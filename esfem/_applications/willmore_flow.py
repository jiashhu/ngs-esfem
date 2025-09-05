from ..willmore_mdr import SDMDR, WillMoreMDR
from geometry import DiscreteMesh, Mesh1dFromPoints, FlowerCurve, MeshSphere, Param2dRot
from ngsolve import *
import numpy as np
from es_utils import pos_transformer
from esfem.ale import VtkOutBnd

class WillmoreMDRRot(WillMoreMDR):
    '''
        Revolutional surface Willmore flow via MDR+KLL formulation 
        初始化参数： mymesh: 初始网格
                Geo_Rot_Obj: Param2dRot类的对象，包含精确曲面信息
                T: 终止时间
                dt: 时间步长
                order: 空间离散阶数
                BDForder: 时间离散阶数
    '''
    def __init__(self, mymesh, Geo_Rot_Obj:Param2dRot, T=0.0001, dt=0.0001, order=1, BDForder=1):
        super().__init__(mymesh, T, dt, order, BDForder)
        self.RBC2d = Geo_Rot_Obj

    def IniByGeoObj(self, geo_rot_obj:Param2dRot):
        # 通过初始几何的Geo_Rot_Obj中的参数表达式，
        # 来初始化初始曲面的高阶形变，高阶法向量，BaseX，高阶的z的插值，高阶的平均曲率以及V的插值
        # 并用初始高阶形变IniDefP来初始化形变Disp
        # compute pointwise mean curvature and normal
        coords_3d = self.interp_obj.get_coords_quad()
        phi_np, theta_np      = geo_rot_obj.Get_Param(coords_3d,Near_opt=True)
        proj_set, norm_set    = geo_rot_obj.get_pos_norm(phi_np, theta_np)
        phi_np, _             = geo_rot_obj.Get_Param(proj_set)
        pointwise_curvature   = geo_rot_obj.H_np(phi_np).reshape(-1,1)
        # compute pointwise LapH and Lapn
        geo_rot_obj.Generate_LapHn_func()
        laplace_curvature, laplace_norm, weingarten_norm = geo_rot_obj.Get_Lap_H_N(phi_np, theta_np)
        # z 是 ∆n + |A|^2 n
        pointwise_z       = laplace_norm + weingarten_norm * laplace_norm
        pointwise_V       = laplace_curvature - 1/2*pointwise_curvature**3 \
                                + weingarten_norm*pointwise_curvature

        GetFesGfu = lambda x: self.interp_obj.return_l2(x)
        def_coords = proj_set - coords_3d
        # list of three gridfunctions
        ModPosx,   ModPosy,   ModPosz   = map(GetFesGfu,[proj_set[:,0],proj_set[:,1],proj_set[:,2]])
        ModNGridx, ModNGridy, ModNGridz = map(GetFesGfu,[norm_set[:,0],norm_set[:,1],norm_set[:,2]])
        DefPosx,   DefPosy,   DefPosz   = map(GetFesGfu,[def_coords[:,0],def_coords[:,1],def_coords[:,2]])
        Modzx,     Modzy,     Modzz     = map(GetFesGfu,[pointwise_z[:,0],pointwise_z[:,1],pointwise_z[:,2]])
        DefPosxyz   = [DefPosx,   DefPosy,   DefPosz]
        ModNGridxyz = [ModNGridx, ModNGridy, ModNGridz]
        ModPosxyz   = [ModPosx,   ModPosy,   ModPosz]
        Modzxyz     = [Modzx,     Modzy,     Modzz]
        for ii in range(self.dim):
            self.init_deformation.components[ii].vec.data = BaseVector(DefPosxyz[ii])
            self.nuold.components[ii].vec.data   = BaseVector(ModNGridxyz[ii])
            self.BaseX.components[ii].vec.data   = BaseVector(ModPosxyz[ii])
            self.zold.components[ii].vec.data    = BaseVector(Modzxyz[ii])
        self.Hold.vec.data    = BaseVector(GetFesGfu(pointwise_curvature[:,0]))
        self.Vold.vec.data    = BaseVector(GetFesGfu(pointwise_V[:,0]))
        self.Disp.vec.data    = self.init_deformation.vec
        self.mesh.SetDeformation(self.Disp)
        
        # L2 error of Position modification
        L2err = lambda x: np.sqrt(Integrate(InnerProduct(x,x), self.mesh, definedon = self.mesh.Boundaries(".*"),element_wise=False))
        errPos = L2err(self.init_deformation)
        print('L2 Error of position on modified surface is {}'.format(errPos))
        Numerical_H = GridFunction(self.fes)
        Numerical_n = GridFunction(self.fesV)
        Numerical_H.Interpolate(Trace(-specialcf.Weingarten(3)),definedon=self.mesh.Boundaries('.*'))
        Numerical_n.Interpolate(-specialcf.normal(3),definedon=self.mesh.Boundaries('.*'))
        err_H = GridFunction(self.fes)
        err_n = GridFunction(self.fesV)
        err_H.vec.data = Numerical_H.vec - self.Hold.vec
        err_n.vec.data = Numerical_n.vec - self.nuold.vec
        print('Error of H is {}, Error of n is {}'.format(L2err(err_H), L2err(err_n)))
        self.LapvSet()

    def ReIni_HN_ByDisPos(self):
        print('Now Reinit time is {}, Reinit by curvature of high order surface with C_order {} and E_order {}'.format(self.t,self.mesh.GetCurveOrder(),self.order))
        self.Hold.Interpolate(Trace(-specialcf.Weingarten(3)),definedon=self.mesh.Boundaries('.*'))
        self.nuold.Interpolate(-specialcf.normal(3),definedon=self.mesh.Boundaries('.*'))
        self.vold.Interpolate(CF((0,0,0)),definedon=self.mesh.Boundaries('.*'))

    def PP_Pic(self,vtk_Obj_bnd:VtkOutBnd=None):
        # 后处理： 保存图像
        perform_res = vtk_Obj_bnd.output(self.mesh,[self.Vold, self.Hold, self.Hgeo], names= ['Vc','Hc','Hg'],tnow=self.t)
        # perform_res = vtk_Obj_bnd.output(self.mesh,[],tnow=self.t)
        # 后处理： 计算网格质量，利用self.mesh的拓扑关系
        if perform_res:
            self.VCoord.Interpolate(CF((x,y,z)),definedon=self.mesh.Boundaries(".*"))
            self.DMesh_Obj.UpdateCoords(pos_transformer(self.VCoord,dim=3))
            Q_Area, Q_Leng = self.DMesh_Obj.MeshQuality()
            self.Mesh_Quality.append([Q_Area,Q_Leng,self.t])
