from ngsolve import *
from es_utils import get_id_cf
from geometry import Mesh_Info_Parse, DiscreteMesh
from es_utils import pos_transformer
import numpy as np
from global_utils import LogTime
import os

class Dziuk():
    '''Dziuk Method MCF'''
    def __init__(self, mymesh, T = 1e-4, tau = 1e-4):
        self.mesh = mymesh
        self.T, self.t, self.finalT = T, 0, 0
        self.dt = Parameter(tau)
        self.fes = H1(self.mesh,order=1)
        self.fesV = VectorH1(self.mesh,order=1)

        self.Solution = GridFunction(self.fesV)
        self.MethodName = 'Dziuk'
        self.lhs, self.rhs = [],[]

        self.counter_print = 0
        self.dim = self.mesh.dim
        print('mesh dim = {}'.format(self.dim))
        # initial historic info: Position, normal vector(ele), weighted n(ver)
        self.X_old = GridFunction(self.fesV)
        self.NvecD = GridFunction(self.fesV)
        self.Disp = GridFunction(self.fesV)
        self.scale = 1
        self.MQ_Measure_Set()

    def IniByDisPos(self):
        # initilize Position by discrete vertices
        Vertices_Coords = np.array([v.point for v in self.mesh.vertices])
        self.X_old.vec.data = BaseVector(Vertices_Coords.flatten('F'))

    def MQ_Measure_Set(self):
        Vertices_Coords = np.array([v.point for v in self.mesh.vertices])
        self.ElVerInd, self.EdVerInd = Mesh_Info_Parse(self.mesh)
        self.DMesh_Obj = DiscreteMesh(Vertices_Coords,self.dim-1,self.dim,self.ElVerInd,self.EdVerInd)
        self.Mesh_Quality = []
        self.Area_set  = []

    def Get_Proper_dt(self,coef=1,h_opt='min'):
        # order h**2, independent of scale
        Vertices_Coords = pos_transformer(self.X_old, dim=self.dim)
        self.DMesh_Obj.UpdateCoords(Vertices_Coords)
        h = min(np.linalg.norm(self.DMesh_Obj.barvec,axis=1))
        hmax = max(np.linalg.norm(self.DMesh_Obj.barvec,axis=1))
        hmean = np.mean(np.linalg.norm(self.DMesh_Obj.barvec,axis=1))
        if h_opt == 'min':
            dt = coef*h**2
        elif h_opt == 'max':
            dt = coef*hmax**2
        elif h_opt == 'mean':
            dt = coef*hmean**2
        print('{} :Using Get_Proper_dt, Now coef is {}, t is {}, min h is {}, dt is {}, scale is {}'.format(LogTime(),coef,self.t,h,format(dt,'0.3e'),self.scale))
        return dt

    def WeakMCF(self):
        '''
            Weak formulation for position
            ∂t X = -Hn = ∆ X
        '''
        solu, ut = self.fesV.TnT()
        self.lhs = BilinearForm(self.fesV)
        self.lhs += (InnerProduct(solu,ut)+self.dt*InnerProduct(grad(solu).Trace(),grad(ut).Trace()))*ds
        self.rhs = LinearForm(self.fesV)
        self.rhs += InnerProduct(self.X_old,ut)*ds

    def PP_Pic(self,vtk_Obj_bnd=None):
        # 后处理： 保存图像
        perform_res = vtk_Obj_bnd.output(self.mesh,[],tnow=self.t)
        # 后处理： 计算网格质量，利用self.mesh的拓扑关系
        if perform_res:
            self.DMesh_Obj.UpdateCoords(pos_transformer(self.X_old))
            Q_Area, Q_Leng = self.DMesh_Obj.MeshQuality()
            self.Mesh_Quality.append([Q_Area,Q_Leng,self.t])

    def PrintDuring(self):
        if self.t > self.T*self.counter_print/5:
            print('Completed {} per cent'.format(int(self.counter_print/5*100)))
            self.counter_print += 1

    def Solving(self,vtk_obj):
        while self.t<self.T:
            tauval = self.dt.Get()
            self.mesh.SetDeformation(self.Disp)
            self.lhs.Assemble()
            self.rhs.Assemble()
            self.Solution.vec.data = self.lhs.mat.Inverse(inverse="pardiso")*self.rhs.vec

            ## Solution.vec 代表的是X^m+1
            self.Disp.vec.data = BaseVector(self.Disp.vec.FV().NumPy() 
                                            + self.Solution.vec.FV().NumPy()
                                            - self.X_old.vec.FV().NumPy())
            self.X_old.vec.data = BaseVector(self.Solution.vec.FV().NumPy())
            self.PP_Pic(vtk_obj)
            self.PrintDuring()
            self.t += tauval