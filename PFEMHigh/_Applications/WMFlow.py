from ..LapVWillmore import LapVSD, LapVWillMore
from Package_Geometry_Obj import Coord1dFlower, DiscreteMesh, Mesh1dFromPoints, FlowerCurve, MeshSphere, Param2dRot
from ngsolve import *
import numpy as np
import os
from Package_MyNgFunc import Pos_Transformer
from Package_ALE_Geometry import Vtk_out_1d, Vtk_out_BND
from Package_MyCode import LogTime

class FlowerLapVSD(LapVSD):
    '''
        T: 总演化时间， dt: time step
        Surface Diffusion是在Willmore的基础上令Q=0，需要在生成weak form的WeakWillmore中采用SD_opt=True的选项。
    '''
    def __init__(self, n_half_blade=120, T=0.0001, dt=0.0001, order=1, BDForder=1, a=0.65, b=7):
        self.Coords,self.theta = Coord1dFlower(n_half_blade, num_blade=b, radparam=a)
        self.DC_obj = DiscreteMesh(self.Coords,dim=1,adim=2)  # for mesh quality
        mymesh = Mesh1dFromPoints(self.Coords,dim=1,adim=2)
        self.Flower_Obj = FlowerCurve()
        super().__init__(mymesh, T, dt, order, BDForder)
        self.Mesh_Quality = []

    def IniByGeoObj(self):
        self.Hfunc, self.Nxfunc, self.Nyfunc = self.Flower_Obj.Hfunc, self.Flower_Obj.Nxfunc, self.Flower_Obj.Nyfunc
        self.LapHfunc, self.zxfunc, self.zyfunc = self.Flower_Obj.LapHfunc, self.Flower_Obj.zxfunc, self.Flower_Obj.zyfunc
        
        self.IniNormal.vec.data = BaseVector(np.append(self.Nxfunc(self.theta),self.Nyfunc(self.theta)))
        self.IniCurv.vec.data = BaseVector(self.Hfunc(self.theta))
        self.IniV.vec.data = BaseVector(self.LapHfunc(self.theta))
        self.Iniz.vec.data = BaseVector(np.append(self.zxfunc(self.theta),self.zyfunc(self.theta)))
        
        self.Hold.vec.data = self.IniCurv.vec
        self.nuold.vec.data = self.IniNormal.vec
        self.Vold.vec.data = self.IniV.vec
        self.zold.vec.data = self.Iniz.vec

        self.LapvSet()

    def Get_Proper_dt(self,coef=0.05):
        # order h**2, independent of scale
        Vertices_Coords = Pos_Transformer(self.IniX,dim=2) + Pos_Transformer(self.Disp,dim=2)
        self.DC_obj.UpdateCoords(Vertices_Coords)
        h = min(np.linalg.norm(self.DC_obj.barvec,axis=1))
        dt = coef*h**2
        print('Using Get_Proper_dt, Now t is {}, min h is {}, dt is {}, scale is {}'.format(self.t,h,format(dt,'0.3e'),self.scale))
        return dt

    def PP_Pic(self,vtk_Obj_1d:Vtk_out_1d):
        # 后处理： 保存图像
        self.VCoord.Interpolate(CF((x,y)),definedon=self.mesh.Boundaries(".*"))
        Coords2d = Pos_Transformer(self.VCoord,dim=self.dim)
        perform_res = vtk_Obj_1d.Output(Coords2d,[],tnow=self.t)
        # 后处理： 计算网格质量，利用self.mesh的拓扑关系
        if perform_res:
            self.DMesh_Obj.UpdateCoords(Coords2d)
            Q_Area, Q_Leng = self.DMesh_Obj.MeshQuality()
            self.Mesh_Quality.append([Q_Area,Q_Leng,self.t]) 

    def Solving(self,vtk_Obj_1d:Vtk_out_1d):
        print('Now time is {}, Solving at t = {}, to T = {} with dt = {}'.format(LogTime(), self.t, self.T, self.dt.Get()))
        while self.t<self.T:
            tauval = self.dt.Get()
            
            self.lhs.Assemble()
            self.rhs.Assemble()
            self.gfu.vec.data = self.lhs.mat.Inverse(inverse="pardiso")*self.rhs.vec
    
            ## Update the Mean curvature and Normal vector old (used in weak formulation)
            self.Hold.vec.data = self.H.vec
            self.nuold.vec.data = self.normal.vec
            self.vold.vec.data = self.velocity.vec
            self.Vold.vec.data = self.V.vec
            self.zold.vec.data = self.z.vec

            BDFInc = self.BDF.Stepping(self.velocity.vec.FV().NumPy()*tauval)
            self.Disp.vec.data += BaseVector(BDFInc)
            self.mesh.SetDeformation(self.Disp)
            self.t += tauval
            self.PrintDuring()
            self.PP_Pic(vtk_Obj_1d)

class FlowerLapVSD_Implicit(FlowerLapVSD):
    def __init__(self, n_half_blade=120, T=0.0001, dt=0.0001, order=1, BDForder=1, a=0.65, b =7):
        super().__init__(n_half_blade, T, dt, order, BDForder, a, b)
        self.ds_lumping = []
        self.GenerateDs()
        self.fes1 = H1(self.mesh, order=1)
        self.n_Norm_ver = GridFunction(self.fes1)

    def GenerateDs(self):
        if self.dim==2:
            ir = IntegrationRule(points = [(0,0), (1,0)], weights = [1/2, 1/2])
            self.ds_lumping = ds(definedon=self.mesh.Boundaries("material"),intrules = { SEGM : ir })
        elif self.dim==3:
            ir = IntegrationRule(points = [(0,0), (1,0), (0,1)], weights = [1/6, 1/6, 1/6])
            self.ds_lumping = ds(intrules = { TRIG : ir })

    def Generate_HN_ByBGN(self):
        # fisrt update the weighted normal 
        N_X = GridFunction(self.fesV)
        N_X.Interpolate(-specialcf.normal(self.dim),definedon=self.mesh.Boundaries('.*'))
        # update mean curvature through H = Lap n
        self.H_X_BGN = GridFunction(self.fes)
        H_trial, H_test = self.fes.TnT()
        rhs = LinearForm(self.fes)
        lhs = BilinearForm(self.fes)
        lhs += H_trial*H_test*self.ds_lumping
        rhs += Trace(grad(N_X).Trace())*H_test*self.ds_lumping
        lhs.Assemble()
        rhs.Assemble()
        self.H_X_BGN.vec.data = lhs.mat.Inverse(inverse='pardiso')*rhs.vec

    def ReIni_HN_ByDisPos(self):
        '''
            2d case: - (Laplace X) * n= H, exterior normal and positive curvature for sphere
            (H,Ht) = sum (grad(X_i),grad(n_i))*Ht + (grad(X_i),grad(Ht))*n_i
        '''
        ## Given high order mesh, deformed by Disp, and compute nuold by interpolation, Hold by weak formulation
        print('Now Reinit time is {}'.format(self.t))
        id_X = GridFunction(self.fesV)
        id_X.Interpolate(CF((x,y)),definedon=self.mesh.Boundaries('.*'))
        n_X = GridFunction(self.fesV)
        # specialcf.normal(3): inward normal vector field
        n_X.Interpolate(-specialcf.normal(self.dim),definedon=self.mesh.Boundaries('.*'))
        H_trial, H_test = self.fes.TnT()
        H_X = GridFunction(self.fes)
        rhs = LinearForm(self.fes)
        rhs += InnerProduct(grad(id_X).Trace(),grad(n_X).Trace())*H_test*ds + InnerProduct(n_X,grad(id_X).Trace()*grad(H_test).Trace())*ds
        lhs = BilinearForm(self.fes)
        lhs += H_trial*H_test*ds
        lhs.Assemble()
        rhs.Assemble()
        H_X.vec.data = lhs.mat.Inverse(inverse='pardiso')*rhs.vec

        err_n = GridFunction(self.fesV)
        err_n.vec.data = n_X.vec - self.nuold.vec
        err_H = GridFunction(self.fes)
        err_H.vec.data = H_X.vec - self.Hold.vec
        self.mesh.SetDeformation(self.IniDefP)
        e_nL2 = np.sqrt(Integrate(InnerProduct(err_n,err_n), self.mesh, element_wise=False, definedon=self.mesh.Boundaries('.*')))
        e_HL2 = np.sqrt(Integrate(err_H**2, self.mesh, element_wise=False, definedon=self.mesh.Boundaries('.*')))
        SurfaceArea = Integrate(1, self.mesh, element_wise=False, definedon=self.mesh.Boundaries('.*'))
        print('e_nL2 is {}, e_HL2 is {} on surface with Area {}'.format(e_nL2,e_HL2,SurfaceArea))
        self.mesh.SetDeformation(self.Disp)

        self.Hold.vec.data = H_X.vec
        self.nuold.vec.data = n_X.vec
        self.LapvSet()

    def WeakSD(self):
        print('Implicit Modification')
        kappa , H , V , z,  nu,  v = self.fesMix.TrialFunction()
        kappat, Ht, Vt, zt, nut, vt= self.fesMix.TestFunction()

        # 预设速度
        vT = self.vold - self.nuold*InnerProduct(self.vold,self.nuold)/InnerProduct(self.nuold,self.nuold)
        ## Weingarten Map
        A = grad(self.nuold).Trace()
        A = A.trans

        Lhs = BilinearForm(self.fesMix,symmetric=False)
        ## self.ds may blow up
        Lhs += (InnerProduct(v,self.nuold)*kappat)*ds-V*kappat*ds
        Lhs += (H*Ht-self.dt*(InnerProduct(grad(V).Trace(),grad(Ht).Trace())-InnerProduct(A,A)*V*Ht\
                            +InnerProduct(vT,grad(H).Trace())*Ht))*ds
        Lhs += (V*Vt+InnerProduct(grad(H).Trace(),grad(Vt).Trace()))*ds

        Lhs += (InnerProduct(z,zt)+InnerProduct(grad(nu).Trace(),grad(zt).Trace()))*ds\
                -(InnerProduct(A,A)*InnerProduct(nu,zt))*ds
        Lhs += InnerProduct(nu,nut)*ds+(self.dt*(-InnerProduct(grad(z).Trace(),grad(nut).Trace())\
                                        -InnerProduct(grad(nu).Trace()*vT,nut)\
                                        -2*InnerProduct(A*grad(self.Hold).Trace(), grad(nut).Trace().trans*nu)\
                                        -InnerProduct(grad(H).Trace(),grad(H).Trace())*InnerProduct(nu,nut)\
                                        -self.Hold*InnerProduct(A*z,nut)\
                                        +InnerProduct(A*z,A*nut)\
                                        -InnerProduct(A*grad(H).Trace(),A*nut)))*ds
        Lhs += (kappa*InnerProduct(self.nuold,vt))*ds+InnerProduct(grad(v).Trace(),grad(vt).Trace())*ds
        self.lhs = Lhs

        Rhs = LinearForm(self.fesMix)
        Rhs += self.Hold*Ht*ds
        Rhs += InnerProduct(self.nuold,nut)*ds
        self.rhs = Rhs

class FlowerLapVSD_v_0531(FlowerLapVSD_Implicit):
    '''
        Using the weak form in WMImplicityVz2
    '''
    def __init__(self, a, b, n_half_blade=120, T=0.0001, dt=0.0001, order=1, BDForder=1):
        super().__init__(n_half_blade, T, dt, order, BDForder, a=a, b=b)

    def WeakSD(self):
        print('Implicit Modification')
        ## Set Curvature
        kappa , H , V , z, nu, v = self.fesMix.TrialFunction()
        kappat, Ht, Vt, zt, nut, vt= self.fesMix.TestFunction()

        ## Weingarten Map
        A0 = grad(self.nuold).Trace()
        A = 1/2*(A0.trans+A0)
        Q = 0
        
        n_unified = self.nuold/Norm(self.nuold)
        Lhs = BilinearForm(self.fesMix,symmetric=False)
        Lhs += (InnerProduct(v,n_unified)*kappat)*ds-V*kappat*ds
        Lhs += (1/self.dt*H*Ht - InnerProduct(grad(V).Trace(),grad(Ht).Trace()))*ds
        Lhs += (V*Vt + InnerProduct(grad(H).Trace(),grad(Vt).Trace()))*ds
        Lhs += (InnerProduct(z,zt) + InnerProduct(grad(nu).Trace(),grad(zt).Trace()))*ds
        Lhs += 1/self.dt*InnerProduct(nu,nut)*ds - InnerProduct(grad(z).Trace(),grad(nut).Trace())*ds
        Lhs += (kappa*InnerProduct(n_unified,vt))*ds + InnerProduct(grad(v).Trace(),grad(vt).Trace())*ds
        # implicit treatment of z and V
        Lhs += -self.Hold*InnerProduct(A*z,nut)*ds + InnerProduct(A*z,A*nut)*ds
        Lhs += -InnerProduct(grad(nu).Trace()*self.vold,nut)*ds
        Lhs += -InnerProduct(self.vold,grad(H).Trace())*Ht*ds + InnerProduct(A,A)*V*Ht*ds
        self.lhs = Lhs

        # 计算需要对Hold,nuold设定初值
        Rhs = LinearForm(self.fesMix)
        Rhs += Q*Vt*ds
        Rhs += 1/self.dt*self.Hold*Ht*ds 
        Rhs += 1/self.dt*InnerProduct(n_unified,nut)*ds
        Rhs += 2*InnerProduct(grad(nut).Trace()*(A*grad(self.Hold).Trace()), n_unified)*ds
        Rhs += (InnerProduct(grad(self.Hold).Trace(),grad(self.Hold).Trace())*InnerProduct(self.nuold,nut))*ds + InnerProduct(A*grad(self.Hold).Trace(),A*nut)*ds
        Rhs += Q*Trace(grad(nut).Trace())*ds - Q*self.Hold*InnerProduct(n_unified,nut)*ds
        Rhs += (InnerProduct(A,A)*InnerProduct(n_unified,zt))*ds
        self.rhs = Rhs

class FlowerLapVSD_v_0601(FlowerLapVSD_Implicit):
    '''
        No unification of normal vector
    '''
    def __init__(self, a, b, n_half_blade=120, T=0.0001, dt=0.0001, order=1, BDForder=1):
        super().__init__(n_half_blade, T, dt, order, BDForder, a=a, b=b)

    def WeakSD(self):
        print('Implicit Modification')
        ## Set Curvature
        kappa , H , V , z, nu, v = self.fesMix.TrialFunction()
        kappat, Ht, Vt, zt, nut, vt= self.fesMix.TestFunction()

        ## Weingarten Map
        A0 = grad(self.nuold).Trace()
        A = 1/2*(A0.trans+A0)
        Q = 0
        
        n_unified = self.nuold
        Lhs = BilinearForm(self.fesMix,symmetric=False)
        Lhs += (InnerProduct(v,n_unified)*kappat)*ds-V*kappat*ds
        Lhs += (1/self.dt*H*Ht - InnerProduct(grad(V).Trace(),grad(Ht).Trace()))*ds
        Lhs += (V*Vt + InnerProduct(grad(H).Trace(),grad(Vt).Trace()))*ds
        Lhs += (InnerProduct(z,zt) + InnerProduct(grad(nu).Trace(),grad(zt).Trace()))*ds
        Lhs += 1/self.dt*InnerProduct(nu,nut)*ds - InnerProduct(grad(z).Trace(),grad(nut).Trace())*ds
        Lhs += (kappa*InnerProduct(n_unified,vt))*ds + InnerProduct(grad(v).Trace(),grad(vt).Trace())*ds
        # implicit treatment of z and V
        Lhs += -self.Hold*InnerProduct(A*z,nut)*ds + InnerProduct(A*z,A*nut)*ds
        Lhs += -InnerProduct(grad(nu).Trace()*self.vold,nut)*ds
        Lhs += -InnerProduct(self.vold,grad(H).Trace())*Ht*ds + InnerProduct(A,A)*V*Ht*ds
        self.lhs = Lhs

        # 计算需要对Hold,nuold设定初值
        Rhs = LinearForm(self.fesMix)
        Rhs += Q*Vt*ds
        Rhs += 1/self.dt*self.Hold*Ht*ds 
        Rhs += 1/self.dt*InnerProduct(n_unified,nut)*ds
        Rhs += 2*InnerProduct(grad(nut).Trace()*(A*grad(self.Hold).Trace()), n_unified)*ds
        Rhs += (InnerProduct(grad(self.Hold).Trace(),grad(self.Hold).Trace())*InnerProduct(self.nuold,nut))*ds + InnerProduct(A*grad(self.Hold).Trace(),A*nut)*ds
        Rhs += Q*Trace(grad(nut).Trace())*ds - Q*self.Hold*InnerProduct(n_unified,nut)*ds
        Rhs += (InnerProduct(A,A)*InnerProduct(n_unified,zt))*ds
        self.rhs = Rhs

class FlowerLapVSD_v_0602(FlowerLapVSD_Implicit):
    '''
        Partial unification of normal vector
    '''
    def __init__(self, a, b, n_half_blade=120, T=0.0001, dt=0.0001, order=1, BDForder=1):
        super().__init__(n_half_blade, T, dt, order, BDForder, a=a, b=b)

    def WeakSD(self):
        print('Implicit Modification')
        ## Set Curvature
        kappa , H , V , z, nu, v = self.fesMix.TrialFunction()
        kappat, Ht, Vt, zt, nut, vt= self.fesMix.TestFunction()

        ## Weingarten Map
        A0 = grad(self.nuold).Trace()
        A = 1/2*(A0.trans+A0)
        Q = 0
        
        n_unified = self.nuold/Norm(self.nuold)
        Lhs = BilinearForm(self.fesMix,symmetric=False)
        Lhs += (InnerProduct(v,n_unified)*kappat)*ds-V*kappat*ds
        Lhs += (1/self.dt*H*Ht - InnerProduct(grad(V).Trace(),grad(Ht).Trace()))*ds
        Lhs += (V*Vt + InnerProduct(grad(H).Trace(),grad(Vt).Trace()))*ds
        Lhs += (InnerProduct(z,zt) + InnerProduct(grad(nu).Trace(),grad(zt).Trace()))*ds
        Lhs += 1/self.dt*InnerProduct(nu,nut)*ds - InnerProduct(grad(z).Trace(),grad(nut).Trace())*ds
        Lhs += (kappa*InnerProduct(n_unified,vt))*ds + InnerProduct(grad(v).Trace(),grad(vt).Trace())*ds
        # implicit treatment of z and V
        Lhs += -self.Hold*InnerProduct(A*z,nut)*ds + InnerProduct(A*z,A*nut)*ds
        Lhs += -InnerProduct(grad(nu).Trace()*self.vold,nut)*ds
        Lhs += -InnerProduct(self.vold,grad(H).Trace())*Ht*ds + InnerProduct(A,A)*V*Ht*ds
        self.lhs = Lhs

        # 计算需要对Hold,nuold设定初值
        Rhs = LinearForm(self.fesMix)
        Rhs += Q*Vt*ds
        Rhs += 1/self.dt*self.Hold*Ht*ds 
        Rhs += 1/self.dt*InnerProduct(self.nuold,nut)*ds
        Rhs += 2*InnerProduct(grad(nut).Trace()*(A*grad(self.Hold).Trace()), self.nuold)*ds
        Rhs += (InnerProduct(grad(self.Hold).Trace(),grad(self.Hold).Trace())*InnerProduct(self.nuold,nut))*ds + InnerProduct(A*grad(self.Hold).Trace(),A*nut)*ds
        Rhs += Q*Trace(grad(nut).Trace())*ds - Q*self.Hold*InnerProduct(self.nuold,nut)*ds
        Rhs += (InnerProduct(A,A)*InnerProduct(self.nuold,zt))*ds
        self.rhs = Rhs

    def PP_Pic(self,vtk_Obj_1d:Vtk_out_1d):
        # 后处理： 保存图像
        self.VCoord.Interpolate(CF((x,y)),definedon=self.mesh.Boundaries(".*"))
        Coords2d = Pos_Transformer(self.VCoord,dim=self.dim)

        n_Norm = GridFunction(self.fes)
        n_Norm.Set(InnerProduct(self.nuold,self.nuold),definedon=self.mesh.Boundaries('.*'))
        # 对高阶元，最前面的自由度是1阶的自由度
        n_Norm_Ver = n_Norm.vec.FV().NumPy()[:Coords2d.shape[0]]
        self.Generate_HN_ByBGN()
        err_H = GridFunction(self.fes)
        err_H.vec.data = self.Hold.vec - self.H_X_BGN.vec
        err_H_Ver = err_H.vec.FV().NumPy()[:Coords2d.shape[0]]
        H_val = self.Hold.vec.FV().NumPy()[:Coords2d.shape[0]]
        pData_dict = {'n_Norm': n_Norm_Ver, 'err_H': err_H_Ver, 'H_val': H_val}
        perform_res = vtk_Obj_1d.Output(Coords2d,pData_dict=pData_dict,tnow=self.t)
        # 后处理： 计算网格质量，利用self.mesh的拓扑关系
        if perform_res:
            self.DMesh_Obj.UpdateCoords(Coords2d)
            Q_Area, Q_Leng = self.DMesh_Obj.MeshQuality()
            self.Mesh_Quality.append([Q_Area,Q_Leng,self.t])

class FlowerLapVSD_v_0603(FlowerLapVSD_v_0602):
    '''
        No unification of normal vector
    '''
    def __init__(self, a, b, n_half_blade=120, T=0.0001, dt=0.0001, order=1, BDForder=1):
        super().__init__(a, b, n_half_blade, T, dt, order, BDForder)
        print('Remove unification modification in surface diffusion!!!')

    def WeakSD(self):
        print('Implicit Modification')
        ## Set Curvature
        kappa , H , V , z, nu, v = self.fesMix.TrialFunction()
        kappat, Ht, Vt, zt, nut, vt= self.fesMix.TestFunction()

        ## Weingarten Map
        A0 = grad(self.nuold).Trace()
        A = 1/2*(A0.trans+A0)
        Q = 0
        
        n_unified = self.nuold
        Lhs = BilinearForm(self.fesMix,symmetric=False)
        Lhs += (InnerProduct(v,n_unified)*kappat)*ds-V*kappat*ds
        Lhs += (1/self.dt*H*Ht - InnerProduct(grad(V).Trace(),grad(Ht).Trace()))*ds
        Lhs += (V*Vt + InnerProduct(grad(H).Trace(),grad(Vt).Trace()))*ds
        Lhs += (InnerProduct(z,zt) + InnerProduct(grad(nu).Trace(),grad(zt).Trace()))*ds
        Lhs += 1/self.dt*InnerProduct(nu,nut)*ds - InnerProduct(grad(z).Trace(),grad(nut).Trace())*ds
        Lhs += (kappa*InnerProduct(n_unified,vt))*ds + InnerProduct(grad(v).Trace(),grad(vt).Trace())*ds
        # implicit treatment of z and V
        Lhs += -self.Hold*InnerProduct(A*z,nut)*ds + InnerProduct(A*z,A*nut)*ds
        Lhs += -InnerProduct(grad(nu).Trace()*self.vold,nut)*ds
        Lhs += -InnerProduct(self.vold,grad(H).Trace())*Ht*ds + InnerProduct(A,A)*V*Ht*ds
        self.lhs = Lhs

        # 计算需要对Hold,nuold设定初值
        Rhs = LinearForm(self.fesMix)
        Rhs += Q*Vt*ds
        Rhs += 1/self.dt*self.Hold*Ht*ds 
        Rhs += 1/self.dt*InnerProduct(self.nuold,nut)*ds
        Rhs += 2*InnerProduct(grad(nut).Trace()*(A*grad(self.Hold).Trace()), self.nuold)*ds
        Rhs += (InnerProduct(grad(self.Hold).Trace(),grad(self.Hold).Trace())*InnerProduct(self.nuold,nut))*ds + InnerProduct(A*grad(self.Hold).Trace(),A*nut)*ds
        Rhs += Q*Trace(grad(nut).Trace())*ds - Q*self.Hold*InnerProduct(self.nuold,nut)*ds
        Rhs += (InnerProduct(A,A)*InnerProduct(self.nuold,zt))*ds
        self.rhs = Rhs

class SphereLapVWillmore(LapVWillMore):
    '''
        Test Convergence of Laplace v = kappa n & Willmore flow on the case of sphere (R=2)
    '''
    def __init__(self, maxh, T=0.0001, dt=0.0001, order=1, BDForder=1):
        mymesh = MeshSphere(maxh,order=order,R=2)
        super().__init__(mymesh, T, dt, order, BDForder)

    def IniByGeoObj(self):
        # sphere of radius 2
        self.IniNormal.Interpolate(CF((x/2,y/2,z/2)),definedon=self.mesh.Boundaries('.*'))
        self.nuold.vec.data = self.IniNormal.vec
        
        self.IniCurv.Interpolate(1,definedon=self.mesh.Boundaries('.*'))
        self.Hold.vec.data = self.IniCurv.vec

        self.IniV.Interpolate(0,definedon=self.mesh.Boundaries('.*'))
        self.Vold.vec.data = self.IniV.vec

        self.Iniv.Interpolate(CF((0,0,0)),definedon=self.mesh.Boundaries('.*'))
        self.vold.vec.data = self.Iniv.vec

        self.Iniz.Interpolate(CF((0,0,0)),definedon=self.mesh.Boundaries('.*'))
        self.zold.vec.data = self.Iniz.vec

        self.IniX.Interpolate(CoefficientFunction((x,y,z)),definedon=self.mesh.Boundaries(".*"))

    def ErrL2H1(self):
        # Unset deformation during solving but keep the initial deformation, 本质上是case by case的写出精确解的表示
        self.mesh.SetDeformation(self.IniDefP)
        H1errfun = lambda xfun: sqrt(Integrate(InnerProduct(grad(xfun).Trace(),grad(xfun).Trace()), self.mesh, element_wise=False, definedon=self.mesh.Boundaries('.*')))
        L2errfun = lambda xfun: sqrt(Integrate(InnerProduct((xfun),(xfun)), self.mesh, element_wise=False, definedon=self.mesh.Boundaries('.*')))

        errX = GridFunction(self.fesV)
        errV = GridFunction(self.fesV)
        ## Sol.IniX 是 Sol.mesh.SetDeformation(Sol.DefPos) 之后的坐标
        errX.vec.data = self.IniX.vec.FV().NumPy() - (self.IniX.vec.FV().NumPy()+self.Disp.vec.FV().NumPy()-self.IniDefP.vec.FV().NumPy())
        # exact velocity is zero
        errV.vec.data = self.vold.vec.FV().NumPy()

        errH = GridFunction(self.fes)
        errn = GridFunction(self.fesV)
        errH.vec.data = self.IniCurv.vec.FV().NumPy() - self.Hold.vec.FV().NumPy()    
        errn.vec.data = self.IniNormal.vec.FV().NumPy() - self.nuold.vec.FV().NumPy()

        errXH1, errHH1, errNH1, errVH1 = map(H1errfun,[errX,errH,errn,errV])
        errXL2, errHL2, errNL2, errVL2 = map(L2errfun,[errX,errH,errn,errV])
        Err_Dict = {
            'errXH1': errXH1,
            'errHH1': errHH1,
            'errNH1': errNH1,
            'errVH1': errVH1,
            'errXL2': errXL2, 
            'errHL2': errHL2, 
            'errNL2': errNL2,
            'errVL2': errVL2}
        return Err_Dict

class RBCLapVWillmore(LapVWillMore):
    def __init__(self, mymesh, Geo_Rot_Obj:Param2dRot, T=0.0001, dt=0.0001, order=1, BDForder=1):
        super().__init__(mymesh, T, dt, order, BDForder)
        self.RBC2d = Geo_Rot_Obj

    def IniByGeoObj(self, Geo_Rot_Obj:Param2dRot):
        # compute pointwise mean curvature and normal
        Coords3d = self.hInterp_Obj.GetCoordsQuad()
        phi_np, theta_np = Geo_Rot_Obj.Get_Param(Coords3d,Near_opt=True)
        Pset, Normset    = Geo_Rot_Obj.Get_Pos_Norm(phi_np, theta_np)
        phi_np, _        = Geo_Rot_Obj.Get_Param(Pset)
        PointwiseH       = Geo_Rot_Obj.H_np(phi_np).reshape(-1,1)
        # compute pointwise LapH and Lapn
        Geo_Rot_Obj.Generate_LapHn_func()
        LapH, Lapn, A_Frob = Geo_Rot_Obj.Get_Lap_H_N(phi_np, theta_np)
        Pointwisez       = Lapn + A_Frob * Lapn
        PointwiseV       = LapH - 1/2*PointwiseH**3 + A_Frob*PointwiseH

        GetFesGfu = lambda x: self.hInterp_Obj.Return2L2(x)
        DefCoords = Pset-Coords3d
        # list of three gridfunctions
        ModPosx,   ModPosy,   ModPosz   = map(GetFesGfu,[Pset[:,0],Pset[:,1],Pset[:,2]])
        ModNGridx, ModNGridy, ModNGridz = map(GetFesGfu,[Normset[:,0],Normset[:,1],Normset[:,2]])
        DefPosx,   DefPosy,   DefPosz   = map(GetFesGfu,[DefCoords[:,0],DefCoords[:,1],DefCoords[:,2]])
        Modzx,     Modzy,     Modzz     = map(GetFesGfu,[Pointwisez[:,0],Pointwisez[:,1],Pointwisez[:,2]])
        DefPosxyz   = [DefPosx,   DefPosy,   DefPosz]
        ModNGridxyz = [ModNGridx, ModNGridy, ModNGridz]
        ModPosxyz   = [ModPosx,   ModPosy,   ModPosz]
        Modzxyz     = [Modzx,     Modzy,     Modzz]
        for ii in range(self.dim):
            self.IniDefP.components[ii].vec.data = DefPosxyz[ii].vec
            self.nuold.components[ii].vec.data   = ModNGridxyz[ii].vec
            self.BaseX.components[ii].vec.data   = ModPosxyz[ii].vec
            self.zold.components[ii].vec.data    = Modzxyz[ii].vec
        self.Vold.vec.data    = GetFesGfu(PointwiseV[:,0]).vec
        self.Hold.vec.data    = GetFesGfu(PointwiseH[:,0]).vec
        self.Disp.vec.data    = self.IniDefP.vec
        self.mesh.SetDeformation(self.Disp)
        # L2 error of Position modification
        L2err = lambda x: np.sqrt(Integrate(InnerProduct(x,x), self.mesh, definedon = self.mesh.Boundaries(".*"),element_wise=False))
        errPos = L2err(self.IniDefP)
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
        pass

    def PP_Pic(self,vtk_Obj_bnd:Vtk_out_BND=None):
        # 后处理： 保存图像
        perform_res = vtk_Obj_bnd.Output(self.mesh,[self.Vold, self.Hold, self.Hgeo], names= ['Vc','Hc','Hg'],tnow=self.t)
        # perform_res = vtk_Obj_bnd.Output(self.mesh,[],tnow=self.t)
        # 后处理： 计算网格质量，利用self.mesh的拓扑关系
        if perform_res:
            self.VCoord.Interpolate(CF((x,y,z)),definedon=self.mesh.Boundaries(".*"))
            self.DMesh_Obj.UpdateCoords(Pos_Transformer(self.VCoord,dim=3))
            Q_Area, Q_Leng = self.DMesh_Obj.MeshQuality()
            self.Mesh_Quality.append([Q_Area,Q_Leng,self.t])

class RBCLapVWM_ImplicitVz_v_0530(RBCLapVWillmore):
    def __init__(self, mymesh, Geo_Rot_Obj: Param2dRot, T=0.0001, dt=0.0001, order=1, BDForder=1):
        super().__init__(mymesh, Geo_Rot_Obj, T, dt, order, BDForder)
        self.Hgeo = GridFunction(self.fes)
        self.ngeo = GridFunction(self.fesV)

    def WeakWillmore(self,SD_opt=False):
        ## Set Curvature
        kappa , H , V , z, nu, v = self.fesMix.TrialFunction()
        kappat, Ht, Vt, zt, nut, vt= self.fesMix.TestFunction()

        ## Weingarten Map
        A0 = grad(self.ngeo).Trace()
        A = 1/2*(A0.trans+A0)
        if SD_opt:
            Q = 0
        else:
            Q = -1/2*self.Hold**3+InnerProduct(A,A)*self.Hold

        Lhs = BilinearForm(self.fesMix,symmetric=False)
        Lhs += (InnerProduct(v,self.nuold)*kappat)*ds-V*kappat*ds
        Lhs += (1/self.dt*H*Ht - InnerProduct(grad(V).Trace(),grad(Ht).Trace()))*ds
        Lhs += (V*Vt + InnerProduct(grad(H).Trace(),grad(Vt).Trace()))*ds
        Lhs += (InnerProduct(z,zt) + InnerProduct(grad(nu).Trace(),grad(zt).Trace()))*ds
        Lhs += 1/self.dt*InnerProduct(nu,nut)*ds - InnerProduct(grad(z).Trace(),grad(nut).Trace())*ds
        Lhs += (kappa*InnerProduct(self.nuold,vt))*ds + InnerProduct(grad(v).Trace(),grad(vt).Trace())*ds
        # implicit treatment of z and V
        Lhs += -self.Hold*InnerProduct(A*z,nut)*ds + InnerProduct(A*z,A*nut)*ds
        Lhs += InnerProduct(A,A)*V*Ht*ds
        Lhs += -(InnerProduct(v,grad(self.Hgeo).Trace())*Ht)*ds - InnerProduct(grad(self.ngeo).Trace()*v,nut)*ds
        self.lhs = Lhs

        # 计算需要对Hold,nuold,vold,Vold,zold设定初值
        Rhs = LinearForm(self.fesMix)
        Rhs += Q*Vt*ds
        Rhs += 1/self.dt*self.Hold*Ht*ds 
        Rhs += 1/self.dt*InnerProduct(self.nuold,nut)*ds
        Rhs += (2*InnerProduct(grad(nut).Trace()*(A*grad(self.Hold).Trace()), self.nuold)\
            + InnerProduct(grad(self.Hold).Trace(),grad(self.Hold).Trace())*InnerProduct(self.nuold,nut)\
            + InnerProduct(A*grad(self.Hold).Trace(),A*nut))*ds
        Rhs += Q*Trace(grad(nut).Trace())*ds - Q*self.Hold*InnerProduct(self.nuold,nut)*ds
        Rhs += (InnerProduct(A,A)*InnerProduct(self.nuold,zt))*ds
        self.rhs = Rhs

    def ReIni_HN_ByDisPos(self):
        '''
            3d case: - (Laplace X) * n= H, exterior normal and positive curvature for sphere

            Solving following equation using high order surface fem
            (H,Ht) = sum (grad(X_i),grad(n_i))*Ht + (grad(X_i),grad(Ht))*n_i

            Q = -1/2*H^3 + |A|^2 H
            V = Lap H + Q
            z = Lap n + |A|^2 n
            Only Reinitialize n, H, (v is actually needed by not updated)
        '''
        print('Now Reinit time is {}, Reinit by curvature of high order surface with C_order {} and E_order {}'.format(self.t,self.mesh.GetCurveOrder(),self.order))
        self.Hold.Interpolate(Trace(-specialcf.Weingarten(3)),definedon=self.mesh.Boundaries('.*'))
        self.nuold.Interpolate(-specialcf.normal(3),definedon=self.mesh.Boundaries('.*'))
        self.vold.Interpolate(CF((0,0,0)),definedon=self.mesh.Boundaries('.*'))

    def Solving(self, vtk_obj:Vtk_out_BND=None):
        while self.t<self.T:
            tauval = self.dt.Get()
            self.ngeo.Interpolate(-specialcf.normal(3),definedon=self.mesh.Boundaries('.*'))
            self.Hgeo.Interpolate(Trace(-specialcf.Weingarten(3)),definedon=self.mesh.Boundaries('.*'))
            self.PP_Pic(vtk_obj) # save pic at t, after one step
            self.lhs.Assemble()
            self.rhs.Assemble()
            self.gfu.vec.data = self.lhs.mat.Inverse(inverse="pardiso")*self.rhs.vec
    
            ## Update the Mean curvature and Normal vector old (used in weak formulation)
            self.Hold.vec.data = self.H.vec
            self.nuold.vec.data = self.normal.vec
            self.vold.vec.data = self.velocity.vec
            self.Vold.vec.data = self.V.vec
            self.zold.vec.data = self.z.vec

            BDFInc = self.BDF.Stepping(self.velocity.vec.FV().NumPy()*tauval)
            self.Disp.vec.data += BaseVector(BDFInc)
            self.mesh.SetDeformation(self.Disp)
            
            self.t += tauval
            self.PrintDuring()
        ## 最后一步计算完的时间可能比T略大
        self.finalT = self.t

class RBCLapVWM_ImplicitVz_v_0531(RBCLapVWM_ImplicitVz_v_0530):
    '''
        Implicitly Treat V,z and the convection term.
    '''
    def __init__(self, mymesh, Geo_Rot_Obj: Param2dRot, T=0.0001, dt=0.0001, order=1, BDForder=1):
        print('class RBCLapVWM_ImplicitVz2 is used!')
        super().__init__(mymesh, Geo_Rot_Obj, T, dt, order, BDForder)

    def WeakWillmore(self,SD_opt=False):
        ## Set Curvature
        kappa , H , V , z, nu, v = self.fesMix.TrialFunction()
        kappat, Ht, Vt, zt, nut, vt= self.fesMix.TestFunction()

        ## Weingarten Map
        A0 = grad(self.nuold).Trace()
        A = 1/2*(A0.trans+A0)
        if SD_opt:
            Q = 0
        else:
            Q = -1/2*self.Hold**3+InnerProduct(A,A)*self.Hold

        n_unified = self.nuold/Norm(self.nuold)
        Lhs = BilinearForm(self.fesMix,symmetric=False)
        Lhs += (InnerProduct(v,n_unified)*kappat)*ds-V*kappat*ds
        Lhs += (1/self.dt*H*Ht - InnerProduct(grad(V).Trace(),grad(Ht).Trace()))*ds
        Lhs += (V*Vt + InnerProduct(grad(H).Trace(),grad(Vt).Trace()))*ds
        Lhs += (InnerProduct(z,zt) + InnerProduct(grad(nu).Trace(),grad(zt).Trace()))*ds
        Lhs += 1/self.dt*InnerProduct(nu,nut)*ds - InnerProduct(grad(z).Trace(),grad(nut).Trace())*ds
        Lhs += (kappa*InnerProduct(n_unified,vt))*ds + InnerProduct(grad(v).Trace(),grad(vt).Trace())*ds
        # implicit treatment of z and V
        Lhs += -self.Hold*InnerProduct(A*z,nut)*ds + InnerProduct(A*z,A*nut)*ds
        Lhs += -InnerProduct(grad(nu).Trace()*self.vold,nut)*ds
        Lhs += -InnerProduct(self.vold,grad(H).Trace())*Ht*ds + InnerProduct(A,A)*V*Ht*ds
        # Lhs += -(InnerProduct(grad(H).Trace(),grad(H).Trace())*InnerProduct(nu,nut))*ds 
        self.lhs = Lhs

        # 计算需要对Hold,nuold设定初值
        Rhs = LinearForm(self.fesMix)
        Rhs += Q*Vt*ds
        Rhs += 1/self.dt*self.Hold*Ht*ds 
        Rhs += 1/self.dt*InnerProduct(n_unified,nut)*ds
        Rhs += 2*InnerProduct(grad(nut).Trace()*(A*grad(self.Hold).Trace()), n_unified)*ds
        Rhs += (InnerProduct(grad(self.Hold).Trace(),grad(self.Hold).Trace())*InnerProduct(self.nuold,nut))*ds + InnerProduct(A*grad(self.Hold).Trace(),A*nut)*ds
        Rhs += Q*Trace(grad(nut).Trace())*ds - Q*self.Hold*InnerProduct(n_unified,nut)*ds
        Rhs += (InnerProduct(A,A)*InnerProduct(n_unified,zt))*ds
        self.rhs = Rhs

class RBCLapVWM_ImplicitVz_v_0602(RBCLapVWM_ImplicitVz_v_0530):
    '''
        Implicitly Treat V,z and the convection term.
    '''
    def __init__(self, mymesh, Geo_Rot_Obj: Param2dRot, T=0.0001, dt=0.0001, order=1, BDForder=1):
        print('class RBCLapVWM_ImplicitVz2 is used!')
        super().__init__(mymesh, Geo_Rot_Obj, T, dt, order, BDForder)

    def WeakWillmore(self,SD_opt=False):
        ## Set Curvature
        kappa , H , V , z, nu, v = self.fesMix.TrialFunction()
        kappat, Ht, Vt, zt, nut, vt= self.fesMix.TestFunction()

        ## Weingarten Map
        A0 = grad(self.nuold).Trace()
        A = 1/2*(A0.trans+A0)
        if SD_opt:
            Q = 0
        else:
            Q = -1/2*self.Hold**3+InnerProduct(A,A)*self.Hold

        n_unified = self.nuold
        Lhs = BilinearForm(self.fesMix,symmetric=False)
        Lhs += (InnerProduct(v,n_unified)*kappat)*ds-V*kappat*ds
        Lhs += (1/self.dt*H*Ht - InnerProduct(grad(V).Trace(),grad(Ht).Trace()))*ds
        Lhs += (V*Vt + InnerProduct(grad(H).Trace(),grad(Vt).Trace()))*ds
        Lhs += (InnerProduct(z,zt) + InnerProduct(grad(nu).Trace(),grad(zt).Trace()))*ds
        Lhs += 1/self.dt*InnerProduct(nu,nut)*ds - InnerProduct(grad(z).Trace(),grad(nut).Trace())*ds
        Lhs += (kappa*InnerProduct(n_unified,vt))*ds + InnerProduct(grad(v).Trace(),grad(vt).Trace())*ds
        # implicit treatment of z and V
        Lhs += -self.Hold*InnerProduct(A*z,nut)*ds + InnerProduct(A*z,A*nut)*ds
        Lhs += -InnerProduct(grad(nu).Trace()*self.vold,nut)*ds
        Lhs += -InnerProduct(self.vold,grad(H).Trace())*Ht*ds + InnerProduct(A,A)*V*Ht*ds
        # Lhs += -(InnerProduct(grad(H).Trace(),grad(H).Trace())*InnerProduct(nu,nut))*ds 
        self.lhs = Lhs

        # 计算需要对Hold,nuold设定初值
        Rhs = LinearForm(self.fesMix)
        Rhs += Q*Vt*ds
        Rhs += 1/self.dt*self.Hold*Ht*ds 
        Rhs += 1/self.dt*InnerProduct(n_unified,nut)*ds
        Rhs += 2*InnerProduct(grad(nut).Trace()*(A*grad(self.Hold).Trace()), n_unified)*ds
        Rhs += (InnerProduct(grad(self.Hold).Trace(),grad(self.Hold).Trace())*InnerProduct(self.nuold,nut))*ds + InnerProduct(A*grad(self.Hold).Trace(),A*nut)*ds
        Rhs += Q*Trace(grad(nut).Trace())*ds - Q*self.Hold*InnerProduct(n_unified,nut)*ds
        Rhs += (InnerProduct(A,A)*InnerProduct(n_unified,zt))*ds
        self.rhs = Rhs

class CubicWM(LapVSD):
    def __init__(self, mymesh, DMesh_Obj: DiscreteMesh, T=0.0001, dt=0.0001, order=1, BDForder=1):
        super().__init__(mymesh, T, dt, order, BDForder)
        self.DMesh_Obj = DMesh_Obj

    def IniByDisPos(self):
        '''
            Initialize the weightedNormal, Curvature, V, v, z, X, by discrete mesh
        '''
        # compute the weighted normal on each vertices
        self.DMesh_Obj.WN_Ver()
        self.IniNormal.vec.data = BaseVector(self.DMesh_Obj.WN_Ver_matrix.flatten('F'))
        self.nuold.vec.data = self.IniNormal.vec
        
        # compute the mean curvature by normal using BGN method, -∆ X = Hn
        H_trial, H_test = self.fes.TnT()
        self.IniCurv = GridFunction(self.fes)
        rhs = LinearForm(self.fes)
        rhs += Trace(grad(self.IniNormal).Trace())*H_test*ds
        lhs = BilinearForm(self.fes)
        lhs += H_trial*H_test*ds
        lhs.Assemble()
        rhs.Assemble()
        self.IniCurv.vec.data = lhs.mat.Inverse(inverse='pardiso')*rhs.vec
        self.Hold.vec.data = self.IniCurv.vec

        self.IniV.Interpolate(0,definedon=self.mesh.Boundaries('.*'))
        self.Vold.vec.data = self.IniV.vec

        self.Iniv.Interpolate(CF((0,0,0)),definedon=self.mesh.Boundaries('.*'))
        self.vold.vec.data = self.Iniv.vec

        self.Iniz.Interpolate(CF((0,0,0)),definedon=self.mesh.Boundaries('.*'))
        self.zold.vec.data = self.Iniz.vec

        self.IniX.Interpolate(CoefficientFunction((x,y,z)),definedon=self.mesh.Boundaries(".*"))