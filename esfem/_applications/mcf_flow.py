from ..mcf_mdr import LapVDNMCF_v2
from ngsolve import *
from geometry import Param2dRot
from es_utils import pos_transformer
from esfem.ale import VtkOutBnd
from ..bgn import BGN_MCF

class DumbbellLapVMCF(LapVDNMCF_v2):
    def __init__(self, mymesh, Geo_Rot_Obj:Param2dRot, T=0.0001, dt=0.0001, order=1, BDForder=1, reset_opt='Rel'):
        super().__init__(mymesh, T, dt, order, BDForder)
        self.Dumbbell2d = Geo_Rot_Obj
        self.Reset_opt = reset_opt
        print('Reset_opt is {}'.format(self.Reset_opt))
        # mean curvature of the rotational geometry
        self.H_np = self.Dumbbell2d.H_np

    def PP_Pic(self,vtk_Obj_bnd:VtkOutBnd=None):
        self.Compu_HN_ByDisPos(opt=self.Reset_opt)
        # 后处理： 保存图像
        perform_res = vtk_Obj_bnd.output(self.mesh,[self.H_X,self.Hold,self.err_H_0,self.n_Norm,self.vTNorm,self.n_err],tnow=self.t,names=['BGN','Hold','eH2_old','nNorm','vTNorm','en2_old'])
        # 后处理： 计算网格质量，利用self.mesh的拓扑关系
        if perform_res:
            self.VCoord.Interpolate(CF((x,y,z)),definedon=self.mesh.Boundaries(".*"))
            self.DMesh_Obj.UpdateCoords(pos_transformer(self.VCoord,dim=3))
            Q_Area, Q_Leng = self.DMesh_Obj.MeshQuality()
            self.Area_set.append([self.DMesh_Obj.Area,self.t,self.scale])
            self.Mesh_Quality.append([Q_Area,Q_Leng,self.t])

class DumbbellLapVMCF_v_0531(DumbbellLapVMCF):
    def __init__(self, mymesh, Geo_Rot_Obj: Param2dRot, T=0.0001, dt=0.0001, order=1, BDForder=1, reset_opt='Rel'):
        super().__init__(mymesh, Geo_Rot_Obj, T, dt, order, BDForder, reset_opt=reset_opt)
        print('Unify all normal by projection to sphere!!')

    def WeakMCF(self):
        ## Set Curvature
        kappa , H , nu, v = self.fesMix.TrialFunction()
        kappat, Ht, nut, vt= self.fesMix.TestFunction()

        ## Weingarten Map A = grad n
        A0 = grad(self.nuold).Trace()
        A = 1/2*(A0.trans+A0)

        n_unified = self.nuold/Norm(self.nuold)
        # update self.vold, self.nuold, self.Hold
        Lhs = BilinearForm(self.fesMix,symmetric=False)
        Lhs += InnerProduct(v,n_unified)*kappat*ds
        Lhs += (kappa*InnerProduct(n_unified,vt))*ds + InnerProduct(grad(v).Trace(),grad(vt).Trace())*ds
        Lhs += 1/self.dt*InnerProduct(nu,nut)*ds + InnerProduct(grad(nu).Trace(),grad(nut).Trace())*ds
        Lhs += 1/self.dt*H*Ht*ds + InnerProduct(grad(H).Trace(),grad(Ht).Trace())*ds
        Lhs += -InnerProduct(grad(nu).Trace()*self.vold,nut)*ds
        Lhs += -InnerProduct(self.vold,grad(H).Trace())*Ht*ds
        self.lhs = Lhs

        Rhs = LinearForm(self.fesMix)
        Rhs += -self.Hold*kappat*ds
        Rhs += 1/self.dt*InnerProduct(n_unified,nut)*ds
        Rhs += (InnerProduct(A,A)*InnerProduct(n_unified,nut))*ds
        Rhs += 1/self.dt*self.Hold*Ht*ds
        Rhs += (InnerProduct(A,A)*self.Hold*Ht)*ds
        self.rhs = Rhs

class DumbbellLapVMCF_v_0601(DumbbellLapVMCF):
    def __init__(self, mymesh, Geo_Rot_Obj: Param2dRot, T=0.0001, dt=0.0001, order=1, BDForder=1):
        super().__init__(mymesh, Geo_Rot_Obj, T, dt, order, BDForder)
        print('Remove all unification of normal!!')

    def WeakMCF(self):
        ## Set Curvature
        kappa , H , nu, v = self.fesMix.TrialFunction()
        kappat, Ht, nut, vt= self.fesMix.TestFunction()

        ## Weingarten Map A = grad n
        A = grad(self.nuold).Trace()

        n_unified = self.nuold
        # update self.vold, self.nuold, self.Hold
        Lhs = BilinearForm(self.fesMix,symmetric=False)
        Lhs += InnerProduct(v,n_unified)*kappat*ds
        Lhs += (kappa*InnerProduct(n_unified,vt))*ds + InnerProduct(grad(v).Trace(),grad(vt).Trace())*ds
        Lhs += 1/self.dt*InnerProduct(nu,nut)*ds + InnerProduct(grad(nu).Trace(),grad(nut).Trace())*ds
        Lhs += 1/self.dt*H*Ht*ds + InnerProduct(grad(H).Trace(),grad(Ht).Trace())*ds
        Lhs += -InnerProduct(grad(nu).Trace()*self.vold,nut)*ds
        Lhs += -InnerProduct(self.vold,grad(H).Trace())*Ht*ds
        self.lhs = Lhs

        Rhs = LinearForm(self.fesMix)
        Rhs += -self.Hold*kappat*ds
        Rhs += 1/self.dt*InnerProduct(n_unified,nut)*ds
        Rhs += (InnerProduct(A,A)*InnerProduct(n_unified,nut))*ds
        Rhs += 1/self.dt*self.Hold*Ht*ds
        Rhs += (InnerProduct(A,A)*self.Hold*Ht)*ds
        self.rhs = Rhs

class DumbbellLapVMCF_v_0602(DumbbellLapVMCF):
    def __init__(self, mymesh, Geo_Rot_Obj: Param2dRot, T=0.0001, dt=0.0001, order=1, BDForder=1):
        super().__init__(mymesh, Geo_Rot_Obj, T, dt, order, BDForder)
        print('partial unified in normal, when performing v.*n!!')

    def WeakMCF(self):
        ## Set Curvature
        kappa , H , nu, v = self.fesMix.TrialFunction()
        kappat, Ht, nut, vt= self.fesMix.TestFunction()

        ## Weingarten Map A = grad n
        A = grad(self.nuold).Trace()

        n_unified = self.nuold/Norm(self.nuold)
        # update self.vold, self.nuold, self.Hold
        Lhs = BilinearForm(self.fesMix,symmetric=False)
        Lhs += InnerProduct(v,n_unified)*kappat*ds
        Lhs += (kappa*InnerProduct(n_unified,vt))*ds + InnerProduct(grad(v).Trace(),grad(vt).Trace())*ds
        Lhs += 1/self.dt*InnerProduct(nu,nut)*ds + InnerProduct(grad(nu).Trace(),grad(nut).Trace())*ds
        Lhs += 1/self.dt*H*Ht*ds + InnerProduct(grad(H).Trace(),grad(Ht).Trace())*ds
        Lhs += -InnerProduct(grad(nu).Trace()*self.vold,nut)*ds
        Lhs += -InnerProduct(self.vold,grad(H).Trace())*Ht*ds
        self.lhs = Lhs

        Rhs = LinearForm(self.fesMix)
        Rhs += -self.Hold*kappat*ds
        Rhs += 1/self.dt*InnerProduct(self.nuold,nut)*ds
        Rhs += (InnerProduct(A,A)*InnerProduct(self.nuold,nut))*ds
        Rhs += 1/self.dt*self.Hold*Ht*ds
        Rhs += (InnerProduct(A,A)*self.Hold*Ht)*ds
        self.rhs = Rhs

class DumbbellLapVMCF_v_0603(DumbbellLapVMCF):
    def __init__(self, mymesh, Geo_Rot_Obj: Param2dRot, T=0.0001, dt=0.0001, order=1, BDForder=1, reset_opt='Rel'):
        super().__init__(mymesh, Geo_Rot_Obj, T, dt, order, BDForder, reset_opt)
        print('Remove all unification of normal!! All terms concerning H implicit!!')

    def WeakMCF(self):
        ## Set Curvature
        kappa , H , nu, v = self.fesMix.TrialFunction()
        kappat, Ht, nut, vt= self.fesMix.TestFunction()

        ## Weingarten Map A = grad n
        A = grad(self.nuold).Trace()

        n_unified = self.nuold
        # update self.vold, self.nuold, self.Hold
        Lhs = BilinearForm(self.fesMix,symmetric=False)
        Lhs += InnerProduct(v,n_unified)*kappat*ds
        Lhs += (kappa*InnerProduct(n_unified,vt))*ds + InnerProduct(grad(v).Trace(),grad(vt).Trace())*ds
        Lhs += 1/self.dt*InnerProduct(nu,nut)*ds + InnerProduct(grad(nu).Trace(),grad(nut).Trace())*ds
        Lhs += 1/self.dt*H*Ht*ds + InnerProduct(grad(H).Trace(),grad(Ht).Trace())*ds
        Lhs += -InnerProduct(grad(nu).Trace()*self.vold,nut)*ds
        Lhs += -InnerProduct(self.vold,grad(H).Trace())*Ht*ds
        Lhs += -(InnerProduct(A,A)*H*Ht)*ds
        self.lhs = Lhs

        Rhs = LinearForm(self.fesMix)
        Rhs += -self.Hold*kappat*ds
        Rhs += 1/self.dt*InnerProduct(n_unified,nut)*ds
        Rhs += (InnerProduct(A,A)*InnerProduct(n_unified,nut))*ds
        Rhs += 1/self.dt*self.Hold*Ht*ds
        self.rhs = Rhs

class DumbbellLapVMCF_v_0604(DumbbellLapVMCF):
    def __init__(self, mymesh, Geo_Rot_Obj: Param2dRot, T=0.0001, dt=0.0001, order=1, BDForder=1):
        super().__init__(mymesh, Geo_Rot_Obj, T, dt, order, BDForder)
        print('Remove all unification of normal!! Treating curvature in normal velocity in an implicity way.')

    def WeakMCF(self):
        ## Set Curvature
        kappa , H , nu, v = self.fesMix.TrialFunction()
        kappat, Ht, nut, vt= self.fesMix.TestFunction()

        ## Weingarten Map A = grad n
        A = grad(self.nuold).Trace()

        n_unified = self.nuold
        # update self.vold, self.nuold, self.Hold
        Lhs = BilinearForm(self.fesMix,symmetric=False)
        Lhs += InnerProduct(v,n_unified)*kappat*ds + H*kappat*ds
        Lhs += (kappa*InnerProduct(n_unified,vt))*ds + InnerProduct(grad(v).Trace(),grad(vt).Trace())*ds
        Lhs += 1/self.dt*InnerProduct(nu,nut)*ds + InnerProduct(grad(nu).Trace(),grad(nut).Trace())*ds
        Lhs += 1/self.dt*H*Ht*ds + InnerProduct(grad(H).Trace(),grad(Ht).Trace())*ds
        Lhs += -InnerProduct(grad(nu).Trace()*self.vold,nut)*ds
        Lhs += -InnerProduct(self.vold,grad(H).Trace())*Ht*ds
        self.lhs = Lhs

        Rhs = LinearForm(self.fesMix)
        Rhs += 1/self.dt*InnerProduct(n_unified,nut)*ds
        Rhs += (InnerProduct(A,A)*InnerProduct(n_unified,nut))*ds
        Rhs += 1/self.dt*self.Hold*Ht*ds
        Rhs += (InnerProduct(A,A)*self.Hold*Ht)*ds
        self.rhs = Rhs

class DumbbellLapVMCF_v_0605(DumbbellLapVMCF):
    def __init__(self, mymesh, Geo_Rot_Obj: Param2dRot, T=0.0001, dt=0.0001, order=1, BDForder=1):
        super().__init__(mymesh, Geo_Rot_Obj, T, dt, order, BDForder)
        print('Treat v\cdot grad H')

    def WeakMCF(self):
        ## Set Curvature
        kappa , H , nu, v = self.fesMix.TrialFunction()
        kappat, Ht, nut, vt= self.fesMix.TestFunction()

        ## Weingarten Map A = grad n
        A = grad(self.nuold).Trace()

        n_unified = self.nuold
        # update self.vold, self.nuold, self.Hold
        Lhs = BilinearForm(self.fesMix,symmetric=False)
        Lhs += InnerProduct(v,n_unified)*kappat*ds + H*kappat*ds
        Lhs += (kappa*InnerProduct(n_unified,vt))*ds + InnerProduct(grad(v).Trace(),grad(vt).Trace())*ds
        Lhs += 1/self.dt*InnerProduct(nu,nut)*ds + InnerProduct(grad(nu).Trace(),grad(nut).Trace())*ds
        Lhs += 1/self.dt*H*Ht*ds + InnerProduct(grad(H).Trace(),grad(Ht).Trace())*ds
        Lhs += -InnerProduct(grad(nu).Trace()*self.vold,nut)*ds
        Lhs += InnerProduct(grad(self.vold).Trace(),grad(nu).Trace())*Ht*ds\
            + InnerProduct(self.vold,grad(nu).Trace()*grad(Ht).Trace())*ds
        self.lhs = Lhs

        Rhs = LinearForm(self.fesMix)
        Rhs += 1/self.dt*InnerProduct(n_unified,nut)*ds
        Rhs += InnerProduct(A,A)*InnerProduct(n_unified,nut)*ds
        Rhs += 1/self.dt*self.Hold*Ht*ds
        self.rhs = Rhs

class DumbbellLapVMCF_v_0604_impv(DumbbellLapVMCF):
    def __init__(self, mymesh, Geo_Rot_Obj: Param2dRot, T=0.0001, dt=0.0001, order=1, BDForder=1):
        super().__init__(mymesh, Geo_Rot_Obj, T, dt, order, BDForder)
        print('Remove all unification of normal!! Treating curvature in normal velocity in an implicity way. Implicit v')
        print('Caution!! Totally wrong!!')

    def WeakMCF(self):
        ## Set Curvature
        kappa , H , nu, v = self.fesMix.TrialFunction()
        kappat, Ht, nut, vt= self.fesMix.TestFunction()

        ## Weingarten Map A = grad n
        A = grad(self.nuold).Trace()

        n_unified = self.nuold
        # update self.vold, self.nuold, self.Hold
        Lhs = BilinearForm(self.fesMix,symmetric=False)
        Lhs += InnerProduct(v,n_unified)*kappat*ds + H*kappat*ds
        Lhs += (kappa*InnerProduct(n_unified,vt))*ds + InnerProduct(grad(v).Trace(),grad(vt).Trace())*ds
        Lhs += 1/self.dt*InnerProduct(nu,nut)*ds + InnerProduct(grad(nu).Trace(),grad(nut).Trace())*ds
        Lhs += 1/self.dt*H*Ht*ds + InnerProduct(grad(H).Trace(),grad(Ht).Trace())*ds
        Lhs += -InnerProduct(grad(nu).Trace()*v,nut)*ds
        Lhs += -InnerProduct(v,grad(H).Trace())*Ht*ds
        self.lhs = Lhs

        Rhs = LinearForm(self.fesMix)
        Rhs += 1/self.dt*InnerProduct(n_unified,nut)*ds
        Rhs += (InnerProduct(A,A)*InnerProduct(n_unified,nut))*ds
        Rhs += 1/self.dt*self.Hold*Ht*ds
        Rhs += (InnerProduct(A,A)*self.Hold*Ht)*ds
        self.rhs = Rhs

class DumbbellLapVMCF_v_0607(DumbbellLapVMCF):
    def __init__(self, mymesh, Geo_Rot_Obj: Param2dRot, T=0.0001, dt=0.0001, order=1, BDForder=1):
        super().__init__(mymesh, Geo_Rot_Obj, T, dt, order, BDForder)
        print('partial unified in normal, when performing v.*n=-H implicitly!!')

    def WeakMCF(self):
        ## Set Curvature
        kappa , H , nu, v = self.fesMix.TrialFunction()
        kappat, Ht, nut, vt= self.fesMix.TestFunction()

        ## Weingarten Map A = grad n
        A = grad(self.nuold).Trace()

        n_unified = self.nuold/Norm(self.nuold)
        # update self.vold, self.nuold, self.Hold
        Lhs = BilinearForm(self.fesMix,symmetric=False)
        Lhs += InnerProduct(v,n_unified)*kappat*ds + H*kappat*ds
        Lhs += (kappa*InnerProduct(n_unified,vt))*ds + InnerProduct(grad(v).Trace(),grad(vt).Trace())*ds
        Lhs += 1/self.dt*InnerProduct(nu,nut)*ds + InnerProduct(grad(nu).Trace(),grad(nut).Trace())*ds
        Lhs += 1/self.dt*H*Ht*ds + InnerProduct(grad(H).Trace(),grad(Ht).Trace())*ds
        Lhs += -InnerProduct(grad(nu).Trace()*self.vold,nut)*ds
        Lhs += -InnerProduct(self.vold,grad(H).Trace())*Ht*ds
        self.lhs = Lhs

        Rhs = LinearForm(self.fesMix)
        Rhs += 1/self.dt*InnerProduct(self.nuold,nut)*ds
        Rhs += (InnerProduct(A,A)*InnerProduct(self.nuold,nut))*ds
        Rhs += 1/self.dt*self.Hold*Ht*ds
        Rhs += (InnerProduct(A,A)*self.Hold*Ht)*ds
        self.rhs = Rhs

class DumbbellLapVMCF_N_modified(DumbbellLapVMCF):
    def __init__(self, mymesh, Geo_Rot_Obj: Param2dRot, T=0.0001, dt=0.0001, order=1, BDForder=1):
        super().__init__(mymesh, Geo_Rot_Obj, T, dt, order, BDForder)
    
    def Solving(self, vtk_obj: VtkOutBnd = None):
        while self.t<self.T:
            tauval = self.dt.Get()
            self.lhs.Assemble()
            self.rhs.Assemble()
            self.gfu.vec.data = self.lhs.mat.Inverse(inverse="pardiso")*self.rhs.vec
    
            ## Update the Mean curvature and Normal vector old (used in weak formulation)
            testvec = sum(self.H.vec.FV().NumPy())
            self.Hold.vec.data = self.H.vec
            self.nuold.vec.data = self.normal.vec
            self.NormalizeN()
            self.vold.vec.data = self.velocity.vec
            BDFInc = self.BDF.Stepping(self.velocity.vec.FV().NumPy()*tauval)
            self.Disp.vec.data += BaseVector(BDFInc)
            self.mesh.SetDeformation(self.Disp)

            self.t += tauval/self.scale**2
            self.PP_Pic(vtk_obj)
            self.PrintDuring()
        ## 最后一步计算完的时间可能比T略大
        self.finalT = self.t

    def WeakMCF(self):
        ## Set Curvature
        kappa , H , nu, v = self.fesMix.TrialFunction()
        kappat, Ht, nut, vt= self.fesMix.TestFunction()

        vT = self.vold
        ## Weingarten Map A = grad n
        A = grad(self.nuold).Trace()
        A = A.trans
        # update self.vold, self.nuold, self.Hold
        Lhs = BilinearForm(self.fesMix,symmetric=False)
        Lhs += InnerProduct(v,self.nuold)*kappat*ds
        Lhs += (kappa*InnerProduct(self.nuold,vt))*ds + InnerProduct(grad(v).Trace(),grad(vt).Trace())*ds
        Lhs += 1/self.dt*InnerProduct(nu,nut)*ds + InnerProduct(grad(nu).Trace(),grad(nut).Trace())*ds
        Lhs += 1/self.dt*H*Ht*ds + InnerProduct(grad(H).Trace(),grad(Ht).Trace())*ds
        self.lhs = Lhs

        Rhs = LinearForm(self.fesMix)
        Rhs += -self.Hold*kappat*ds
        Rhs += 1/self.dt*InnerProduct(self.nuold,nut)*ds
        Rhs += (InnerProduct(A,A)*InnerProduct(self.nuold,nut)+InnerProduct(grad(self.nuold).Trace()*vT,nut))*ds
        Rhs += 1/self.dt*self.Hold*Ht*ds
        Rhs += (InnerProduct(A,A)*self.Hold*Ht+InnerProduct(vT,grad(self.Hold).Trace())*Ht)*ds
        self.rhs = Rhs

class Dumbbell_BGN_MCF(BGN_MCF):
    def __init__(self, mymesh, T=0.0001, dt=0.0001):
        super().__init__(mymesh, T, dt)

    def PP_Pic(self,vtk_Obj_bnd:VtkOutBnd=None):
        # 后处理： 保存图像
        perform_res = vtk_Obj_bnd.output(self.mesh,[],tnow=self.t)
        # 后处理： 计算网格质量，利用self.mesh的拓扑关系
        if perform_res:
            self.DMesh_Obj.UpdateCoords(pos_transformer(self.X_old,dim=3))
            Q_Area, Q_Leng = self.DMesh_Obj.MeshQuality()
            self.Area_set.append([self.DMesh_Obj.Area,self.t,self.scale])
            self.Mesh_Quality.append([Q_Area,Q_Leng,self.t])

class Dumbbell_BGN_MCF_Mod(BGN_MCF):
    def __init__(self, mymesh, T=0.0001, dt=0.0001):
        super().__init__(mymesh, T, dt)
    
    def PP_Pic(self,vtk_Obj_bnd:VtkOutBnd=None):
        # 后处理： 保存图像
        perform_res = vtk_Obj_bnd.output(self.mesh,[],tnow=self.t)
        # 后处理： 计算网格质量，利用self.mesh的拓扑关系
        if perform_res:
            self.DMesh_Obj.UpdateCoords(pos_transformer(self.X_old,dim=3))
            Q_Area, Q_Leng = self.DMesh_Obj.MeshQuality()
            self.Mesh_Quality.append([Q_Area,Q_Leng,self.t])
    
    def WeakMCF(self):
        '''
            Mixed Formulation for kappa and velocity
            v n = -kappa (曲率纠正符号)    ->      v n + kappa = 0
            (in weak form)
            v n chi + kappa chi = 0
            -kappa n = Lap Xnew 
            (in weak form)
            -kappa n eta + <grad Xnew, grad eta> = 0      
            ->       
            -kappa n eta + dt * <grad v, grad eta> =  -<grad Xold, grad eta>
        '''
        kappa, v = self.fesMix.TrialFunction()
        chi, eta = self.fesMix.TestFunction()

        self.lhs = BilinearForm(self.fesMix)
        self.lhs += InnerProduct(v,self.Nvec)*chi*self.ds_lumping + kappa*chi*self.ds_lumping
        self.lhs += -InnerProduct(eta,self.Nvec)*kappa*self.ds_lumping + self.dt*InnerProduct(grad(v).Trace(),grad(eta).Trace())*ds

        self.rhs = LinearForm(self.fesMix)
        self.rhs += -InnerProduct(grad(self.X_old).Trace(),grad(eta).Trace())*ds

    def Solving(self,vtk_obj):
        while self.t<self.T:
            tauval = self.dt.Get()
            self.mesh.SetDeformation(self.Disp)
            self.GetNvec()
            self.lhs.Assemble()
            self.rhs.Assemble()
            self.Solution.vec.data = self.lhs.mat.Inverse(inverse="pardiso")*self.rhs.vec

            self.Disp.vec.data = BaseVector(self.Disp.vec.FV().NumPy() + tauval*self.Solution.components[1].vec.FV().NumPy())
            self.X_old.vec.data = BaseVector(self.X_old.vec.FV().NumPy() + tauval*self.Solution.components[1].vec.FV().NumPy())
            self.PP_Pic(vtk_obj)
            self.PrintDuring()
            self.t += tauval/self.scale**2

class LapVMCF_v_0605(DumbbellLapVMCF_v_0605):
    def __init__(self, mymesh, Geo_Rot_Obj: Param2dRot, T=0.0001, dt=0.0001, order=1, BDForder=1):
        super().__init__(mymesh, Geo_Rot_Obj, T, dt, order, BDForder)