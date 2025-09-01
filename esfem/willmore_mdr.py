from ngsolve import *
from ngsolve.comp import IntegrationRuleSpaceSurface
import numpy as np
from esfem.ode import BDF
from esfem.utils import Pos_Transformer, NgMFSave, SurfacehInterp
from esfem.ale import Vtk_out_BND
from geometry import DiscreteMesh, Param2dRot
from global_utils import LogTime
import os
from tqdm import tqdm

class WillMoreMDR():
    def __init__(self, mymesh, T = 1e-4, dt = 1e-4, order = 1, BDForder = 1):
        self.MethodName = 'MDRWillmore'
        self.mesh = mymesh
        self.order = order
        self.fes = H1(self.mesh,order=order,definedon=self.mesh.Boundaries(".*"))
        print('fes order is {}, with ndof is {}'.format(self.order, self.fes.ndof))
        self.fesV = VectorH1(self.mesh,order=order,definedon=self.mesh.Boundaries(".*"))
        self.fesir = IntegrationRuleSpaceSurface(self.mesh, order=order, definedon=self.mesh.Boundaries('.*'))
        self.fesV1 = VectorH1(self.mesh,order=1,definedon=self.mesh.Boundaries('.*'))  # interpolate vertices coordinate
        self.VCoord = GridFunction(self.fesV1)

        self.dim = self.mesh.dim # 2维中的1维曲线dim=2
        self.Disp = GridFunction(self.fesV)  # piecewise high order for displacement --- high order isoparametric
        self.scale = 1

        self.IniDefP = GridFunction(self.fesV) # store the initial deformation
        self.t, self.T, self.finalT = 0, T, 0
        self.dt = Parameter(dt)
        self.lhs, self.rhs = [],[]
        ##              kappa     H        V       z          nu        v
        self.fesMix = self.fes*self.fes*self.fes*self.fesV*self.fesV*self.fesV
        ##              kappa      v   单独求解Laplace v = kappa n以及vn = V的方程
        self.fes_vk = self.fes*self.fesV
        self.gfu    = GridFunction(self.fesMix)
        self.gfuold = GridFunction(self.fesMix)
        self.kappa,    self.H,    self.V,    self.z,    self.normal, self.velocity = self.gfu.components
        self.kappaold, self.Hold, self.Vold, self.zold, self.nuold,  self.vold     = self.gfuold.components
        
        ## 初始化初始值
        self.Iniu   = GridFunction(self.fesMix)
        _, self.IniCurv, self.IniV, self.Iniz, self.IniNormal, self.Iniv = self.Iniu.components
        self.IniX = GridFunction(self.fesV)
        self.BaseX = GridFunction(self.fesV)
        if self.dim==2:
            self.BaseX.Interpolate(CF((x,y)),definedon=self.mesh.Boundaries(".*"))
        elif self.dim==3:
            self.BaseX.Interpolate(CF((x,y,z)),definedon=self.mesh.Boundaries(".*"))

        self.BDF = BDF(order = BDForder)
        self.BDForder = BDForder
        
        ## For Interpolation 
        self.hInterp_Obj = SurfacehInterp(self.mesh,self.order)
        self.counter_print = 1
        self.MQ_Measure_Set()

    def LapvSet(self):
        '''
            MDR的初始的切向速度，需要通过初始的V和n来求解
            Given self.Vold and self.nuold and deformed mesh, solving v from

            <grad v, grad vt> + <n, vt>*kappa = 0

                                <v, n>*kappat = V*kappat
        '''
        kappa,  v  = self.fes_vk.TrialFunction()
        kappat, vt = self.fes_vk.TestFunction()
        v_lhs = BilinearForm(self.fes_vk, symmetric=True)
        v_lhs += (InnerProduct(grad(v).Trace(),grad(vt).Trace()) + kappa*InnerProduct(self.nuold, vt)\
                 + kappat*InnerProduct(self.nuold,v))*ds
        v_rhs = LinearForm(self.fes_vk)
        v_rhs += self.Vold*kappat*ds
        gfu = GridFunction(self.fes_vk)
        gfu.vec.data = v_lhs.Assemble().mat.Inverse(inverse="umfpack")*(v_rhs.Assemble().vec)
        self.vold.vec.data = gfu.components[-1].vec
        
    def MQ_Measure_Set(self):
        Vertices_Coords = self.Get_Vertices_Coords()
        self.ElVerInd, self.EdVerInd = self.Mesh_Info_Parse()
        self.DMesh_Obj = DiscreteMesh(Vertices_Coords,self.dim-1,self.dim,self.ElVerInd,self.EdVerInd)
        self.Mesh_Quality = []
              
    def Mesh_Info_Parse(self):
        if self.dim==3:
            ElVerInd = []
            EdVerInd = []
            for el in self.mesh.ngmesh.Elements2D():
                # v.nr 从1开始计数
                ElVerInd.append([v.nr-1 for v in el.vertices])
            for ed in self.mesh.edges:
                # v.nr 从0开始计数（ngmesh与ngsolve.Mesh之间的区别）
                EdVerInd.append([v.nr for v in ed.vertices])
            ElVerInd, EdVerInd = np.array(ElVerInd), np.array(EdVerInd)
        elif self.dim==2:
            ElVerInd = None
            EdVerInd = None
        return ElVerInd, EdVerInd

    def Get_Vertices_Coords(self):
        if self.dim==3:
            self.VCoord.Interpolate(CF((x,y,z)),definedon=self.mesh.Boundaries(".*"))
            Vertices_Coords = Pos_Transformer(self.VCoord,dim=3)
        elif self.dim==2:
            self.VCoord.Interpolate(CF((x,y)),definedon=self.mesh.Boundaries(".*"))
            Vertices_Coords = Pos_Transformer(self.VCoord,dim=2)
        return Vertices_Coords

    def Get_Proper_dt(self,coef=1,h_opt='min'):
        # 通过顶点的三角形网格来计算最小边长，然后计算需要的时间步长
        # order h**2, independent of scale
        Vertices_Coords = Pos_Transformer(self.Disp, dim=self.dim) + Pos_Transformer(self.BaseX, dim=self.dim)
        self.DMesh_Obj.UpdateCoords(Vertices_Coords)
        h = min(np.linalg.norm(self.DMesh_Obj.barvec,axis=1))
        hmean = np.mean(np.linalg.norm(self.DMesh_Obj.barvec,axis=1))
        if h_opt == 'min':
            dt = coef*h**2
        elif h_opt == 'mean':
            dt = coef*hmean**2
        print('{} :Using Get_Proper_dt, Now coef is {}, t is {}, min h is {}, dt is {}, scale is {}'.format(LogTime(),coef,self.t,h,format(dt,'0.3e'),self.scale))
        return dt

    def IniByGeoObj(self):
        pass

    def IniByFEM(self):
        self.Hold.Interpolate(Trace(-specialcf.Weingarten(3)),definedon=self.mesh.Boundaries('.*'))
        self.nuold.Interpolate(-specialcf.normal(3),definedon=self.mesh.Boundaries('.*'))
        self.vold.Interpolate(CF((0,0,0)),definedon=self.mesh.Boundaries('.*'))
        
    def IniBySaveDat(self,SavePath):
        '''
            Set Disp, t, scale, Hold, nuold, vold, Vold, zold by Saved Data
        '''
        info = np.load(os.path.join(SavePath,'info.npy'),allow_pickle=True).item()
        self.t, self.scale = [info[ii] for ii in ['t','scale']]
        self.Disp.Load(os.path.join(SavePath,'Disp'))
        self.Hold.Load(os.path.join(SavePath,'Hold'))
        self.nuold.Load(os.path.join(SavePath,'nuold'))
        self.vold.Load(os.path.join(SavePath,'vold'))
        self.Vold.Load(os.path.join(SavePath,'Vold'))
        self.zold.Load(os.path.join(SavePath,'zold'))

    def SaveFunc(self,BaseDirPath):
        Case_Name = 'C_'+format(self.T,'0.3e').replace('.','_').replace('-','_')
        flist = [self.Disp, self.Hold, self.vold, self.nuold, self.Vold, self.zold]
        fnlist = ['Disp', 'Hold', 'vold', 'nuold', 'Vold', 'zold']
        t_scale_dict = {'t': self.t, 'scale': self.scale}
        NgMFSave(CaseName=Case_Name, BaseDirPath=BaseDirPath, mesh=self.mesh, funclist=flist, func_name_list=fnlist, data_dict=t_scale_dict)
        print('Now T={}, Save Functions in {}'.format(self.T,Case_Name))

    def WeakWillmore(self,SD_opt=False,is_unified=True):
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

        if is_unified:
            n_unified = self.nuold/Norm(self.nuold)
        else:
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
        
    def PP_Pic(self,vtk_obj:Vtk_out_BND=None):
        # Post Process: Saving Picture
        pass

    def Solving(self, vtk_obj:Vtk_out_BND=None, sceneu=None):
        '''
            从t时间求解到T时间 
        '''
        tauval = self.dt.Get()
        num_steps = int((self.T-self.t) / tauval)
        with tqdm(total=num_steps, desc="Time stepping") as pbar:
            while self.t<self.T:
                # tauval = self.dt.Get() # Willmore flow应该不需要每一步都更新dt
                if vtk_obj is not None:
                    self.PP_Pic(vtk_obj) # save pic at t, after one step
                self.lhs.Assemble()
                self.rhs.Assemble()
                self.gfu.vec.data = self.lhs.mat.Inverse(inverse="umfpack")*self.rhs.vec
        
                ## Update the Mean curvature and Normal vector old (used in weak formulation)
                self.gfuold.vec.data = self.gfu.vec

                BDFInc = self.BDF.Stepping(self.velocity.vec.FV().NumPy()*tauval)
                self.Disp.vec.data += BaseVector(BDFInc)
                self.mesh.SetDeformation(self.Disp)
                
                if sceneu is not None:
                    sceneu.Redraw()
                self.t += tauval
                pbar.update(1)
        ## 最后一步计算完的时间可能比T略大
        self.finalT = self.t

class WillmoreKLL(WillMoreMDR):
    def __init__(self, mymesh, T=0.0001, dt=0.0001, order=1, BDForder=1):
        super().__init__(mymesh, T, dt, order, BDForder)
        
        ##                H        V       z          nu      
        self.fesMix = self.fes*self.fes*self.fesV*self.fesV
        self.gfu    = GridFunction(self.fesMix)
        self.gfuold = GridFunction(self.fesMix)
        self.H,    self.V,    self.z,    self.normal = self.gfu.components
        self.Hold, self.Vold, self.zold, self.nuold  = self.gfuold.components
        self.velocity = GridFunction(self.fesV)  # only for storing velocity

    def WeakWillmore(self,SD_opt=False,is_unified=True):
        # KLL算法的弱形式
        ## Set Curvature
        H , V , z,  nu  = self.fesMix.TrialFunction()
        Ht, Vt, zt, nut = self.fesMix.TestFunction()

        ## Weingarten Map
        A0 = grad(self.nuold).Trace()
        A = 1/2*(A0.trans+A0)
        if SD_opt:
            Q = 0
        else:
            Q = -1/2*self.Hold**3+InnerProduct(A,A)*self.Hold

        if is_unified:
            n_unified = self.nuold/Norm(self.nuold)
        else:
            n_unified = self.nuold
        Lhs = BilinearForm(self.fesMix,symmetric=False)
        Rhs = LinearForm(self.fesMix)
        # H的演化方程离散
        Lhs += (1/self.dt*H*Ht - InnerProduct(grad(V).Trace(),grad(Ht).Trace()))*ds
        Lhs += InnerProduct(A,A)*V*Ht*ds
        Rhs += 1/self.dt*self.Hold*Ht*ds 
        # V的弱形式
        Lhs += (V*Vt + InnerProduct(grad(H).Trace(),grad(Vt).Trace()))*ds
        Rhs += Q*Vt*ds
        # n的演化方程离散
        Lhs += 1/self.dt*InnerProduct(nu,nut)*ds - InnerProduct(grad(z).Trace(),grad(nut).Trace())*ds
        Rhs += 1/self.dt*InnerProduct(n_unified,nut)*ds
        Lhs += -self.Hold*InnerProduct(A*z,nut)*ds + InnerProduct(A*z,A*nut)*ds
        Rhs += (InnerProduct(grad(self.Hold).Trace(),grad(self.Hold).Trace())*InnerProduct(self.nuold,nut))*ds \
                + InnerProduct(A*grad(self.Hold).Trace(),A*nut)*ds \
                + 2*InnerProduct(grad(nut).Trace()*(A*grad(self.Hold).Trace()), n_unified)*ds
        Rhs += Q*Trace(grad(nut).Trace())*ds - Q*self.Hold*InnerProduct(n_unified,nut)*ds
        # z的弱形式
        Lhs += (InnerProduct(z,zt) + InnerProduct(grad(nu).Trace(),grad(zt).Trace()))*ds
        Rhs += (InnerProduct(A,A)*InnerProduct(n_unified,zt))*ds
        self.lhs = Lhs
        self.rhs = Rhs
        
    def Solving(self, vtk_obj:Vtk_out_BND=None, sceneu=None):
        '''
            从t时间求解到T时间 
        '''
        tauval = self.dt.Get()
        num_steps = int((self.T-self.t) / tauval)
        with tqdm(total=num_steps, desc="Time stepping") as pbar:
            while self.t<self.T:
                # tauval = self.dt.Get() # Willmore flow应该不需要每一步都更新dt
                if vtk_obj is not None:
                    self.PP_Pic(vtk_obj) # save pic at t, after one step
                self.lhs.Assemble()
                self.rhs.Assemble()
                self.gfu.vec.data = self.lhs.mat.Inverse(inverse="umfpack")*self.rhs.vec
        
                ## Update the Mean curvature and Normal vector old (used in weak formulation)
                self.gfuold.vec.data = self.gfu.vec
                
                self.velocity.Interpolate(self.gfu.components[1]*self.gfu.components[-1],
                                          definedon=self.mesh.Boundaries(".*"))            

                BDFInc = self.BDF.Stepping(self.velocity.vec.FV().NumPy()*tauval)
                self.Disp.vec.data += BaseVector(BDFInc)
                self.mesh.SetDeformation(self.Disp)
                
                if sceneu is not None:
                    sceneu.Redraw()
                self.t += tauval
                pbar.update(1)
        ## 最后一步计算完的时间可能比T略大
        self.finalT = self.t
        
class SDMDR(WillMoreMDR):
    def __init__(self, mymesh, T=0.0001, dt=0.0001, order=1, BDForder=1):
        super().__init__(mymesh, T, dt, order, BDForder)

    def WeakSD(self):
        # Surface Diffusion is a variant of Willmore with Q = 0
        super().WeakWillmore(SD_opt=True)