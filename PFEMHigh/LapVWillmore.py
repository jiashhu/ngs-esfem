from ngsolve import *
from ngsolve.comp import IntegrationRuleSpaceSurface
import numpy as np
from Package_ODE import BDF
from Package_MyNgFunc import Pos_Transformer, NgMFSave, SurfacehInterp
from Package_ALE_Geometry import Vtk_out_BND
from Package_Geometry_Obj import DiscreteMesh, Param2dRot
from Package_MyCode import LogTime
import os

class LapVWillMore():
    def __init__(self, mymesh, T = 1e-4, dt = 1e-4, order = 1, BDForder = 1):
        self.MethodName = 'LapVWillmore'
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
        self.Iniu   = GridFunction(self.fesMix)
        self.kappa,    self.H,    self.V,    self.z,    self.normal, self.velocity = self.gfu.components
        self.kappaold, self.Hold, self.Vold, self.zold, self.nuold,  self.vold     = self.gfuold.components
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

    def MQ_Measure_Set(self):
        Vertices_Coords = self.Get_Vertices_Coords()
        self.ElVerInd, self.EdVerInd = self.Mesh_Info_Parse()
        self.DMesh_Obj = DiscreteMesh(Vertices_Coords,self.dim-1,self.dim,self.ElVerInd,self.EdVerInd)
        self.Mesh_Quality = []

    def Get_Proper_dt(self,coef=1,h_opt='min'):
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

    def LapvSet(self):
        '''
            Given Vold and nuold and deformed mesh, solving v from

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
        gfu.vec.data = v_lhs.Assemble().mat.Inverse(inverse="pardiso")*(v_rhs.Assemble().vec)
        self.vold.vec.data = gfu.components[-1].vec

    def PrintDuring(self):
        if self.t > self.T*self.counter_print/5:
            print('Completed {} per cent'.format(int(self.counter_print/5*100)))
            self.counter_print += 1

    def IniByGeoObj(self):
        pass

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

    def WeakWillmore(self,SD_opt=False):
        ## Set Curvature
        kappa , H , V , z, nu, v = self.fesMix.TrialFunction()
        kappat, Ht, Vt, zt, nut, vt= self.fesMix.TestFunction()

        # 预设速度
        vT = self.vold - self.nuold*InnerProduct(self.vold,self.nuold)/InnerProduct(self.nuold,self.nuold)
        ## Weingarten Map
        A = grad(self.nuold).Trace()
        A = 1/2*(A.trans+A)
        if SD_opt:
            Q = 0
        else:
            Q = -self.Hold**3/2+InnerProduct(A,A)*self.Hold

        Lhs = BilinearForm(self.fesMix,symmetric=False)
        ## self.ds may blow up
        Lhs += (InnerProduct(v,self.nuold)*kappat)*ds-V*kappat*ds
        Lhs += (1/self.dt*H*Ht - InnerProduct(grad(V).Trace(),grad(Ht).Trace()))*ds
        Lhs += (V*Vt + InnerProduct(grad(H).Trace(),grad(Vt).Trace()))*ds
        Lhs += (InnerProduct(z,zt) + InnerProduct(grad(nu).Trace(),grad(zt).Trace()))*ds
        Lhs += 1/self.dt*InnerProduct(nu,nut)*ds - InnerProduct(grad(z).Trace(),grad(nut).Trace())*ds
        Lhs += (kappa*InnerProduct(self.nuold,vt))*ds + InnerProduct(grad(v).Trace(),grad(vt).Trace())*ds
        self.lhs = Lhs

        # 计算需要对Hold,nuold,vold,Vold,zold设定初值
        Rhs = LinearForm(self.fesMix)
        Rhs += Q*Vt*ds
        Rhs += 1/self.dt*self.Hold*Ht*ds 
        Rhs += (InnerProduct(vT,grad(self.Hold).Trace())*Ht-InnerProduct(A,A)*self.Vold*Ht)*ds
        Rhs += 1/self.dt*InnerProduct(self.nuold,nut)*ds
        Rhs += (InnerProduct(grad(self.nuold).Trace()*vT,nut)\
            + self.Hold*InnerProduct(A*self.zold,nut) - InnerProduct(A*self.zold,A*nut)\
            + 2*InnerProduct(grad(nut).Trace()*(A*grad(self.Hold).Trace()), self.nuold)\
            + InnerProduct(grad(self.Hold).Trace(),grad(self.Hold).Trace())*InnerProduct(self.nuold,nut)\
            + InnerProduct(A*grad(self.Hold).Trace(),A*nut))*ds
        Rhs += Q*Trace(grad(nut).Trace())*ds - Q*self.Hold*InnerProduct(self.nuold,nut)*ds
        Rhs += (InnerProduct(A,A)*InnerProduct(self.nuold,zt))*ds
        self.rhs = Rhs
        
    def PP_Pic(self,vtk_obj:Vtk_out_BND=None):
        # Post Process: Saving Picture
        pass

    def Solving(self, vtk_obj:Vtk_out_BND=None):
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
            self.PP_Pic(vtk_obj) # save pic at t, after one step
        ## 最后一步计算完的时间可能比T略大
        self.finalT = self.t

class LapVSD(LapVWillMore):
    def __init__(self, mymesh, T=0.0001, dt=0.0001, order=1, BDForder=1):
        super().__init__(mymesh, T, dt, order, BDForder)

    def WeakSD(self):
        # Surface Diffusion is a variant of Willmore with Q = 0
        super().WeakWillmore(SD_opt=True)