from ngsolve import *
import numpy as np
from esfem.ode import BDF
from es_utils import pos_transformer, NgMFSave
from viz.vtk_out import VtkOutBnd
from geometry import DiscreteMesh, Param2dRot
from global_utils import LogTime
from ._unsolved_pack import *
import os

class LapVDNMCF_v2():
    '''
        Algorithm: Laplace v = kappa n, Mean curvature flow case Curvature convention: sphere of radius R: 2/R.
    '''
    def __init__(self, mymesh, T = 1e-4, dt = 1e-4, order = 1, BDForder = 1, T_begin=0, scale=1):
        self.MethodName = 'LapVDNMCF'
        self.mesh = mymesh
        self.order = order
        self.scale = scale
        self.fes = H1(self.mesh,order=order,definedon=self.mesh.Boundaries(".*"))
        print('fes ndofs is {}'.format(self.fes.ndof))
        self.fesV = VectorH1(self.mesh,order=order,definedon=self.mesh.Boundaries(".*"))
        self.fesV1 = VectorH1(self.mesh,order=1,definedon=self.mesh.Boundaries('.*'))  # interpolate vertices coordinate
        self.dim = 3
        self.Disp = GridFunction(self.fesV)  # piecewise high order for displacement --- high order isoparametric
        self.IniDefP = GridFunction(self.fesV) # store the initial deformation
        for i in range(self.dim):
            self.Disp.components[i].vec.data = BaseVector(np.zeros(self.fes.ndof))
        self.t, self.T, self.finalT = T_begin, T, 0
        self.dt = Parameter(dt)
        self.lhs, self.rhs = [], []
        ##               gam        H           nu           v
        self.fesMix =  self.fes * self.fes * self.fesV * self.fesV
        self.gfu, self.gfuold = GridFunction(self.fesMix), GridFunction(self.fesMix) 
        self.gam, self.H, self.normal, self.velocity = self.gfu.components
        self.gamold, self.Hold, self.nuold, self.vold = self.gfuold.components

        # Update Position = BaseX + Disp
        self.BaseX = GridFunction(self.fesV)
        self.BaseX.Interpolate(CF((x,y,z)),definedon=self.mesh.Boundaries(".*"))
        self.VCoord = GridFunction(self.fesV1)
        # Time discretize scheme
        self.BDF = BDF(order = BDForder)
        self.BDForder = BDForder
        self.counter_print = 1
        self.Ntotal = int(self.T/dt)
        # Initial position for error measuring (after modification)
        self.IniX = GridFunction(self.fesV)
        self.IniX.Interpolate(CF((x,y,z)),definedon=self.mesh.Boundaries(".*"))
        self.IniCurv, self.IniNormal = GridFunction(self.fes), GridFunction(self.fesV)
        self.Iniv = GridFunction(self.fesV)
        # Inverse of mass matrix of the scalar fespace
        self.L2u, self.L2v = self.fes.TnT()
        mass = BilinearForm(self.L2u*self.L2v*ds).Assemble().mat
        self.invmass = mass.Inverse(inverse="sparsecholesky")
        # Mixed fespace for solving Laplace v = kappa n 
        self.fes_vk = self.fes*self.fesV

    def IniByGeoObj(self,Geo_Rot_Obj:Param2dRot):
        '''
            Given profile spline, determine modification of position, normal and curvature of the initial surface by analytic expression 
        '''
        InterpolateOrder = self.order
        self.fesir = IntegrationRuleSpaceSurface(self.mesh, order=InterpolateOrder, definedon=self.mesh.Boundaries('.*'))
        self.irs = self.fesir.GetIntegrationRules()
        # Interpolate x,y,z of mesh on quadrature points
        gfuirx = GridFunction(self.fesir)
        gfuirx.Interpolate(x,definedon=self.mesh.Boundaries('.*'))
        gfuiry = GridFunction(self.fesir)
        gfuiry.Interpolate(y,definedon=self.mesh.Boundaries('.*'))
        gfuirz = GridFunction(self.fesir)
        gfuirz.Interpolate(z,definedon=self.mesh.Boundaries('.*'))
        Coords3d = np.zeros((self.fesir.ndof,3))
        Coords3d[:,0] = gfuirx.vec.FV().NumPy()
        Coords3d[:,1] = gfuiry.vec.FV().NumPy()
        Coords3d[:,2] = gfuirz.vec.FV().NumPy()
        phi_np, theta_np = Geo_Rot_Obj.Get_Param(Coords3d,Near_opt=True)
        Pset, Normset = Geo_Rot_Obj.get_pos_norm(phi_np, theta_np)
        DefCoords = Pset-Coords3d
        tmpfunc = lambda x: self.return_l2(x)
        ModPosx,ModPosy,ModPosz = map(tmpfunc,[Pset[:,0],Pset[:,1],Pset[:,2]])
        ModNGridx,ModNGridy,ModNGridz = map(tmpfunc,[Normset[:,0],Normset[:,1],Normset[:,2]])
        DefPosx,DefPosy,DefPosz = map(tmpfunc,[DefCoords[:,0],DefCoords[:,1],DefCoords[:,2]])
        # Mean curvature of the modified points
        phi_np, _ = Geo_Rot_Obj.Get_Param(Pset)
        IntCurvature = Geo_Rot_Obj.H_np(phi_np)
        ModCurv = tmpfunc(IntCurvature)
        # Initial deformation
        self.DefPos = GridFunction(self.fesV) # store the initial deformation
        self.DefPos.components[0].vec.data = DefPosx.vec
        self.DefPos.components[1].vec.data = DefPosy.vec
        self.DefPos.components[2].vec.data = DefPosz.vec
        self.IniDefP.vec.data = self.DefPos.vec
        self.Disp.vec.data = self.DefPos.vec
        self.mesh.SetDeformation(self.Disp)
        # L2 error of Position modification
        L2err = lambda x: np.sqrt(Integrate(InnerProduct(x,x), self.mesh, definedon = self.mesh.Boundaries(".*"),element_wise=False))
        errPos = L2err(self.DefPos)
        print('L2 Error of position on modified surface is {}'.format(errPos))

        # For error measurement
        self.IniNormal.components[0].vec.data = ModNGridx.vec
        self.IniNormal.components[1].vec.data = ModNGridy.vec
        self.IniNormal.components[2].vec.data = ModNGridz.vec
        self.IniX.components[0].vec.data = ModPosx.vec
        self.IniX.components[1].vec.data = ModPosy.vec
        self.IniX.components[2].vec.data = ModPosz.vec
        self.IniCurv.vec.data = ModCurv.vec
        # Set Initial data for curvature, normal and velocity
        self.Hold.vec.data = self.IniCurv.vec
        self.nuold.vec.data = self.IniNormal.vec
        self.LapvSet()
        self.Iniv.vec.data = self.vold.vec
    
    def LapvSet(self):
        '''
            Solving Laplace v = kappa n; v n = -H, after Hold and nuold are set
        '''
        kappa, v = self.fes_vk.TrialFunction()
        kappat, vt = self.fes_vk.TestFunction()
        v_lhs = BilinearForm(self.fes_vk, symmetric=True)
        v_lhs += (InnerProduct(grad(v).Trace(),grad(vt).Trace()) + kappa*InnerProduct(self.nuold, vt)\
                 + kappat*InnerProduct(self.nuold,v))*ds
        v_rhs = LinearForm(self.fes_vk)
        v_rhs += -self.Hold*kappat*ds
        gfu = GridFunction(self.fes_vk)
        gfu.vec.data = v_lhs.Assemble().mat.Inverse(inverse="pardiso")*(v_rhs.Assemble().vec)
        self.vold.vec.data = gfu.components[-1].vec
        
    def Compu_HN_ByDisPos(self,opt):
        self.n_X = GridFunction(self.fesV)
        # specialcf.normal(3): inward normal vector field
        self.n_X.Interpolate(-specialcf.normal(3),definedon=self.mesh.Boundaries('.*'))
        # self.DMesh_Obj.N_Area_El()
        
        if opt == 'Rel':
            id_X = GridFunction(self.fesV)
            id_X.Interpolate(CF((x,y,z)),definedon=self.mesh.Boundaries('.*'))
            H_trial, H_test = self.fes.TnT()
            self.H_X = GridFunction(self.fes)
            rhs = LinearForm(self.fes)
            rhs += InnerProduct(grad(id_X).Trace(),grad(self.n_X).Trace())*H_test*ds + InnerProduct(self.n_X,grad(id_X).Trace()*grad(H_test).Trace())*ds
            lhs = BilinearForm(self.fes)
            lhs += H_trial*H_test*ds
            lhs.Assemble()
            rhs.Assemble()
            self.H_X.vec.data = lhs.mat.Inverse(inverse='pardiso')*rhs.vec
        elif opt == 'Interp':
            self.H_X = GridFunction(self.fes)
            self.H_X.Interpolate(-Trace(specialcf.Weingarten(3)),definedon=self.mesh.Boundaries('.*'))
        elif opt == 'BGN':
            H_trial, H_test = self.fes.TnT()
            self.H_X = GridFunction(self.fes)
            rhs = LinearForm(self.fes)
            rhs += Trace(grad(self.n_X).Trace())*H_test*ds
            lhs = BilinearForm(self.fes)
            lhs += H_trial*H_test*ds
            lhs.Assemble()
            rhs.Assemble()
            self.H_X.vec.data = lhs.mat.Inverse(inverse='pardiso')*rhs.vec

        # Compute error of the mean curvature and the normal
        self.err_H_0 = GridFunction(self.fes)
        self.err_H_0.vec.data = self.H_X.vec - self.Hold.vec
        self.n_Norm = GridFunction(self.fes)
        self.n_Norm.Interpolate(Norm(self.nuold),definedon=self.mesh.Boundaries('.*'))
        self.n_err = GridFunction(self.fesV)
        self.n_err.vec.data = self.n_X.vec - self.nuold.vec
        self.vTNorm = GridFunction(self.fes)
        self.vTNorm.Interpolate(Norm(self.vold)**2-InnerProduct(self.vold,specialcf.normal(3))**2,definedon=self.mesh.Boundaries('.*'))

    def ReIni_HN_ByDisPos(self,opt,threhold):
        self.Compu_HN_ByDisPos(opt=opt)
        reset_tag = False
        self.mesh.SetDeformation(self.IniDefP)
        e_nL2 = np.sqrt(Integrate(InnerProduct(self.n_err,self.n_err), self.mesh, element_wise=False, definedon=self.mesh.Boundaries('.*')))
        e_HL2 = np.sqrt(Integrate(self.err_H_0**2, self.mesh, element_wise=False, definedon=self.mesh.Boundaries('.*')))
        HGeoL2 = np.sqrt(Integrate(self.H_X**2, self.mesh, element_wise=False, definedon=self.mesh.Boundaries('.*')))
        if e_HL2/HGeoL2 > threhold:
            reset_tag = True
        self.mesh.SetDeformation(self.Disp)

        if reset_tag:
            print('Now Reinit time is {}'.format(self.t))
            self.Hold.vec.data  = self.H_X.vec
            self.nuold.vec.data = self.n_X.vec
            self.LapvSet()

    def IniWithScale(self,scale):
        # total scale
        self.scale *= scale 
        Base_Coord = pos_transformer(self.BaseX, dim=3)
        Disp_np = scale * (Base_Coord + pos_transformer(self.Disp, dim=3)) - Base_Coord
        self.Disp.vec.data = BaseVector(Disp_np.flatten('F'))
        # only work for 1st order
        self.Hold.vec.data = BaseVector(1/scale*self.Hold.vec.FV().NumPy())
        # nuold keeps unchange
        self.vold.vec.data = BaseVector(1/scale*self.vold.vec.FV().NumPy())
        self.mesh.SetDeformation(self.Disp)

    def IniBySaveDat(self,SavePath):
        '''
            Set Disp, t, scale, Hold, nuold, vold by Saved Data
        '''
        info = np.load(os.path.join(SavePath,'info.npy'),allow_pickle=True).item()
        self.t, self.scale = [info[ii] for ii in ['t','scale']]
        self.Disp.Load(os.path.join(SavePath,'Disp'))
        self.Hold.Load(os.path.join(SavePath,'Hold'))
        self.nuold.Load(os.path.join(SavePath,'nuold'))
        self.vold.Load(os.path.join(SavePath,'vold'))
        self.mesh.SetDeformation(self.Disp)

    def return_l2(self,vec):
        '''
            vec: Interpolated values at quadrature nodes
        '''
        IntFunc = GridFunction(self.fesir)
        IntFunc.vec.data = BaseVector(vec)
        L2Func = GridFunction(self.fes)
        rhs = LinearForm(IntFunc*self.L2v*ds(intrules=self.irs))
        rhs.Assemble()
        L2Func.vec.data = self.invmass * rhs.vec
        return L2Func

    def WeakMCF(self):
        ## Set Curvature
        kappa , H , nu, v = self.fesMix.TrialFunction()
        kappat, Ht, nut, vt= self.fesMix.TestFunction()

        vT = self.vold - self.nuold*InnerProduct(self.vold,self.nuold)/InnerProduct(self.nuold,self.nuold)
        ## Weingarten Map A = grad n
        A = grad(self.nuold).Trace()
        A = A.trans
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

    def Get_Vertices_Coords(self):
        self.VCoord.Interpolate(CF((x,y,z)),definedon=self.mesh.Boundaries(".*"))
        Vertices_Coords = pos_transformer(self.VCoord,dim=3)
        return Vertices_Coords

    def MQ_Measure_Set(self):
        Vertices_Coords = self.Get_Vertices_Coords()
        self.ElVerInd, self.EdVerInd = self.Mesh_Info_Parse()
        self.DMesh_Obj = DiscreteMesh(Vertices_Coords,2,3,self.ElVerInd, self.EdVerInd)
        self.Mesh_Quality = []
        self.Area_set = []

    def Get_Proper_Scale(self):
        # make diameter to be order 1
        Vertices_Coords = self.Get_Vertices_Coords()
        diam = max(np.linalg.norm(Vertices_Coords,axis=1))
        scale = 1/diam
        return scale
    
    def Get_Proper_dt(self,coef=1):
        # order h**2, independent of scale
        Vertices_Coords = pos_transformer(self.Disp, dim=self.dim) + pos_transformer(self.BaseX, dim=self.dim)
        self.DMesh_Obj.UpdateCoords(Vertices_Coords)
        h = min(np.linalg.norm(self.DMesh_Obj.barvec,axis=1))
        dt = coef*h**2
        log_str = '{} :Using Get_Proper_dt, Now coef is {}, t is {}, min h is {}, dt is {}, scale is {}'
        print(log_str.format(LogTime(),coef,self.t,h,format(dt,'0.3e'),self.scale))
        return dt

    def Mesh_Info_Parse(self):
        ElVerInd = []
        EdVerInd = []
        for el in self.mesh.ngmesh.Elements2D():
            # v.nr 从1开始计数
            ElVerInd.append([v.nr-1 for v in el.vertices])
        for ed in self.mesh.edges:
            # v.nr 从0开始计数（ngmesh与ngsolve.Mesh之间的区别）
            EdVerInd.append([v.nr for v in ed.vertices])
        return np.array(ElVerInd), np.array(EdVerInd)

    def PrintDuring(self):
        if self.t > self.T*self.counter_print/5:
            print('Completed {} per cent'.format(int(self.counter_print/5*100)))
            self.counter_print += 1

    def PP_Pic(self,vtk_obj:VtkOutBnd=None):
        # Post Process: Saving Picture
        pass

    def SaveFunc(self,BaseDirPath,Spstr=''):
        Case_Name = 'C_'+format(self.T,'0.3e').replace('.','_').replace('-','_')+Spstr
        flist = [self.Disp, self.Hold, self.vold, self.nuold]
        fnlist = ['Disp', 'Hold', 'vold', 'nuold']
        t_scale_dict = {'t': self.t, 'scale': self.scale}
        NgMFSave(CaseName=Case_Name, BaseDirPath=BaseDirPath, mesh=self.mesh, funclist=flist, func_name_list=fnlist, data_dict=t_scale_dict)
        print('Now T={}, Save Functions in {}'.format(self.T,Case_Name))

    def Solving(self,vtk_obj:VtkOutBnd=None):
        '''
            if vtk_obj not None, output vtk file.
        '''
        while self.t<self.T:
            tauval = self.dt.Get()
            self.lhs.Assemble()
            self.rhs.Assemble()
            self.gfu.vec.data = self.lhs.mat.Inverse(inverse="pardiso")*self.rhs.vec
    
            ## Update the Mean curvature and Normal vector old (used in weak formulation)
            self.Hold.vec.data = self.H.vec
            self.nuold.vec.data = self.normal.vec
            self.vold.vec.data = self.velocity.vec
            BDFInc = self.BDF.Stepping(self.velocity.vec.FV().NumPy()*tauval)
            self.Disp.vec.data += BaseVector(BDFInc)
            self.mesh.SetDeformation(self.Disp)

            self.t += tauval/self.scale**2
            self.PP_Pic(vtk_obj)
            self.PrintDuring()
        ## 最后一步计算完的时间可能比T略大
        self.finalT = self.t

    def NormalizeN(self):
        n_mat = pos_transformer(self.nuold,dim=3)
        unit_n_mat = n_mat/(np.linalg.norm(n_mat,axis=1)[:,None])
        self.nuold.vec.data = BaseVector(unit_n_mat.flatten('F'))