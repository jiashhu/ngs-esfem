'''
    convention：A = nabla n，从而圆周的平均曲率H为正
    类的继承关系：
    基本的ESFEM的class有：
        LapVDNMCF: Laplace v对应的平均曲率流
        LapVWillMore: Laplace v对应的Willmore flow
    特定几何的类：继承基本的ESFEM的类：
        初始化给定mymesh
        重写IniH方法：即设定演化方程所需要的初值，初始的position，曲率，法向量等
        重写ErrL2H1方法：这部分似乎可以写成ESFEM的类方法，然后在每个example中给出精确解的表达式
'''

from logging import exception
import netgen.meshing as ngm
from netgen.csg import *
from ngsolve import *
import numpy as np
from Package_MyCode import * 
from Package_MyNgFunc import * 
from Package_ODE import *
from ngsolve.comp import IntegrationRuleSpaceSurface
from Package_Geometry_Obj import Coord1dFlower, DiscreteMesh, FlowerCurve, Mesh1dFromPoints, HNCoef, DumbbellSpline, phi, Param1dCurve, Param1dSpline, Param2dRot
from Package_ALE_Geometry import Pos_Transformer, Vtk_out_1d, Vtk_out_BND
from PFEMMd import Mesh1d
import sympy as sym
from sympy.physics.vector import *
from time import strftime, gmtime

N = ReferenceFrame('N')
phi = sym.Symbol('phi')

def FindNearestRotZ(Profile_Obj,Coords):
    '''
        关于z轴旋转的旋转几何体的投影，例如xz平面绕z轴旋转的Angenent Torus
    '''
    nv2, _ = Coords.shape
    assert _ == 3
    ModCoord = np.zeros((nv2,2))
    ModCoord[:,0] = np.linalg.norm(Coords[:,:2],axis=1)
    ModCoord[:,1] = Coords[:,-1]
    # x = Rho * cos Phi, y = Rho * sin Phi
    Phi = np.arctan2(Coords[:,1],Coords[:,0])
    
    Nearest_p, Normset2d = Profile_Obj.NearestPoint(ModCoord)
    NewCoord = np.zeros(Coords.shape)
    NewCoord[:,0] = Nearest_p[:,0]*np.cos(Phi)
    NewCoord[:,1] = Nearest_p[:,0]*np.sin(Phi)
    NewCoord[:,2] = Nearest_p[:,1]
    Normset = np.zeros(Coords.shape)
    Normset[:,0] = Normset2d[:,0]*np.cos(Phi)
    Normset[:,1] = Normset2d[:,0]*np.sin(Phi)
    Normset[:,2] = Normset2d[:,1]
    return NewCoord, Normset

class LapVDNMCF():
    '''
        Algorithm: Laplace v = kappa n, Mean curvature flow case，尚未修改Curvature的符号
    '''
    def __init__(self, mymesh, T = 1e-4, dt = 1e-4, order = 1, BDForder = 1):
        self.MethodName = 'LapVDNMCF'
        self.mesh = mymesh
        self.order = order
        self.fes = H1(self.mesh,order=order,definedon=self.mesh.Boundaries(".*"))
        self.fesV = VectorH1(self.mesh,order=order,definedon=self.mesh.Boundaries(".*"))
        InterpolateOrder = order
        self.fesir = IntegrationRuleSpaceSurface(self.mesh, order=InterpolateOrder, definedon=self.mesh.Boundaries('.*'))

        self.dim = 3
        self.Disp = GridFunction(self.fesV)  # piecewise high order for displacement --- high order isoparametric
        for i in range(self.dim):
            self.Disp.components[i].vec.data = BaseVector(np.zeros(self.fes.ndof))
        self.DefPos = GridFunction(self.fesV) # store the initial deformation
        self.t, self.T, self.finalT = 0, T, 0
        self.dt = Parameter(dt)
        self.lhs, self.rhs = [], []
        ##               gam        H           nu           v
        self.fesMix =  self.fes * self.fes * self.fesV * self.fesV
        self.gfu, self.gfuold = GridFunction(self.fesMix), GridFunction(self.fesMix) 
        self.gam, self.H, self.normal, self.velocity = self.gfu.components
        self.gamold, self.Hold, self.nuold, self.vold = self.gfuold.components

        self.IniX = GridFunction(self.fesV)
        self.IniX.Interpolate(CoefficientFunction((x,y,z)),definedon=self.mesh.Boundaries(".*"))

        self.counter_print = 1
        self.Ntotal = int(self.T/dt)

        self.BDF = BDF(order = BDForder)
        self.BDForder = BDForder
        self.IniCurv, self.IniNormal = GridFunction(self.fes), GridFunction(self.fesV)

        ## For Interpolation 
        self.irs = self.fesir.GetIntegrationRules()
        self.L2u, self.L2v = self.fes.TnT()
        mass = BilinearForm(self.L2u*self.L2v*ds).Assemble().mat
        self.invmass = mass.Inverse(inverse="sparsecholesky")

    def PrintDuring(self):
        if self.t > self.T*self.counter_print/5:
            print('Completed {} per cent'.format(int(self.counter_print/5*100)))
            self.counter_print += 1

    def IniDataSet(self, MyAngenent=None, MeshName=None):
        '''
            Initial values for Normal vector field and Mean curvature,
            By Almost exact values!!!
        '''
        if MyAngenent:
            # For case of Angenent Torus
            print('Interpolated mesh and HN')
            gfuirx = GridFunction(self.fesir)
            gfuirx.Interpolate(x,definedon=self.mesh.Boundaries('.*'))
            gfuiry = GridFunction(self.fesir)
            gfuiry.Interpolate(y,definedon=self.mesh.Boundaries('.*'))
            gfuirz = GridFunction(self.fesir)
            gfuirz.Interpolate(z,definedon=self.mesh.Boundaries('.*'))
            Coords = np.zeros((self.fesir.ndof,3))
            Coords[:,0] = gfuirx.vec.FV().NumPy()
            Coords[:,1] = gfuiry.vec.FV().NumPy()
            Coords[:,2] = gfuirz.vec.FV().NumPy()
            NewCoord, Normset = FindNearestRotZ(MyAngenent,Coords)
            # key formula for initial Curvature
            IntCurvature = -1/2*np.sum(NewCoord*Normset,axis=1)
            DefCoords = NewCoord-Coords
            tmpfunc = lambda x: self.Return2L2(x)
            ModPosx,ModPosy,ModPosz = map(tmpfunc,[NewCoord[:,0],NewCoord[:,1],NewCoord[:,2]])
            ModNGridx,ModNGridy,ModNGridz = map(tmpfunc,[Normset[:,0],Normset[:,1],Normset[:,2]])
            DefPosx,DefPosy,DefPosz = map(tmpfunc,[DefCoords[:,0],DefCoords[:,1],DefCoords[:,2]])
            ModCurv = tmpfunc(IntCurvature)

            self.DefPos.components[0].vec.data = DefPosx.vec
            self.DefPos.components[1].vec.data = DefPosy.vec
            self.DefPos.components[2].vec.data = DefPosz.vec

            self.Disp.vec.data = self.DefPos.vec
            self.mesh.SetDeformation(self.DefPos)

            ## error of position
            L2err = lambda x: np.sqrt(Integrate(InnerProduct(x,x), self.mesh, definedon = self.mesh.Boundaries(".*"),element_wise=False))
            errPos = L2err(self.DefPos)
            print('L2 Error of position on modified surface is {}'.format(errPos))

            self.IniNormal.components[0].vec.data = ModNGridx.vec
            self.IniNormal.components[1].vec.data = ModNGridy.vec
            self.IniNormal.components[2].vec.data = ModNGridz.vec
            
            self.IniX.components[0].vec.data = ModPosx.vec
            self.IniX.components[1].vec.data = ModPosy.vec
            self.IniX.components[2].vec.data = ModPosz.vec

            self.IniCurv.vec.data = ModCurv.vec
            self.Hold.vec.data = self.IniCurv.vec
            self.nuold.vec.data = self.IniNormal.vec
        elif MeshName == 'sphere':
            # sphere of radius 2
            self.nuold.Interpolate(CoefficientFunction((x/2,y/2,z/2)),definedon=self.mesh.Boundaries('.*'))
            self.Hold.Interpolate(-1,definedon=self.mesh.Boundaries('.*'))
            self.IniCurv.vec.data = self.Hold.vec
            self.IniNormal.vec.data = self.nuold.vec
        else:
            self.nuold.Set(specialcf.normal(3),definedon=self.mesh.Boundaries('.*'))
            Hi,Hit = self.fes.TnT()
            Lhs1 = BilinearForm(self.fes,symmetric=True)
            Lhs1 += Hi*Hit*ds
            Rhs1 = LinearForm(self.fes)
            Rhs1 += -InnerProduct(grad(self.IniX).Trace(), grad(self.nuold).Trace())*Hit*ds\
                    -InnerProduct(grad(self.IniX).Trace()*self.nuold, grad(Hit).Trace())*ds
            Lhs1.Assemble()
            Rhs1.Assemble()
            self.Hold.vec.data = Lhs1.mat.Inverse()*Rhs1.vec
            print("Initialize mean curvature for self-shrinker!!")
            self.Hold.Set(-1/2*InnerProduct(specialcf.normal(3),CoefficientFunction((x,y,z))),definedon=self.mesh.Boundaries('.*'))
            ## Initial curvature and normal by local L2 projection.
            self.IniCurv.vec.data = self.Hold.vec
            self.IniNormal.vec.data = self.nuold.vec

    def Return2L2(self,vec):
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

    def IniDraw(self):
        Draw(self.Disp, self.mesh, "deformation")
        visoptions.vecfunction  = "deformation"
        SetVisualization(deformation=True)
        Draw(self.Hold, self.mesh, "Hold")
        visoptions.scalfunction  = "Hold"

    def WeakMCF(self):
        ## Set Curvature
        gam , H , nu, v = self.fesMix.TrialFunction()
        gamt, Ht, nut, vt= self.fesMix.TestFunction()

        vT = self.vold - self.nuold*InnerProduct(self.vold,self.nuold)/InnerProduct(self.nuold,self.nuold)
        ## Weingarten Map
        A = -grad(self.nuold).Trace()
        A = A.trans

        Lhs = BilinearForm(self.fesMix,symmetric=False)
        Lhs += (InnerProduct(v,self.nuold)*gamt)*ds-H*gamt*ds
        Lhs += (H*Ht+self.dt*(InnerProduct(grad(H).Trace(),grad(Ht).Trace())\
                    -InnerProduct(A,A)*H*Ht\
                    -InnerProduct(vT,grad(H).Trace())*Ht))*ds
        Lhs += InnerProduct(nu,nut)*ds+self.dt*(-InnerProduct(grad(nu).Trace()*vT,nut)\
                    +InnerProduct(grad(nu).Trace(),grad(nut).Trace())\
                    -InnerProduct(A,A)*InnerProduct(nu,nut))*ds
        Lhs += (gam*InnerProduct(self.nuold,vt))*ds\
                    +InnerProduct(grad(v).Trace(),grad(vt).Trace())*ds
        self.lhs = Lhs

        Rhs = LinearForm(self.fesMix)
        Rhs += self.Hold*Ht*ds
        Rhs += InnerProduct(self.nuold,nut)*ds
        self.rhs = Rhs
        
    def Solving(self,vtk_obj:Vtk_out_BND=None):
        while self.t<self.T:
            tauval = self.dt.Get()
            self.lhs.Assemble()
            self.rhs.Assemble()
            self.gfu.vec.data = self.lhs.mat.Inverse(inverse="pardiso")*self.rhs.vec
    
            ## BDF -> deformation, H, nu
            if self.t<(self.BDForder-1)*tauval:
                pass
            else:
                ## Update the Mean curvature and Normal vector old (used in weak formulation)
                self.Hold.vec.data = self.H.vec
                self.nuold.vec.data = self.normal.vec
                self.vold.vec.data = self.velocity.vec
                BDFInc = self.BDF.Stepping(self.velocity.vec.FV().NumPy()*tauval)
                self.Disp.vec.data += BaseVector(BDFInc)
                self.mesh.SetDeformation(self.Disp)
            if vtk_obj is not None:
                vtk_obj.Output(self.mesh)
            self.t += tauval
            self.PrintDuring()
        ## 最后一步计算完的时间可能比T略大
        self.finalT = self.t

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
        Pset, Normset = Geo_Rot_Obj.Get_Pos_Norm(phi_np, theta_np)
        DefCoords = Pset-Coords3d
        tmpfunc = lambda x: self.Return2L2(x)
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
        
    def ReIni_HN_ByDisPos(self):
        '''
            - (Laplace X) * n= H, exterior normal and positive curvature for sphere
            (H,Ht) = sum (grad(X_i),grad(n_i))*Ht + (grad(X_i),grad(Ht))*n_i
        '''
        ## Given high order mesh, deformed by Disp, and compute nuold by interpolation, Hold by weak formulation
        print('Now Reinit time is {}'.format(self.t))
        id_X = GridFunction(self.fesV)
        id_X.Interpolate(CF((x,y,z)),definedon=self.mesh.Boundaries('.*'))
        n_X = GridFunction(self.fesV)
        # specialcf.normal(3): inward normal vector field
        n_X.Interpolate(-specialcf.normal(3),definedon=self.mesh.Boundaries('.*'))
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

    def IniWithScale(self,scale):
        # total scale
        self.scale *= scale 
        Base_Coord = Pos_Transformer(self.BaseX, dim=3)
        Disp_np = scale * (Base_Coord + Pos_Transformer(self.Disp, dim=3)) - Base_Coord
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

    def Return2L2(self,vec):
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

    def Get_Vertices_Coords(self):
        self.VCoord.Interpolate(CF((x,y,z)),definedon=self.mesh.Boundaries(".*"))
        Vertices_Coords = Pos_Transformer(self.VCoord,dim=3)
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
        Vertices_Coords = Pos_Transformer(self.Disp, dim=self.dim) + Pos_Transformer(self.BaseX, dim=self.dim)
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

    def PP_Pic(self,vtk_obj:Vtk_out_BND=None):
        # Post Process: Saving Picture
        pass

    def SaveFunc(self,BaseDirPath):
        Case_Name = 'C_'+format(self.T,'0.3e').replace('.','_').replace('-','_')
        flist = [self.Disp, self.Hold, self.vold, self.nuold]
        fnlist = ['Disp', 'Hold', 'vold', 'nuold']
        t_scale_dict = {'t': self.t, 'scale': self.scale}
        NgMFSave(CaseName=Case_Name, BaseDirPath=BaseDirPath, mesh=self.mesh, funclist=flist, func_name_list=fnlist, data_dict=t_scale_dict)
        print('Now T={}, Save Functions in {}'.format(self.T,Case_Name))

    def Solving(self,vtk_obj:Vtk_out_BND=None):
        '''
            if vtk_obj not None, output vtk file.
        '''
        while self.t<self.T:
            tauval = self.dt.Get()
            self.lhs.Assemble()
            self.rhs.Assemble()
            self.gfu.vec.data = self.lhs.mat.Inverse(inverse="pardiso")*self.rhs.vec
    
            ## Update the Mean curvature and Normal vector old (used in weak formulation)
            testvec = sum(self.H.vec.FV().NumPy())
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
        n_mat = Pos_Transformer(self.nuold,dim=3)
        unit_n_mat = n_mat/(np.linalg.norm(n_mat,axis=1)[:,None])
        self.nuold.vec.data = BaseVector(unit_n_mat.flatten('F'))

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
        A = A.trans
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

class FlowerLapVSD(LapVSD):
    '''
        T: 总演化时间， dt: time step
        Surface Diffusion是在Willmore的基础上令Q=0，需要在生成weak form的WeakWillmore中采用SD_opt=True的选项。
    '''
    def __init__(self, n_half_blade=120, T=0.0001, dt=0.0001, order=1, BDForder=1):
        self.Coords,self.theta = Coord1dFlower(n_half_blade, num_blade=7)
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
    def __init__(self, n_half_blade=120, T=0.0001, dt=0.0001, order=1, BDForder=1):
        super().__init__(n_half_blade, T, dt, order, BDForder)
        self.ds_lumping = []
        self.GenerateDs()

    def GenerateDs(self):
        if self.dim==2:
            ir = IntegrationRule(points = [(0,0), (1,0)], weights = [1/2, 1/2])
            self.ds_lumping = ds(definedon=self.mesh.Boundaries("material"),intrules = { SEGM : ir })
        elif self.dim==3:
            ir = IntegrationRule(points = [(0,0), (1,0), (0,1)], weights = [1/6, 1/6, 1/6])
            self.ds_lumping = ds(intrules = { TRIG : ir })

    def ReIni_HN_ByBGN(self):
        assert self.order == 1
        # fisrt update the weighted normal 
        N_X = GridFunction(self.fesV)

        # update mean curvature through H = Lap n
        H_X = GridFunction(self.fes)
        H_trial, H_test = self.fes.TnT()
        rhs = LinearForm(self.fes)
        lhs = BilinearForm(self.fes)
        lhs += H_trial*H_test*self.ds_lumping
        rhs += Trace(grad(N_X).Trace())*H_test*self.ds_lumping
        lhs.Assemble()
        rhs.Assemble()
        H_X.vec.data = lhs.mat.Inverse(inverse='pardiso')*rhs.vec

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
        n_X.Interpolate(-specialcf.normal(2),definedon=self.mesh.Boundaries('.*'))
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

class SphereLapVWillmore(LapVWillMore):
    '''
        Test Convergence of Laplace v = kappa n & Willmore flow on the case of sphere (R=2)
    '''
    def __init__(self, maxh, T=0.0001, dt=0.0001, order=1, BDForder=1):
        sphere = Sphere(Pnt(0,0,0),2)
        sphere.bc("Outer")
        dom = sphere
        geo = CSGeometry()
        geo.Add(dom)
        mesh = geo.GenerateMesh(maxh = maxh, optsteps2d=3, perfstepsend=ngm.MeshingStep.MESHSURFACE)
        mymesh = Mesh(mesh)
        mymesh.Curve(order)
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

class DumbbellLapVMCF(LapVDNMCF_v2):
    def __init__(self, mymesh, Geo_Rot_Obj:Param2dRot, T=0.0001, dt=0.0001, order=1, BDForder=1):
        super().__init__(mymesh, T, dt, order, BDForder)
        self.Dumbbell2d = Geo_Rot_Obj
        # mean curvature of the rotational geometry
        self.H_np = self.Dumbbell2d.H_np

    def PP_Pic(self,vtk_Obj_bnd:Vtk_out_BND=None):
        # 后处理： 保存图像
        perform_res = vtk_Obj_bnd.Output(self.mesh,[],tnow=self.t)
        # 后处理： 计算网格质量，利用self.mesh的拓扑关系
        if perform_res:
            self.VCoord.Interpolate(CF((x,y,z)),definedon=self.mesh.Boundaries(".*"))
            self.DMesh_Obj.UpdateCoords(Pos_Transformer(self.VCoord,dim=3))
            Q_Area, Q_Leng = self.DMesh_Obj.MeshQuality()
            self.Area_set.append(self.DMesh_Obj.Area)
            self.Mesh_Quality.append([Q_Area,Q_Leng,self.t])

class DumbbellLapVMCF_N_modified(DumbbellLapVMCF):
    def __init__(self, mymesh, Geo_Rot_Obj: Param2dRot, T=0.0001, dt=0.0001, order=1, BDForder=1):
        super().__init__(mymesh, Geo_Rot_Obj, T, dt, order, BDForder)
    
    def Solving(self, vtk_obj: Vtk_out_BND = None):
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

class RBCLapVWM_ImplicitVz(RBCLapVWillmore):
    def __init__(self, mymesh, Geo_Rot_Obj: Param2dRot, T=0.0001, dt=0.0001, order=1, BDForder=1):
        super().__init__(mymesh, Geo_Rot_Obj, T, dt, order, BDForder)
        self.Hgeo = GridFunction(self.fes)
        self.ngeo = GridFunction(self.fesV)

    def WeakWillmore(self,SD_opt=False):
        ## Set Curvature
        kappa , H , V , z, nu, v = self.fesMix.TrialFunction()
        kappat, Ht, Vt, zt, nut, vt= self.fesMix.TestFunction()

        # 预设速度
        vT = self.vold
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

class RBCLapVWM_ImplicitVz2(RBCLapVWM_ImplicitVz):
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

        # 计算需要对Hold,nuold,vold,Vold,zold设定初值
        Rhs = LinearForm(self.fesMix)
        Rhs += Q*Vt*ds
        Rhs += 1/self.dt*self.Hold*Ht*ds 
        Rhs += 1/self.dt*InnerProduct(n_unified,nut)*ds
        Rhs += 2*InnerProduct(grad(nut).Trace()*(A*grad(self.Hold).Trace()), n_unified)*ds
        Rhs += (InnerProduct(grad(self.Hold).Trace(),grad(self.Hold).Trace())*InnerProduct(self.nuold,nut))*ds + InnerProduct(A*grad(self.Hold).Trace(),A*nut)*ds
        Rhs += Q*Trace(grad(nut).Trace())*ds - Q*self.Hold*InnerProduct(n_unified,nut)*ds
        Rhs += (InnerProduct(A,A)*InnerProduct(n_unified,zt))*ds
        self.rhs = Rhs

