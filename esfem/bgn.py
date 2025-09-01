from ngsolve import *
from esfem.utils import GetIdCF
from geometry import Mesh_Info_Parse, DiscreteMesh
from esfem.utils import Pos_Transformer, NgMFSave
from global_utils import LogTime
import numpy as np
import os

def mydiv(tensor_Gfu):
    '''
        Given a 2-tensor m_obj, for each row, compute the divergence
    '''
    cols = tensor_Gfu.dims[1]
    divu = []
    for ii in range(cols):
        divu[ii] = grad(tensor_Gfu[ii,0])[0] + grad(tensor_Gfu[ii,1])[1] + grad(tensor_Gfu[ii,2])[2]
    return divu

def myInnerProduct(v1,v2):
    '''
        Innerproduct of two lists of gridfunctions
    '''
    nv1 = len(v1)
    nv2 = len(v2)
    assert nv1 == nv2
    for ii in range(nv1):
        if ii == 0:
            res = v1[0]*v2[0]
        else:
            res += v1[ii]*v2[ii]
    return res

class BGN():
    def __init__(self, mymesh, T = 1e-4, tau = 1e-4):
        self.mesh = mymesh
        self.T_seg_begin = 0
        self.T, self.t, self.finalT = T, 0, 0
        self.dt = Parameter(tau)
        self.fes = H1(self.mesh,order=1)
        self.fesV = VectorH1(self.mesh,order=1)
        self.fesV0 = VectorSurfaceL2(self.mesh,order=0)
        self.fesMix = self.fes*self.fesV
        self.lhs, self.rhs = [],[]
        self.Solution = GridFunction(self.fesMix)
        self.MethodName = 'BGN'

        self.counter_print = 0
        self.dim = self.mesh.dim
        print('mesh dim = {}'.format(self.dim))
        self.BaseX = GridFunction(self.fesV)
        self.BaseX.Interpolate(GetIdCF(self.dim),definedon=self.mesh.Boundaries(".*"))
        # initial historic info: Position, normal vector(ele), weighted n(ver)
        self.X_old = GridFunction(self.fesV)
        self.Nvec = GridFunction(self.fesV0)
        self.NvecD = GridFunction(self.fesV)
        self.Disp = GridFunction(self.fesV)
        # set mass lumping inner product
        self.ds_lumping = []
        self.GenerateDs()
        self.scale = 1
        self.MQ_Measure_Set()

    def MQ_Measure_Set(self):
        Vertices_Coords = np.array([v.point for v in self.mesh.vertices])
        self.ElVerInd, self.EdVerInd = Mesh_Info_Parse(self.mesh)
        self.DMesh_Obj = DiscreteMesh(Vertices_Coords,self.dim-1,self.dim,self.ElVerInd,self.EdVerInd)
        self.Mesh_Quality = []
        self.Area_set  = []

    def IniWithScale(self,scale):
        # total scale
        self.scale *= scale 
        Base_Coord = Pos_Transformer(self.BaseX, dim=self.dim)
        Disp_np = scale * (Base_Coord + Pos_Transformer(self.Disp, dim=self.dim)) - Base_Coord
        self.Disp.vec.data = BaseVector(Disp_np.flatten('F'))
        # only work for 1st order
        self.X_old.vec.data = BaseVector(scale*self.X_old.vec.FV().NumPy())
        # nuold keeps unchange
        self.mesh.SetDeformation(self.Disp)

    def IniBySaveDat(self,SavePath):
        '''
            Set Disp, t, scale, Position, Nvec by Saved Data
        '''
        info = np.load(os.path.join(SavePath,'info.npy'),allow_pickle=True).item()
        self.t, self.scale = [info[ii] for ii in ['t','scale']]
        self.Disp.Load(os.path.join(SavePath,'Disp'))
        self.X_old.Load(os.path.join(SavePath,'Position'))
        self.Nvec.Load(os.path.join(SavePath,'Nvec'))
        self.mesh.SetDeformation(self.Disp)

    def IniByDisPos(self):
        # initilize Position and normal vector field(ele) by discrete vertices
        Vertices_Coords = np.array([v.point for v in self.mesh.vertices])
        self.X_old.vec.data = BaseVector(Vertices_Coords.flatten('F'))
        self.GetNvec()

    def GenerateDs(self):
        if self.dim==2:
            ir = IntegrationRule(points = [(0,0), (1,0)], weights = [1/2, 1/2])
            self.ds_lumping = ds(definedon=self.mesh.Boundaries("material"),intrules = { SEGM : ir })
        elif self.dim==3:
            ir = IntegrationRule(points = [(0,0), (1,0), (0,1)], weights = [1/6, 1/6, 1/6])
            self.ds_lumping = ds(intrules = { TRIG : ir })

    def GetNvec(self):
        '''获取Nvec(分片常数)以及NvecD(线性)'''
        Coords_Update = Pos_Transformer(self.X_old,dim=self.dim)
        self.DMesh_Obj.UpdateCoords(Coords_Update)
        N_El_matrix, Leng_matrix, Area_matrix = self.DMesh_Obj.N_Area_El()
        self.Area = sum(Area_matrix)
        # element-wise normal vector
        self.Nvec.vec.data = BaseVector((-N_El_matrix).flatten('F'))
        self.NvecD.Interpolate(-specialcf.normal(3),definedon=self.mesh.Boundaries('.*'))

    def PP_Pic(self,vtk_obj=None):
        pass
    
    def PrintDuring(self,LoadT=0):
        if self.t > LoadT+(self.T-LoadT)*(1+self.counter_print)/5:
            print('Completed {} per cent'.format(int(self.counter_print/5*100)))
            self.counter_print += 1

    def Get_Proper_Scale(self):
        # make diameter to be order 1
        Vertices_Coords = Pos_Transformer(self.X_old, dim=self.dim)
        diam = max(np.linalg.norm(Vertices_Coords,axis=1))
        scale = 1/diam
        return scale
    
    def Get_Proper_dt(self,coef=1,h_opt='min'):
        # order h**2, independent of scale
        Vertices_Coords = Pos_Transformer(self.X_old, dim=self.dim)
        self.DMesh_Obj.UpdateCoords(Vertices_Coords)
        h = min(np.linalg.norm(self.DMesh_Obj.barvec,axis=1))
        hmean = np.mean(np.linalg.norm(self.DMesh_Obj.barvec,axis=1))
        if h_opt == 'min':
            dt = coef*h**2
        elif h_opt == 'mean':
            dt = coef*hmean**2
        print('{} :Using Get_Proper_dt, Now coef is {}, t is {}, min h is {}, dt is {}, scale is {}'.format(LogTime(),coef,self.t,h,format(dt,'0.3e'),self.scale))
        return dt

    def Solving(self,vtk_obj):
        while self.t<self.T:
            tauval = self.dt.Get()
            self.mesh.SetDeformation(self.Disp)
            self.GetNvec()
            self.lhs.Assemble()
            self.rhs.Assemble()
            self.Solution.vec.data = self.lhs.mat.Inverse(inverse="pardiso")*self.rhs.vec

            ## Solution.components[1] 代表的是X^m+1-X^m
            self.Disp.vec.data = BaseVector(self.Disp.vec.FV().NumPy() + self.Solution.components[1].vec.FV().NumPy())
            self.X_old.vec.data = BaseVector(self.X_old.vec.FV().NumPy() + self.Solution.components[1].vec.FV().NumPy())
            self.PP_Pic(vtk_obj)
            self.PrintDuring(LoadT=self.T_seg_begin)
            self.t += tauval/self.scale**2
        self.T_seg_begin = self.T

    def SaveFunc(self,BaseDirPath):
        Case_Name = 'C_'+format(self.T,'0.3e').replace('.','_').replace('-','_')
        flist = [self.Disp, self.X_old, self.Nvec]
        fnlist = ['Disp', 'Position', 'Nvec']
        t_scale_dict = {'t': self.t, 'scale': self.scale}
        NgMFSave(CaseName=Case_Name, BaseDirPath=BaseDirPath, mesh=self.mesh, funclist=flist, func_name_list=fnlist, data_dict=t_scale_dict)
        print('Now T={}, Save Functions in {}'.format(self.T,Case_Name))

class BGN_MCF(BGN):
    def __init__(self, mymesh, T=0.0001, tau=0.0001):
        super().__init__(mymesh, T, tau)

    def WeakMCF(self):
        '''
            Mixed Formulation for kappa and velocity
            v n = -kappa (曲率纠正符号)   +    D = dt*v     ->      D n + dt*kappa = 0
            -kappa n = Lap Xnew 
            (in weak form)
            -kappa n eta + <grad Xnew, grad eta> = 0      
            ->       
            -kappa n eta + <grad D, grad eta> =  -<grad Xold, grad eta>
        '''
        kappa, D = self.fesMix.TrialFunction()
        chi, eta = self.fesMix.TestFunction()

        self.lhs = BilinearForm(self.fesMix)
        self.lhs += InnerProduct(D,self.Nvec)*chi*self.ds_lumping + self.dt*kappa*chi*self.ds_lumping
        self.lhs += -InnerProduct(eta,self.Nvec)*kappa*self.ds_lumping + InnerProduct(grad(D).Trace(),grad(eta).Trace())*ds

        self.rhs = LinearForm(self.fesMix)
        self.rhs += -InnerProduct(grad(self.X_old).Trace(),grad(eta).Trace())*ds

class BGN_WM(BGN):
    def __init__(self, mymesh, T=0.0001, dt=0.0001):
        super().__init__(mymesh, T, dt)
        self.Mfes = MatrixValued(H1(self.mesh,order=1),dim=self.dim)
        self.W_sol = GridFunction(self.Mfes)
        self.H_sol = GridFunction(self.fes)
        self.fesMix = self.fes*self.fesV
        self.gfu = GridFunction(self.fesMix)
        self.gfu_old = GridFunction(self.fesMix)
        self.kappa, self.X = self.gfu.components
        self.kappa_old, self.X_old = self.gfu_old.components
        self.nu_old = GridFunction(self.fesV)
        self.Base_Coord = Pos_Transformer(self.BaseX)

    def IniByDisPos(self):
        # initilize Position and normal and additional mean curvature
        Vertices_Coords = np.array([v.point for v in self.mesh.vertices])
        self.X_old.vec.data = BaseVector(Vertices_Coords.flatten('F'))
        self.GetNvec()
        self.WeakCurvature_Ini()
        self.H_Lhs.Assemble()
        self.H_Rhs.Assemble()
        self.kappa_old.vec.data = self.H_Lhs.mat.Inverse(inverse='pardiso')*self.H_Rhs.vec

    def WeakWeigarten(self):
        '''
            Solving Weingarten mapping as 2 tensor from W = grad nu, by integration by part. Given 2 tensor chi = (chi_1, chi_2, chi_3):

            <grad nu, chi> = sum <grad nu_i, chi_i> = - sum <nu_i, div chi_i> + kappa nu^T chi^T nu
        '''
        
        W, chi = self.Mfes.TnT()
        self.W_Lhs = BilinearForm(self.Mfes)
        self.W_Lhs += InnerProduct(W,chi)*self.ds_lumping 
        self.W_Rhs = LinearForm(self.Mfes)
        self.W_Rhs += InnerProduct(self.nu_old, chi*self.nuold)*self.kappa_old*self.ds_lumping - myInnerProduct(self.nu_old, mydiv(chi))*ds

    def WeakMain(self):
        '''
            Approximation of the identity
        '''
        kappa, X = self.fesMix.TrialFunction()
        chi, eta = self.fesMix.TestFunction()
        self.WM_Lhs = BilinearForm(self.fesMix)
        self.WM_Rhs = LinearForm(self.fesMix)
        self.WM_Lhs += 1/self.dt*InnerProduct(X, self.nu_old)*chi*self.ds_lumping + InnerProduct(grad(kappa).Trace(),grad(chi).Trace())*ds + 1/2*self.kappa_old**2*kappa*chi*self.ds_lumping
        self.WM_Lhs += -InnerProduct(self.nu_old,eta)*kappa*self.ds_lumping + InnerProduct(grad(X).Trace(),grad(eta).Trace())*ds
        self.WM_Rhs += 1/self.dt*InnerProduct(self.X_old, self.nu_old)*chi*self.ds_lumping
        self.WM_Rhs += self.kappa_old*InnerProduct(grad(self.nu_old).Trace(),grad(self.nu_old).Trace())*chi*self.ds_lumping
    
    def WeakCurvature_Ini(self):
        '''
            Set the initial data for mean curvature from discrete position
        '''
        kappa, chi = self.fes.TnT()
        self.H_Lhs = BilinearForm(self.fes)
        self.H_Lhs += kappa*chi*self.ds_lumping
        self.H_Rhs = LinearForm(self.fes)
        self.H_Rhs += Trace(grad(self.nu_old).Trace())*chi*self.ds_lumping

    def Solving(self, vtk_obj):
        while self.t<self.T:
            tauval = self.dt.Get()
            self.mesh.SetDeformation(self.Disp)
            UpdatedCoords = self.Base_Coord + Pos_Transformer(self.Disp)
            self.DMesh_Obj.UpdateCoords(UpdatedCoords)
            self.DMesh_Obj.WN_Ver()
            # might be in the negetive direction
            self.nu_old.vec.data = BaseVector(self.DMesh_Obj.WN_Ver_matrix.flatten('F'))
            # solving Position and mean curvature using grad(nu) as Weingarten mapping
            self.WM_Lhs.Assemble()
            self.WM_Rhs.Assemble()
            self.gfu.vec.data = self.WM_Lhs.mat.Inverse(inverse='pardiso')*self.WM_Rhs.vec
            self.Disp.vec.data += self.X.vec - self.X_old.vec
            # update position and mean curvature
            self.gfu_old.vec.data = self.gfu.vec
        
            self.PP_Pic(vtk_obj)
            self.PrintDuring()
            self.t += tauval/self.scale**2

    def PP_Pic(self,vtk_Obj_bnd=None):
        # 后处理： 保存图像
        perform_res = vtk_Obj_bnd.Output(self.mesh,[],tnow=self.t)
        # 后处理： 计算网格质量，利用self.mesh的拓扑关系
        if perform_res:
            self.DMesh_Obj.UpdateCoords(Pos_Transformer(self.X_old))
            Q_Area, Q_Leng = self.DMesh_Obj.MeshQuality()
            self.Mesh_Quality.append([Q_Area,Q_Leng,self.t])