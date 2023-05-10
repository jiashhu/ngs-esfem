from pdb import post_mortem
import netgen.meshing as ngm
from ngsolve import *

import numpy as np
import sympy as sym

import os
from scipy.sparse import coo_matrix
import math

import sympy as sym
from sympy.physics.vector import *

import ngsolve.internal as internal
from Package_MyCode import *
from Package_MyNgFunc import * 
from Package_ALE_Geometry import Vtk_out_1d, Vtk_out_BND
from Package_MyNgFunc import Pos_Transformer
from Package_Geometry_Obj import Mesh_Info_Parse, DiscreteMesh
from time import strftime, gmtime


phi = sym.Symbol('phi')
Par_x = (0.65*sym.sin(7*phi)+1)*sym.cos(phi)
Par_y = (0.65*sym.sin(7*phi)+1)*sym.sin(phi)
N = ReferenceFrame('N')
Par_v = Par_x*N.x + Par_y*N.y
Par_func = [sym.lambdify(phi,fun_i) for fun_i in Par_v.to_matrix(N)]

def HNCoef(Par_v,phi,frame):
    '''
        计算平面曲线的曲率：X=(f,g)
        H = -(f' * g'' - g' * f'')/(f'^2+g'^2)^(3/2)
        圆周为 -1
        返回symbolic的标量函数
    '''
    dv = Par_v.diff(phi,frame)
    ddv = dv.diff(phi,frame)
    # rphi rotate clockwise 90 to normal
    normalv = dv^frame.z
    # normalize the normal vector
    normalv = normalv.normalize()
    ## Generate Curvature
    H = -(dv.to_matrix(frame)[0]*ddv.to_matrix(frame)[1]-dv.to_matrix(frame)[1]*ddv.to_matrix(frame)[0])/(dv.to_matrix(frame)[0]**2+dv.to_matrix(frame)[1]**2)**(3/2)
    nx_S = normalv.dot(frame.x)
    ny_S = normalv.dot(frame.y)
    return H, nx_S, ny_S

def NgCoef(H, nx_S, ny_S):
    '''
        Transform symbolic expression with polar coordinate into Ngsolve Coeff Func
    '''
    theta = atan2(y,x)
    Hfun = eval(str(H).replace('phi','theta'))
    Nxfun = eval(str(nx_S).replace('phi','theta'))
    Nyfun = eval(str(ny_S).replace('phi','theta'))
    Hfunc = Hfun.Compile()
    Nxfunc = Nxfun.Compile()
    Nyfunc = Nyfun.Compile()
    return Hfunc, Nxfunc, Nyfunc

def ComputeWeight(arr):
    res = np.zeros(arr.shape)
    res[0] += arr[-1]+arr[0]
    res[1:] += arr[:-1]+arr[1:]
    res /= 2
    return res

class Mesh1d:
    '''
        生成平面单参数曲线对象
        输入：参数值的ndarray
        属性： 
            p: 第三分量为0的坐标
            nr: 节点个数
            mesh: 对应节点构成的顺序连接的网格
    '''
    def __init__(self,theta):
        self.theta = theta
        self.dim = 2
        self.p = []
        self.nr = len(theta)
        self.Parameterization()
        self.mesh = self.MeshFrom1dPoints()
        
    def Parameterization(self):
        ## Parameterization of curve
        self.p = np.array([[Par_func[0](phi),Par_func[1](phi),0] for phi in self.theta])

    def MeshFrom1dPoints(self):
        mesh0= ngm.Mesh(dim=self.dim)
        pids = []
        for i in range(self.nr):
            pids.append (mesh0.Add (ngm.MeshPoint(ngm.Pnt(self.p[i]))))  

        idx = mesh0.AddRegion("material", dim=self.dim-1)   ## dim=1: Boundary For a Mesh of dim=2
        for i in range(self.nr):
            if i<self.nr-1:
                mesh0.Add(ngm.Element1D([pids[i],pids[i+1]],index=idx))
            else:
                mesh0.Add(ngm.Element1D([pids[i],pids[0]],index=idx))
        mymesh = Mesh(mesh0)
        return mymesh

class NgMeshMod:
    def __init__(self,mymesh):
        self.mesh = Mesh(mymesh.ngmesh.Copy())
        self.nr = mymesh.nv
        self.dim = mymesh.dim
        self.fes = H1(self.mesh,order=1)
        self.fesV = VectorH1(self.mesh,order=1)
        self.barvec = np.zeros((self.nr,2))
        self.fesV0 = VectorSurfaceL2(self.mesh,order=0)
        self.Nvec = GridFunction(self.fesV0)
        self.NvecD = GridFunction(self.fesV)
        self.Disp = GridFunction(self.fesV)
        self.Position = GridFunction(self.fesV)
        self.Reg_Position = GridFunction(self.fesV)
        self.SetPosition()
        self.ds_lumping = []
        self.GenerateDs()
        self.Dismesh = []
        self.finalT = 0
        
    def SetPosition(self):
        '''获得mesh所有vertices的坐标'''
        p = np.array([list(ver.point) for ver in self.mesh.vertices])
        for i in range(self.dim):
            self.Position.components[i].vec.data = BaseVector(p[:,i])
            self.Disp.components[i].vec.data = BaseVector(np.zeros(self.nr))
    
    def GenerateDs(self):
        if self.dim==2:
            ir = IntegrationRule(points = [(0,0), (1,0)], weights = [1/2, 1/2])
            self.ds_lumping = ds(definedon=self.mesh.Boundaries("material"),intrules = { SEGM : ir })
        elif self.dim==3:
            ir = IntegrationRule(points = [(0,0), (1,0), (0,1)], weights = [1/6, 1/6, 1/6])
            self.ds_lumping = ds(intrules = { TRIG : ir })

class NgMesh1d(NgMeshMod):
    def __init__(self,mymesh,T=1e-4, tau=1e-4):
        self.MethodName = ''
        self.dim = 2
        self.p = []
        self.nr = mymesh.nv
        NgMeshMod.__init__(self,mymesh)
        self.t = 0
        self.T = T
        self.dt = Parameter(tau)
        self.Savetime = np.linspace(0,1e-4,100)
        
    def GetNvec(self):
        Position = np.zeros((self.nr,2))
        Position[:,0] = self.Position.components[0].vec.FV().NumPy()
        Position[:,1] = self.Position.components[1].vec.FV().NumPy()
        for i in range(self.nr):
            if i<self.nr-1:
                self.barvec[i] = Position[i+1]-Position[i]
            else:
                self.barvec[i] = Position[0]-Position[i]
        
        norm = np.linalg.norm(self.barvec,axis=1)
        ## Rotate ClockWise 90
        n2 = -self.barvec[:,0]/norm
        n1 = self.barvec[:,1]/norm

        self.Nvec.components[0].vec.data = BaseVector(n1)
        self.Nvec.components[1].vec.data = BaseVector(n2)
        
        wn1 = ComputeWeight(n1*norm)
        wn2 = ComputeWeight(n2*norm)
        wnn = np.sqrt(wn1**2+wn2**2)
        wn1 /= wnn
        wn2 /= wnn
        self.NvecD.components[0].vec.data = BaseVector(wn1)
        self.NvecD.components[1].vec.data = BaseVector(wn2)
    
    def Visualize(self, Draw_opt):
        newmesh = Mesh(self.mesh.ngmesh.Copy())
        Position = np.zeros((self.nr,2))
        Position[:,0] = self.Position.components[0].vec.FV().NumPy()
        Position[:,1] = self.Position.components[1].vec.FV().NumPy()
        for i,vec in enumerate(newmesh.ngmesh.Points()):
            vec[0] = Position[i,0]
            vec[1] = Position[i,1]
        if Draw_opt:
            if self.t==0:
                Draw(newmesh)
            else:
                Redraw(blocking=True)
        return newmesh
    
    def MySnapShot(self,filepath):
        NameStr = self.MethodName+'_tau_'+str(self.dt.Get())+'T_'+str(self.t)
        internal.SnapShot(filepath+(NameStr.replace('.','_').replace('-','_')))

class NgMesh2d(NgMeshMod):
    def __init__(self, mymesh ,T=1e-4, tau=1e-4):
        self.MethodName = ''
        self.dim = mymesh.dim
        NgMeshMod.__init__(self,mymesh)
        
        self.t = 0
        self.T = T
        self.dt = Parameter(tau)
        self.Savetime = np.linspace(0,1e-4,100)
        self.Dismesh = Mesh(mymesh.ngmesh.Copy())
        self.h = np.inf
        self.R = np.inf
        self.Area = 0
    
    def GetR(self):
        p = np.array([list(ver.point) for ver in self.Dismesh.vertices])
        self.R = min(self.R, max(np.linalg.norm(p,axis=1)))

    def GetNvec(self):
        '''获取Nvec(分片常数)以及NvecD(线性)'''
        wnormal = np.zeros((self.Dismesh.nv,3))
        snormal = np.zeros((self.Dismesh.nface,3))

        p = np.array([list(ver.point) for ver in self.Dismesh.vertices])
        self.Area = 0
        for i,el in enumerate(self.Dismesh.ngmesh.Elements2D()):
            ## 边界上（遍历）三角形（2d element in 3d）的顶点的index，start from 1
            ver_ind = [ver.nr-1 for ver in el.vertices] 
            ver_cor = p[ver_ind,:] # 3x3
            bar = np.zeros((ver_cor.shape[0]-1,ver_cor.shape[1])) # 2x3
            bar = ver_cor[1:]-ver_cor[:-1]
            ## 排列顺序，逆时针顺序
            wni = np.cross(bar[0,:],bar[1,:]) 
            S = np.linalg.norm(wni)
            ## weighted norm -- on node -- weighted normal + normalization
            for ind_ in ver_ind:
                wnormal[ind_,:] += wni
            snormal[i,:] = wni/S
            self.h = min(self.h, min(np.linalg.norm(bar,axis=1)))
            self.Area += S
        
        self.GetR()
        ## normalization
        for i in range(self.mesh.nv):
            wnormal[i,:] /= np.linalg.norm(wnormal[i,:])
        
        for i in range(self.dim):
            self.Nvec.components[i].vec.data = BaseVector(snormal[:,i])
            self.NvecD.components[i].vec.data = BaseVector(wnormal[:,i])
    
    def Reg_Info(self):
        El2Ver = []
        row_vec, col_vec = [],[]
        nv_s = self.Dismesh.nv
        for el in self.Dismesh.Elements(BND):
            el_row = [el.nr]*3
            ver_col = [v.nr for v in el.vertices]
            row_vec += el_row
            col_vec += ver_col
            El2Ver.append(ver_col)
        A = NGMO.myCOO(row_vec,col_vec,[1]*len(row_vec),len(El2Ver),nv_s,'scipy').tocsc()
        ## 2d list, 第i行是包含ver i的Element的id
        Ver2El = [Ai.nonzero()[1].tolist() for Ai in A.transpose()]
        Ver_Coord = np.array([list(ver.point) for ver in self.Dismesh.vertices])
        ne_s = len(El2Ver)
        ## Weighted normal at all vertices
        Ver_WN = np.zeros((nv_s,3))
        ## Barycenter of all elements
        El_BaryCenter = np.zeros((ne_s,3))

        for ii,el in enumerate(self.Dismesh.Elements(BND)):
            ## 遍历单元，计算每一个点的加权法向量，计算每个单元的质心坐标
            ver_ind = [v.nr for v in el.vertices] 
            ver_cor = Ver_Coord[ver_ind,:] # 3x3
            ## Barycentry
            El_BaryCenter[ii] = ver_cor.sum(axis=0)/3
            ## **************
            ##     0
            ##     | -
            ##     |   -
            ##     1 ---- 2
            bar = np.zeros((ver_cor.shape[0]-1,ver_cor.shape[1])) # 2x3
            bar = ver_cor[1:]-ver_cor[:-1]
            wni = np.cross(bar[0,:],bar[1,:])
            for ind_ in ver_ind:
                Ver_WN[ind_,:] += wni
        Ver_WN /= np.linalg.norm(Ver_WN,axis=1)[:,None]
        return nv_s, Ver2El, El2Ver, Ver_Coord, El_BaryCenter, Ver_WN

    def Regularize(self):
        '''返回Regularized之后的网格节点坐标，nvx3维度的ndarray'''
        nv_s, Ver2El, El2Ver, Ver_Coord, El_BaryCenter, Ver_WN = self.Reg_Info()
        ## G-S类型的迭代，直接更新Ver_Coord的节点坐标
        for ii in range(nv_s):
            El_supp = Ver2El[ii]
            # 获得所有support element的质心
            res = El_BaryCenter[El_supp]
            zhat = res.sum(axis=0)/len(res)
            ## 计算法向搜索的距离
            wn_ = Ver_WN[ii]
            nominator = 0
            divisor = 0
            for jj in El_supp:
                # 遍历support element
                ver_order = El2Ver[jj]
                tmpind = ver_order.index(ii)
                # 将当前ver调整到第一个，zT_新顺序的z
                neworder = ver_order[tmpind:]+ver_order[:tmpind]
                zT_ = Ver_Coord[neworder]
                nominator += np.cross(zT_[0]-zhat,zT_[1]-zhat)@(zT_[2]-zhat)
                divisor += np.cross(wn_,zT_[1]-zhat)@(zT_[2]-zhat)
            t = nominator/divisor
            Ver_Coord[ii] = zhat + t*wn_
        for jj in range(self.dim):
            self.Reg_Position.components[jj].vec.data = BaseVector(Ver_Coord[:,jj])
        for i in range(self.dim):
            Reg_Vec = self.Reg_Position.components[i].vec.FV().NumPy()-self.Position.components[i].vec.FV().NumPy()
            self.Disp.components[i].vec.data += BaseVector(Reg_Vec)
            ## Position的目的既是画图，也用在rhs中
            self.Position.components[i].vec.data += BaseVector(Reg_Vec)
        self.mesh.SetDeformation(self.Disp)

    def Visualize(self, Draw_opt):
        '''绘制self.mesh对应于self.Position的网格结果'''
        mynewmesh = Mesh(self.mesh.ngmesh.Copy())
        Position = np.zeros((self.nr,self.dim))
        for i in range(self.dim):
            Position[:,i] = self.Position.components[i].vec.FV().NumPy()
        
        for i,vec in enumerate(mynewmesh.ngmesh.Points()):
            vec[0] = Position[i,0]
            vec[1] = Position[i,1]
            vec[2] = Position[i,2]
        
        if Draw_opt:
            if self.t==0:
                Draw(mynewmesh)
            else:
                Redraw(blocking=True)
            
        return mynewmesh
    
    def MySnapShot(self,filepath):
        NameStr = self.MethodName+'_tau_'+str(self.dt.Get())+'T_'+str(self.t)
        internal.SnapShot(filepath+(NameStr.replace('.','_').replace('-','_')))
    
def ChooseDim(mydim):
    if mydim==1:
        NgMesh = NgMesh1d
    elif mydim==2:
        NgMesh = NgMesh2d
        
    class Barret2008(NgMesh):
        def __init__(self, mymesh, T = 1e-4, tau = 1e-4):
            NgMesh.__init__(self, mymesh, T, tau)
            self.lhs = []
            self.rhs = []
            self.fesMix = self.fes*self.fesV
            self.Solution = GridFunction(self.fesMix)
            self.MethodName = 'Barret'

        def SetSolution(self):
            '''Solution vector initialized by zero'''
            for i in range(self.dim):
                self.Solution.components[1].components[i].vec.data = BaseVector(np.zeros(self.nr))
            self.Solution.components[0].vec.data = BaseVector(np.zeros(self.nr))

        ## NVec, Position, 
        def WeakSD(self):
            kap, D = self.fesMix.TrialFunction()
            chi, eta = self.fesMix.TestFunction()

            self.lhs = BilinearForm(self.fesMix)
            self.lhs += InnerProduct(D,self.Nvec)*chi*self.ds_lumping\
                        - self.dt*InnerProduct(grad(kap).Trace(),grad(chi).Trace())*self.ds_lumping
            self.lhs += InnerProduct(eta,self.Nvec)*kap*self.ds_lumping \
                        + InnerProduct(grad(D).Trace(),grad(eta).Trace())*ds

            self.rhs = LinearForm(self.fesMix)
            self.rhs += -InnerProduct(grad(self.Position).Trace(),grad(eta).Trace())*ds

        def WeakMCF(self):
            '''
                Mixed Formulation for kappa and velocity
                v n = kappa (曲率差一个符号)   +    D = dt*v     ->      D n - dt*kappa = 0
                kappa n = Lap Xnew 
                (in weak form)
                kappa n eta + <grad Xnew, grad eta> = 0      
                ->       
                kappa n eta + <grad D, grad eta> =  -<grad Xold, grad eta>
            '''
            kappa, D = self.fesMix.TrialFunction()
            chi, eta = self.fesMix.TestFunction()

            self.lhs = BilinearForm(self.fesMix)
            self.lhs += InnerProduct(D,self.Nvec)*chi*self.ds_lumping - self.dt*kappa*chi*self.ds_lumping
            self.lhs += InnerProduct(eta,self.Nvec)*kappa*self.ds_lumping + InnerProduct(grad(D).Trace(),grad(eta).Trace())*ds

            self.rhs = LinearForm(self.fesMix)
            self.rhs += -InnerProduct(grad(self.Position).Trace(),grad(eta).Trace())*ds

        def Solving(self,Draw_opt):
            while self.t<self.T:
                tauval = self.dt.Get()
                self.GetNvec()
                self.lhs.Assemble()
                self.rhs.Assemble()
                self.Solution.vec.data = self.lhs.mat.Inverse(inverse="pardiso")*self.rhs.vec

                ## Solution.components[1] 代表的是X^m+1-X^m
                for i in range(self.dim):
                    self.Disp.components[i].vec.data += self.Solution.components[1].components[i].vec.data  
                    self.Position.components[i].vec.data += self.Solution.components[1].components[i].vec.data
                self.mesh.SetDeformation(self.Disp)
                self.Dismesh = self.Visualize(Draw_opt)
                self.t += tauval

    class Dziuk(NgMesh):
        '''Dziuk Method MCF'''
        def __init__(self, mymesh, T = 1e-4, tau = 1e-4):
            NgMesh.__init__(self, mymesh, T, tau)
            self.lhs = []
            self.rhs = []
            self.Solution = GridFunction(self.fesV)
            self.MethodName = 'Dziuk'

        def WeakMCF(self):
            '''Weak formulation for position'''
            solu, ut = self.fesV.TnT()
            self.lhs = BilinearForm(self.fesV)
            self.lhs += (InnerProduct(solu,ut)+self.dt*InnerProduct(grad(solu).Trace(),grad(ut).Trace()))*ds

            self.rhs = LinearForm(self.fesV)
            self.rhs += InnerProduct(self.Position,ut)*ds

        def Solving(self,Draw_opt):
            state = FO.Process_Message(self.t,self.T,20)
            while self.t<self.T:
                state = FO.Process_Message(self.t,self.T,20,state)
                tauval = self.dt.Get()
                self.lhs.Assemble()
                self.rhs.Assemble()
                self.Solution.vec.data = self.lhs.mat.Inverse(inverse="pardiso")*self.rhs.vec

                ## Solution.vec 代表的是X^m+1
                for i in range(self.dim):
                    self.Disp.components[i].vec.data += BaseVector(self.Solution.components[i].vec.FV().NumPy()-\
                                                        self.Position.components[i].vec.FV().NumPy())
                    self.Position.components[i].vec.data = self.Solution.components[i].vec.data
                self.mesh.SetDeformation(self.Disp)
                self.Dismesh = self.Visualize(Draw_opt)
                self.t += tauval

    class BMN(NgMesh):
        def __init__(self, mymesh, T = 1e-4, tau = 1e-4):
            NgMesh.__init__(self, mymesh, T, tau)
            self.lhs = []
            self.rhs = []
            #               kappa       H       V         v
            self.fesMix = self.fesV*self.fes*self.fes*self.fesV
            self.Solution = GridFunction(self.fesMix)
            self.MethodName = 'BMN2005'
        
        def WeakSD(self):
            kap, H, V, v = self.fesMix.TrialFunction()
            kapt, Ht, Vt, vt = self.fesMix.TestFunction()

            self.lhs = BilinearForm(self.fesMix)
            self.lhs += (InnerProduct(kap,kapt) + self.dt * InnerProduct(grad(v).Trace(),grad(kapt).Trace()))*ds
            self.lhs += (H*Ht-InnerProduct(kap,specialcf.normal(3))*Ht)*ds
            self.lhs += (V*Vt-InnerProduct(grad(H).Trace(),grad(Vt).Trace()))*ds
            self.lhs += (InnerProduct(v,vt)-V*InnerProduct(vt,specialcf.normal(3)))*ds
            self.rhs = LinearForm(self.fesMix)
            self.rhs += -InnerProduct(grad(self.Position).Trace(),grad(kapt).Trace())*ds

        def Solving(self,Draw_opt):
            while self.t<self.T:
                tauval = self.dt.Get()

                self.lhs.Assemble()
                self.rhs.Assemble()
                self.Solution.vec.data = self.lhs.mat.Inverse(inverse="pardiso")*self.rhs.vec

                for i in range(self.dim):
                    self.Disp.components[i].vec.data += BaseVector(self.Solution.components[3].components[i].vec.FV().NumPy()*tauval)
                    ## Position的目的既是画图，也用在rhs中
                    self.Position.components[i].vec.data += BaseVector(self.Solution.components[3].components[i].vec.FV().NumPy()*tauval)
                self.mesh.SetDeformation(self.Disp)
                self.Dismesh = self.Visualize(Draw_opt)
                self.Regularize()
                self.Dismesh = self.Visualize(Draw_opt)
                self.t += tauval
            return self.Dismesh

    class LapVDNSD(NgMesh):
        '''
            Algorithm: Laplace v = kappa n, Surface diffusion
        '''
        def __init__(self, mymesh, T = 1e-4, tau = 1e-4,Adjust_time = [],Draw_opt=False):
            NgMesh.__init__(self, mymesh, T, tau)
            self.lhs = []
            self.rhs = []
            self.Adjust_time = Adjust_time
            ##                gam     H        V       z          nu        v
            self.fesMix = self.fes*self.fes*self.fes*self.fesV*self.fesV*self.fesV
            self.NvecD = GridFunction(self.fesV)   # piecewise linear
            self.Vvec = GridFunction(self.fesV)
            self.Hvec = GridFunction(self.fes)
            self.Solution = GridFunction(self.fesMix)
            self.Draw_opt = Draw_opt
            self.MethodName = 'LapVDN'
            self.Mesh_Quality = []

        def IniH(self):
            print('Not yet')

        def WeakSD(self):
            print('Check Modification')
            gam , H , V , z, nu, v = self.fesMix.TrialFunction()
            gamt, Ht, Vt, zt, nut, vt= self.fesMix.TestFunction()

            vT = self.Solution.components[-1] - self.NvecD*\
                InnerProduct(self.Solution.components[-1],self.NvecD)/InnerProduct(self.NvecD,self.NvecD)
            ## Weingarten Map
            A = -grad(self.NvecD).Trace()
            A = A.trans

            Lhs = BilinearForm(self.fesMix,symmetric=False)
            ## self.ds may blow up
            Lhs += (InnerProduct(v,self.NvecD)*gamt)*self.ds_lumping-V*gamt*ds
            Lhs += (H*Ht+self.dt*(InnerProduct(grad(V).Trace(),grad(Ht).Trace())-InnerProduct(A,A)*V*Ht\
                             -InnerProduct(vT,grad(H).Trace())*Ht))*ds
            Lhs += (V*Vt-InnerProduct(grad(H).Trace(),grad(Vt).Trace()))*ds

            Lhs += (InnerProduct(z,zt)+InnerProduct(grad(nu).Trace(),grad(zt).Trace()))*ds\
                    -(InnerProduct(A,A)*InnerProduct(nu,zt))*ds
            Lhs += InnerProduct(nu,nut)*ds+(self.dt*(-InnerProduct(grad(z).Trace(),grad(nut).Trace())\
                                            -InnerProduct(grad(nu).Trace()*vT,nut)\
                                            -2*InnerProduct(grad(nut).Trace()*(A*grad(self.Hvec).Trace()), nu)\
                                            -InnerProduct(grad(self.Hvec).Trace(),grad(self.Hvec).Trace())*InnerProduct(nu,nut)\
                                            -self.Hvec*InnerProduct(A*z,nut)\
                                            +InnerProduct(A*z,A*nut) + InnerProduct(A*grad(H).Trace(),A*nut)))*ds
            Lhs += (gam*InnerProduct(self.NvecD,vt))*ds+InnerProduct(grad(v).Trace(),grad(vt).Trace())*ds
            self.lhs = Lhs

            Rhs = LinearForm(self.fesMix)
            Rhs += self.Hvec*Ht*ds
            Rhs += InnerProduct(self.NvecD,nut)*ds
            self.rhs = Rhs

        def Solving(self,vtk_Obj_1d:Vtk_out_1d):
            while self.t<self.T:
                tauval = self.dt.Get()

                if sum(abs(self.t-self.Adjust_time)<tauval*0.5):
                    self.IniH('LapV')
                    print('Reinit at {}'.format(self.t))

                self.lhs.Assemble()
                self.rhs.Assemble()
                self.Solution.vec.data = self.lhs.mat.Inverse(inverse="pardiso")*self.rhs.vec

                self.Hvec.vec.data = self.Solution.components[1].vec.data
                Normvec = np.zeros((self.nr,2))
                for i in range(self.dim):
                    Normvec[:,i] = self.Solution.components[-2].components[i].vec.FV().NumPy()

                for i in range(self.dim):
                    self.NvecD.components[i].vec.data = BaseVector(Normvec[:,i])
                    self.Disp.components[i].vec.data += BaseVector(self.Solution.components[-1].components[i].vec.FV().NumPy()*tauval)
                    self.Position.components[i].vec.data += BaseVector(self.Solution.components[-1].components[i].vec.FV().NumPy()*tauval)

                self.mesh.SetDeformation(self.Disp)

                self.Dismesh = self.Visualize(self.Draw_opt)
                self.t += tauval
                
                # 后处理： 保存图像
                Coords = np.array([v.point for v in self.Dismesh.vertices])
                vtk_Obj_1d.Output(Coords,tnow=self.t)
                # 后处理： 计算网格质量
                Coords_order = np.array(sorted(Coords,key=lambda x: np.arctan2(x[1],x[0])))
                dis_set = np.linalg.norm(Coords_order[:-1,:]-Coords_order[1:,:],axis=1)
                dis_ = np.linalg.norm(Coords_order[0,:]-Coords_order[-1,:])
                min_dis = min(min(dis_set),dis_)
                max_dis = max(max(dis_set),dis_)
                self.Mesh_Quality.append([max_dis/min_dis,self.t])
            return self.Dismesh

    class LapVDN_SD_HI1(LapVDNSD):
        '''Geometric Curvature as Initial data'''
        def IniH(self,opt='init'):
            if opt=='init':
                print('Initial by exact parameterization')
                HStr, NxStr, NyStr = HNCoef(Par_v,phi,N)
                Hfunc = sym.lambdify(phi,HStr,'numpy')
                Nxfunc = sym.lambdify(phi,NxStr,'numpy')
                Nyfunc = sym.lambdify(phi,NyStr,'numpy')
                Coords = np.array([v.point for v in self.mesh.vertices])
                theta = np.arctan2(Coords[:,1],Coords[:,0])
                self.Hvec.vec.data = BaseVector(Hfunc(theta))
                self.NvecD.vec.data = BaseVector(np.append(Nxfunc(theta),Nyfunc(theta)))
            else:
                print('Initial by Laplace X = Hn')
                self.MethodName = 'LapVDNHI1'
                ## Set Normal
                self.GetNvec()
                ## Set Curvature
                Hi,Hit = self.fes.TnT()
                Lhs = BilinearForm(self.fes,symmetric=True)
                Lhs += Hi*Hit*ds
                Rhs = LinearForm(self.fes)
                Rhs += -InnerProduct(grad(self.Position).Trace(), grad(self.NvecD).Trace())*Hit*self.ds_lumping\
                        -InnerProduct(grad(self.Position).Trace()*grad(Hit).Trace(), self.NvecD)*ds
                Lhs.Assemble()
                Rhs.Assemble()
                self.Hvec.vec.data = Lhs.mat.Inverse()*Rhs.vec

    class LapVDNMCF(NgMesh):
        '''
            Algorithm: Laplace v = kappa n, Mean curvature flow case
        '''
        def __init__(self, mymesh, T = 1e-4, tau = 1e-4, Adjust_time = [], Savetime = [], Printtime = []):
            NgMesh.__init__(self, mymesh, T, tau)
            self.lhs = []
            self.rhs = []
            ##                gam     H        nu        v
            self.fesMix = self.fes*self.fes*self.fesV*self.fesV
            self.NvecD = GridFunction(self.fesV)   # piecewise linear
            self.Vvec = GridFunction(self.fesV)
            self.Hvec = GridFunction(self.fes)
            self.Solution = GridFunction(self.fesMix)
            self.MethodName = 'LapVDNMCF'
            self.Adjust_time = np.array(Adjust_time)
            self.Save_time = np.array(Savetime)
            self.Print_time = np.array(Printtime)
            self.Scale = 1
            self.NMaxNorm = 0

        def SetScale(self,scaT):
            self.Scale = scaT

        def ScalarSolution(self,HistSol,myscale):
            self.NvecD.vec.data = BaseVector(HistSol.components[2].vec.FV().NumPy())
            self.Hvec.vec.data = BaseVector(HistSol.components[1].vec.FV().NumPy()/myscale)
            ## to set tangential velocity
            self.Solution.vec.data = BaseVector(HistSol.vec.data)

        def SetSolution(self):
            self.Solution.components[0].vec.data = BaseVector(np.zeros(self.nr))
            self.Solution.components[1].vec.data = BaseVector(np.zeros(self.nr))
            self.Solution.components[2].vec.data = BaseVector(np.zeros(self.dim*self.nr))
            self.Solution.components[3].vec.data = BaseVector(np.zeros(self.dim*self.nr))

        def IniH(self):
            self.GetNvec()
            ## Set Curvature
            Hi,Hit = self.fes.TnT()
            Lhs1 = BilinearForm(self.fes,symmetric=True)
            Lhs1 += Hi*Hit*ds
            Rhs1 = LinearForm(self.fes)
            Rhs1 += -InnerProduct(grad(self.Position).Trace(), grad(self.NvecD).Trace())*Hit*self.ds_lumping\
                    -InnerProduct(grad(self.Position).Trace()*self.NvecD, grad(Hit).Trace())*ds
            Lhs1.Assemble()
            Rhs1.Assemble()
            self.Hvec.vec.data = Lhs1.mat.Inverse()*Rhs1.vec

        def WeakMCF(self):
            gam , H , nu, v = self.fesMix.TrialFunction()
            gamt, Ht, nut, vt= self.fesMix.TestFunction()

            vT = self.Solution.components[3] - self.NvecD*\
                InnerProduct(self.Solution.components[3],self.NvecD)/InnerProduct(self.NvecD,self.NvecD)
            ## Weingarten Map
            A = -grad(self.NvecD).Trace()
            A = A.trans

            Lhs = BilinearForm(self.fesMix,symmetric=False)
            Lhs += (InnerProduct(v,self.NvecD)*gamt)*ds-H*gamt*ds
            Lhs += (H*Ht+self.dt*(InnerProduct(grad(H).Trace(),grad(Ht).Trace())-InnerProduct(A,A)*H*Ht\
                             -InnerProduct(vT,grad(H).Trace())*Ht))*ds
            Lhs += InnerProduct(nu,nut)*ds+self.dt*(-InnerProduct(grad(nu).Trace()*vT,nut)\
                                             +InnerProduct(grad(nu).Trace(),grad(nut).Trace())\
                                            -InnerProduct(A,A)*InnerProduct(nu,nut))*ds
            Lhs += (gam*InnerProduct(self.NvecD,vt))*ds+InnerProduct(grad(v).Trace(),grad(vt).Trace())*ds
            self.lhs = Lhs

            Rhs = LinearForm(self.fesMix)
            Rhs += self.Hvec*Ht*ds
            Rhs += InnerProduct(self.NvecD,nut)*ds
            self.rhs = Rhs

        def CopyNvecD(self):
            Normvec = np.zeros((self.nr,self.dim))
            for i in range(self.dim):
                Normvec[:,i] = self.Solution.components[2].components[i].vec.FV().NumPy()
            for i in range(self.dim):
                self.NvecD.components[i].vec.data = BaseVector(Normvec[:,i])
            self.NMaxNorm = max(np.linalg.norm(Normvec,axis=1))

        def Solving(self,Draw_opt):
            while self.t<self.T:
                tauval = self.dt.Get()

                if sum(abs(self.t-self.Print_time)<tauval*0.51):
                    print('Now time is:'+str(self.t))

                if sum(abs(self.t-self.Adjust_time)<tauval*0.51):
                    self.IniH()
                    print('Adjust time at '+str(self.t))

                self.lhs.Assemble()
                self.rhs.Assemble()
                self.Solution.vec.data = self.lhs.mat.Inverse(inverse="pardiso")*self.rhs.vec

                self.Hvec.vec.data = self.Solution.components[1].vec.data
                self.CopyNvecD()

                for i in range(self.dim):
                    self.Disp.components[i].vec.data += BaseVector(self.Solution.components[3].components[i].vec.FV().NumPy()*tauval)
                    self.Position.components[i].vec.data += BaseVector(self.Solution.components[3].components[i].vec.FV().NumPy()*tauval)
                self.mesh.SetDeformation(self.Disp)
                self.Dismesh = self.Visualize(Draw_opt)
                self.t += tauval
            ## 最后一步计算完的时间可能比self.T略大
            self.finalT = self.t
            return self.Dismesh

    class LapVDNMCF_Mod(LapVDNMCF):
        def CopyNvecD(self):
            Normvec = np.zeros((self.nr,self.dim))
            for i in range(self.dim):
                Normvec[:,i] = self.Solution.components[2].components[i].vec.FV().NumPy()
            for row in Normvec:
                rnorm = np.linalg.norm(row)
                row /= rnorm
            for i in range(self.dim):
                self.NvecD.components[i].vec.data = BaseVector(Normvec[:,i])

    myDict = {'Barret': Barret2008, 'LapVDNHI1': LapVDN_SD_HI1, \
             'LapVDNMCF': LapVDNMCF, 'LapVDNMCF_Mod': LapVDNMCF_Mod, 'Dziuk': Dziuk,
             'BMN':BMN}
    return myDict

class BGN():
    def __init__(self, mymesh, T = 1e-4, tau = 1e-4):
        self.mesh = mymesh
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

    def PP_Pic(self,vtk_obj=None):
        pass
    
    def PrintDuring(self):
        if self.t > self.T*self.counter_print/5:
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
            self.PrintDuring()
            self.t += tauval/self.scale**2

    def SaveFunc(self,BaseDirPath):
        Case_Name = 'C_'+format(self.T,'0.3e').replace('.','_').replace('-','_')
        flist = [self.Disp, self.X_old, self.Nvec]
        fnlist = ['Disp', 'Position', 'Nvec']
        t_scale_dict = {'t': self.t, 'scale': self.scale}
        NgMFSave(CaseName=Case_Name, BaseDirPath=BaseDirPath, mesh=self.mesh, funclist=flist, func_name_list=fnlist, data_dict=t_scale_dict)
        print('Now T={}, Save Functions in {}'.format(self.T,Case_Name))

class Dumbbell_BGN_MCF(BGN):
    def __init__(self, mymesh, T=0.0001, dt=0.0001):
        super().__init__(mymesh, T, dt)

    def PP_Pic(self,vtk_Obj_bnd:Vtk_out_BND=None):
        # 后处理： 保存图像
        perform_res = vtk_Obj_bnd.Output(self.mesh,[],tnow=self.t)
        # 后处理： 计算网格质量，利用self.mesh的拓扑关系
        if perform_res:
            self.DMesh_Obj.UpdateCoords(Pos_Transformer(self.X_old,dim=3))
            Q_Area, Q_Leng = self.DMesh_Obj.MeshQuality()
            self.Mesh_Quality.append([Q_Area,Q_Leng,self.t])

class Dumbbell_BGN_MCF_Mod(BGN):
    def __init__(self, mymesh, T=0.0001, dt=0.0001):
        super().__init__(mymesh, T, dt)
    
    def PP_Pic(self,vtk_Obj_bnd:Vtk_out_BND=None):
        # 后处理： 保存图像
        perform_res = vtk_Obj_bnd.Output(self.mesh,[],tnow=self.t)
        # 后处理： 计算网格质量，利用self.mesh的拓扑关系
        if perform_res:
            self.DMesh_Obj.UpdateCoords(Pos_Transformer(self.X_old,dim=3))
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