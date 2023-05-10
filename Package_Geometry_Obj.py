import sympy as sym
import sympy.physics.vector as spv
import numpy as np
import math
from scipy.sparse import coo_matrix
import netgen.meshing as ngm
from netgen.csg import Pnt,SplineCurve2d,CSGeometry,Revolution,Sphere
from ngsolve import *
from netgen.meshing import MeshingParameters
from netgen.csg import unit_cube

# 曲线的参数化的symbol
phi = sym.Symbol('phi')
# 三维坐标系！！
N = spv.ReferenceFrame('N')

class DiscreteMesh:
    '''
        示例：
            ElVerInd, EdVerInd = Mesh_Info_Parse(mesh)
            DMesh_Obj = DiscreteMesh(Vertices_Coords,dim=2,adim=3,ElVerInd=ElVerInd,EdVerInd=EdVerInd)
        输入：
            节点坐标Coords，网格dim，背景维度adim
            单元节点index矩阵ElVer，边节点index矩阵EdVer
            光是邻接矩阵不够，没有单元定向信息
        methods:
            UpdateCoords: update coords and bar vectors
            N_Area_El: element-wised normal and area
            WN_Ver: vertex_wised weighted normal
    '''
    def __init__(self,Coords,dim,adim,ElVerInd=None,EdVerInd=None) -> None:
        self.dim = dim
        self.adim = adim
        self.nv = Coords.shape[0]
        if (ElVerInd is None) and (EdVerInd is None):
            # 特用来处理曲线的顺序节点的情形
            ind_set = (np.append(range(self.nv),0)).reshape(-1,1)
            ElVerInd = np.concatenate((ind_set[:-1],ind_set[1:]),axis=1)
            EdVerInd = ElVerInd.copy()
        self.ElVerInd = ElVerInd
        self.EdVerInd = EdVerInd
        self.nel = self.ElVerInd.shape[0]
        self.ned = self.EdVerInd.shape[0]
        self.VerElInd, self.VerEdInd = self.RevInd()
        self.UpdateCoords(Coords=Coords)
        self.Area = 0

    def UpdateCoords(self,Coords):
        self.Coords = Coords
        self.barvec = np.array(
            [Coords[vind[1],:] - Coords[vind[0],:] for vind in self.EdVerInd]
        )

    def RevInd(self):
        # 计算每个顶点连接的单元与边的index
        ElVer = np.zeros((self.nel,self.nv))
        for ii, vind in enumerate(self.ElVerInd):
            ElVer[ii, vind] = 1
        # 顶点共享单元ind，个数不同，因此是arrayxarray
        VerElInd = np.array([line.nonzero()[0].tolist() for line in ElVer.transpose()],dtype=list)
        EdVer = np.zeros((self.ned,self.nv))
        for ii, vind in enumerate(self.EdVerInd):
            EdVer[ii, vind] = 1
        VerEdInd = np.array([line.nonzero()[0].tolist() for line in EdVer.transpose()],dtype=list)
        return VerElInd, VerEdInd

    def N_Area_El(self):
        # 计算每个单元的单位法向量以及单元面积以及边长
        N_El_matrix = np.zeros((self.nel,self.adim))
        Leng_matrix = np.zeros((self.ned,1))
        Area_matrix = np.zeros((self.nel,1))
        if self.adim == 2:
            vec2 = np.array([0,0,1])
            ECoords = np.concatenate((self.Coords,np.zeros((self.nv,1))),axis=1)
            for ii,ver_ind in enumerate(self.ElVerInd):
                assert(len(ver_ind)==2)
                bar = ECoords[ver_ind[1]] - ECoords[ver_ind[0]]
                n3d = np.cross(bar, vec2)
                N_El_matrix[ii,:] = n3d[:-1]/np.linalg.norm(n3d)
                Area_matrix[ii,:] = np.linalg.norm(n3d)
            for ii,ver_ind in enumerate(self.EdVerInd):
                bar = ECoords[ver_ind[1]] - ECoords[ver_ind[0]]
                Leng_matrix[ii,:] = np.linalg.norm(bar)
        elif self.adim == 3:
            # 三维中的二维曲面
            ECoords = self.Coords.copy()
            for ii,ver_ind in enumerate(self.ElVerInd):
                bar = ECoords[ver_ind[1:]] - ECoords[ver_ind[:-1]]
                n3d = np.cross(bar[0,:]/np.linalg.norm(bar[0,:]),
                            bar[1,:]/np.linalg.norm(bar[1,:]))
                N_El_matrix[ii,:] = n3d/np.linalg.norm(n3d)
                Area_matrix[ii,:] = np.linalg.norm(n3d)/2\
                    *np.linalg.norm(bar[0,:])*np.linalg.norm(bar[1,:])
            for ii,ver_ind in enumerate(self.EdVerInd):
                bar = ECoords[ver_ind[1]] - ECoords[ver_ind[0]]
                Leng_matrix[ii,:] = np.linalg.norm(bar)
        self.Area = sum(Area_matrix)
        return N_El_matrix, Leng_matrix, Area_matrix
    
    def WN_Ver(self):
        # 计算每个节点的加权法向量
        WN_Ver_matrix = np.zeros((self.nv,self.adim))
        N_El_matrix, Leng_matrix, Area_matrix = self.N_Area_El()
        for ii in range(self.nv):
            # 获取顶点所在的单元index
            adj_el_ind = self.VerElInd[ii]
            tmp = (N_El_matrix[adj_el_ind,:]*Area_matrix[adj_el_ind,:]).sum(axis=0)
            WN_Ver_matrix[ii,:] = tmp/np.linalg.norm(tmp)
        self.WN_Ver_matrix = WN_Ver_matrix

    def MeshQuality(self):
        _, Leng_matrix, Area_matrix = self.N_Area_El()
        # 最大面积与最小面积之比
        Q_Area = max(Area_matrix)/min(Area_matrix)
        # 最大边长与最小边长之比
        Q_Leng = max(Leng_matrix)/min(Leng_matrix)
        return Q_Area, Q_Leng

class Param1dCurve:
    '''
        Example of computing 
        import sympy as sym
        import sympy.physics.vector as spv
        N = spv.ReferenceFrame('N')

        phi = sym.Symbol('phi')
        Param = [sym.cos(phi),(0.6*sym.cos(phi)**2+0.4)*sym.sin(phi)]
        Param_np = [sym.lambdify(phi,expr) for expr in Param]

        Test_Obj = Param1dcurve(Param)
        n = 21
        Coords = np.zeros((n-1,2))
        theta = np.linspace(0,2*np.pi,n)[:-1]

        Coords[:,0] = Param_np[0](theta)+0.05*np.random.random(n-1)
        Coords[:,1] = Param_np[1](theta)+0.05*np.random.random(n-1)
        theta_set = Test_Obj.Projection(Coords)

        theta_ref = np.linspace(0,2*np.pi,2**10)
        plt.plot(Param_np[0](theta_ref),Param_np[1](theta_ref),'-')
        plt.plot(Coords[:,0],Coords[:,1],'o')
        plt.plot(Param_np[0](theta_set),Param_np[1](theta_set),'go')
        plt.gca().set_aspect('equal')
    ''' 
    def __init__(self,Param,T_min=0,T_max=2*np.pi,theta_dis=None) -> None:
        # Param: 2d list of symbolic expression
        self.Xparam, self.Yparam = Param
        self.Param_np = [sym.lambdify(phi,expr) for expr in Param]
        self.dX, self.dY = self.Xparam.diff(phi), self.Yparam.diff(phi)
        self.theta_np = sym.lambdify(phi,sym.atan2(self.dY,self.dX))
        self.ddX, self.ddY = self.dX.diff(phi), self.dY.diff(phi)
        Par_v = self.Xparam*N.x + self.Yparam*N.y
        # 曲率，法向量的string表达
        self.Hsym, self.Nxsym, self.Nysym, self.LapHsym, self.zxsym, self.zysym = HNCoef(Par_v,phi,N,LapHn=True)
        self.Hfunc, self.Nxfunc, self.Nyfunc, self.LapHfunc, self.zxfunc, self.zyfunc = [sym.lambdify(phi, expr) for expr in [
            self.Hsym, self.Nxsym, self.Nysym, self.LapHsym, self.zxsym, self.zysym]]
        
        # (x-X(s))*X'(s) + (y-Y(s))*Y'(s) = 0
        pyvars = sym.symbols('x:2')
        self.f = sym.lambdify((phi,pyvars),((pyvars[0] - self.Xparam)*self.dX + \
                            (pyvars[1] - self.Yparam)*self.dY))
        self.df = sym.lambdify((phi,pyvars),((pyvars[0] - self.Xparam)*self.ddX - self.dX**2 + \
                                             (pyvars[1] - self.Yparam)*self.ddY - self.dY**2))
        self.T_max = T_max
        self.T_min = T_min
        self.X_fun = sym.lambdify(phi,Param) # list
        if theta_dis is None:
            # 如果没有事先指定粗筛的theta集合的话
            self.N_dis = 2**12
            self.theta_dis = np.linspace(self.T_min,self.T_max,self.N_dis)[:-1]
            self.X_dis = np.array(self.X_fun(self.theta_dis)).transpose() # nx2 ndarray
    
    def Theta_By_DisMin(self,Coord):
        dis = np.linalg.norm(self.X_dis - Coord[None,:], axis=1)
        # 最小三个数的index
        ind_M3 = np.sort(np.argpartition(dis,3)[:3])
        theta = self.theta_dis[ind_M3]
        # 处理周期的参数情况
        if max(ind_M3) - min(ind_M3) > self.N_dis/2:
            for ii,ind in enumerate(ind_M3):
                if ind > self.N_dis/2:
                    theta[ii] -= self.T_max
        return np.sort(theta)

    def Get_Param_Projection(self,Coords2d:np.ndarray,threshold=1e-6, val_thres=1e-6,iter_max=15):
        theta_set = np.zeros((Coords2d.shape[0],1))
        for ii,Coord in enumerate(Coords2d):
            # 通过粗筛选定Newton迭代的初值以及上下界限
            bef_phi,init_phi,end_phi = self.Theta_By_DisMin(Coord)
            iter_num = 0
            xold = init_phi
            xnew = xold - self.f(xold,Coord)/self.df(xold,Coord)
            while np.abs(xnew - xold) > threshold or abs(self.f(xnew,Coord))>val_thres:
                xold = xnew.copy()
                xnew = xold - self.f(xold,Coord)/self.df(xold,Coord)
                iter_num += 1
                # 额外情况处理
                if xnew>end_phi or xnew<bef_phi:
                    xnew = np.arctan2(Coord[1],Coord[0])
                    print('wrong range')
                    break
                if iter_num > iter_max:
                    # 权宜之计
                    xnew = np.arctan2(Coord[1],Coord[0])
                    print('iter_num={}'.format(iter_num))
                    break
            theta_set[ii] = xnew
        return theta_set

    def Generate_Pos_Norm(self,param:np.ndarray):
        Pset = np.zeros((len(param),2))
        Normset = np.zeros((len(param),2))
        Pset[:,0] = self.Param_np[0](param).flatten()
        Pset[:,1] = self.Param_np[1](param).flatten()
        Normset[:,0] = self.Nxfunc(param).flatten()
        Normset[:,1] = self.Nyfunc(param).flatten()
        return Pset, Normset

    def Generate_Pos_Angle(self, param):
        newx = self.Param_np[0](param)
        newy = self.Param_np[1](param)
        newalpha = self.theta_np(param)
        return newx, newy, newalpha

    def PlotCurve(self,plt):
        theta_set = np.linspace(self.T_min,self.T_max,2**10)
        X_dis = np.array(self.X_fun(theta_set)).transpose()
        plt.plot(X_dis[:,0],X_dis[:,1],'-')
        plt.gca().set_aspect('equal')

class Param1dSpline(Param1dCurve):
    '''
        Parameteric, spline object in x-z plane
    '''
    def __init__(self, Param, T_max, N_Spline, eps, c_tag, T_min=0, theta_dis=None) -> None:
        super().__init__(Param, T_min, T_max, theta_dis)
        self.ctrbase, self.ctr = self.Generate_Control_Point(N_Spline,eps,c_tag)
    
    def Get_Param(self, Coords):
        # Compute parameter from 2d, rewritten by case
        pass

    def Generate_Control_Point(self,N_Spline,eps,c_tag):
        sparam = np.linspace(self.T_min,self.T_max,N_Spline) 
        newx, newz, newalpha = self.Generate_Pos_Angle(sparam)
        ctr = []
        if c_tag:
            # closed spline
            c_tag_len = len(sparam)
        else:
            # open spline
            c_tag_len = len(sparam) - 1
        for ii in range(c_tag_len):
            indnext = ii+1
            if indnext>len(sparam)-1:
                indnext = 0
            alpha0, alpha1 = newalpha[ii], newalpha[indnext]
            x0, x1 = newx[ii], newx[indnext]
            z0, z1 = newz[ii], newz[indnext]
            A = np.array([[np.sin(alpha0), -np.cos(alpha0)], 
                        [np.sin(alpha1), -np.cos(alpha1)]])
            b = np.array([[np.sin(alpha0)*x0 - np.cos(alpha0)*z0],
                        [np.sin(alpha1)*x1 - np.cos(alpha1)*z1]])
            res = np.linalg.solve(A, b).flatten()
            ctr.append([res[0]-eps,res[1]+eps])
        ctr = np.array(ctr)
        ctrbase = np.zeros((len(newz),2))
        ctrbase[:,0] = newx
        ctrbase[:,1] = newz
        return ctrbase, ctr

    def Generate_Control_Point_c0(self,N_Spline,eps,c_tag):
        '''
            Special treat for T_min=0, T_max=pi, more points at two ends
        '''
        sparam = np.linspace(self.T_min,self.T_max,N_Spline) 
        sparam = (np.cos(sparam)-1)*(-np.pi/2)
        newx, newz, newalpha = self.Generate_Pos_Angle(sparam)
        ctr = []
        if c_tag:
            # closed spline
            c_tag_len = len(sparam)
        else:
            # open spline
            c_tag_len = len(sparam) - 1
        for ii in range(c_tag_len):
            indnext = ii+1
            if indnext>len(sparam)-1:
                indnext = 0
            alpha0, alpha1 = newalpha[ii], newalpha[indnext]
            x0, x1 = newx[ii], newx[indnext]
            z0, z1 = newz[ii], newz[indnext]
            A = np.array([[np.sin(alpha0), -np.cos(alpha0)], 
                        [np.sin(alpha1), -np.cos(alpha1)]])
            b = np.array([[np.sin(alpha0)*x0 - np.cos(alpha0)*z0],
                        [np.sin(alpha1)*x1 - np.cos(alpha1)*z1]])
            res = np.linalg.solve(A, b).flatten()
            ctr.append([res[0]-eps,res[1]+eps])
        ctr = np.array(ctr)
        ctrbase = np.zeros((len(newz),2))
        ctrbase[:,0] = newx
        ctrbase[:,1] = newz
        return ctrbase, ctr

class EllipseSpline(Param1dSpline):
    def __init__(self, a, b, T_max=np.pi, N_Spline=3, eps=1e-5, c_tag=False, T_min=0, theta_dis=None) -> None:
        self.a, self.b = a, b
        self.T_max, self.T_min = T_max, T_min
        self.Param = [a*sym.cos(phi),b*sym.sin(phi)]
        super().__init__(Param=self.Param, T_max=self.T_max, N_Spline=N_Spline, eps=eps, c_tag=c_tag, T_min=T_min)

    def Get_Param(self, Coords):
        x_coord = Coords[:,0]
        z_coord = Coords[:,1]
        phi_np = np.arctan2(z_coord/self.b, x_coord/self.a)
        return phi_np

class DumbbellSpline(Param1dSpline):
    '''
        Profile of the Dumbbell:
        x = cos phi 
        y = (a cos^2 phi + b) cos theta sin phi
        z = (a cos^2 phi + b) sin theta sin phi
        by Rotation around the x axis
        x = cos phi 
        z = (a cos^2 phi + b) sin phi
        Example:

        DPObj = DumbbellProfile()
        CoordRef = DPObj.DisCurveObj.X_dis
        plt.plot(CoordRef[:,0],CoordRef[:,1],'-')
        Coords = np.array([
            [0.5,0.45],
            [0.6,0.3]
        ])
        NewCoords, Normals = DPObj.NearestPoint(Coords)
        plt.plot(Coords[:,0],Coords[:,1],'ro')
        plt.plot(NewCoords[:,0],NewCoords[:,1],'ro')
        plt.quiver(NewCoords[:,0],NewCoords[:,1],Normals[:,0],Normals[:,1])
        plt.gca().set_aspect('equal')
    '''
    def __init__(self,a,b,N_Spline,T_max,eps,c_tag, T_min=0) -> None:
        self.a, self.b = a, b
        self.T_max = T_max
        self.T_min = T_min
        # sym.cos(phi)*N.x + (a*sym.cos(phi)**2+b)*sym.sin(phi)*N.z
        self.Param = [sym.cos(phi),(a*sym.cos(phi)**2+b)*sym.sin(phi)]
        super().__init__(Param=self.Param, T_max=self.T_max, N_Spline=N_Spline, eps=eps, c_tag=c_tag)

    def Get_Param(self, Coords):
        ## by arctan2, get values in [0,pi] (symmetric). What if point not on the curve???
        x_coord = Coords[:,0]
        z_coord = Coords[:,1]
        phi_np = np.arctan2(z_coord/(self.a*x_coord**2+self.b), x_coord)
        return phi_np

class RBCSpline(Param1dSpline):
    '''
        x = -(b-a (cos phi^2-1)^2) sin phi
        z = c*cos phi 
    '''
    def __init__(self,a,b,c,N_Spline,T_min,T_max,eps,c_tag) -> None:
        self.a, self.b, self.c = a, b, c
        self.T_max = T_max
        self.T_min = T_min
        self.Param = [-(b-a*(sym.cos(phi)**2-1)**2)*sym.sin(phi), c*sym.cos(phi)]
        super().__init__(Param=self.Param, T_min = self.T_min, T_max=self.T_max, N_Spline=N_Spline, eps=eps, c_tag=c_tag)

    def Get_Param(self, Coords):
        ## by arctan2, get values in [0,pi] (symmetric). What if point not on the curve???
        x_coord = Coords[:,0]
        z_coord = Coords[:,1]/self.c
        phi_np = np.arctan2(-x_coord/(self.b-self.a*(z_coord**2-1)**2), z_coord)
        return phi_np

class FlowerCurve(Param1dCurve):
    def __init__(self, T_max=2 * np.pi, theta_dis=None, a=0.65, b=7) -> None:
        Param = [(a*sym.sin(b*phi)+1)*sym.cos(phi), (a*sym.sin(b*phi)+1)*sym.sin(phi)]
        super().__init__(Param = Param, T_max = T_max, theta_dis = theta_dis)    



## Rotational 2d surface: curvature, Laplace curvature, normal and etc, GenerateMesh
class Param2dRot():
    '''
        Rotational 2d surface by profile curve in x-z plane (f(phi), g(phi)).
        Around x axis: (f(phi), g(phi)cos theta, g(phi)sin theta).
        Around z axis: (f(phi)cos theta, f(phi)sin theta, g(phi))
    '''
    def __init__(self,Spline_Obj:Param1dSpline,axis_opt,c_tag) -> None:
        self.Spline_Profile_Obj = Spline_Obj
        self.Rot_Axis = axis_opt
        self.Closed_Tag = c_tag

        f, g = self.Spline_Profile_Obj.Xparam, self.Spline_Profile_Obj.Yparam
        df, dg = f.diff(phi), g.diff(phi)
        ddf, ddg = df.diff(phi), dg.diff(phi)
        # normal of profile curve
        self.nhat_1 = dg/sym.sqrt(df**2+dg**2)
        self.nhat_2 = -df/sym.sqrt(df**2+dg**2)
        if self.Rot_Axis == 'x':
            self.H_sym = -((dg*ddf-ddg*df)/(df**2+dg**2)**(3/2)+df/(g*sym.sqrt(df**2+dg**2)))
            self.H_np = sym.lambdify(phi, self.H_sym)
            # metric (orthogonal), diagnoal
            self.gij = [df**2+dg**2, g**2]
            self.invgij = [1/(df**2+dg**2), 1/g**2]
            self.detg = (df**2+dg**2)*g**2
        elif self.Rot_Axis == 'z':
            # wait for check
            self.H_sym = -((df*ddg-ddg*dg)/(df**2+dg**2)**(3/2)+dg/(f*sym.sqrt(df**2+dg**2)))
            self.H_np = sym.lambdify(phi, self.H_sym)

    def LapS_theta_indep(self,fexpr):
        # surface Laplace of function independent of theta - due to rotational symmetry
        Lapf = 1/sym.sqrt(self.detg)*(
            sym.sqrt(self.detg)*self.invgij[0]*fexpr.diff(phi)
        ).diff(phi)
        return Lapf

    def Generate_LapHn_func(self):
        '''
            Compute some intermediate results symbolically, pre-step for Get_Lap_H_N
        '''
        self.LapH_func, self.Lapnhat1_func, self.Lapnhat2_func = [
            sym.lambdify(phi, self.LapS_theta_indep(fexpr)) 
            for fexpr in [self.H_sym, self.nhat_1, self.nhat_2]
        ]
        self.g22nhat2 = sym.lambdify(phi,self.invgij[1]*self.nhat_2)
        n_prim_square = (self.nhat_1.diff(phi))**2+(self.nhat_2.diff(phi))**2
        self.A2_func  = sym.lambdify(phi, self.invgij[0]*n_prim_square 
                                    + self.invgij[1]*self.nhat_2**2)
    
    def Get_Lap_H_N(self,phi:np.ndarray, theta:np.ndarray):
        '''
            Computing pointwise ∆ H and ∆ n and 
            
            by transforming ∆ H and ∆ n symbolic expression to numpy functions.
        '''
        n = len(phi)
        assert len(phi) == len(theta)
        LapH, Lapn = np.zeros((n,1)), np.zeros((n,3))
        if self.Rot_Axis == 'x':
            LapH[:,0] = self.LapH_func(phi)
            Lapn[:,0] = self.Lapnhat1_func(phi)
            Lapn[:,1] = self.Lapnhat2_func(phi)*np.cos(theta) - self.g22nhat2(phi)*np.cos(theta)
            Lapn[:,2] = self.Lapnhat2_func(phi)*np.sin(theta) - self.g22nhat2(phi)*np.sin(theta)
            A_Frob = self.A2_func(phi)
        return LapH, Lapn, A_Frob.reshape(-1,1)

    def Get_Profile_Coord(self,Coords3d):
        Coord_Profile = np.zeros((Coords3d.shape[0],2))
        if self.Rot_Axis == 'x':
            # x-z curve rotate around x axis (f(phi), g(phi)cos theta, g(phi)sin theta)
            Coord_Profile[:,0] = Coords3d[:,0]
            Coord_Profile[:,1] = np.linalg.norm(Coords3d[:,1:],axis=1)
        elif self.Rot_Axis == 'z':
            # x-z curve rotate around z axis (f(phi)cos theta, f(phi)sin theta, g(phi))
            Coord_Profile[:,0] = np.linalg.norm(Coords3d[:,:-1],axis=1)
            Coord_Profile[:,1] = Coords3d[:,2]
        return Coord_Profile

    def Get_Param(self,Coords3d,Near_opt=False):
        '''
            Determin Parameter of 3d Coords by Rotation axis and Profile Curve
        '''
        Coord_Profile = self.Get_Profile_Coord(Coords3d)
        if self.Rot_Axis == 'x':
            # x-z curve rotate around x axis (f(phi), g(phi)cos theta, g(phi)sin theta)
            theta_np = np.arctan2(Coords3d[:,2],Coords3d[:,1])
        elif self.Rot_Axis == 'z':
            # x-z curve rotate around z axis (f(phi)cos theta, f(phi)sin theta, g(phi))
            theta_np = np.arctan2(Coords3d[:,1],Coords3d[:,0])
        if not Near_opt:
            # point exactly on the profile curve
            phi_np = self.Spline_Profile_Obj.Get_Param(Coord_Profile)
        else:
            # point not on the profile curve
            phi_np = self.Spline_Profile_Obj.Get_Param_Projection(Coord_Profile)
        return phi_np, theta_np

    def Get_Pos_Norm(self,phi:np.ndarray, theta:np.ndarray):
        n = len(phi)
        assert len(phi) == len(theta)
        Pos_Profile, Norm_Profile = self.Spline_Profile_Obj.Generate_Pos_Norm(phi)
        Pset, Normset = np.zeros((n,3)), np.zeros((n,3))
        if self.Rot_Axis == 'x':
            Pset[:,0] = Pos_Profile[:,0]
            Pset[:,1] = Pos_Profile[:,1]*np.cos(theta)
            Pset[:,2] = Pos_Profile[:,1]*np.sin(theta)
            Normset[:,0] = Norm_Profile[:,0]
            Normset[:,1] = Norm_Profile[:,1]*np.cos(theta)
            Normset[:,2] = Norm_Profile[:,1]*np.sin(theta)
        elif self.Rot_Axis == 'z':
            print('Rotation around z-s Normal is not available yet!')
        return Pset, Normset
    
    def Generate_Mesh(self,maxh,order,RN=0,Local_h=None):
        if self.Rot_Axis == 'x':
            axis_0, axis_1 = Pnt(0,0,0), Pnt(1,0,0)
        elif self.Rot_Axis == 'z':
            axis_0, axis_1 = Pnt(0,0,0), Pnt(0,0,1)
        spline = SplineCurve2d() # create a 2d spline
        # define the control points -- anti clockwise
        ctrbase, ctr = self.Spline_Profile_Obj.ctrbase, self.Spline_Profile_Obj.ctr
        if self.Closed_Tag == False:
            # open spline on x-axis, rotate around x-axis
            ctrbase[-1,-1] = 0
            ctrbase[0,-1] = 0
            n_ctr = len(ctr)
        else:
            # closed spline, number of ctr = number of ctrbase
            n_ctr = len(ctr) + 1
        n_spline = n_ctr
        # collect pnts
        pnts = []
        for ii in range(len(ctrbase)):
            tmp = ctrbase[ii].tolist()
            pnts.append(tuple(tmp))
            if ii < n_ctr:
                pnts.append(tuple(ctr[ii]))
        # collect splines
        segs = []
        for ii in range(n_spline):
            ind0, ind1, ind2 = 2*ii, 2*ii+1, 2*ii+2
            if ii >= len(ctrbase)-1:
                ind2 = 0
            segs.append([ind0,ind1,ind2])
        # add the points and segments to the spline
        for pnt in pnts:
            spline.AddPoint (*pnt)
        for seg in segs:
            spline.AddSegment (*seg)
        # revolve the spline
        rev = Revolution(axis_0, axis_1, spline)
        # create a geometry and add the revolved object
        self.geo = CSGeometry()
        self.geo.Add (rev.col([1,0,0]))

        mp = MeshingParameters (maxh=maxh, perfstepsend=ngm.MeshingStep.MESHSURFACE)

        if not Local_h is None:
            x_set,y_set,z_set,h_set = Local_h
            for ii in range(len(x_set)):
                x = x_set[ii]
                y = y_set[ii]
                z = z_set[ii]
                h = h_set[ii]
                if abs(x)<np.inf:
                    mp.RestrictH(x=x,y=y,z=z,h=h)

        self.mesh = Mesh(self.geo.GenerateMesh(mp = mp))
        Coords = np.array([v.point for v in self.mesh.vertices])
        eta2 = []
        for el in self.mesh.Elements(BND):
            x_average = 0
            ii = 0
            for v in el.vertices:
                x_average += Coords[v.nr][0]
                ii += 1
            eta2.append(x_average/ii)
        self.mesh.Curve(order)
        for el in self.mesh.Elements(BND):
            self.mesh.SetRefinementFlag(el, abs(eta2[el.nr]) > 0.85)
        for ii in range(RN):
            self.mesh.Refine(mark_surface_elements=True)
        print('Mesh with ne {}, nface {}, nv {}'.format(self.mesh.ne, self.mesh.nface, self.mesh.nv))

class Ellipsoid(Param2dRot):
    def __init__(self, a, b, eps, NS) -> None:
        Spline_Obj = EllipseSpline(a=2, b=1, eps=eps, N_Spline=NS)
        super().__init__(Spline_Obj, axis_opt='x', c_tag=False)

class RBC_Rot_Obj(Param2dRot):
    '''
        MyRBC_Obj = RBC_Obj(a=0.4,b=1,c=2,N_Spline=9)
        MyRBC_Obj.Generate_Mesh(maxh=0.2,order=1)
        mesh = RBC_Obj.mesh
    '''
    def __init__(self, a, b, c, N_Spline) -> None:
        Profile_Obj = RBCSpline(a=a,b=b,c=c,N_Spline=N_Spline,T_min=-np.pi/2,T_max=np.pi/2,eps=1e-7,c_tag=False)
        super().__init__(Profile_Obj, axis_opt='x', c_tag=False)

class Dumbbell_Rot_Obj(Param2dRot):
    def __init__(self, Spline_Obj: Param1dSpline, axis_opt, c_tag) -> None:
        super().__init__(Spline_Obj, axis_opt, c_tag)

## other types of 2d surface mesh
class CubicSurface():
    def __init__(self) -> None:
        pass

    def Generate_Mesh(self,maxh):
        self.mesh=Mesh(unit_cube.GenerateMesh(maxh=maxh, optsteps2d=3, perfstepsend=ngm.MeshingStep.MESHSURFACE))
        ElVerInd, EdVerInd = Mesh_Info_Parse(self.mesh)
        Vertices_Coords = np.array([v.point for v in self.mesh.vertices])
        self.DMesh_Obj = DiscreteMesh(Vertices_Coords,dim=2,adim=3,ElVerInd=ElVerInd,EdVerInd=EdVerInd)

class AngenentTorus_Rot_Obj(Param2dRot):
    '''
        Not prepared.
    '''
    def __init__(self, Spline_Obj: Param1dSpline, axis_opt, c_tag) -> None:
        super().__init__(Spline_Obj, axis_opt, c_tag)

## 1 dimensional curve mesh - first order
class FlowerCurveMesh:
    def __init__(self, box, nnode, dt, symm, a=0.65, b=7):
        self.box = box
        self.symm = symm
        self.dt = dt
        # Flower curve parameterization --  symbolic version
        Par_x = (a*sym.sin(b*phi)+1)*sym.cos(phi)
        Par_y = (a*sym.sin(b*phi)+1)*sym.sin(phi)
        Par_v = Par_x*N.x + Par_y*N.y
        self.Par_func = [sym.lambdify(phi,fun_i) for fun_i in Par_v.to_matrix(N)]
        self.Fscale = 1.05
        if symm:
            theta = np.linspace(box[0],box[1],nnode+1)
            dsize = (box[1]-box[0])/nnode
            self.theta = np.append(theta-dsize/2,box[1]+dsize/2)
            self.nnode = nnode+2
            self.nvs = nnode
        else:
            self.theta = np.linspace(box[0],box[1],nnode)
            self.nnode = nnode
            self.nvs = nnode-2
        
        self.p = np.zeros((self.nnode,2))
        self.ComputParam()
        indp = list(range(self.nnode))
        self.bars = np.transpose(np.array([indp[1:],indp[:-1]]))
        self.barvec = np.zeros((self.nnode-1,2)) 
        ## Normal of visual points
        self.nvec = np.zeros((self.nvs,2))
        self.Fvis = np.zeros((self.nvs,2))
    
    def ComputParam(self):
        phi = self.theta
        self.p[:,0] = self.Par_func[0](phi)
        self.p[:,1] = self.Par_func[1](phi)
        
    def ComputBar(self):
        self.barvec = self.p[self.bars[:,0],:]-self.p[self.bars[:,1],:]
    
    def WeightedNorm(self):
        nvec = np.zeros(self.barvec.shape)
        Norm = np.sqrt((self.barvec**2).sum(axis=1))
        nvec[:,0] = self.barvec[:,1]/Norm
        nvec[:,1] = -self.barvec[:,0]/Norm
        Norm = Norm.reshape(-1,1)
        self.nvec = nvec[:-1]*Norm[:-1]+nvec[1:]*Norm[1:]
        for i in range(self.nvec.shape[0]):
            self.nvec[i] /= np.linalg.norm(self.nvec[i])
        
    def Constrain(self):
        '''
            根据theta角度以及单参数表达式进行拉回
            找到 (-pi/14,pi/14) 内部的点的坐标（首尾根据对称性再次确定），按照幅角排序
            断言第二个点比 -pi/14 更大，如果第一个点小于，则取两者平均
            再调整首尾端点取关于box对称的点
        '''
        pvis = self.p[1:-1]
        # By theta
        newtheta = np.sort([math.atan2(pj[1],pj[0]) for pj in pvis])
        if self.symm:
            if newtheta[1]<self.box[0] or newtheta[-2]>self.box[1]:
                print('wrong')
            elif newtheta[0]<self.box[0]:
                newtheta[0] = (self.box[0]+newtheta[1])/2
            elif newtheta[-1]>self.box[1]:
                newtheta[-1] = (self.box[1]+newtheta[-2])/2

            self.theta[0] = 2*self.box[0]-newtheta[0]
            self.theta[1:-1] = newtheta
            self.theta[-1] = 2*self.box[1]-newtheta[-1]
        else:
            self.theta[1:-1] = newtheta
        self.ComputParam()
        
    def Force(self):
        barvec = self.barvec
        bars = self.bars
        L = np.linalg.norm(barvec,axis=1)
        L = L.reshape(-1,1)
        hbars = np.ones((self.nnode-1,1))
        L0 = hbars*self.Fscale*np.sqrt((L**2).sum()/(hbars**2).sum())
        F = np.array([[(L0-L)[ii][0]] for ii in range(len(L))])
        Fvec=np.kron(F/L,[1,1])*barvec
        row = bars[:,[0,0,1,1]].flatten()
        column = np.kron(np.ones(F.shape),[0,1,0,1]).flatten()
        val = np.concatenate((Fvec,-Fvec),axis=1).flatten()
        Ftot = coo_matrix((val,(row,column)),shape=(self.nnode,2)).todense()
        
        Ftot = np.array(Ftot)
        
        self.Fvis = Ftot[1:-1]
        ## Projection to Tangent Part
        for i,F in enumerate(self.Fvis):
            self.Fvis[i] = F-(F*self.nvec[i]).sum()*self.nvec[i]
        
    def Deformation(self):
        self.p[1:-1] += self.dt*self.Fvis
        self.Constrain()
    
    def SymmetryMesh(self):
        '''
            返回p:nx2坐标矩阵以及PhiTot:幅角
        '''
        if self.symm:
            # res = np.append(self.box[0], self.theta[1:-1])
            # res = np.append(res, self.box[1])
            # res = np.append(res, np.sort(2*self.box[1]-self.theta[1:-1]))
            res = np.append(self.theta[1:-1], np.sort(2*self.box[1]-self.theta[1:-1]))
        else:
            res = np.concatenate((self.theta[:], np.sort(2*self.box[1]-self.theta[1:-1])))
        PhiTot = []
        for i in range(7):
            PhiTot = np.append(PhiTot, res+i*2*(self.box[1]-self.box[0]))
        p = np.zeros((len(PhiTot),2))
        phi = PhiTot
        p[:,0] = self.Par_func[0](phi)
        p[:,1] = self.Par_func[1](phi)
        return p,PhiTot

def Coord1dFlower(n_half_blade,num_blade=7,radparam=0.65):
    '''
        生成花型网格，输入参数为对称的半个叶片曲线上的采样点个数。输出 p: nx2 坐标矩阵； PhiTot: 极角坐标
    '''
    EquivMesh = FlowerCurveMesh([-np.pi/(2*num_blade),np.pi/(2*num_blade)],n_half_blade,0.02,True, a=radparam, b=num_blade)
    while True:
        EquivMesh.ComputBar()
        EquivMesh.WeightedNorm()
        EquivMesh.Force()
        if np.var(np.linalg.norm((EquivMesh.barvec),axis=1))>1e-5:
            EquivMesh.Deformation()
        else: 
            break
    p,PhiTot = EquivMesh.SymmetryMesh()
    return p,PhiTot

def Mesh1dFromPoints(Coords,dim,adim):
    nr = Coords.shape[0]
    mesh0= ngm.Mesh(dim=adim)
    pids = []
    for i in range(nr):
        pids.append (mesh0.Add (ngm.MeshPoint(ngm.Pnt(np.append(Coords[i],np.array([0]))))))  

    idx = mesh0.AddRegion("material", dim=dim)   ## dim=1: Boundary For a Mesh of adim=2
    for i in range(nr):
        if i<nr-1:
            mesh0.Add(ngm.Element1D([pids[i],pids[i+1]],index=idx))
        else:
            mesh0.Add(ngm.Element1D([pids[i],pids[0]],index=idx))
    mymesh = Mesh(mesh0)
    return mymesh

def Mesh_Info_Parse(mesh):
    ElVerInd = []
    EdVerInd = []
    for el in mesh.ngmesh.Elements2D():
        # v.nr 从1开始计数
        ElVerInd.append([v.nr-1 for v in el.vertices])
    for ed in mesh.edges:
        # v.nr 从0开始计数（ngmesh与ngsolve.Mesh之间的区别）
        EdVerInd.append([v.nr for v in ed.vertices])
    return np.array(ElVerInd), np.array(EdVerInd)

def MeshSphere(maxh,R=2,order=1,maxh_Out=0.4):
    sphere = Sphere(Pnt(0,0,0),R)
    sphere.bc("Outer").maxh(maxh_Out)
    dom = sphere
    geo = CSGeometry()
    geo.Add(dom)
    mesh = geo.GenerateMesh(maxh = maxh,optsteps2d=3, perfstepsend=ngm.MeshingStep.MESHSURFACE)
    mymesh = Mesh(mesh)
    mymesh.Curve(order)
    return mymesh

def HNCoef(Par_v:spv.vector.Vector, phi, frame, LapHn=False):
    '''
        计算平面曲线的曲率：X=(f,g)
        H = (f' * g'' - g' * f'')/(f'^2+g'^2)^(3/2)
        圆周为 1
        返回symbolic的标量函数
    '''
    dv = Par_v.diff(phi,frame)
    ddv = dv.diff(phi,frame)
    # rphi rotate clockwise 90 to normal
    normalv = dv^frame.z
    # normalize the normal vector
    normalv = normalv.normalize()
    ## Generate Curvature
    H = (dv.to_matrix(frame)[0]*ddv.to_matrix(frame)[1]-dv.to_matrix(frame)[1]*ddv.to_matrix(frame)[0])/(dv.to_matrix(frame)[0]**2+dv.to_matrix(frame)[1]**2)**(3/2)
    nx_S = normalv.dot(frame.x)
    ny_S = normalv.dot(frame.y)
    if LapHn == True:
        s = sym.sqrt(dv.dot(dv))
        LapH_val = (H.diff(phi)/s).diff(phi)/s
        # z = Lap n + |A|^2 n, |A|^2 = H^2
        zx_val = ((nx_S.diff(phi)/s).diff(phi)/s) + H**2*nx_S
        zy_val = ((ny_S.diff(phi)/s).diff(phi)/s) + H**2*ny_S
    else:
        LapH_val = None
        zx_val, zy_val = None, None
    return H, nx_S, ny_S, LapH_val, zx_val, zy_val

class D1_FlowerCurve(Param1dCurve):
    '''
        一维的花型曲线：一种特殊的一维参数曲线
            输入：
                确定花型的解析表达的半径参数a与周期参数b
                确定花型离散网格的半单周期网格点数nnode
            解析表示：
                x = (a*sin(b*phi)+1)*cos(phi)
                y = (a*sin(b*phi)+1)*sin(phi)
            方法：
                计算参数化的数值函数，参数化的符号表达str，法向量和曲率符号表达str
                生成坐标决定的参数化曲线
    ''' 
    def __init__(self,params:list=[0.65,7],nnode:int=12) -> None:
        '''
            ParSymStr: list of str; 
            ParFunc: func; 
            DMesh: DiscreteCurve;
            Projection 
        '''
        self.params = params
        a,b = self.params
        # 解析的参数化设置
        self.ParSymStr = ['({}*sym.sin({}*phi)+1)*sym.cos(phi)'.format(a,b),\
        '({}*sym.sin({}*phi)+1)*sym.sin(phi)'.format(a,b)]
        # 生成参数化的符号表示以及Numpy表达式
        Par_Np = [component.replace('sym','np') for component in self.ParSymStr]
        # position的数值计算函数 
        ## 输入必须是ndarray，并且是行向量，决不能是list
        self.ParFunc = lambda phi: np.append(eval(Par_Np[0])[:,None], eval(Par_Np[1])[:,None], axis=1)
        # 继承参数化曲线的法向投影-曲率-法向量的string表达
        super().__init__([eval(component) for component in self.ParSymStr])
        self.DMesh = self.Generate1dMesh(symm=True, nnode=nnode)

    def Generate1dMesh(self,symm=True,dt=0.02,nnode=6) -> DiscreteMesh:
        freq = self.params[1]
        semi_period = np.pi/freq
        # 对应最小与最大半径
        symm_box = [-semi_period/2, semi_period/2]
        if symm:
            theta = np.linspace(symm_box[0],symm_box[1],nnode+1)
            dsize = semi_period/nnode
            # 最小与最大半径之间的节点theta
            theta = np.append(theta-dsize/2,symm_box[1]+dsize/2)[1:-1]
            # 对称延拓到单周期
            Etheta_per = np.array(
                [theta, np.sort(semi_period-theta)]
            )
            # 延拓到所有周期
            Etheta = []
            for ii in range(freq):
                Etheta = np.append(Etheta, Etheta_per+ii*2*semi_period)
        Coords = self.ParFunc(phi = Etheta)
        nv = Coords.shape[0]
        ElVerMatrix = np.array(
            [[ii, ii+1] for ii in range(nv)]
        )
        ElVerMatrix[-1,-1] = 0
        EdVerMatrix = ElVerMatrix.copy()
        return DiscreteMesh(Coords,1,2,ElVerMatrix,EdVerMatrix)

class ReParam_BarMod:
    '''
        坐标调整，根据网格的对称性可以选择对称选项来处理自由的边界点
        最重要的属性：
            法向量：切向速度调整
            Force：确定速度大小
    '''
    def __init__(self,ParDisCurve:D1_FlowerCurve,Symm=True,dt=0.02) -> None:
        self.dt = dt
        self.ParDisCurve = ParDisCurve
        self.ParFunc = ParDisCurve.ParFunc
        self.ParSymStr = [eval(component) for component in ParDisCurve.ParSymStr]
        self.Fscale = 1.05
        if Symm:
            # 1d对称情形：首尾是内点对称出来的，内点个数减2
            self.DMesh = ParDisCurve.DMesh
            self.Fvis = np.zeros((self.DMesh.nv,2))
        else:
            pass

    def ModifiedMesh(self):
        '''
            调整网格，首先计算 
        '''
        newcoords = self.DMesh.Coords.copy()
        while True:
            # 更新加权法向量，计算切向的force
            self.DMesh.WN_Ver()
            self.Force()
            # 等距网格的停机准则
            if np.var(np.linalg.norm((self.DMesh.barvec),axis=1))>1e-4:
                self.CheckSymm(self.DMesh.Coords[-1,:],self.DMesh.Coords[0,:])
                newcoords += self.dt*self.Fvis
                newcoords = self.Constrain(newcoords)
            else:
                break
            self.DMesh.UpdateCoords(newcoords)
            Q_Area, Q_Leng = self.DMesh.MeshQuality()
            print('Mesh Quality: {}, {}'.format(Q_Area, Q_Leng))
        
    def Constrain(self,coords):
        theta = self.ParDisCurve.Get_Param_Projection(coords, threshold=1e-5)
        newcoords = self.ParFunc(theta.flatten())
        return newcoords
        
    def CheckSymm(self, vec1: np.ndarray, vec2: np.ndarray):
        proximal = np.array(
            [np.cos(-np.pi/14), np.sin(-np.pi/14)]
        )
        v1normal = np.dot(vec1, proximal)
        v2normal = np.dot(vec2, proximal)
        print('normal diff is {}'.format(v1normal - v2normal))
        diff_tan = np.linalg.norm(
            vec2 - v2normal * proximal + (vec1 - v1normal * proximal)
        )
        print('tangential diff is {}'.format(diff_tan))

    def Force(self):
        '''
            前置：计算加权法向量 
        '''
        _, Leng_matrix, _ = self.DMesh.N_Area_El()
        hbars = np.ones((self.DMesh.ned,1)) # edge的权重
        L0 = hbars*self.Fscale*np.sqrt((Leng_matrix**2).sum()/(hbars**2).sum())
        F = np.array([[(L0-Leng_matrix)[ii][0]] for ii in range(self.DMesh.ned)])
        # barforce, in direction of barvec
        Fvec=np.kron(F/Leng_matrix,[1,1])*self.DMesh.barvec
        # 若边的端点为i->j,则Ftot的[jjii]行,[0,1,0,1]列取值[F1,F2,-F1,-F2]
        # 即j点（终点）的Force为F,i点（起点）的Force为-F,能保持轴对称性质
        row = (self.DMesh.EdVerInd[:,[1,1,0,0]]).flatten()
        column = np.kron(np.ones(F.shape),[0,1,0,1]).flatten()
        val = np.concatenate((Fvec,-Fvec),axis=1).flatten()
        # i-th row -- the force of i-th vertex
        self.Ftot = np.array(
            coo_matrix((val,(row,column)),shape=(self.DMesh.nv,2)).todense()
        )
        ## Projection to Tangent Part
        for i,F in enumerate(self.Ftot):
            nvec = self.DMesh.WN_Ver_matrix[i]
            self.Fvis[i] = F # -(F*nvec).sum()*nvec
