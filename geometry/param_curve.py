import sympy as sym
import sympy.physics.vector as spv
import numpy as np

# 曲线的参数化的symbol
phi = sym.Symbol('phi')
# 三维坐标系！！
N = spv.ReferenceFrame('N')

def HNCoef(Par_v:spv.vector.Vector, phi, frame, LapHn=False):
    '''
        计算平面曲线的曲率: X=(f,g)
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

class Param1dCurve:
    '''
        Example of computing 
        import sympy as sym
        phi = sym.Symbol('phi')
        Param = [sym.cos(phi),(0.6*sym.cos(phi)**2+0.4)*sym.sin(phi)]
        CurveObj = Param1dcurve(Param)
        n = 21
        Coords = np.zeros((n-1,2))
        theta = np.linspace(0,2*np.pi,n)[:-1]

        Coords[:,0] = CurveObj.Param_np[0](theta)+0.05*np.random.random(n-1)
        Coords[:,1] = CurveObj.Param_np[1](theta)+0.05*np.random.random(n-1)
        theta_set = CurveObj.Get_Param_Projection(Coords)
        ProjPos = np.hstack([CurveObj.Param_np[0](theta_set),CurveObj.Param_np[1](theta_set)])

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
            # 通过粗筛选定Newton迭代的下界限，初值以及上界限
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
                    print('max iter_num has been reached!')
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
    def __init__(self,a,b,c,N_Spline,T_min,T_max,eps,is_close) -> None:
        self.a, self.b, self.c = a, b, c
        self.T_max = T_max
        self.T_min = T_min
        self.Param = [-(b-a*(sym.cos(phi)**2-1)**2)*sym.sin(phi), c*sym.cos(phi)]
        super().__init__(Param=self.Param, T_min = self.T_min, T_max=self.T_max, N_Spline=N_Spline, eps=eps, c_tag=is_close)

    def Get_Param(self, Coords):
        ## by arctan2, get values in [0,pi] (symmetric). What if point not on the curve???
        x_coord = Coords[:,0]
        z_coord = Coords[:,1]/self.c
        phi_np = np.arctan2(-x_coord/(self.b-self.a*(z_coord**2-1)**2), z_coord)
        return phi_np
    
class CliffordTorusSpline(Param1dSpline):
    '''
        (1- sqrt(x^2 + y^2))^2 + z^2 = 1/2, Dziuk 2008, Numerische Mathematik
        x = 1 - sqrt(1/2)*sin phi (computed by setting y = 0)
        z = sqrt(1/2)*cos phi 
    '''
    def __init__(self,N_Spline,T_min=0,T_max=np.pi*2,eps=1e-4,is_close=True) -> None:
        self.T_max = T_max
        self.T_min = T_min
        self.a = np.sqrt(1/2)
        self.Param = [1-self.a*sym.sin(phi), self.a*sym.cos(phi)]
        super().__init__(Param=self.Param, T_min = self.T_min, T_max=self.T_max, N_Spline=N_Spline, eps=eps, c_tag=is_close)

    def Get_Param(self, Coords):
        ## by arctan2, get values in [0,pi] (symmetric). What if point not on the curve???
        x_coord = Coords[:,0]
        z_coord = Coords[:,1]
        phi_np = np.arctan2( 1 - x_coord, z_coord)
        return phi_np

class FlowerCurve(Param1dCurve):
    def __init__(self, T_max=2 * np.pi, theta_dis=None, a=0.65, b=7) -> None:
        Param = [(a*sym.sin(b*phi)+1)*sym.cos(phi), (a*sym.sin(b*phi)+1)*sym.sin(phi)]
        super().__init__(Param = Param, T_max = T_max, theta_dis = theta_dis)