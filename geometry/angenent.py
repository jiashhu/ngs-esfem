import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
from scipy import interpolate
import sympy as sym
import sympy.physics.vector as spv
try:
    import netgen.meshing as ngm
    from netgen.csg import *
    from ngsolve import *
except:
    print("No packages called netgen and ngsolve")

def hit_ground1(t, y):
    if t==0:
        return 1
    else:
        return y[1]

ground_event = lambda t,x: hit_ground1(t,x)
ground_event.terminal = True
ground_event.direction = 1
phi = sym.Symbol('phi')
N = spv.ReferenceFrame('N')

class AngenentTorusProfile():
    '''
        Profile curve of the angenent torus: generation and find the nearest point
        Start from the shooting paramter: tguess, 
    '''
    
    def __init__(self,max_step=1e-4) -> None:
        tguess = 0.43712396709646023085227284
        self.max_step = max_step
        sol = solve_ivp(self.F, y0=[tguess,0,0],t_span=(0,10), events=ground_event, max_step=self.max_step)
        print('final time t = {}, x is {} and y is {}, alpha is {} '.format(sol.t[-1], (sol.y[0][-1]), (sol.y[1][-1]), sol.y[2][-1]))
        ## t -> y[0], y[1] linear interpolation
        self.rhot = interpolate.interp1d(sol.t,sol.y[0])
        self.zt = interpolate.interp1d(sol.t,sol.y[1])
        self.alphat = interpolate.interp1d(sol.t,sol.y[2])
        self.Tmax = sol.t[-1]
        self.pos2d = np.zeros((len(sol.t),2))
        self.pos2d[:,0] = sol.y[0]
        self.pos2d[:,1] = sol.y[1]   
        self.alpha_sample = sol.y[2]

    def NearestPoint(self,xset):
        '''
            xset: nx2 coords set 2d case
        '''
        Nearest_p = []
        Normset = []
        for coord in xset:
            ## find the nearest first
            ind = np.argmin(np.linalg.norm(coord-self.pos2d,axis=1))
            Nearest_p.append(self.pos2d[ind])
            alpha = self.alpha_sample[ind]
            Normset.append([-np.cos(alpha), np.sin(alpha)])
        Nearest_p = np.array(Nearest_p)
        Normset = np.array(Normset)
        return Nearest_p, Normset

    def F(self,t,X):
        '''
            ODE for angenent torus
        '''
        rho,z,alpha = X
        y = np.zeros(3)
        y[0] = np.sin(alpha)
        y[1] = np.cos(alpha)
        y[2] = (1/rho-rho/2)*np.cos(alpha) + z/2*np.sin(alpha)
        return y

    def Generate_Param_Profile(self,sparam):
        '''
            Input: theta: ndarray for parametrization of circle, 
            Output: ndarray for position of points on profile of Angenent Torus.
            rho' = sin alpha, z' = cos alpha.
        '''        
        newrho = self.rhot(sparam)
        newz = self.zt(sparam)
        newalpha = self.alphat(sparam)
        return newrho, newz, newalpha

    def Generate_Coord(self,phi,theta):
        '''
            Input: Rotational coordinate,
            phi: param for profile curve, 
            theta: rotational angle,
            sscaled = t for the shooting method.
            Output: Coordinates, Normals and Mean Curvature
        '''
        Coord3d = np.zeros((len(phi),3))
        sscaled = phi/(2*np.pi)*self.Tmax
        newrho, newz, newalpha = self.Generate_Param_Profile(sscaled)
        Coord3d[:,0] = newrho*np.cos(theta)
        Coord3d[:,1] = newrho*np.sin(theta)
        Coord3d[:,2] = newz
        ## (sin alpha cos theta, sin alpha sin theta, cos alpha) x (-sin theta, cos theta, 0)
        Normal3d = np.zeros(Coord3d.shape)
        Normal3d[:,0] = - np.cos(newalpha)*np.cos(theta)
        Normal3d[:,1] = - np.cos(newalpha)*np.sin(theta)
        Normal3d[:,2] = np.sin(newalpha)
        MC1d = np.zeros(Coord3d.shape[0])
        for i in range(Coord3d.shape[0]):
            MC1d[i] = -1/2*(Normal3d[i,:].dot(Coord3d[i,:]))
        return Coord3d, Normal3d, MC1d     

    def objective(self,tguess):
        T = 10
        sol = solve_ivp(self.F, y0=[tguess, 0, 0],t_span=(0,T), events=ground_event,max_step=0.01)
        err = abs(sol.y[0][-1] - tguess)
        return err

    def Findt0(self):
        t0 = fsolve(self.objective,0.3)
        return t0

class AngenentTorus():
    '''
        Generate the angenent torus (self-shrinker) with factor sqrt(1-t),
        from reparameterizing Canonical Torus: R = 1, r = 0.7
    '''
    def __init__(self, maxh = 0.1) -> None:
        self.maxh = maxh 
        self.AngProfile = AngenentTorusProfile()

    def Angenent(self,maxh = 0.1,save_path = None):
        '''
            使用从标准环面的离散曲面近似
            再通过尺度变换得到Angenent Torus的多面体近似
            适用于一阶方法的网格生成
        '''
        spline = SplineCurve2d() # create a 2d spline
        R = 1                    # define the major radius
        r = 0.7                  # define the minor radius
        eps = 1e-6
        # define the control points
        pnts = [ (0,R-r), (-r+eps,R-r+eps), (-r,R),
                (-r+eps,R+r-eps), (0,R+r), (r-eps,R+r-eps), (r,R), (r-eps,R-r+eps) ]
        # define the splines using the control points
        segs = [ (0,1,2), (2,3,4), (4,5,6), (6,7,0) ]
        # add the points and segments to the spline
        for pnt in pnts:
            spline.AddPoint (*pnt)
        for seg in segs:
            spline.AddSegment (*seg)
        # revolve the spline
        rev = Revolution ( Pnt(0,0,0), Pnt(0,0,1), spline)
        # create a geometry and add the revolved object
        geo = CSGeometry()
        geo.Add (rev.col([1,0,0]))
        Torus_Canon = Mesh(geo.GenerateMesh(maxh=maxh, optsteps2d=3, perfstepsend=ngm.MeshingStep.MESHSURFACE))
        # region: Torus coords, phi: param for profile curve, theta: rotational angle
        Torus_Pos = np.array([list(v.point) for v in Torus_Canon.vertices])
        theta = np.arctan2(Torus_Pos[:,1],Torus_Pos[:,0])
        z = Torus_Pos[:,2]
        rho = np.sqrt(Torus_Pos[:,0]**2+Torus_Pos[:,1]**2)
        phi = np.arctan2(z,R-rho)
        res = phi[phi<0]
        phi[phi<0] = res+2*np.pi
        # endregion
        Coord3d, Normal3d, MC1d = self.AngProfile.Generate_Coord(phi,theta)

        ATmesh = Torus_Canon.ngmesh.Copy()
        for i,vec in enumerate(ATmesh.Points()):
            vec[0] = Coord3d[i,0]
            vec[1] = Coord3d[i,1]
            vec[2] = Coord3d[i,2]
        mymesh = Mesh(ATmesh)
        if save_path:
            mymesh.ngmesh.Save(save_path)
        return mymesh, Normal3d, MC1d

    def Distance(self,CoordVec,ComputT):
        '''
            CoordVec: nx3 ndarray, records vertices of n points,
            Function: Computing minimal distance of the points to the AngenentTorus shrinked at ComputT
        '''
        newrho = np.sqrt(CoordVec[:,0]**2+CoordVec[:,1]**2)/np.sqrt(1-ComputT)
        newz = CoordVec[:,2]/np.sqrt(1-ComputT)
        dist = []
        myTmax = self.AngProfile.Tmax
        myrhot = self.AngProfile.rhot(np.linspace(0,myTmax,2**15))
        myzt = self.AngProfile.zt(np.linspace(0,myTmax,2**15))
        for r,z in zip(newrho,newz):
            dist.append(min( np.sqrt((r-myrhot)**2 + (z-myzt)**2)))
        min_dist = np.sqrt(1-ComputT)*max(dist)
        return min_dist

# AngTorus = AngenentTorusProfile()
# theta = np.linspace(0,2*np.pi,1024)
# newrho, newz = AngTorus.Generate_Param_Profile(theta)

# plt.figure(figsize = (10, 8))
# plt.plot(newrho, newz)
# plt.show()

