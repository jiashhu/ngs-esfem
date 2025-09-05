import sympy as sym
import sympy.physics.vector as spv
import numpy as np
import netgen.meshing as ngm
from ngsolve import *
from netgen.meshing import MeshingParameters
from geometry import *
from .param_curve import Param1dSpline, Param1dCurve, phi, N, RBCSpline, DumbbellSpline
from .discrete_mesh import DiscreteMesh
from netgen.occ import SplineApproximation, Pnt, Axis, Face, Wire, Segment, Revolve, OCCGeometry, Z, X, Y
from netgen.csg import Pnt,SplineCurve2d,CSGeometry,Revolution,Sphere

# 后面的 axis中用到的Pnt是 netgen.csg 中的Pnt
def Mesh2dRotSpline(Spline_Obj:Param1dSpline,axis_opt,c_tag,maxh,order):
    if axis_opt == 'x':
        axis_0, axis_1 = Pnt(0,0,0), Pnt(1,0,0) 
    elif axis_opt == 'z':
        axis_0, axis_1 = Pnt(0,0,0), Pnt(0,0,1)
    spline = SplineCurve2d() # create a 2d spline
    # define the control points -- anti clockwise
    ctrbase, ctr = Spline_Obj.ctrbase, Spline_Obj.ctr
    if c_tag == False:
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
        pnts.append(tuple(ctrbase[ii]))
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
    geo = CSGeometry()
    geo.Add (rev.col([1,0,0]))
    mesh = Mesh(geo.GenerateMesh(maxh = maxh, perfstepsend=ngm.MeshingStep.MESHSURFACE))
    mesh.Curve(order)
    return mesh
    
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
    
class Param2dRot():
    '''
        Rotational 2d surface by profile curve in x-z plane (f(phi), g(phi)).
        Around x axis: (f(phi), g(phi)cos theta, g(phi)sin theta).
        Around z axis: (f(phi)cos theta, f(phi)sin theta, g(phi))
    '''
    def __init__(self,spline_obj:Param1dSpline,axis_opt,c_tag) -> None:
        self.spline_profile_obj = spline_obj
        self.rot_axis = axis_opt
        self.is_closed = c_tag

        f, g = self.spline_profile_obj.Xparam, self.spline_profile_obj.Yparam
        df, dg = f.diff(phi), g.diff(phi)
        ddf, ddg = df.diff(phi), dg.diff(phi)
        # normal of profile curve
        self.nhat_1 = dg/sym.sqrt(df**2+dg**2)
        self.nhat_2 = -df/sym.sqrt(df**2+dg**2)
        if self.rot_axis == 'x':
            self.curvature_sym = -((dg*ddf-ddg*df)/(df**2+dg**2)**(3/2)+df/(g*sym.sqrt(df**2+dg**2)))
            self.curvature_np = sym.lambdify(phi, self.curvature_sym)
            # metric (orthogonal)
            self.gij = [df**2+dg**2, g**2]
            self.invgij = [1/(df**2+dg**2), 1/g**2]
            self.detg = (df**2+dg**2)*g**2
        elif self.rot_axis == 'z':
            # wait for check
            self.curvature_sym = -((df*ddg-ddg*dg)/(df**2+dg**2)**(3/2)+dg/(f*sym.sqrt(df**2+dg**2)))
            self.curvature_np = sym.lambdify(phi, self.curvature_sym)

    def laplace_theta_indep(self,fexpr):
        # surface Laplace of function independent of theta
        Lapf = 1/sym.sqrt(self.detg)*(
            sym.sqrt(self.detg)*self.invgij[0]*fexpr.diff(phi)
        ).diff(phi)
        return Lapf

    def laplace_Hn(self):
        self.LapH_func, self.Lapnhat1_func, self.Lapnhat2_func = [
            sym.lambdify(phi, self.laplace_theta_indep(fexpr)) 
            for fexpr in [self.curvature_sym, self.nhat_1, self.nhat_2]
        ]
        self.g22nhat2 = sym.lambdify(phi,self.invgij[1]*self.nhat_2)
        n_prim_square = (self.nhat_1.diff(phi))**2+(self.nhat_2.diff(phi))**2
        self.A2_func  = sym.lambdify(phi, self.invgij[0]*n_prim_square + self.invgij[1]*self.nhat_2**2)
    
    def get_laplace_H_N(self,phi:np.ndarray, theta:np.ndarray):
        n = len(phi)
        assert len(phi) == len(theta)
        LapH, Lapn = np.zeros((n,1)), np.zeros((n,3))
        if self.rot_axis == 'x':
            LapH[:,0] = self.LapH_func(phi)
            Lapn[:,0] = self.Lapnhat1_func(phi)
            Lapn[:,1] = self.Lapnhat2_func(phi)*np.cos(theta) - self.g22nhat2(phi)*np.cos(theta)
            Lapn[:,2] = self.Lapnhat2_func(phi)*np.sin(theta) - self.g22nhat2(phi)*np.sin(theta)
            A_Frob = self.A2_func(phi)
        return LapH, Lapn, A_Frob.reshape(-1,1)

    def get_profile_coord(self,Coords3d):
        Coord_Profile = np.zeros((Coords3d.shape[0],2))
        if self.rot_axis == 'x':
            # x-z curve rotate around x axis (f(phi), g(phi)cos theta, g(phi)sin theta)
            Coord_Profile[:,0] = Coords3d[:,0]
            Coord_Profile[:,1] = np.linalg.norm(Coords3d[:,1:],axis=1)
        elif self.rot_axis == 'z':
            # x-z curve rotate around z axis (f(phi)cos theta, f(phi)sin theta, g(phi))
            Coord_Profile[:,0] = np.linalg.norm(Coords3d[:,:-1],axis=1)
            Coord_Profile[:,1] = Coords3d[:,2]
        return Coord_Profile

    def get_param(self,coords_3d,near_opt=False):
        '''
            Determin Parameter of 3d Coords by Rotation axis and Profile Curve
        '''
        Coord_Profile = self.get_profile_coord(coords_3d)
        if self.rot_axis == 'x':
            # x-z curve rotate around x axis (f(phi), g(phi)cos theta, g(phi)sin theta)
            theta_np = np.arctan2(coords_3d[:,2],coords_3d[:,1])
        elif self.rot_axis == 'z':
            # x-z curve rotate around z axis (f(phi)cos theta, f(phi)sin theta, g(phi))
            theta_np = np.arctan2(coords_3d[:,1],coords_3d[:,0])
        if not near_opt:
            # point exactly on the profile curve
            phi_np = self.spline_profile_obj.get_param(Coord_Profile)
        else:
            # point not on the profile curve
            phi_np = self.spline_profile_obj.get_param_projection(Coord_Profile)
        return phi_np, theta_np

    def get_pos_norm(self,phi:np.ndarray, theta:np.ndarray):
        n = len(phi)
        assert len(phi) == len(theta)
        Pos_Profile, Norm_Profile = self.spline_profile_obj.Generate_Pos_Norm(phi)
        Pset, Normset = np.zeros((n,3)), np.zeros((n,3))
        if self.rot_axis == 'x':
            Pset[:,0] = Pos_Profile[:,0]
            Pset[:,1] = Pos_Profile[:,1]*np.cos(theta)
            Pset[:,2] = Pos_Profile[:,1]*np.sin(theta)
            Normset[:,0] = Norm_Profile[:,0]
            Normset[:,1] = Norm_Profile[:,1]*np.cos(theta)
            Normset[:,2] = Norm_Profile[:,1]*np.sin(theta)
        elif self.rot_axis == 'z':
            print('Rotation around z-s Normal is not available yet!')
        return Pset, Normset
    
    def Generate_Mesh(self,maxh,order,RN=0,Local_h=None):
        if self.rot_axis == 'x':
            axis_0, axis_1 = Pnt(0,0,0), Pnt(1,0,0)
        elif self.rot_axis == 'z':
            axis_0, axis_1 = Pnt(0,0,0), Pnt(0,0,1)
        spline = SplineCurve2d() # create a 2d spline
        # define the control points -- anti clockwise
        ctrbase, ctr = self.spline_profile_obj.ctrbase, self.spline_profile_obj.ctr
        if self.is_closed == False:
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
        
def Mesh2dDumbbell(maxh,order,a=0.6,b=0.4,N_Spline=7,eps=2e-7):
    '''
        Profile curve on the x-z plane, rotate around the x axis, 由于参数去除最后一个，NumSpline为奇数时对称
    '''
    T_max = np.pi
    c_tag = False
    Spline_Obj = DumbbellSpline(a=a,b=b,N_Spline=N_Spline,T_max=T_max,eps=eps,c_tag=c_tag)
    mesh = Mesh2dRotSpline(Spline_Obj,axis_opt='x',c_tag=False,maxh=maxh,order=order)
    return mesh, Spline_Obj

def GetRotMesh(CurveFunc,msize,T_min=-np.pi/2,T_max=np.pi/2,axis=Z,is_close=False,n=100):
    '''
        Generate Rotational Mesh by revolving a 2d curve (CurveFunc) around an axis (axis)
        CurveFunc: function of phi, return (x(phi), 0, z(phi)) for example
        axis: axis of rotation, default z-axis: curve on x-z plane around z-axis
        is_close: whether the curve is closed (first point = last point)
        n: number of points on the curve
        T_min, T_max: parameter range of the curve
        msize: mesh size
        return: mesh 
    '''
    pnts = [CurveFunc(phi) for phi in np.linspace(T_min,T_max,n)]
    spline = SplineApproximation(pnts, tol=1e-4)
    f = Face(Wire([spline]))
    torus = f.Revolve(Axis((0,0,0), axis), 360)
    mesh = Mesh(OCCGeometry(torus).GenerateMesh(maxh=msize,
                perfstepsend=ngm.MeshingStep.MESHSURFACE))
    return mesh

# MyRBC_Obj = RBC_Rot_Obj(a=0.4,b=1,c=2,N_Spline=9)
# MyRBC_Obj.Generate_Mesh(maxh=0.2,order=1)
# mesh = MyRBC_Obj.mesh
# print('good')