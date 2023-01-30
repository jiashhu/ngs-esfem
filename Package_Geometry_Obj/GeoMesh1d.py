import netgen.meshing as ngm
from ngsolve import *
import numpy as np
from Package_MyNgFunc import SurfacehInterp
from Package_FEM_B import FEM1d
from SM_util import Param1dCurve
import sympy as sym

phi = sym.Symbol('phi')

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

class CircleMesh():
    '''
        Create mesh of a circle with 
        radius: rad
        number of vertices: n

        Example:
        MeshObj = CircleMesh(1,4)
        MeshObj.MeshHo(order = 2)
        mymesh = MeshObj.mesh
    '''
    def __init__(self,rad,n,cx=0,cy=0):
        self.r = rad
        self.nv = n
        self.r_pos = np.array([cx,cy])

    def ProjMap(self,x:np.ndarray):
        '''
            projection from neighborhood of surface onto surface
        '''
        try:
            assert(x.shape[1]==2)
        except:
            print('second dimension of x should be 2!!')
        x = x - self.r_pos
        if len(x.shape) == 1:
            y = self.r_pos + self.r*x/np.linalg.norm(x)
        else: 
            y = self.r_pos + self.r*x/np.linalg.norm(x,axis=1)[:,None]
        return y

    def IsoProjQP(self,x_ref,X_El,order):
        '''
            X_El = [[X_0], [X_1]]: points of dofs of the isoparametric element on line
            x_ref in [0,1], reference coordinate of quadrature nodes

            Construct isoparametric interplation a_k as high order functions of line segment, by mapping DOFs of ho using projection a. Then compute the image of quadrature nodes under a_k
        '''
        XDof = self.femobj.XDof
        X_Dofs = np.array([
            (1-coef)*X_El[0,:] + coef*X_El[1,:] for coef in XDof
        ])
        Proj_X_Dofs = self.ProjMap(X_Dofs)
        # compute the image of quadrature points x under ak = \sum phi_i(x) PjX_i
        Val_QP = np.array([
            np.array([f(xval[0]) for f in self.BRfunc]) @ Proj_X_Dofs for xval in x_ref
        ])
        return Val_QP

    def MeshPoly(self):
        self.vertices = np.array([[self.r_pos[0]+np.cos(theta)*self.r, self.r_pos[1]+np.sin(theta)*self.r] 
                            for theta in np.linspace(0,2*np.pi,self.nv+1)[:-1]])
        self.mesh = Mesh1dFromPoints(self.vertices,dim=1,adim=2)
        mod_nv = lambda n: n-self.nv if n>=self.nv else n
        self.ElVerInd = np.array([[ii, mod_nv(ii+1)] for ii in range(self.nv)])

    def MeshHo(self,order):
        '''
            Generate high order surface by deformation on the polygon
            1. Generate polygon mesh
            2. Set up surface quadrature interpolation space, get the quad rule (number for quad points shall be larger than ndof of isoparametric)
        '''
        self.MeshPoly()
        self.order = order
        mymesh = self.mesh
        Coords = self.vertices
        Coords_circ = np.vstack([Coords,Coords[0,:]])
        self.hofes = H1(mymesh,order=order)
        self.hofesV = VectorH1(mymesh,order=order)
        self.ini_disp = GridFunction(self.hofesV)
        self.femobj = FEM1d(order=order)
        self.BRfunc = self.femobj.BRFunc
        # compute the initial displacement and deform the boundary by interpolation of a
        self.SInterp = SurfacehInterp(mymesh,order=order)
        if order > 1:
            # coords on the polygon
            Coord_interp_quad = self.SInterp.GetCoordsQuad()
            # quadrature nodes in [0,1] -- 1d case
            self.quad_ws = np.array(self.SInterp.irs[SEGM].points)
            # compute the image of quadrature nodes after ak (isoparametric mapping)
            trans_quad_nodes = None
            # each straight element
            for ii in range(len(Coords_circ)):
                if ii > 0:
                    xstarts = Coords_circ[ii]
                    xends = Coords_circ[ii-1]
                    if trans_quad_nodes is None:
                        trans_quad_nodes = self.IsoProjQP(self.quad_ws,np.vstack([xstarts,xends]),order=order)
                    else:
                        trans_quad_nodes = np.vstack(
                            [trans_quad_nodes,
                            self.IsoProjQP(self.quad_ws,np.vstack([xstarts,xends]),order=order)]
                        )
            disp_np = trans_quad_nodes-Coord_interp_quad
            self.ini_disp.vec.data = BaseVector(np.append(self.SInterp.Return2L2(disp_np[:,0]),
                                                    self.SInterp.Return2L2(disp_np[:,1])))

    def InterpHo(self, func):
        '''
            The isoparametric surface interpolates the exact one, for high order interpolation, from values of function on the interpolated points (deformed dofs of straight line)

            First, we get the position of the deformed dofs (trans_quad_nodes), then get their values by function, create the high order polynomial then obtain values on the quadrature nodes. Finally mapping back to a high order fes function.
        '''
        if self.order > 1:
            Val_QP = None
            for Elver in self.ElVerInd:
                X_El = self.vertices[[Elver[1],Elver[0]]] ## element vertices match the reference mapping rule
                XDof = self.femobj.XDof
                X_Dofs = np.array([
                    (1-coef)*X_El[0,:] + coef*X_El[1,:] for coef in XDof
                ])
                Proj_X_Dofs = self.ProjMap(X_Dofs)

                fx_Dofs = np.array([func(Proj_X_Dofs[ii,:]) for ii in range(Proj_X_Dofs.shape[0])])
                Val_QP_El = np.array([
                    np.array([f(xval[0]) for f in self.BRfunc]) @ fx_Dofs for xval in self.quad_ws
                ])
                if Val_QP is None:
                    Val_QP = Val_QP_El.copy()
                else:
                    Val_QP = np.hstack([Val_QP,Val_QP_El])
            hoGFnp = self.SInterp.Return2L2(Val_QP)
        elif self.order == 1:
            hoGFnp = np.array([func(xx) for xx in self.vertices])
        return hoGFnp
        
    def GetX_IsoDof(self):
        '''
            Get the position of dofs of the isoparametric surface from the deformed surface mesh by SInterp
        '''
        # Get poisition of quadrature nodes on the deformed surface
        Coord_interp_quad = self.SInterp.GetCoordsQuad()
        
class EllipseMesh(CircleMesh):
    '''
        Create high order isoparametric mesh of an ellipse with 
        radius: rL (x-axis) and rs (y-axis)
        number of vertices: n

        Example:
        MeshObj = EllipseMesh(rL = 2, rs = 1, 4)
        MeshObj.MeshHo(order = 2)
        mymesh = MeshObj.mesh
    '''
    def __init__(self, cx, cy, rL, rs, n):
        self.cx = cx
        self.cy = cy
        self.rL = rL
        self.rs = rs
        self.nv = n
        Param = [self.cx + self.rL*sym.cos(phi), self.cy + self.rs**sym.sin(phi)]
        self.CurveObj = Param1dCurve(Param)

    def MeshPoly(self):
        theta = np.linspace(0,2*np.pi,self.nv+1)[:-1]
        Param_np = self.CurveObj.Param_np
        self.vertices = np.hstack([Param_np[0](theta), Param_np[1](theta)])
        self.mesh = Mesh1dFromPoints(self.vertices,dim=1,adim=2)

    def ProjMap(self,x:np.ndarray):
        '''
            projection from neighborhood of surface onto surface, not normal
        '''
        try:
            assert(x.shape[1]==2)
        except:
            print('second dimension of x should be 2!!')
        theta_set = self.CurveObj.Get_Param_Projection(x)
        ProjPos = np.hstack([self.CurveObj.Param_np[0](theta_set),self.CurveObj.Param_np[1](theta_set)])
        return ProjPos


