'''
    Reference: Documents/2021/Surface%20PDE/Surface_PDE/Mesh2d/ModifiedMesh2d.ipynb
'''
import netgen.meshing as ngm
from netgen.meshing import *
from netgen.csg import *
from ngsolve import *
import numpy as np
import math
import open3d as o3d
from scipy import optimize
from scipy.sparse import coo_matrix
from Package_MyCode import *
from Package_Angenent import AngenentTorusProfile

class Dumbbell:
    dptol = .001    # 单次迭代位移小于便停止
    ttol = .1
    def __init__(self, coef=(0.6,0.4), h0=0.05, Fscale=1.05, deltat=.2) -> None:
        '''
        Fscale: Control Internal Pressure,
        deltat: Time Step in Euler,
        '''
        self.coef, self.h0, self.Fscale, self.deltat = coef, h0, Fscale, deltat
        self.geps = 0.001*h0
        self.deps = np.sqrt(1e-16)*h0
        self.mymesh = []
        self.ParamDumbbell(*coef)

    def MySphere(self,center=(0,0,0),rad=1):
        '''A Surface Mesh with a 3d Sphere'''
        print(center)
        geo          = CSGeometry()
        sphere       = Sphere(Pnt(center), rad).bc("sphere")
        geo.Add(sphere)
        mymesh = Mesh(geo.GenerateMesh(maxh=0.08, perfstepsend=MeshingStep.MESHSURFACE))
        return mymesh

    def ParamDumbbell(self,p1=0.6,p2=0.4):
        '''
            球坐标下的参数化Dumbell曲面，简单的ngmesh.copy不行？
            Function: Initialize self.mymesh
        '''
        myngmesh = ngm.Mesh()
        fd_outside = myngmesh.Add (FaceDescriptor(bc=1,domin=0,surfnr=1))
        pmap1 = { }
        SphereMesh = self.MySphere()
        pmall = {vert: vert.point for vert in SphereMesh.vertices}
        for e in SphereMesh.Elements(BND):
            for v in e.vertices:
                if (v not in pmap1):
                    tmpx = pmall[v][0]
                    tmpy = pmall[v][1]
                    tmpz = pmall[v][2]
                    tmpr = np.sqrt(tmpy**2+tmpz**2)
                    exactr = (p1*tmpx**2+p2)*sqrt(1-tmpx**2)
                    if tmpr != 0:
                        newpos = [tmpx, tmpy*exactr/tmpr, tmpz*exactr/tmpr]
                    else:
                        print(tmpx)
                        newpos = [1, tmpy, tmpz]
                    # Tuple
                    newcoord = tuple(newpos)
                    pmap1[v] = myngmesh.Add (MeshPoint(newcoord))
        for e in SphereMesh.Elements(BND):
            myngmesh.Add (Element2D (fd_outside, [pmap1[v] for v in e.vertices]))
        self.mymesh = Mesh(myngmesh)

    def getangle(self,v_coord):
        [x,y,z] = np.array(v_coord)
        phi = math.acos(x)
        theta = math.atan2(z,y)
        return phi,theta
        
    def GenerateNormal(self,phi,theta):
        '''给了1个点的球坐标,输出Dumbbell上对应点的法向量'''
        tx = -np.sin(phi)
        ty = -2*self.coef[0]*np.cos(phi)*np.sin(phi)**2*np.cos(theta) +\
            (self.coef[0]*np.cos(phi)**2+self.coef[1])*np.cos(theta)*np.cos(phi)
        tz = -2*self.coef[0]*np.cos(phi)*np.sin(phi)**2*np.sin(theta) +\
            (self.coef[0]*np.cos(phi)**2+self.coef[1])*np.sin(theta)*np.cos(phi)
        
        sx,sy,sz = 0,-np.sin(theta),np.cos(theta)

        nx = ty*sz - tz*sy
        ny = tz*sx - tx*sz
        nz = tx*sy - ty*sx

        length = sqrt(nx*nx + ny*ny + nz*nz)
        nx /= length
        ny /= length
        nz /= length
        return nx, ny, nz

    def GetCoordNormal(self,pxy):
        '''给了n个点的坐标:nx3,输出Dumbbell上对应点的法向量'''
        nxy = np.zeros(pxy.shape)

        iangle_set = []
        jangle_set = []
        for i in range(pxy.shape[0]):
            [iangle,jangle] = self.getangle(pxy[i])
            iangle_set.append(iangle)
            jangle_set.append(jangle)
        for i in range(pxy.shape[0]):
            nxy[i] = self.GenerateNormal(iangle_set[i],jangle_set[i])
        return nxy

    def GenerateTri(self,pxy,nxy):
        '''
        Generate PointCloud from Normal:
        输入节点坐标和法向量，
        输出节点和三角元的构造
        '''
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pxy)
        pcd.normals = o3d.utility.Vector3dVector(nxy)
        # VisualizeCloud(pcd)

        radii = [0.01, 0.1, 0.2,0.3,0.4, 0.5]
        rec_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd, o3d.utility.DoubleVector(radii))
        # o3d.visualization.draw_geometries([pcd, rec_mesh])
        npVec = np.asarray(rec_mesh.vertices)
        npTri = np.asarray(rec_mesh.triangles)
        return npVec, npTri

    def VisualNgsolve(self,npVec,npTri):
        myngmesh = ngm.Mesh(dim=3)
        fd_outside = myngmesh.Add (FaceDescriptor(bc=1,domin=0,surfnr=1))
        pmall = [myngmesh.Add(MeshPoint(Pnt(ver))) for ver in npVec]
        for ii in range(npTri.shape[0]):
            myngmesh.Add (Element2D (fd_outside, [pmall[ind] for ind in npTri[ii]]))
        m1 = Mesh(myngmesh)
        return m1

    def fd(self,x,y,z):
        '''Level Set Function'''
        ## x,y,z: Nx1 ndarray
        return (self.coef[0]*x**2+self.coef[1])**2*(x**2-1)+y**2+z**2
    def fdgrad(self,x,y,z):
        gradx = 2*(self.coef[0]*x**2+self.coef[1])*1.2*x*(x**2-1) \
                    + 2*x*(self.coef[0]*x**2+self.coef[1])**2
        grady = 2*y
        gradz = 2*z
        return gradx, grady, gradz

    def hd(self,p):
        '''Weight distribution of the length of bars'''
        res = []
        for i in range(p.shape[0]):
            if abs(p[i][0])<0.4:
                res.append(0.4)
            elif abs(p[i][0])<0.5:
                res.append(0.45)
            elif abs(p[i][0])<0.6:
                res.append(0.5)
            else:
                res.append(0.5)
        return np.array(res)
    # hd = lambda p: np.array([1 for i in range(p.shape[0])])

    def SingleMove(self,t,p,nxy):
        bars=np.concatenate((t[:,[0,1]], t[:,[0,2]], t[:,[1,2]]), axis = 0)
        bars=np.unique(np.sort(bars),axis=0)

        barvec=p[bars[:,0],:]-p[bars[:,1],:]
        ## Length of bars
        L=(np.sqrt((barvec**2).sum(axis=1)))
        L = L.reshape(-1,1)

        N=p.shape[0]
        ## center of bar
        hbars = self.hd((p[bars[:,0],:]+p[bars[:,1],:])/2)

        L0 = hbars*self.Fscale*np.sqrt((L**2).sum()/(hbars**2).sum())
        L0 = L0.reshape(len(L0),1)

        ## L0 = Desired lengths
        ## Bar forces (scalars)
        ## Bar forces (x,y,z-components)
        # F=np.array([[max((L0-L)[ii][0],0)] for ii in range(len(L0))])
        F=np.array([[(L0-L)[ii][0]] for ii in range(len(L0))])
        Fvec=np.kron(F/L,[1,1,1])*barvec
        row = bars[:,[0,0,0,1,1,1]].flatten()
        column = np.kron(np.ones(F.shape),[0,1,2,0,1,2]).flatten()
        val = np.concatenate((Fvec,-Fvec),axis=1).flatten()
        Ftot = coo_matrix((val,(row,column)),shape=(N,3)).todense()
        Ftot = np.array(Ftot)
        ## Projection to Tangent Part
        for i,F in enumerate(Ftot):
            Ftot[i] = F-(F*nxy[i]).sum()*nxy[i]

        p=p+self.deltat*Ftot
        p=np.array(p)
        d=self.fd(p[:,0],p[:,1],p[:,2])
        ix=d>0
        dgradx=(self.fd(p[ix,0]+self.deps,p[ix,1],p[ix,2])-d[ix])/self.deps ### Numerical 
        dgrady=(self.fd(p[ix,0],p[ix,1]+self.deps,p[ix,2])-d[ix])/self.deps ### gradient 
        dgradz=(self.fd(p[ix,0],p[ix,1],p[ix,2]+self.deps)-d[ix])/self.deps ### gradient 
        ### Project back to boundary
        p[ix,:]=p[ix,:]-np.concatenate(((d[ix]*dgradx).reshape(-1,1),(d[ix]*dgrady).reshape(-1,1),(d[ix]*dgradz).reshape(-1,1)),axis=1)
        return p

    def GetBack(self,p):
        ## input: a coordinate array
        gradx, grady, gradz = self.fdgrad(*p)
        gradv = np.array([gradx,grady,gradz])
        func = lambda lam: self.fd(*(p+lam*gradv))
        lam0 = optimize.newton(func,0)
        p0 = p+lam0*gradv
        d = self.fd(*p0)
        try:
            assert d<1e-10
            return p0
        except:
            return p

    def ModifyMesh(self,N=200):
        pxy = np.array([list(ver.point) for ver in self.mymesh.vertices])
        for ii in range(N):
            if ii%20==0:
                print(ii)
                for jj in range(len(pxy)):
                    pxy[jj] = self.GetBack(pxy[jj])
            nxy = self.GetCoordNormal(pxy)
            npVec,npTri = self.GenerateTri(pxy,nxy)
            self.mymesh = self.VisualNgsolve(npVec,npTri)
            Redraw(blocking=True)
            pxy = self.SingleMove(npTri,npVec,nxy)
            if ii==N-2:
                for jj in range(len(pxy)):
                    pxy[jj] = self.GetBack(pxy[jj])
        return self.mymesh

class AngenentTorus():
    dptol = .001    # 单次迭代位移小于便停止
    ttol = .1
    def __init__(self, h0=0.05, Fscale=1.05, deltat=.2) -> None:
        '''
        Fscale: Control Internal Pressure,
        deltat: Time Step in Euler,
        '''
        self.h0, self.Fscale, self.deltat = h0, Fscale, deltat
        self.geps = 0.001*h0
        self.deps = np.sqrt(1e-16)*h0
        self.mymesh = []
        self.Angenent()

    def Angenent(self,maxh = 0.1):
        '''
            Generate the angenent torus from reparameterizing Canonical Torus: R = 1, r = 0.7
        '''
        AngTorus = AngenentTorusProfile()
        spline = SplineCurve2d() # create a 2d spline
        R = 1                    # define the major radius
        r = 0.7                  # define the minor radius
        eps = 1e-8
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
        Torus_Pos = np.array([list(v.point) for v in Torus_Canon.vertices])
        phi = np.arctan2(Torus_Pos[:,1],Torus_Pos[:,0])
        z = Torus_Pos[:,2]
        rho = np.sqrt(Torus_Pos[:,0]**2+Torus_Pos[:,1]**2)
        theta = np.arctan2(z,R-rho)
        res = theta[theta<0]
        theta[theta<0] = res+2*np.pi
        Coord3d = AngTorus.Generate_Coord(theta,phi)

        ATmesh = Torus_Canon.ngmesh.Copy()
        for i,vec in enumerate(ATmesh.Points()):
            vec[0] = Coord3d[i,0]
            vec[1] = Coord3d[i,1]
            vec[2] = Coord3d[i,2]
        self.mymesh = Mesh(ATmesh)

