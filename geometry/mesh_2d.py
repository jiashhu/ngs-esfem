from netgen.occ import Face, Wire, OCCGeometry, SplineApproximation, WorkPlane
from netgen.meshing import MeshingParameters
from .levelset_curve import level_set_quadfoil
from ngsolve import Mesh
import numpy as np

def quadfoil(mesh_size):
    '''
        Quat in a Circle, centered at origin with radius r_circ
    '''
    pnts = [level_set_quadfoil(z) for z in np.linspace(0,2*np.pi,100)]
    spline = SplineApproximation(pnts, tol=1e-4)
    f = Face(Wire([spline]))
    geo = OCCGeometry(f)
    mp = MeshingParameters(maxh=mesh_size,curvaturesafety=3,grading=0.2)   # 控制最大单元大小
    mesh = Mesh(geo.GenerateMesh(mp))
    return mesh
    
def quadfoil_in_circ(mesh_size,r_circ,bnd_in='bnd_in',bnd_out='bnd_out'):
    '''
        Quat in a Circle, centered at origin with radius r_circ
    '''
    pnts = [level_set_quadfoil(z) for z in np.linspace(0,2*np.pi,100)]
    spline = SplineApproximation(pnts, tol=1e-4)
    f = Face(Wire([spline]))
    geo = OCCGeometry(f)
    
    wp = WorkPlane().Circle(0,0,r_circ)
    face0 = wp.Face()
    face0.edges[0].name = bnd_out
    f = Face(Wire(spline))
    f.edges[0].name = bnd_in
    geo = OCCGeometry(face0-f,dim=2)
    mp = MeshingParameters(maxh=mesh_size,curvaturesafety=3,grading=0.2)   # 控制最大单元大小
    mesh = Mesh(geo.GenerateMesh(mp))
    return mesh

