"""
===========================================================
2d Meshes generatad by netgen.geom2d packages
-----------------------------------------------------------
- Ellipse
- Circle
===========================================================
"""

from ngsolve import Mesh
import numpy as np
from netgen.geom2d import SplineGeometry

def Ellipse(msize,bndname='bnd'):
    geo = SplineGeometry()
    ctr_point = [(2,0), (2,1), (0,1), (-2,1), (-2,0), 
                (-2,-1), (0,-1), (2,-1)]
    ctr_point = [(0.25*x, 0.25*y) for x,y in ctr_point]
    p1,p2,p3,p4,p5,p6,p7,p8 = [ geo.AppendPoint(x,y) for x,y in ctr_point]
    geo.Append (["spline3", p1, p2, p3],bc=bndname,
                     leftdomain=1, rightdomain=0)
    geo.Append (["spline3", p3, p4, p5],bc=bndname,
                     leftdomain=1, rightdomain=0)
    geo.Append (["spline3", p5, p6, p7],bc=bndname,
                     leftdomain=1, rightdomain=0)
    geo.Append (["spline3", p7, p8, p1],bc=bndname,
                     leftdomain=1, rightdomain=0)
    # geo.SetMaterial (1, "outer")
    mesh = Mesh(geo.GenerateMesh(maxh=msize))
    return mesh, ctr_point

def Circle(msize,bndname='bnd'):
    geo = SplineGeometry()
    geo.AddCircle(c=(0,0),
                r=1,
                bc=bndname,
                leftdomain=1,
                rightdomain=0)
    geo.SetMaterial(1, "inner")
    mesh = Mesh(geo.GenerateMesh(maxh=msize))
    return mesh