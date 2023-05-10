'''
    Test high order convergence of surface PDE on circle with isoparametric approximation 
    
    For order = 1, convergence rate in H1 is 2, super-convergence
    For order = 2,3, convergence rate in H1 is little higher than 2,3
'''

from Package_Geometry_Obj import CircleMesh, EllipseMesh
from ngsolve import *
import numpy as np

# for circle with radius 1, ∆X = -X, the elliptic PDE writes -∆(x+2y) = (x+2y)
order = 3
errs = []
for nk in range(3,6):
    meshobj = CircleMesh(rad = 1, n = 2**(2+nk))
    meshobj.MeshHo(order = order)

    mymesh = meshobj.mesh
    fes = meshobj.hofes
    fesV = meshobj.hofesV
    ini_disp = meshobj.ini_disp

    mymesh.SetDeformation(ini_disp)

    Sol = GridFunction(fes)
    Err = GridFunction(fes)
    Sol_Ref = GridFunction(fes)
    Sol_Ref.Set(x+2*y,definedon=mymesh.Boundaries('.*'))

    u,v = fes.TnT()
    Lhs = BilinearForm(fes)
    Lhs += InnerProduct(grad(u).Trace(),grad(v).Trace())*ds+u*v*ds
    Rhs = LinearForm(fes)
    Rhs += 2*(x+2*y)*v*ds

    Lhs.Assemble()
    Rhs.Assemble()
    Sol.vec.data = Lhs.mat.Inverse(inverse='pardiso')*Rhs.vec
    Err.vec.data = Sol.vec - Sol_Ref.vec

    H1errfun = lambda xfun: sqrt(Integrate(InnerProduct(grad(xfun).Trace(),grad(xfun).Trace())
                                            +InnerProduct((xfun),(xfun)), mymesh, 
                                            element_wise=False, definedon=mymesh.Boundaries('.*')))
    H1Err = H1errfun(Err)
    errs.append(H1Err)
print('H1 errs are {}'.format(errs))
print('Convergence Rates for order {} is {}'.format(
    order, [np.log(errs[ii-1]/errs[ii])/np.log(2) for ii in range(len(errs)) if ii > 0]
))



