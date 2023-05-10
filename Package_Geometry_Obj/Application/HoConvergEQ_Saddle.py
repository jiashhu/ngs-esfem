'''
    Test high order convergence of surface PDE on circle with isoparametric approximation 
    
    For order = 1, convergence rate in H1 is 2, super-convergence
    For order = 2,3, convergence rate in H1 is little higher than 2,3
'''

from Package_Geometry_Obj import CircleMesh, EllipseMesh
from ngsolve import *
import numpy as np

# for circle with radius 1, ∆X = -X, the elliptic PDE writes -∆(x+2y) = (x+2y)
order = 2
errs = []
errsD = []
for nk in range(3,6):
    meshobj = CircleMesh(rad = 1/2, n = 2**(2+nk))
    meshobj.MeshHo(order = order)
    mymesh = meshobj.mesh
    fes = meshobj.hofes
    fesV = meshobj.hofesV
    ini_disp = meshobj.ini_disp
    mymesh.SetDeformation(ini_disp)

    r = sqrt(x**2+y**2)
    u = CF(((1-r)*x,(1-r)*y))
    u_np = [
        lambda x: x[0]*(1-np.linalg.norm(x)),
        lambda x: x[1]*(1-np.linalg.norm(x))
    ]
    V_Interp = GridFunction(fesV)
    V_Interp.vec.data = BaseVector(np.hstack([meshobj.InterpHo(u_np[0]), 
                                        meshobj.InterpHo(u_np[1])]))
    V_Set = GridFunction(fesV)
    V_Set.Set(u,definedon=mymesh.Boundaries('.*'))
    Err_D = GridFunction(fesV)
    Err_D.vec.data = V_Set.vec - V_Interp.vec

    n_proj = GridFunction(fesV)
    rCF = sqrt(x**2+y**2)
    nCF_lift = CF((x/rCF,y/rCF))
    DnCF_lift = CF((y**2/rCF**3, -x*y/rCF**3, -x*y/rCF**3, x**2/rCF**3),dims=(2,2))
    n_proj.vec.data = BaseVector(meshobj.SInterp.ReturnByRitzV(nCF_lift,DnCF_lift))
    # n_proj.Set(specialcf.normal(2),definedon=mymesh.Boundaries('.*'))

    fes_saddle = fes*fesV
    chi, v_trial = fes_saddle.TrialFunction()
    chit, v_test = fes_saddle.TestFunction()
    v_chi_sol = GridFunction(fes_saddle)
    chi_o_sol, v_o_sol = v_chi_sol.components
    Lhs_v = BilinearForm(fes_saddle)
    Rhs_v = LinearForm(fes_saddle)
    Lhs_v += InnerProduct(grad(v_trial).Trace(),grad(v_test).Trace())*ds\
            + chi*InnerProduct(n_proj,v_test)*ds\
            + chit*InnerProduct(n_proj,v_trial)*ds
    Rhs_v += chit*InnerProduct(n_proj,u)*ds

    Lhs_v.Assemble()
    Rhs_v.Assemble()
    v_chi_sol.vec.data = Lhs_v.mat.Inverse(inverse='pardiso')*Rhs_v.vec
    Ref_V = GridFunction(fesV)
    # Ref_V.vec.data = BaseVector(1/4*n_proj.vec.FV().NumPy())
    # Ref_V.vec.data = V_Interp.vec
    Ref_V.vec.data = V_Set.vec
    Err = GridFunction(fesV)
    Err.vec.data = v_o_sol.vec - Ref_V.vec

    H1errfun = lambda xfun: sqrt(Integrate(InnerProduct(grad(xfun).Trace(),grad(xfun).Trace())
                                            +InnerProduct((xfun),(xfun)), mymesh, 
                                            element_wise=False, order=7, definedon=mymesh.Boundaries('.*')))
    H1Err = H1errfun(Err)
    errs.append(H1Err)
    errsD.append(H1errfun(Err_D))
print('H1 errs are {}'.format(errs))
print('Convergence Rates for order {} is {}'.format(
    order, [np.log(errs[ii-1]/errs[ii])/np.log(2) for ii in range(len(errs)) if ii > 0]
))
print('H1 errs are {}'.format(errsD))
print('Convergence Rates for order {} is {}'.format(
    order, [np.log(errsD[ii-1]/errsD[ii])/np.log(2) for ii in range(len(errsD)) if ii > 0]
))



