from ngsolve import *
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/Users/liubocheng/Documents/PythonPackage/FEMGeoPackage/')
from Package_Geometry_Obj import *
T = 1
H1errfun = lambda xfun: sqrt(Integrate(InnerProduct(grad(xfun),grad(xfun))
                                    +InnerProduct((xfun),(xfun)), mymesh, element_wise=False))
L2errfun = lambda xfun: sqrt(Integrate(xfun*xfun, mymesh, element_wise=False))


Coords = np.array([[ii,0] for ii in np.linspace(0,1,2**8)])
mymesh = Mesh1dFromPoints(Coords, dim = 1, adim = 1, Circ=False)

t = Parameter(0)
x_spatial = Coords[:,0]
u_exact = sin(x*np.pi)*exp(-np.pi**2*t)
u_e_np  = lambda x,t: np.sin(x*np.pi)*exp(-np.pi**2*t)

order = 1
fes = H1(mymesh,order = order, dirichlet=mymesh.Boundaries(".*"))
u,v = fes.TnT()

tau_set = np.array([2**j for j in [-6,-7,-8,-9,-10]])
LinfL2_e_set = []
L2H1_e_set = []
for tauval in tau_set:
    tau = Parameter(tauval)
    u_sol = GridFunction(fes)
    u_sol_old = GridFunction(fes)

    Lhs = BilinearForm(fes)
    Rhs = LinearForm(fes)
    Lhs += 1/tau*u*v*dx + grad(u)*grad(v)*dx 
    Rhs += 1/tau*u_sol_old*v*dx

    tnow = 0
    t.Set(tnow)
    u_sol_old.Interpolate(u_exact)
    Lhs.Assemble()
    u_ref = GridFunction(fes)
    u_err = GridFunction(fes)
    err_H1_set = []
    err_L2_set = []
    while tnow < T and abs(T-tnow)>1e-8:
        Rhs.Assemble()
        u_sol.vec.data = Lhs.mat.Inverse(inverse="pardiso",freedofs=fes.FreeDofs())*Rhs.vec
        u_sol_old.vec.data = u_sol.vec
        tnow += tauval

        t.Set(tnow)
        u_ref.Interpolate(u_exact)
        u_err.vec.data = BaseVector(u_ref.vec.FV().NumPy()-u_sol.vec.FV().NumPy())
        err_H1_set.append(H1errfun(u_err))
        err_L2_set.append(L2errfun(u_err))

    LinfL2_e = max(err_L2_set)
    L2H1_e = np.sqrt(tauval*sum(np.array(err_H1_set)**2))
    
    LinfL2_e_set.append(LinfL2_e)
    L2H1_e_set.append(L2H1_e)

# plt.plot(x_spatial,u_e_np(x_spatial,T),'k-')
# plt.plot(x_spatial,u_sol.vec.FV().NumPy(),'k-')
# plt.show()
plt.loglog(tau_set,L2H1_e_set,'o-')
plt.loglog(tau_set,LinfL2_e_set,'k-')
plt.loglog(tau_set,tau_set,'k--')
plt.show()
print(L2H1_e_set)
print(LinfL2_e_set)