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

BDF_order = 1
if BDF_order == 1:
    ext_BDF = [1]
    CBDF    = [1,1]
elif BDF_order == 2:
    ext_BDF = [2, -1]
    CBDF    = [3/2, 2, -1/2] 
elif BDF_order == 3:
    ext_BDF = [3,-3,1]
    CBDF    = [11/6, 3, -3/2, 1/3] 
elif BDF_order == 4:
    # ext_BDF = [3,-3,1]
    CBDF    = [25/12, 4, -3, 4/3, -1/4] 

hN = 2**8
Coords = np.array([[ii,0] for ii in np.linspace(0,1,hN)])
mymesh = Mesh1dFromPoints(Coords, dim = 1, adim = 1, Circ=False)

t = Parameter(0)
x_spatial = Coords[:,0]
u_exact = sin(x*np.pi)*exp(-np.pi**2*t)
u_e_np  = lambda x,t: np.sin(x*np.pi)*exp(-np.pi**2*t)

order = 2
fes = H1(mymesh,order = order, dirichlet=mymesh.Boundaries(".*"))
u,v = fes.TnT()

tau_set = np.array([2**j for j in [-6,-7,-8,-9,-10]])
LinfL2_e_set = []
L2H1_e_set = []
for tauval in tau_set:
    tau = Parameter(tauval)

    u_hist = []
    for ii in range(BDF_order):
        # For BDF scheme, save order-1 historic terms, the first is the nearest one 
        jj = BDF_order-1-ii
        locals()['u_sol_'+str(ii)] = GridFunction(fes)
        # Initialize 
        texact = jj*tauval
        t.Set(texact)
        locals()['u_sol_'+str(ii)].Interpolate(u_exact)
        u_hist.append(locals()["u_sol_"+str(ii)]) 
    
    tnow = (BDF_order-1)*tauval
    Lhs = BilinearForm(fes)
    Rhs = LinearForm(fes)
    Lhs += CBDF[0]/tau*u*v*dx + grad(u)*grad(v)*dx 
    for jj in range(BDF_order):
        Rhs += CBDF[jj+1]/tau*u_hist[jj]*v*dx
    
    Lhs.Assemble()
    u_ref = GridFunction(fes)
    u_err = GridFunction(fes)
    err_H1_set = []
    err_L2_set = []
    while tnow < T and abs(T-tnow)>1e-8:
        Rhs.Assemble()
        for ii in range(BDF_order):
            jj = BDF_order - 1 - ii
            if jj > 0:
                u_hist[jj].vec.data = BaseVector(u_hist[jj-1].vec.FV().NumPy())
            elif jj == 0:
                u_hist[jj].vec.data = Lhs.mat.Inverse(inverse="pardiso",freedofs=fes.FreeDofs())*Rhs.vec
        tnow += tauval

        t.Set(tnow)
        u_ref.Interpolate(u_exact)
        u_err.vec.data = BaseVector(u_ref.vec.FV().NumPy()-u_hist[0].vec.FV().NumPy())
        err_H1_set.append(H1errfun(u_err))
        err_L2_set.append(L2errfun(u_err))

    LinfL2_e = max(err_L2_set)
    L2H1_e = np.sqrt(tauval*sum(np.array(err_H1_set)**2))
    
    LinfL2_e_set.append(LinfL2_e)
    L2H1_e_set.append(L2H1_e)

fig = plt.figure(figsize = (12, 12))
ax = fig.add_subplot(111)
plt.loglog(tau_set,L2H1_e_set,'o-',linewidth=3,label='$|| e_u||_{L^2_{\\tau} H^1}$',markersize=18,markerfacecolor='none',markeredgewidth=4)
plt.loglog(tau_set,LinfL2_e_set,'-^',linewidth=3,label='$|| e_u||_{L^\infty_{\\tau} L^2}$',markersize=18,markerfacecolor='none',markeredgewidth=4)
plt.loglog(tau_set,tau_set**BDF_order,'k--',linewidth=4,label="$O(\\tau^{})$".format(BDF_order))
# plt.gca().set_aspect('equal')
plt.legend(loc='upper left',prop={'size': 18})
plt.grid(True,which="both",ls="--")
plt.gca().set_xlabel('$\\tau$')
plt.gca().set_title('BDF-{}'.format(BDF_order))

for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(20)
plt.show()
fig.savefig('./FEM_Coding/Fig/hN_{}_ord_{}.pdf'.format(hN,BDF_order),dpi=600,format='pdf')
print(L2H1_e_set)
print(LinfL2_e_set)