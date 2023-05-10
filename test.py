from Package_Geometry_Obj import *
import numpy as np
import netgen.meshing as ngm
N_set = np.array([2**(3+ii)+1 for ii in range(1)])
H1err_set = []

for N in N_set:
    xlin = np.linspace(0,1,N)
    print(N)
    Coords = np.vstack([xlin,np.zeros(xlin.shape)]).T
    adim = 1
    dim = 1
    nr = Coords.shape[0]
    mesh0= ngm.Mesh(dim=adim)
    pids = []
    for i in range(nr):
        pids.append (mesh0.Add (ngm.MeshPoint(ngm.Pnt(np.append(Coords[i],np.array([0]))))))  

    idx = mesh0.AddRegion("material_left", dim=dim)   ## dim=1: Boundary For a Mesh of adim=2
    idx2 = mesh0.AddRegion("material_right", dim=dim)   ## dim=1: Boundary For a Mesh of adim=2
    for i in range(nr):
        if i<(nr-1)/2:
            print('1{}'.format(nr))
            mesh0.Add(ngm.Element1D([pids[i],pids[i+1]],index=idx))
        elif i<nr-1:
            print('1{}'.format(i))
            mesh0.Add(ngm.Element1D([pids[i],pids[i+1]],index=idx2))
    idx2 = mesh0.AddRegion("bnd", dim=0)   ## dim=1: Boundary For a Mesh of adim=2
    mesh0.Add(ngm.Element0D(pids[0],index=idx2))
    mesh0.Add(ngm.Element0D(pids[-1],index=idx2))
    mymesh = Mesh(mesh0)

    from ngsolve import *

    fes = H1(mymesh,order =1,dirichlet="bnd")
    fes2 = H1(mymesh,order =2)

    gfu = GridFunction(fes)

    u,v = fes.TnT()
    Lhs = BilinearForm(fes)
    Rhs = LinearForm(fes)

    Lhs += IfPos(x-0.5,2,1)*InnerProduct(grad(u),grad(v))*dx+u*v*dx
    Rhs += IfPos(x-0.5,2*x**2-3*x-7,-4*x**2+2*x+8)*v*dx

    Lhs.Assemble()
    Rhs.Assemble()

    gfu.vec.data = Lhs.mat.Inverse(freedofs=fes.FreeDofs())*Rhs.vec

    res = gfu.vec.FV().NumPy()
    exactu = GridFunction(fes)
    exactu.Interpolate(IfPos(x-0.5,2*(x-1)**2+(x-1),-4*x**2+2*x))

    eu = IfPos(x-0.5,2*(x-1)**2+(x-1),-4*x**2+2*x)
    deu = IfPos(x-0.5,4*(x-1)+1,-8*x+2)
    deu_new = GridFunction(fes)
    deu_new.Interpolate(deu)
    err = GridFunction(fes)
    err.vec.data = gfu.vec - exactu.vec
# #     H1err_set.append(sqrt(Integrate(err**2, mymesh, element_wise=False,order=5)))
    H1err_set.append(sqrt( Integrate((grad(gfu)-deu_new)**2 + err**2, mymesh, \
                                    element_wise=False, definedon=mymesh.Materials("material_right"))))
#                     + Integrate((grad(gfu)-deu_new)**2 + err**2, mymesh, \
#                                     element_wise=False, definedon=mymesh.Materials("material_right"))))
    