'''
    Test high order convergence of material derivative on circle with isoparametric approximation 
'''

from Package_Geometry_Obj import CircleMesh, EllipseMesh
from Package_MyNgFunc import Pos_Transformer
from ngsolve import *
import numpy as np

def TensorProduct(u,v):
    assert(type(u)==list)
    assert(type(v)==list)
    return CF((u[0]*v[0],u[0]*v[1],u[1]*v[0],u[1]*v[1]),dims=(2,2))

class VinRad():
    def __init__(self) -> None:
        r = sqrt(x**2+y**2)
        # Setting of velocity field, along radius, with velocity (1-r)r, started from r = 1/2
        self.vR = lambda r: (1-r)*r
        self.u = CF(((1-r)*x,(1-r)*y))
        # Exact solution: r = 1/(1+e^-t)

        # derived Plane Hessian for each component of velocity
        self.mat_D2u = [
            CF((x/r*(x**2/r**2-3), y/r*(x**2/r**2-1), 
                y/r*(x**2/r**2-1), x/r*(y**2/r**2-1)), dims=(2,2)),
            CF((y/r*(x**2/r**2-1), x/r*(y**2/r**2-1), 
                x/r*(y**2/r**2-1), y/r*(y**2/r**2-3)), dims=(2,2))
        ]
        # Plane Gradient -- Du: first row: D u_0, second row D u_1
        self.mat_Du = [
            [-x**2/r+(1-r), -x*y/r],
            [-x*y/r, -y**2/r+(1-r)]
        ]

        self.exactr = lambda t: 1/(1+np.exp(-t))

# BDF test for material derivative of surface gradient of normal
order = 3
BDForder = 2
errs = []
errsD = []
H1errfun = lambda xfun: sqrt(Integrate(InnerProduct(grad(xfun).Trace(),grad(xfun).Trace())
                                    +InnerProduct((xfun),(xfun)), mymesh, 
                                    element_wise=False, order=7, definedon=mymesh.Boundaries('.*')))
tau0 = 0.01
T = 0.1
tau_set = tau0/2**np.array(range(1,6)) # 1,2,3,4
tau_set = [min(tau_set)]
# tau_set = tau0/2**np.array(range(1,5)) # 1,2,3,4
for nk in [1,2,3,4]:
# for nk in [9]:
    for tauval in tau_set:
        meshobj = CircleMesh(rad = 1/2, n = 2**(2+nk))
        meshobj.MeshHo(order = order)
        mymesh = meshobj.mesh
        fes = meshobj.hofes
        fesV = meshobj.hofesV

        Info_VF = VinRad()
        u = Info_VF.u
        mat_D2u = Info_VF.mat_D2u
        mat_Du = Info_VF.mat_Du
        r_0 = Info_VF.exactr(0)
        r_dt = Info_VF.exactr(tauval)

        PolyPos = GridFunction(fesV)
        PolyPos.Set(CF((x,y)),definedon=mymesh.Boundaries('.*'))

        ini_disp = meshobj.ini_disp
        mymesh.SetDeformation(ini_disp)

        Ini_Pos = GridFunction(fesV)
        Ini_Pos.vec.data = BaseVector(PolyPos.vec.FV().NumPy()+ini_disp.vec.FV().NumPy())
        Ini_Pos_dt = GridFunction(fesV)
        Ini_Pos_dt.vec.data = BaseVector(r_dt/r_0*Ini_Pos.vec.FV().NumPy())
        deform_old = GridFunction(fesV)
        deform_old.vec.data = ini_disp.vec
        deform = GridFunction(fesV)
        deform.vec.data = BaseVector(Ini_Pos_dt.vec.FV().NumPy()-PolyPos.vec.FV().NumPy())

        MatrixL2 = FESpace( [fes,fes,fes] )
        #         n  weingarten
        fes_pq = fesV*MatrixL2
        p, qxx, qxy, qyy = fes_pq.TrialFunction()
        pt, qtxx, qtxy, qtyy = fes_pq.TestFunction()
        q = CoefficientFunction( (qxx, qxy,
                                qxy, qyy), dims=(2,2))
        qt = CoefficientFunction( (qtxx, qtxy,
                                qtxy, qtyy), dims=(2,2))
        gfuold = GridFunction(fes_pq)
        p_o_sol, q_o_sxx, q_o_sxy, q_o_syy = gfuold.components
        q_o_mat = CF((q_o_sxx,q_o_sxy,q_o_sxy,q_o_syy),dims=(2,2))

        gfuold2 = GridFunction(fes_pq)
        p_o_sol2, q_o_sxx2, q_o_sxy2, q_o_syy2 = gfuold2.components
        q_o_mat2 = CF((q_o_sxx2,q_o_sxy2,q_o_sxy2,q_o_syy2),dims=(2,2))

        gfuold0 = GridFunction(fes_pq)
        p_o_sol0, q_o_sxx0, q_o_sxy0, q_o_syy0 = gfuold0.components
        q_o_mat0 = CF((q_o_sxx0,q_o_sxy0,q_o_sxy0,q_o_syy0),dims=(2,2))

        Lhs_pq = BilinearForm(fes_pq)
        Rhs_pq = LinearForm(fes_pq)

        ext_BDF = [2, -1]
        CBDF    = [3/2, 2, -1/2] # additional - except the first item

        tau = Parameter(tauval)
        v_o_sol = GridFunction(fesV)

        p_ext_v = GridFunction(fesV)
        p_ext_1,p_ext_2 = p_ext_v.components
        rCF = sqrt(x**2+y**2)
        nCF_lift = CF((x/rCF,y/rCF))
        DnCF_lift = CF((y**2/rCF**3, -x*y/rCF**3, -x*y/rCF**3, x**2/rCF**3),dims=(2,2))
        Initial_N = GridFunction(fesV)
        Initial_N.Set(nCF_lift, definedon=mymesh.Boundaries('.*'))
        p_ext_v.vec.data = Initial_N.vec
        mat_pv = TensorProduct([p_ext_1,p_ext_2],[CF(0),CF(0)])
        mat_Projp = TensorProduct([p_ext_1,p_ext_2],[p_ext_1,p_ext_2])
        Id = CF((1,0,0,1),dims=(2,2))
        mat_Ptn = Id - mat_Projp
        mat_DuCF = CF((mat_Du[0][0],mat_Du[0][1],mat_Du[1][0],mat_Du[1][1]),dims=(2,2))
        Du = mat_DuCF*mat_Ptn
        
        q_ext = 2*q_o_mat-q_o_mat2
        Lhs_pq += CBDF[0]/tau*InnerProduct(q,qt)*ds
        Rhs_pq += CBDF[1]/tau*InnerProduct(q_o_mat,qt)*ds \
                    + CBDF[2]/tau*InnerProduct(q_o_mat2,qt)*ds
        Rhs_pq += InnerProduct(mat_pv*q_ext**2,qt)*ds\
                    - InnerProduct(q_ext*Du,qt)*ds\
                    - InnerProduct(Du.trans*q_ext + mat_Projp*grad(v_o_sol).Trace()*q_ext,qt)*ds 

        Hessu = [mat_Ptn*mat_D2u[ii]*mat_Ptn for ii in range(2)]
        Rhs_pq += -(InnerProduct(Hessu[0],qt)*p_ext_1+InnerProduct(Hessu[1],qt)*p_ext_2)*ds
        Rhs_pq += InnerProduct(q_ext*mat_DuCF*mat_Projp,qt)*ds
        Rhs_pq += InnerProduct(mat_DuCF*p_ext_v,p_ext_v)*InnerProduct(q_ext,qt)*ds

        rCF = sqrt(x**2+y**2)
        nCF_lift = CF((x/rCF,y/rCF))
        DnCF_lift = CF((y**2/rCF**3, -x*y/rCF**3, -x*y/rCF**3, x**2/rCF**3),dims=(2,2))
        PCF = CF((1-(x/rCF)**2, -x*y/rCF**2, -x*y/rCF**2, 1-(y/rCF)**2),dims = (2,2))
        sgnCF = PCF*DnCF_lift
        # sgn is mean curvature (1/r) times ...
        Exact_sgn00_2 = GridFunction(fes)
        Exact_sgn01_2 = GridFunction(fes)
        Exact_sgn11_2 = GridFunction(fes)
        Exact_sgn00_2.Set(sgnCF[0,0],definedon=mymesh.Boundaries('.*'))
        Exact_sgn01_2.Set(sgnCF[0,1],definedon=mymesh.Boundaries('.*'))
        Exact_sgn11_2.Set(sgnCF[1,1],definedon=mymesh.Boundaries('.*'))
        q_o_sxx2.vec.data = Exact_sgn00_2.vec
        q_o_sxy2.vec.data = Exact_sgn01_2.vec
        q_o_syy2.vec.data = Exact_sgn11_2.vec

        q_o_sxx.vec.data = BaseVector(q_o_sxx2.vec.FV().NumPy()*r_0/r_dt)
        q_o_sxy.vec.data = BaseVector(q_o_sxy2.vec.FV().NumPy()*r_0/r_dt)
        q_o_syy.vec.data = BaseVector(q_o_syy2.vec.FV().NumPy()*r_0/r_dt)
        Exact_V0 = GridFunction(fesV)
        Exact_V0.Set(u,definedon=mymesh.Boundaries('.*'))
        Exact_V0_np = Exact_V0.vec.FV().NumPy().copy()

        told = (BDForder-1)*tauval
        X_test = GridFunction(fesV)
        Ref_V = GridFunction(fes)
        print(told)
        while told <= T:
            Lhs_pq.Assemble()
            Rhs_pq.Assemble()
            gfuold2.vec.data = BaseVector(gfuold.vec.FV().NumPy())
            gfuold.vec.data = Lhs_pq.mat.Inverse(inverse='pardiso')*Rhs_pq.vec

            v_o_sol.vec.data = BaseVector(Info_VF.vR(Info_VF.exactr(told))/Info_VF.vR(r_0)*Exact_V0_np.flatten('F'))
            tmp = 2*deform.vec.FV().NumPy() - deform_old.vec.FV().NumPy() + tauval*v_o_sol.vec.FV().NumPy()
            deform_old.vec.data = deform.vec
            deform.vec.data = BaseVector(tmp)
            mymesh.SetDeformation(deform)
            told += tauval
            Ref_V.vec.data = BaseVector(Exact_sgn00_2.vec.FV().NumPy()*r_0/Info_VF.exactr(told))

            if (abs(told-T))<1e-5:
                Err = GridFunction(fes)
                Err.vec.data = q_o_sxx.vec - Ref_V.vec
                H1Err = H1errfun(Err)
                errs.append(H1Err)


print('H1 errs of BDF2s are {}'.format(errs))
print('Convergence Rates for order {} is {}'.format(
    order, [np.log(errs[ii-1]/errs[ii])/np.log(2) for ii in range(len(errs)) if ii > 0]
))


