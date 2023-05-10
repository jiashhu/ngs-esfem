# import matplotlib.pylab as plt
from scipy.sparse import linalg
from scipy.sparse.linalg.dsolve.linsolve import spsolve
from netgen.csg import *
from Package_MyNgFunc import *
from Package_Numerical import bestappro
from scipy.sparse import bmat
from scipy.sparse import spdiags
import scipy.sparse.linalg as spla
from scipy.sparse.linalg import lgmres
import numpy as np
from line_profiler import LineProfiler
from ngsolve import *
import sys

ngsglobals.msg_level = 0

def CompCoeff(n,dom = [0.1,100]):
    '''
        dom: approximating interval
    '''
    pole, weight = bestappro(n,dom[0],dom[1])
    print("Chebyshev Rational Approximation Domain = [{},{}], order = {}".format(*[str(a) for a in dom],n))
    tpole = np.array([-(p_-1)/4 for p_ in pole])
    tweight = np.zeros(len(weight)+2)
    tweight[:-2] = -1/4*weight[:-2]
    tweight[-2] = 4*weight[-2]
    tweight[-1] = 1+(np.array(tweight[:-2])/tpole).sum()

    qn = 1/4
    am = -tweight[:-2]
    bm = tweight[:-2]/(2*(tpole-qn))
    Psi_n0 = tweight[-1]-tweight[-2]*qn-sum(tweight[:-2]/(tpole-qn))
    bn = Psi_n0/2
    dm = -tweight[:-2]/(2*tpole)
    dn = 1
    ###### -alphaC
    cm = tweight[:-2]/(2*tpole) - tweight[:-2]/(tpole-qn)
    cn = -Psi_n0
    cn_1 = 2
    em = 3/2*tweight[:-2]/tpole - tweight[:-2]/(tpole-qn)
    en = -Psi_n0
    ## 注意cn=en
    p_ = len(pole)
    return p_, tpole, tweight, am, bm, cm, dm, em, bn, cn, cn_1, dn, en, qn

def GenerateBeta(*PhiMat,CoefList):
    '''纵向排列，分块矩阵'''
    if len(PhiMat)==1:
        ## 把CoefList每一个元素乘以矩阵列向排列
        BaseMat = PhiMat[0]
        for ii in range(len(CoefList)):
            if ii > 0:
                res = bmat([[res],[BaseMat*CoefList[ii]]])
            else:
                res = BaseMat*CoefList[0]
    else:
        ## PhiMat是多个矩阵，则依次乘以Coef的系数列向排列
        for ii in range(len(CoefList)):
            if ii > 0:
                res = bmat([[res],[PhiMat[ii]*CoefList[ii]]])
            else:
                res = PhiMat[0]*CoefList[0]
    return res

def BlockM(*Matrix_list):
    for ii in range(len(Matrix_list)):
        if ii > 0:
            res = bmat([[res, None],[None, Matrix_list[ii]]])
        else:
            res = Matrix_list[ii]
    return res

def SparsePattern(SpMatrix,MatFlag=False,Draw_opt=False):
    '''
        MatFlag=True: 输入是个SparseMatrix，
        Draw_opt=True: 绘制Spy
    '''
    if MatFlag:
        row,col,val = SpMatrix.COO()
        tmp = NGMO.myCOO(row,col,val,*SpMatrix.shape,'scipy')
    else:
        row,col,val = SpMatrix.mat.COO()
        tmp = NGMO.myCOO(row,col,val,*SpMatrix.mat.shape,'scipy')
    if Draw_opt:
        plt.spy(tmp)
    return tmp

def GetBoundaryDof(fes_inn_V,B_Mat = 'ABCs'):
    ind_outer = []
    for el in fes_inn_V.Elements(BND):
        if el.mat == B_Mat:
            ind_outer.append(fes_inn_V.GetDofNrs(el))
    res2 = np.unique(ind_outer)
    return res2

def wedge(U,At):
    '''
        U向量，At标量，U.(梯度At x n)
        | gradA[0] gradA[1] gradA[2] |
        |   n[0]     n[1]     n[2]   |
        |   U[0]     U[1]     U[2]   |    
    '''
    gradA = grad(At).Trace()
    n = specialcf.normal(3)
    res = gradA[0]*(n[1]*U[2] - n[2]*U[1]) - gradA[1]*(n[0]*U[2]-n[2]*U[0]) + \
            gradA[2]*(n[0]*U[1]-n[1]*U[0])
    return res
    
def myDiv(u):
    '''Comput Div uT'''
    res = (grad(u).Trace()[0,0]+grad(u).Trace()[1,1]+grad(u).Trace()[2,2]) \
           -2*InnerProduct(u,specialcf.normal(3))
    return res

def mycurl(u):
    '''计算curl uT'''
    res = grad(u).Trace()
    n = specialcf.normal(3)
    curls_uT = -res[0,1]*n[2] + res[0,2]*n[1] + res[1,0]*n[2] - res[1,2]*n[0]\
                - res[2,0]*n[1] + res[2,1]*n[0]
    return curls_uT

def GenerateMesh(maxh,save_opt = False):
    '''
        生成几何体，一个大的球(ABCs)减去一个小的球(Interface)，
        返回Ngsolve.comp中的Mesh
    '''
    ## 生成外部和内部边界，相减得到domain
    sphere = Sphere(Pnt(0,0,0),1)
    sphere.bc("ABCs").maxh(maxh/2)
    spherei = Sphere(Pnt(0,0,0),0.5)
    spherei.bc("Interface").maxh(maxh)
    dom = sphere-spherei
    ## 构造CSG几何体
    geo = CSGeometry()
    geo.Add(dom)
    ## 生成网格
    mesh = geo.GenerateMesh(maxh = maxh)
    mymesh = Mesh(mesh)
    ## 保存网格vtk格式
    if save_opt:
        vtk = VTKOutput(ma=mymesh,names=['sol'],filename='sphere_mesh',subdivision=1)
        vtk.Do()
    mymesh.Curve(2)
    return mymesh

def GenerateFes(mymesh,rat_order=2,dom=[0.1,100],maxiter=200):
    '''
        Generate the finite element spaces for velocity, pressure and auxiliary variables. Mixed fem -- P2-P1 element is used.
        Auxiliary variables on boundaries are using 
    '''
    # region: Basic FEMs
    fes_ABC = H1(mymesh,order=2,definedon=mymesh.Boundaries("ABCs"))
    # 如果不compress，则fes_ABC和fes的ndof一样多，当然并不是真的自由度
    fes_ABC = Compress(fes_ABC)
    fes_ABC_V = VectorH1(mymesh,order=2,definedon=mymesh.Boundaries("ABCs"))
    fes_ABC_V = Compress(fes_ABC_V)
    fes_inn_V = VectorH1(mymesh,order=2,dirichlet="Interface")
    fes_inn_p = H1(mymesh,order=1)
    fes_N = NumberSpace(mymesh,definedon=mymesh.Boundaries("ABCs"))
    # endregion

    MAux, Beta, MU, Mp, Mb, MAux_Inv, MUinv= GenerateBlock(fes_ABC_V,fes_ABC,fes_inn_V,fes_inn_p,fes_N,rat_order,dom)
    
    # region: Dirichlet Boundary Condition
    r = sqrt(x**2+y**2+z**2)
    exactU0 = 3/(8*r)+1/(32*r**3)+3*(r**2-1/4)*x**2/(8*r**5)
    exactU1 = 3*(r**2-1/4)*x*y/(8*r**5)
    exactU2 = 3*(r**2-1/4)*x*z/(8*r**5)
    exactp = 3/4*x/r**3
    U_Dirichlet = GridFunction(fes_inn_V)
    U_Dirichlet.Set(CoefficientFunction((exactU0,exactU1,exactU2)),definedon=mymesh.Boundaries('Interface'))
    D_bnd = GetBoundaryDof(fes_inn_V,B_Mat = 'Interface')
    D_vec = U_Dirichlet.vec.FV().NumPy()

    U_exact = GridFunction(fes_inn_V)
    p_exact = GridFunction(fes_inn_p)
    U_exact.Set(CoefficientFunction((exactU0,exactU1,exactU2)))
    p_exact.Set(exactp)

    # D_vec[D_bnd] = U_exact.vec.FV().NumPy()[D_bnd]    
    # Bnd_err = np.linalg.norm(D_vec[D_bnd]-U_exact.vec.FV().NumPy()[D_bnd])
    # print('Bnd err = {}'.format(Bnd_err))
    # endregion

    # region: operator form
    N_list = [Mat.shape[0] for Mat in [MAux,MU,Mb]]
    print("NAux = {}, NU = {}, Np = {}. ".format(*N_list))
    N_A_op = sum(N_list)
    x0 = np.zeros(N_A_op)
    ## 算子表示矩阵，处理Dirichlet边界条件，gmres迭代，输出迭代次数与对应的residue
    A_op = lambda x: ATotal(MAux,Beta,MU,D_bnd,Mb,x,N_list)
    Non_D_ind = np.setdiff1d(np.arange(N_list[1]),D_bnd)
    AT = spla.LinearOperator((N_A_op, N_A_op), A_op)
    FTotal = np.zeros(len(x0))
    ## No source term, generated from the Dirichlet BC
    FTotal = -np.concatenate([Beta@D_vec,MU@D_vec,Mb@D_vec],axis=0)
    FTotal[MAux.shape[0]+D_bnd] = D_vec[D_bnd]
    # 预处理算子，给出的是一个求逆的过程 M_x = lambda x: spla.spsolve(A, x)
    M_op = lambda x: Pre(MAux_Inv,Beta,MUinv,D_bnd,Mb,Mp,x,N_list)
    MPre = spla.LinearOperator((N_A_op, N_A_op), M_op)
    # endregion

    counter = gmres_counter(A = A_op, b = FTotal)
    # Sol, exitCode = lgmres(AT, FTotal, x0=None, atol=0.03, maxiter=10, M=MPre, callback=counter, inner_m=10, outer_k=1, outer_v=None, store_outer_Av=True,prepend_outer_v=False)

    # Sol, exitCode = gmres(AT, FTotal, tol = 1e-3, maxiter=maxiter, restart=15,\
                            # callback=counter,M = MPre)
    
    # region: 测试AT
    tmp = Beta@U_exact.vec.FV().NumPy()
    Aux_Val = MatAuxInv(MAux_Inv,-tmp)
    print('辅助变量方程误差为{}'.format(np.linalg.norm(MAux@Aux_Val+tmp)))
    res = A_op(np.concatenate((Aux_Val,U_exact.vec.FV().NumPy(),p_exact.vec.FV().NumPy()),axis=0))
    print('精确解的残差为{}，其中辅助变量通过消元法计算得到'.format(np.linalg.norm(res-FTotal)))
    print('处理D氏边界的右端项norm为{}'.format(np.linalg.norm(FTotal)))

    [res0,res1] = testsol(MAux_Inv,MU,Beta,Mb,U_exact.vec.FV().NumPy(),p_exact.vec.FV().NumPy())
    print('res0为Beta*Aux+MU*U+bT*p = {}'.format(np.linalg.norm(res0[Non_D_ind])))
    print('res1为b*u = {}'.format(np.linalg.norm(res1)))
    # endregion

    ## 解向量+基函数 -> 有限元函数
    # SolU = GridFunction(fes_inn_V)
    # SolU.vec.data = BaseVector(Sol[N_list[0]:sum(N_list[:2])])
    # Solp = GridFunction(fes_inn_p)
    # Solp.vec.data = BaseVector(Sol[sum(N_list[:2]):sum(N_list)])
    # return SolU, Solp
    return 1,0

def testsol(MAux_Inv,MU,Beta,Mb,U_exact_vec,p_exact_vec):
    tmp = Beta@U_exact_vec
    res = MatAuxInv(MAux_Inv,-tmp)
    res0 = Beta.T@res + MU@U_exact_vec + Mb.T@p_exact_vec
    res1 = Mb@U_exact_vec
    return res0, res1

def testcase(fes_inn_V,phi4):
    '''
        3(P1+I)^{-1} = 3/2 \sum tw_m/tp_m (- \Laplace+tp_m)^{-1} + 3/2 tw_n
    '''
    ## 测试curls (e_i)_T = 0
    testv = GridFunction(fes_inn_V)
    testv.Set(CoefficientFunction((1,0,0)))
    res = Norm(BaseVector(phi4.mat*testv.vec))
    ## 测试wedge与curls的关系
    phi4_t1 = BilinearForm(trialspace=fes_inn_V,testspace=fes_ABC)
    phi4_t1 += ft*mycurl(u)*ds
    phi4_t1.Assemble()
    NGMO.Mat_Cf(phi4.mat,phi4_t1.mat,[range(phi4.mat.shape[0]),range(phi4.mat.shape[1])])

def GenerateBlock(fes_ABC_V,fes_ABC,fes_inn_V,fes_inn_p,fes_N,rat_order,dom):
    # region: Bilinear: Mass, Stiff
    phi, phi_ = fes_ABC.TnT()
    Mass_Surface = BilinearForm(fes_ABC)
    Stiff_Surface = BilinearForm(fes_ABC)
    Mass_Surface += phi*phi_*ds
    Stiff_Surface += InnerProduct(grad(phi).Trace(),grad(phi_).Trace())*ds

    phiV, phiV_ = fes_ABC_V.TnT()
    MassV_Surface = BilinearForm(fes_ABC_V)
    MassV_Surface += phiV*phiV_*ds
    StiffV_Surface = BilinearForm(fes_ABC_V)
    StiffV_Surface += InnerProduct(grad(phiV).Trace(),grad(phiV_).Trace())*ds

    Mass_Surface.Assemble()   
    Stiff_Surface.Assemble() 
    MassV_Surface.Assemble()
    StiffV_Surface.Assemble()
    # endregion

    # region: Mixed Bilinear phi
    A,At = fes_ABC_V.TnT()
    f,ft = fes_ABC.TnT()
    u,ut = fes_inn_V.TnT()
    ds_ABC = ds(definedon=mymesh.Boundaries("ABCs"))
    phi1 = BilinearForm(trialspace=fes_inn_V,testspace=fes_ABC_V)
    phi1 += InnerProduct(u,At)*ds_ABC
    phi2 = BilinearForm(trialspace=fes_inn_V,testspace=fes_ABC)
    phi2 += InnerProduct(u,specialcf.normal(3))*ft*ds_ABC
    phi3 = BilinearForm(trialspace=fes_inn_V,testspace=fes_ABC)
    phi3 += -InnerProduct(grad(ft).Trace(),u)*ds_ABC
    phi4 = BilinearForm(trialspace=fes_inn_V,testspace=fes_ABC)
    phi4 += wedge(u,ft)*ds_ABC

    phi1.Assemble()
    phi2.Assemble()
    phi3.Assemble()
    phi4.Assemble()
    # endregion

    # region: Bilinear: Lagrangian for Stiff
    fes_Lag = fes_ABC*fes_N
    (v,lam),(vt,mu) = fes_Lag.TnT()
    Stiff_Lag = BilinearForm(fes_Lag)
    Stiff_Lag += (InnerProduct(grad(v).Trace(),grad(vt).Trace()) + v*mu + vt*lam)*ds
    Stiff_Lag.Assemble()

    phi3cn1 = BilinearForm(trialspace=fes_inn_V,testspace=fes_Lag)
    phi3cn1 += -InnerProduct(u,grad(vt).Trace())*ds
    phi4cn1 = BilinearForm(trialspace=fes_inn_V,testspace=fes_Lag)
    phi4cn1 += wedge(u,vt)*ds

    phi3cn1.Assemble()
    phi4cn1.Assemble()
    # endregion

    # region: ngsolve -> scipy.sparse
    T_Phi_S = SparsePattern(StiffV_Surface.mat,MatFlag=True)
    T_Phi_M = SparsePattern(MassV_Surface.mat,MatFlag=True)
    Phi_S   = SparsePattern(Stiff_Surface.mat,MatFlag=True)
    Phi_M   = SparsePattern(Mass_Surface.mat,MatFlag=True)
    Alpha_1 = SparsePattern(phi1.mat,MatFlag=True)
    Alpha_2 = SparsePattern(phi2.mat,MatFlag=True)
    Alpha_3 = SparsePattern(phi3.mat,MatFlag=True)
    Alpha_4 = SparsePattern(phi4.mat,MatFlag=True)
    Alpha_5 = Alpha_2+Alpha_3
    Phi_S_ext = SparsePattern(Stiff_Lag.mat,MatFlag=True)
    Alpha_3_ext = SparsePattern(phi3cn1.mat,MatFlag=True)
    Alpha_4_ext = SparsePattern(phi4cn1.mat,MatFlag=True)
    # endregion

    # region: MAux
    p_, tpole, tweight, am, bm, cm, dm, em, bn, cn, cn_1, dn, en, qn = CompCoeff(rat_order,dom=dom)
    for m in range(p_):
        AMm = -am[m]*T_Phi_S - am[m]*tpole[m]*T_Phi_M
        BMm = -bm[m]*Phi_S - bm[m]*tpole[m]*Phi_M
        CMm = -cm[m]*Phi_S - cm[m]*tpole[m]*Phi_M
        DMm = -dm[m]*Phi_S - dm[m]*tpole[m]*Phi_M
        EMm = -em[m]*Phi_S - em[m]*tpole[m]*Phi_M
        if m > 0:
            MA = bmat([[MA, None],[None, AMm]])
            MB = bmat([[MB, None],[None, BMm]])
            MC = bmat([[MC, None],[None, CMm]])
            MD = bmat([[MD, None],[None, DMm]])
            ME = bmat([[ME, None],[None, EMm]])
        else: 
            MA, MB, MC, MD, ME = AMm, BMm, CMm, DMm, EMm
    B0 = -bn*Phi_S - bn*qn*Phi_M
    Cp = -cn*Phi_S - cn*qn*Phi_M
    E0 = -en*Phi_S - en*qn*Phi_M
    C0 = -cn_1*Phi_S_ext
    D0 = -dn*Phi_S_ext
    MB0_Cp_E0_C0_D0 = BlockM(B0,Cp,E0,C0,D0)
    MAux = BlockM(MA,MB,MC,MD,ME,MB0_Cp_E0_C0_D0).tocsc()
    # endregion

    # region: sci_sparse -> ng_sparse -> Inverse
    MAux_Ng = [NGMO.myCOO(list(Mat_.row),list(Mat_.col),list(Mat_.data),*Mat_.shape,'ngsolve') for Mat_ in [MA,MB,MC,MD,ME,B0.tocoo(),Cp.tocoo(),E0.tocoo(),C0.tocoo(),D0.tocoo()]]
    MAux_Inv = []
    for Mat in MAux_Ng:
        MAux_Inv.append(Mat.Inverse(inverse='pardiso'))
    # endregion

    # region: Generate Beta in sci_sparse
    Beta_a = GenerateBeta(Alpha_1,CoefList=am)
    Beta_b = GenerateBeta(Alpha_2,CoefList=bm)
    Beta_c = GenerateBeta(Alpha_3,CoefList=cm)
    Beta_d = GenerateBeta(Alpha_4,CoefList=dm)
    Beta_e = GenerateBeta(Alpha_5,CoefList=em)
    Beta_B0_Cp_E0_C0_D0 = GenerateBeta(Alpha_2,Alpha_3,Alpha_5,Alpha_3_ext,Alpha_4_ext,\
                                    CoefList=[bn,cn,en,cn_1,dn])
    Beta = GenerateBeta(Beta_a,Beta_b,Beta_c,Beta_d,Beta_e,Beta_B0_Cp_E0_C0_D0,\
                        CoefList=[1]*6).tocsc()
    # endregion

    # region: MU
    viscosity = 1
    Auv = BilinearForm(fes_inn_V)
    Auv += 1/2*viscosity*InnerProduct(grad(u)+grad(u).trans,grad(ut)+grad(ut).trans)*dx
    uN = InnerProduct(u,specialcf.normal(3))
    utN = InnerProduct(ut,specialcf.normal(3))
    ds_ABC = ds(definedon=mymesh.Boundaries("ABCs"))
    ## 注意积分区域是外边界，不然会多出内边界的自由度
    # tAuv = BilinearForm(fes_inn_V)
    Auv += tweight[-2]*InnerProduct(grad(u).Trace(),grad(ut).Trace())*ds_ABC
    Auv += tweight[-1]*InnerProduct(u,ut)*ds_ABC
    Auv += tweight[-2]/2*uN*utN*ds_ABC
    Auv += -tweight[-2]/2*(myDiv(u)*myDiv(ut)+mycurl(u)*mycurl(ut))*ds_ABC
    Auv += tweight[-2]/2*(uN+myDiv(u))*(utN+myDiv(ut))*ds_ABC
    Auv.Assemble()
    MU = SparsePattern(Auv.mat,MatFlag=True).tocsc()
    MUinv = Auv.mat.Inverse(inverse='pardiso',freedofs=fes_inn_V.FreeDofs())
    # endregion

    # region: MD Test
    # len_set = [x.shape[1] for x in [MA,MB,MC,MD,ME]] 
    # index_list = []
    # for i in range(len(len_set)):
    #     index_list += [i]*len_set[i]
    # len_set2 = [x.shape[1] for x in [B0,Cp,E0,C0,D0]]
    # index_list += [1]*len_set2[0]+[2]*len_set2[1]+[4]*len_set2[2]+[2]*len_set2[3]+[3]*len_set2[-1]
    # index_list = np.array(index_list)
    # A_index, B_index, C_index, D_index, E_index = [index_list==i for i in range(5)]
    
    # ## curl u_T, curl v_T
    # Auv = BilinearForm(fes_inn_V)
    # Auv += -tweight[-2]/2*mycurl(u)*mycurl(ut)*ds_ABC
    # ## Vectorial Spherical Harmonic T_1^0 = (-y,x,0)
    # Uvec = GridFunction(fes_inn_V)
    # f = GridFunction(H1(mymesh,order=2,dirichlet="Interface"))
    # T10 = CoefficientFunction((-y,x,0))
    # N10 = CoefficientFunction((x,y,z))
    # I10 = CoefficientFunction(6*z*(-z*x,-z*y,x**2+y**2)) + 2*(3*z**2-1)*N10
    # Uvec.Set(T10+N10)
    # res = (Beta@Uvec.vec.FV().NumPy())*D_index  # 保留D_index部分
    # DM_vec = MatAuxInv(MAux_Inv,-res)
    # res2 = BaseVector(Auv.Assemble().mat*Uvec.vec)
    # errD = np.linalg.norm(Beta.T@DM_vec + res2.FV().NumPy())
    # print('errD = {}'.format(errD))
    # endregion

    # region: MB Test
    # Uvec.Set(N10)
    # Auv = BilinearForm(fes_inn_V)
    # Auv += tweight[-2]/2*uN*utN*ds_ABC
    # res2 = BaseVector(Auv.Assemble().mat*Uvec.vec)
    # f.Set(2,definedon = mymesh.Boundaries("ABCs"))
    # Rhs = LinearForm(fes_inn_V)
    # Rhs += f*utN*ds_ABC
    # Rhs.Assemble()

    # res = (Beta@Uvec.vec.FV().NumPy())*B_index
    # BM_vec = MatAuxInv(MAux_Inv,-res)
    # errB = np.linalg.norm(Beta.T@BM_vec + res2.FV().NumPy() - Rhs.vec.FV().NumPy())    
    # print('errB = {}'.format(errB))
    # endregion

    # region: MC Test 
    # Uvec.Set(I10)
    # Auv = BilinearForm(fes_inn_V)
    # Auv += -tweight[-2]/2*myDiv(u)*myDiv(ut)*ds_ABC
    # res2 = BaseVector(Auv.Assemble().mat*Uvec.vec)
    # f.Set(4/5*(3*z**2-1),definedon = mymesh.Boundaries("ABCs"))
    # Rhs = LinearForm(fes_inn_V)
    # Rhs += f*myDiv(ut)*ds_ABC
    # Rhs.Assemble()

    # res = (Beta@Uvec.vec.FV().NumPy())*C_index
    # CM_vec = MatAuxInv(MAux_Inv,-res)
    # errC = np.linalg.norm(Beta.T@CM_vec + res2.FV().NumPy() - Rhs.vec.FV().NumPy())    
    # print('errC = {}'.format(errC))
    # endregion

    # (sum(cm/(6+tpole))-1/2*tweight[-2]+cn/(6+qn)+2/6)*-6

    # region: Mixed Bilinear For Uxp
    p,pt = fes_inn_p.TnT()
    Bilinear_p = BilinearForm(fes_inn_p)
    Bilinear_p += p*pt*dx
    # Mp = SparsePattern(Bilinear_p.Assemble().mat,MatFlag=True).tocsc()
    Mp = Bilinear_p.Assemble().mat

    Bilinear_b = BilinearForm(trialspace=fes_inn_V,testspace=fes_inn_p)
    Bilinear_b += -(grad(u)[0,0]+grad(u)[1,1]+grad(u)[2,2])*pt*dx
    Mb = SparsePattern(Bilinear_b.Assemble().mat,MatFlag=True).tocsc()
    # endregion
    return MAux, Beta, MU, Mp, Mb, MAux_Inv, MUinv

def GenerateBlock_Beta(MS_Sur,Phi_Mat,coef,tpole,Sp_Tag='ngsolve'):
    '''
        MS_Sur == 1, for Lagrangian multiplier
    '''
    Block_res = []
    Beta_res = []
    if len(MS_Sur) == 2:
        MM,MS = [SparsePattern(Mat,MatFlag=True).tocsc() for Mat in MS_Sur]
        PM = SparsePattern(Phi_Mat,MatFlag=True).tocsc()
        for am,qm in zip(coef,tpole):
            tmp = (-am*MS - am*qm*MM).tocoo()
            Block_res.append(NGMO.myCOO(list(tmp.row),list(tmp.col),list(tmp.data),*tmp.shape,Sp_Tag))
            tb = (am*PM).tocoo()
            Beta_res.append(NGMO.myCOO(list(tb.row),list(tb.col),list(tb.data),*tb.shape,Sp_Tag))
    else:
        MS = [SparsePattern(Mat,MatFlag=True).tocsc() for Mat in MS_Sur]
        PM = SparsePattern(Phi_Mat,MatFlag=True).tocsc()
        for am,qm in zip(coef,tpole):
            tmp = (-am*MS).tocoo()
            Block_res.append(NGMO.myCOO(list(tmp.row),list(tmp.col),list(tmp.data),*tmp.shape,Sp_Tag))
            tb = (am*PM).tocoo()
            Beta_res.append(NGMO.myCOO(list(tb.row),list(tb.col),list(tb.data),*tb.shape,Sp_Tag))
    return Block_res,Beta_res

def L2Err(SolU,Solp,mymesh,Bid="Interface"):
    r = sqrt(x**2+y**2+z**2)
    exactU0 = 3/(8*r)+1/(32*r**3)+3*(r**2-1/4)*x**2/(8*r**5)
    exactU1 = 3*(r**2-1/4)*x*y/(8*r**5)
    exactU2 = 3*(r**2-1/4)*x*z/(8*r**5)
    exactp = 3/4*x/r**3
    U0dx = 3*x**3/(4*r**5)-5*x**3*(3*r**2-3/4)/(8*r**7)-3*x/(8*r**3)+x*(3*r**2-3/4)/(4*r**5)-3*x/(32*r**5)
    U0dy = 3*x**2*y/(4*r**5)-5*x**2*y*(3*r**2-3/4)/(8*r**7)-3*y/(8*r**3)-3*y/(32*r**5)
    U0dz = 3*x**2*z/(4*r**5)-5*x**2*z*(3*r**2-3/4)/(8*r**7)-3*z/(8*r**3)-3*z/(32*r**5)
    gradU0 = CoefficientFunction((U0dx, U0dy, U0dz))
    err = [SolU.components[0]-exactU0,SolU.components[1]-exactU1,SolU.components[2]-exactU2]
    L2err0 = sqrt(Integrate(err[0]**2, mymesh, element_wise=False,definedon = mymesh.Boundaries(Bid)))
    L2err1 = sqrt(Integrate(err[1]**2, mymesh, element_wise=False,definedon = mymesh.Boundaries(Bid)))
    L2err2 = sqrt(Integrate(err[2]**2, mymesh, element_wise=False,definedon = mymesh.Boundaries(Bid)))
    L2U = sqrt(Integrate(err[0]**2+err[1]**2+err[2]**2, mymesh, element_wise=False,order=6))
    H1U = sqrt(Integrate( InnerProduct(grad(SolU.components[0])-gradU0,grad(SolU.components[0])-gradU0), mymesh, order=5, element_wise=False))
    L2p = sqrt(Integrate((exactp-Solp)**2, mymesh, element_wise=False))
    print("exactp的均值为{}".format(Integrate((exactp),mymesh, element_wise=False)))
    res = Integrate(1,mymesh,element_wise=False)
    print('1积分为{}'.format(res))
    meanSolp = Integrate(Solp,mymesh,element_wise=False)/res
    print('Solp的常数为{}'.format(meanSolp))
    L2pt = sqrt(Integrate((exactp-Solp+meanSolp)**2, mymesh, element_wise=False))
    print('减去常数的结果={}'.format(L2pt))
    print("L2 error -- {} -- of Ux is {}, of Uy is {}, of Uz is {}".format(Bid,L2err0,L2err1,L2err2))
    print("L2 error of U is {}".format(L2U))
    print("H1 error of U is {}".format(H1U))
    print("L2 error of p is {}".format(L2p))

def Pre(MAuxInv,Beta,MUinv,D_bnd,Mb,Mp,x_vec,N_list):
    '''
        Precondition:
        | I       |  | I        |   | K^-1   |
        |    S^-1 |  | 0 -Mb  I |   |      I |
        K:
        | MAux       Beta |
        | Beta.T      MU  |
        K^-1 substituted by 
        | I         |  | I          |  | MAux^-1     |  | x_A |
        |    MU^-1  |  | -Beta.T  I |  |          I  |  | x_U |
    '''
    N_Aux, N_U, N_p = N_list
    Non_D_ind = np.setdiff1d(np.arange(N_U),D_bnd)
    N_x = len(x_vec)
    res = np.zeros(N_x)
    res[:N_Aux+N_U] = Kinv(MAuxInv,Beta,MUinv,D_bnd,x_vec[:N_Aux+N_U],N_Aux,N_U)
    res[N_Aux+N_U:] = -Mb[:,Non_D_ind]@res[N_Aux+Non_D_ind] + x_vec[N_Aux+N_U:]
    res[N_Aux+N_U:] = BaseVector(-Mp.Inverse(inverse="pardiso")*BaseVector(res[N_Aux+N_U:])).FV().NumPy()
    return res

def Kinv(MAuxInv,Beta,MUinv,D_bnd,x_AU,N_Aux,N_U):
    Ddata = x_AU[N_Aux+D_bnd]
    ## 辅助变量的分块逆
    res = np.zeros(len(x_AU))
    res[:N_Aux] = MatAuxInv(MAuxInv,x_AU[:N_Aux])
    
    Non_D_ind = np.setdiff1d(np.arange(N_U),D_bnd)
    res[N_Aux+Non_D_ind] = x_AU[N_Aux+Non_D_ind] -Beta.T[Non_D_ind]@res[:N_Aux]
    res[N_Aux:] = BaseVector(MUinv*BaseVector(x_AU[N_Aux:])).FV().NumPy()
    res[N_Aux+D_bnd] = Ddata
    return res

def MatAuxInv(MAuxInv,x_A):
    ## MAuxInv分别是Aux的Ngsolve矩阵逆
    Nbegin = 0
    for Mat_inv in MAuxInv:
        N_Mat = Mat_inv.shape[0]
        x_A[Nbegin:Nbegin+N_Mat] = BaseVector(Mat_inv*BaseVector(x_A[Nbegin:Nbegin+N_Mat]))
        Nbegin += N_Mat
    return x_A

class gmres_counter(object):
    def __init__(self, A, b, disp=True):
        self._disp = disp
        self.niter = 0
        self.Afun = A
        self.b = b
    def __call__(self, xk=None):
        self.niter += 1
        if self._disp & (self.niter%1==0):
            print('iter %3i\trk = %s' % (self.niter, str(np.linalg.norm(self.b-self.Afun(xk)))))

def ATotal(MAux,Beta,MU,D_bnd,Mb,x_vec,N_list):
    '''
        D_bnd: Dirichlet boundary index w.r.t MU, and Beta,
        | MAux       Beta      0    | | x_Aux | 
        | Beta.T      MU      Mb.T  | |  x_U  | 
        |  0          Mb       0    | |  x_p  |
    '''
    N_Aux, N_U, N_p = N_list
    N_x = len(x_vec)
    assert(N_x==(N_Aux+N_U+N_p))

    res = np.zeros(N_x)
    x_Aux, x_U, x_p = x_vec[:N_Aux], x_vec[N_Aux:N_Aux+N_U], x_vec[N_Aux+N_U:]
    Non_D_ind = np.setdiff1d(np.arange(N_U),D_bnd)
    ## Beta为csc矩阵
    res[:N_Aux] = MAux@x_Aux + Beta[:,Non_D_ind]@x_U[Non_D_ind]
    res[N_Aux:N_Aux+N_U] = Beta.T@x_Aux+MU[:,Non_D_ind]@x_U[Non_D_ind]+Mb.T@x_p
    ## 对应于D_bnd的块为单位阵
    res[N_Aux+D_bnd] = x_U[D_bnd] 
    res[N_Aux+N_U:] = Mb[:,Non_D_ind]@x_U[Non_D_ind]
    return res

def Prepare_Pre(MAux_data,MU,D_bnd,p_):
    # MA,MB,MC,MD,ME,B0,Cp,E0,C0,D0 = MAux_data
    MAux_LU = MAux_data
    ## MU设置Dirichlet边界0,1
    N_U = MU.shape[1]
    chi_interior = np.ones(N_U)
    chi_interior[D_bnd] = 0.0
    I_interior = spdiags(chi_interior, [0], N_U, N_U).tocsc()
    chi_boundary = np.zeros(N_U)
    chi_boundary[D_bnd] = 1.0
    I_boundary = spdiags(chi_boundary, [0], N_U, N_U).tocsc()
    MU_mod = I_interior @ MU @ I_interior + I_boundary
    return MAux_LU, MU_mod

mymesh = GenerateMesh(maxh=0.16)
# prof = LineProfiler(GenerateFes)  # 把函数传递到性能分析器中
# prof.add_function(Kinv)
# prof.enable()  # 开始性能分析
SolU, Solp = GenerateFes(mymesh,rat_order=3,dom = [0.1,100],maxiter=150)
# prof.disable()  # 停止性能分析
# prof.print_stats(sys.stdout)  # 打印性能分析结果
# L2Err(SolU,Solp,mymesh,Bid="ABCs")