from ngsolve import *
from scipy.sparse import coo_matrix
import numpy as np
import os
from global_utils import FO
from ngsolve.comp import IntegrationRuleSpaceSurface
import copy

def GetGeoHighOrderQuant(mymesh,order,dof_linear_num):
    mymesh.Curve(order)
    fes = H1(mymesh,order=order)
    fesV = VectorH1(mymesh, order=order)
    np_vec_sets = []
    sgnCF = specialcf.Weingarten(3)
    Exact_sgn00_2 = GridFunction(fes)
    Exact_sgn01_2 = GridFunction(fes)
    Exact_sgn02_2 = GridFunction(fes)
    Exact_sgn11_2 = GridFunction(fes)
    Exact_sgn12_2 = GridFunction(fes)
    Exact_sgn22_2 = GridFunction(fes)
    Exact_sgn00_2.Set(sgnCF[0,0],definedon=mymesh.Boundaries('.*'))
    Exact_sgn01_2.Set(sgnCF[0,1],definedon=mymesh.Boundaries('.*'))
    Exact_sgn02_2.Set(sgnCF[0,2],definedon=mymesh.Boundaries('.*'))
    Exact_sgn11_2.Set(sgnCF[1,1],definedon=mymesh.Boundaries('.*'))
    Exact_sgn12_2.Set(sgnCF[1,2],definedon=mymesh.Boundaries('.*'))
    Exact_sgn22_2.Set(sgnCF[2,2],definedon=mymesh.Boundaries('.*'))
    np_vec_sets.append(Exact_sgn00_2.vec.FV().NumPy()[:dof_linear_num])
    np_vec_sets.append(Exact_sgn01_2.vec.FV().NumPy()[:dof_linear_num])
    np_vec_sets.append(Exact_sgn02_2.vec.FV().NumPy()[:dof_linear_num])
    np_vec_sets.append(Exact_sgn11_2.vec.FV().NumPy()[:dof_linear_num])
    np_vec_sets.append(Exact_sgn12_2.vec.FV().NumPy()[:dof_linear_num])
    np_vec_sets.append(Exact_sgn22_2.vec.FV().NumPy()[:dof_linear_num])
    # reset the mesh
    mymesh.Curve(1)
    return np_vec_sets

h1_err_fun = lambda xfun,mymesh: sqrt(Integrate(InnerProduct(grad(xfun).Trace(),grad(xfun).Trace())
                                        +InnerProduct((xfun),(xfun)), mymesh, 
                                        element_wise=False, definedon=mymesh.Boundaries('.*')))
l2_err_fun = lambda xfun,mymesh: sqrt(Integrate(InnerProduct((xfun),(xfun)), mymesh, 
                                        element_wise=False, definedon=mymesh.Boundaries('.*')))

class FEM1d():
    def __init__(self,order) -> None:
        self.order = order
        self.Info_BR_XDof()

    def Info_BR_XDof(self):
        if self.order == 1:
            self.BRFunc = [
                lambda x: 1-x,
                lambda x: x
            ]
            self.XDof = [
                0, 1
            ]
        elif self.order == 2:
            self.BRFunc = [
                lambda x: 2*(x-1)*(x-1/2),
                lambda x: -4*(x-1)*x,
                lambda x: 2*x*(x-1/2)
            ]
            self.XDof = [
                0, 1/2, 1
            ]
        elif self.order == 3:
            self.BRFunc = [
                lambda x: -9/2*(x-1/3)*(x-2/3)*(x-1),
                lambda x: 27/2*x*(x-2/3)*(x-1),
                lambda x: -27/2*x*(x-1/3)*(x-1),
                lambda x: 9/2*x*(x-1/3)*(x-2/3),
            ]
            self.XDof = [
                0, 1/3, 2/3, 1
            ]
        elif self.order == 4:
            self.BRFunc = [
                lambda x: 32/3*(x-1/4)*(x-2/4)*(x-3/4)*(x-1),
                lambda x: -128/3*x*(x-2/4)*(x-3/4)*(x-1),
                lambda x: 64*x*(x-1/4)*(x-3/4)*(x-1),
                lambda x: -128/3*x*(x-1/4)*(x-2/4)*(x-1),
                lambda x: 32/3*x*(x-1/4)*(x-2/4)*(x-3/4),
            ]
            self.XDof = [
                0, 1/4, 1/2, 3/4, 1
            ]

def cf_tensor_product(u,v):
    '''
        tensorproduct of list objects. 

        input: two lists, u,v
        output: ngsolve matrix CF uxv with rows (dim u), columms (dim v)
    '''
    dimu = len(u)
    dimv = len(v)
    # for ii in range(dim):
    #    for jj in range(dim): 的简写
    tensor_ele = [u[ii]*v[jj] for ii in range(dimu) for jj in range(dimv)]
    return CF(tuple(tensor_ele),dims=(dimu,dimv))
    
def get_id_cf(dim):
    if dim == 2:
        idCF = CF((x,y))
    elif dim == 3:
        idCF = CF((x,y,z))
    return idCF

def pos_transformer(pos_grid_func,dim=None):
    if dim is None:
        dim = pos_grid_func.dim
    else:
        assert(pos_grid_func.dim == dim)
    num = int(len(pos_grid_func.vec)/dim)
    coords = pos_grid_func.vec.Reshape(num).NumPy().copy()
    return coords.T
        
class SurfacehInterp():
    '''
        High order Lagrangian interpolation, 
        by interpolation on Gaussian quadrature nodes of 1 order higher 
        and L2 projection back
        1. build hInterp_Obj = SurfacehInterp(mesh,order)
        2. 创建曲面积分规则，以及能够得到对应点插值位置的有限元空间fesir
    '''
    def __init__(self,mesh,order) -> None:
        self.order = order
        self.mesh = mesh
        self.dim = self.mesh.dim
        self.fesir = IntegrationRuleSpaceSurface(self.mesh, order=self.order, 
                                                 definedon=self.mesh.Boundaries('.*'))
        self.fesir_vector = self.fesir**self.dim
        self.irs = self.fesir.GetIntegrationRules()
        self.fes = H1(self.mesh,order=self.order)
        self.fes_vector = VectorH1(self.mesh,order=self.order)
        self.L2u, self.L2v = self.fes.TnT()
        self.L2u_vector, self.L2v_vector = self.fes_vector.TnT()
        mass = BilinearForm(self.L2u*self.L2v*ds).Assemble().mat
        self.inv_mass = mass.Inverse(inverse="sparsecholesky")
        self.id_coef_func = get_id_cf(self.dim)
        self.rhs, self.rhsV = None, None
    
    def return_l2(self,vec,deform=None):
        '''
            vec: Interpolated values at quadrature nodes
            return: ndarray containing vector values of GridFunction on fes
        ''' 
        if deform is not None:
            self.mesh.SetDeformation(deform)
        quad_interp_func = GridFunction(self.fesir)
        quad_interp_func.vec.data = BaseVector(vec)
        res  = GridFunction(self.fes)
        rhs  = LinearForm(quad_interp_func * self.L2v * ds(intrules=self.irs))
        rhs.Assemble()
        res.vec.data = self.inv_mass * rhs.vec
        return copy.deepcopy(res.vec.FV().NumPy())


    def get_coords_quad(self):
        '''
            Get values of the identity(position) on quadrature nodes
        '''
        return self.get_values_quad(self.id_coef_func)

    def get_values_quad(self,coef_func):
        '''
            Get values of CF on quadrature nodes
        '''
        ndim = coef_func.dim
        cf_values = np.zeros((self.fesir.ndof,ndim))
        interp_fun = GridFunction(self.fesir)
        for ii in range(ndim):
            # for 1d CF, CF[0] is right
            interp_fun.Interpolate(coef_func[ii],self.mesh.Boundaries('.*')) 
            cf_values[:,ii] = interp_fun.vec.FV().NumPy().copy()
        return cf_values

class Ng_Matrix_Oper():
    def SparseMatPlus(A,B,a=1,b=1):
        '''输入sparsematrixd A, B, 返回aA+bB'''
        A_data = list(A.COO())
        B_data = list(B.COO())
        # A_data[2]为VectorD
        A_data[2] = A_data[2].NumPy().tolist()
        B_data[2] = B_data[2].NumPy().tolist()
        
        A_coo = Ng_Matrix_Oper.myCOO(A_data[0],A_data[1],A_data[2],*(A.shape),'scipy')
        B_coo = Ng_Matrix_Oper.myCOO(B_data[0],B_data[1],B_data[2],*(B.shape),'scipy')
        LinComb = A_coo*a+B_coo*b
        LinComb = LinComb.tocoo()
        val,row,col = map(list,[LinComb.data,LinComb.row,LinComb.col])
        res = Ng_Matrix_Oper.myCOO(row,col,val,*(A.shape),'ngsolve')
        return res

    def myCOO(row:list,col:list,val:list,m,n,tag):
        '''
            tag可选参数为 'scipy.sparse' or 'ngsolve.la.Spar...', row, col, val均为list
        '''
        if tag == 'scipy':
            return coo_matrix((val,(row,col)),(m,n))
        elif tag == 'ngsolve':
            return la.SparseMatrixd.CreateFromCOO(row,col,val,m,n)

    def Mat_Cf(A,B,indA,indB=None):
        '''
        比较A,B的indA和indB的部分是否相同，方法是作用在随机生成的向量上，
        A,B的类型可以是ngsolve.la.SparseMatrixd
        '''
        Asize = list(map(len,indA))
        if not B:
            ## B=None
            B = la.SparseMatrixd.CreateFromCOO([],[],[],*Asize)
        if not indB:
            indB = (list(range(B.shape[0])),list(range(B.shape[1])))
        if Asize != list(map(len,indB)):
            print("index不匹配")
        else:
            ## 根据列数来extend测试向量
            ncolB = B.shape[1]
            ncolA = A.shape[1]
            ## slice of row of B
            srowB, scolB = indB
            srowA, scolA = indA

            testv = np.random.rand(len(scolB))
            tmpB = np.zeros(ncolB)
            tmpB[scolB] = testv
            tmpA = np.zeros(ncolA)
            tmpA[scolA] = testv
            testvB = BaseVector(tmpB)
            testvA = BaseVector(tmpA)
            resB = BaseVector(B*testvB)
            resA = BaseVector(A*testvA)
            err = np.linalg.norm((resB.FV().NumPy())[srowB]-(resA.FV().NumPy())[srowA])
            if err<1e-10:
                print("误差为{}，矩阵相同".format(err))
            else:
                print("误差为{}，结果可能不同".format(err))
            print("测试向量L^2norm为{}".format(np.linalg.norm(testv)))

def NgMFSave(CaseName,BaseDirPath,mesh,funclist:list,func_name_list:list,data_dict:dict):
    '''
        Save Mesh: Ngsolve.comp.mesh, save ngmesh and curve order
        Save GF function: defindon_type, order, vector
    '''
    CaseDirPath = os.path.join(BaseDirPath,CaseName)
    if not os.path.exists(CaseDirPath):
        os.mkdir(CaseDirPath)
    CurveOrder = mesh.GetCurveOrder()
    mesh.ngmesh.Save(os.path.join(CaseDirPath,'Mesh.vol'))
    # save info of each GF: defindon_type, order, space_type 
    DataDict = data_dict
    DataDict['mesh_order'] = CurveOrder
    FuncDict = {}
    for GF,GF_name in zip(funclist,func_name_list):
        GF.Save(os.path.join(CaseDirPath,GF_name))
        order = GF.space.flags.ToDict()['order']
        FuncDict[GF_name] = ['bnd',int(order),GF.space.type]
    DataDict['Func'] = FuncDict
    np.save(file=os.path.join(CaseDirPath,'info.npy'),arr=DataDict,allow_pickle=True)

class LoadNgMF():
    def __init__(self,CaseName,BaseDirPath) -> None:
        self.FolderPath = os.path.join(BaseDirPath,CaseName)
        self.mesh_path = os.path.join(self.FolderPath,'Mesh.vol')
        self.info_path = os.path.join(self.FolderPath,'info.npy')
        self.Data_Dict = np.load(file=self.info_path,allow_pickle=True).item()
        self.mesh_order = self.Data_Dict['mesh_order']
    
    def RecoverMesh(self):
        # recover high order mesh
        mesh = Mesh(self.mesh_path)
        mesh.Curve(self.mesh_order)
        return mesh
        
    def RecoverGF(self,gf_in,gf_name:str):
        gf_in.Load(os.path.join(self.FolderPath,gf_name))
        return gf_in
