from ngsolve import *
from scipy.sparse import coo_matrix
import numpy as np
import os
from Package_MyCode import FO
from ngsolve.comp import IntegrationRuleSpaceSurface

def TensorProduct(u,v):
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
    
def GetIdCF(dim):
    if dim == 2:
        idCF = CF((x,y))
    elif dim == 3:
        idCF = CF((x,y,z))
    return idCF

class SurfacehInterp():
    '''
        High order Lagrangian interpolation, by interpolation on Gaussian quadrature nodes of 1 order higher and L2 projection back

    '''
    def __init__(self,mesh,order) -> None:
        self.order = order
        self.mesh = mesh
        self.dim = self.mesh.dim
        self.fesir = IntegrationRuleSpaceSurface(self.mesh, order=self.order, definedon=self.mesh.Boundaries('.*'))
        self.irs = self.fesir.GetIntegrationRules()
        self.fes = H1(self.mesh,order=self.order)
        self.fesV = VectorH1(self.mesh,order=self.order)
        self.L2u, self.L2v = self.fes.TnT()
        self.L2uV, self.L2vV = self.fesV.TnT()
        mass = BilinearForm(self.L2u*self.L2v*ds).Assemble().mat
        massV = BilinearForm(InnerProduct(self.L2uV,self.L2vV)*ds).Assemble().mat
        self.invmass = mass.Inverse(inverse="sparsecholesky")
        self.invmassV = massV.Inverse(inverse="sparsecholesky")
        RitzV = BilinearForm(InnerProduct(self.L2uV,self.L2vV)*ds
                            +InnerProduct(grad(self.L2uV).Trace(),grad(self.L2vV).Trace())*ds).Assemble().mat
        self.invRitzV = RitzV.Inverse(inverse="sparsecholesky")
        self.fesirV = self.fesir**self.dim
        self.idCF = GetIdCF(self.dim)
        self.rhs, self.rhsV = None, None

    def GetCoordsQuad(self):
        '''
            Get values of the identity(position) on quadrature nodes
        '''
        return self.GetValuesQuad(self.idCF)

    def GetValuesQuad(self,CF):
        '''
            Get values of CF on quadrature nodes
        '''
        ndim = CF.dim
        CF_values_np = np.zeros((self.fesir.ndof,ndim))
        Interp_fun = GridFunction(self.fesir)
        for ii in range(ndim):
            # for 1d CF, CF[0] is right
            Interp_fun.Interpolate(CF[ii],self.mesh.Boundaries('.*')) 
            CF_values_np[:,ii] = Interp_fun.vec.FV().NumPy().copy()
        return CF_values_np

    def Return2L2Byrhs(self,L2Func):
        ndim = L2Func.dim
        if ndim == 1:
            self.rhs.Assemble()
            L2Func.vec.data = self.invmass * self.rhs.vec
        else:
            self.rhsV.Assemble()
            L2Func.vec.data = self.invmassV * self.rhsV.vec

    def Return2L2(self,vec):
        '''
            vec: Interpolated values at quadrature nodes
            
            return: GridFunction on fes
        ''' 
        IntFunc = GridFunction(self.fesir)
        IntFunc.vec.data = BaseVector(vec)
        L2Func  = GridFunction(self.fes)
        rhs     = LinearForm(IntFunc*self.L2v*ds(intrules=self.irs))
        rhs.Assemble()
        L2Func.vec.data = self.invmass * rhs.vec
        return L2Func.vec.FV().NumPy().copy()

    def ReturnByRitzV(self,u,Du):
        '''
            u: CF function
        '''
        L2Func = GridFunction(self.fesV)
        rhs  = LinearForm(self.fesV)
        rhs += InnerProduct(u,self.L2vV)*ds + InnerProduct(Du,grad(self.L2vV).Trace())*ds
        rhs.Assemble()
        L2Func.vec.data = self.invRitzV * rhs.vec
        return L2Func.vec.FV().NumPy().copy()
    
    def ReturnByL2(self,u):
        '''
            u: CF function
        '''
        L2Func = GridFunction(self.fes)
        rhs  = LinearForm(self.fes)
        rhs += InnerProduct(u,self.L2v)*ds
        rhs.Assemble()
        L2Func.vec.data = self.invmass * rhs.vec
        return L2Func.vec.FV().NumPy().copy()

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

def Pos_Transformer(Pos_GF,dim=None):
    if dim is None:
        dim = Pos_GF.dim
    else:
        assert(Pos_GF.dim == dim)
    N = int(len(Pos_GF.vec)/dim)
    coords = Pos_GF.vec.Reshape(N).NumPy().copy()
    return coords.T
    
class NGMO():
    def SparseMatPlus(A,B,a=1,b=1):
        '''输入sparsematrixd A, B, 返回aA+bB'''
        A_data = list(A.COO())
        B_data = list(B.COO())
        # A_data[2]为VectorD
        A_data[2] = A_data[2].NumPy().tolist()
        B_data[2] = B_data[2].NumPy().tolist()
        
        A_coo = NGMO.myCOO(A_data[0],A_data[1],A_data[2],*(A.shape),'scipy')
        B_coo = NGMO.myCOO(B_data[0],B_data[1],B_data[2],*(B.shape),'scipy')
        LinComb = A_coo*a+B_coo*b
        LinComb = LinComb.tocoo()
        val,row,col = map(list,[LinComb.data,LinComb.row,LinComb.col])
        res = NGMO.myCOO(row,col,val,*(A.shape),'ngsolve')
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


