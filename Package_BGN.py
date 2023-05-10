import numpy as np
import netgen.meshing as ngm
from ngsolve import *


def CircleSegs(index_set):
    '''
        输入：index_set为ndarray
    '''
    tmp = index_set.reshape(len(index_set),1)
    tmp1 = tmp.copy()
    tmp1[:-1] = tmp[1:]
    tmp1[-1] = tmp[0]
    v = np.hstack((tmp,tmp1))
    return v

class BGN1d_TangentialMod():
    '''
        对给定的逆时针排列的节点构成的多边形使用切向速度BGN方法演化的结果
    '''
    
    def __init__(self,vertices) -> None:
        self.vertices = vertices
        self.mesh = self.GeneratingMesh(vertices)
        pass

    def GeneratingMesh(self):
        ngmesh = ngm.Mesh(dim=2)
        pnums = []
        for v in self.vertices:
            pnums.append(ngmesh.Add(ngm.MeshPoint(ngm.Pnt(*v,0))))
        ## 添加边界
        idx_bnd = ngmesh.AddRegion("bnd", dim=1)
        segments = CircleSegs(np.arange(len(self.vertices)))
        for bnd in segments:
            ngmesh.Add(ngm.Element1D([pnums[ii] for ii in bnd],index=idx_bnd))
        return Mesh(ngmesh)

class BGN_V_Lap():
    '''
        为了探究BGN的连续格式，将BGN中的速度描述方式改成 Laplace v = kappa n
        整个格式为：
        X = Xn + dt*v
                                  (chi, v\cdot n)    = (chi, v0\cdot n)
        (kappa n\cdot eta)   +  (nabla v, nabla eta) = 0

        kappa is scalar and v is vectorial.
    '''
    
    def __init__(self,mesh,order=1):
        self.mesh = mesh
        self.fes = H1(mesh,order=order)
        self.fesV = VectorH1(mesh,order=order)
        self.mixfes = self.fes*self.fesV

    def Weak_MCF():
        pass

