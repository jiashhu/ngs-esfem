from ngsolve import *
import numpy as np

class DiscreteMesh:
    '''
        示例：
            DMesh_Obj = DiscreteMesh(Coords,dim=2,adim=3,ElVerInd=ElVerInd,EdVerInd=EdVerInd)
        输入：
            节点坐标Coords，网格dim，背景维度adim
            单元节点index矩阵ElVer，边节点index矩阵EdVer
            光是邻接矩阵不够，没有单元定向信息
        methods:
            UpdateCoords: update coords and bar vectors
            N_Area_El: element-wised normal and area
            WN_Ver: vertex_wised weighted normal
    '''
    def __init__(self,Coords,dim,adim,ElVerInd=None,EdVerInd=None) -> None:
        self.dim = dim
        self.adim = adim
        self.nv = Coords.shape[0]
        if (ElVerInd is None) and (EdVerInd is None):
            # 特用来处理曲线的顺序节点的情形
            ind_set = (np.append(range(self.nv),0)).reshape(-1,1)
            ElVerInd = np.concatenate((ind_set[:-1],ind_set[1:]),axis=1)
            EdVerInd = ElVerInd.copy()
        self.ElVerInd = ElVerInd
        self.EdVerInd = EdVerInd
        self.nel = self.ElVerInd.shape[0]
        self.ned = self.EdVerInd.shape[0]
        self.VerElInd, self.VerEdInd = self.RevInd()
        self.UpdateCoords(Coords=Coords)
        self.Area = 0

    def UpdateCoords(self,Coords):
        self.Coords = Coords
        self.barvec = np.array(
            [Coords[vind[1],:] - Coords[vind[0],:] for vind in self.EdVerInd]
        )

    def RevInd(self):
        # 计算每个顶点连接的单元与边的index
        ElVer = np.zeros((self.nel,self.nv))
        for ii, vind in enumerate(self.ElVerInd):
            ElVer[ii, vind] = 1
        # 顶点共享单元ind，个数不同，因此是arrayxarray
        VerElInd = np.array([line.nonzero()[0].tolist() for line in ElVer.transpose()],dtype=list)
        EdVer = np.zeros((self.ned,self.nv))
        for ii, vind in enumerate(self.EdVerInd):
            EdVer[ii, vind] = 1
        VerEdInd = np.array([line.nonzero()[0].tolist() for line in EdVer.transpose()],dtype=list)
        return VerElInd, VerEdInd

    def N_Area_El(self):
        # 计算每个单元的单位法向量以及单元面积以及边长
        N_El_matrix = np.zeros((self.nel,self.adim))
        Leng_matrix = np.zeros((self.ned,1))
        Area_matrix = np.zeros((self.nel,1))
        if self.adim == 2:
            vec2 = np.array([0,0,1])
            ECoords = np.concatenate((self.Coords,np.zeros((self.nv,1))),axis=1)
            for ii,ver_ind in enumerate(self.ElVerInd):
                assert(len(ver_ind)==2)
                bar = ECoords[ver_ind[1]] - ECoords[ver_ind[0]]
                n3d = np.cross(bar, vec2)
                N_El_matrix[ii,:] = n3d[:-1]/np.linalg.norm(n3d)
                Area_matrix[ii,:] = np.linalg.norm(n3d)
            for ii,ver_ind in enumerate(self.EdVerInd):
                bar = ECoords[ver_ind[1]] - ECoords[ver_ind[0]]
                Leng_matrix[ii,:] = np.linalg.norm(bar)
        elif self.adim == 3:
            # 三维中的二维曲面
            ECoords = self.Coords.copy()
            for ii,ver_ind in enumerate(self.ElVerInd):
                bar = ECoords[ver_ind[1:]] - ECoords[ver_ind[:-1]]
                n3d = np.cross(bar[0,:]/np.linalg.norm(bar[0,:]),
                            bar[1,:]/np.linalg.norm(bar[1,:]))
                N_El_matrix[ii,:] = n3d/np.linalg.norm(n3d)
                Area_matrix[ii,:] = np.linalg.norm(n3d)/2\
                    *np.linalg.norm(bar[0,:])*np.linalg.norm(bar[1,:])
            for ii,ver_ind in enumerate(self.EdVerInd):
                bar = ECoords[ver_ind[1]] - ECoords[ver_ind[0]]
                Leng_matrix[ii,:] = np.linalg.norm(bar)
        self.Area = sum(Area_matrix)
        return N_El_matrix, Leng_matrix, Area_matrix
    
    def WN_Ver(self):
        # 计算每个节点的加权法向量
        WN_Ver_matrix = np.zeros((self.nv,self.adim))
        N_El_matrix, Leng_matrix, Area_matrix = self.N_Area_El()
        for ii in range(self.nv):
            # 获取顶点所在的单元index
            adj_el_ind = self.VerElInd[ii]
            tmp = (N_El_matrix[adj_el_ind,:]*Area_matrix[adj_el_ind,:]).sum(axis=0)
            WN_Ver_matrix[ii,:] = tmp/np.linalg.norm(tmp)
        self.WN_Ver_matrix = WN_Ver_matrix

    def MeshQuality(self):
        _, Leng_matrix, Area_matrix = self.N_Area_El()
        # 最大面积与最小面积之比
        Q_Area = max(Area_matrix)/min(Area_matrix)
        # 最大边长与最小边长之比
        Q_Leng = max(Leng_matrix)/min(Leng_matrix)
        return Q_Area, Q_Leng