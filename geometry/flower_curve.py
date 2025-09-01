from scipy.sparse import coo_matrix
import math

class FlowerCurveMesh:
    def __init__(self, box, nnode, dt, symm, a=0.65, b=7):
        self.box = box
        self.symm = symm
        self.dt = dt
        # Flower curve parameterization --  symbolic version
        Par_x = (a*sym.sin(b*phi)+1)*sym.cos(phi)
        Par_y = (a*sym.sin(b*phi)+1)*sym.sin(phi)
        Par_v = Par_x*N.x + Par_y*N.y
        self.Par_func = [sym.lambdify(phi,fun_i) for fun_i in Par_v.to_matrix(N)]
        self.Fscale = 1.05
        if symm:
            theta = np.linspace(box[0],box[1],nnode+1)
            dsize = (box[1]-box[0])/nnode
            self.theta = np.append(theta-dsize/2,box[1]+dsize/2)
            self.nnode = nnode+2
            self.nvs = nnode
        else:
            self.theta = np.linspace(box[0],box[1],nnode)
            self.nnode = nnode
            self.nvs = nnode-2
        
        self.p = np.zeros((self.nnode,2))
        self.ComputParam()
        indp = list(range(self.nnode))
        self.bars = np.transpose(np.array([indp[1:],indp[:-1]]))
        self.barvec = np.zeros((self.nnode-1,2)) 
        ## Normal of visual points
        self.nvec = np.zeros((self.nvs,2))
        self.Fvis = np.zeros((self.nvs,2))
    
    def ComputParam(self):
        phi = self.theta
        self.p[:,0] = self.Par_func[0](phi)
        self.p[:,1] = self.Par_func[1](phi)
        
    def ComputBar(self):
        self.barvec = self.p[self.bars[:,0],:]-self.p[self.bars[:,1],:]
    
    def WeightedNorm(self):
        nvec = np.zeros(self.barvec.shape)
        Norm = np.sqrt((self.barvec**2).sum(axis=1))
        nvec[:,0] = self.barvec[:,1]/Norm
        nvec[:,1] = -self.barvec[:,0]/Norm
        Norm = Norm.reshape(-1,1)
        self.nvec = nvec[:-1]*Norm[:-1]+nvec[1:]*Norm[1:]
        for i in range(self.nvec.shape[0]):
            self.nvec[i] /= np.linalg.norm(self.nvec[i])
        
    def Constrain(self):
        '''
            根据theta角度以及单参数表达式进行拉回
            找到 (-pi/14,pi/14) 内部的点的坐标（首尾根据对称性再次确定），按照幅角排序
            断言第二个点比 -pi/14 更大，如果第一个点小于，则取两者平均
            再调整首尾端点取关于box对称的点
        '''
        pvis = self.p[1:-1]
        # By theta
        newtheta = np.sort([math.atan2(pj[1],pj[0]) for pj in pvis])
        if self.symm:
            if newtheta[1]<self.box[0] or newtheta[-2]>self.box[1]:
                print('wrong')
            elif newtheta[0]<self.box[0]:
                newtheta[0] = (self.box[0]+newtheta[1])/2
            elif newtheta[-1]>self.box[1]:
                newtheta[-1] = (self.box[1]+newtheta[-2])/2

            self.theta[0] = 2*self.box[0]-newtheta[0]
            self.theta[1:-1] = newtheta
            self.theta[-1] = 2*self.box[1]-newtheta[-1]
        else:
            self.theta[1:-1] = newtheta
        self.ComputParam()
        
    def Force(self):
        barvec = self.barvec
        bars = self.bars
        L = np.linalg.norm(barvec,axis=1)
        L = L.reshape(-1,1)
        hbars = np.ones((self.nnode-1,1))
        L0 = hbars*self.Fscale*np.sqrt((L**2).sum()/(hbars**2).sum())
        F = np.array([[(L0-L)[ii][0]] for ii in range(len(L))])
        Fvec=np.kron(F/L,[1,1])*barvec
        row = bars[:,[0,0,1,1]].flatten()
        column = np.kron(np.ones(F.shape),[0,1,0,1]).flatten()
        val = np.concatenate((Fvec,-Fvec),axis=1).flatten()
        Ftot = coo_matrix((val,(row,column)),shape=(self.nnode,2)).todense()
        
        Ftot = np.array(Ftot)
        
        self.Fvis = Ftot[1:-1]
        ## Projection to Tangent Part
        for i,F in enumerate(self.Fvis):
            self.Fvis[i] = F-(F*self.nvec[i]).sum()*self.nvec[i]
        
    def Deformation(self):
        self.p[1:-1] += self.dt*self.Fvis
        self.Constrain()
    
    def SymmetryMesh(self):
        '''
            返回p:nx2坐标矩阵以及PhiTot:幅角
        '''
        if self.symm:
            # res = np.append(self.box[0], self.theta[1:-1])
            # res = np.append(res, self.box[1])
            # res = np.append(res, np.sort(2*self.box[1]-self.theta[1:-1]))
            res = np.append(self.theta[1:-1], np.sort(2*self.box[1]-self.theta[1:-1]))
        else:
            res = np.concatenate((self.theta[:], np.sort(2*self.box[1]-self.theta[1:-1])))
        PhiTot = []
        for i in range(7):
            PhiTot = np.append(PhiTot, res+i*2*(self.box[1]-self.box[0]))
        p = np.zeros((len(PhiTot),2))
        phi = PhiTot
        p[:,0] = self.Par_func[0](phi)
        p[:,1] = self.Par_func[1](phi)
        return p,PhiTot


def Coord1dFlower(n_half_blade,num_blade=7,radparam=0.65):
    '''
        生成花型网格，输入参数为对称的半个叶片曲线上的采样点个数。输出 p: nx2 坐标矩阵； PhiTot: 极角坐标
    '''
    EquivMesh = FlowerCurveMesh([-np.pi/(2*num_blade),np.pi/(2*num_blade)],n_half_blade,0.02,True, a=radparam, b=num_blade)
    while True:
        EquivMesh.ComputBar()
        EquivMesh.WeightedNorm()
        EquivMesh.Force()
        if np.var(np.linalg.norm((EquivMesh.barvec),axis=1))>1e-5:
            EquivMesh.Deformation()
        else: 
            break
    p,PhiTot = EquivMesh.SymmetryMesh()
    return p,PhiTot


def Mesh_Info_Parse(mesh):
    ElVerInd = []
    EdVerInd = []
    for el in mesh.ngmesh.Elements2D():
        # v.nr 从1开始计数
        ElVerInd.append([v.nr-1 for v in el.vertices])
    for ed in mesh.edges:
        # v.nr 从0开始计数（ngmesh与ngsolve.Mesh之间的区别）
        EdVerInd.append([v.nr for v in ed.vertices])
    return np.array(ElVerInd), np.array(EdVerInd)



class D1_FlowerCurve(Param1dCurve):
    '''
        一维的花型曲线：一种特殊的一维参数曲线
            输入：
                确定花型的解析表达的半径参数a与周期参数b
                确定花型离散网格的半单周期网格点数nnode
            解析表示：
                x = (a*sin(b*phi)+1)*cos(phi)
                y = (a*sin(b*phi)+1)*sin(phi)
            方法：
                计算参数化的数值函数,参数化的符号表达str,法向量和曲率符号表达str
                生成坐标决定的参数化曲线
    ''' 
    def __init__(self,params:list=[0.65,7],nnode:int=12) -> None:
        '''
            ParSymStr: list of str; 
            ParFunc: func; 
            DMesh: DiscreteCurve;
            Projection 
        '''
        self.params = params
        a,b = self.params
        # 解析的参数化设置
        self.ParSymStr = ['({}*sym.sin({}*phi)+1)*sym.cos(phi)'.format(a,b),\
        '({}*sym.sin({}*phi)+1)*sym.sin(phi)'.format(a,b)]
        # 生成参数化的符号表示以及Numpy表达式
        Par_Np = [component.replace('sym','np') for component in self.ParSymStr]
        # position的数值计算函数 
        ## 输入必须是ndarray，并且是行向量，决不能是list
        self.ParFunc = lambda phi: np.append(eval(Par_Np[0])[:,None], eval(Par_Np[1])[:,None], axis=1)
        # 继承参数化曲线的法向投影-曲率-法向量的string表达
        super().__init__([eval(component) for component in self.ParSymStr])
        self.DMesh = self.Generate1dMesh(symm=True, nnode=nnode)

    def Generate1dMesh(self,symm=True,dt=0.02,nnode=6) -> DiscreteMesh:
        freq = self.params[1]
        semi_period = np.pi/freq
        # 对应最小与最大半径
        symm_box = [-semi_period/2, semi_period/2]
        if symm:
            theta = np.linspace(symm_box[0],symm_box[1],nnode+1)
            dsize = semi_period/nnode
            # 最小与最大半径之间的节点theta
            theta = np.append(theta-dsize/2,symm_box[1]+dsize/2)[1:-1]
            # 对称延拓到单周期
            Etheta_per = np.array(
                [theta, np.sort(semi_period-theta)]
            )
            # 延拓到所有周期
            Etheta = []
            for ii in range(freq):
                Etheta = np.append(Etheta, Etheta_per+ii*2*semi_period)
        Coords = self.ParFunc(phi = Etheta)
        nv = Coords.shape[0]
        ElVerMatrix = np.array(
            [[ii, ii+1] for ii in range(nv)]
        )
        ElVerMatrix[-1,-1] = 0
        EdVerMatrix = ElVerMatrix.copy()
        return DiscreteMesh(Coords,1,2,ElVerMatrix,EdVerMatrix)



class ReParam_BarMod:
    '''
        坐标调整，根据网格的对称性可以选择对称选项来处理自由的边界点
        最重要的属性：
            法向量：切向速度调整
            Force：确定速度大小
    '''
    def __init__(self,ParDisCurve:D1_FlowerCurve,Symm=True,dt=0.02) -> None:
        self.dt = dt
        self.ParDisCurve = ParDisCurve
        self.ParFunc = ParDisCurve.ParFunc
        self.ParSymStr = [eval(component) for component in ParDisCurve.ParSymStr]
        self.Fscale = 1.05
        if Symm:
            # 1d对称情形：首尾是内点对称出来的，内点个数减2
            self.DMesh = ParDisCurve.DMesh
            self.Fvis = np.zeros((self.DMesh.nv,2))
        else:
            pass

    def ModifiedMesh(self):
        '''
            调整网格，首先计算 
        '''
        newcoords = self.DMesh.Coords.copy()
        while True:
            # 更新加权法向量，计算切向的force
            self.DMesh.WN_Ver()
            self.Force()
            # 等距网格的停机准则
            if np.var(np.linalg.norm((self.DMesh.barvec),axis=1))>1e-4:
                self.CheckSymm(self.DMesh.Coords[-1,:],self.DMesh.Coords[0,:])
                newcoords += self.dt*self.Fvis
                newcoords = self.Constrain(newcoords)
            else:
                break
            self.DMesh.UpdateCoords(newcoords)
            Q_Area, Q_Leng = self.DMesh.MeshQuality()
            print('Mesh Quality: {}, {}'.format(Q_Area, Q_Leng))
        
    def Constrain(self,coords):
        theta = self.ParDisCurve.Get_Param_Projection(coords, threshold=1e-5)
        newcoords = self.ParFunc(theta.flatten())
        return newcoords
        
    def CheckSymm(self, vec1: np.ndarray, vec2: np.ndarray):
        proximal = np.array(
            [np.cos(-np.pi/14), np.sin(-np.pi/14)]
        )
        v1normal = np.dot(vec1, proximal)
        v2normal = np.dot(vec2, proximal)
        print('normal diff is {}'.format(v1normal - v2normal))
        diff_tan = np.linalg.norm(
            vec2 - v2normal * proximal + (vec1 - v1normal * proximal)
        )
        print('tangential diff is {}'.format(diff_tan))

    def Force(self):
        '''
            前置：计算加权法向量 
        '''
        _, Leng_matrix, _ = self.DMesh.N_Area_El()
        hbars = np.ones((self.DMesh.ned,1)) # edge的权重
        L0 = hbars*self.Fscale*np.sqrt((Leng_matrix**2).sum()/(hbars**2).sum())
        F = np.array([[(L0-Leng_matrix)[ii][0]] for ii in range(self.DMesh.ned)])
        # barforce, in direction of barvec
        Fvec=np.kron(F/Leng_matrix,[1,1])*self.DMesh.barvec
        # 若边的端点为i->j,则Ftot的[jjii]行,[0,1,0,1]列取值[F1,F2,-F1,-F2]
        # 即j点（终点）的Force为F,i点（起点）的Force为-F,能保持轴对称性质
        row = (self.DMesh.EdVerInd[:,[1,1,0,0]]).flatten()
        column = np.kron(np.ones(F.shape),[0,1,0,1]).flatten()
        val = np.concatenate((Fvec,-Fvec),axis=1).flatten()
        # i-th row -- the force of i-th vertex
        self.Ftot = np.array(
            coo_matrix((val,(row,column)),shape=(self.DMesh.nv,2)).todense()
        )
        ## Projection to Tangent Part
        for i,F in enumerate(self.Ftot):
            nvec = self.DMesh.WN_Ver_matrix[i]
            self.Fvis[i] = F # -(F*nvec).sum()*nvec
