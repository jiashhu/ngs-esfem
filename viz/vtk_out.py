from global_utils import FO
from ngsolve import *

class VtkOut:
    def __init__(self, t_set_old, n_set, pathname, t_begin=0):
        '''
            Generate to be saved Tsets
        '''
        t_set = [t_i for t_i in t_set_old if t_i>t_begin]
        n_set = [n_set[ii] for ii in range(len(n_set)) if t_set_old[ii]>t_begin]
        if len(t_set)==1:
            self.t_sets = np.linspace(t_begin,t_set[0],n_set[0]+1)
        else:
            self.t_sets = np.zeros(1)
            for ii in range(len(t_set)):
                if ii == 0:
                    init,endt = t_begin,t_set[0]
                else:
                    init,endt = t_set[ii-1],t_set[ii]
                self.t_sets = np.append(self.t_sets,np.linspace(init,endt,n_set[ii]+1)[1:])

        self.index = 0
        self.fillnum = len(str(max(n_set)))
        self.pathname = pathname
        self.done = False
        if not os.path.exists(pathname):
            os.makedirs(pathname)
            mystr = '  vtk files are saved in {}  '.format(pathname)
            print('{:#^60}'.format(mystr))
        else:
            self.done = True
        self.rel_mapping = {}
        np.save(file=os.path.join(pathname,'Tset.npy'),arr=self.t_sets)
    
    def generate_mapping(self, file_path, load_t=0):
        '''
            LoadT = 0 version
        '''
        vtu_list = [fname for fname in FO.Get_File_List(file_path) if fname.endswith('vtu')]
        n_file = len(vtu_list)
        t_sets = [t for t in self.t_sets if t>=load_t]
        for ii, T in enumerate(t_sets):
            if ii<n_file:
                self.rel_mapping[vtu_list[ii].split('.vtu')[0]] = T
        np.save(file=os.path.join(file_path,'Rel_Mapping.npy'),arr=self.rel_mapping,allow_pickle=True)

    def output(self, mesh, function, tnow, command='', names=['sol'],subdivision=0):
        perform = False
        if command == 'do':
            perform = True
        elif tnow >= self.t_sets[self.index]:
            perform = True
        if perform:
            vtk = VTKOutput(ma=mesh,coefs=function,names=names,\
                            filename=os.path.join(self.pathname,('{}'.format(self.index)).zfill(self.fillnum)),\
                            subdivision=subdivision,legacy=False)
            vtk.Do()
            self.index += 1

    def Generate_PVD(self,pvd_name):
        # generate only one path
        FO.PVD_Generate(pvd_path=self.pathname,folder_path_set=[''], pvd_name=pvd_name)
        
class VtkOutBnd(VtkOut):
    def __init__(self, T_set, n_set, pathname, T_begin=0):
        super().__init__(T_set, n_set, pathname, T_begin)
    
    def output(self, mesh, function, tnow, command='', names=['sol']):
        perform = False
        if command == 'do':
            perform = True
        elif tnow >= self.t_sets[self.index]:
            perform = True
        if perform:
            local_name = ('{}'.format(self.index)).zfill(self.fillnum)
            file_path = os.path.join(self.pathname,local_name)
            vtk = VTKOutput(ma=mesh,coefs=function,names=names,\
                            filename=file_path,\
                            subdivision=0,legacy=False)
            self.rel_mapping[local_name] = tnow
            vtk.Do(vb=BND)
            self.index += 1
        return perform

class Vtk_out_1d(VtkOut):
    '''
        Save vtk file of 1d curve in plane (z=0) by coordinates

        Example: 
            # generate vtk output object
            T     = [1]
            n_vtk = [10]
            VTU_Path = './vtkfile'
            Coords2d0 = np.array([[np.cos(theta), np.sin(theta)] 
                        for theta in np.linspace(0,2*np.pi,20)[:-1]])
            vtk_Obj_1d = Vtk_out_1d(T,n_vtk,VTU_Path)
            for t in np.linspace(0,1,20):
                # radius changes from 2 to 1
                Coords2d = (2-t)*Coords2d0
                perform_res = vtk_Obj_1d.output(Coords2d,None,tnow=t)
            np.save(file=os.path.join(VTU_Path,'Rel_Mapping.npy'),arr=vtk_Obj_1d.Rel_Mapping,allow_pickle=True)
            vtk_Obj_1d.Generate_PVD('test.pvd')
    '''
    
    def __init__(self, T, n, pathname, T_begin=0):
        super().__init__(T, n, pathname, T_begin)
        self.VertsCoords = {}
    
    def output(self, vertices, pData_dict=None, tnow=None):
        '''
            输入一维曲线的逆时针序节点，输出vtk文件
        '''
        perform = False
        if pData_dict is None:
            pData_dict = {'zero': np.zeros(vertices.shape[0])}
        for key,val in pData_dict.items():
            pData_dict[key] = np.append(val,val[0])
        if self.index<len(self.t_sets) and (tnow is None or tnow >= self.t_sets[self.index]):
            # Positions of points that define lines
            npoints = len(vertices)
            x = np.zeros(npoints+1)
            y = np.zeros(npoints+1)
            z = np.zeros(npoints+1)

            # First line
            for ii, v in enumerate(vertices):
                x[ii], y[ii], z[ii] = vertices[ii,0], vertices[ii,1], 0.0
            x[-1], y[-1], z[-1] = vertices[0,0], vertices[0,1], 0.0

            # Connectivity of the lines
            pointsPerLine = np.zeros(1)
            pointsPerLine[0] = len(vertices)+1

            local_name = ('{}'.format(self.index)).zfill(self.fillnum)
            file_path = os.path.join(self.pathname,local_name)
            polyLinesToVTK(
                file_path,
                x,
                y,
                z,
                pointsPerLine=pointsPerLine,
                pointData=pData_dict
            )
            self.rel_mapping[local_name] = tnow
            self.index += 1
            perform = True
        return perform
        
    def LineSave(self, vertices, tnow=None):
        if self.index<len(self.t_sets) and (tnow is None or tnow >= self.t_sets[self.index]):
            self.VertsCoords[tnow] = (vertices)
            self.index += 1

    def SaveNpy(self,pvd_name):
        np.save(os.path.join(self.pathname,pvd_name), self.VertsCoords, allow_pickle=True)

class yxt_1d_out(VtkOut):
    def __init__(self, x_save_ref, T, n, pathname, T_begin=0):
        super().__init__(T, n, pathname, T_begin)
        self.Data_List = []
        self.t_List = []
        self.x_ref = x_save_ref
        Mesh_Obj = Mesh1d(min(x_save_ref),max(x_save_ref),len(x_save_ref)-1)
        self.mesh     = Mesh(Mesh_Obj.ngmesh)
        self.save_fem = H1(self.mesh)
        self.funvalue = GridFunction(self.save_fem)

    def output(self, mesh, function, tnow, command='', names=['sol'], subdivision=0):
        assert(len(function)==1)
        func = function[0]
        perform = False
        if command == 'do':
            perform = True
        elif tnow >= self.t_sets[self.index]:
            perform = True
        if perform:
            self.funvalue.Set(func)
            self.Data_List.append(self.funvalue.vec.FV().NumPy().copy())
            self.t_List.append(tnow)
            self.index += 1
    