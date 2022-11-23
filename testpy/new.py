import numpy as np
from Package_ALE_Geometry import Vtk_out_1d
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
    perform_res = vtk_Obj_1d.Output(Coords2d,None,tnow=t)
vtk_Obj_1d.Generate_PVD('test.pvd')