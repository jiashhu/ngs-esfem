from Package_Geometry_Obj import EllipseMesh
import numpy as np
import matplotlib.pyplot as plt

Ellp_Obj = EllipseMesh(0,0,2,1,40)
Ellp_Obj.MeshPoly()
per_vertices = Ellp_Obj.vertices + 0.1*np.random.random(Ellp_Obj.vertices.shape)
proj_vertices = Ellp_Obj.ProjMap(per_vertices)
plt.plot(Ellp_Obj.vertices[:,0],Ellp_Obj.vertices[:,1],'r-')
plt.plot(per_vertices[::2,0],per_vertices[::2,1],'o')
plt.plot(proj_vertices[::2,0],proj_vertices[::2,1],'k^')
plt.gca().set_aspect('equal')
plt.show()