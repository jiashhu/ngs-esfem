import matplotlib.pyplot as plt
import numpy as np

from Package_MyCode import Pic

def myquiver(position_set,vector_set,Ndraw=20,scale=1,xlim=[-1,1],ylim=[-1,1]):
    '''
        变量1：位置：nx2 ndarray
        变量2：速度：nx2 ndarray
        Ndraw：个数 
    '''
    plt.plot(position_set[:Ndraw,0],position_set[:Ndraw,1],'o')
    plt.quiver(position_set[:Ndraw,0],position_set[:Ndraw,1],\
           vector_set[:Ndraw,0],vector_set[:Ndraw,1],scale=scale)
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.gca().set_aspect('equal', adjustable='box')
           
def triangle(tr1,tr2,ratio,text1,text2,pos_adjust):
    '''
        tr1,tr2,ratio = [4e-3,2e-1], 1e-3, 1/2
        triangle(tr1,tr2,ratio,'2','1',1.05)
                          (x1,y1)
                      *      *
                *            *
        (x0,y0) ---- ---- (x1,y0)
    '''
    x0,y0 = tr1 
    x1 = tr2
    y1 = y0*(x1/x0)**ratio
    plt.loglog([x0,x1],[y0,y0],'-k',linewidth=0.85)
    plt.loglog([x1,x1],[y0,y1],'-k',linewidth=0.85)
    plt.loglog([x1,x0],[y1,y0],'-k',linewidth=0.85)
    plt.text(np.sqrt(x0*x1),y0*pos_adjust,text1,fontsize=16)
    plt.text(x1*pos_adjust,np.sqrt(y0*y1),text2,fontsize=16)

class Fig():
    def __init__(self,figsize=(10,8)) -> None:
        self.fig = plt.figure(figsize=figsize)
        self.ax = self.fig.add_subplot(111)
        pass

# class MyLogLog(Fig):
#     def __init__(self, hset, figsize=(10, 8)) -> None:
#         super().__init__(figsize=figsize)
#         self.h = hset

#     def Add
