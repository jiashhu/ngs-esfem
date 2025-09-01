'''
    convention：A = nabla n，从而圆周的平均曲率H为正
    类的继承关系：
    基本的ESFEM的class有：
        LapVDNMCF: Laplace v对应的平均曲率流
        LapVWillMore: Laplace v对应的Willmore flow
    特定几何的类：继承基本的ESFEM的类：
        初始化给定mymesh
        重写IniH方法：即设定演化方程所需要的初值，初始的position，曲率，法向量等
        重写ErrL2H1方法：这部分似乎可以写成ESFEM的类方法，然后在每个example中给出精确解的表达式
'''
from .mcf_mdr import *
from .willmore_mdr import *
from .bgn import *
from .dziuk import *
from ._applications.mcf_flow import *
from ._applications.willmore_flow import *
from .utils import *