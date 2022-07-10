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
from .LapVMCF import (
    LapVDNMCF_v2
)

from .LapVWillmore import (
    LapVWillMore,
    LapVSD
)

from .BGN import (
    BGN,
    BGN_MCF,
    BGN_WM
)

from ._Applications.MCFFlow import (
    DumbbellLapVMCF,
    DumbbellLapVMCF_N_modified,
    Dumbbell_BGN_MCF,
    Dumbbell_BGN_MCF_Mod,
    DumbbellLapVMCF_v_0531,
    DumbbellLapVMCF_v_0601,
    DumbbellLapVMCF_v_0602,
    DumbbellLapVMCF_v_0603,
    DumbbellLapVMCF_v_0604,
    DumbbellLapVMCF_v_0604_impv,
    DumbbellLapVMCF_v_0605,
    DumbbellLapVMCF_v_0607,
    LapVMCF_v_0605
)

from ._Applications.WMFlow import (
    FlowerLapVSD,
    FlowerLapVSD_Implicit,
    SphereLapVWillmore,
    RBCLapVWillmore,
    RBCLapVWM_ImplicitVz_v_0530,
    RBCLapVWM_ImplicitVz_v_0531,
    FlowerLapVSD_v_0531,
    FlowerLapVSD_v_0601,
    FlowerLapVSD_v_0602,
    RBCLapVWM_ImplicitVz_v_0602,
    FlowerLapVSD_v_0603
)

__all__ = [
    'DumbbellLapVMCF',
    'DumbbellLapVMCF_N_modified',
    'Dumbbell_BGN_MCF',
    'Dumbbell_BGN_MCF_Mod',
    'FlowerLapVSD',
    'FlowerLapVSD_Implicit',
    'SphereLapVWillmore',
    'RBCLapVWillmore',
    'RBCLapVWM_ImplicitVz_v_0530',
    'RBCLapVWM_ImplicitVz_v_0531',
    'FlowerLapVSD_v_0531',
    'DumbbellLapVMCF_v_0531',
    'FlowerLapVSD_v_0601',
    'DumbbellLapVMCF_v_0601',
    'DumbbellLapVMCF_v_0602',
    'FlowerLapVSD_v_0602',
    'RBCLapVWM_ImplicitVz_v_0602',
    'FlowerLapVSD_v_0603',
    'DumbbellLapVMCF_v_0603',
    'DumbbellLapVMCF_v_0604',
    'DumbbellLapVMCF_v_0604_impv',
    'DumbbellLapVMCF_v_0605',
    'DumbbellLapVMCF_v_0607',
    'LapVMCF_v_0605'
]