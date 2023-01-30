import numpy as np

class FEM1d():
    def __init__(self,order) -> None:
        self.order = order
        self.Info_BR_XDof()

    def Info_BR_XDof(self):
        if self.order == 1:
            self.BRFunc = [
                lambda x: 1-x,
                lambda x: x
            ]
            self.XDof = [
                0, 1
            ]
        elif self.order == 2:
            self.BRFunc = [
                lambda x: 2*(x-1)*(x-1/2),
                lambda x: -4*(x-1)*x,
                lambda x: 2*x*(x-1/2)
            ]
            self.XDof = [
                0, 1/2, 1
            ]
        elif self.order == 3:
            self.BRFunc = [
                lambda x: -9/2*(x-1/3)*(x-2/3)*(x-1),
                lambda x: 27/2*x*(x-2/3)*(x-1),
                lambda x: -27/2*x*(x-1/3)*(x-1),
                lambda x: 9/2*x*(x-1/3)*(x-2/3),
            ]
            self.XDof = [
                0, 1/3, 2/3, 1
            ]
        elif self.order == 4:
            self.BRFunc = [
                lambda x: 32/3*(x-1/4)*(x-2/4)*(x-3/4)*(x-1),
                lambda x: -128/3*x*(x-2/4)*(x-3/4)*(x-1),
                lambda x: 64*x*(x-1/4)*(x-3/4)*(x-1),
                lambda x: -128/3*x*(x-1/4)*(x-2/4)*(x-1),
                lambda x: 32/3*x*(x-1/4)*(x-2/4)*(x-3/4),
            ]
            self.XDof = [
                0, 1/4, 1/2, 3/4, 1
            ]
