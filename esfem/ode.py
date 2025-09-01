import numpy as np
from collections import deque

class BDF():
    '''
        BE:     y_n+1 - y_n = h f_n+1,
        BDF2:   y_n+2 - y_n+1 = 1/3*(y_n+1-y_n) + 2/3*h*f_n+2,
        BDF3:   y_n+3 - y_n+2 = 7/11*(y_n+2-y_n+1) - 2/11*(y_n+1-y_n) + 6/11*h*f_n+3
    '''
    def __init__(self,order = 1) -> None:
        self.nstored = order-1
        self.hist = deque([],self.nstored)

    def Stepping(self,hfnew):
        '''
            hfnew: increment, e.g., y_n+1 - y_n
        '''
        tmpn = len(self.hist)
        if tmpn < self.nstored:
            ## Initialize historic info -- exact or modified
            res = self.IntRule(tmpn,hfnew)
            self.hist.append(res)
        else:
            ## Update historic info
            res = self.IntRule(self.nstored,hfnew)
            self.hist.append(res)
        return res

    def IntRule(self,nstored,hfnew):
        '''
            Integrate rule depends on the num of stored data
        '''
        if nstored == 0:
            res = hfnew
        elif nstored == 1:
            res = 1/3*self.hist[0] + 2/3*hfnew
        elif nstored == 2:
            res = 7/11*self.hist[1] - 2/11*self.hist[0] + 6/11*hfnew
        return res