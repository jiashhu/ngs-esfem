import numpy as np

def mysolve(tmp1,aj,bj):
    ## Root of denomina = x-tmp1*(x+aj*bj)**2/4
    ## a0*x**2+a1*x+a2
    a = -tmp1/4
    b = 1-tmp1*aj*bj/2
    c = -(aj*bj)**2*tmp1/4
    Delta = b**2-4*a*c
    root = [(-b-np.sqrt(Delta))/(2*a),(-b+np.sqrt(Delta))/(2*a)]
    return root

def bestappro(n,aj,bj):
    '''
        Chebychev best rational approximation of (2^n,2^n-1) on (aj,bj),
        Return pole and weight
    '''
    aj1 = np.sqrt(aj*bj)
    bj1 = (aj+bj)/2
    aj2 = np.sqrt(aj1*bj1)
    bj2 = (aj1+bj1)/2
    bj3 = (aj2+bj2)/2

    if n == 1:
        pole = np.array([-aj1**2])
        weight = np.array([-(aj1**2*aj2**2)/bj3, 1/(4*bj3), (4*aj2**2+aj1**2)/(4*bj3)])
    else:
        tmppole, tmpweight = bestappro(n-1,1/bj1,1/aj1)
        L = len(tmppole)
        # 一个pole变成两个pole
        pole = np.zeros(2*L+1)
        weight = np.zeros(2*L+3)
        # x term --- end-1指标，出来一个pole和一个常数项
        pole[0] = -aj*bj
        # x term 的系数
        tmp = tmpweight[-2]
        weight[0] = -2*aj*bj*tmp
        weight[-1] = weight[-1]+2*tmp
        # 常数项的变换，end指标，出来x和常数项
        tmp = tmpweight[-1]
        weight[-2] = weight[-2] + tmp/2
        weight[-1] = weight[-1] + tmp*aj*bj/2
        # 各个pole项的变化,原来一个pole会分解成两个pole，所以
        # 第ii个pole会对这一层的第2*ii，2*ii+1的pole有贡献，这里先不考虑重根。
        for ii in range(L):
            tmp1 = tmppole[ii]
            tmp2 = tmpweight[ii]
            pole_sin = mysolve(tmp1,aj,bj)
            target = lambda x: -4/tmp1*(x+aj*bj)/2*x
            weight1 = target(pole_sin[0])/(pole_sin[0]-pole_sin[1])
            weight2 = target(pole_sin[1])/(pole_sin[1]-pole_sin[0])
            pole[2*ii+1] = pole_sin[0]
            pole[2*ii+2] = pole_sin[1]
            weight[2*ii+1] = weight1*tmp2/tmp1
            weight[2*ii+2] = weight2*tmp2/tmp1
            # x term -1/tmp1*(x+aj*bj)/2
            weight[-2] = weight[-2] - (tmp2/tmp1)/2
            # constant
            weight[-1] = weight[-1] - 2*tmp2/tmp1**2 - (tmp2/tmp1)/2*aj*bj
    return pole, weight
