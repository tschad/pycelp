
from numba import njit
import numpy as np

fact = np.zeros(101,dtype=np.float)
fact[0] = 1.
for i in range(1,101): fact[i] = fact[i-1] * np.float(i)

@njit(cache=True,fastmath=True)
def w3js(j1,j2,j3,m1,m2,m3):
    """
    This function calculates the 3-j symbol
    J_i and M_i have to be twice the actual value of J and M
    originally from hazel maths.f90 fact is array of read(kind=8)
    factorial numbers 0:301

    See Appendix 1 of Landi Degl'Innocenti & Landolfi
    Also equation 2.19 and 2.22
    """
    if (m1+m2+m3 != 0):
        return 0.0
    ia = j1 + j2
    if (j3 > ia):
        return 0.0
    ib = j1 - j2
    if (j3 < abs(ib)):
        return 0.0
    if (abs(m1) > j1):
        return 0.0
    if (abs(m2) > j2):
        return 0.0
    if (abs(m3) > j3):
        return 0.0
    jsum = j3 + ia
    ic = j1 - m1
    id = j2 - m2
    ie = j3 - j2 + m1
    im = j3 - j1 - m2
    zmin = max([0,-ie,-im])
    ig = ia - j3
    ih = j2 + m2
    zmax = min([ig,ih,ic])
    cc = 0.0
    for z in range(zmin,zmax+2,2):
       denom = fact[z//2]*fact[(ig-z)//2]*fact[(ic-z)//2]*fact[(ih-z)//2]*fact[(ie+z)//2]*fact[(im+z)//2]
       if (z%4 != 0):
           cc = cc - 1.0/denom
       else:
           cc = cc + 1.0/denom

    cc1 = fact[ig//2]*fact[(j3+ib)//2]*fact[(j3-ib)//2]/fact[(jsum+2)//2]
    cc2 = fact[(j1+m1)//2]*fact[ic//2]*fact[ih//2]*fact[id//2]*fact[(j3-m3)//2]*fact[(j3+m3)//2]
    cc = cc * (cc1*cc2)**(0.5)

    if ((ib-m3)%4 != 0):
        cc = -cc
    if (abs(cc) < 1.e-8):
        return 0.0
    return cc

@njit(cache=True,fastmath=True)
def w6js(j1,j2,j3,l1,l2,l3):
    """
    This function calculates the 6-j symbol
    J_i and M_i have to be twice the actual value of J and M
    originally from hazel maths.f90
    """

    ##  Equation 2.36a -- TSCHAD 12 March 2021
    #if (j3 == 0):
    #    if (j1 != j2): return 0.0
    #    if (l1 != l2): return 0.0
    #    return (-1.)**(0.5*(j1+l2+l3)) / ((j1+1.)*(l1+1.))**0.5

    ## Equation 2.36d
    ## runs into issue for j1 = 0
    #if (j3 == 2) and (j1 == j2) and (l1 == l2):
    #    a = j1/2.
    #    b = l1/2.
    #    c = l3/2.
    #    return (-1)**(a+b+c+1)*0.5*(a*(a+1) + b*(b+1) - c*(c+1)) / (a*(a+1)*(2*a+1)*b*(b+1)*(2*b+1))**0.5

    ia = j1 + j2
    if (ia < j3): return 0.0
    ib = j1 - j2
    if (abs(ib) > j3): return 0.0
    ic = j1 + l2
    if (ic < l3): return 0.0
    id = j1 - l2
    if (abs(id) > l3): return 0.0
    ie = l1 + j2
    if (ie < l3): return 0.0
    iif = l1 - j2
    if (abs(iif) > l3): return 0.0
    ig = l1 + l2
    if (ig < j3): return 0.0
    ih = l1 - l2
    if (abs(ih) > j3): return 0.0

    sum1=ia + j3
    sum2=ic + l3
    sum3=ie + l3
    sum4=ig + j3
    wmin = max([sum1, sum2, sum3, sum4])
    ii = ia + ig
    ij = j2 + j3 + l2 + l3
    ik = j3 + j1 + l3 + l1
    wmax = min([ii,ij,ik])
    omega = 0.0
    for w in range(wmin,wmax+2,2):
       denom = fact[(w-sum1)//2]*fact[(w-sum2)//2]*fact[(w-sum3)//2]* \
               fact[(w-sum4)//2]*fact[(ii-w)//2]*fact[(ij-w)//2]* \
               fact[(ik-w)//2]
       if (w%4 != 0):
           omega = omega - fact[w//2+1] / denom
       else:
           omega = omega + fact[w//2+1] / denom

    theta1 = fact[(ia-j3)//2]*fact[(j3+ib)//2]*fact[(j3-ib)//2]/fact[sum1//2+1]
    theta2 = fact[(ic-l3)//2]*fact[(l3+id)//2]*fact[(l3-id)//2]/fact[sum2//2+1]
    theta3 = fact[(ie-l3)//2]*fact[(l3+iif)//2]*fact[(l3-iif)//2]/fact[sum3//2+1]
    theta4 = fact[(ig-j3)//2]*fact[(j3+ih)//2]*fact[(j3-ih)//2]/fact[sum4//2+1]
    theta = theta1 * theta2 * theta3 * theta4
    w6js = omega * theta**(0.5)
    if (abs(w6js) < 1.e-8):
        return 0.0
    return w6js

@njit(cache=True,fastmath=True)
def w9js(j1,j2,j3,j4,j5,j6,j7,j8,j9):
    """
    This function calculates the 9-j symbol
    J_i and M_i have to be twice the actual value of J and M
    originally from hazel maths.f90
    """
    kmin = abs(j1-j9)
    kmax = j1 + j9
    i = abs(j4-j8)
    if (i > kmin): kmin = i
    i = j4 + j8
    if (i < kmax): kmax = i
    i = abs(j2-j6)
    if (i > kmin): kmin = i
    i = j2 + j6
    if (i < kmax): kmax = i
    x = 0.0
    if (kmin%2 != 0):
       s = -1.0
    else:
       s = 1.0
    for k in range(kmin, kmax+2, 2):
       x1 = w6js(j1,j9,k,j8,j4,j7)
       x2 = w6js(j2,j6,k,j4,j8,j5)
       x3 = w6js(j1,j9,k,j6,j2,j3)
       x = x + s*x1*x2*x3*(np.float(k)+1.)
    return x
