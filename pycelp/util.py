"""
This module provides main utility functions needed by pycelp.
"""

from numba import njit,jit
import numpy as np
import pycelp.wigner as wigner

## ALLEN -- lambdaCLV in Angstrom
## better to keep it as a global variable since NUMBA will know its immutable?
lambdaCLV = 1.e4* np.asarray([0.20,0.22,0.245,0.265,0.28,0.30,0.32,0.35,
                0.37,0.38,0.40,0.45,0.50,0.55,0.60,0.80,1.0,
                1.5,2.0,3.0,5.0,10.0])
uCLV = np.asarray([0.12,-1.3,-0.1,-0.1,0.38,0.74,0.88,0.98,1.03,0.92,0.91,
                    0.99,0.97,0.93,0.88,0.73,0.64,0.57,0.48,0.35,0.22,0.15])
vCLV = np.asarray([0.33,1.6,0.85,0.90,0.57, 0.20, 0.03,-0.1,-0.16,-0.05,
                    -0.05,-0.17,-0.22,-0.23,-0.23,-0.22,-0.20,-0.21,-0.18,
                    -0.12,-0.07,-0.07])

def vac2air(wvlamb):
    """ Converts wavelengths from vacuum to air-equivalent """
    ww = (wvlamb < 2000.)
    convl = np.zeros_like(wvlamb)
    convl[ww] = wvlamb[ww]
    ww = (wvlamb >= 2000.)
    convl[ww] =  wvlamb[ww]/(1. + 2.735182e-4 + 131.4182/(wvlamb[ww]**2) + 2.76249e8/(wvlamb[ww]**4))
    return convl

@njit(cache=True)
def newint(x,a,b,u,v,up,vp):
    """
    Adapted from https://arxiv.org/pdf/2001.09253.pdf
    Originally Released under a CC0 license by Haysn Hornbeck
    """
    if b < a:
        print('Warning: b must be > a')
    ba = ( b - a )
    xa = ( x - a )
    inv_ba = 1. / ba
    bx = ( b - x )
    ba2 = ba * ba
    lower = xa * v + bx * u
    C = ( xa * xa - ba2 ) * xa * vp
    D = ( bx * bx - ba2 ) * bx * up
    return ( lower + (.16666666666666666666) *( C + D ) ) * inv_ba

@njit(cache=True)
def new_second_derivative(knots, values, start_deriv, end_deriv):
    """
    Adapted from https://arxiv.org/pdf/2001.09253.pdf
    Originally Released under a CC0 license by Haysn Hornbeck
    """
    n = len(knots)
    if n != len(values):
        print('warning: knots and values should have same length')
    if n <= 2:
        print('warning: must have more than 2 knot value pairs')
    for i in range(1,n):
        if knots[i] < knots[i-1]:
            print('knots must be sorted in increasing order')

    c_p = np.zeros(n)
    ypp = np.zeros(n)

    # recycle these values in later routines
    new_x = knots[1]
    new_y = values[1]
    cj = knots[1] - knots[0]
    new_dj = (values[1] - values[0]) / cj

    # initialize the forward substitution
    if start_deriv > .99e30:
        c_p[0] = 0
        ypp[0] = 0
    else:
        c_p[0] = 0.5
        ypp[0] = 3 * ( new_dj - start_deriv ) / cj

    # forward substitution portion
    j = 1
    while j < (n-1):
        # shuffle new values to old
        old_x = new_x
        old_y = new_y
        aj = cj
        old_dj = new_dj
        # generate new quantities
        new_x = knots[j+1]
        new_y = values[j+1]
        cj = new_x - old_x
        new_dj = ( new_y - old_y ) / cj
        bj = 2*(cj + aj)
        inv_denom = 1. / ( bj - aj * c_p[j-1])
        dj = 6*(new_dj - old_dj)
        ypp[j] = (dj - aj * ypp[j-1]) * inv_denom
        c_p[j] = cj * inv_denom
        j += 1

    # handle the end derivative
    if end_deriv > .99e30 :
        c_p[j] = 0
        ypp[j] = 0
    else :
        old_x = new_x
        old_y = new_y
        aj = cj
        old_dj = new_dj
        cj = 0 # this has the same effect as skipping c_n
        new_dj = end_deriv
        bj = 2*(cj + aj)
        inv_denom = 1. / (bj - aj * c_p[j-1])
        dj = 6*(new_dj - old_dj)
        ypp[j] = (dj - aj * ypp [j-1]) * inv_denom
        c_p[j] = cj * inv_denom

    # as we 're storing d_j in y ''_j , y ''_n = d_n is a no-op
    # backward substitution portion
    while j > 0:
        j -= 1
        ypp[j] = ypp[j] - c_p[j]* ypp[j+1]

    return ypp

@njit(cache=True)
def spintarr(xnew,knots,values,ydiv2):
    ynew = np.zeros_like(xnew)
    i = 0
    for n,x in enumerate(xnew):
        while (i+1 < len(knots)) and (knots[i+1] < x) :
            i += 1
        ynew[n] =  newint(x,knots[i],knots[i+1],values[i],values[i+1],ydiv2[i],ydiv2[i+1])
    return ynew

@njit(cache=True)
def spintone(xnew,knots,values,ydiv2):
    i = 0
    while (i+1 < len(knots)) and (knots[i+1] < xnew):
        i += 1
    ynew =  newint(xnew,knots[i],knots[i+1],values[i],values[i+1],ydiv2[i],ydiv2[i+1])
    return ynew

def get_eTransType(elvl_data,scups_data):
    qns = elvl_data['s']
    qnj = elvl_data['j']
    qnl = elvl_data['l']
    iqnp = elvl_data['parity']  ## 0 == even; 1 == old parity
    lowlev = scups_data['lower_level_index']-1
    upplev = scups_data['upper_level_index']-1
    ettype = np.zeros(len(lowlev),dtype = int) -1   ## initialized all to -1
    sallow = (qns[upplev] == qns[lowlev])   ## SPIN ALLOWED?
    pallow = (iqnp[upplev] != iqnp[lowlev]) ## parity change?
    isumm  = (qnj[upplev] + qnj[lowlev]).astype(int)
    idiff  = (abs(qnj[upplev] - qnj[lowlev])).astype(int)
    wE1 = sallow * pallow * (isumm != 0) * (idiff <= 1)  #E1 transitions
    wE2 = sallow * (~pallow) * (isumm >= 2) * (idiff <= 2) ## E2 transitions
    wM1 = sallow * (~pallow) * (isumm != 0) * (idiff <= 1) ## M1 transitions
    wE1_spinchanging = (~sallow)*pallow*(isumm!=0)*(idiff<=1)
    wE1_forbidden_spinchanging = (~sallow)*(~pallow)*(isumm != 0)*(idiff <= 2)
    ## the order of these matter and all need to be included
    ## M1 and E2 are similar but all with idiff isumm>=2 and idiff<=1 are
    ## assumed to be M1
    ettype[wE1] = 1
    ettype[wE2] = 2
    ettype[wM1] = -1
    ettype[wE1_spinchanging] = -1
    ettype[wE1_forbidden_spinchanging] = -1

    return ettype

@njit(cache=True)
def rad_field_bframe(wlang,thetab,rphot,include_limbdark = True,photo_temp = 6000.):
    """ Calculates the radiation field tensor in the cylindrically symmetric case

    Parameters
    ----------
    wlang : float, array-like (unit: angstroms)
        wavelength of transition
    thetab : float (unit: radians)
        inclination of the magnetic field wrt solar vertical in radians
    rphot : float (unit: fraction of solar radius)
        height above surface in radius units
    """

    nline = len(wlang)
    amu_b = np.cos(thetab)
    factor = (0.5)*(3.0*amu_b*amu_b-1.0)

    ## LANDI DEGL'INNOCENTI & LANDOLFI (2004) Equation 12.32
    sg = 1.0 / (1.0 + rphot)   ## sin(gamma)
    cg2 = 1.-sg**2
    cg  = np.sqrt(cg2)
    cg3 = cg2*cg
    logval = (np.log1p(sg) - np.log(cg))  ## old: np.log((1.+sg)/cg)
    b0 = (1.-cg3)/3.
    b1 = (8.*cg3 - 3.*cg2 - 2.) / 24. - cg**4 / (8.*sg) * logval
    b2 = (cg-1.)*(3.*cg3 + 6.*cg2 + 4.*cg + 2.) / (15.*(cg+1.))
    a0 = 1. - cg
    a1 = cg - 0.5 - 0.5*cg2/sg*logval
    a2 = (cg+2.)*(cg-1.) / (3.*(cg+1.))

    cc = 2.99792458e10
    hh = 6.626176e-27
    bk = 1.380662e-16

    nu = 1.e8 * (cc/wlang)
    boltzm1 = np.expm1(hh*nu/(bk*photo_temp))
    planck = 2.0*hh*(nu**3)/(cc*cc*boltzm1)

    ## NEED TO IMPLEMENT LIMB DARK FLAG HERE
    uc = np.interp(wlang,lambdaCLV,uCLV)
    vc = np.interp(wlang,lambdaCLV,vCLV)
    if not include_limbdark:
        uc = 0.*uc
        vc = 0.*vc

    ## LANDI DEGL'INNOCENTI & LANDOLFI (2004) Equation 12.34
    Jnu = 0.5 * planck * (a0 + a1*uc + a2*vc)
    Knu = 0.5 * planck * (b0 + b1*uc + b2*vc)

    radj = np.zeros((3,nline))
    radj[0,:] = Jnu
    radj[2,:] = (3.0*Knu-Jnu)/(2.0*(2**0.5)) * factor  

    return radj

@njit(cache=True)
def calcLande(J,S,L):
    ## equation 3.8 in LD&L (2004)
    g = np.zeros(len(J))
    ## if J equal to zero, make lande g = 0
    g[(J == 0)] = 0.
    wg = (J != 0)
    J,S,L = J[wg],S[wg],L[wg]
    g[wg] = 1. + 0.5 * (J*(J+1) + S*(S+1) - L*(L+1)) / (J*(J+1))
    return g

@njit(cache=True)
def getDcoeff(Jupp,Jlow):
    ## D = w^(K=2)_JuJl  from equation 10.10 in LD&L (2004)
    ## equation 13.28 from LD&L (2004)
    nline = len(Jupp)
    d_coeff = np.zeros(nline)
    for n in range(nline):
        Ju,Jl = Jupp[n],Jlow[n]
        Ju2 = int(2*Ju)
        Jl2 = int(2*Jl)
        d_coeff[n] = (-1)**(1.+Jl+Ju) * np.sqrt(3.*(Ju2+1.)) * wigner.w6js(2,2,4,Ju2,Ju2,Jl2)
    return d_coeff

@njit(cache=True)
def getEcoeff(Jupp,Jlow,gupp,glow):
    ## D = w^(K=2)_JuJl  from equation 10.10 in LD&L (2004)
    ## equation 13.28 from LD&L (2004)
    nline = len(Jupp)
    e_coeff = np.zeros(nline)
    for n in range(nline):
        Ju,Jl,gu,gl = Jupp[n],Jlow[n],gupp[n],glow[n]
        Ju2 = int(2*Ju)
        Jl2 = int(2*Jl)
        factor = gu*(-1)**(1.+Jl-Ju) * np.sqrt(Ju*(Ju+1)*(Ju2+1)) * wigner.w6js(4,2,2,Ju2,Ju2,Ju2) * \
                wigner.w6js(2,2,2,Jl2,Ju2,Ju2) + \
                + gl*np.sqrt(Jl*(Jl+1)*(2.*Jl+1))*wigner.w9js(Jl2,Ju2,2,Jl2,Ju2,2,2,4,2)
        e_coeff[n] = -3.*np.sqrt(2.*Ju + 1) * factor
    return e_coeff

def setupSEE(qnj,all_ks=False):

    ## Determine the indices of the statistical equilibrium matrix elements
    ## - No-coherence hypothesis is assumed, thus Q = 0 throughout
    ## - LD&L (2004) Section 13.5 discusses that K can be limited to even values.
    ##   Below, the option to use all k values remains, but it is likely not
    ##   necessary unless the code is extended to other use cases.
    ## - The total number of equations is given for the Q=0,K=even case from
    ##   the equations in Section 13.5.  For all K, each level contributes
    ##   2*J + 1 equations for each of its K values

    if all_ks:
        see_dk  = 1
        see_neq = int(np.sum(2*qnj+1))
    else:
        see_dk = 2
        if (qnj[0]%1 == 0.0):
            see_neq   = int(np.sum(qnj+1))
        else:
            see_neq   = int(np.sum(qnj+0.5))

    see_index = np.arange(see_neq)             ## index of the SEE table element
    see_lev   = np.zeros(see_neq,dtype = int)  ## energy level index
    see_k     = np.zeros(see_neq,dtype = int)  ## value of k

    nlev = len(qnj)

    i = 0
    for lev in range(nlev):
        for k in range(0,int(2*qnj[lev]+1),see_dk):
            see_lev[i] = lev
            see_k[i] = k
            i += 1

    return see_neq,see_index,see_lev,see_k,see_dk

@njit(cache=True)
def seeSolve(aa,weight,see_lev,see_k):
    """
    Solves the statistical equilibrium equations
    """

    ## particle conservation equation
    aa[0,:] = weight
    bb = np.zeros(weight.shape[0])
    bb[0] = 1.

    ## solve system
    xx = np.linalg.solve(aa,bb)

    ## check solution
    wk0 = (weight != 0)
    ntot = np.sum(weight[wk0]*xx[wk0])
    if (abs(1-ntot) > 1.e-8):
        print('Warning:  Solution may not be accurate')

    ## reshape into rho matrix
    rho = np.zeros((max(see_lev)+1,max(see_k)+1))
    for i in range(weight.shape[0]):
        rho[see_lev[i],see_k[i]] = xx[i]

    return rho
