"""
This module provides functions necessary for calculating collisional rate 
coefficients.
"""

from numba import njit,jit
import numpy as np
from pycelp.wigner import w6js,w3js,w9js
import pycelp.util as util

@njit(cache=True)
def getSecondDerivatives(bt_t,bt_u,n_t):
    """
    Calculates the 2nd Derivatives needed for the cubic spline interpolation
    bt_t -- scaled temperatures
    bt_u -- scaled upsilons
    n_t -- number of knots (i.e. temperatures) included
    """
    ncol = bt_t.shape[0]
    yd2 = np.zeros_like(bt_t)
    for ic in range(ncol):
        n = n_t[ic]
        yd2[ic,0:n] = util.new_second_derivative(bt_t[ic,0:n],bt_u[ic,0:n],1e100,1e100)
    return yd2

@njit(cache=True)
def intErates(lowlev,upplev,qnj,dE,ccA,ttA,btT,btU,btY,etemp,edens):

    ncol        = len(btT)
    ups_all     = np.zeros(ncol)
    erates_up   = np.zeros(ncol)
    erates_down = np.zeros(ncol)
    kte_all     = etemp/dE/1.57888e5

    for ic in range(ncol):
        de = dE[ic]
        cc = ccA[ic]
        tt = ttA[ic]
        kte = kte_all[ic]

        if (tt==1) or (tt == 4):
            xt = 1 - np.log(cc)/(np.log(kte + cc))
        else:
            xt = kte/(kte +cc)

        sups = util.spintone(xt,btT[ic,:],btU[ic,:],btY[ic,:])

        if tt == 1:
            ups=sups*np.log(kte + np.e)
        elif tt == 2:
            ups=sups
        elif tt == 3:
            ups=sups/(kte+1.)
        elif tt == 4:
            ups=sups*np.log(kte+cc)
        elif tt == 5:
            ups=sups/(kte)
        elif tt == 6:
            ups=10.**sups
        else:
            print(' t_type ne 1,2,3,4,5,6 ')

        ## clip at zero
        if ups<0: ups = 0.
        ups_all[ic] = ups

    gl = 2*qnj[lowlev] + 1
    gu = 2*qnj[upplev] + 1
    ## boltzman constant = 8.61733034e-5 eV / K
    ## dE is in rydberg
    ## 1 Rydberg =  13.6056980659 eV

    expv  = np.exp(-dE*13.6056980659 / ( 8.61733034e-5*etemp))
    erates_up  = 8.628e-06/np.sqrt(etemp) * expv/gl * ups_all * edens
    erates_down = 8.628E-06/gu/np.sqrt(etemp) * ups_all * edens

    return erates_up,erates_down


@njit(cache=True)
def intPrates(lowlev,upplev,qnj,dE,ccA,ttA,btT,btU,btY,etemp,edens):

    ncol        = len(btT)
    ups_all     = np.zeros(ncol)
    kte_all     = etemp/dE/1.57888e5

    for ic in range(ncol):
        de = dE[ic]
        cc = ccA[ic]
        tt = ttA[ic]
        kte = kte_all[ic]

        if (tt==1) or (tt == 4):
            xt = 1 - np.log(cc)/(np.log(kte + cc))
        else:
            xt = kte/(kte +cc)

        sups = util.spintone(xt,btT[ic,:],btU[ic,:],btY[ic,:])

        if tt == 1:
            ups=sups*np.log(kte + np.e)
        elif tt == 2:
            ups=sups
        elif tt == 3:
            ups=sups/(kte+1.)
        elif tt == 4:
            ups=sups*np.log(kte+cc)
        elif tt == 5:
            ups=sups/(kte)
        elif tt == 6:
            ups=10.**sups
        else:
            print(' t_type ne 1,2,3,4,5,6 ')

        ## clip at zero
        if ups<0: ups = 0.
        ups_all[ic] = ups

    gl = 2*qnj[lowlev] + 1
    gu = 2*qnj[upplev] + 1
    ## boltzman constant = 8.61733034e-5 eV / K
    ## dE is in rydberg
    ## 1 Rydberg =  13.6056980659 eV
    ## need to follow up on these equations once more is it dE or -dE

    expv  = np.exp(dE*13.6056980659 / ( 8.61733034e-5*etemp))
    prates_up  = ups_all * edens
    prates_down = prates_up * gl/gu * expv

    return prates_up,prates_down


@njit(cache=True)
def setup_ecoll(lowlev,upplev,ettype,qnj,see_index,see_lev,see_k,see_dk):

    ## figure out maximum number of coefficents -- will be reduced if too many
    ktot = 0
    for t in range(len(lowlev)):
        Jl = qnj[lowlev[t]]
        Ju = qnj[upplev[t]]
        ktot += int(min(2*Jl+1,2*Ju+1))   ## because only ku to kl can exist for the min value

    ktot = ktot*4
    print(' how to decide ktot here  where I also add the relaxation')

    ciK = np.zeros(ktot) ## ci
    ciK_indx = np.zeros((3,ktot),dtype = np.int32) ## tindx_low,tindx_upp,transindx
    csK = np.zeros(ktot) ## cs
    csK_indx = np.zeros((3,ktot),dtype = np.int32) ## tindx_upp,tindx_low,transindx

    icnt = 0
    scnt = 0

    see_neq = len(see_lev)

    for t in range(len(lowlev)):

        Jl = qnj[lowlev[t]]
        Ju = qnj[upplev[t]]
        kmax = int(min(2*Jl,2*Ju)) ## because only ku to kl can exist for the min value

        if (ettype[t] == 1):

            ## CI_K(a_J,al_Jl)
            demon_i = w6js(int(2*Ju),int(2*Ju),0,int(2*Jl),int(2*Jl),2)
            for k in range(0,kmax+1,see_dk):
                ci = (-1.)**(k) * w6js(int(2*Ju),int(2*Ju),int(2*k),int(2*Jl),int(2*Jl),2) / demon_i
                if (ci != 0):
                    ## transfer rate
                    idx_src = see_index[((see_lev == lowlev[t])*(see_k == k))][0]
                    idx_dst = see_index[((see_lev == upplev[t])*(see_k == k))][0]
                    ciK[icnt] = np.sqrt((2.*Jl+1.)/(2.*Ju+1.)) * ci
                    ciK_indx[:,icnt] = idx_src,idx_dst,t
                    icnt +=1
                    ## relaxation rate
                    if (k == 0):
                        for f in range(see_neq):
                            if see_lev[f] == lowlev[t]:
                                ciK[icnt] = - ci
                                ciK_indx[:,icnt] = f,f,t
                                icnt +=1

            ## CS_K(a_J,au_Ju)
            demon_s = w6js(int(2*Jl),int(2*Jl),0,int(2*Ju),int(2*Ju),2)
            for k in range(0,kmax+1,see_dk):
                cs = (-1.)**(k) * w6js(int(2*Jl),int(2*Jl),int(2*k),int(2*Ju),int(2*Ju),2) /  demon_s
                if (cs != 0):
                    idx_src = see_index[((see_lev == upplev[t])*(see_k == k))][0]
                    idx_dst = see_index[((see_lev == lowlev[t])*(see_k == k))][0]
                    csK[scnt] = np.sqrt((2.0*Ju+1.)/(2.0*Jl+1.0)) * cs
                    csK_indx[:,scnt] =  idx_src,idx_dst,t
                    scnt += 1
                    ## relaxation rate
                    if (k == 0):
                        for f in range(see_neq):
                            if see_lev[f] == upplev[t]:
                                csK[scnt] = - cs
                                csK_indx[:,scnt] = f,f,t
                                scnt +=1

        if (ettype[t] == 2):

            ## CI_K(a_J,al_Jl)
            demon_i = w6js(int(2*Ju),int(2*Ju),0,int(2*Jl),int(2*Jl),4)
            for k in range(0,kmax+1,see_dk):
                ci = (-1.)**(k) * w6js(int(2*Ju),int(2*Ju),int(2*k),int(2*Jl),int(2*Jl),4) / demon_i
                if (ci != 0):
                    idx_src = see_index[((see_lev == lowlev[t])*(see_k == k))][0]
                    idx_dst = see_index[((see_lev == upplev[t])*(see_k == k))][0]
                    ciK[icnt] = np.sqrt((2.*Jl+1.)/(2.*Ju+1.))*ci
                    ciK_indx[:,icnt] = idx_src,idx_dst,t
                    icnt +=1
                    ## relaxation rate
                    if (k == 0):
                        for f in range(see_neq):
                            if see_lev[f] == lowlev[t]:
                                ciK[icnt] = - ci
                                ciK_indx[:,icnt] = f,f,t
                                icnt +=1

            ## CS_K(a_J,au_Ju)
            demon_s = w6js(int(2*Jl),int(2*Jl),0,int(2*Ju),int(2*Ju),4)
            for k in range(0,kmax+1,see_dk):
                cs = (-1.)**(k) * w6js(int(2*Jl),int(2*Jl),int(2*k),int(2*Ju),int(2*Ju),4) /  demon_s
                if (cs != 0):
                    idx_src = see_index[((see_lev == upplev[t])*(see_k == k))][0]
                    idx_dst = see_index[((see_lev == lowlev[t])*(see_k == k))][0]
                    csK[scnt] = np.sqrt((2.0*Ju+1.)/(2.0*Jl+1.0)) * cs
                    csK_indx[:,scnt] =  idx_src, idx_dst,t
                    scnt += 1
                    ## relaxation rate
                    if (k == 0):
                        for f in range(see_neq):
                            if see_lev[f] == upplev[t]:
                                csK[scnt] = - cs
                                csK_indx[:,scnt] = f,f,t
                                scnt +=1

        if (ettype[t] == -1):

            ## CI_K(a_J,al_Jl)
            f3b = np.zeros(int(2*Jl+1)+1)
            for k in range(0,kmax+1,see_dk):
                citmp = 0.0
                for ani in range(1,int(2*Ju+1)+1):
                    an = -Ju + (ani-1.)
                    f3 = w3js(int(2*Ju),int(2*Ju),int(2*k),int(2*an),int(-2*an),0)
                    for an1i in range(1,int(2*Jl+1)+1):
                        an1 = -Jl + (an1i-1.)
                        if (ani == 1): f3b[an1i] = w3js(int(2*Jl),int(2*Jl),int(2*k),int(2*an1),int(-2*an1),0)
                        citmp = citmp + ((-1.)**(int(Ju-an+Jl-an1)))*f3*f3b[an1i]
                ci = np.sqrt((2.0*Ju+1.0)/(2.0*Jl+1.0))*citmp
                ci = ci/(2.0*Ju+1.0)
                if (ci != 0):
                    idx_src = see_index[((see_lev == lowlev[t])*(see_k == k))][0]
                    idx_dst = see_index[((see_lev == upplev[t])*(see_k == k))][0]
                    ciK[icnt] = np.sqrt((2.*Jl+1.)/(2.*Ju+1.)) * ci
                    ciK_indx[:,icnt] = idx_src, idx_dst,t
                    icnt +=1
                    ## relaxation rate
                    if (k == 0):
                        for f in range(see_neq):
                            if see_lev[f] == lowlev[t]:
                                ciK[icnt] = - ci
                                ciK_indx[:,icnt] = f,f,t
                                icnt +=1

            ## CS_K(a_J,au_Ju)
            f3b = np.zeros(int(2*Ju+1)+1)
            for k in range(0,kmax+1,see_dk):
                cstmp = 0.0
                for ani in range(1,int(2*Jl+1)+1):
                    an = -Jl + (ani-1.)
                    f3 = w3js(int(2*Jl),int(2*Jl),int(2*k),int(2*an),int(-2*an),0)
                    for an1i in range(1,int(2*Ju+1)+1):
                        an1 = -Ju + (an1i-1.)
                        if (ani == 1): f3b[an1i] = w3js(int(2*Ju),int(2*Ju),int(2*k),int(2*an1),int(-2*an1),0)
                        cstmp = cstmp + ((-1.)**(int(Jl-an+Ju-an1)))*f3*f3b[an1i]
                cs = np.sqrt((2.*Jl+1.)/(2.*Ju+1.))*cstmp
                cs = cs/(2.0*Jl+1.0)
                if (cs != 0):
                    idx_src = see_index[((see_lev == upplev[t])*(see_k == k))][0]
                    idx_dst = see_index[((see_lev == lowlev[t])*(see_k == k))][0]
                    csK[scnt] = np.sqrt((2.0*Ju+1.)/(2.0*Jl+1.0)) * cs
                    csK_indx[:,scnt] =  idx_src,idx_dst,t
                    scnt += 1
                    ## relaxation rate
                    if (k == 0):
                        for f in range(see_neq):
                            if see_lev[f] == upplev[t]:
                                csK[scnt] = - cs
                                csK_indx[:,scnt] = f,f,t
                                scnt +=1


    ## cull zeros at the end for now
    ciK = ciK[0:icnt]
    ciK_indx = ciK_indx[:,0:icnt]
    csK = csK[0:scnt]
    csK_indx = csK_indx[:,0:scnt]

    return ciK,ciK_indx,csK,csK_indx


@njit(cache=True)
def getElectronSEE(ciK,ciK_indx,csK,csK_indx,erates_up,erates_down,see_neq):

    ecmat = np.zeros((see_neq,see_neq))

    for cit in range(ciK.shape[0]):
        #idx_src,idx_dst,t = ciK_indx[:,cit]
        ecmat[ciK_indx[1,cit],ciK_indx[0,cit]] += ciK[cit] * erates_up[ciK_indx[2,cit]]

    for cit in range(csK.shape[0]):
        #idx_src,idx_dst,t = csK_indx[:,cit]
        ecmat[csK_indx[1,cit],csK_indx[0,cit]] += csK[cit] * erates_down[csK_indx[2,cit]]

    return ecmat
