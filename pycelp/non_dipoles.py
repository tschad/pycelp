
from numba import njit,jit
import numpy as np
from pycle.wigner import w6js,w3js,w9js
import pycle.util as util

@njit(cache=True)
def setup_nonDipoles(lowlev,upplev,qnj,b_low2up,a_up2low,b_up2low,
                     see_index,see_lev,see_k,see_dk):

    deltaJ = np.abs(qnj[upplev] - qnj[lowlev])
    nrad = len(lowlev)

    ## figure out maximum number of coefficents
    ktot_nonD = 0
    for t in range(nrad):
        if (deltaJ[t] >1):
            Jl = qnj[lowlev[t]]
            Ju = qnj[upplev[t]]
            ktot_nonD += int(min(2*Jl+1,2*Ju+1))   ## because only ku to kl can exist for the min value

    ktot_nonD *= 5

    taK = np.zeros(ktot_nonD) ## prefactor,ta
    taK_indx = np.zeros((3,ktot_nonD),dtype = np.int32) ## tindx_low,tindx_upp,transindx
    teK = np.zeros(ktot_nonD) ## prefactor,te
    teK_indx = np.zeros((3,ktot_nonD),dtype = np.int32) ## tindx_upp,tindx_low,transindx
    tsK = np.zeros(ktot_nonD) ## prefactor,ts
    tsK_indx = np.zeros((3,ktot_nonD),dtype = np.int32) ## tindx_upp,tindx_low,transindx

    acnt = 0
    ecnt = 0
    scnt = 0

    see_neq = len(see_lev)

    print(' getting non-dipole rate factors')

    for t in range(nrad):

        if (deltaJ[t] <= 1) and (deltaJ[t]%1 == 0): continue    ## skip over dipole lines

        Jl = qnj[lowlev[t]]
        Ju = qnj[upplev[t]]
        kmax = int(min(2*Jl,2*Ju)) ## because only ku to kl can exist for the min value

        ## three total rates here
        cup = b_low2up[t]  ## will get multiplied by Jrad_(Q=0)^(K=0) [line]
        cdown_spon = a_up2low[t]  ## spontaneous
        cdown_stim = b_up2low[t]  ## will get multiplied by Jrad_(Q=0)^(K=0) [line]

        ## TA equivalent  -- treat it like inelastic excitation 'strong coupling'
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
            idx_src = see_index[((see_lev == lowlev[t])*(see_k == k))][0]
            idx_dst = see_index[((see_lev == upplev[t])*(see_k == k))][0]
            ci = ci * cup/(2.0*Ju+1.0)
            taK[acnt] = np.sqrt((2.*Jl+1.)/(2.*Ju+1.))*ci
            taK_indx[:,acnt] = idx_src, idx_dst,t
            acnt +=1
            ## relaxation rate
            if (k == 0):
                for f in range(see_neq):
                    if see_lev[f] == lowlev[t]:
                        taK[acnt] = - ci
                        taK_indx[:,acnt] = f,f,t
                        acnt +=1

        ## TE equivalent  -- treat it like superelastic deexcitation 'strong coupling'
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
            idx_src = see_index[((see_lev == upplev[t])*(see_k == k))][0]
            idx_dst = see_index[((see_lev == lowlev[t])*(see_k == k))][0]
            cs = cs * cdown_spon/(2.0*Jl+1.0)
            teK[ecnt] = np.sqrt((2.0*Ju+1.)/(2.0*Jl+1.0)) * cs
            teK_indx[:,ecnt] =  idx_src,idx_dst,t
            ecnt += 1
            ## relaxation rate
            if (k == 0):
                for f in range(see_neq):
                    if see_lev[f] == upplev[t]:
                        teK[ecnt] = - cs
                        teK_indx[:,ecnt] = f,f,t
                        ecnt +=1

        ## TS equivalent  -- treat it like superelastic deexcitation 'strong coupling'
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
            idx_src = see_index[((see_lev == upplev[t])*(see_k == k))][0]
            idx_dst = see_index[((see_lev == lowlev[t])*(see_k == k))][0]
            cs = cs * cdown_stim/(2.0*Jl+1.0)
            tsK[scnt] = np.sqrt((2.0*Ju+1.)/(2.0*Jl+1.0))*cs
            tsK_indx[:,scnt] =  idx_src,idx_dst,t
            scnt += 1
            ## relaxation rate
            if (k == 0):
                for f in range(see_neq):
                    if see_lev[f] == upplev[t]:
                        tsK[scnt] = - cs
                        tsK_indx[:,scnt] = f,f,t
                        scnt +=1

    ## cull zeros at the end for now
    taK = taK[0:acnt]
    taK_indx = taK_indx[:,0:acnt]
    teK = teK[0:ecnt]
    teK_indx = teK_indx[:,0:ecnt]
    tsK = tsK[0:scnt]
    tsK_indx = tsK_indx[:,0:scnt]

    ## get the spontaneous emission matrix
    see_neq = len(see_lev)
    nonD_spon =  np.zeros((see_neq,see_neq))

    for cit in range(teK.shape[0]):
        ## superelastic SPONTANEOUS emission rate
        #idx_src,idx_dst,t = teK_indx[:,cit]
        #nonDmat[idx_dst,idx_src] += teK[cit] ## no radiation term
        nonD_spon[teK_indx[1,cit],teK_indx[0,cit]] += teK[cit]  ## no radiation term

    ## sort them for faster memory access later
    ## put them all into one array

    tnD = np.hstack((taK,tsK))
    tnD_indx = np.hstack((taK_indx,tsK_indx))

    return tnD,tnD_indx,nonD_spon

@njit(cache=True)
def getNonDipoleSEE(tnD,tnD_indx,radj,nonDmat):

    for cit in range(tnD.shape[0]):
        ## inelastic ABSORPTION rate
        #idx_src,idx_dst,t = taK_indx[:,cit]
        #nonDmat[idx_dst,idx_src] += taK[cit] * radj[0,t]
        nonDmat[tnD_indx[1,cit],tnD_indx[0,cit]] += tnD[cit] * radj[0,tnD_indx[2,cit]]

    return nonDmat
