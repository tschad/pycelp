
from numba import njit,jit
import numpy as np
from pycle.wigner import w6js,w3js,w9js
import pycle.util as util 

@njit(cache=True)
def setup_Dipoles(lowlev,upplev,qnj,b_low2up,a_up2low,b_up2low,
                  see_index,see_lev,see_k,see_dk):

    ktot_D = 1000000  ## how to determine this value?

    taD = np.zeros((ktot_D)) ## prefactor,ta
    taD_indx = np.zeros((4,ktot_D),dtype = np.int32) ## tindx_low,tindx_upp,kr,transindx
    tsD = np.zeros((ktot_D)) ## prefactor,ts
    tsD_indx = np.zeros((4,ktot_D),dtype = np.int32) ## tindx_upp,tindx_low,kr,transindx
    raD = np.zeros((ktot_D)) ## prefactor,ta
    raD_indx = np.zeros((4,ktot_D),dtype = np.int32) ## tindx_low,tindx_upp,kr,transindx
    rsD = np.zeros((ktot_D)) ## prefactor,ts
    rsD_indx = np.zeros((4,ktot_D),dtype = np.int32) ## tindx_upp,tindx_low,kr,transindx

    teD = np.zeros((ktot_D)) ## prefactor,te
    teD_indx = np.zeros((3,ktot_D),dtype = np.int32) ## tindx_upp,tindx_low,transindx
    reD = np.zeros((ktot_D)) ## prefactor,te
    reD_indx = np.zeros((3,ktot_D),dtype = np.int32) ## tindx_upp,tindx_low,transindx

    tacnt = 0
    tscnt = 0
    racnt = 0
    rscnt = 0
    tecnt = 0
    recnt = 0

    deltaJ = np.abs(qnj[upplev] - qnj[lowlev])
    nrad = len(upplev)

    for t in range(nrad):

        ## Note:  In the Chianti v9 database there is one Fe 11 level with half-integer J
        if (deltaJ[t] > 1) or (deltaJ[t]%1 != 0): continue    ## skip over non-dipole lines

        Jl = qnj[lowlev[t]]
        Ju = qnj[upplev[t]]

        ##  TA -- EQUATION 7.14a (Q = 0)

        for kl in range(0,int(2*Jl)+1,see_dk):
            for ku in range(0,int(2*Ju)+1,see_dk):
                for kr in range(3):
                    val =  (2.*Jl+1) * b_low2up[t] * \
                        np.sqrt(3.*(2*ku+1)*(2*kl+1)*(2*kr+1)) * \
                        (-1)**(kl) * \
                        w9js(int(2*Ju),int(2*Jl),2,int(2*Ju),int(2*Jl),2,int(2*ku),int(2*kl),int(2*kr)) * \
                        w3js(int(2*ku),int(2*kl),int(2*kr),0,0,0)
                    if (val != 0):
                        idx_src = see_index[((see_lev == lowlev[t])*(see_k == kl))][0]
                        idx_dst = see_index[((see_lev == upplev[t])*(see_k == ku))][0]
                        taD[tacnt] = val
                        taD_indx[:,tacnt] = idx_src,idx_dst,kr,t
                        tacnt += 1

        ## RA - 7.14d also 7.11
        for k in range(0,int(2*Jl)+1,see_dk):
            for kpr in range(0,int(2*Jl)+1,see_dk):
                for kr in range(3):
                    val =  (2.*Jl+1)*b_low2up[t]* \
                         np.sqrt(3.*(2.*k+1.)*(2.*kpr+1.)*(2.*kr+1.)) * \
                         (-1)**(1+Ju-Jl+kr) * \
                         w6js(int(2*k),int(2*kpr),int(2*kr),int(2*Jl),int(2*Jl),int(2*Jl)) * \
                         w6js(2,2,int(2*kr),int(2*Jl),int(2*Jl),int(2*Ju)) * \
                         w3js(int(2*k),int(2*kpr),int(2*kr),0,0,0) * \
                         (0.5*(1. + (-1)**(k+kpr+kr)))
                    if (val != 0):
                        idx_src = see_index[((see_lev == lowlev[t])*(see_k == kpr))][0]
                        idx_dst = see_index[((see_lev == lowlev[t])*(see_k == k))][0]
                        raD[racnt] = val
                        raD_indx[:,racnt] = idx_src,idx_dst,kr,t
                        racnt += 1

        ## TS - Equation 7.14c All Q = 0
        for kl in range(0,int(2*Jl)+1,see_dk):
            for ku in range(0,int(2*Ju)+1,see_dk):
                for kr in range(3):
                    val =  (2.*Ju+1) * b_up2low[t] * \
                        np.sqrt(3.*(2*kl+1)*(2*ku+1)*(2*kr+1)) * \
                        (-1)**(kr+ku) * \
                        w9js(int(2*Jl),int(2*Ju),2,int(2*Jl),int(2*Ju),2,int(2*kl),int(2*ku),int(2*kr)) * \
                        w3js(int(2*kl),int(2*ku),int(2*kr),0,0,0)
                    if (val != 0):
                        idx_src = see_index[((see_lev == upplev[t])*(see_k == ku))][0]
                        idx_dst = see_index[((see_lev == lowlev[t])*(see_k == kl))][0]
                        tsD[tscnt] = val
                        tsD_indx[:,tscnt] = idx_src,idx_dst,kr,t
                        tscnt += 1

        ## RS - 7.14f
        for k in range(0,int(2*Ju)+1,see_dk):
            for kpr in range(0,int(2*Ju)+1,see_dk):
                for kr in range(3):
                    val = (2.*Ju+1) * b_up2low[t] * \
                     np.sqrt(3.*(2.*k+1.)*(2.*kpr+1.)*(2.*kr+1.)) * \
                     (-1)**(1+Jl-Ju) * \
                      w6js(int(2*k),int(2*kpr),int(2*kr),int(2*Ju),int(2*Ju),int(2*Ju)) * \
                      w6js(2,2,int(2*kr),int(2*Ju),int(2*Ju),int(2*Jl)) * \
                      w3js(int(2*k),int(2*kpr),int(2*kr),0,0,0) * \
                      (0.5*(1. + (-1)**(k+kpr+kr)))
                    if (val != 0):
                        idx_src = see_index[((see_lev == upplev[t])*(see_k == kpr))][0]
                        idx_dst = see_index[((see_lev == upplev[t])*(see_k == k))][0]
                        rsD[rscnt] = val
                        rsD_indx[:,rscnt] = idx_src,idx_dst,kr,t
                        rscnt += 1

        ## TE
        for kl in range(0,int(2*Jl)+1,see_dk):
            for ku in range(0,int(2*Ju)+1,see_dk):
                if (kl == ku):
                    val = (2*Ju+1)*a_up2low[t]* \
                        (-1)**(1.+Jl+Ju+kl)*w6js(int(2*Ju),int(2*Ju),int(2*kl),int(2*Jl),int(2*Jl),2)
                    idx_src = see_index[((see_lev == upplev[t])*(see_k == ku))][0]
                    idx_dst = see_index[((see_lev == lowlev[t])*(see_k == kl))][0]
                    if (val != 0):
                        teD[tecnt] = val
                        teD_indx[:,tecnt] = idx_src,idx_dst,t
                        tecnt += 1

        ## RE
        for ku in range(0,int(2*Ju)+1,see_dk):
            val = a_up2low[t]
            if (val != 0):
                idx_src = see_index[((see_lev == upplev[t])*(see_k == ku))][0]
                idx_dst = idx_src
                reD[recnt] = val
                reD_indx[:,recnt] = idx_src,idx_dst,t
                recnt += 1

    ## cull zeros at the end for now

    taD = taD[0:tacnt]
    taD_indx = taD_indx[:,0:tacnt]
    tsD = tsD[0:tscnt]
    tsD_indx = tsD_indx[:,0:tscnt]
    teD = teD[0:tecnt]
    teD_indx = teD_indx[:,0:tecnt]

    raD = raD[0:racnt]
    raD_indx = raD_indx[:,0:racnt]
    rsD = rsD[0:rscnt]
    rsD_indx = rsD_indx[:,0:rscnt]
    reD = reD[0:recnt]
    reD_indx = reD_indx[:,0:recnt]

    ## get the spontaneous emission matrix
    see_neq = len(see_lev)
    Dmat_spon =  np.zeros((see_neq,see_neq))

    for cit in range(teD.shape[0]):
        idx_src,idx_dst,t = teD_indx[:,cit]
        Dmat_spon[idx_dst,idx_src] += teD[cit]

    for cit in range(reD.shape[0]):
        idx_src,idx_dst,t = reD_indx[:,cit]
        Dmat_spon[idx_dst,idx_src] -= reD[cit]

    ## sort them for faster memory access later
    ## put them all into one array

    tD = np.hstack((taD,tsD,-raD,-rsD))
    tD_indx = np.hstack((taD_indx,tsD_indx,raD_indx,rsD_indx))

    return tD,tD_indx,Dmat_spon

@njit(cache=True)
def getDipoleSEE(tD,tD_indx,radj,Dmat_spon):

    for cit in range(tD.shape[0]):
        #idx_src,idx_dst,kr,t = tD_indx[:,cit]
        Dmat_spon[tD_indx[1,cit],tD_indx[0,cit]] += tD[cit] * radj[tD_indx[2,cit],tD_indx[3,cit]]

    return Dmat_spon
