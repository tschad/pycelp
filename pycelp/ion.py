'''
This module contains the Ion class for pycelp.
'''

import numpy as np
from pycelp.chianti import *
from pycelp.collisions import *
from pycelp.non_dipoles import *
from pycelp.dipoles import *
import pycelp.util as util

class Ion:
    """
    Ion class

    The ion object is the primary class used by pycelp for calculations of the
    polarized emission for a particular transition. Upon initialization, an Ion
    object loads all necessary atomic data from the CHIANTI database and
    pre-calculates all pre-factors and static terms of the statistical
    equilibrium rate equations.

    Parameters
    ----------
    ion_name : `str`
        Name of ion, e.g. 'fe_13'

    Other Parameters
    ----------------
    nlevels:  'int'
    ioneqFile : `str`, optional
        Ionization equilibrium dataset
    abundFile : `str`, optional
        Abundance dataset
    all_ks : `bool`, optional
        Flag to

    Examples
    --------
    """

    def __init__(self, ion_name=None,nlevels=None,ioneqFile=None,
                abundFile=None,all_ks = False):

        ## READ CHIANTI ATOMIC DATA
        elvl_data    = elvlcRead(ion_name)   ## ENERGY LEVEL DATA
        wgfa_data    = wgfaRead(ion_name)    ## RADIATIVE TRANSITION DATA
        scups_data   = scupsRead(ion_name)   ## ELECTRON COLLISIONAL DATA
        splups_data  = splupsRead(ion_name)  ## PROTON COLLISIONAL DATA
        abund_data   = abundRead('temp')     ## not selectable yet
        ioneq_data   = ioneqRead('temp')     ## not selectable yet

        ### REDUCE NUMBER OF CONSIDERED LEVELS
        if nlevels != None:
            elvl_data  = limit_levels(elvl_data,nlevels,type = 'elvl')
            wgfa_data  = limit_levels(wgfa_data,nlevels,type = 'wgfa')
            scups_data = limit_levels(scups_data,nlevels,type = 'scups')
            splups_data = limit_levels(splups_data,nlevels,type = 'splups')

        nlevels     = len(elvl_data['energy'])

        ### DERIVE NECESSARY ATOMIC PARAMETERS
        element, ion_stage = ion_name.split('_')
        ion_stage     = int(ion_stage)
        ionZ          = getIonZ(ion_name)
        atomicWeight  = getAtomicWeight(element)
        element_abund = abund_data['abund_val'][np.where(abund_data['abund_z'] == ionZ)][0]
        eq_logtemp    = np.copy(ioneq_data['temp'])
        eq_frac       = np.copy(ioneq_data['ionfrac'][:,ionZ-1,ion_stage-1]).clip(1.e-30)
        eq_logfrac    = np.log10(eq_frac)
        yderiv2       = util.new_second_derivative(eq_logtemp,eq_logfrac,1e100,1e100)
        qnj           = elvl_data['j']

        ######### SETUP INDICES OF THE SEE MATRIX AND GET WEIGHTS
        see_neq,see_index,see_lev,see_k,see_dk = util.setupSEE(qnj,all_ks=all_ks)
        weight = np.zeros(see_neq)
        weight[see_k == 0] =  np.sqrt((2.*qnj[see_lev[see_k == 0]]+1))

        ######### ELECTRON COLLISION RATE INITIALIZATION CALCULATIONS
        ## precalculate the spline interpolations
        scups_data['yd2'] = getSecondDerivatives(scups_data['bt_t'],scups_data['bt_upsilon'],scups_data['n_t'])
        ettype = util.get_eTransType(elvl_data,scups_data)
        elowlev = scups_data['lower_level_index']-1
        eupplev = scups_data['upper_level_index']-1
        print(' setting up electron collision rate factors')
        ciK,ciK_indx,csK,csK_indx = setup_ecoll(elowlev,eupplev,ettype,qnj,
                                                see_index,see_lev,see_k,see_dk)

        ######### PROTON COLLISION RATE INITIALIZATION CALCULATIONS
        splups_data['yd2'] = getSecondDerivatives(splups_data['bt_t'],splups_data['bt_upsilon'],splups_data['n_t'])
        plowlev = splups_data['lower_level_index']-1
        pupplev = splups_data['upper_level_index']-1
        pttype = np.zeros(len(plowlev)) - 1
        print(' setting up proton  collision rate factors')
        ciKp,ciKp_indx,csKp,csKp_indx = setup_ecoll(plowlev,pupplev,pttype,qnj,
                                                see_index,see_lev,see_k,see_dk)

        ######### RADIATIVE RATES INITIALIZATION CALCULATIONS

        rupplev = wgfa_data['upper_level_index']-1
        rlowlev = wgfa_data['lower_level_index']-1
        gup    = 2*qnj[rupplev]+1
        glo    = 2*qnj[rlowlev]+1
        alamb  = 1.e8 / (elvl_data['energy'][rupplev] - elvl_data['energy'][rlowlev])
        a_up2low = wgfa_data['A_einstein']
        hh = 6.626176e-27  ## ergs sec (planck's constant);
        cc = 2.99792458e10 ## cm s^-1 (speed of light)
        b_up2low = (alamb**3)/2./hh/cc/1.e24*a_up2low
        b_low2up = gup/glo*b_up2low
        nrad = len(rlowlev)

        wv_air = util.vac2air(alamb)

        ## --> Non-dipoles
        print(' setting up non-dipole radiative rate factors')
        tnD,tnD_indx,nonD_spon= setup_nonDipoles(rlowlev,rupplev,qnj,b_low2up,a_up2low,
                                                b_up2low,see_index,see_lev,see_k,see_dk)

        ## ---> Dipoles
        print(' setting up dipole radiative rate factors')
        tD,tD_indx,Dmat_spon = setup_Dipoles(rlowlev,rupplev,qnj,b_low2up,a_up2low,b_up2low,see_index,see_lev,see_k,see_dk)

        #################################################
        ## put into instance information
        self.ion_name = ion_name
        self.all_ks = all_ks
        ########################
        self.nlevels = nlevels
        self.elvl_data    = elvl_data
        self.wgfa_data    = wgfa_data
        self.scups_data   = scups_data
        self.splups_data  = splups_data
        self.abund_data   = abund_data
        self.ioneq_data   = ioneq_data
        ########################
        self.element = element
        self.ion_stage = ion_stage
        self.ionZ = ionZ
        self.atomicWeight = atomicWeight
        self.element_abund = element_abund
        self.ioneq_logtemp = eq_logtemp
        self.ioneq_frac    = eq_frac
        self.ioneq_logfrac = eq_logfrac
        self.ioneq_yderiv2 = yderiv2
        self.qnj = qnj
        ########################
        self.see_neq    = see_neq
        self.see_index  = see_index
        self.see_lev    = see_lev
        self.see_k      = see_k
        self.see_dk     = see_dk
        self.weight     = weight
        ########################
        self.scups_data = scups_data
        self.elowlev    = elowlev
        self.eupplev    = eupplev
        self.ciK        = ciK
        self.ciK_indx   = ciK_indx
        self.csK        = csK
        self.csK_indx   = csK_indx
        ########################
        self.splups_data = splups_data
        self.plowlev     = plowlev
        self.pupplev     = pupplev
        self.ciKp        = ciKp
        self.ciKp_indx   = ciKp_indx
        self.csKp        = csKp
        self.csKp_indx   = csKp_indx
        ########################
        ## radiative transition info
        self.rlowlev    = rlowlev
        self.rupplev    = rupplev
        self.alamb      = alamb
        self.wv_air     = wv_air
        self.a_up2low   = a_up2low
        self.b_up2low   = b_up2low
        self.b_low2up   = b_low2up
        self.tnD        = tnD
        self.tnD_indx   = tnD_indx
        self.nonD_spon  = nonD_spon
        self.tD         = tD
        self.tD_indx    = tD_indx
        self.Dmat_spon  = Dmat_spon

    def __repr__(self):
         return f"""pyCELP Ion class
    ---------------------
    Ion Name: {self.ion_name}
    Number of energy levels included: {self.nlevels}
    Number of SEE equations: {self.see_neq}
    Number of Radiative Transitions: {len(self.alamb)}
    Ionization Equilbrium Filename: {self.ioneq_data['filename']}"""

    def calc_ecoll_matrix_standard(self,edens,etemp):
        wkzero = np.argwhere(self.see_k == 0)[:,0]
        ecmat_std = np.zeros((self.nlevels,self.nlevels))
        for n in range(self.nlevels):
            ecmat_std[n,:] = self.ecmat[wkzero[n],wkzero] * self.weight[wkzero[n]]/self.weight[wkzero]
        return ecmat_std

    def calc_pcoll_matrix_standard(self):
        wkzero = np.argwhere(self.see_k == 0)[:,0]
        pcmat_std = np.zeros((self.nlevels,self.nlevels))
        for n in range(self.nlevels):
            pcmat_std[n,:] = self.pcmat[wkzero[n],wkzero] * self.weight[wkzero[n]]/self.weight[wkzero]
        return pcmat_std

    def calc_dipole_matrix_standard(self,ht,thetab):
        radj    = util.rad_field_bframe(self.alamb,thetab,ht,limbd_flag = True)
        Dmat    = getDipoleSEE(self.tD,self.tD_indx,radj,np.copy(self.Dmat_spon))
        wkzero = np.argwhere(self.see_k == 0)[:,0]
        Dmat_std = np.zeros((self.nlevels,self.nlevels))
        for n in range(self.nlevels):
            Dmat_std[n,:] = Dmat[wkzero[n],wkzero] * self.weight[wkzero[n]]/self.weight[wkzero]
        return Dmat_std

    def calc_nonDipole_matrix_standard(self,ht,thetab):
        radj    = util.rad_field_bframe(self.alamb,thetab,ht,limbd_flag = True)
        nonDmat = getNonDipoleSEE(self.tnD,self.tnD_indx,radj,np.copy(self.nonD_spon))
        wkzero = np.argwhere(self.see_k == 0)[:,0]
        nonDmat_std = np.zeros((self.nlevels,self.nlevels))
        for n in range(self.nlevels):
            nonDmat_std[n,:] = nonDmat[wkzero[n],wkzero] * self.weight[wkzero[n]]/self.weight[wkzero]
        return nonDmat_std

    def calc_rad_matrix_standard(self,ht,thetab):
        radj    = util.rad_field_bframe(self.alamb,thetab,ht,limbd_flag = True)
        nonDmat = getNonDipoleSEE(self.tnD,self.tnD_indx,radj,np.copy(self.nonD_spon))
        Dmat    = getDipoleSEE(self.tD,self.tD_indx,radj,np.copy(self.Dmat_spon))
        DD = nonDmat + Dmat
        wkzero = np.argwhere(self.see_k == 0)[:,0]
        radmat_std = np.zeros((self.nlevels,self.nlevels))
        for n in range(self.nlevels):
            radmat_std[n,:] = DD[wkzero[n],wkzero] * self.weight[wkzero[n]]/self.weight[wkzero]
        return radmat_std

    def calc(self,edens,etemp,ht,thetab,include_limbdark = True,include_protons = True):
        """
        Calculates rho
        """
        thetab = np.deg2rad(thetab)
        ptemp = etemp
        toth  = 0.85*edens
        pdens = 1.*toth
        if not include_protons:
            pdens = 0.

        self.edens = edens
        self.pdens = pdens
        self.toth  = toth
        self.thetab_rad = thetab

        eq_frac_int  = 10.**(util.spintone(np.log10(etemp),self.ioneq_logtemp,self.ioneq_logfrac,self.ioneq_yderiv2))
        totn         = 10.**(self.element_abund-12.)*toth*eq_frac_int

        erates_up,erates_down = intErates(self.elowlev,self.eupplev,self.qnj, \
                                          self.scups_data['delta_energy'],self.scups_data['bt_c'], \
                                          self.scups_data['bt_type'],self.scups_data['bt_t'], \
                                          self.scups_data['bt_upsilon'],self.scups_data['yd2'], \
                                          etemp,edens)

        prates_up,prates_down = intPrates(self.plowlev,self.pupplev,self.qnj, \
                                          self.splups_data['delta_energy'],self.splups_data['bt_c'], \
                                          self.splups_data['bt_type'],self.splups_data['bt_t'], \
                                          self.splups_data['bt_upsilon'],self.splups_data['yd2'], \
                                          ptemp,pdens)

        self.radj    = util.rad_field_bframe(self.alamb,thetab,ht,include_limbdark = include_limbdark)

        ecmat   = getElectronSEE(self.ciK,self.ciK_indx,self.csK,self.csK_indx,
                                 erates_up,erates_down,self.see_neq)
        pcmat   = getElectronSEE(self.ciKp,self.ciKp_indx,self.csKp,self.csKp_indx,
                                 prates_up,prates_down,self.see_neq)
        nonDmat = getNonDipoleSEE(self.tnD,self.tnD_indx,self.radj,np.copy(self.nonD_spon))
        Dmat    = getDipoleSEE(self.tD,self.tD_indx,self.radj,np.copy(self.Dmat_spon))

        self.collmat = ecmat + pcmat
        self.radmat = Dmat + nonDmat
        self.see_matrix = self.collmat + self.radmat

        rho     = util.seeSolve(self.see_matrix,self.weight,self.see_lev,self.see_k)

        self.rho = rho
        self.totn = totn

    def get_totn(self):
        return self.totn

    def get_rho(self):
        return self.rho

    def get_maxtemp(self):
        """ Returns the temperature at maximum ionization fraction """
        logt = np.linspace(5,7,50)
        eq_frac_int  = 10.**(util.spintarr(logt,self.ioneq_logtemp,self.ioneq_logfrac,self.ioneq_yderiv2))
        logt_max = logt[np.argmax(eq_frac_int)]
        return 10.**logt_max

    def get_lower_level_alignment(self,wv_air):
        """ Returns the atomic alignment for upper level of given transition
        input -- wv_air == air wavelength in Angstrom
        """
        ww = np.argmin(np.abs(self.wv_air - wv_air))
        lowlev = self.rlowlev[ww]
        alignment = self.rho[lowlev,2] / self.rho[lowlev,0]
        return alignment

    def get_upper_level_alignment(self,wv_air):
        """ Returns the atomic alignment for upper level of given transition
        input -- wv_air == air wavelength in Angstrom
        """
        ww = np.argmin(np.abs(self.wv_air - wv_air))
        upplev = self.rupplev[ww]
        alignment = self.rho[upplev,2] / self.rho[upplev,0]
        return alignment

    def get_upper_level_rho00(self,wv_air):
        """ Returns the atomic alignment for upper level of given transition
        input -- wv_air == air wavelength in Angstrom
        """
        ww = np.argmin(np.abs(self.wv_air - wv_air))
        upplev = self.rupplev[ww]
        rho00 = self.rho[upplev,0]
        return rho00

    def calc_Iemiss(self,wv_air):
        """ returns the intensity
        Note:  currently does not include geometry or atomic polarization
        """
        ww = np.argmin(np.abs(self.wv_air - wv_air))
        if ((self.wv_air[ww] - wv_air)/wv_air > 0.05):
            print(' warning: requested wavelength for calculation does have a good match')
            print(' requested: ', wv_air)
            print(' closest:   ', self.wv_air[ww])
            raise

        upplev = self.rupplev[ww]
        hh = 6.626176e-27  ## ergs sec (planck's constant);
        cc = 2.99792458e10 ## cm s^-1 (speed of light)
        hnu = hh*cc / (self.alamb[ww]/1.e8)
        Ju = self.qnj[upplev]
        ## convert units
        sr2arcsec = (180./np.pi)**2.*3600.**2.
        phergs = hh*(3.e8)/(self.alamb[ww] * 1.e-10)
        val = hnu/4./np.pi*self.a_up2low[ww] *np.sqrt(2.*Ju+1)*self.rho[upplev,0] * self.totn
        val = val/sr2arcsec/phergs
        return val

    def show_lines(self,nlines=None,start=0):
        """ prints out information on the radiative transitions"""

        if (nlines == None):
            nlines = len(self.alamb)
            start = 0

        assert start >= 0
        assert start <= len(self.alamb)
        assert nlines <= len(self.alamb)
        print(' Index -- WV_VAC [A] -- WV_AIR [A] -- TRANSITION')
        for ln in range(start,start+nlines):
            wv = np.round(self.alamb[ln],3)
            wvair = np.round(self.wv_air[ln],3)
            upplev,lowlev = self.rupplev[ln],self.rlowlev[ln]
            print(ln, wv,wvair, self.elvl_data['full_level'][upplev], ' --> ', self.elvl_data['full_level'][lowlev])


    ## get stokes
    ## plot
