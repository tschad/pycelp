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
    The ion object is the primary class used by pycelp for calculations of the
    polarized emission for a particular transition. Upon initialization, an Ion
    object loads all necessary atomic data from the CHIANTI database and
    pre-calculates all pre-factors and static terms of the statistical
    equilibrium rate equations.

    Parameters
    ----------
    ion_name : str
        Name of ion (e.g., 'fe_13')
    nlevels:  int (default = None)
        Number of energy levels to include (default is all)
    ioneqFile : str (default = None)
        Ionization equilibrium filename (defaults to Chianti default file)
    abundFile : str (default = None)
        Abundance filename (defaults to sun_photospheric_2009_asplund.abund)
    all_ks : bool (default = False)
        Flag to include all multipole order Ks in the calculation
        The default is to include only the even values of K for the no
        coherence hypothesis, as discussed in LD&L (2004) Section 13.5.

    References
    ----------
    Egidio Landi Degl’innocenti and Marco Landolfi (2004)
    "Polarization in Spectral Lines"
    <https://link.springer.com/book/10.1007/1-4020-2415-0>

    """

    def __init__(self, ion_name,nlevels=None,ioneqFile=None,
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

        ### DERIVE D and E coefficients
        ss,ll,jj = elvl_data['s'],elvl_data['l'],elvl_data['j']
        landeg = util.calcLande(jj,ss,ll)
        Jupp,Jlow = jj[rupplev],jj[rlowlev]
        gupp,glow = landeg[rupplev],landeg[rlowlev]
        Dcoeff = util.getDcoeff(Jupp,Jlow)
        Ecoeff = util.getEcoeff(Jupp,Jlow,gupp,glow)
        geff = 0.5*(glow+gupp) + 0.25 * (glow-gupp) * (Jlow*(Jlow+1.) - Jupp*(Jupp+1.))
        
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
        self.landeg       = landeg
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
        self.Dcoeff     = Dcoeff
        self.Ecoeff     = Ecoeff
        self.geff       = geff
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

    def get_maxtemp(self):
        """ Returns the temperature at maximum ionization fraction in Kelvin """
        logt = np.linspace(5,7,50)
        eq_frac_int  = 10.**(util.spintarr(logt,self.ioneq_logtemp,self.ioneq_logfrac,self.ioneq_yderiv2))
        logt_max = logt[np.argmax(eq_frac_int)]
        return 10.**logt_max

    def calc_rho_sym(self,edens,etemp,height,thetab,include_limbdark = True,
                    include_protons = True):
        """
        Calculates the elements of the atomic density matrix (rho) for the
        case of a cylindrically symmetric radiation field.

        Parameters
        ----------
        edens : float (units: cm^-3)
            Electron density (e.g., 1e8)
        etemp : float (units: K)
            Electron temperature (e.g., 1.e6)
        height : float (units: fraction of a solar radius)
            Height above the solar photosphere
        thetab : float (units: degrees)
            Inclination angle of the magnetic field relative to the solar
            vertical (i.e. 0 == vertical, 90 = horizontal)

        Other Parameters
        ----------
        include_limbdark:  bool (default: True)
            Flag to include limb darkening in the radiation field calculation
        include_protons:  bool (default: True)
            Flat to include protons in the collisional rates

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

        self.radj    = util.rad_field_bframe(self.alamb,thetab,height,include_limbdark = include_limbdark)

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

    def calc_rho_radj(self,edens,etemp,radj,include_protons = True):
        """
        Calculates the elements of the atomic density matrix (rho) in the case
        that the radiation field tensor components are given as a input.

        Parameters
        ----------
        edens : float (units: cm^-3)
            Electron density (e.g., 1e8)
        etemp : float (units: K)
            Electron temperature (e.g., 1.e6)
        radj : float,nparray (shape is (3, number of radiative transitions))
            Radiation field tensor components (QK = 00,01,02) for all
            radiative transitions in the ion

        Other Parameters
        ----------
        include_protons:  bool (default: True)
            Flat to include protons in the collisional rates

        """

        ptemp = etemp
        toth  = 0.85*edens
        pdens = 1.*toth
        if not include_protons:
            pdens = 0.

        self.edens = edens
        self.pdens = pdens
        self.toth  = toth

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

        self.radj    = radj

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

    def calc_ecoll_matrix_standard(self):
        """ to be documented - dev use only for now """
        wkzero = np.argwhere(self.see_k == 0)[:,0]
        ecmat_std = np.zeros((self.nlevels,self.nlevels))
        for n in range(self.nlevels):
            ecmat_std[n,:] = self.ecmat[wkzero[n],wkzero] * self.weight[wkzero[n]]/self.weight[wkzero]
        return ecmat_std

    def calc_pcoll_matrix_standard(self):
        """ to be documented - dev use only for now """
        wkzero = np.argwhere(self.see_k == 0)[:,0]
        pcmat_std = np.zeros((self.nlevels,self.nlevels))
        for n in range(self.nlevels):
            pcmat_std[n,:] = self.pcmat[wkzero[n],wkzero] * self.weight[wkzero[n]]/self.weight[wkzero]
        return pcmat_std

    def calc_dipole_matrix_standard(self,ht,thetab):
        """ to be documented - dev use only for now """
        radj    = util.rad_field_bframe(self.alamb,thetab,ht,limbd_flag = True)
        Dmat    = getDipoleSEE(self.tD,self.tD_indx,radj,np.copy(self.Dmat_spon))
        wkzero = np.argwhere(self.see_k == 0)[:,0]
        Dmat_std = np.zeros((self.nlevels,self.nlevels))
        for n in range(self.nlevels):
            Dmat_std[n,:] = Dmat[wkzero[n],wkzero] * self.weight[wkzero[n]]/self.weight[wkzero]
        return Dmat_std

    def calc_nonDipole_matrix_standard(self,ht,thetab):
        """ to be documented - dev use only for now """
        radj    = util.rad_field_bframe(self.alamb,thetab,ht,limbd_flag = True)
        nonDmat = getNonDipoleSEE(self.tnD,self.tnD_indx,radj,np.copy(self.nonD_spon))
        wkzero = np.argwhere(self.see_k == 0)[:,0]
        nonDmat_std = np.zeros((self.nlevels,self.nlevels))
        for n in range(self.nlevels):
            nonDmat_std[n,:] = nonDmat[wkzero[n],wkzero] * self.weight[wkzero[n]]/self.weight[wkzero]
        return nonDmat_std

    def calc_rad_matrix_standard(self,ht,thetab):
        """ to be documented - dev use only for now """
        radj    = util.rad_field_bframe(self.alamb,thetab,ht,limbd_flag = True)
        nonDmat = getNonDipoleSEE(self.tnD,self.tnD_indx,radj,np.copy(self.nonD_spon))
        Dmat    = getDipoleSEE(self.tD,self.tD_indx,radj,np.copy(self.Dmat_spon))
        DD = nonDmat + Dmat
        wkzero = np.argwhere(self.see_k == 0)[:,0]
        radmat_std = np.zeros((self.nlevels,self.nlevels))
        for n in range(self.nlevels):
            radmat_std[n,:] = DD[wkzero[n],wkzero] * self.weight[wkzero[n]]/self.weight[wkzero]
        return radmat_std

    def get_lower_level_alignment(self,wv_air):
        """ Returns the atomic alignment for the lower level of given transition

        Parameters
        ----------
        wv_air : float (unit: angstroms)
            Air wavelength of spectral line
        """
        ww = np.argmin(np.abs(self.wv_air - wv_air))

        if ((self.wv_air[ww] - wv_air)/wv_air > 0.05):
            print(' warning: requested wavelength for calculation does have a good match')
            print(' requested: ', wv_air)
            print(' closest:   ', self.wv_air[ww])
            raise

        lowlev = self.rlowlev[ww]
        alignment = self.rho[lowlev,2] / self.rho[lowlev,0]
        return alignment

    def get_upper_level_alignment(self,wv_air):
        """ Returns the atomic alignment for the upper level of given transition

        Parameters
        ----------
        wv_air : float (unit: angstroms)
            Air wavelength of spectral line
        """
        ww = np.argmin(np.abs(self.wv_air - wv_air))

        if ((self.wv_air[ww] - wv_air)/wv_air > 0.05):
            print(' warning: requested wavelength for calculation does have a good match')
            print(' requested: ', wv_air)
            print(' closest:   ', self.wv_air[ww])
            raise

        upplev = self.rupplev[ww]
        alignment = self.rho[upplev,2] / self.rho[upplev,0]
        return alignment

    def get_upper_level_rho00(self,wv_air):
        """ Returns rho(Q=0,K=0) for the upper level of given transition

        Parameters
        ----------
        wv_air : float (unit: angstroms)
            Air wavelength of spectral line
        """
        ww = np.argmin(np.abs(self.wv_air - wv_air))

        if ((self.wv_air[ww] - wv_air)/wv_air > 0.05):
            print(' warning: requested wavelength for calculation does have a good match')
            print(' requested: ', wv_air)
            print(' closest:   ', self.wv_air[ww])
            raise

        upplev = self.rupplev[ww]
        rho00 = self.rho[upplev,0]
        return rho00

    def get_EinsteinA(self,wv_air):
        """ Returns the Einstein A for a selected transition

        Parameters
        ----------
        wv_air : float (unit: angstroms)
            Air wavelength of spectral line
        """
        ww = np.argmin(np.abs(self.wv_air - wv_air))

        if ((self.wv_air[ww] - wv_air)/wv_air > 0.05):
            print(' warning: requested wavelength for calculation does have a good match')
            print(' requested: ', wv_air)
            print(' closest:   ', self.wv_air[ww])
            raise

        return self.a_up2low[ww]

    def get_Dcoeff(self,wv_air):
        """ Returns the D coefficent for a selected transition

        Parameters
        ----------
        wv_air : float (unit: angstroms)
            Air wavelength of spectral line
        """
        ww = np.argmin(np.abs(self.wv_air - wv_air))

        if ((self.wv_air[ww] - wv_air)/wv_air > 0.05):
            print(' warning: requested wavelength for calculation does have a good match')
            print(' requested: ', wv_air)
            print(' closest:   ', self.wv_air[ww])
            raise

        return self.Dcoeff[ww]

    def get_Ecoeff(self,wv_air):
        """ Returns the E coefficent for a selected transition

        Parameters
        ----------

        wv_air : float (unit: angstroms)
            Air wavelength of spectral line
        """
        ww = np.argmin(np.abs(self.wv_air - wv_air))

        if ((self.wv_air[ww] - wv_air)/wv_air > 0.05):
            print(' warning: requested wavelength for calculation does have a good match')
            print(' requested: ', wv_air)
            print(' closest:   ', self.wv_air[ww])
            raise

        return self.Ecoeff[ww]

    def get_Jupp(self, wv_air):
        """ Returns the E coefficent for a selected transition

        Parameters
        ----------

        wv_air : float (unit: angstroms)
            Air wavelength of spectral line
        """
        ww = np.argmin(np.abs(self.wv_air - wv_air))

        if ((self.wv_air[ww] - wv_air)/wv_air > 0.05):
            print(' warning: requested wavelength for calculation does have a good match')
            print(' requested: ', wv_air)
            print(' closest:   ', self.wv_air[ww])
            raise

        upplev = self.rupplev[ww]
        return self.qnj[upplev]

    def calc_Iemiss(self,wv_air,thetaBLOS = np.rad2deg(np.arccos(1./np.sqrt(3.))) ):
        """ returns the intensity emission coefficent for a selected transition
        return units are photons

        Parameters
        ----------
        wv_air : float (unit: angstroms)
            Air wavelength of spectral line
        thetaBLOS : float (unit: degrees)
            inclination angle of the magnetic field relative to the line of sight
            default is van vleck
        """
        ww = np.argmin(np.abs(self.wv_air - wv_air))

        if ((self.wv_air[ww] - wv_air)/wv_air > 0.05):
            print(' warning: requested wavelength for calculation does have a good match')
            print(' requested: ', wv_air)
            print(' closest:   ', self.wv_air[ww])
            raise

        upplev = self.rupplev[ww]
        Dcoeff = self.Dcoeff[ww]
        sigma = self.get_upper_level_alignment(wv_air)
        thetaBLOS = np.deg2rad(thetaBLOS)

        hh = 6.626176e-27  ## ergs sec (planck's constant);
        cc = 2.99792458e10 ## cm s^-1 (speed of light)
        hnu = hh*cc / (self.alamb[ww]/1.e8)
        Ju = self.qnj[upplev]

        ## convert units to
        sr2arcsec = (180./np.pi)**2.*3600.**2.
        phergs = hh*(3.e8)/(self.alamb[ww] * 1.e-10)
        val = hnu/4./np.pi*self.a_up2low[ww] * np.sqrt(2.*Ju+1)*self.rho[upplev,0] * self.totn
        val = val * (1. + 3./(2.*np.sqrt(2.)) * (np.cos(thetaBLOS)**2 - (1./3.)  )   )
        val = val/sr2arcsec/phergs

        return val

    def show_lines(self,nlines=None,start=0):
        """ prints out information for the radiative transitions

        Parameters
        ----------
        nlines : int (default = None)
            The number of spectral lines to print
        start:  int (default = 0)
            Starting index of lines printed.
            Lines are ordered roughly by energy level indices
        """

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

    def energy_levels_to_dataframe(self): 
        
        try: 
            import pandas as pd
        except: 
            print(' To print Energy Level Information using dataframe requires pandas package to be installed')
            
        energy_units = self.elvl_data['energy_units']
        
        d = {'Index'         : self.elvl_data['index'], 
             'Ion_name'      : np.repeat(self.ion_name,self.nlevels),
             'Ion_z'         : np.repeat(self.elvl_data['ion_z'],self.nlevels),
             'Configuration' : self.elvl_data['conf'], 
             'Conf Index'    : self.elvl_data['conf_index'],
             'Term'          : self.elvl_data['term'], 
             'Level'         : self.elvl_data['level'], 
             'Full Level'    : self.elvl_data['full_level'],
             'Label'         : self.elvl_data['label'],
             'Spin Multiplicity'  : self.elvl_data['mult'],
             'S'      : self.elvl_data['s'],
             'L'      : self.elvl_data['l'],
             'L Symbol' : self.elvl_data['l_sym'],
             'J'      : self.elvl_data['j'],
             'Lande g' : self.landeg,
             'Parity' : self.elvl_data['parity'],
             'Parity Str' : self.elvl_data['parity_str'],
             'Stat. Weight'  : self.elvl_data['weight'],
             'Obs Energy [' + energy_units + ']' : self.elvl_data['obs_energy'],
             'Theory Energy [' + energy_units + ']' : self.elvl_data['theory_energy'],
             'Energy [' + energy_units + ']' : self.elvl_data['energy']}
 
        df = pd.DataFrame(data=d)
    
        return df 
