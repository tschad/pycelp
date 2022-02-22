'''
This module contains the emissionLine class for pycelp.
'''

import numpy as np

hh = 6.626176e-27  ## ergs sec (planck's constant);
cc = 2.99792458e10 ## cm s^-1 (speed of light)

class emissionLine:
    
    def __init__(self,Ion,wv_air):  
        
        ww = np.argmin(np.abs(Ion.wv_air - wv_air))

        if ((Ion.wv_air[ww] - wv_air)/wv_air > 0.05):
            print(' warning: requested wavelength for calculation does have a good match')
            print(' requested: ', wv_air)
            print(' closest:   ', Ion.wv_air[ww])
            raise
          
        self.ion_name = Ion.ion_name
        self.atomic_weight = Ion.atomicWeight
        self.transitionIndex = ww
        self.wavelength_in_air = Ion.wv_air[ww]
        self.wavelength_in_vacuum = Ion.alamb[ww]
        self.upper_level_index = Ion.rupplev[ww]
        self.lower_level_index = Ion.rlowlev[ww]      
        self.upper_level_config = Ion.elvl_data['full_level'][self.upper_level_index]
        self.lower_level_config = Ion.elvl_data['full_level'][self.lower_level_index]   
        self.geff = Ion.geff[ww]  ## get Lande geff in LS coupling
        self.Einstein_A = Ion.a_up2low[ww]
        self.Dcoeff = Ion.Dcoeff[ww]
        self.Ecoeff = Ion.Ecoeff[ww]
        self.Jupp = Ion.qnj[self.upper_level_index]
        wv_vac_cm = self.wavelength_in_vacuum * 1.e-8 
        hnu = hh*cc / (wv_vac/1.e8)
        self.hnu = hnu         
        try: 
            self.upper_level_alignment = Ion.rho[self.upper_level_index,2] / Ion.rho[self.upper_level_index,0]
            self.upper_level_rho00 = Ion.rho[self.upper_level_index,0]
            self.lower_level_alignment = Ion.rho[self.lower_level_index,2] / Ion.rho[self.lower_level_index,0]
            self.total_ion_population = Ion.totn
            self.upper_level_pop_frac =  np.sqrt(2.*self.Jupp+1)*self.upper_level_rho00  ## Calculate population in standard representation 
            self.electron_temperature = Ion.etemp 
            self.electron_density = Ion.edens
            self.C_coeff =   self.hnu/4./np.pi * self.Einstein_A * self.upper_level_pop_frac * self.total_ion_population
        except: 
            self.upper_level_alignment  = None 
            self.upper_level_rho00 = None
            self.lower_level_alignment = None
            self.total_ion_population = None 
            self.upper_level_pop_frac = None 
            self.C_coeff = None
            self.electron_temperature = None
            self.electron_density = None
        
    def __repr__(self):
        print("pyCELP emissionLine class instance") 
        print("----------------------------------")
        for ky in self.__dict__: 
            print('   ',ky,'  : ',self.__dict__[ky])
        return """----------------------------------"""
        
    def calc_PolEmissCoeff(self,magnetic_field_amplitude,thetaBLOSdeg,azimuthBLOSdeg=0.): 
        """ returns the polarized emission coefficent for the emission line
        
        return units: 
        I,Q,U :: photons cm$^{-3}$ s$^{-1}$ arcsec$^{-2}$
        V     ::  photons cm$^{-3}$ s$^{-1}$ arcsec$^{-2}$ Angstrom^{-1}

        Parameters
        ----------
        magnetic_field_amplitude: float (unit: gauss)
            Total magnitude of the magnetic field 
        thetaBLOSdeg : float (unit: degrees)
            inclination angle of the magnetic field relative to the line of sight
        azimuthBLOSdeg : float (unit: degrees) [default = 0] 
            azimuth angle of magnetic field relative to coordinate frame aligned with 
            the line-of-sight projected magnetic field orientation. 
        """
    
        ## convert to radians 
        thetaBLOS = np.deg2rad(thetaBLOSdeg) 
        azimuthBLOS = np.deg2rad(azimuthBLOSdeg)
        ALARMOR = 1399612.2*magnetic_field_amplitude    ## Get Larmor frequency in units of s^-1 

        ## scaling coefficent for the Stokes V emission coefficient 
        wv_vac = self.wavelength_in_vacuum
        wv_vac_cm = wv_vac * 1.e-8 
        Vscl = - (wv_vac_cm)**2   / cc  * 1.e8  ## units of Angstrom * s 
        hnu = hh*cc / (wv_vac/1.e8)
        
        ## Common coefficent related to populations        
        C_coeff = hnu/4./np.pi * self.Einstein_A * self.upper_level_pop_frac * self.total_ion_population
        epsI = C_coeff * (1. + 3./(2.*np.sqrt(2.)) * self.Dcoeff * self.upper_level_alignment* (np.cos(thetaBLOS)**2 - (1./3.)  )   )
        epsQ = C_coeff*(3./(2.*np.sqrt(2.)))*(np.sin(thetaBLOS)**2)*self.Dcoeff*self.upper_level_alignment
        epsU = 0
        epsV = Vscl * C_coeff*np.cos(thetaBLOS)*ALARMOR*(self.geff + self.Ecoeff*self.upper_level_alignment)

        ## rotate for the azimuthal direction 
        epsQr =  np.cos(2.*azimuthBLOS)*epsQ+ np.sin(2.*azimuthBLOS)*epsU
        epsUr = -np.sin(2.*azimuthBLOS)*epsQ + np.cos(2.*azimuthBLOS)*epsU
        epsQ = epsQr
        epsU = epsUr 
        
        ## convert to returned units 
        sr2arcsec = (180./np.pi)**2.*3600.**2.
        phergs = hh*(3.e8)/(wv_vac * 1.e-10)
        self.epsI = epsI/sr2arcsec/phergs        
        self.epsQ = epsQ/sr2arcsec/phergs        
        self.epsU = epsU/sr2arcsec/phergs        
        self.epsV = epsV/sr2arcsec/phergs        
        
        return epsI, epsQ, epsU, epsV 
    
    def calc_Iemiss(self,thetaBLOSdeg = np.rad2deg(np.arccos(1./np.sqrt(3.))) ):
        """ returns the line-integrated intensity emission coefficent for a selected transition
        
        return units are photons cm$^{-3}$ s$^{-1}$ arcsec$^{-2}$

        Parameters
        ----------
        wv_air : float (unit: angstroms)
            Air wavelength of spectral line
        thetaBLOS : float (unit: degrees)
            inclination angle of the magnetic field relative to the line of sight
            default is van vleck
        """
        magnetic_field_amplitude = 0.
        epsI, epsQ, epsU, epsV = self.calc_PolEmissCoeff(magnetic_field_amplitude,thetaBLOSdeg)
        epsI_units = r'photons cm$^{-3}$ s$^{-1}$ arcsec$^{-2}$'
        return epsI,epsI_units

    def calc_stokesSpec(self,magnetic_field_amplitude,thetaBLOSdeg,
                       azimuthBLOSdeg=0., 
                       doppler_velocity = 0.,
                       non_thermal_turb_velocity = 0.,
                       doppler_spectral_range = (-120,120),
                       specRes_wv_over_dwv = 100000):        
        
        """ calculate Stokes spectra after the statistical equilibrium has been solved 
        
        Assumes current electron temperature dictates the thermal line width. 
        
        return units: 
        I,Q,U ::  photons cm$^{-3}$ s$^{-1}$ arcsec$^{-2} Angstrom^{-1}$
        V     ::  photons cm$^{-3}$ s$^{-1}$ arcsec$^{-2} Angstrom^{-1}$ 

        Parameters
        ----------
        wv_air : float (unit: angstroms)
            Rest air wavelength of spectral line (needs to be close to database value)
        magnetic_field_amplitude: float (unit: gauss)
            Total magnitude of the magnetic field 
        thetaBLOS : float (unit: degrees)
            inclination angle of the magnetic field relative to the line of sight
        azimuthBLOS : float (unit: degrees) [default = 0] 
            azimuth angle of magnetic field relative to coordinate frame aligned with 
            the line-of-sight projected magnetic field orientation. 
        doppler_velocity : float (unit: km/s)
            Doppler velocity of the spectral line 
        non_thermal_turb_velocity  : float (unit: km/s) 
            Non-thermal velocity convolved with the thermal component of the line width 
        doppler_spectral_range : float 2-tuple (unit : km/2) 
            Range of the spectrum calculated in Doppler velocity space relative to the line rest wavelength
        specRes_wv_over_dwv: float (unitless) 
            The sampling resolution of the wavelength vector given as the ratio of the wavelength to the sampling. 
            
        """
   
        ## Get polarized emission coefficients 
        epsI, epsQ, epsU, epsV = self.calc_PolEmissCoeff(magnetic_field_amplitude,  thetaBLOSdeg, azimuthBLOSdeg)
        
        ## setup wavelength vector 
        assert doppler_spectral_range[1]>doppler_spectral_range[0]
        dVel = 3e5 / specRes_wv_over_dwv
        nwv = np.ceil((doppler_spectral_range[1] - doppler_spectral_range[0]) / dVel).astype(int)
        velvec = np.linspace(*doppler_spectral_range,nwv)  ##;; velocity range used for the spectral axis
        wvvec = (self.wavelength_in_air  *1.e-10)*(1. + velvec/3.e5)  ## in units of meters at this point 

        ## Calculate Gaussian Line Width        
        awgt = self.atomic_weight 
        M = (awgt*1.6605655e-24)/1000.   ## kilogram
        kb = 1.380648e-23  ## J K^-1 [ = kg m^2 s^-2 K^-1]
        etemp = self.electron_temperature
        turbv = non_thermal_turb_velocity
        sig = (1./np.sqrt(2.))*(self.wavelength_in_air*1.e-10/3.e8)*np.sqrt(2.*kb*etemp/M + (turbv*1000.)**2.)

        ## calculate line center position 
        wv0 = (self.wavelength_in_air*1.e-10) + (doppler_velocity/3.e5)*(self.wavelength_in_air*1.e-10)    ## in meters 
        wv0 = wv0*1.e10     ## convert to Angstrom 
        sig = sig*1.e10      ## convert to Angstrom 
        wvvec = wvvec*1.e10  ## convert to Angstrom 
        
        ## normalized Gaussian profile
        wprof  = 1./(np.sqrt(2.*np.pi)*sig) * np.exp(-(wvvec-wv0)**2./(2.*sig**2.))
        ## normalized Gaussian derivative profiles 
        wprof_deriv = (- (wvvec-wv0) / sig**2)  * wprof 

        stokes = np.zeros((nwv,4) )
        stokes[:,0] = epsI*wprof
        stokes[:,1] = epsQ*wprof
        stokes[:,2] = epsU*wprof
        stokes[:,3] = epsV*wprof_deriv

        return wvvec,stokes