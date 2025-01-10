"""
Provides access to solar spectral distribution information available in Allen's
Astrophysical Quantities 4th Edition (Section 14.6), including disk center
spectral radiance and the center-to-limb behavior.
"""
import numpy as np

########################################
## GLOBAL DATA FIRST
## TABLE 14.13: Wavelengths in microns and I_lambda^prime(0)
## Ipr is "intensity of the center of the Sun's disk between spectrum lines"

lambdaIpr = 1.e4 * np.asarray([0.2 , 0.22, 0.24, 0.26, 0.28, 0.3 , 0.32, 0.34, 0.36,
               0.37, 0.38, 0.39, 0.4 , 0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.48,
               0.5 , 0.55, 0.6 , 0.65, 0.7 , 0.75, 0.8 , 0.9 , 1.  , 1.1 , 1.2 ,
               1.4 , 1.6 , 1.8 , 2.  , 2.5 , 3.  , 4.  , 5.  ])
lambdaIpr_unit = r'$\AA$'

Ipr = 1.e3 * np.asarray([0.02,0.19,0.21,0.53,1.21,2.39,2.94,
                         3.30,3.47,3.60,4.14,4.41,4.58,4.63,
                         4.66,4.67,4.62,4.55,4.44,4.22,4.08,
                         3.63,3.24,2.90,2.52,2.24,1.97,1.58,
                         1.26,1.01,0.84,0.56,0.40,0.27,0.18,
                         0.081,0.041,0.0135,0.0057])
Ipr_unit = r'W m$^{-2}$ sr$^{-1}$ $\AA^{-1}$'

Fpr = 1.e3 *np.asarray([0.014,0.10,0.13,0.27,0.68,1.48,1.97,
                        2.39,2.56,2.67,2.99,3.21,3.35,3.42,
                        3.47,3.50,3.49,3.47,3.41,3.28,3.20,
                        2.93,2.67,2.41,2.13,1.92,1.71,1.39,
                        1.12,0.90,0.76,0.51,0.37,0.25,0.17,
                        0.076,0.039,0.013,0.0055])

## convert to 'erg / (s cm^2 sr AA)'
## W = J/s = 1.e7 ergs/s
## 1/m^2 * (1 m^2)/(100.^2 cm^2 ) = 1./100**2. = 1/1.e4
Ipr = Ipr * 1e7 / 1.e4
Ipr_unit = r'erg s$^{-1}$ cm$^{-2}$ sr$^{-1}$ $\AA^{-1}$'

Fpr = Fpr *1e7/1.e4
Fpr_unit = r'erg s$^{-1}$ cm$^{-2}$ sr$^{-1}$ $\AA^{-1}$'

## Table 14.17 - limb darkening coefficients
lambdaCLV = 1.e4* np.asarray([0.20,0.22,0.245,0.265,0.28,0.30,0.32,0.35,
                0.37,0.38,0.40,0.45,0.50,0.55,0.60,0.80,1.0,
                1.5,2.0,3.0,5.0,10.0])
lambdaCLV_unit = r'$\AA$'

uCLV = np.asarray([0.12,-1.3,-0.1,-0.1,0.38,0.74,0.88,0.98,1.03,0.92,0.91,
                    0.99,0.97,0.93,0.88,0.73,0.64,0.57,0.48,0.35,0.22,0.15])
vCLV = np.asarray([0.33,1.6,0.85,0.90,0.57, 0.20, 0.03,-0.1,-0.16,-0.05,
                    -0.05,-0.17,-0.22,-0.23,-0.23,-0.22,-0.20,-0.21,-0.18,
                    -0.12,-0.07,-0.07])

## get max and min of wavelengths in Angstrom
minlambda = np.max(np.asarray([lambdaIpr.min(),lambdaCLV.min()]))
maxlambda = np.min(np.asarray([lambdaIpr.max(),lambdaCLV.max()]))

########################################

def interpData(wvAng):
    """
    Inputs:
        wvAng - wavelength in Angstrom
    Returns:
        i0 - disk center intensity at wvAng between spectrum lines
        uc - limb darkening u coefficient at wvAng
        vc - limb darkening v coefficient at wvAng
        i0_unit - units of i0
    """
    if (wvAng < minlambda).any() or (wvAng > maxlambda).any():
        raise ValueError('Input wavelength out of data range')
    uc = np.interp(wvAng, lambdaCLV, uCLV)
    vc = np.interp(wvAng, lambdaCLV, vCLV)
    i0 = np.interp(wvAng, lambdaIpr, Ipr)
    i0_unit = Ipr_unit
    return i0,uc,vc,i0_unit

def applyCLV(i0,uc,vc,i0_unit,muAngle):
    i0 = (1.0 - uc - vc + uc * muAngle + vc * muAngle**2)* i0
    return i0,i0_unit

def mean_sun_lambda(wvAng): 
    """ Returns mean solar intensity per wavelength units at a specific wavelength 
    lambda:     wavelength in angstrom
    """
    f0 = np.interp(wvAng,lambdaIpr,Fpr)
    return f0,Fpr_unit 

def i_lambda(wvAng,muAngle):
    """
    Returns solar intensity in per wavelength units at a specific
    wavelength and heliocentric angle
    lambda:     wavelength in angstrom
    muAngle:    cosine of the heliocentric angle
    """
    if (muAngle < 0) or (muAngle > 1):
        raise ValueError('muAngle must be >0 and <1')
    i0,uc,vc,i0_unit = interpData(wvAng)
    i0,i0_unit = applyCLV(i0,uc,vc,i0_unit,muAngle)
    return i0, i0_unit

def i_lambda_2_i_nu(i0,i0_unit,wvAng):
    ## convert to frequency units
    c = 2.99792458e+8 *1e10 ## Angstrom s^-1
    i0 = i0 * (wvAng**2)/c
    i0_unit = r'erg s$^{-1}$ cm$^{-2}$ sr$^{-1}$ Hz$^{-1}$'
    return i0,i0_unit

def i_nu(wvAng, muAngle):
    """
    Return the solar intensity in units per frequency units at a specific
    wavelength and heliocentric angle
    wavelength: wavelength in angstrom
    muAngle:    cosine of the heliocentric angle
    """
    i0, i0_unit = i_lambda(wvAng,muAngle)  ## i0_unit = 'erg s^-1 cm^-2 sr^-1 AA^-1)'
    i0, i0_unit = i_lambda_2_i_nu(i0,i0_unit,wvAng)
    return i0, i0_unit

def bright_temp(wvAng, muAngle):
    """ Returns the brightness temperature implied by intensity at wavelength"""
    i0, i0_unit = i_lambda(wvAng,muAngle)  ## 'erg s^-1 cm^-2 sr^-1 AA^-1'
    c     = 2.99792458e+10   ## 'cm s^-1'
    h     = 6.62607015e-27  ##  'erg s'
    k_B   = 1.380649e-16     ##  'erg / K'
    wv_cm = wvAng/1.e10*1.e2
    mf = (2.*h*c**2)/(wv_cm**5)   ## erg s cm^2 s^-2 / cm^5 ==> erg /s /cm^-3
    i0 = i0 * 1.e8  ##  'erg s^-1 cm^-2 sr^-1 cm^-1'
    Tb = (h*c/np.log(mf/i0 + 1.))/(wv_cm*k_B)
    Tb_unit = r'K'
    return Tb, Tb_unit

def J00_sym(wvAng, height):
    """
    Calculates the J00 component of the solar radiation
    field tensor assuming cylindrically symmetric limb darkened
    radiation.

    wavelength -  Angstrom
    height - kilometers

    Output units are Unit("erg / (cm2 Hz s sr)")

    References:  Degl'Innocenti & Landolfi (2004) Section 12.3
    """
    R_sun = 6.957e5   ## km
    height = np.clip(height,0.0001,1.e100)
    sg = R_sun / (R_sun + height)
    cg = np.sqrt(1.-sg**2.)
    a0 = 1. - cg
    a1 = cg - 0.5 - 0.5*cg**2/sg*np.log((1.+sg)/cg)
    a2 = (cg+2.)*(cg-1.) / (3.*(cg+1.))

    i0_lambda,uc,vc,i0_lambda_unit = interpData(wvAng)
    i0_nu,i0_nu_unit = i_lambda_2_i_nu(i0_lambda,i0_lambda_unit,wvAng)
    J00 = 0.5 * i0_nu * (a0 + a1*uc + a2*vc)
    J00_unit = i0_nu_unit

    return J00,J00_unit

def Knu_sym(wvAng, height):
    """
    Calculates the J02 component of the solar radiation
    field tensor assuming cylindrically symmetric limb darkened
    radiation.

    wavelenght -  Angstrom
    height - kilometers

    Output units are Unit("erg / (cm2 Hz s sr)")

    References:  Degl'Innocenti & Landolfi (2004) Section 12.3
    """
    R_sun = 6.957e5   ## km
    height = np.clip(height,0.0001,1.e100)
    sg = R_sun / (R_sun + height)
    cg = np.sqrt(1.-sg**2)
    b0 = (1.-cg**3)/3.
    b1 = (8.*cg**3 - 3.*cg**2 - 2.) / 24. - cg**4 / (8.*sg) * np.log((1.+sg)/cg)
    b2 = (cg-1.)*(3.*cg**3 + 6.*cg**2 + 4.*cg + 2.) / (15.*(cg+1.))
    i0_lambda,uc,vc,i0_lambda_unit = interpData(wvAng)
    i0_nu,i0_nu_unit = i_lambda_2_i_nu(i0_lambda,i0_lambda_unit,wvAng)
    Knu = 0.5 * i0_nu * (b0 + b1*uc + b2*vc)
    Knu_unit = i0_nu_unit

    return Knu, Knu_unit

def nbar(wvAng, height):
    """ photons per mode (units photons per sr)"""
    c     = 2.99792458e+10   ## 'cm s^-1'
    h     =  6.62607015e-27  ##  'erg s'
    k_B   = 1.380649e-16     ##  'erg / K'
    wv_cm = wvAng/1.e10*1.e2
    nu    = c/wv_cm
    J00,J_unit = J00_sym(wvAng,height)
    nbar = c**2 / (2.*h*nu**3.) * J00
    nbar_unit = 'photons per mode (per sr)'
    return nbar,nbar_unit

def get_omega(wvAng, height):
    """ 
    Calculate omega
    Inputs:
    wvAng -- wavelength in Angstroms
    height -- kilometers
    """
    J,J_unit = J00_sym(wvAng,height)
    K,K_unit = Knu_sym(wvAng,height)
    omega = (3.*K-J)/(2.*J)
    return omega

def J02_sym(wvAng, height):
    omega = get_omega(wvAng, height)
    J00,J_unit = J00_sym(wvAng,height)
    J02 = omega / np.sqrt(2.)*J00
    return J02,J_unit

if __name__ == "__main__":

    import matplotlib.pyplot as plt

    fig,ax = plt.subplots(1,3,figsize = (8,3))
    ax = ax.flatten()
    ax[0].plot(lambdaIpr,Ipr)
    ax[0].set_xlabel('Wavelength [' + lambdaIpr_unit + ']')
    ax[0].set_ylabel(Ipr_unit)
    ax[0].set_yscale('log')

    c =  2.99792458e+8
    nu = (lambdaIpr*1.e-8)/c
    i_nu,i_nu_unit = i_nu(lambdaIpr,1.)
    ax[1].plot(nu,i_nu)
    ax[1].set_xscale('log')
    ax[1].set_yscale('log')
    ax[1].set_xlabel('Frequency [Hz]')
    ax[1].set_ylabel('[' + i_nu_unit + ']')

    Tb,Tb_unit = bright_temp(lambdaIpr,1.)
    ax[2].plot(lambdaIpr,Tb)
    ax[2].set_xlabel('Wavelength [' + lambdaIpr_unit + ']')
    ax[2].set_ylabel(r'Brightness Temp [K]')

    fig.suptitle('Solar Disk Center Spectral Radiance [Allens Astrophysical Quantities]')

    fig.tight_layout(pad = 0.2)
    fig.subplots_adjust(top = 0.88)

    plt.show()

    ## Recreate Figure 4 from Asensio Ramos
    wvA = np.linspace(2000,12000,100)
    fig, ax= plt.subplots(1,2,figsize = (7,3))
    ax = ax.flatten()

    arc2km = 712.
    ax[0].plot(wvA,nbar(wvA, 0.*arc2km)[0],linestyle = '--',label = 'h = 0 arcsec')
    ax[0].plot(wvA,nbar(wvA, 3.*arc2km)[0],linestyle = '-',label = 'h = 3 arcsec')
    ax[0].plot(wvA,nbar(wvA,10.*arc2km)[0],linestyle = '-.',label = 'h = 10 arcsec')
    ax[0].plot(wvA,nbar(wvA,40.*arc2km)[0],linestyle = ':',label = 'h = 40 arcsec')
    ax[0].set_yscale('log')
    ax[0].set_xlabel(r'Wavelength [$\AA$]')
    ax[0].set_ylabel(r'$\bar{n}$ [photons/sr]')
    ax[0].set_ylim(1.e-7,1.e-1)
    ax[0].legend()

    ax[1].plot(wvA,get_omega(wvA, 0.*arc2km),linestyle = '--',label = 'h = 0 arcsec')
    ax[1].plot(wvA,get_omega(wvA, 3.*arc2km),linestyle = '-',label = 'h = 3 arcsec')
    ax[1].plot(wvA,get_omega(wvA,10.*arc2km),linestyle = '-.',label = 'h = 10 arcsec')
    ax[1].plot(wvA,get_omega(wvA,40.*arc2km),linestyle = ':',label = 'h = 40 arcsec')
    ax[1].set_xlabel(r'Wavelength [$\AA$]')
    ax[1].set_ylabel(r'$\omega$')
    ax[1].legend()

    fig.tight_layout()
    plt.show()
