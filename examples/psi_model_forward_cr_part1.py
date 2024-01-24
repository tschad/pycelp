

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator as rgi
from scipy.interpolate import interp1d
import pickle
import time
import sys
import os
import sunpy.coordinates.sun as sun
from pyhdf.SD import SD, SDC

plt.ion()
plt.close('all')

def read_psi_hdf(fn):
    file = SD(fn, SDC.READ)
    dat  = file.select('Data-Set-2').get()
    s0   = file.select('fakeDim0').get()
    s1   = file.select('fakeDim1').get()
    s2   = file.select('fakeDim2').get()
    return dat,s0,s1,s2

def read_interp_psi(fn,points_ref):
    print(' read_interp_psi: ',fn)
    ## read data and get interpolating function
    dat,lons,lats,rs = read_psi_hdf(fn)
    fInt = rgi((lons,lats,rs),dat,method = 'linear',fill_value = 0.,bounds_error = False)
    ## get meshgrid of points for the new reference coordinates
    lons_ref,lats_ref,rs_ref = points_ref
    npoints = np.meshgrid(lons_ref,lats_ref,rs_ref,indexing = 'ij')
    flat = np.array([m.flatten() for m in npoints])
    ## interpolate and reshape
    datInt = fInt(flat.T).reshape(*npoints[0].shape)
    return datInt

def euler_ry(alpha):
    '''Euler rotation matrix about y axis '''
    ry = np.array([[ np.cos(alpha), 0., np.sin(alpha)],
                   [            0., 1., 0.],
                   [-np.sin(alpha), 0., np.cos(alpha)]])
    return ry

def euler_rz(alpha):
    '''Euler rotation matrix about z axis '''
    rz = np.array([[ np.cos(alpha), -np.sin(alpha), 0.],
                   [ np.sin(alpha),  np.cos(alpha), 0.],
                   [            0.,             0., 1.]])
    return rz

## Interpolate PSI model onto an unstaggered grid...
## Use lons from br and lats,rs from bp as the reference coordinates
br,lons,latsB,rsB = read_psi_hdf('./corona/br002.hdf')
bp,lonsB,lats,rs = read_psi_hdf('./corona/bp002.hdf')

temp = read_interp_psi('./corona/t002.hdf',(lons,lats,rs))
br = read_interp_psi('./corona/br002.hdf',(lons,lats,rs))
bt = read_interp_psi('./corona/bt002.hdf',(lons,lats,rs))
bp = read_interp_psi('./corona/bp002.hdf',(lons,lats,rs))
ne = read_interp_psi('./corona/rho002.hdf',(lons,lats,rs))

## have to convert MAS normalized units into physical units
temp = temp*(2.807067e7) ##  K
br = br*2.2068908   ## Gauss
bt = bt*2.2068908   ## Gauss
bp = bp*2.2068908   ## Gauss
ne = ne*1.e8        ## cm^-3

##spherical coordinates and cartesian coordinates
phi3d,theta3d,r3d = np.meshgrid(lons,lats,rs,indexing = 'ij')
x3d = r3d*np.sin(theta3d)*np.cos(phi3d)
y3d = r3d*np.sin(theta3d)*np.sin(phi3d)
z3d = r3d*np.cos(theta3d)

## cartesian vectors of the magnetic field
bx = br*np.sin(theta3d)*np.cos(phi3d) + bt*np.cos(theta3d)*np.cos(phi3d) - bp*np.sin(phi3d)
by = br*np.sin(theta3d)*np.sin(phi3d) + bt*np.cos(theta3d)*np.sin(phi3d) + bp*np.cos(phi3d)
bz = br*np.cos(theta3d)               - bt*np.sin(theta3d)

## get inclinations in local solar reference frame
## using dot product between local radial and the magnetic field
rlen = np.sqrt(x3d**2 + y3d**2 + z3d**2)
blen = np.sqrt(bx**2 + by**2 + bz**2)
localinc  = np.arccos(((bx*x3d + by*y3d + bz*z3d)/(rlen*blen)).clip(min = -1,max=1))

## this should be enough to get the alignments and rho_00 by using PYCLE
## someday may try to make this into a lookup table version
pht     = rlen.flatten() - 1
petemp  = temp.flatten()
pedens  = ne.flatten()
pbgaus  = blen.flatten()
pbthet  = np.rad2deg(localinc).flatten()
save_var = pht,petemp,pedens,pbgaus,pbthet
pickle_file = './psi_model_params4pycle.pickle'
pickle.dump(save_var,open(pickle_file,"wb"))

# pickle_file = './psi_model_params4pycle.pickle'
# pht,petemp,pedens,pbgaus,pbthet  = pickle.load(open(pickle_file,"rb"))
#  >>> GO RUN psi_pycle_mpi.py on DSSC cluster for different ions
# this will give me the alignments, etc....

def psi_model_forward(crn,pycle_pickle_file,wvAng_synth):

    ## get observing geometry and setup observer's field of view
    crt        = sun.carrington_rotation_time(crn)
    obsLon     = sun.L0(crt).deg
    b0         = sun.B0(crt).deg
    Obs_Sun_AU = sun.earth_distance(crt).value
    rsunarc    = sun.angular_radius(crt).value  ## radius of sun in arcseconds .. later replace with sun ephemeris
    fov_arc    = 6.*960.  ## +/- 3 rsun
    arcsamp    = 20. ## 10.   ## sampling in arcsecond

    ##################
    ## recall the simulation geometry and plot
    ## compare to GONG maps of the CR 2189

    rObs     = Obs_Sun_AU * (1.495978707e11/6.96340e8)
    thetaObs = np.pi/2. - np.deg2rad(b0)
    phiObs   = np.deg2rad(obsLon)
    xObs,yObs,zObs = rObs*np.sin(thetaObs)*np.cos(phiObs),rObs*np.sin(thetaObs)*np.sin(phiObs),rObs*np.cos(thetaObs)

    lonsd = np.rad2deg(lons)
    latsd = np.flip(np.rad2deg(lats)-90.,0)  ## flip as in observers frame angles increase towards north
    extrad = (lons[0],lons[-1],lats[-1],lats[0])
    extdeg = (lonsd[0],lonsd[-1],latsd[-1],latsd[0])
    #fig,ax = plt.subplots(nrows = 3,ncols = 1,figsize = (6,8))
    #ax = ax.flatten()
    #ax[0].imshow(br[:,:,0].T,extent = extrad)
    #ax[1].imshow(br[:,:,0].T,extent = extdeg)
    #ax[2].imshow(z3d[:,:,0].T,extent = extrad)
    #ax[0].plot(np.zeros(1)+phiObs,np.zeros(1)+thetaObs,'x',markersize = 5,color= 'black')
    #ax[1].plot(np.zeros(1)+phiObs,np.zeros(1)+thetaObs,'x',markersize = 5,color= 'black')

    ## setup the synthesized field of view

    yarc = np.linspace(-fov_arc/2.,fov_arc/2.,np.int(np.ceil(fov_arc/arcsamp)))
    zarc = yarc
    yya,zza  = np.meshgrid(yarc,zarc,indexing = 'ij')
    rra = np.sqrt(yya**2. + zza**2.)
    m_behind_sun = 1.*(rra>rsunarc)

    ################
    ## define new coordinate system with x towards
    ## observer, and z towards north pole
    ## get points in the plane

    xxObs = np.zeros_like(yya)
    yyObs = rObs * np.tan(np.deg2rad(yya/3600.))
    zzObs = rObs * np.tan(np.deg2rad(zza/3600.))

    ## rotate these points into the model geometry with Euler rotation
    rotm = np.matmul(euler_rz(-phiObs),euler_ry(-(thetaObs-np.pi/2.)))
    b = np.stack((xxObs.flatten(),yyObs.flatten(),zzObs.flatten()))
    xyz_model = np.matmul(rotm.T, b)
    #plt.imshow(xyz_model[1,:].reshape(xxObs.shape).T,origin = 'lower')

    ## NOW WITH THE XYZ_MODEL POINTS AND THE LOCATION OF THE OBSERVER,
    ## come up with the parametric equations for the los of sight
    ## and then get the spherical coordinates, interpolate for rho and temps, etc.
    ## synthesize and integrate for integrated I,Q,U, (V?)

    losvec = np.stack((xObs-xyz_model[0,:],yObs-xyz_model[1,:],zObs - xyz_model[2,:]))
    losveclen = np.linalg.norm(losvec,axis=0,ord=2,keepdims = True)
    losvec = losvec / losveclen
    startpt = np.stack((xyz_model[0,:],xyz_model[1,:],xyz_model[2,:]))

    #############################################
    ## get pycle data
    pht,petemp,pedens,pbgaus,pbthet,rho,atom,atomrad,abnd,awgt,atomid,eq_frac,eq_logtemp = pickle.load(open(pycle_pickle_file,"rb"))
    eq_int = interp1d(eq_logtemp,eq_frac,kind = 'cubic',bounds_error = False,fill_value = 0.)
    totn   = 10.**(abnd-12.)*(0.85*ne) * eq_int(np.log10(temp))    ## where totH = (0.85*ne)
    ln     = np.argmin(np.abs(atomrad['wv_air']-wvAng_synth))
    wv_air = atomrad['wv_air'][ln]
    upplev = np.int(atomrad['upper_lev_indx'][ln])
    qnj    = atom['qnj'][upplev-1]
    A_ein  = atomrad['A_einstein'][ln]
    hnu      = (6.626176e-27)*(2.99792458e10)/(wv_air/1.e8 )   ## CGS
    D_coeff  = atomrad['D_coeff'][ln]
    E_coeff  = atomrad['E_coeff'][0]
    geff     = atomrad['geff'][0]
    print(wv_air,upplev,qnj)
    rho00 = (rho[:,upplev-1,0]).reshape(rlen.shape)
    sig   = (rho[:,upplev-1,2]/rho[:,upplev-1,0]).reshape(rlen.shape)
    sig[np.isnan(sig)] = 0.

    #### Get interpolating functions for all necessary simulation data
    totnInt = rgi((lons,lats,rs),totn,method = 'linear',fill_value = 0.,bounds_error = False)
    rhoInt  = rgi((lons,lats,rs),rho00,method = 'linear',fill_value = 0.,bounds_error = False)
    sigInt  = rgi((lons,lats,rs),sig,method = 'linear',fill_value = 0.,bounds_error = False)
    bxInt   = rgi((lons,lats,rs),bx,method = 'linear',fill_value = 0.,bounds_error = False)
    byInt   = rgi((lons,lats,rs),by,method = 'linear',fill_value = 0.,bounds_error = False)
    bzInt   = rgi((lons,lats,rs),bz,method = 'linear',fill_value = 0.,bounds_error = False)

    ## SYNTHESIS
    totIQU = np.zeros((xxObs.shape[0],xxObs.shape[1],4))
    steps = np.linspace(-3,3,np.int(np.ceil(6./(10./960.))))  ## radii units
    for nstep,losstep in enumerate(steps):
        print(nstep,len(steps))
        t0 = time.time()
        ## get points along line of sight in cartesian and spherical coords
        xyz_next = startpt + losvec*losstep
        rm = np.sqrt(np.sum(xyz_next**2,axis = 0))
        tm = np.arccos(xyz_next[2,:]/rm)
        pm = np.arctan2(xyz_next[1,:],xyz_next[0,:])
        pm[pm<0] += 2*np.pi
        flat = np.array([pm,tm,rm])
        ## get ThetaB (inclination of B wrt to LOS)
        bxyzs = np.stack((bxInt(flat.T),byInt(flat.T),bzInt(flat.T)))
        blens = np.linalg.norm(bxyzs,axis =0)
        ## losvec len is 1
        thetaBlos = np.arccos((np.sum(bxyzs*losvec,axis=0)/blens).clip(min = -1,max=1))
        thetaBlos[np.isnan(thetaBlos)] = 0.
        ## angle between losvec and the vector from disk center
        rlens = np.linalg.norm(xyz_next,axis=0)
        thetaDClos = np.arccos((np.sum(xyz_next*losvec,axis=0)/rlens).clip(min = -1,max=1))
        ## get projection of B onto plane perpendicular to LOS
        ## and the projection of DC vector onto same plane
        Bperp  = bxyzs -  (blens*np.cos(thetaBlos))*losvec
        DCperp = xyz_next - (rlens*np.cos(thetaDClos))*losvec
        ## find angle between Bperp and DCperp, which is the azimuthal angle relative to disk center
        costhetaAzi = np.sum((Bperp*DCperp),axis=0)/(np.linalg.norm(Bperp,axis = 0)*np.linalg.norm(DCperp,axis = 0))
        thetaAzi = np.arccos(costhetaAzi.clip(min =-1,max=1))
        thetaAzi[np.isnan(thetaAzi)] = 0.
        ## total population and atomic alignment
        C_coeff = totnInt(flat.T)*np.sqrt(2.*qnj+1.)*rhoInt(flat.T)*A_ein*hnu/(4.*np.pi)
        sigma   = sigInt(flat.T)
        ## STOKES COEFFICIENTS
        epsI   = C_coeff*(1.0+(1./(2.*np.sqrt(2.)))*(3.*np.cos(thetaBlos)**2 - 1.)*D_coeff*sigma)
        epsQnr = C_coeff*(3./(2.*np.sqrt(2.)))*(np.sin(thetaBlos)**2)*D_coeff*sigma
        epsQ   = np.cos(2.*thetaAzi)*epsQnr
        epsU   = -np.sin(2.*thetaAzi)*epsQnr
        epsV   = C_coeff*np.cos(thetaBlos)*(1399612.2*blens)*(geff + E_coeff*sigma)
        ## is it behind the Sun?
        if (losstep < 0):
            epsI = epsI.reshape(xxObs.shape)*m_behind_sun
            epsQ = epsQ.reshape(xxObs.shape)*m_behind_sun
            epsU = epsU.reshape(xxObs.shape)*m_behind_sun
            epsV = epsV.reshape(xxObs.shape)*m_behind_sun
        else:
            epsI = epsI.reshape(xxObs.shape)
            epsQ = epsQ.reshape(xxObs.shape)
            epsU = epsU.reshape(xxObs.shape)
            epsV = epsV.reshape(xxObs.shape)
        ###
        totIQU[:,:,0] += epsI
        totIQU[:,:,1] += epsQ
        totIQU[:,:,2] += epsU
        totIQU[:,:,3] += epsV
        t1 = time.time()
        print(t1-t0)
        sys.stdout.flush()
    ##
    plt.clf()
    plt.imshow(np.log10(totIQU[:,:,0]))
    return totIQU

##########################
if __name__ == "__main__":

    ions = ['fe11','fe13','fe13']
    wvAng_synth = [7892.,10747.,10798.]
    crn_all = np.linspace(2189,2190,np.int(360./10.))
    nions = len(ions)
    for nln in range(nions):
        for nn,crn in enumerate(crn_all):
            pycle_pickle_file = './pycle_calcs/psi_pycle_outputs_'+ions[nln]+'.pickle'
            totIQU = psi_model_forward(crn,pycle_pickle_file,wvAng_synth[nln])
            pickle_file = './synth/'+ions[nln]+'_'+str(np.int(wvAng_synth[nln]))+ '_' + str(nn).zfill(3) + '_emiss.pickle'
            print(pickle_file)
            save_var = totIQU,crn
            pickle.dump(save_var,open(pickle_file,"wb"))

## display
## end synth loop
#plt.imshow(np.log10(totIQU[:,:,0]).T,origin = 'lower')   ## this looks like the right orientation
#plt.imshow(np.abs(totIQU[:,:,1]/totIQU[:,:,0]).T,origin = 'lower')
#plt.imshow(totIQU[:,:,1].T,origin = 'lower')
#plt.imshow(totIQU[:,:,2].T,origin = 'lower')
#plt.imshow(0.5*np.arctan2(totIQU[:,:,2],totIQU[:,:,1]).T,origin = 'lower')
