
'''
This module contains the psiModel class for reading PSI coronal model data 
'''

import numpy as np
from scipy.interpolate import RegularGridInterpolator as rgi
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as R
from pyhdf.SD import SD, SDC
import os

def read_psi_hdf(fn):
    file = SD(fn, SDC.READ)
    dat  = file.select('Data-Set-2').get()
    s0   = file.select('fakeDim0').get()
    s1   = file.select('fakeDim1').get()
    s2   = file.select('fakeDim2').get()
    return dat,s0,s1,s2

def read_interp_psi(fn,points_ref):
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

class Model:
    
    def __init__(self, model_data_directory): 
        
        self.dir = os.path.join(os.path.dirname(model_data_directory),'')
        
        ## check if the right files exist in directory 
        
        ## Interpolate PSI model onto an unstaggered grid...
        ## Use lons from br and lats,rs from bp as the reference coordinates
        ## this is because they have different shapes normally due to end points 
        br,lons,latsB,rsB = read_psi_hdf(self.dir + 'br002.hdf')
        bp,lonsB,lats,rs = read_psi_hdf(self.dir + 'bp002.hdf')
        temp = read_interp_psi(self.dir + 't002.hdf',(lons,lats,rs))
        br = read_interp_psi(self.dir + 'br002.hdf',(lons,lats,rs))
        bt = read_interp_psi(self.dir + 'bt002.hdf',(lons,lats,rs))
        bp = read_interp_psi(self.dir + 'bp002.hdf',(lons,lats,rs))
        ne = read_interp_psi(self.dir + 'rho002.hdf',(lons,lats,rs))
        
        ## convert MAS normalized units into physical units
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

        self.lons = lons
        self.lats = lats 
        self.rs = rs 
        self.temp = temp 
        self.br   = br
        self.bt   = bt
        self.bp   = bp
        self.ne   = ne
        self.bx   = bx
        self.by   = by
        self.bz   = bz
        self.rlen = rlen 
        self.blen = blen 
        self.b_localinc = localinc
                
"""

## INPUTS
obsLon     = 185.     ## observer's longitude
b0         = -5.57    ## observer's heliographic latitude -- b0 angle
Obs_Sun_AU = 1.     ## replace with ephermeris data
fov_rsun   = 6.     ## +/- 3 rsun
arcsamp    = 10.    ## sampling in arcsecond
rsunarc    = 960.  ## radius of sun in arcseconds .. later replace with sun ephemeris
pycle_pickle_file = './pycle_calcs/psi_pycle_outputs_fe11.pickle'
wvAng_synth = 7892.

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

##################
## recall the simulation geometry and plot
## compare to GONG maps of the CR 2189

rObs     = Obs_Sun_AU * (1.495978707e11/6.96340e8)
thetaObs = np.pi/2. - np.deg2rad(b0)
phiObs   = np.deg2rad(obsLon)
xObs,yObs,zObs = rObs*np.sin(thetaObs)*np.cos(phiObs),rObs*np.sin(thetaObs)*np.sin(phiObs),rObs*np.cos(thetaObs)

lonsd = np.rad2deg(lons)
latsd = np.flip(np.rad2deg(lats)-90.)  ## flip as in observers frame angles increase towards north
extrad = (lons[0],lons[-1],lats[-1],lats[0])
extdeg = (lonsd[0],lonsd[-1],latsd[-1],latsd[0])
fig,ax = plt.subplots(nrows = 3,ncols = 1,figsize = (6,8))
ax = ax.flatten()
ax[0].imshow(br[:,:,0].T,extent = extrad)
ax[1].imshow(br[:,:,0].T,extent = extdeg)
ax[2].imshow(z3d[:,:,0].T,extent = extrad)
ax[0].plot(np.zeros(1)+phiObs,np.zeros(1)+thetaObs,'x',markersize = 5,color= 'black')
ax[1].plot(np.zeros(1)+phiObs,np.zeros(1)+thetaObs,'x',markersize = 5,color= 'black')

## setup the synthesized field of view

yarc = np.linspace(-fov_rsun/2.*rsunarc,fov_rsun/2.*rsunarc,np.int(np.ceil(fov_rsun*rsunarc/arcsamp)))
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
r = R.from_euler('yz',[-(thetaObs-np.pi/2.),-(phiObs)])
b = np.stack((xxObs.flatten(),yyObs.flatten(),zzObs.flatten()))
xyz_model = np.matmul(r.as_matrix(), b)
plt.imshow(xyz_model[1,:].reshape(xxObs.shape).T,origin = 'lower')

## NOW WITH THE XYZ_MODEL POINTS AND THE LOCATION OF THE OBSERVER,
## come up with the parametric equations for the los of sight
## and then get the spherical coordinates, interpolate for rho and temps, etc.
## synthesize and integrate for integrated I,Q,U, (V?)

losvec = np.stack((xObs-xyz_model[0,:],yObs-xyz_model[1,:],zObs - xyz_model[2,:]))
losveclen = np.linalg.norm(losvec,axis=0,ord=2,keepdims = True)
losvec = losvec / losveclen
startpt = np.stack((xyz_model[0,:],xyz_model[1,:],xyz_model[2,:]))


 
"""
