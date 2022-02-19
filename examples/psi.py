
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

        ## Grid sample locations 
        self.lons = lons
        self.lats = lats 
        self.rs   = rs
        
        ## cartesian coordinats
        self.x3d = x3d 
        self.y3d = y3d 
        self.z3d = z3d 
        
        self.temp = temp 
        self.ne   = ne
        
        ## spherical magnetic field components
        self.br   = br
        self.bt   = bt
        self.bp   = bp
        
        ## cartesian magnetic field components
        self.bx   = bx
        self.by   = by
        self.bz   = bz

        ## Field magnitude and local frame inclination angle 
        self.bmag = blen 
        self.thetaBlocal = localinc
        
    def __repr__(self):
         return f"""psi Model class
    ---------------------
    Data Directory Names: {self.dir}
    Number of longitude samples: {len(self.lons)}
    Number of latitude samples: {len(self.lats)}
    Number of radial samples: {len(self.rs)}
    Data shape: {self.temp.shape}
    
    Variables: 
    lons -- Longitudes
    lats -- Latitudes
    rs   -- Radial samples 
    br,bt,bp  -- Spherical components of magnetic field 
    bx,by,bz  -- Cartesian components of magnetic field 
    bmag      -- total magnetic field intensity
    thetaBlocal -- location inclination of magnetic field in solar frame
    """
