

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import readsav
import matplotlib as mpl
plt.ion()
from cycler import cycler
mpl.rcParams['axes.prop_cycle'] = cycler(color='rkbgym')

import pycle

##########################
## DO PYCLE CALCULATIONS

fe14 = pycle.Ion('fe_14',nlevels = 100)
fe13 = pycle.Ion('fe_13',nlevels = 100)
fe11 = pycle.Ion('fe_11',nlevels = 100)
si10 = pycle.Ion('si_10',nlevels = 100)
si9 = pycle.Ion('si_9')
models = fe14,fe11,fe13,fe13,si10,si9

wvls = 5303,7892,10746,10798,14301,39343
rphot = 0.5
iontemps =  10.**np.array([6.3,6.1,6.25,6.25,6.15,6.05])
dens = 1.e5

nt = 100
thetabs = np.linspace(0,90,nt)
py_align = np.zeros((6,nt))
py_rho00 = np.zeros((6,nt))

for n in range(6):
    print(n)
    for t,thetab in enumerate(thetabs):
        models[n].calc(dens,iontemps[n],rphot,thetab,include_limbdark = True,include_protons = True)
        py_align[n,t] = models[n].get_upper_level_alignment(wvls[n])
        py_rho00[n,t] = models[n].get_upper_level_rho00(wvls[n])

py_rho00_ref = np.zeros(6)
vv =  np.rad2deg(np.arccos(1./np.sqrt(3.)))  ## Van Vleck
for n in range(6):
    models[n].calc(dens,iontemps[n],rphot,vv,include_limbdark = True,include_protons = True)
    py_rho00_ref[n] = models[n].get_upper_level_rho00(wvls[n])

fig,ax = plt.subplots(1,2,figsize = (8,4))
ax = ax.flatten()

lw0 = 0.8
for n in range(6):
    ax[0].plot(thetabs,py_rho00[n,:]/py_rho00_ref[n],lw = lw0,linestyle = 'solid')
    ax[1].plot(thetabs,py_align[n,:],lw = lw0,linestyle = 'solid')

    ## LABELS
    #ax[n].set_xlabel(r'Electron Density [cm$^{-3}$]')
    #ax[n].set_ylabel(r'Upper Level Alignment ($\sigma^2_{0}$)')
    #ax[n].set_xscale('log')
    #ax[n].set_title(titles[n])
