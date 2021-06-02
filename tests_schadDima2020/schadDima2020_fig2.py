
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import readsav
import matplotlib as mpl
plt.ion()
from cycler import cycler
mpl.rcParams['axes.prop_cycle'] = cycler(color='rkbgym')

import pycle

##########################
## LOAD UP CHIANTI VERSION 9 DATA

savname = './genChianti/chianti_temp_contfnc_data.sav'  # ,logt,ch_int,ions,wvl,dens,rht
print('savname: ',savname)
idls = readsav(savname)
logt = idls['logt']
ch_int = np.copy(idls['ch_int']) ## units are  INT_UNITS   STRING   'photons cm-2 sr-1 s-1'
ions = idls['ions']
wvl = idls['wvl']
dens = idls['dens']
rht = idls['rht']
## remove bad low point
ch_int[:,0] = 1.e-35

##########################
## DO PYCLE CALCULATIONS

fe14 = pycle.Ion('fe_14',nlevels = 300) # -- issue with scups read for now !
fe13 = pycle.Ion('fe_13',nlevels = 300)
fe11 = pycle.Ion('fe_11',nlevels = 300)
si10 = pycle.Ion('si_10')
si9 = pycle.Ion('si_9')

temps = 10.**logt
pycle_int = np.zeros((6,len(temps)))
edens = idls['dens']
rphot = rht-1.
thetab = np.rad2deg(np.arccos(1./np.sqrt(3.)))  ## Van Vleck

print(' ')
print(' Starting calculations -- with many levels, so it may be slower')
print(' Paper includes all...this script currently truncates at 300 levels for speed')
print(' The comparison should still be very good')
print(' ')
print(fe14,fe11,fe13,si10,si9)
print(' ')
print(' Electron density, height, and thetaB: ')
print(edens,rphot,thetab)
print(' ')

for n,t in enumerate(temps):
    print(n,' of ',len(temps), ' temperatures')
    fe14.calc(edens,t,rphot,thetab,include_limbdark = False,include_protons = True)
    fe11.calc(edens,t,rphot,thetab,include_limbdark = False,include_protons = True)
    fe13.calc(edens,t,rphot,thetab,include_limbdark = False,include_protons = True)
    si10.calc(edens,t,rphot,thetab,include_limbdark = False,include_protons = True)
    si9.calc(edens,t,rphot,thetab,include_limbdark = False,include_protons = True)
    pycle_int[0,n] = fe14.calc_Iemiss(5303)
    pycle_int[1,n] = fe11.calc_Iemiss(7892)
    pycle_int[2,n] = fe13.calc_Iemiss(10746)
    pycle_int[3,n] = fe13.calc_Iemiss(10798)
    pycle_int[4,n] = si10.calc_Iemiss(14301)
    pycle_int[5,n] = si9.calc_Iemiss(39343)

########
## PLOT DATA FOR COMPARISON

plt.close('all')
fig,ax = plt.subplots(nrows=1,ncols=1,figsize = (7,7./1.6/1.5))

labs = [r'[739] Fe XIV $\lambda5303$',
        r'[996] Fe XI $\lambda7892$',
        r'[749] Fe XIII $\lambda10746$',
        r'[749] Fe XIII $\lambda10798$',
        r'[204] Si X $\lambda14301$',
        r'[ 46] Si IX $\lambda39343$']
sr2arcsec = (180./np.pi)**2.*3600.**2.

for zz in range(0,len(wvl)):
    ax.plot((10.**logt)/1.e6,ch_int[zz,:]/sr2arcsec,label = labs[zz])
for n in range(6):
    ax.plot(temps/1e6,pycle_int[n,:]/edens/(0.85*edens),'s',markersize = 3,fillstyle = 'none')

ax.set_xlim(0.5,3.5)
ax.set_yscale('log')
ax.set_ylim(1.e-27,1.e-24)
ax.set_ylabel(r'photons cm$^{+3}$ s$^{-1}$ arcsec$^{-2}$')
ax.set_xlabel('Temperature [MK]')
ax.text(0.02,0.9,r'N$_{e}$ = $10^{8.5}$ [cm$^{-3}$]; Photospheric Abundances',transform = ax.transAxes)
ax.text(0.02,0.82,r'Height = 0.1 R$_\odot$ (Limb Darkening Disabled)',transform = ax.transAxes)
ax.text(0.02,0.74,r'$\theta_{B} = 54.74^{\circ}$ (pyCLE)',transform = ax.transAxes)
ax.set_title('Contribution Functions: Chianti (solid) and pyCLE (symbols)')
ax.legend(fontsize = 8)
fig.tight_layout()
