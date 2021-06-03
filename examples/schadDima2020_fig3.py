

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import readsav
import matplotlib as mpl
plt.ion()
from cycler import cycler
mpl.rcParams['axes.prop_cycle'] = cycler(color='rkbgym')

import pycelp

##########################
## LOAD UP CHIANTI VERSION 9 DATA

savname = './genChianti/chianti_density_contfnc_data.sav'
print('savname: ',savname)
idls = readsav(savname)
dens = idls['dens']
levels = idls['levels']
rht = idls['rht']
iontemps = idls['iontemps']
ch_int = np.copy(idls['ch_int']) ## units are  INT_UNITS   STRING   'photons cm-2 sr-1 s-1'
ch_int[:,:,0,:] = 1.  ## bad point at low density
sr2arcsec = (180./np.pi)**2.*3600.**2.
ch_int = ch_int / sr2arcsec


##########################
## DO pycelp CALCULATIONS

py_int = np.zeros_like(ch_int)

fe14 = pycelp.Ion('fe_14',nlevels = 27)
fe13 = pycelp.Ion('fe_13',nlevels = 27)
fe11 = pycelp.Ion('fe_11',nlevels = 27)
si10 = pycelp.Ion('si_10',nlevels = 27)
si9 = pycelp.Ion('si_9',nlevels = 27)

models = fe14,fe11,fe13,fe13,si10,si9
wvls = 5303,7892,10746,10798,14301,39343
rphot = rht-1.
thetab = np.rad2deg(np.arccos(1./np.sqrt(3.)))  ## Van Vleck

lev27 = 0

for n in range(6):
    for d,dens0 in enumerate(dens):
        print(n,d)
        models[n].calc(dens0,iontemps[n],rphot,thetab,include_limbdark = False,include_protons = True)
        py_int[n,lev27,d,0]  = models[n].calc_Iemiss(wvls[n]) /dens0 / 0.85 / dens0
        models[n].calc(dens0,iontemps[n],rphot,thetab,include_limbdark = False,include_protons = False)
        py_int[n,lev27,d,1]  = models[n].calc_Iemiss(wvls[n]) / dens0 / 0.85 / dens0


fe14 = pycelp.Ion('fe_14',nlevels = 100)
fe13 = pycelp.Ion('fe_13',nlevels = 100)
fe11 = pycelp.Ion('fe_11',nlevels = 100)
si10 = pycelp.Ion('si_10',nlevels = 100)
si9 = pycelp.Ion('si_9')

models = fe14,fe11,fe13,fe13,si10,si9
lev100 = 1
for n in range(6):
    for d,dens0 in enumerate(dens):
        print(n,d)
        models[n].calc(dens0,iontemps[n],rphot,thetab,include_limbdark = False,include_protons = True)
        py_int[n,lev100,d,0]  = models[n].calc_Iemiss(wvls[n]) /dens0 / 0.85 / dens0
        models[n].calc(dens0,iontemps[n],rphot,thetab,include_limbdark = False,include_protons = False)
        py_int[n,lev100,d,1]  = models[n].calc_Iemiss(wvls[n]) / dens0 / 0.85 / dens0


titles = r'Fe XIV $\lambda5303$ (T$_{e} = 10^{6.3}$ K)', \
        r'Fe XI $\lambda7892$ (T$_{e} = 10^{6.1}$ K)', \
        r'Fe XIII $\lambda10746$ (T$_{e} = 10^{6.25}$ K)',\
        r'Fe XIII $\lambda10798$ (T$_{e} = 10^{6.25}$ K)',\
        r'Si X $\lambda14301$ (T$_{e} = 10^{6.15}$ K)',\
        r'Si IX $\lambda39343$ (T$_{e} = 10^{6.05}$ K)',


fig,ax = plt.subplots(3,2,figsize = (7,8))
ax = ax.flatten()

lw0 = 0.8
for n in range(6):
    ax[n].plot(dens,ch_int[n,2,:,0]/ch_int[n,2,:,0],color = 'black',lw = lw0,label = str(levels[n,2]))
    if n != 5: ax[n].plot(dens,ch_int[n,1,:,0]/ch_int[n,2,:,0],color = 'green',lw = lw0,label = str(levels[n,1]))
    ax[n].plot(dens,ch_int[n,0,:,0]/ch_int[n,2,:,0],color = 'blue',lw = lw0,label = str(levels[n,0]))
    ax[n].plot(dens,ch_int[n,2,:,1]/ch_int[n,2,:,0],color = 'black',linestyle = 'dashed',lw = lw0)
    if n != 5: ax[n].plot(dens,ch_int[n,1,:,1]/ch_int[n,2,:,0],color = 'green',linestyle = 'dashed',lw = lw0)
    ax[n].plot(dens,ch_int[n,0,:,1]/ch_int[n,2,:,0],color = 'blue',linestyle = 'dashed',lw = lw0)

    ## pycelp
    ax[n].plot(dens,py_int[n,0,:,0]/ch_int[n,2,:,0],'s',color = 'blue',fillstyle = 'none',markersize = 1)
    ax[n].plot(dens,py_int[n,0,:,1]/ch_int[n,2,:,0],'x',color = 'blue',fillstyle = 'none',markersize = 1)
    if n != 5: ax[n].plot(dens,py_int[n,1,:,0]/ch_int[n,2,:,0],'s',color = 'green',fillstyle = 'none',markersize = 1)
    if n != 5: ax[n].plot(dens,py_int[n,1,:,1]/ch_int[n,2,:,0],'x',color = 'green',fillstyle = 'none',markersize = 1)
    if n == 5: ax[n].plot(dens,py_int[n,1,:,0]/ch_int[n,2,:,0],'s',color = 'black',fillstyle = 'none',markersize = 1)
    if n == 5: ax[n].plot(dens,py_int[n,1,:,1]/ch_int[n,2,:,0],'x',color = 'black',fillstyle = 'none',markersize = 1)

    ## LABELS
    ax[n].set_xlabel(r'Electron Density [cm$^{-3}$]')
    ax[n].set_ylabel(r'I / (I$_{Chianti}$ [' + str(levels[n,2]) + ' levels])')
    ax[n].set_xscale('log')
    ax[n].set_ylim(0.7,1.05)
    ax[n].set_title(titles[n])
    ax[n].text(0.8e5,0.725,
       r"$\theta_{B} = 54.74^{\circ}$" "\n"
       r"Height: 0.5 R$\odot$" "\n"
       "Solid: Chianti" "\n"
       "Symbols: PYCELP" "\n"
       "Dashed: Chianti no protons",   fontsize = 8,
       bbox=dict(facecolor='white', alpha=0.7))
    ax[n].legend(loc = 'lower right',title = '# levels')

for z in range(5): fig.tight_layout()
