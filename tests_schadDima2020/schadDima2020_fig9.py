
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
iontemps =  10.**np.array([6.3,6.1,6.25,6.25,6.15,6.05])

nd = 200
dens = 10.**np.linspace(5,11,nd)
py_align = np.zeros((6,nd,2))

for n in range(6):
    print(n)
    for d,dens0 in enumerate(dens):
        models[n].calc(dens0,iontemps[n],0.1,0,include_limbdark = True,include_protons = True)
        py_align[n,d,0] = models[n].get_upper_level_alignment(wvls[n])

        models[n].calc(dens0,iontemps[n],0.5,0,include_limbdark = True,include_protons = True)
        py_align[n,d,1] = models[n].get_upper_level_alignment(wvls[n])


fig,ax = plt.subplots(1,2,figsize = (8,4))
ax = ax.flatten()

for n in range(6):
    ax[0].plot(dens,py_align[n,:,0],lw = 0.8)
    ax[0].plot(dens,py_align[n,:,1],linestyle = 'dashed',lw = 0.8)
    ax[1].plot(dens,np.abs(py_align[n,:,0]),lw = 0.5)
    ax[1].plot(dens,np.abs(py_align[n,:,1]),linestyle = 'dashed',lw = 0.5)

ax[0].set_xscale('log')
ax[1].set_xscale('log')
ax[1].set_yscale('log')

for z in range(5): fig.tight_layout()
