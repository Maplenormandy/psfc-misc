# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 13:52:05 2018

Intended to be run in python 3

@author: normandy
"""

import matplotlib as mpl
import matplotlib.pyplot as plt

import numpy as np
import scipy.io, scipy.signal

font = {'family' : 'serif',
        'serif': ['Computer Modern'],
        'size'   : 9}

mpl.rc('font', **font)

# %% Figure 5, the cartoon

fig9 = plt.figure(5, figsize=(3.395, 3.395))


gs9 = mpl.gridspec.GridSpec(2, 1, height_ratios=[3,4])
gs9_inner = mpl.gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs9[0], wspace=0.2)

ax90 = plt.subplot(gs9[1])
ax910 = plt.subplot(gs9_inner[0])
ax911 = plt.subplot(gs9_inner[1])

kplot = np.linspace(0.02, 1.4, 128)

def lorentzian(ky, k0, g):
    return 1.0/(np.pi * g * (1 + ((ky-k0)/g)**2))

def model_wqi(ky):
    return 2.0*(1-(ky/1.6)**0.5)

#gmax = np.max(gamma(kplot))
#mixing = gamma(kplot)/kplot

#plt.plot(kplot, wqi(kplot))

loc_spectrum = lorentzian(kplot, 0.4, 0.25) * model_wqi(kplot)
soc_spectrum = lorentzian(kplot, 0.4, 0.16) * model_wqi(kplot)


loc_spectrum = loc_spectrum / np.sum(loc_spectrum) * 2 / 1.8 * 100
soc_spectrum = soc_spectrum / np.sum(soc_spectrum) * 2 / 1.8 * 100

ax90.plot(kplot, loc_spectrum, c='r')
ax90.plot(kplot, soc_spectrum, c='b')
ax90.set_ylim(bottom=0)

xbar = np.arange(6)
barlabels=['ITGa', 'ITGb', 'ITGc', 'TEMa', 'TEMb', 'ETG']
soc_values = [0.3, 1.8, 0.2, 0.0, 0.4, 0.18]
loc_values = [0.6, 1.4, 0.4, 0.15, 0.4, 0.15]

#plt.xticks(x, labels, rotation='vertical')

ax90.set_xlabel(r'$k_y \rho_s$')
ax90.set_ylabel(r'$W_{Qi,k} \left\langle \bar{\phi}_{k_1}^2 \right\rangle$ (arb. units)')
ax90.set_title('Cartoon Ion Heat Flux Spectrum')
ax90.xaxis.set_major_locator(mpl.ticker.MultipleLocator(0.5))

ax910.bar(xbar, loc_values, color='r', align='center', tick_label=barlabels)
ax910.set_xticklabels(barlabels, rotation=45)
ax911.bar(xbar, soc_values, color='b', align='center', tick_label=barlabels)
ax911.set_xticklabels(barlabels, rotation=45)

ax910.set_title('LOC')
ax911.set_title('SOC')
ax910.set_ylim([0,2])
ax911.set_ylim([0,2])
ax910.set_ylabel('Family-Integrated\nIntensity (a.u.)')

plt.tight_layout()
plt.tight_layout()

plt.savefig('fig_cartoon.eps', format='eps', dpi=1200, facecolor='white')

