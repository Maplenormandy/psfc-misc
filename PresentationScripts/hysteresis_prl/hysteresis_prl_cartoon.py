# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate

import matplotlib as mpl


font = {'family' : 'serif',
        'serif': ['Computer Modern'],
        'size'   : 10}

mpl.rc('font', **font)

# %%

radii = np.loadtxt('./cgyro_outputs/input.norm.radii')
ky = np.loadtxt('./cgyro_outputs/input.norm.ky')

flux_avg = np.load('./cgyro_outputs/out.norm.flux_avg.npy')
flux_std = np.load('./cgyro_outputs/out.norm.flux_std.npy')

particle_weights = np.append([0.0, 0.0, 0.0, 1.0], np.zeros(8))
qe_weights = np.zeros(12)
qe_weights[7] = 1.0
qi_weights = np.zeros(12)
qi_weights[4] = 1.0
#qi_weights[5] = 1.0
#qi_weights[6] = 1.0
mom_weights = np.zeros(12)
mom_weights[8] = 1.0
#mom_weights[9] = 1.0
#mom_weights[10] = 1.0

pflux = np.einsum('i,ijk',particle_weights, flux_avg)
pflux_std = np.sqrt(np.einsum('i,ijk',particle_weights, flux_std**2))

qeflux = np.einsum('i,ijk',qe_weights, flux_avg)
qeflux_std = np.sqrt(np.einsum('i,ijk',qe_weights, flux_std**2))

qiflux = np.einsum('i,ijk',qi_weights, flux_avg)
qiflux_std = np.sqrt(np.einsum('i,ijk',qi_weights, flux_std**2))

momflux = np.einsum('i,ijk',mom_weights, flux_avg)
momflux_std = np.sqrt(np.einsum('i,ijk',mom_weights, flux_std**2))

rad_ind = np.searchsorted(radii, 0.57)


cgyro_freqs = np.load('./cgyro_outputs/all_freqs.npz')


freqs = cgyro_freqs['soc_mid']
wqi = scipy.interpolate.interp1d(ky, qiflux[rad_ind,:])
wqe = scipy.interpolate.interp1d(ky, qeflux[rad_ind,:])
wp = scipy.interpolate.interp1d(ky, pflux[rad_ind,:])
omega = scipy.interpolate.interp1d(ky, freqs[1,:])
gamma = scipy.interpolate.interp1d(ky, freqs[2,:])

# %%
fig7 = plt.figure(7, figsize=(3.375, 3.375))


gs7 = mpl.gridspec.GridSpec(2, 1, height_ratios=[1,2])
gs7_inner = mpl.gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs7[0], wspace=0.5)

ax70 = plt.subplot(gs7[1])
ax710 = plt.subplot(gs7_inner[0])
ax711 = plt.subplot(gs7_inner[1])

kplot = np.linspace(0.02, 1.4, 128)

def lorentzian(ky, k0, g):
    return 1.0/(np.pi * g * (1 + ((ky-k0)/g)**2))

def model_wqi(ky):
    return 2.0*(1-(ky/1.6)**0.5)

#gmax = np.max(gamma(kplot))
#mixing = gamma(kplot)/kplot

#plt.plot(kplot, wqi(kplot))

loc_spectrum = lorentzian(kplot, 0.5, 0.25) * model_wqi(kplot)
soc_spectrum = lorentzian(kplot, 0.5, 0.16) * model_wqi(kplot)


loc_spectrum = loc_spectrum / np.sum(loc_spectrum) * 2 / 1.8 * 100
soc_spectrum = soc_spectrum / np.sum(soc_spectrum) * 2 / 1.8 * 100

ax70.plot(kplot, loc_spectrum, c='r')
ax70.plot(kplot, soc_spectrum, c='b')
ax70.set_ylim(bottom=0)

xbar = np.arange(4)
barlabels=['Ia', 'Ib', 'II', 'III']
soc_values = [0.3, 1.8, 0.0, 0.33]
loc_values = [0.6, 1.4, 0.2, 0.3]

ax70.set_xlabel(r'$k_y \rho_s$')
ax70.set_ylabel(r'$W_{Qi,k} \left\langle \bar{\phi}_{k_1}^2 \right\rangle$ (arb. units)')
ax70.set_title('Example Ion Heat Flux Spectrum')

ax710.bar(xbar, loc_values, color='r', align='center', tick_label=barlabels)
ax711.bar(xbar, soc_values, color='b', align='center', tick_label=barlabels)

ax710.set_title('LOC')
ax711.set_title('SOC')
ax710.set_ylim([0,2])
ax711.set_ylim([0,2])
ax710.set_ylabel('Intensity (a.u.)')

plt.tight_layout()
plt.tight_layout()

plt.savefig('figure7.eps', format='eps', dpi=1200, facecolor='white')