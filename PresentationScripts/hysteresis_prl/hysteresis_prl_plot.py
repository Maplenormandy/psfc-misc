# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 13:52:05 2018

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

# %% Figure 1: time traces and rotation profiles

fig1_1 = np.load('figure1_1.npz')

nl04time = fig1_1['nl04time']
nl04data = fig1_1['nl04data']
vtime = fig1_1['vtime']
vdata = fig1_1['vdata']

fig1 = plt.figure(1, figsize=(3.375, 3.375*1.1))
gs1 = mpl.gridspec.GridSpec(2, 1, height_ratios=[2,1])
gs1_inner = mpl.gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs1[0], hspace=0.0)

ax10 = plt.subplot(gs1_inner[0])
ax10.axvspan(0.57, 0.63, color=(0.5,0.5,1.0))
ax10.axvspan(0.93, 0.99, color=(1.0,0.5,0.5))
ax10.plot(nl04time, nl04data/1e20/0.6, c='k')
ax10.yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.1))
ax10.set_ylabel(r'$\bar{n}_e$ ($10^{20} \mathrm{m}^{-3}$)')
ax10.set_ylim([0.68,1.02])
plt.setp(ax10.get_xticklabels(), visible=False)

offset = ((7.0)-4.0)/(25.0-7.0)*6

ax11 = plt.subplot(gs1_inner[1], sharex=ax10)
ax11.axvspan(0.57, 0.63, color=(0.5,0.5,1.0))
ax11.axvspan(0.93, 0.99, color=(1.0,0.5,0.5))
ax11.plot(vtime, vdata+offset, c='k', marker='.')
ax11.axhline(c='k', ls='--')
ax11.yaxis.set_major_locator(mpl.ticker.FixedLocator([-10,0,10]))
ax11.set_ylabel('$v_{tor}$ (km/s)')
ax11.set_xlabel('time (sec)')
ax11.set_xlim([0.5, 1.5])

fig12 = np.load('figure1_2.npz')

ax12 = plt.subplot(gs1[1])
ax12.axvspan(0.56, 0.59, color=(0.7, 1.0, 1.0))
ax12.errorbar(fig12['soc_roa'], fig12['soc_pro'], fig12['soc_perr'], c='b')
ax12.errorbar(fig12['loc_roa'], fig12['loc_pro'], fig12['loc_perr'], c='r')
ax12.axhline(c='k', ls='--')
ax12.xaxis.set_major_locator(mpl.ticker.MultipleLocator(0.2))
ax12.yaxis.set_major_locator(mpl.ticker.MultipleLocator(4))
ax12.set_ylabel('$f_{tor}$ (kHz)')
ax12.set_xlabel('r/a')

plt.tight_layout()
plt.tight_layout()

plt.savefig('figure1.eps', format='eps', dpi=1200)

# %% Figure 2: Hysteresis plot

fig2 = plt.figure(2, figsize=(3.375, 3.375*0.75))

fig2d = np.load('figure2.npz')

plt.plot(fig2d['hys01'][0,:], fig2d['hys01'][1,:], marker='.', c=(1.0, 0.5, 0.0))
plt.plot(fig2d['hys02'][0,:], fig2d['hys02'][1,:], marker='.', c='m')
plt.plot(fig2d['hys03'][0,:], fig2d['hys03'][1,:], marker='.', c='c')

plt.axhspan(3, 8, xmax=0.73, color=(1.0,0.7,0.7))
plt.axhspan(-16, -11, xmin=0.36, color=(0.7,0.7,1.0))


#fig2.add_subplot(111, frameon=False)
#plt.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
#plt.grid(False)
#plt.xlabel(r'$\bar{n}_e$ [$10^{20} \mathrm{m}^{-3}$]')
#plt.ylabel(r'$v_{tor}$ [km/s]')

plt.ylabel(r'$v_{tor}$ (km/s)')
plt.xlabel(r'$\bar{n}_e$ ($10^{20} \mathrm{m}^{-3}$)')

plt.tight_layout()
plt.tight_layout()

plt.savefig('figure2.eps', format='eps', dpi=1200, facecolor='white')

# %% Figure 3: Profile matched plots


fig3 = plt.figure(3, figsize=(3.375*2,3.375*0.75))
gs3o = mpl.gridspec.GridSpec(1, 2, width_ratios=[3,1])
gs3 = mpl.gridspec.GridSpecFromSubplotSpec(2, 3, subplot_spec=gs3o[0], hspace=0.0)
gs3i = mpl.gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs3o[1], hspace=0.0)
#gs3 = mpl.gridspec.GridSpec(2, 4, hspace=0.00)

ax300 = plt.subplot(gs3[0,0])
ax310 = plt.subplot(gs3[1,0], sharex=ax300)

ax301 = plt.subplot(gs3[0,1], sharex=ax300)
ax311 = plt.subplot(gs3[1,1], sharex=ax300)

ax302 = plt.subplot(gs3[0,2], sharex=ax300)
ax312 = plt.subplot(gs3[1,2], sharex=ax300)

ax303 = plt.subplot(gs3i[0])
ax313 = plt.subplot(gs3i[1], sharex=ax303)

fig3_profs = np.load('fig3_data1.npy')
fig3_nedata_loc = np.load('fig3_data2a.npy')
fig3_nedata_soc = np.load('fig3_data2b.npy')
fig3_tedata = np.load('fig3_data3.npy')

r3 = fig3_profs[0,:]

ax300.errorbar(fig3_nedata_loc[0,:], fig3_nedata_loc[1,:], yerr=fig3_nedata_loc[2,:], c='r', fmt='.')
ax300.errorbar(fig3_nedata_soc[0,:], fig3_nedata_soc[1,:], yerr=fig3_nedata_soc[2,:], c='b', fmt='.')

ax300.errorbar(r3, fig3_profs[1,:], yerr=fig3_profs[2,:], c='r')
ax300.errorbar(r3, fig3_profs[3,:], yerr=fig3_profs[4,:], c='b')
#ax300.plot(r3, fig3_profs[1,:], c='r')
#ax300.fill_between(r3, fig3_profs[1,:]-2*fig3_profs[2,:], fig3_profs[1,:]+2*fig3_profs[2,:], alpha=0.2, facecolor='r', linewidth=0)
#ax300.fill_between(r3, fig3_profs[1,:]-fig3_profs[2,:], fig3_profs[1,:]+fig3_profs[2,:], alpha=0.2, facecolor='r', linewidth=0)

ax310.errorbar(r3, fig3_profs[5,:], yerr=fig3_profs[6,:], c='r')
ax310.errorbar(r3, fig3_profs[7,:], yerr=fig3_profs[8,:], c='b')

ax301.errorbar(fig3_tedata[0,:], fig3_tedata[1,:], yerr=fig3_tedata[2,:], c='r', fmt='.')
ax301.errorbar(fig3_tedata[3,:], fig3_tedata[4,:], yerr=fig3_tedata[5,:], c='b', fmt='.')
ax301.errorbar(r3, fig3_profs[9,:], yerr=fig3_profs[10,:], c='r')
ax301.errorbar(r3, fig3_profs[11,:], yerr=fig3_profs[12,:], c='b')

ax311.errorbar(r3, fig3_profs[13,:], yerr=fig3_profs[14,:], c='r')
ax311.errorbar(r3, fig3_profs[15,:], yerr=fig3_profs[16,:], c='b')

plt.setp(ax300.get_xticklabels(), visible=False)
plt.setp(ax301.get_xticklabels(), visible=False)
plt.setp(ax302.get_xticklabels(), visible=False)


fig32 = np.load('figure3_2.npz')

ax302.errorbar(fig32['loc_rho'], fig32['loc_prof'], fig32['loc_err'], c='r')
ax302.errorbar(fig32['soc_rho'], fig32['soc_prof'], fig32['soc_err'], c='b')

ax312.errorbar(fig32['loc_rho'], fig32['alti_loc'], fig32['alti_loc_err'], c='r')
ax312.errorbar(fig32['soc_rho'], fig32['alti_soc'], fig32['alti_soc_err'], c='b')

ax300.set_xlim([0.0, 1.0])
#ax310.set_ylim([-0.05, 2.4])
ax301.set_ylim([0.0, 2.1])
ax302.set_ylim([0.0, 2.1])
ax311.set_ylim([-0.05, 7.2])
#ax312.set_ylim([-0.05, 7.2])

ax310.set_xlabel('r/a')
ax311.set_xlabel('r/a')
ax312.set_xlabel('r/a')

ax300.text(0.1, 0.2, '$n_e$ ($10^{20} \mathrm{m}^{-3}$)', transform=ax300.transAxes)
ax301.text(0.1, 0.2, '$T_e$ (keV)', transform=ax301.transAxes)
ax302.text(0.1, 0.2, '$T_i$ (keV)', transform=ax302.transAxes)
ax310.text(0.1, 0.7, '$a/L_{ne}$', transform=ax310.transAxes)
ax311.text(0.1, 0.7, '$a/L_{Te}$', transform=ax311.transAxes)
ax312.text(0.1, 0.7, '$a/L_{Ti}$', transform=ax312.transAxes)


ax51 = ax303
ax52 = ax313

cgyro_freqs = np.load('./cgyro_outputs/all_freqs.npz')

for foldb in reversed(cgyro_freqs['folders']):
    fold = foldb.decode("utf-8")
    data = cgyro_freqs[fold]
    
    ky = data[0,:]
    #kmax = np.searchsorted(ky, 1.42)
    kmax=len(ky)
    kmin=0
    #kmin = np.searchsorted(ky, 1.42)
    omega = data[1,:]
    gamma = data[2,:]

    c = (0.7, 0.7, 0.7)
    alpha = 1.0
    if fold == 'soc_mid':
        c = 'b'
        alpha = 1.0
    elif fold == 'loc_mid':
        c = 'r'
        alpha = 1.0
    elif fold == 'soc_low':
        c = (0.5, 0.5, 1.0)
        alpha = 1.0
    elif fold == 'loc_hig':
        c = (1.0, 0.5, 0.5)
        alpha = 1.0

    ax51.plot(ky[kmin:kmax], omega[kmin:kmax], marker='.', c=c)
    ax52.plot(ky[kmin:kmax], gamma[kmin:kmax], marker='.', c=c)

ax51.axhline(ls='--', c='k')

ax51.text(0.15, 0.38, 'r/a = 0.575')

ax52.axhline(ls='--', c='k')

ax51.set_ylim([-0.59, 0.59])
ax52.set_ylim([-0.08, 0.22])

ax52.set_xlim([0.1, 1.42])


ax51.set_ylabel(r'$\omega_R$ ($c_s/a$)')
plt.setp(ax51.get_xticklabels(), visible=False)

ax52.set_ylabel(r'$\gamma$ ($c_s/a$)')
#ax52.yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.1))
#ax52.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.05))
ax52.set_xlabel(r'$k_y \rho_s$')

plt.tight_layout(h_pad=0.0, w_pad=0.08)
plt.tight_layout(h_pad=0.0, w_pad=0.08)

plt.savefig('figure3.eps', format='eps', dpi=1200, facecolor='white')

# %% Figure 4: Reflecometer plots

fig4 = plt.figure(4, figsize=(3.375, 3.375*0.75))

fig4d = np.load('figure4.npz')

Sxx_loc_down = fig4d['Sxx_loc_down']
Sxx_soc_down = fig4d['Sxx_soc_down']
f_down = fig4d['f_down']

plt.semilogy(f_down, Sxx_soc_down, c='b')
plt.semilogy(f_down, Sxx_loc_down, c='r')
plt.xlim([-450,450])
plt.xlabel('f (kHz)')
plt.ylim([1e-13, 1e-8])
plt.ylabel('(arb. units)')
plt.text(160, 1e-9, 'r/a = 0.53')

plt.tight_layout()
plt.tight_layout()

plt.savefig('figure4.eps', format='eps', dpi=1200, facecolor='white')

# %% Figure 5: Growth rates / real frequencies.
# Note: The data here is generated from ~/hys2/plot_all_freqs.py


# %% Figure 6: QL weights - need to go on engaging, also what to do with flux data?

fig6 = plt.figure(6, figsize=(3.375,3.375*0.75))
gs6 = mpl.gridspec.GridSpec(1, 2, width_ratios=[4,1])

ax60 = plt.subplot(gs6[0])
ax61 = plt.subplot(gs6[1])

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

ax60.axvline(0.465, 0, 0.85, ls=':', c=(0.7, 0.7, 0.7))
ax60.axvline(1.15, 0, 0.95, c=(0.7, 0.7, 0.7))
ax60.axvline(1.65, 0, 0.95, c=(0.7, 0.7, 0.7))

ax60.plot(ky, qiflux[rad_ind,:], marker='.', label='$W {Q_i}$', c='b')
ax60.plot(ky, qeflux[rad_ind,:], marker='.', label='$W {Q_e}$', c='g')
ax60.plot(ky, pflux[rad_ind,:], marker='.', label='$W {\Gamma_e}$', c='r')
#plt.errorbar(ky, momflux[rad_ind,:], yerr=momflux_std[rad_ind,:], marker='.', label='W,Pi')

ax60.set_xscale('log')
ax60.set_ylim([-1.0,3.6])
ax60.set_xlim([0.1,25])
ax60.set_ylabel('(GB units)')
ax60.set_xlabel(r'$k_y \rho_s$')

ax60.axhline(0.0, ls='--', c='black')

#plt.title('r/a = ' + str(radii[rad_ind]))
pos = np.array([0.5, 1.5, 2.5])
vals = [2.0, 3.0, 0.01]
ax61.bar(pos, vals, width=1.0, color=('b', 'g', 'r'), tick_label=('$Q_i$','$Q_e$','$\Gamma_e$'), align='center')

ax61.yaxis.tick_right()
ax61.xaxis.set_ticks_position('bottom')
#ax61.set_xlabel('Anomalous Flux (GB units)')

plt.tight_layout(w_pad=0.2)
plt.tight_layout(w_pad=0.2)

ax60.text(0.16, 3.1, 'Ion-Scale')
ax60.text(1.8, 3.1, 'Elec.-Scale')
ax60.text(0.2, 2.5, 'Ia')
ax60.text(0.6, 2.5, 'Ib')
ax60.text(1.25, 2.5, 'II')
ax60.text(6.0, 2.5, 'III')

plt.savefig('figure6.eps', format='eps', dpi=1200, facecolor='white')


# %% Figure 7, the cartoon

fig7 = plt.figure(7, figsize=(3.375, 3.375*0.8))


gs7 = mpl.gridspec.GridSpec(2, 1, height_ratios=[3,4])
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