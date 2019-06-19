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

# %%

tifit = np.interp(r3, fig32['soc_rho'], fig32['soc_prof'])

plt.figure()
plt.plot(r3, tifit*fig3_profs[5,:])
plt.title('$v_* \propto T_i / L_n$')
