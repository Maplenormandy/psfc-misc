# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 13:52:05 2018

@author: normandy
"""

import matplotlib as mpl
import matplotlib.pyplot as plt

import numpy as np
#import scipy.interpolate

font = {'family' : 'serif',
        'serif': ['Computer Modern'],
        'size'   : 9}

mpl.rc('font', **font)

# %%

data1_soc = np.loadtxt('soc_mid_input_1.txt', skiprows=1)
data2_soc = np.loadtxt('soc_mid_input_2.txt', skiprows=1)

data1_loc = np.loadtxt('loc_mid_input_1.txt', skiprows=1)
data2_loc = np.loadtxt('loc_mid_input_2.txt', skiprows=1)

# %%



fig3 = plt.figure(3, figsize=(3.375*2,3.375*0.75))
gs3o = mpl.gridspec.GridSpec(1, 2, width_ratios=[2,1])
gs3 = mpl.gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs3o[0], hspace=0.0)
ax3i = plt.subplot(gs3o[1])
#gs3 = mpl.gridspec.GridSpec(2, 4, hspace=0.00)

#ax300 = plt.subplot(gs3[0,0])
ax310 = plt.subplot(gs3[0])

#ax301 = plt.subplot(gs3[0,1], sharex=ax300)
ax311 = plt.subplot(gs3[1], sharex=ax310)

#ax302 = plt.subplot(gs3[0,2], sharex=ax300)
#ax312 = plt.subplot(gs3[1,2], sharex=ax300)
ax312 = ax311

fig3_profs = np.load('fig3_data1.npy')
fig3_nedata_loc = np.load('fig3_data2a.npy')
fig3_nedata_soc = np.load('fig3_data2b.npy')
fig3_tedata = np.load('fig3_data3.npy')

r3 = fig3_profs[0,:]

#ax300.errorbar(fig3_nedata_loc[0,:], fig3_nedata_loc[1,:], yerr=fig3_nedata_loc[2,:], c='r', fmt='.')
#ax300.errorbar(fig3_nedata_soc[0,:], fig3_nedata_soc[1,:], yerr=fig3_nedata_soc[2,:], c='b', fmt='.')

#ax300.errorbar(r3, fig3_profs[1,:], yerr=fig3_profs[2,:], c='r')
#ax300.errorbar(r3, fig3_profs[3,:], yerr=fig3_profs[4,:], c='b')
#ax300.plot(r3, fig3_profs[1,:], c='r')
#ax300.fill_between(r3, fig3_profs[1,:]-2*fig3_profs[2,:], fig3_profs[1,:]+2*fig3_profs[2,:], alpha=0.2, facecolor='r', linewidth=0)
#ax300.fill_between(r3, fig3_profs[1,:]-fig3_profs[2,:], fig3_profs[1,:]+fig3_profs[2,:], alpha=0.2, facecolor='r', linewidth=0)

ax310.errorbar(r3, fig3_profs[5,:], yerr=fig3_profs[6,:], c='r')
ax310.errorbar(r3, fig3_profs[7,:], yerr=fig3_profs[8,:], c='b')

#ax301.errorbar(fig3_tedata[0,:], fig3_tedata[1,:], yerr=fig3_tedata[2,:], c='r', fmt='.')
#ax301.errorbar(fig3_tedata[3,:], fig3_tedata[4,:], yerr=fig3_tedata[5,:], c='b', fmt='.')
#ax301.errorbar(r3, fig3_profs[9,:], yerr=fig3_profs[10,:], c='r')
#ax301.errorbar(r3, fig3_profs[11,:], yerr=fig3_profs[12,:], c='b')

ax311.errorbar(r3, fig3_profs[13,:], yerr=fig3_profs[14,:], c='r')
ax311.errorbar(r3, fig3_profs[15,:], yerr=fig3_profs[16,:], c='b')

#plt.setp(ax300.get_xticklabels(), visible=False)
#plt.setp(ax301.get_xticklabels(), visible=False)
#plt.setp(ax302.get_xticklabels(), visible=False)


fig32 = np.load('figure3_2.npz')

#ax302.errorbar(fig32['loc_rho'], fig32['loc_prof'], fig32['loc_err'], c='r')
#ax302.errorbar(fig32['soc_rho'], fig32['soc_prof'], fig32['soc_err'], c='b')

ax312.errorbar(fig32['loc_rho'], fig32['alti_loc'], fig32['alti_loc_err'], c='r', marker='x')
ax312.errorbar(fig32['soc_rho'], fig32['alti_soc'], fig32['alti_soc_err'], c='b', marker='x')

#ax300.set_xlim([0.0, 1.0])
ax310.set_ylim([-0.05, 2])
#ax301.set_ylim([0.0, 2.1])
#ax302.set_ylim([0.4, 2.1])
ax311.set_ylim([-0.05, 7.2])
#ax312.set_ylim([-0.05, 7.2])

#ax310.set_xlim([0.0, 1.0])

ax310.set_xlabel('r/a')
ax311.set_xlabel('r/a')
#ax312.set_xlabel('r/a')

#ax300.text(0.1, 0.2, '$n_e$ ($10^{20} \mathrm{m}^{-3}$)', transform=ax300.transAxes)
#ax301.text(0.1, 0.2, '$T_e$ (keV)', transform=ax301.transAxes)
#ax302.text(0.1, 0.2, '$T_i$ (keV)', transform=ax302.transAxes)

ax310.text(0.2, 0.6, '$a/L_{ne}$', transform=ax310.transAxes)
ax311.text(0.2, 0.6, '$a/L_{Te}$', transform=ax311.transAxes)
ax311.text(0.6, 0.2, '$a/L_{Ti}$', transform=ax312.transAxes)




def calculateShear(data1, data2):
    roa = data1[1:-1,0]
    r = data1[:,1]
    q = data1[1:-1,3]
    omega0 = data1[:,4]
    te = data2[1:-1,1]

    domega0dr = (omega0[2:] - omega0[:-2]) / (r[2:] - r[:-2]) # 1/m/s

    mD = 1876544.16 # mass of Deuterium in keV
    c = 2.998e8
    cs = np.sqrt(te/mD) * c # sound speed in m/s

    gammae = -r[1:-1] * domega0dr / q
    a = 0.218

    gammae_norm = gammae * (a/cs)
    return roa, gammae_norm

roa_soc, gammae_soc = calculateShear(data1_soc, data2_soc)
roa_loc, gammae_loc = calculateShear(data1_loc, data2_loc)

rgam = np.array([ 0.35  , 0.425 , 0.5  ,  0.575 , 0.65 ,  0.725 , 0.8  ])
ggam = np.array([ 0.027435 , 0.076341,  0.12212 ,  0.16693 ,  0.20676 ,  0.22962 ,  0.52844  ])


ax3i.plot(roa_soc[:], -gammae_soc[:], c='b')
ax3i.plot(roa_loc[:], -gammae_loc[:], c='r')
ax3i.plot(rgam[:5], ggam[:5], marker='o', c='b')
ax3i.axhline(ls='--', c='k')
ax3i.set_ylabel(r'$[a/c_s]$')
ax3i.set_xlabel('r/a')
ax3i.set_xlim([0.25, 0.75])
ax3i.set_ylim([-.05, .19])

ax3i.text(0.2, 0.65, '$\gamma_{\mathrm{max}}$', transform=ax3i.transAxes)
ax3i.text(0.5, 0.26, '$\gamma_{E}$', transform=ax3i.transAxes)


plt.tight_layout(h_pad=0.0, w_pad=0.08)
plt.tight_layout(h_pad=0.0, w_pad=0.08)

plt.savefig('shear_locsoc.pdf', format='pdf', dpi=1200, facecolor='white')
