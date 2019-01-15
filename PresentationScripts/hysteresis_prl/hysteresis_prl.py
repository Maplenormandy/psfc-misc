# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 13:52:05 2018

@author: normandy
"""

import matplotlib as mpl
import matplotlib.pyplot as plt

import numpy as np
import scipy.io, scipy.signal

import MDSplus

import eqtools

font = {'family' : 'serif',
        'serif': ['Computer Modern'],
        'size'   : 10}

mpl.rc('font', **font)

# %% General function definitions

class ThacoData:
    def __init__(self, thtNode, shot=None, tht=None, path='.HELIKE.PROFILES.Z', time=0.95):
        if (thtNode == None):
            self.shot = shot
            self.specTree = MDSplus.Tree('spectroscopy', shot)

            if (tht == 0):
                self.tht = ''
            else:
                self.tht = str(tht)

            self.thtNode = self.specTree.getNode('\SPECTROSCOPY::TOP.HIREXSR.ANALYSIS' + self.tht + path)
        else:
            self.thtNode = thtNode

        e = eqtools.CModEFITTree(shot)

        proNode = self.thtNode.getNode('PRO')
        perrNode = self.thtNode.getNode('PROERR')
        rhoNode = self.thtNode.getNode('RHO')

        rpro = proNode.data()
        rperr = perrNode.data()
        rrho = rhoNode.data()
        rtime = rhoNode.dim_of()

        goodTimes = (rtime > 0).sum()

        self.time = rtime.data()[:goodTimes]
        self.rho = rrho[0,:] # Assume unchanging rho bins
        self.roa = e.psinorm2roa(self.rho, time)
        self.pro = rpro[:,:goodTimes,:len(self.rho)]
        self.perr = rperr[:,:goodTimes,:len(self.rho)]

def trimNodeData(node, t0=0.5, t1=1.5):
    time = node.dim_of().data()
    data = node.data()
    i0, i1 = np.searchsorted(time, (t0, t1))
    return time[i0:i1+1], data[i0:i1+1]
def trimData(time, data, t0=0.5, t1=1.5):
    i0, i1 = np.searchsorted(time, (t0, t1))
    return time[i0:i1+1], data[i0:i1+1]

# %% Data loading
elecTree = MDSplus.Tree('electrons', 1160506007)
specTree = MDSplus.Tree('spectroscopy', 1160506007)

nl04time, nl04data = trimNodeData(elecTree.getNode(r'\ELECTRONS::TOP.TCI.RESULTS:NL_04'))
vnode = specTree.getNode(r'\SPECTROSCOPY::TOP.HIREX_SR.ANALYSIS.W:VEL')
vtime, vdata = trimData(vnode.dim_of().data(), vnode.data()[0])

# %%

tifit_soc = scipy.io.readsav('/home/normandy/fits/tifit_1120106012_THT0.dat')
omfit_soc = scipy.io.readsav('/home/normandy/fits/omfit_1120106012_THT0.dat')
tifit_loc = scipy.io.readsav('/home/normandy/fits/tifit_1120106016_THT0.dat')
omfit_loc = scipy.io.readsav('/home/normandy/fits/omfit_1120106016_THT0.dat')

thacodata_soc = ThacoData(None, 1120106012, 0)
thacodata_loc = ThacoData(None, 1120106016, 0)


# %% Figure 1: time traces and rotation profiles\

fig1 = plt.figure(1, figsize=(3.375, 3.375*1.2))
gs1 = mpl.gridspec.GridSpec(2, 1, height_ratios=[2,1])
gs1_inner = mpl.gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs1[0], hspace=0.0)

ax10 = plt.subplot(gs1_inner[0])
ax10.axvspan(0.57, 0.63, color=(0.5,0.5,1.0))
ax10.axvspan(0.93, 0.99, color=(1.0,0.5,0.5))
ax10.plot(nl04time, nl04data/1e20/0.6, c='k')
ax10.yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.1))
ax10.set_ylabel(r'$\bar{n}_e$ ($10^{19} \mathrm{m}^{-3}$)')
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

time_index = 20
radial_index = 20

bsom_soc = omfit_soc.bsom[time_index]['fit'][0]
bsom_loc = omfit_loc.bsom[time_index]['fit'][0]

ax12 = plt.subplot(gs1[1])
ax12.axvspan(0.35, 0.65, color=(0.7, 1.0, 1.0))
ax12.errorbar(thacodata_soc.roa[:radial_index], thacodata_soc.pro[1,time_index,:radial_index], thacodata_soc.perr[1, time_index,:radial_index], c='b')
ax12.errorbar(thacodata_loc.roa[:radial_index], thacodata_loc.pro[1,time_index,:radial_index], thacodata_loc.perr[1, time_index,:radial_index], c='r')
ax12.axhline(c='k', ls='--')
ax12.xaxis.set_major_locator(mpl.ticker.MultipleLocator(0.2))
ax12.yaxis.set_major_locator(mpl.ticker.MultipleLocator(4))
ax12.set_ylabel('$f_{tor}$ (kHz)')
ax12.set_xlabel('r/a')

plt.tight_layout(h_pad=1.6)
plt.tight_layout(h_pad=1.6)

plt.savefig('figure1.eps', format='eps', dpi=1200)

# %% Figure 2: Hysteresis plot

def plotHysteresis(shot, ax, c='b'):
    elecTree = MDSplus.Tree('electrons', shot)

    nl04Node = elecTree.getNode('\ELECTRONS::TOP.TCI.RESULTS:NL_04')
    specTree = MDSplus.Tree('spectroscopy', shot)
    velNode = specTree.getNode(r'\SPECTROSCOPY::TOP.HIREX_SR.ANALYSIS.W:VEL')

    vtime = velNode.dim_of().data()
    nltime = nl04Node.dim_of().data()


    vlow = np.searchsorted(vtime, 0.55)
    vhigh = np.searchsorted(vtime, 1.25)+2

    vtime = vtime[vlow:vhigh]
    vdata = velNode.data()[0]
    vdata = vdata[vlow:vhigh]

    nlData = np.interp(vtime, nltime, nl04Node.data())/1e20/0.6
    offset = ((shot%100)-4.0)/(25.0-7.0)*6

    ax.plot(nlData, vdata+offset, label=str(shot), marker='.', c=c)
    
fig2 = plt.figure(2, figsize=(3.375, 3.375*1.5))
gs2 = mpl.gridspec.GridSpec(2,1)

ax20 = plt.subplot(gs2[0])
ax21 = plt.subplot(gs2[1])

plotHysteresis(1160506007, ax20, c='k')
plotHysteresis(1160506008, ax20, c='c')
plotHysteresis(1160506024, ax20, c='m')

plotHysteresis(1160506009, ax21, c='c')
plotHysteresis(1160506010, ax21, c='m')

ax20.axhspan(3, 8, xmax=0.73, color=(1.0,0.7,0.7))
ax20.axhspan(-16, -11, xmin=0.36, color=(0.7,0.7,1.0))

ax21.set_xlim([1.05, 1.55])
ax21.axhspan(17, 25, xmax=0.63, color=(1.0,0.7,0.7))
ax21.axhspan(-7, -2, xmin=0.38, color=(0.7,0.7,1.0))

#fig2.add_subplot(111, frameon=False)
#plt.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
#plt.grid(False)
#plt.xlabel(r'$\bar{n}_e$ [$10^{19} \mathrm{m}^{-3}$]')
#plt.ylabel(r'$v_{tor}$ [km/s]')

ax20.set_ylabel(r'$v_{tor}$ (km/s)')
ax20.set_xlabel(r'$\bar{n}_e$ ($10^{19} \mathrm{m}^{-3}$)')
ax21.set_xlabel(r'$\bar{n}_e$ ($10^{19} \mathrm{m}^{-3}$)')
ax21.set_ylabel(r'$v_{tor}$ (km/s)')

plt.tight_layout(h_pad=1.08)
plt.tight_layout(h_pad=1.08)

plt.savefig('figure2.eps', format='eps', dpi=1200, facecolor='white')

# %% Figure 3: Profile matched plots


fig3 = plt.figure(3, figsize=(3.375*2,3.375))
gs3 = mpl.gridspec.GridSpec(2, 3, hspace=0.00)

ax300 = plt.subplot(gs3[0,0])
ax310 = plt.subplot(gs3[1,0], sharex=ax300)

ax301 = plt.subplot(gs3[0,1], sharex=ax300)
ax311 = plt.subplot(gs3[1,1], sharex=ax300)

ax302 = plt.subplot(gs3[0,2], sharex=ax300)
ax312 = plt.subplot(gs3[1,2], sharex=ax300)

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

ax300.set_xlim([0.0, 1.0])
ax310.set_ylim([-0.05, 2.4])
ax311.set_ylim([-0.05, 7.8])
ax301.set_ylim([0.0, 2.1])

time_index = 20
radial_index = 25

bsti_soc = tifit_soc.bsti[time_index][0][0][0]
bsti_loc = tifit_loc.bsti[time_index][0][0][0]

ax302.errorbar(bsti_loc['rho'][::2], bsti_loc['prof'][::2]-0.12, bsti_loc['err'][::2], c='r')
ax302.errorbar(bsti_soc['rho'][::2], bsti_soc['prof'][::2]-0.12, bsti_soc['err'][::2], c='b')

alti_soc = -bsti_soc['dprof']/(bsti_soc['prof']-0.12)
alti_soc_err = np.sqrt(alti_soc**2 * (bsti_soc['err']**2 / (bsti_soc['prof']-0.12)**2 + bsti_soc['derr']**2/bsti_soc['dprof']**2))
alti_loc = -bsti_loc['dprof']/(bsti_loc['prof']-0.12)
alti_loc_err = np.sqrt(alti_loc**2 * (bsti_loc['err']**2 / (bsti_loc['prof']-0.12)**2 + bsti_loc['derr']**2/bsti_loc['dprof']**2))

ax312.errorbar(bsti_loc['rho'][::2], alti_loc[::2], alti_loc_err[::2], c='r')
ax312.errorbar(bsti_soc['rho'][::2], alti_soc[::2], alti_soc_err[::2], c='b')

ax302.set_ylim([0.0, 2.1])
ax312.set_ylim([-0.05, 4.8])

ax310.set_xlabel('r/a')
ax311.set_xlabel('r/a')
ax312.set_xlabel('r/a')

ax300.text(0.1, 0.2, '$n_e$ ($10^{20} \mathrm{m}^{-3}$)', transform=ax300.transAxes)
ax301.text(0.1, 0.2, '$T_e$ (keV)', transform=ax301.transAxes)
ax302.text(0.1, 0.2, '$T_i$ (keV)', transform=ax302.transAxes)
ax310.text(0.1, 0.7, '$a/L_{ne}$', transform=ax310.transAxes)
ax311.text(0.1, 0.7, '$a/L_{Te}$', transform=ax311.transAxes)
ax312.text(0.1, 0.7, '$a/L_{Ti}$', transform=ax312.transAxes)

plt.tight_layout()
plt.tight_layout()

plt.savefig('figure3.eps', format='eps', dpi=1200, facecolor='white')

# %%


# %% Figure 4: Reflecometer plots

elecTree = MDSplus.Tree('electrons', 1160506007)

# 9 and 10 are normally the best
sig88ui = elecTree.getNode('\ELECTRONS::TOP.REFLECT.CPCI:DT132_1:INPUT_09').data()
sig88uq = elecTree.getNode('\ELECTRONS::TOP.REFLECT.CPCI:DT132_1:INPUT_10').data()
sig88ut = elecTree.getNode('\ELECTRONS::TOP.REFLECT.CPCI:DT132_1:INPUT_09').dim_of().data()

ci = np.mean(sig88ui)
cq = np.mean(sig88uq)

nl04Node = elecTree.getNode('\ELECTRONS::TOP.TCI.RESULTS:NL_04')


t1, t2 = np.searchsorted(sig88ut, (0.4,1.6))
#0.5944-0.5954
#0.9625-0.9650

si = sig88ui[t1:t2]
sq = sig88uq[t1:t2]
st = sig88ut[t1:t2]
ci = np.median(si)
cq = np.median(sq)
z = (si-ci)+1j*(sq-cq)


# %%
t_soc, t_loc = np.searchsorted(st, (0.5949, 0.9637))

total_samples=4096*2
down_samples =512

z_soc = z[t_soc-total_samples/2:t_soc+total_samples/2]
z_loc = z[t_loc-total_samples/2:t_loc+total_samples/2]

fz_soc = np.fft.fftshift(np.fft.fft(z_soc))/2e3/2e3
fz_loc = np.fft.fftshift(np.fft.fft(z_loc))/2e3/2e3
freqs = np.fft.fftshift(np.fft.fftfreq(total_samples, 1.0/2e6))

fig4 = plt.figure(4, figsize=(3.375, 3.375*0.75))

Sxx_loc_down = np.real(np.average(np.reshape(fz_loc*np.conjugate(fz_loc), (down_samples, -1)), axis=-1))
Sxx_soc_down = np.real(np.average(np.reshape(fz_soc*np.conjugate(fz_soc), (down_samples, -1)), axis=-1))

f_down = np.average(np.reshape(freqs/1e3, (down_samples, -1)), axis=-1)

plt.semilogy(f_down, Sxx_soc_down, c='b')
plt.semilogy(f_down, Sxx_loc_down, c='r')
plt.xlim([-450,450])
plt.xlabel('f (kHz)')
plt.ylim([1e-13, 1e-8])
plt.ylabel('(arb. units)')
plt.text(160, 1e-9, 'r/a = 0.53')

plt.tight_layout(h_pad=1.08)
plt.tight_layout(h_pad=1.08)

plt.savefig('figure4.eps', format='eps', dpi=1200, facecolor='white')

# %% Figure 5: Growth rates / real frequencies.
# Note: The data here is generated from ~/hys2/plot_all_freqs.py

fig5 = plt.figure(5, figsize=(3.375, 3.375*0.75))
gs5 = mpl.gridspec.GridSpec(2, 1, hspace=0.0)

ax51 = plt.subplot(gs5[0])
ax52 = plt.subplot(gs5[1], sharex=ax51)

cgyro_freqs = np.load(file('./cgyro_outputs/all_freqs.npz'))

for fold in reversed(cgyro_freqs['folders']):
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

plt.tight_layout(h_pad=0.0)
plt.tight_layout(h_pad=0.0)

plt.savefig('figure5.eps', format='eps', dpi=1200, facecolor='white')

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
ax60.set_ylim([-1.0,3.5])
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

ax60.text(0.16, 3.0, 'Ion-Scale')
ax60.text(1.8, 3.0, 'Elec.-Scale')
ax60.text(0.2, 2.5, 'Ia')
ax60.text(0.6, 2.5, 'Ib')
ax60.text(1.25, 2.5, 'II')
ax60.text(6.0, 2.5, 'III')

plt.savefig('figure6.eps', format='eps', dpi=1200, facecolor='white')
