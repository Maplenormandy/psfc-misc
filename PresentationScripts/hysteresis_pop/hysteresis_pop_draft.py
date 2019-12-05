# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 13:52:05 2018

Note that the order of figures in the code does not necessarily reflect the order
of figures in the paper

@author: normandy
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors

import numpy as np
import scipy.io, scipy.signal

import cPickle as pkl

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




# %% Figure 1: time traces and rotation profiles\



#plt.savefig('figure1.eps', format='eps', dpi=1200)

# %% Figure 2: Hysteresis loops

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
    
    nlT0 = np.searchsorted(nltime, vtime-np.median(np.diff(vtime))/2)
    nlT1 = np.searchsorted(nltime, vtime+np.median(np.diff(vtime))/2)
    
    nlData = np.zeros(vtime.shape)
    for j in range(len(vtime)):
        nlData[j] = np.average(nl04Node.data()[nlT0[j]:nlT1[j]])/1e20/0.6

    #nlData = np.interp(vtime, nltime, nl04Node.data())/1e20/0.6
    
    offset = ((shot%100)-4.0)/(25.0-7.0)*6

    ax.plot(nlData, vdata+offset, label=str(shot), marker='.', c=c)
    
    
def plotRFHysteresis(shot, ax, c='b'):
    elecTree = MDSplus.Tree('electrons', shot)

    teNode = elecTree.getNode(r'\ELECTRONS::GPC2_TE0')
    specTree = MDSplus.Tree('spectroscopy', shot)
    velNode = specTree.getNode(r'\SPECTROSCOPY::TOP.HIREX_SR.ANALYSIS.W:VEL')

    vtime = velNode.dim_of().data()
    nltime = teNode.dim_of().data()


    vlow = np.searchsorted(vtime, 0.55)
    vhigh = np.searchsorted(vtime, 1.25)+2

    vtime = vtime[vlow:vhigh]
    vdata = velNode.data()[0]
    vdata = vdata[vlow:vhigh]
    
    nlT0 = np.searchsorted(nltime, vtime-np.median(np.diff(vtime))/2)
    nlT1 = np.searchsorted(nltime, vtime+np.median(np.diff(vtime))/2)
    
    nlData = np.zeros(vtime.shape)
    for j in range(len(vtime)):
        nlData[j] = (np.percentile(teNode.data()[nlT0[j]:nlT1[j]],95) + np.percentile(teNode.data()[nlT0[j]:nlT1[j]],5))/2

    #nlData = np.interp(vtime, nltime, nl04Node.data())/1e20/0.6
    
    offset = ((shot%100)-4.0)/(25.0-7.0)*6

    ax.plot(nlData, vdata+offset, label=str(shot), marker='.', c=c)
    
    
fig2 = plt.figure(2, figsize=(3.375*2, 3.375*0.65))
gs2 = mpl.gridspec.GridSpec(1,3)

ax20 = plt.subplot(gs2[0])
ax21 = plt.subplot(gs2[1])
ax22 = plt.subplot(gs2[2])

plotHysteresis(1160506007, ax20, c='k')
plotHysteresis(1160506008, ax20, c='c')
plotHysteresis(1160506024, ax20, c='m')

plotHysteresis(1160506009, ax21, c='c')
plotHysteresis(1160506010, ax21, c='m')

plotRFHysteresis(1160506015, ax22, c='m')

ax20.axhspan(3, 8, xmax=0.73, color=(1.0,0.7,0.7))
ax20.axhspan(-16, -11, xmin=0.36, color=(0.7,0.7,1.0))
ax20.set_title('Case I (0.8 MA Ohmic)', fontsize=10)

ax21.axhspan(17, 25, xmax=0.63, color=(1.0,0.7,0.7))
ax21.axhspan(-7, -2, xmin=0.38, color=(0.7,0.7,1.0))
ax21.set_title('Case II (1.1 MA Ohmic)', fontsize=10)

ax22.axhspan(25, 35, xmin=0.5, color=(1.0,0.7,0.7))
ax22.axhspan(-15, -5, xmax=0.8, color=(0.7,0.7,1.0))
ax22.set_title('Case III (0.8 MA +ICRF)', fontsize=10)

ax20.set_ylabel(r'$v_{tor}$ (km/s)')
ax20.set_xlabel(r'$\bar{n}_e$ ($10^{20} \mathrm{m}^{-3}$)')
ax20.xaxis.set_major_locator(mpl.ticker.MultipleLocator(0.1))

#ax21.set_ylabel(r'$v_{tor}$ (km/s)')
ax21.set_xlabel(r'$\bar{n}_e$ ($10^{20} \mathrm{m}^{-3}$)')
ax21.xaxis.set_major_locator(mpl.ticker.MultipleLocator(0.2))

#ax22.set_ylabel(r'$v_{tor}$ (km/s)')
ax22.set_xlabel(r'$T_e$ (keV)')
ax22.xaxis.set_major_locator(mpl.ticker.MultipleLocator(0.2))

plt.tight_layout()
plt.tight_layout()

plt.savefig('fig_hysteresis.eps', format='eps', dpi=1200, facecolor='white')

# %% Figure 3: Profile matched plots

fig3 = plt.figure(figsize=(3.375*2, 3.375*2.0))
fig3_rows = 6
fig3_cols = 3
gs3 = mpl.gridspec.GridSpec(fig3_rows,fig3_cols)

ax3 = np.array([[plt.subplot(gs3[i,j]) for j in range(fig3_cols)] for i in range(fig3_rows)])

# Comes from mcRaytracing.py
itemp = np.array([ 0.58171643,  0.58086416,  0.58097034,  0.58196616,  0.58472968,
        0.58757312,  0.5930036 ,  0.59770901,  0.60584412,  0.61244813,
        0.62335314,  0.6319093 ,  0.60002627,  0.59638661,  0.59233809,
        0.59035054,  0.58865633,  0.58829467,  0.58894016,  0.59020199,
        0.59318791,  0.59607954,  0.60142963,  0.60597674,  0.61373905,
        0.61998311,  0.63022579,  0.63822313])

def plot_ion(shot, iondata, axti, axvtor, color='b'):
    #offset = ((shot%100)-4.0)/(25.0-7.0)*6
    offset = 0
    
    meas_avg = iondata['meas_avg']
    meas_avg[:,4] = np.nan
    
    #specTree = MDSplus.Tree('spectroscopy', shot)
    #pos = specTree.getNode(r'\SPECTROSCOPY::TOP.HIREXSR.ANALYSIS6.HELIKE.MOMENTS.W:POS').data()
    
    nch = iondata['meas_avg'].shape[1]
    ch = list(reversed(range(nch)))
    
    axti.errorbar(ch, meas_avg[2,:]-itemp, yerr=iondata['meas_std'][2,:], c=color, fmt='.')
    axvtor.errorbar(ch, meas_avg[1,:]+offset, yerr=iondata['meas_std'][1,:], c=color, fmt='.')
    
    
    

def plot_ion_pair(shot, r0):
    socdata = np.load('/home/normandy/git/bsfc/bsfc_fits/fit_data/mf_%d_nh3_t0.npz'%shot)
    locdata = np.load('/home/normandy/git/bsfc/bsfc_fits/fit_data/mf_%d_nh3_t2.npz'%shot)
    
    plot_ion(shot, locdata, ax3[r0,2], ax3[r0+1,2], 'r')
    plot_ion(shot, socdata, ax3[r0,2], ax3[r0+1,2], 'b')
    

plot_ion_pair(1160506007, 0)
plot_ion_pair(1160506009, 2)
plot_ion_pair(1160506015, 4)


def plot_profs(shot, folder, axp, axd, color='b', data='ne'):
    prof = pkl.load(file('/home/normandy/git/psfc-misc/Fitting/FitsPoP2019/%s/%s_dict_fit_%d.pkl'%(folder, data, shot)))
    
    x_max = np.searchsorted(prof['X'], 0.99)
    x_max_d = np.searchsorted(prof['X'], 0.9)

    
    axp.errorbar(prof['X'][:x_max:4], prof['y'][:x_max:4], yerr=prof['err_y'][:x_max:4], c=color, linewidth=0.8)
    axd.errorbar(prof['X'][:x_max_d:4], prof['a_Ly'][:x_max_d:4], yerr=prof['err_a_Ly'][:x_max_d:4], c=color, linewidth=0.8)
    
    axp.errorbar(prof['data_x'], prof['data_y'], yerr=prof['data_err_y'], c=color, fmt='.')
    
    axp.set_xlim([0.0, 1.0])
    axd.set_xlim([0.0, 1.0])

def plot_pair(shot, folder_base, r0):
    plot_profs(shot, folder_base+'_loc', ax3[r0,0], ax3[r0+1,0], 'r', 'ne')
    plot_profs(shot, folder_base+'_soc', ax3[r0,0], ax3[r0+1,0], 'b', 'ne')
    
    plot_profs(shot, folder_base+'_loc', ax3[r0,1], ax3[r0+1,1], 'r', 'te')
    plot_profs(shot, folder_base+'_soc', ax3[r0,1], ax3[r0+1,1], 'b', 'te')

plot_pair(1160506007, '007', 0)
plot_pair(1160506009, '009', 2)
plot_pair(1160506015, '015', 4)

#plt.savefig('fig_hysteresis.eps', format='eps', dpi=1200, facecolor='white')

# %% Plots of growth rates



cgyro_data = np.load(file('/home/normandy/git/psfc-misc/PresentationScripts/hysteresis_pop/1120216_cgyro_data.npz'))

case1_transp = scipy.io.netcdf.netcdf_file('/home/pablorf/Cao_transp/12345B05/12345B05.CDF')
case2_transp = scipy.io.netcdf.netcdf_file('/home/pablorf/Cao_transp/12345A05/12345A05.CDF')

fig4 = plt.figure(4, figsize=(3.375*2, 3.375*1.2))
gs4 = mpl.gridspec.GridSpec(2, 3)

ax400 = plt.subplot(gs4[0,0])
ax410 = plt.subplot(gs4[1,0])

ax401 = plt.subplot(gs4[0,1])
ax411 = plt.subplot(gs4[1,1])

gs402 = mpl.gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs4[0,2], hspace=0.0)
gs412 = mpl.gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs4[1,2], hspace=0.0)

ax4020 = plt.subplot(gs402[0])
ax4021 = plt.subplot(gs402[1])

ax4120 = plt.subplot(gs412[0])
ax4121 = plt.subplot(gs412[1])

gamma_plot = np.copy(cgyro_data['gamma'])
gamma_plot[gamma_plot<0] = 0
gamma_plot = gamma_plot * np.sign(cgyro_data['omega'])

norm = colors.SymLogNorm(linthresh=0.2, linscale=0.2, vmin=-2.0, vmax=2.0)

ky_max = 12
ky_ion = 6

def shiftByHalf(x):
    y = np.zeros(len(x)+1)
    y[1:-1] = (x[1:] + x[:-1])/2
    y[0] = 2*x[0] - y[1]
    y[-1] = 2*x[-1] - y[-2]
    
    return y

ky0 = shiftByHalf(cgyro_data['ky'][0,:ky_max])
r0 = shiftByHalf(cgyro_data['radii'][0,:])
gmax_ion0 = np.max(cgyro_data['gamma'][0,:,:ky_ion], axis=1)
ax400.pcolormesh(r0, ky0, (gamma_plot[0,:,:ky_max]/gmax_ion0[:,np.newaxis]).T, norm=norm, cmap='PiYG')


ky1 = shiftByHalf(cgyro_data['ky'][1,:ky_max])
r1 = shiftByHalf(cgyro_data['radii'][1,:])
gmax_ion1 = np.max(cgyro_data['gamma'][1,:,:ky_ion], axis=1)
ax410.pcolormesh(r1, ky1, (gamma_plot[1,:,:ky_max]/gmax_ion1[:,np.newaxis]).T, norm=norm, cmap='PiYG')

# TODO: Figure out how to get the actual midplane radius?
def plotTransp(ax, transp, times, rbounds):
    time = transp.variables['TIME'].data
    roa = transp.variables['RMNMP'].data / 21.878
    
    shear = transp.variables['SREXB_NCL'].data    
    te = transp.variables['TE'].data

    t0, t1 = np.searchsorted(time, times)    
    r0, r1 = np.searchsorted(roa[t0-1,:], rbounds)
    cs = np.sqrt(te) * 692056.111 # c_s in cm/s
    
    ax.plot(roa[t0-1,r0:r1], shear[t0-1,r0:r1]/cs[t0-1,r0:r1]*21.878, c='r')
    ax.plot(roa[t1-1,r0:r1], shear[t1-1,r0:r1]/cs[t0-1,r0:r1]*21.878, c='b')


plotTransp(ax401, case1_transp, (0.95, 1.47), (r0[0], r0[-1]))
ax401.plot(cgyro_data['radii'][0,:], gmax_ion0, c='k', marker='.')
plotTransp(ax411, case2_transp, (1.25, 1.41), (r1[0], r1[-1]))
ax411.plot(cgyro_data['radii'][1,:], gmax_ion1, c='k', marker='.')

ax4020.plot(cgyro_data['ky'][0,:ky_max], cgyro_data['omega'][0,3,:ky_max], c='k', marker='+')
ax4021.plot(cgyro_data['ky'][0,:ky_max], cgyro_data['gamma'][0,3,:ky_max], c='k', marker='+')
ax4020.axhline(ls='--', c='k')
ax4021.axhline(ls='--', c='k')
plt.setp(ax4020.get_xticklabels(), visible=False)

ax4120.plot(cgyro_data['ky'][1,:ky_max], cgyro_data['omega'][1,2,:ky_max], c='k', marker='+')
ax4121.plot(cgyro_data['ky'][1,:ky_max], cgyro_data['gamma'][1,2,:ky_max], c='k', marker='+')
ax4120.axhline(ls='--', c='k')
ax4021.axhline(ls='--', c='k')
plt.setp(ax4120.get_xticklabels(), visible=False)

ax400.xaxis.set_major_locator(mpl.ticker.MultipleLocator(0.1))
ax410.xaxis.set_major_locator(mpl.ticker.MultipleLocator(0.1))
ax401.xaxis.set_major_locator(mpl.ticker.MultipleLocator(0.1))
ax411.xaxis.set_major_locator(mpl.ticker.MultipleLocator(0.1))

ax401.yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.1))
ax411.yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.1))

ax4020.yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.5))
ax4120.yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.5))
ax4021.yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.1))
ax4121.yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.1))

ax400.set_title(r'$k_y \rho_s$', loc='left')
ax401.set_title(r'$[c_s/a]$', loc='left')
ax4020.set_title(r'$[c_s/a]$', loc='left')

ax410.set_xlabel('r/a')
ax411.set_xlabel('r/a')
ax4121.set_xlabel(r'$k_y \rho_s$')

ax400.set_ylabel('Case I (0.8 MA Ohmic)')
ax410.set_ylabel('Case II (1.1 MA Ohmic)')


ax401.text(0.5, 0.27, r'$\gamma_{\mathrm{max}}$')
ax401.text(0.55, 0.07, r'$\gamma_{E}$')

ax411.text(0.55, 0.23, r'$\gamma_{\mathrm{max}}$')
ax411.text(0.6, 0.1, r'$\gamma_{E}$')

ax4020.text(0.05, 0.75, r'$\omega_R$', transform=ax4020.transAxes)
ax4021.text(0.05, 0.75, r'$\gamma$', transform=ax4021.transAxes)

ax4120.text(0.05, 0.75, r'$\omega_R$', transform=ax4120.transAxes)
ax4121.text(0.05, 0.75, r'$\gamma$', transform=ax4121.transAxes)

plt.tight_layout()
plt.tight_layout()

plt.savefig('fig_lin_cgyro.eps', format='eps', dpi=1200, facecolor='white')

# %% Figure 5: Growth rates / real frequencies.
# Note: The data here is generated from ~/hys2/plot_all_freqs.py

fig5 = plt.figure(5, figsize=(3.375*2, 3.375*1.2))
gs5 = mpl.gridspec.GridSpec(2, 3, wspace=0.0)

ax500 = plt.subplot(gs5[0,0])
ax510 = plt.subplot(gs5[1,0])
ax501 = plt.subplot(gs5[0,1])
ax511 = plt.subplot(gs5[1,1])
ax502 = plt.subplot(gs5[0,2])
ax512 = plt.subplot(gs5[1,2])

def plotQlWeight(ax, ky, pflux, qeflux, qiflux, gamma, alpha):
    gamma_cut = np.max(gamma[:6])*alpha
    subdominant = gamma < gamma_cut
    signchange = np.not_equal(np.roll(subdominant, 1), subdominant)
    
    ky_sub = ky[signchange]
    j = 0
    for i in range(signchange.shape[0]):
        if i == 0:
            j += 1
            continue
        elif signchange[i]:
            ky_sub[j] = (ky[i]-ky[i-1]) * np.abs((gamma_cut-gamma[i-1])/(gamma[i]-gamma[i-1])) + ky[i-1]
            j += 1
            
    for region in range(ky_sub.shape[0]/2):
        ax.axvspan(ky_sub[2*region], ky_sub[2*region+1], color=(0.75,0.75,0.75))
    
    ax.plot(ky, qiflux, c=mpl.cm.PiYG(0.0), marker='+', label='W,Qi')
    ax.plot(ky, qeflux, c=mpl.cm.PiYG(1.0), marker='+', label='W,Qe')
    ax.plot(ky, pflux, c=(0.5,0.0,1.0), marker='+', label=r'W,$\Gamma$e')
    ax.axhline(c='k', ls='--')
    ax.set_xscale('log')
    ax.set_xlim([0.08, 26.0])
    plt.setp(ax.get_yticklabels(), visible=False)
    ax.xaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
    
    flux_signchange = np.not_equal(np.sign(pflux[1:]), np.sign(pflux[:-1]))
    ky_fs = ky[:-1][flux_signchange]
    j = 0
    for i in range(flux_signchange.shape[0]):
        if flux_signchange[i]:
            ky_fs[j] = (ky[i+1]-ky[i]) * np.abs(pflux[i]/(pflux[i+1]-pflux[i])) + ky[i]
    ax.scatter(ky_fs[0], 0, marker='*', c=(1.0, 0.5, 0.0), s=120, zorder=20, linewidth=0)
    
plotQlWeight(ax500, cgyro_data['ky'][0,:], cgyro_data['pflux'][0,1,:], cgyro_data['qeflux'][0,1,:], cgyro_data['qiflux'][0,1,:], cgyro_data['gamma'][0,1,:], 0.4)
plotQlWeight(ax501, cgyro_data['ky'][0,:], cgyro_data['pflux'][0,3,:], cgyro_data['qeflux'][0,3,:], cgyro_data['qiflux'][0,3,:], cgyro_data['gamma'][0,3,:], 0.4)
plotQlWeight(ax502, cgyro_data['ky'][0,:], cgyro_data['pflux'][0,5,:], cgyro_data['qeflux'][0,5,:], cgyro_data['qiflux'][0,5,:], cgyro_data['gamma'][0,5,:], 0.4)

plotQlWeight(ax510, cgyro_data['ky'][1,:], cgyro_data['pflux'][1,1,:], cgyro_data['qeflux'][1,1,:], cgyro_data['qiflux'][1,1,:], cgyro_data['gamma'][1,1,:], 0.7)
plotQlWeight(ax511, cgyro_data['ky'][1,:], cgyro_data['pflux'][1,2,:], cgyro_data['qeflux'][1,2,:], cgyro_data['qiflux'][1,2,:], cgyro_data['gamma'][1,2,:], 0.7)
plotQlWeight(ax512, cgyro_data['ky'][1,:], cgyro_data['pflux'][1,3,:], cgyro_data['qeflux'][1,3,:], cgyro_data['qiflux'][1,3,:], cgyro_data['gamma'][1,3,:], 0.7)

plt.setp(ax500.get_yticklabels(), visible=True)
plt.setp(ax510.get_yticklabels(), visible=True)

ax500.set_ylabel('Case I (0.8 MA Ohmic)')
ax510.set_ylabel('Case II (1.1 MA Ohmic)')

l = ax512.legend(loc='center right', fontsize=10)
l.set_zorder(100)

ax500.set_title('r/a=0.5', fontsize=10)
ax501.set_title('r/a=0.6', fontsize=10)
ax502.set_title('r/a=0.7', fontsize=10)
ax510.set_title('r/a=0.55', fontsize=10)
ax511.set_title('r/a=0.6', fontsize=10)
ax512.set_title('r/a=0.65', fontsize=10)



plt.tight_layout()
plt.tight_layout()

plt.savefig('fig_ql_cgyro.eps', format='eps', dpi=1200, facecolor='white')

# %% Figure 6: QL weights - need to go on engaging, also what to do with flux data?

"""
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

#plt.savefig('figure6.eps', format='eps', dpi=1200, facecolor='white')
"""

# %% Figure 7 on vgr vs. vph


fig7 = plt.figure(7, figsize=(3.375, 3.375*0.7))
gs7 = mpl.gridspec.GridSpec(2, 1, hspace=0.0)
ax7 = map(plt.subplot, gs7)

def plotWaveVelocities(ky, omega, ax):
    elec = omega>0
    ion = omega<0
    vph = omega/ky
    
    vgr_e = np.diff(omega[elec])/np.diff(ky[elec])
    vgr_i = np.diff(omega[ion])/np.diff(ky[ion])
    ky_e = ky[elec]
    ky_i = ky[ion]
    ky_e = (ky_e[1:] + ky_e[:-1])/2
    ky_i = (ky_i[1:] + ky_i[:-1])/2
    
    ax.plot(ky[elec], vph[elec], c=mpl.cm.PiYG(1.0), linestyle='-', label='$v_{ph}$')
    ax.plot(ky_e, vgr_e, c=mpl.cm.PiYG(1.0), linestyle=':', label='$v_{gr}$')
    
    ax.plot(ky[ion], vph[ion], c=mpl.cm.PiYG(0.0), linestyle='-')
    ax.plot(ky_i, vgr_i, c=mpl.cm.PiYG(0.0), linestyle=':')
    
    ax.set_xscale('log')
    ax.set_xlim([0.1, 30.0])
    ax.set_ylim([-0.8, 0.8])
    

ky = cgyro_data['ky']
omega1 = cgyro_data['omega'][1,2,:]

plotWaveVelocities(ky[0,:], cgyro_data['omega'][0,3,:], ax7[0])
plotWaveVelocities(ky[1,:], cgyro_data['omega'][1,2,:], ax7[1])

plt.setp(ax7[0].get_xticklabels(), visible=False)
ax7[1].set_xlabel(r'$k_y \rho_s$')
ax7[1].legend(handletextpad=0.1, borderpad=0.1, labelspacing=0.0, loc='lower right')

ax7[0].yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.3))
ax7[1].yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.3))

ax7[0].set_ylabel(r'$[c_s \rho_s / a]$')
ax7[1].set_ylabel(r'$[c_s \rho_s / a]$')

ax7[0].text(0.05, 0.75, 'Case I', transform=ax7[0].transAxes)
ax7[1].text(0.05, 0.75, 'Case II', transform=ax7[1].transAxes)

ax7[1].text(0.15, 0.30, 'ITG', transform=ax7[1].transAxes, color=mpl.cm.PiYG(0.0))
ax7[1].text(0.40, 0.6, 'TEM/ETG', transform=ax7[1].transAxes, color=mpl.cm.PiYG(1.0))

plt.tight_layout()
plt.tight_layout()