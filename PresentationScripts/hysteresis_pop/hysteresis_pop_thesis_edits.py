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

from mpl_toolkits.axes_grid1.inset_locator import inset_axes

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

elecTree = MDSplus.Tree('electrons', 1160506009)
specTree = MDSplus.Tree('spectroscopy', 1160506009)

nl04time, nl04data = trimNodeData(elecTree.getNode(r'\ELECTRONS::TOP.TCI.RESULTS:NL_04'))
vnode = specTree.getNode(r'\SPECTROSCOPY::TOP.HIREX_SR.ANALYSIS.W:VEL')
vtime, vdata = trimData(vnode.dim_of().data(), vnode.data()[0])

td = ThacoData(None, 1120216017, 9)

# %% Figure 1: time traces and rotation profiles

fig1 = plt.figure(1, figsize=(3.375, 3.375*1.2))
gs1 = mpl.gridspec.GridSpec(2, 1, height_ratios=[2,1])
gs1_inner = mpl.gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs1[0], hspace=0.0)

ax10 = plt.subplot(gs1_inner[0])
ax10.axvspan(0.69, 0.75, color=(0.5,0.5,1.0))
ax10.axvspan(0.89, 0.95, color=(1.0,0.5,0.5))
ax10.plot(nl04time, nl04data/1e20/0.6, c='k')
ax10.yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.1))
ax10.set_ylim([1.05,1.55])
ax10.set_ylabel(r'$\bar{n}_e$ ($10^{20} \mathrm{m}^{-3}$)')
plt.setp(ax10.get_xticklabels(), visible=False)

offset = ((7.0)-4.0)/(25.0-7.0)*6

ax11 = plt.subplot(gs1_inner[1], sharex=ax10)
ax11.axvspan(0.69, 0.75, color=(0.5,0.5,1.0))
ax11.axvspan(0.89, 0.95, color=(1.0,0.5,0.5))
ax11.plot(vtime, vdata+offset, c='k', marker='.')
ax11.axhline(c='k', ls='--')
ax11.yaxis.set_major_locator(mpl.ticker.MultipleLocator(5))
ax11.set_ylim([-14,29])
ax11.set_ylabel('$v_{tor}$ (km/s)')
ax11.set_xlabel('time (sec)')
ax11.set_xlim([0.5, 1.5])

tloc, tsoc = np.searchsorted(td.time, (1.21, 1.43))
rmax = np.searchsorted(td.roa, 0.9)
vloc = np.average(td.pro[1,tloc-1:tloc+1,:],axis=0)
verrloc = np.average(td.perr[1,tloc-1:tloc+1,:],axis=0)
vsoc = np.average(td.pro[1,tsoc-1:tsoc+1,:],axis=0)
verrsoc = np.average(td.perr[1,tsoc-1:tsoc+1,:],axis=0)

ax12 = plt.subplot(gs1[1])
#ax12.axvspan(0.35, 0.65, color=(0.7, 1.0, 1.0))
ax12.errorbar(td.roa[:rmax], vloc[:rmax]*2*np.pi, verrloc[:rmax]*2*np.pi, c='r')
ax12.errorbar(td.roa[:rmax], vsoc[:rmax]*2*np.pi, verrsoc[:rmax]*2*np.pi, c='b')
#ax12.errorbar(td.roa[:], vloc[:], verrloc[:], c='r')
#ax12.errorbar(td.roa[:], vsoc[:], verrsoc[:], c='b')
ax12.axhline(c='k', ls='--')
ax12.xaxis.set_major_locator(mpl.ticker.MultipleLocator(0.2))
#ax12.yaxis.set_major_locator(mpl.ticker.MultipleLocator(4))
ax12.set_ylabel('$\omega_{tor}$ (kRad/s)')
ax12.set_xlabel('r/a')
ax12.set_xlim([0.0, 1.0])

plt.tight_layout(h_pad=1.6)
plt.tight_layout(h_pad=1.6)

plt.savefig('fig_timetraces.eps', format='eps', dpi=1200)

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
plotHysteresis(1160506008, ax20, c='m')
plotHysteresis(1160506024, ax20, c='c')

plotHysteresis(1160506009, ax21, c='k')
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

gs3_outer = mpl.gridspec.GridSpec(3, 1, hspace=0.08)
gs3_inners = map(lambda gs: mpl.gridspec.GridSpecFromSubplotSpec(2, 3, subplot_spec=gs, hspace=0.0), gs3_outer)
ax3 = np.array(map(lambda gs: np.array(map(plt.subplot, gs)).reshape((2,3)), gs3_inners))

#ax3 = np.array(map(plt.subplot, gs3)).reshape((fig3_rows, fig3_cols))

#ax3 = np.array([[plt.subplot(gs3[i,j]) for j in range(fig3_cols)] for i in range(fig3_rows)])

# Comes from mcRaytracing.py
itemp = np.array([ 0.58171643,  0.58086416,  0.58097034,  0.58196616,  0.58472968,
        0.58757312,  0.5930036 ,  0.59770901,  0.60584412,  0.61244813,
        0.62335314,  0.6319093 ,  0.60002627,  0.59638661,  0.59233809,
        0.59035054,  0.58865633,  0.58829467,  0.58894016,  0.59020199,
        0.59318791,  0.59607954,  0.60142963,  0.60597674,  0.61373905,
        0.61998311,  0.63022579,  0.63822313])

def plot_ion(shot, roa, iondata, axti, axvtor, color='b'):
    #offset = ((shot%100)-4.0)/(25.0-7.0)*6
    offset = -2

    meas_avg = iondata['meas_avg']
    meas_avg[:,4] = np.nan

    #specTree = MDSplus.Tree('spectroscopy', shot)
    #pos = specTree.getNode(r'\SPECTROSCOPY::TOP.HIREXSR.ANALYSIS6.HELIKE.MOMENTS.W:POS').data()

    #nch = iondata['meas_avg'].shape[1]
    #ch = list(reversed(range(nch)))

    axti.errorbar(roa, meas_avg[2,:]-itemp, yerr=iondata['meas_std'][2,:], c=color, fmt='.')
    axvtor.errorbar(roa, (meas_avg[1,:]+offset)*4, yerr=iondata['meas_std'][1,:], c=color, fmt='.')

    plt.setp(axti.get_xticklabels(), visible=False)
    plt.setp(axvtor.get_xticklabels(), visible=False)

    axti.set_xlim([0.0, 1.0])
    axvtor.set_xlim([0.0, 1.0])

    #axti.xaxis.set_major_locator(mpl.ticker.MultipleLocator(4))
    #axvtor.xaxis.set_major_locator(mpl.ticker.MultipleLocator(4))


    #axti.axvline(15.5, c='k')
    #axvtor.axvline(15.5, c='k')


def plot_ion_pair(shot, ax_base, t_loc, t_soc):
    socdata = np.load('/home/normandy/git/bsfc/bsfc_fits/fit_data/mf_%d_nh3_t0.npz'%shot)
    locdata = np.load('/home/normandy/git/bsfc/bsfc_fits/fit_data/mf_%d_nh3_t2.npz'%shot)
    
    momNode = specTree.getNode(r'\SPECTROSCOPY::TOP.HIREXSR.ANALYSIS.HELIKE.MOMENTS.W:MOM')
    psin_raw = momNode.dim_of(0).data()
    time = momNode.dim_of(1).data()
    time = time[time>0]
    
    
    tind_loc, tind_soc = np.searchsorted(time, (t_loc, t_soc))

    psin_loc = psin_raw[tind_loc,:locdata['meas_avg'].shape[1]]
    psin_soc = psin_raw[tind_soc,:locdata['meas_avg'].shape[1]]
    e = eqtools.CModEFITTree(shot)
    
    roa_loc = e.psinorm2roa(psin_loc, t_loc)
    roa_soc = e.psinorm2roa(psin_soc, t_soc)
    
    plot_ion(shot, roa_loc, locdata, ax_base[0,2], ax_base[1,2], 'r')
    plot_ion(shot, roa_soc, socdata, ax_base[0,2], ax_base[1,2], 'b')


plot_ion_pair(1160506007, ax3[0,:,:], 0.96, 0.6)
plot_ion_pair(1160506009, ax3[1,:,:], 0.92, 0.72)
plot_ion_pair(1160506015, ax3[2,:,:], 0.9, 0.68)


def plot_profs(shot, folder, axp, axd, color='b', data='ne'):
    prof = pkl.load(file('/home/normandy/git/psfc-misc/Fitting/FitsPoP2019/%s/%s_dict_fit_%d.pkl'%(folder, data, shot)))

    x_max = np.searchsorted(prof['X'], 0.99)
    x_max_d = np.searchsorted(prof['X'], 0.9)

    axp.errorbar(prof['X'][:x_max:4], prof['y'][:x_max:4], yerr=prof['err_y'][:x_max:4], c=color, linewidth=0.8)
    axd.errorbar(prof['X'][:x_max_d:4], prof['a_Ly'][:x_max_d:4], yerr=prof['err_a_Ly'][:x_max_d:4], c=color, linewidth=0.8)

    axp.errorbar(prof['data_x'], prof['data_y'], yerr=prof['data_err_y'], c=color, fmt='.')

    axp.set_xlim([0.0, 1.0])
    axd.set_xlim([0.0, 1.0])

    plt.setp(axp.get_xticklabels(), visible=False)
    plt.setp(axd.get_xticklabels(), visible=False)


def plot_pair(shot, folder_base, ax_base):
    plot_profs(shot, folder_base+'_loc', ax_base[0,0], ax_base[1,0], 'r', 'ne')
    plot_profs(shot, folder_base+'_soc', ax_base[0,0], ax_base[1,0], 'b', 'ne')

    plot_profs(shot, folder_base+'_loc', ax_base[0,1], ax_base[1,1], 'r', 'te')
    plot_profs(shot, folder_base+'_soc', ax_base[0,1], ax_base[1,1], 'b', 'te')

plot_pair(1160506007, '007', ax3[0,:,:])
plot_pair(1160506009, '009', ax3[1,:,:])
plot_pair(1160506015, '015', ax3[2,:,:])

plt.setp(ax3[2,1,0].get_xticklabels(), visible=True)
plt.setp(ax3[2,1,1].get_xticklabels(), visible=True)
ax3[2,1,0].set_xlabel('r/a')
ax3[2,1,1].set_xlabel('r/a')

plt.setp(ax3[2,1,2].get_xticklabels(), visible=True)
ax3[2,1,2].set_xlabel('Spatial Channel')

ax3[0,0,2].set_ylim([-0.1, 1.4])
ax3[0,1,2].set_ylim([-20,14])
ax3[1,0,2].set_ylim([-0.1, 1.4])
ax3[1,1,2].set_ylim([-10,34])
ax3[2,0,2].set_ylim([-0.25, 2.0])
ax3[2,1,2].set_ylim([-20,44])
ax3[0,1,2].yaxis.set_major_locator(mpl.ticker.MultipleLocator(5))
ax3[1,1,2].yaxis.set_major_locator(mpl.ticker.MultipleLocator(10))
ax3[2,1,2].yaxis.set_major_locator(mpl.ticker.MultipleLocator(10))

ax3[0,0,1].set_ylim([0.0, 2.0])
ax3[0,1,1].set_ylim([-0.5, 7.5])
ax3[1,0,1].set_ylim([0.0, 2.0])
ax3[1,1,1].set_ylim([-0.5, 7.5])
ax3[2,0,1].set_ylim([0.0, 2.5])
ax3[2,1,1].set_ylim([-0.5, 9.5])
ax3[0,1,1].yaxis.set_major_locator(mpl.ticker.MultipleLocator(2))
ax3[1,1,1].yaxis.set_major_locator(mpl.ticker.MultipleLocator(2))

ax3[0,0,0].set_ylim([0.0, 1.5])
ax3[0,1,0].set_ylim([-0.25, 1.8])
ax3[1,0,0].set_ylim([0.0, 1.8])
ax3[1,1,0].set_ylim([-0.25, 1.8])
ax3[2,0,0].set_ylim([0.0, 1.8])
ax3[2,1,0].set_ylim([-0.25, 1.8])
ax3[0,0,0].yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.3))
ax3[1,0,0].yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.3))
ax3[2,0,0].yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.3))

for i in range(3):
    ax3[i,0,0].text(0.08, 0.2, r'$n_e$ [$10^{20}$ m$^{-3}$]', transform=ax3[i,0,0].transAxes, va='center', ha='left')
    ax3[i,1,0].text(0.08, 0.7, r'$a/L_{ne}$', transform=ax3[i,1,0].transAxes, va='center', ha='left')

    ax3[i,0,1].text(0.08, 0.2, r'$T_e$ [keV]', transform=ax3[i,0,1].transAxes, va='center', ha='left')
    ax3[i,1,1].text(0.08, 0.7, r'$a/L_{Te}$', transform=ax3[i,1,1].transAxes, va='center', ha='left')

    ax3[i,0,2].text(0.08, 0.5, r'$\overline{T}_i$ [keV]', transform=ax3[i,0,2].transAxes, va='center', ha='left')
    ax3[i,1,2].text(0.08, 0.5, r'$\bar{v}_{tor}$ [km/s]', transform=ax3[i,1,2].transAxes, va='center', ha='left')

ax3[0,0,0].annotate(r'Case I (0.8 MA Ohmic)', xy=(-0.25,0.0), ha='right', va='center', xycoords='axes fraction', rotation='90')
ax3[1,0,0].annotate(r'Case II (1.1 MA Ohmic)', xy=(-0.25,0.0), ha='right', va='center', xycoords='axes fraction', rotation='90')
ax3[2,0,0].annotate(r'Case III (0.8 MA +ICRF)', xy=(-0.25,0.0), ha='right', va='center', xycoords='axes fraction', rotation='90')
#ax3[0,0,0].annotate(-0.05, 0.0, )

plt.savefig('fig_profiles.eps', format='eps', dpi=1200, facecolor='white')

# %% Figure 4: Plots of growth rates

cgyro_data = np.load(file('/home/normandy/git/psfc-misc/PresentationScripts/hysteresis_pop/1120216_cgyro_data.npz'))

case1_transp = scipy.io.netcdf.netcdf_file('/home/normandy/hysteresis_transp/12345B05/12345B05.CDF')
case2_transp = scipy.io.netcdf.netcdf_file('/home/normandy/hysteresis_transp/12345A05/12345A05.CDF')


# %% Do actual plotting here

fig4 = plt.figure(4, figsize=(3.375*2, 3.375*1.4))
gs4_outer = mpl.gridspec.GridSpec(2, 1, height_ratios=[40,1])
gs4 = mpl.gridspec.GridSpecFromSubplotSpec(2, 3, subplot_spec=gs4_outer[0])

gs4_bot = mpl.gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs4_outer[1])
ax4_cax = plt.subplot(gs4_bot[0])

#gs4 = mpl.gridspec.GridSpec(2, 3)

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

norm = colors.Normalize(vmin=-1.5, vmax=1.5)

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
pc = ax400.pcolormesh(r0, ky0, (gamma_plot[0,:,:ky_max]/gmax_ion0[:,np.newaxis]).T, norm=norm, cmap='PiYG')


ky1 = shiftByHalf(cgyro_data['ky'][1,:ky_max])
r1 = shiftByHalf(cgyro_data['radii'][1,:])
gmax_ion1 = np.max(cgyro_data['gamma'][1,:,:ky_ion], axis=1)
ax410.pcolormesh(r1, ky1, (gamma_plot[1,:,:ky_max]/gmax_ion1[:,np.newaxis]).T, norm=norm, cmap='PiYG')

cb = plt.colorbar(pc, cax=ax4_cax, orientation='horizontal', label=r'$(\gamma / \gamma_{\mathrm{max,ion}}) \mathrm{sign}(\omega_R)$', ticks=[-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5])
cb.ax.set_xticklabels(['<-1.5','-1.0','-0.5','0.0','0.5','1.0','>1.5'])

def plotTranspShear(ax, transp, times, rbounds, ax_gamma, gamma_roa):
    time = transp.variables['TIME'].data
    roa = transp.variables['RMNMP'].data / 21.878

    shear = transp.variables['SREXB_NCL'].data
    te = transp.variables['TE'].data

    t0, t1 = np.searchsorted(time, times)
    r0, r1 = np.searchsorted(roa[t0-1,:], rbounds)
    cs = np.sqrt(te) * 692056.111 # c_s in cm/s

    ax.plot(roa[t0-1,r0:r1], shear[t0-1,r0:r1]/cs[t0-1,r0:r1]*21.878, c='r')
    ax.plot(roa[t1-1,r0:r1], shear[t1-1,r0:r1]/cs[t0-1,r0:r1]*21.878, c='b')

    rind = rind = np.searchsorted(roa[t0-1,:], gamma_roa)
    ax_gamma.axhline(shear[t0-1,rind]/cs[t0-1,rind]*21.878, c='r')
    ax_gamma.axhline(shear[t1-1,rind]/cs[t0-1,rind]*21.878, c='b')


plotTranspShear(ax401, case1_transp, (0.95, 1.47), (r0[0], r0[-1]), ax4021, 0.55)
ax401.plot(cgyro_data['radii'][0,:], gmax_ion0, c='k', marker='.')
plotTranspShear(ax411, case2_transp, (1.21, 1.41), (r1[0], r1[-1]), ax4121, 0.6)
ax411.plot(cgyro_data['radii'][1,:], gmax_ion1, c='k', marker='.')

ax4020.plot(cgyro_data['ky'][0,:ky_max], cgyro_data['omega'][0,2,:ky_max], c='k', marker='.')
ax4021.plot(cgyro_data['ky'][0,:ky_max], cgyro_data['gamma'][0,2,:ky_max], c='k', marker='.')
ax4020.axhline(ls='--', c='k')
ax4021.axhline(ls='--', c='k')
plt.setp(ax4020.get_xticklabels(), visible=False)

ax4120.plot(cgyro_data['ky'][1,:ky_max], cgyro_data['omega'][1,2,:ky_max], c='k', marker='.')
ax4121.plot(cgyro_data['ky'][1,:ky_max], cgyro_data['gamma'][1,2,:ky_max], c='k', marker='.')
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

#ax4021.set_ylim([0.0, 0.35])
#ax4121.set_ylim([0.0, 0.35])

#ax400.set_ylabel(r'$k_y \rho_s$')
#ax401.set_ylabel(r'$[c_s/a]$')
#ax4020.set_ylabel(r'$[c_s/a]$')

ax400.annotate(r'$k_y \rho_s$', xy=(0,1.03), ha='right', va='bottom', xycoords='axes fraction')
ax401.annotate(r'$[c_s/a]$', xy=(0,1.03), ha='right', va='bottom', xycoords='axes fraction')
ax4020.annotate(r'$[c_s/a]$', xy=(0,1.06), ha='right', va='bottom', xycoords='axes fraction')


ax410.set_xlabel('r/a')
ax411.set_xlabel('r/a')
ax4121.set_xlabel(r'$k_y \rho_s$')

ax400.set_title('Elec./Ion Dominance', fontsize=10)
ax401.set_title(r'Growth vs. Shear', fontsize=10)
ax4020.set_title('Eigenspectrum', fontsize=10)

ax400.set_ylabel('(0.8 MA Ohmic)')
ax410.set_ylabel('(1.1 MA Ohmic)')


ax401.text(0.5, 0.27, r'$\gamma_{\mathrm{max}}$')
ax401.text(0.55, 0.07, r'$\gamma_{E}$')

ax411.text(0.55, 0.23, r'$\gamma_{\mathrm{max}}$')
ax411.text(0.6, 0.1, r'$\gamma_{E}$')

ax4020.text(0.05, 0.75, r'$\omega_R$ ($r/a=0.55$)', transform=ax4020.transAxes)
ax4021.text(0.05, 0.75, r'$\gamma$', transform=ax4021.transAxes)
ax4021.text(0.85, 0.28, r'$\gamma_E$', transform=ax4021.transAxes)

ax4120.text(0.05, 0.75, r'$\omega_R$ ($r/a=0.6$)', transform=ax4120.transAxes)
ax4121.text(0.05, 0.75, r'$\gamma$', transform=ax4121.transAxes)
ax4121.text(0.85, 0.28, r'$\gamma_E$', transform=ax4121.transAxes)


plt.tight_layout()
plt.tight_layout()

plt.savefig('fig_lin_cgyro.eps', format='eps', dpi=1200, facecolor='white')

# %% Plots of quasilinear weights
# Note: The data here is generated from ~/hys2/plot_all_freqs.py

fig5 = plt.figure(5, figsize=(3.375*2, 3.375*1.2))
gs5_outer = mpl.gridspec.GridSpec(1,2, width_ratios=[12,3])
gs5_left = mpl.gridspec.GridSpecFromSubplotSpec(2, 3, subplot_spec=gs5_outer[0], wspace=0.0)
gs5_right = mpl.gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs5_outer[1])

ax5 = np.array(map(plt.subplot, gs5_left)).reshape(2,3)
ax5_right = np.array(map(plt.subplot, gs5_right))

#gs5 = mpl.gridspec.GridSpec(2, 3, wspace=0.0)
#ax5 = np.array(map(plt.subplot, gs5)).reshape(2,3)

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
    ax.set_ylim([-0.5, 3.5])
    plt.setp(ax.get_yticklabels(), visible=False)
    ax.xaxis.set_major_formatter(mpl.ticker.ScalarFormatter())

    flux_signchange = np.not_equal(np.sign(pflux[1:]), np.sign(pflux[:-1]))
    ky_fs = ky[:-1][flux_signchange]
    j = 0
    for i in range(flux_signchange.shape[0]):
        if flux_signchange[i]:
            ky_fs[j] = (ky[i+1]-ky[i]) * np.abs(pflux[i]/(pflux[i+1]-pflux[i])) + ky[i]
    #ax.scatter(ky_fs[0], 0, marker='*', c=(1.0, 0.5, 0.0), s=120, zorder=20, linewidth=0)

plotQlWeight(ax5[0,0], cgyro_data['ky'][0,:], cgyro_data['pflux'][0,1,:], cgyro_data['qeflux'][0,1,:], cgyro_data['qiflux'][0,1,:], cgyro_data['gamma'][0,1,:], 0.4)
plotQlWeight(ax5[0,1], cgyro_data['ky'][0,:], cgyro_data['pflux'][0,3,:], cgyro_data['qeflux'][0,3,:], cgyro_data['qiflux'][0,3,:], cgyro_data['gamma'][0,3,:], 0.4)
plotQlWeight(ax5[0,2], cgyro_data['ky'][0,:], cgyro_data['pflux'][0,5,:], cgyro_data['qeflux'][0,5,:], cgyro_data['qiflux'][0,5,:], cgyro_data['gamma'][0,5,:], 0.4)

plotQlWeight(ax5[1,0], cgyro_data['ky'][1,:], cgyro_data['pflux'][1,1,:], cgyro_data['qeflux'][1,1,:], cgyro_data['qiflux'][1,1,:], cgyro_data['gamma'][1,1,:], 0.7)
plotQlWeight(ax5[1,1], cgyro_data['ky'][1,:], cgyro_data['pflux'][1,2,:], cgyro_data['qeflux'][1,2,:], cgyro_data['qiflux'][1,2,:], cgyro_data['gamma'][1,2,:], 0.7)
plotQlWeight(ax5[1,2], cgyro_data['ky'][1,:], cgyro_data['pflux'][1,3,:], cgyro_data['qeflux'][1,3,:], cgyro_data['qiflux'][1,3,:], cgyro_data['gamma'][1,3,:], 0.7)

plt.setp(ax5[0,0].get_yticklabels(), visible=True)
plt.setp(ax5[1,0].get_yticklabels(), visible=True)

ax5[0,0].set_ylabel('(0.8 MA Ohmic)')
ax5[1,0].set_ylabel('(1.1 MA Ohmic)')

l = ax5[1,2].legend(loc='center right', fontsize=8)
l.set_zorder(100)

ax5[0,0].set_title('r/a=0.5', fontsize=10)
ax5[0,1].set_title('r/a=0.6', fontsize=10)
ax5[0,2].set_title('r/a=0.7', fontsize=10)
ax5[1,0].set_title('r/a=0.55', fontsize=10)
ax5[1,1].set_title('r/a=0.6', fontsize=10)
ax5[1,2].set_title('r/a=0.65', fontsize=10)

plt.setp(ax5[0,0].get_xticklabels(), visible=False)
plt.setp(ax5[0,1].get_xticklabels(), visible=False)
plt.setp(ax5[0,2].get_xticklabels(), visible=False)

ax5[1,1].set_xlabel(r'$k_y \rho_s$')

def plot_transp(filename, ax):
    fluxes = np.load(filename)

    ax.plot(fluxes['roa'], fluxes['qi_anom'], marker='.', c=mpl.cm.PiYG(0.0), label='Qi')
    ax.plot(fluxes['roa'], fluxes['qe_anom'], marker='.', c=mpl.cm.PiYG(1.0), label='Qe')
    ax.plot(fluxes['roa'], fluxes['fe_anom'], marker='.', c=(0.5,0.0,1.0), label=r'$\Gamma$e')
    ax.axhline(ls='--', c='k')

plot_transp('fluxes_030.npz', ax5_right[0])
plot_transp('fluxes_017.npz', ax5_right[1])

ax5_right[1].set_xlabel('r/a')
ax5_right[1].legend(loc='upper left', fontsize=8)
ax5_right[0].xaxis.set_major_locator(mpl.ticker.MultipleLocator(0.1))
ax5_right[1].xaxis.set_major_locator(mpl.ticker.MultipleLocator(0.1))
ax5_right[0].set_title('Anom. Fluxes', fontsize=10)

ax5[0,0].annotate(r'[GB]', xy=(0,1.03), ha='right', va='bottom', xycoords='axes fraction')
ax5_right[0].annotate(r'[GB]', xy=(0,1.03), ha='right', va='bottom', xycoords='axes fraction')

plt.tight_layout()
plt.tight_layout()

plt.savefig('fig_ql_cgyro.eps', format='eps', dpi=1200, facecolor='white')

# %% Figure 7 on vgr vs. vph


fig7 = plt.figure(7, figsize=(3.375, 3.375*0.7))
gs7 = mpl.gridspec.GridSpec(2, 1, hspace=0.00)
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

ax7[0].text(0.02, 0.8, '(0.8 MA Ohmic)', transform=ax7[0].transAxes)
ax7[1].text(0.02, 0.8, '(1.1 MA Ohmic)', transform=ax7[1].transAxes)

ax7[1].text(0.15, 0.30, 'ITG', transform=ax7[1].transAxes, color=mpl.cm.PiYG(0.0))
ax7[1].text(0.40, 0.6, 'TEM/ETG', transform=ax7[1].transAxes, color=mpl.cm.PiYG(1.0))

plt.tight_layout()
plt.tight_layout()


plt.savefig('fig_wavevel.eps', format='eps', dpi=1200, facecolor='white')

# %% Figure 8, PCI plots

fig8 = plt.figure(8, figsize=(3.375*2, 3.375*1.5))
gs8_outer = mpl.gridspec.GridSpec(2, 2, height_ratios=[50,1], width_ratios=[2,2])
gs8 = mpl.gridspec.GridSpecFromSubplotSpec(3, 2, subplot_spec=gs8_outer[0,0], wspace=0.0, hspace=0.08)
#gs8_right = mpl.gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs8_outer[1], wspace=0.0)
ax8_cax = plt.subplot(gs8_outer[1,0])

gs8_right = mpl.gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs8_outer[0,1], hspace=0.08)

#gs8 = mpl.gridspec.GridSpec(4, 2, wspace=0.0, height_ratios=[10,10,10,1])
ax8 = np.array(map(plt.subplot, gs8)).reshape((3,2))
ax8_right = np.array(map(plt.subplot, gs8_right))
#ax8_cax = plt.subplot(gs8[3,:])

ax8_last = plt.subplot(gs8_outer[1,1])


def plotPci(shot, ax_loc, ax_soc, t_loc, t_soc):
    idlsav = scipy.io.readsav('/home/normandy/%d_pci.sav'%shot)

    t = idlsav.spec.t[0]
    f = idlsav.spec.f[0]
    k = idlsav.spec.k[0]
    s = idlsav.spec.spec[0]
    #shot = idlsav.spec.shot[0]

    t0, t1 = np.searchsorted(t, (t_soc, t_loc))

    kplot = shiftByHalf(k)
    #kplot = np.concatenate((k, [-k[0]]))
    fplot = shiftByHalf(f)

    ax_soc.pcolormesh(kplot, fplot, s[t0,:,:], cmap='cubehelix', norm=mpl.colors.LogNorm(vmin=1e-6, vmax=1e-1), rasterized=True)
    pc = ax_loc.pcolormesh(kplot, fplot, s[t1,:,:], cmap='cubehelix', norm=mpl.colors.LogNorm(vmin=1e-6, vmax=1e-1), rasterized=True)

    plt.setp(ax_loc.get_yticklabels(), visible=False)

    plt.setp(ax_loc.get_xticklabels(), visible=False)
    plt.setp(ax_soc.get_xticklabels(), visible=False)
    ax_soc.xaxis.set_major_locator(mpl.ticker.MultipleLocator(10))
    ax_loc.xaxis.set_major_locator(mpl.ticker.MultipleLocator(10))
    ax_loc.yaxis.set_major_locator(mpl.ticker.MultipleLocator(200))
    ax_soc.yaxis.set_major_locator(mpl.ticker.MultipleLocator(200))

    plt.setp(ax_soc.get_yticklabels(), rotation=45)

    #plt.colorbar(pc)

    ax_soc.set_xlim([-25, 25])
    ax_loc.set_xlim([-25, 25])
    ax_soc.set_ylim([0, 750])
    ax_loc.set_ylim([0, 750])

    return pc

plotPci(1160506007, ax8[0,1], ax8[0,0], 0.96, 0.6)
plotPci(1160506009, ax8[1,1], ax8[1,0], 0.92, 0.72)
pc = plotPci(1160506015, ax8[2,1], ax8[2,0], 0.9, 0.68)

plt.setp(ax8[2,0].get_xticklabels(), visible=True)
plt.setp(ax8[2,1].get_xticklabels(), visible=True)

ax8[0,0].annotate(r'f [kHz]', xy=(0.02,1.02), ha='right', va='bottom', xycoords='axes fraction')



ax8[2,0].set_xlabel(r'$k_R\ [\mathrm{cm}^{-1}]$')
ax8[2,1].set_xlabel(r'$k_R\ [\mathrm{cm}^{-1}]$')

ax8[0,0].set_title('SOC', color='b')
ax8[0,1].set_title('LOC', color='r')

#plt.text(0.0, 0.5, '0.8 MA Ohmic', horizontalalignment='center', verticalalignment='center', transform=ax8[0,0].transAxes, bbox=dict(facecolor='white'), rotation='90')
#plt.text(0.0, 0.5, '1.1 MA Ohmic', horizontalalignment='center', verticalalignment='center', transform=ax8[1,0].transAxes, bbox=dict(facecolor='white'), rotation='90')
#plt.text(0.0, 0.5, '0.8 MA +ICRF', horizontalalignment='center', verticalalignment='center', transform=ax8[2,0].transAxes, bbox=dict(facecolor='white'), rotation='90')

ax8[0,0].set_ylabel('(0.8 MA Ohmic)')
ax8[1,0].set_ylabel('(1.1 MA Ohmic)')
ax8[2,0].set_ylabel('(0.8 MA +ICRF)')


cb = plt.colorbar(pc, cax=ax8_cax, use_gridspec=True, orientation='horizontal')
cb.set_label(r'Intensity [($10^{18}$ m$^{-2}$)$^2$/cm$^{-1}$/kHz]')

plt.tight_layout()
plt.tight_layout()


def plotPciAsym(shot, ax):
    idlsav = scipy.io.readsav('/home/normandy/%d_pci.sav'%shot)

    t = idlsav.spec.t[0]
    f = idlsav.spec.f[0]
    k = idlsav.spec.k[0]
    s = idlsav.spec.spec[0]

    fc = np.searchsorted(f, (50, 200, 200, 700))

    #sn = s[:,:,1:16]
    #sp = s[:,:,17:32]

    #pn = np.sum(sn, axis=2)
    #pp = np.sum(sp, axis=2)

    #ax.pcolormesh(t, f, pp.T, norm=mpl.colors.LogNorm(vmin=1e-5, vmax=1.0), cmap='cubehelix')
    #ax.pcolormesh(t, -f, pn.T, norm=mpl.colors.LogNorm(vmin=1e-5, vmax=1.0), cmap='cubehelix')



    #kp = np.sum(s[:,:,18:24], axis=2)
    #kn = np.sum(s[:,:,10:16], axis=2)
    kp = np.sum(s[:,:,17:32], axis=2)
    kn = np.sum(s[:,:,1:16], axis=2)

    #kall = np.concatenate((kn[:,299:29:-1], kp[:,30:300]), axis=1)
    #fall = np.concatenate((-f[299:29:-1], f[30:300]))

    totalp = np.sum(kp[:,fc[0]:fc[1]], axis=1)
    totaln = np.sum(kn[:,fc[0]:fc[1]], axis=1)
    asym = (totalp-totaln)/(totalp+totaln)*2
    ax.plot(t, asym, c=mpl.cm.PiYG(0.0))

    totalp = np.sum(kp[:,fc[2]:fc[3]], axis=1)
    totaln = np.sum(kn[:,fc[2]:fc[3]], axis=1)
    asym = (totalp-totaln)/(totalp+totaln)*2
    ax.plot(t, asym, c=mpl.cm.PiYG(1.0))

    ax.axhline(c='k', ls='--')
    ax.set_ylim([-0.7, 0.7])

    plt.setp(ax.get_xticklabels(), visible=False)

ax8_right[0].clear()

ax8_right[0].axvspan(0.5, 0.75, color=(0.8, 0.8, 1.0))
ax8_right[0].axvspan(0.84, 1.15, color=(1.0, 0.8, 0.8))
ax8_right[0].axvspan(1.22, 1.4, color=(0.8, 0.8, 1.0))

ax8_right[1].clear()

ax8_right[1].axvspan(0.5, 0.75, color=(0.8, 0.8, 1.0))
ax8_right[1].axvspan(0.83, 1.05, color=(1.0, 0.8, 0.8))
ax8_right[1].axvspan(1.15, 1.39, color=(0.8, 0.8, 1.0))

ax8_right[2].clear()

ax8_right[2].axvspan(0.5, 0.71, color=(0.8, 0.8, 1.0))
ax8_right[2].axvspan(0.87, 0.95, color=(1.0, 0.8, 0.8))
ax8_right[2].axvspan(1.12, 1.37, color=(0.8, 0.8, 1.0))

plotPciAsym(1160506007, ax8_right[0])
plotPciAsym(1160506009, ax8_right[1])
plotPciAsym(1160506015, ax8_right[2])

ax8_last.axis('off')
ax8_last.annotate(r'$50<f<200$', xy=(0.15,0.5), ha='left', va='center', xycoords='axes fraction', color=mpl.cm.PiYG(0.0))
ax8_last.annotate(r'$200<f<700$', xy=(0.85,0.5), ha='right', va='center', xycoords='axes fraction', color=mpl.cm.PiYG(1.0))
ax8_last.annotate(r'[kHz]', xy=(0.5,-2), ha='center', va='center', xycoords='axes fraction')

plt.setp(ax8_right[2].get_xticklabels(), visible=True)
ax8_right[2].set_xlabel('time [sec]')
ax8_right[0].set_title('PCI Asymmetry')


plt.savefig('fig_pci_spec.eps', format='eps', dpi=240, facecolor='white')
