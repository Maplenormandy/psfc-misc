# -*- coding: utf-8 -*-
"""
Created on Fri May 15 12:15:27 2020

@author: normandy
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors

import numpy as np
import scipy.io, scipy.signal

import sys
sys.path.append('/home/normandy/bin')

import fftcece8 as fftcece

import cPickle as pkl

import MDSplus

import eqtools

font = {'family' : 'serif',
        'serif': ['Computer Modern'],
        'size'   : 10}

mpl.rc('font', **font)

# %% Calculate CECE data


def calcCECEdata(shot, t_loc, t_soc):
    cdata = fftcece.cecedata()
    cdata.fftBin = 128
    cdata.lagWindow = 0
    cdata.lowpassf = 1.5e6
    cdata.readMDS(shot,4)
    cdata.lowpass()
    
    hanningBase = np.r_[0:(cdata.samplingRate):(float(cdata.samplingRate)/
                                                             cdata.fftBin)]

    f0 = hanningBase[0]    #Hz
    f1 = hanningBase[-1]    #Hz
    
    #Convert frequencies to indices
    f0index = 0
    f1index = len(hanningBase)-1
    
    """
    T~/T Calculations, from Creely's code
    """

    #Calculate Te~ over Te
    Bif = 100000000.0      #Hz


    #Signal bandwidth
    Bsig = f1-f0

    #Conversion factor from normalized coherence to (Te_tilde/Te)^2/kHz
    ttfactor = (1000.0/Bif) 
    #ttfactor = 1
    
    
    cdata.timeBegin = t_loc[0]
    cdata.timeEnd = t_loc[1]
    
    cdata.calcAutoOverlap()
    cdata.calcCrossOverlap()
    cdata.calcCoherence()
    cdata.calcCrossCorr()

    cdata.coherence[0,:] = np.nan
    tt_loc = ttfactor*cdata.coherence[:,:]
    tterr_loc = ttfactor*cdata.coherVar[:,:]
    ttstat_loc = ttfactor*cdata.statlimit[:,:]
    
    cdata.timeBegin = t_soc[0]
    cdata.timeEnd = t_soc[1]
    
    cdata.calcAutoOverlap()
    cdata.calcCrossOverlap()
    cdata.calcCoherence()
    cdata.calcCrossCorr()
    
    cdata.coherence[0,:] = np.nan
    tt_soc = ttfactor*cdata.coherence[:,:]
    tterr_soc = ttfactor*cdata.coherVar[:,:]
    ttstat_soc = ttfactor*cdata.statlimit[:,:]
    
    return hanningBase/1000.0, (tt_loc, tt_soc), (tterr_loc, tterr_soc), (ttstat_loc, ttstat_soc)
    
cece1 = calcCECEdata(1160506007, (0.93, 0.99), (0.57, 0.63))    
cece2 = calcCECEdata(1160506009, (0.89, 0.95), (0.69, 0.75))
cece1a = calcCECEdata(1160506008, (0.93, 0.99), (0.57, 0.63))
cece2a = calcCECEdata(1160506010, (0.89, 0.95), (0.69, 0.75))

#cece1 = calcCECEdata(1160506007, (0.85, 1.05), (0.55, 0.75))    
#cece2 = calcCECEdata(1160506009, (0.85, 1.05), (0.55, 0.75))
#cece1a = calcCECEdata(1160506008, (0.85, 1.05), (0.55, 0.75))
#cece2a = calcCECEdata(1160506010, (0.85, 1.05), (0.55, 0.75))
cece3 = calcCECEdata(1160506015, (0.87, 0.93), (0.65, 0.71))

# %% CECE plots

fig_cece = plt.figure(figsize=(3.375*2, 3.375))
gs_cece = mpl.gridspec.GridSpec(2, 3, hspace=0.00)
ax_cece = np.array(map(plt.subplot, gs_cece)).reshape(2,3)


def plotCECE(shot, cece, ax_r0, ax_r1, cecea=None):
    if cecea == None:    
        freq = cece[0]
        tt = cece[1]
        tterr = cece[2]
        ttstat = cece[3]
    else:
        freq = cece[0]
        tt = ((cece[1][0]+cecea[1][0])/2, (cece[1][1]+cecea[1][1])/2)
        tterr = (np.sqrt((cece[2][0]**2+cecea[2][0]**2)/2), np.sqrt((cece[2][1]**2+cecea[2][1]**2)/2))
        ttstat = cece[3]
    
    
    ax_r1.fill_between(freq, ttstat[0][:,0]*1e6, color=(0.8, 0.8, 0.8))
    ax_r0.fill_between(freq, ttstat[0][:,5]*1e6, color=(0.8, 0.8, 0.8))
    ax_r1.plot(freq, ttstat[0][:,0]*1e6, c='k', ls='--')
    ax_r0.plot(freq, ttstat[0][:,5]*1e6, c='k', ls='--')
    
    ax_r1.errorbar(freq, tt[0][:,0]*1e6, yerr=tterr[0][:,0]*1e6, c='r')
    ax_r0.errorbar(freq, tt[0][:,5]*1e6, yerr=tterr[0][:,5]*1e6, c='r')
    
    ax_r1.errorbar(freq, tt[1][:,0]*1e6, yerr=tterr[1][:,0]*1e6, c='b')
    ax_r0.errorbar(freq, tt[1][:,5]*1e6, yerr=tterr[1][:,5]*1e6, c='b')
    
    ax_r0.set_xlim([0, 500])
    ax_r1.set_xlim([0, 500])
    ax_r0.set_ylim([0,1.0])
    ax_r1.set_ylim([0,0.99])
    
    ax_r0.ticklabel_format(style='sci', scilimits=(-2,8), axis='y')
    ax_r1.ticklabel_format(style='sci', scilimits=(-2,8), axis='y')
    
    plt.setp(ax_r0.get_xticklabels(), visible=False)
    #plt.setp(ax_r1.get_xticklabels(), visible=False)
    plt.setp(ax_r0.get_yticklabels(), visible=False)
    plt.setp(ax_r1.get_yticklabels(), visible=False)
    
    ax_r0.xaxis.set_major_locator(mpl.ticker.MultipleLocator(100))
    ax_r1.xaxis.set_major_locator(mpl.ticker.MultipleLocator(100))

plotCECE(1160506007, cece1, ax_cece[0,0], ax_cece[1,0])
plotCECE(1160506009, cece2, ax_cece[0,1], ax_cece[1,1])
plotCECE(1160506015, cece3, ax_cece[0,2], ax_cece[1,2])

plt.setp(ax_cece[0,0].get_yticklabels(), visible=True)
plt.setp(ax_cece[1,0].get_yticklabels(), visible=True)
#ax_cece[2,1].set_xlabel('Freq [kHz]')
#ax_cece[2,0].set_xlabel('Freq [kHz]')

ax_cece[0,0].set_title('(0.8 MA Ohmic)')
ax_cece[0,1].set_title('(1.1 MA Ohmic)')
ax_cece[0,2].set_title('(0.8 MA +ICRF)')

ax_cece[1,1].set_xlabel('Frequency [kHz]')

ax_cece[0,0].set_ylabel('Cross power spectrum [$10^{-6} (\\tilde{T}_e / T_e)^2 / \\mathrm{kHz}$]')
ax_cece[0,0].yaxis.set_label_coords(-0.2, 0.0)
ax_cece[0,0].annotate(r'r/a$\sim$0.65', xy=(0.05,0.95), xycoords='axes fraction', ha='left', va='top')
ax_cece[1,0].annotate(r'r/a$\sim$0.7', xy=(0.05,0.95), xycoords='axes fraction', ha='left', va='top')
ax_cece[0,1].annotate(r'r/a$\sim$0.65', xy=(0.05,0.95), xycoords='axes fraction', ha='left', va='top')
ax_cece[1,1].annotate(r'r/a$\sim$0.7', xy=(0.05,0.95), xycoords='axes fraction', ha='left', va='top')
ax_cece[0,2].annotate(r'r/a$\sim$0.65', xy=(0.05,0.95), xycoords='axes fraction', ha='left', va='top')
ax_cece[1,2].annotate(r'r/a$\sim$0.7', xy=(0.05,0.95), xycoords='axes fraction', ha='left', va='top')

#fig_cece.suptitle('Cross power spectrum', fontsize=15)

plt.tight_layout()
plt.tight_layout()

plt.savefig('fig_cece.eps', format='eps', dpi=1200, facecolor='white')

# %%

def get_refl_spectra(shot, t_plot):
    elecTree = MDSplus.Tree('electrons', 1160506007)
    
    # 9 and 10 are normally the best
    sig88ui = elecTree.getNode('\ELECTRONS::TOP.REFLECT.CPCI:DT132_1:INPUT_09').data()
    sig88uq = elecTree.getNode('\ELECTRONS::TOP.REFLECT.CPCI:DT132_1:INPUT_10').data()
    sig88ut = elecTree.getNode('\ELECTRONS::TOP.REFLECT.CPCI:DT132_1:INPUT_09').dim_of().data()
    
    ci = np.mean(sig88ui)
    cq = np.mean(sig88uq)
    
    
    t1, t2 = np.searchsorted(sig88ut, (0.5,1.5))
    t1 = 0
    t2 = len(sig88ut)-1
    
    si = sig88ui[t1:t2]
    sq = sig88uq[t1:t2]
    st = sig88ut[t1:t2]
    ci = np.mean(si)
    cq = np.mean(sq)
    
    z = (si-ci)+1j*(sq-cq)
    
    t_loc, t_soc = np.searchsorted(st, t_plot)
    #t_soc, t_loc = np.searchsorted(st, (0.1, 0.9637))
    
    total_samples=4096*4*4
    down_samples =1024
    
    z_null = z[-total_samples:]
    z_soc = z[t_soc-total_samples/2:t_soc+total_samples/2]
    z_loc = z[t_loc-total_samples/2:t_loc+total_samples/2]
    
    fz_null = np.fft.fftshift(np.fft.fft(z_null))/2e3/2e3
    fz_soc = np.fft.fftshift(np.fft.fft(z_soc))/2e3/2e3
    fz_loc = np.fft.fftshift(np.fft.fft(z_loc))/2e3/2e3
    freqs = np.fft.fftshift(np.fft.fftfreq(total_samples, 1.0/2e6))
    
    Sxx_null = fz_null*np.conjugate(fz_null)
    Sxx_loc_down = np.real(np.average(np.reshape(fz_loc*np.conjugate(fz_loc)-Sxx_null, (down_samples, -1)), axis=-1))
    Sxx_soc_down = np.real(np.average(np.reshape(fz_soc*np.conjugate(fz_soc)-Sxx_null, (down_samples, -1)), axis=-1))
    
    f_down = np.average(np.reshape(freqs/1e3, (down_samples, -1)), axis=-1)
    
    return f_down, Sxx_loc_down, Sxx_soc_down

refl1 = get_refl_spectra(1160506007, (0.96, 0.6))
refl2 = get_refl_spectra(1160506015, (0.9, 0.68))


def find_refl_cutoff(shot, folder, freq):
    data = 'ne'
    prof = pkl.load(file('/home/normandy/git/psfc-misc/Fitting/FitsPoP2019/%s/%s_dict_fit_%d.pkl'%(folder, data, shot)))
    
    # Real frequency of plasma frequency at 1e20 m^-3
    ompe = 89.7866282
    
    om_prof = np.sqrt(prof['y'][10:])*ompe
    roa_prof = prof['X'][10:]
    
    if np.all(freq > om_prof):
        return -1.0
    else:
        return scipy.interpolate.interp1d(om_prof, roa_prof)(freq)

print find_refl_cutoff(1160506007, '007_loc', 88.5)
print find_refl_cutoff(1160506009, '009_loc', 88.5)
print find_refl_cutoff(1160506015, '015_loc', 88.5)

# %%


def plot_refl(ax, refl, roa):
    f_down, Sxx_loc_down, Sxx_soc_down = refl
    
    ax.semilogy(f_down, Sxx_soc_down, c='b')
    ax.semilogy(f_down, Sxx_loc_down, c='r')
    ax.set_xlim([-450,450])
    ax.set_xlabel('f (kHz)')
    ax.set_ylim([1e-12, 1e-8])
    ax.text(120, 1e-9, 'r/a$\\sim$'+str(roa))

fig_refl = plt.figure(4, figsize=(3.375*1.67, 3.375*0.67))
gs_refl = mpl.gridspec.GridSpec(1, 2)
ax_refl = np.array(map(plt.subplot, gs_refl))

plot_refl(ax_refl[0], refl1, 0.57)
plot_refl(ax_refl[1], refl2, 0.63)


ax_refl[0].set_ylabel('(arb. units)')
plt.setp(ax_refl[1].get_yticklabels(), visible=False)

ax_refl[0].set_title('(0.8 MA Ohmic)')
ax_refl[1].set_title('(1.1 MA +ICRF)')

plt.tight_layout()
plt.tight_layout()

plt.savefig('fig_refl.eps', format='eps', dpi=1200, facecolor='white')