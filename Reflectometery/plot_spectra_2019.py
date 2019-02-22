# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 16:02:10 2017

@author: normandy
"""

import sys
sys.path.append('/home/normandy/git/psfc-misc/Common')

import readline
import MDSplus

readline

import numpy as np
import matplotlib.pyplot as plt
#import matplotlib as mpl

#from scipy import signal
#from scipy import stats
#from scipy import optimize

#import ShotAnalysisTools as sat


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

def plotSpectra(total_samples, time_slices):
    t_soc, t_loc = np.searchsorted(st, (0.5949, 0.9637))

    z_soc = z[t_soc-total_samples/2*time_slices:t_soc+total_samples/2*time_slices]
    z_loc = z[t_loc-total_samples/2*time_slices:t_loc+total_samples/2*time_slices]
    z_soc = z_soc.reshape((time_slices, total_samples))
    z_loc = z_loc.reshape((time_slices, total_samples))

    fz_soc = np.fft.fftshift(np.fft.fft(z_soc, axis=1), axes=1)/2e3/2e3
    fz_loc = np.fft.fftshift(np.fft.fft(z_loc, axis=1), axes=1)/2e3/2e3
    freqs = np.fft.fftshift(np.fft.fftfreq(total_samples, 1.0/2e6))

    Sxx_loc_down = np.real(np.average(fz_loc*np.conjugate(fz_loc), axis=0))
    Sxx_soc_down = np.real(np.average(fz_soc*np.conjugate(fz_soc), axis=0))

    f_down = freqs/1e3

    plt.semilogy(f_down, Sxx_loc_down, c='r')
    plt.semilogy(f_down, Sxx_soc_down, c='b')
    plt.xlim([-450, 450])

    plt.show()

plotSpectra(2048, 64)
