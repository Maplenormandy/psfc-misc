# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 10:32:34 2017

@author: normandy

See also:
Transp/getHeatFluxes.py
cgyro/plot_fluxes.py

Some plotting routines are meant to be run on the PSFC cluster
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

import readline
import MDSplus

from scipy import signal

font = {'family': 'normal', 'size': 18}
mpl.rc('font', **font)

# %% Load reflectometry data


elecTree = MDSplus.Tree('electrons', 1160506007)

sig88ui = elecTree.getNode('\ELECTRONS::TOP.REFLECT.CPCI:DT132_1:INPUT_09').data()
sig88uq = elecTree.getNode('\ELECTRONS::TOP.REFLECT.CPCI:DT132_1:INPUT_10').data()
sig88ut = elecTree.getNode('\ELECTRONS::TOP.REFLECT.CPCI:DT132_1:INPUT_09').dim_of().data()

# %% Process data

t1, t2 = np.searchsorted(sig88ut, (0.3,1.8))
#0.5944-0.5954
#0.9625-0.9650

si = sig88ui[t1:t2]
sq = sig88uq[t1:t2]
st = sig88ut[t1:t2]
z = si+1j*sq
ampli = si**2 + sq**2

b, a = signal.butter(4, (0.01, 0.7), btype='bandpass')
sshsi = signal.filtfilt(b, a, si)
sshsq = signal.filtfilt(b, a, sq)

t3, t4 = np.searchsorted(st, (0.3,1.8))

hsi = sshsi[t3:t4]
hsq = sshsq[t3:t4]
hst = st[t3:t4]

hz = hsi+1j*hsq


# %%


f, t, Sxx = signal.spectrogram(z, fs=2e6, nperseg=1024, return_onesided=False)
f, t, hSxx = signal.spectrogram(hz, fs=2e6, nperseg=1024, return_onesided=False)

f2 = np.fft.fftshift(f, axes=0)
Sxx2 = np.fft.fftshift(Sxx, axes=0)
hSxx2 = np.fft.fftshift(hSxx, axes=0)

# %%

plt.figure()
plt.pcolormesh(t+0.5, f2/1000, Sxx2, cmap='cubehelix', norm=mpl.colors.LogNorm(vmin=10**-5, vmax=10**-10))
plt.colorbar()
plt.xlabel('time [sec]')
plt.ylabel('freq [kHz]')


# %% Hysteresis plot

nsamp = 128
avgWindow = np.ones(nsamp)/(1.0*nsamp)
avgWindow2 = np.ones((1,nsamp))/(1.0*nsamp)


nl04Node = elecTree.getNode('\ELECTRONS::TOP.TCI.RESULTS:NL_04')
nlData = np.convolve(nl04Node.data(), avgWindow, mode='valid') # 0.0005 per time bin
nlTime = np.convolve(nl04Node.dim_of().data(), avgWindow, mode='valid')

rfData = np.convolve(np.sum(hSxx2, axis=0), avgWindow, mode='valid')
rfTime = np.convolve(t+0.3, avgWindow, mode='valid')

nl0, nl1 = np.searchsorted(nlTime, (0.5, 1.5))

summedData = np.interp(nlTime[nl0:nl1], rfTime, rfData)

plt.plot(nlData[nl0:nl1], summedData, marker='.')

# %% Plot spectra



# %%

nsamp = 32
avgWindow = np.ones(nsamp)/(1.0*nsamp)
avgWindow2 = np.ones((1,nsamp))/(1.0*nsamp)

fm = np.convolve(f2, avgWindow, mode='valid')
tm = np.convolve(t, avgWindow, mode='valid')
t4, t5 = np.searchsorted(tm+0.3, (0.6, 0.96))
powers = signal.convolve2d(Sxx2, avgWindow2, mode='valid')


fmin, fmax = np.searchsorted(f2, (-4.5e5, 4.5e5))

plt.semilogy(f2[fmin:fmax]/1e3, powers[fmin:fmax,t4], c='b')
plt.semilogy(f2[fmin:fmax]/1e3, powers[fmin:fmax,t5], c='r')
plt.xlim([-450, 450])
plt.xlabel('f [kHz]')

#plt.semilogy(f2, Sxx2[:,t4], c='b')
#plt.semilogy(f2, Sxx2[:,t5], c='r')