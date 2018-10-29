# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 16:02:10 2017

@author: normandy
"""

import readline
import MDSplus

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from scipy import signal
from scipy import stats
from scipy import optimize

# %%

elecTree = MDSplus.Tree('electrons', 1160506007)

# 9 and 10 are normally the best
sig88ui = elecTree.getNode('\ELECTRONS::TOP.REFLECT.CPCI:DT132_1:INPUT_09').data()
sig88uq = elecTree.getNode('\ELECTRONS::TOP.REFLECT.CPCI:DT132_1:INPUT_10').data()
sig88ut = elecTree.getNode('\ELECTRONS::TOP.REFLECT.CPCI:DT132_1:INPUT_09').dim_of().data()

ci = np.mean(sig88ui)
cq = np.mean(sig88uq)


# %%
t1, t2 = np.searchsorted(sig88ut, (0.5,1.5))
#0.5944-0.5954
#0.9625-0.9650

si = sig88ui[t1:t2]
sq = sig88uq[t1:t2]
st = sig88ut[t1:t2]
z = si+1j*sq
ampli = si**2 + sq**2

lb, la = signal.butter(4, 1e-4, btype='lowpass')
hb, ha = signal.butter(4, 1e-4, btype='highpass')

lsi = signal.filtfilt(lb, la, si)
lsq = signal.filtfilt(lb, la, sq)

hsi = signal.filtfilt(hb, ha, si)
hsq = signal.filtfilt(hb, ha, sq)

ci = np.median(si)
cq = np.median(sq)

sz = (si-ci) + 1j*(sq-cq)

H, xedges, yedges = np.histogram2d(hsi, hsq, bins=128)
H = H.T

plt.figure()
x,y = np.meshgrid(xedges, yedges)
plt.pcolormesh(x, y, np.clip(H, np.min(H), np.percentile(H, 99)), cmap='cubehelix')
plt.axis('square')

# 2 MHz sampling frequency -> 1 MHz Nyquist frequency
# 100Hz

# %% Reshape into 10ms time windows

nsamp = 1999

shsi = np.reshape(si, (-1, nsamp))
shsq = np.reshape(sq, (-1, nsamp))

csi = shsi - np.mean(shsi, axis=-1, keepdims=True)
csq = shsq - np.mean(shsq, axis=-1, keepdims=True)

rcsi = np.ravel(csi)
rcsq = np.ravel(csq)

dsi = np.gradient(rcsi)
dsq = np.gradient(rcsq)

mag = np.abs(sz)+1e-2

#dpar = (dsi*(si-ci) + dsq*(sq-cq))/mag
#dper = (dsq*(si-ci) - dsi*(sq-cq))/mag

dpar = (dsi*rcsi + dsq*rcsq)/mag
dper = (dsq*rcsi - dsi*rcsq)/mag


dperr = np.reshape(dper, (-1, nsamp))
dperavg = np.mean(dperr, axis=-1, keepdims=True)
dper2 = np.ravel(dperr - dperavg)

par = np.cumsum(dpar)
per = np.cumsum(dper2)

hper = per

avgt = np.mean(np.reshape(st, (-1, nsamp)), axis=-1)
rper = np.reshape(per, (-1, nsamp))
m1 = np.mean(rper, axis=-1)
drper = rper-m1[:,np.newaxis]
m2 = np.mean(drper**2, axis=-1)
m3 = np.mean(drper**3, axis=-1)
m4 = np.mean(drper**4, axis=-1)


k2 = (nsamp/(nsamp-1.0))*m2
k3 = nsamp*nsamp*m3/((nsamp-1.0)*(nsamp-2.0))
k4 = nsamp*nsamp*((nsamp+1)*m4-3*(nsamp-1)*m2**2)/((nsamp-1.0)*(nsamp-2.0)*(nsamp-3.0))

# %%

f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
ax2.plot(avgt, k2)
ax1.plot(avgt, k3/(k2**1.5))

# %%

plt.plot(avgt, np.sqrt(np.mean(shsi-ci, axis=-1)**2 + np.mean(shsq-cq, axis=-1)**2))

# %%
H, xedges, yedges = np.histogram2d(st, np.abs(sz), bins=256)
H = H.T

plt.figure()
x,y = np.meshgrid(xedges, yedges)
plt.pcolormesh(x, y, np.clip(H, np.min(H), np.percentile(H, 99)), cmap='cubehelix')

# %%

f, t, Sxx = signal.spectrogram(dper2, fs=2e6, nperseg=8192, return_onesided=False)

f2 = np.fft.fftshift(f, axes=0)
Sxx2 = np.fft.fftshift(Sxx, axes=0)

plt.figure()
plt.pcolormesh(t+0.5, f2/1000, Sxx2, cmap='cubehelix', norm=mpl.colors.LogNorm(vmin=10**-5, vmax=10**-10))
plt.colorbar()
plt.xlabel('time [sec]')
plt.ylabel('freq [kHz]')