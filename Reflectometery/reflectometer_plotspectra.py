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

b, a = signal.butter(4, 0.2, btype='lowpass')
#sshsi = signal.filtfilt(b, a, si)
#sshsq = signal.filtfilt(b, a, sq)

sshsi = si
sshsq = sq

t3, t4 = np.searchsorted(st, (0.5,1.5))

hsi = sshsi[t3:t4]
hsq = sshsq[t3:t4]
hst = st[t3:t4]
hsz = hsi+1j*hsq

def getPhiWrapped(si, sq, ci, cq):
    phi = np.arctan2(si-ci, sq-cq)

    phidiff = np.diff(phi)
    phidiff1 = phidiff > np.pi
    phidiff2 = phidiff < -np.pi
    
    phidiffsum1 = np.cumsum(phidiff1)*-2*np.pi
    phidiffsum2 = np.cumsum(phidiff2)*2*np.pi
    
    phdiffsum = np.concatenate(([0], phidiffsum1+phidiffsum2))
    return (phi+phdiffsum)

c2 = getPhiWrapped(sshsi, sshsq, 0, 0)
c3 = signal.filtfilt(b, a, c2)



f, t, Sxx = signal.spectrogram(hsz, fs=2e6, nperseg=8192, return_onesided=False)

f2 = np.fft.fftshift(f, axes=0)
Sxx2 = np.fft.fftshift(Sxx, axes=0)

plt.figure()
plt.pcolormesh(t+0.5, f2/1000, Sxx2, cmap='cubehelix', norm=mpl.colors.LogNorm(vmin=10**-5, vmax=10**-10))
plt.colorbar()
plt.xlabel('time [sec]')
plt.ylabel('freq [kHz]')

"""
H, xedges, yedges = np.histogram2d(hsi, hsq, bins=128)
H = H.T

plt.figure()
x,y = np.meshgrid(xedges, yedges)
plt.pcolormesh(x, y, np.clip(H, np.min(H), np.percentile(H, 99)), cmap='cubehelix')
plt.axis('square')
"""

# %%

nsamp = 16
avgWindow = np.ones(nsamp)/(1.0*nsamp)



Sxxm2 = np.array([np.convolve(Sxx2[:,i], avgWindow, mode='valid') for i in range(Sxx2.shape[1])]).T
Sxxm4 = np.array([np.convolve(Sxx2[:,i]**2, avgWindow, mode='valid') for i in range(Sxx2.shape[1])]).T
fm = np.convolve(f2, avgWindow, mode='valid')

Sxxk2 = (nsamp/(nsamp-1.0))*Sxxm2
Sxxk4 = nsamp*nsamp*((nsamp+1)*Sxxm4-3*(nsamp-1)*Sxxm2**2)/((nsamp-1.0)*(nsamp-2.0)*(nsamp-3.0))


Sxxm22 = np.array([np.convolve(Sxx2[i,:], avgWindow, mode='valid') for i in range(Sxx2.shape[0])])
Sxxm42 = np.array([np.convolve(Sxx2[i,:]**2, avgWindow, mode='valid') for i in range(Sxx2.shape[0])])
tm = np.convolve(t, avgWindow, mode='valid')


Sxxk22 = (nsamp/(nsamp-1.0))*Sxxm22
Sxxk42 = nsamp*nsamp*((nsamp+1)*Sxxm42-3*(nsamp-1)*Sxxm22**2)/((nsamp-1.0)*(nsamp-2.0)*(nsamp-3.0))


# %%

plt.figure()
#plt.pcolormesh(t+0.5, f2/1000.0, Sxx2, cmap='RdBu', vmin=0, vmax=1e-10)
#plt.colorbar()

# 1160506007 0.57-0.63 or 0.93-0.99
# 1160506015 0.67-0.73 or 0.87-0.93
t5, t6 = np.searchsorted(t+0.5, (0.6,0.96))

#plt.semilogy(fm/1000.0, Sxxm2[:,t5])
#plt.semilogy(fm/1000.0, Sxxm2[:,t6])

#plt.plot(f2/1000.0, Sxxk42[:,t5]/Sxxk22[:,t5]**2, c='b', lw=2)
#plt.loglog(-fm/1000.0, Sxxk4[:,t5], c='b')
#plt.plot(f2/1000.0, Sxxk42[:,t6]/Sxxk22[:,t6]**2, c='r', lw=2)
#plt.loglog(-fm/1000.0, Sxxk4[:,t6-1], c='r')

plt.semilogy(fm/1000.0, Sxxk2[:,t5], c='b', lw=2)

plt.semilogy(fm/1000.0, Sxxk2[:,t6], c='r', lw=2)
plt.xlabel('freq [kHz]')
plt.ylabel('smoothed spectral power density')

# %% FFT stuff
t7, t8 = np.searchsorted(st, (0.7,0.9))
zf = np.fft.fft(z[t7-512:t7+512])

zf2 = np.fft.fftshift(zf)

# %% Stuff

"""
crossphase = np.outer(zf2, np.conjugate(zf2))
powers = np.abs(np.diag(crossphase))

nsamp = 32
avgWindow = np.ones(nsamp)/(1.0*nsamp)
avgWindow2 = np.ones((nsamp,nsamp))/(1.0*nsamp*nsamp)

crossphase = signal.convolve2d(crossphase, avgWindow2, mode='valid')
powers = np.convolve(powers, avgWindow, mode='valid')
fm = np.convolve(f2, avgWindow, mode='valid')

crossphase = crossphase / np.sqrt(np.outer(powers, powers))
"""


# %%

plt.pcolormesh(fm/1000.0, fm/1000.0, np.abs(crossphase))
plt.colorbar()

# %% Ion vs electron direction spectral power

hsii = hsi**2
hsqq = hsq**2
hsiq = hsi*hsq

hsii2 = np.mean(np.reshape(hsii, (-1, 1000)), 1)
hsqq2 = np.mean(np.reshape(hsqq, (-1, 1000)), 1)
hsiq2 = np.mean(np.reshape(hsiq, (-1, 1000)), 1)
hst2 = np.mean(np.reshape(hst, (-1, 1000)), 1)

Sxxlo = np.sum(Sxx2[:f2.size/2], axis=0)
Sxxhi = np.sum(Sxx2[f2.size/2:], axis=0)

hsdet = np.sqrt((hsii2 - hsqq2)**2 + 4*hsiq2**2)

hse1 = 0.5 * (hsii2+hsqq2 + hsdet)
hse2 = 0.5 * (hsii2+hsqq2 - hsdet)

v1x = -(hsqq2 - hsii2 + hsdet) / (2 * hsiq2)
angle = np.arctan(v1x) / np.pi * 180.0

plt.figure()
#plt.plot(t+0.5, Sxxlo, c='g')
#plt.plot(t+0.5, Sxxhi, lw=2, c='g')
plt.plot(hst2, hse1, lw=3, c='g')
plt.plot(hst2, hse2, lw=3, c='b')

plt.figure()
plt.plot(hst2, angle, lw=3, c='r')

# %%

plt.figure()
plt.pcolormesh(tm+0.5, f2/1000, Sxxk42/Sxxk22**2, cmap='BrBG', vmin=-2, vmax=2)
plt.colorbar()
plt.xlabel('time [sec]')
plt.ylabel('freq [kHz]')