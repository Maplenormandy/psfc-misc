# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 18:29:48 2017

@author: normandy
"""

import numpy as np
import matplotlib.pyplot as plt

import readline
import MDSplus


from scipy import signal
from scipy.stats import norm

# %% Get the real sampled time

elecTree = MDSplus.Tree('electrons', 1160506015)
sig88ut = elecTree.getNode('\ELECTRONS::TOP.REFLECT.CPCI:DT132_1:INPUT_05').dim_of().data()
t1, t2 = np.searchsorted(sig88ut, (0.5,1.5))
st = sig88ut[t1:t2]

# %% Function def


def getPhiWrapped(si, sq, ci, cq):
    phi = np.arctan2(si-ci, sq-cq)

    phidiff = np.diff(phi)
    phidiff1 = phidiff > np.pi
    phidiff2 = phidiff < -np.pi
    
    phidiffsum1 = np.cumsum(phidiff1)*-2*np.pi
    phidiffsum2 = np.cumsum(phidiff2)*2*np.pi
    
    phdiffsum = np.concatenate(([0], phidiffsum1+phidiffsum2))
    return (phi+phdiffsum)

# %% Generate a false signal

phi_base = np.sin(st*200e3*2*np.pi) + np.sin(st*157e3*2*np.pi) + np.sin(st*245e3*2*np.pi)

sigTotal = np.exp(1j*phi_base)+1.01*np.exp(1j*st*19e3*2*np.pi)

si = np.real(sigTotal)
sq = np.imag(sigTotal)

#b, a = signal.butter(8, [0.1, 0.5], btype='bandpass')
b,a = signal.butter(8, 0.1, btype='highpass')

hsi = signal.filtfilt(b, a, si)
hsq = signal.filtfilt(b, a, sq)

c2 = getPhiWrapped(hsi, hsq, 0, 0)
c3 = signal.filtfilt(b, a, c2)
    
c1 = getPhiWrapped(si, sq, 0, 0)

c4 = signal.filtfilt(b, a, phi_base)

# %%
#plt.plot(st, c1)
plt.plot(st[1000000:1001000], -c3[1000000:1001000])
#plt.plot(st[1000000:1010000], c2[1000000:1010000])
plt.plot(st[1000000:1001000], phi_base[1000000:1001000])
#plt.plot(st[1000000:1010000], c2[1000000:1010000])