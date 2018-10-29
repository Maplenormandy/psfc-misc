# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 16:02:10 2017

@author: normandy
"""

import readline
import MDSplus

readline

import numpy as np
import matplotlib.pyplot as plt
#import matplotlib as mpl

from scipy import optimize
#from scipy import signal
#from scipy import stats

# %%

elecTree = MDSplus.Tree('electrons', 1160506007)

# 9 and 10 are normally the best
sig88ui = elecTree.getNode('\ELECTRONS::TOP.REFLECT.CPCI:DT132_1:INPUT_09').data()
sig88uq = elecTree.getNode('\ELECTRONS::TOP.REFLECT.CPCI:DT132_1:INPUT_10').data()
sig88ut = elecTree.getNode('\ELECTRONS::TOP.REFLECT.CPCI:DT132_1:INPUT_09').dim_of().data()

ci = np.mean(sig88ui)
cq = np.mean(sig88uq)

nl04Node = elecTree.getNode('\ELECTRONS::TOP.TCI.RESULTS:NL_04')

# %%
t1, t2 = np.searchsorted(sig88ut, (0.5,1.6))
t2 = t1+(2048*1000)
#0.5944-0.5954
#0.9625-0.9650

# Get raw signals over proper time window
si = sig88ui[t1:t2]
sq = sig88uq[t1:t2]
st = sig88ut[t1:t2]

ci = np.median(si)
cq = np.median(sq)

sz = si-ci + 1j*(sq-cq)

# Filter signals
#b, a = signal.butter(4, 1e-1, btype='highpass')
#filt_si = signal.filtfilt(b, a, si-ci)
#filt_sq = signal.filtfilt(b, a, sq-cq)

def getPhiWrapped(si, sq, ci, cq):
    phi = np.arctan2(si-ci, sq-cq)

    phidiff = np.diff(phi)
    phidiff1 = phidiff > np.pi
    phidiff2 = phidiff < -np.pi

    phidiffsum1 = np.cumsum(phidiff1)*-2*np.pi
    phidiffsum2 = np.cumsum(phidiff2)*2*np.pi

    phdiffsum = np.concatenate(([0], phidiffsum1+phidiffsum2))
    return (phi+phdiffsum)

#t3, t4 = np.searchsorted(st, (0.5,1.5))
#t4 = t3+(2048*1000)

#st = st[t3:t4]
#sz = filt_si[t3:t4] + 1j*filt_sq[t3:t4]

#c2 = getPhiWrapped(si, sq, ci, cq)

#hsi = filt_si[t3:t4]
#hsq = filt_sq[t3:t4]
#hst = st[t3:t4]
#hsz = hsi+1j*hsq
#hc2 = c2[t3:t4]


# %%

nsamp = 2048
szr = np.reshape(sz/(np.abs(sz)+1e-8), (-1, nsamp))
szt = np.reshape(st, (-1, nsamp))
szt2 = szt - np.mean(szt, axis=-1, keepdims=True)

#fszt = np.abs(np.fft.fft(szr, axis=-1))

def phiVar(x, t):
    rot = np.exp(-1j*x*szt2[t,:])
    newZ = szr[t,:]*rot
    return -2.0*np.log(np.abs(np.mean(newZ)))

csr = np.zeros(szr.shape[0])
runaway = np.zeros(szr.shape[0])

for i in range(szr.shape[0]):
    sol = optimize.minimize(phiVar, np.array([0.0]), args=(i,))
    csr[i] = phiVar(sol.x, i)
    runaway[i] = sol.x

csra = -2.0*np.log(np.abs(np.mean(szr, axis=-1)))

#csr = -2.0*np.log(np.abs(np.mean(szr, axis=-1)))
#csr = -2.0*np.log(np.amax(fszt, axis=-1))
#csr = getPhiWrapped(np.real(np.mean(szr, axis=-1)), np.imag(np.mean(szr, axis=-1)), 0, 0)
#csr = np.angle(np.mean(szr, axis=-1))
cst = np.mean(szt, axis=-1)

# %%
print csr.shape
nsamp2 = 20480*4/nsamp

csr2 = np.mean(np.reshape(csr, (-1, nsamp2)), axis=-1)
csra2 = np.mean(np.reshape(csra, (-1, nsamp2)), axis=-1)
stdcsr2 = np.std(np.reshape(csr, (-1, nsamp2)), axis=-1)
cst2 = np.mean(np.reshape(cst, (-1, nsamp2)), axis=-1)
runaway2 = np.mean(np.reshape(runaway, (-1, nsamp2)), axis=-1)
stdrunaway2 = np.std(np.reshape(runaway, (-1, nsamp2)), axis=-1)

plt.figure()
plt.errorbar(cst2, csr2, yerr=stdcsr2/np.sqrt(nsamp2), marker='.')
#plt.errorbar(cst2, csra2, yerr=stdcsr2/np.sqrt(nsamp2), marker='.')
plt.axhline(np.mean(csr2), ls='--')
plt.axhline(np.mean(csr2)+np.std(csr2), ls=':')
plt.axhline(np.mean(csr2)-np.std(csr2), ls=':')

plt.xlabel('time [s]')
plt.ylabel('Var($\phi$) [rad${}^2$]')

#plt.figure()
#plt.errorbar(cst2, runaway2, yerr=stdrunaway2/np.sqrt(nsamp2), marker='.')
#plt.plot(cst2, -np.cumsum(runaway2), marker='.')

plt.show()
