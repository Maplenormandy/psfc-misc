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
import matplotlib as mpl

from scipy import signal
#from scipy import stats
from scipy import optimize

import ShotAnalysisTools as sat

# %%

shot = 1160506015

elecTree = MDSplus.Tree('electrons', shot)

# 9 and 10 are normally the best
sig88ui = elecTree.getNode('\ELECTRONS::TOP.REFLECT.CPCI:DT132_1:INPUT_09').data()
sig88uq = elecTree.getNode('\ELECTRONS::TOP.REFLECT.CPCI:DT132_1:INPUT_10').data()
sig88ut = elecTree.getNode('\ELECTRONS::TOP.REFLECT.CPCI:DT132_1:INPUT_09').dim_of().data()

ci = np.mean(sig88ui)
cq = np.mean(sig88uq)

nl04Node = elecTree.getNode('\ELECTRONS::TOP.TCI.RESULTS:NL_04')
nltime = nl04Node.dim_of().data()
nlData = nl04Node.data()
nl1, nl2 = np.searchsorted(nltime, (0.5, 1.5))

specTree = MDSplus.Tree('spectroscopy', shot)
velNode = specTree.getNode(r'\SPECTROSCOPY::TOP.HIREX_SR.ANALYSIS.W:VEL')
vtime = velNode.dim_of().data()
vData = velNode.data()[0]
v1, v2 = np.searchsorted(vtime, (0.5, 1.5))

gpc = elecTree.getNode('\gpc_t0')

# %% Load data and set times
t1, t2 = np.searchsorted(sig88ut, (0.5,1.5))
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
b, a = signal.butter(2, 1e-4, btype='lowpass')
filt_si = signal.filtfilt(b, a, sig88ui-ci)
filt_sq = signal.filtfilt(b, a, sig88uq-cq)

def getPhiWrapped(si, sq, ci, cq):
    phi = np.arctan2(si-ci, sq-cq)

    phidiff = np.diff(phi)
    phidiff1 = phidiff > np.pi
    phidiff2 = phidiff < -np.pi

    phidiffsum1 = np.cumsum(phidiff1)*-2*np.pi
    phidiffsum2 = np.cumsum(phidiff2)*2*np.pi

    phdiffsum = np.concatenate(([0], phidiffsum1+phidiffsum2))
    return (phi+phdiffsum)


c2 = getPhiWrapped(si, sq, ci, cq)

hsi = filt_si[t1:t2]
hsq = filt_sq[t1:t2]
hsz = hsi+1j*hsq

hc2 = getPhiWrapped(hsi, hsq, 0, 0)

# %% Calculate mean angles and such

nsamp = 512
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
cst = np.mean(szt, axis=-1)

meanAngle = np.angle(np.mean(szr, axis=-1))
rot = np.exp(-1j*(runaway[:, np.newaxis]*szt2 + meanAngle[:, np.newaxis]))

szrRot = szr*rot

angles = np.ravel(np.angle(szrRot))

# %% Plot histograms

t_soc, t_loc = np.searchsorted(sig88ut, (0.5949, 0.9637))
#t_soc, t_loc = np.searchsorted(sig88ut, (0.6, 0.96))
window = 1024*20

soc_data = angles[t_soc-window:t_soc+window]
loc_data = angles[t_loc-window:t_loc+window]

df1, db1 = np.histogram(loc_data, bins=128, density=True)
df2, db2 = np.histogram(soc_data, bins=128, density=True)

dbc1 = (db1[1:] + db1[:-1])/2
dbc2 = (db2[1:] + db2[:-1])/2

plt.figure()
plt.fill_between(dbc1/np.pi, df1*1.0, 0, facecolor='red', alpha=0.5, label='LOC')
plt.fill_between(dbc2/np.pi, df2*1.0, 0, facecolor='blue', alpha=0.5, label='SOC')

plt.xlabel('$\phi$ / $\pi$')
plt.ylabel('probability of observing')
plt.legend()
plt.show()

print np.std(soc_data), np.std(loc_data)

# %%

nsamp2 = 2048*40
times2 = np.mean(np.reshape(st, (-1, nsamp2)), axis=-1)
#angles2 = np.std(np.reshape(angles, (-1, nsamp2)), axis=-1) 
angles2 = -2.0*np.log(np.abs(np.mean(np.reshape(np.ravel(szrRot), (-1, nsamp2)), axis=-1) ))



f, ax1 = plt.subplots()
ax1.plot(times2, angles2, marker='.')

ax2 = ax1.twinx()
ax2.plot(nltime[nl1:nl2], nlData[nl1:nl2], c='g')
#ax2.plot(vtime[v1:v2], vData[v1:v2], c='g', marker='.')

# %%
saw = sat.findSawteeth(gpc.dim_of().data(), gpc.data(), 0.5, 1.49, threshold=-0.05)
sawtimes = gpc.dim_of().data()[saw]
sawData = gpc.data()[saw]

H, xedges, yedges = np.histogram2d(st, -np.angle(np.ravel(sz))/np.pi, bins=(8000, 128))
H = H.T

f, ax1 = plt.subplots()
x,y = np.meshgrid(xedges, yedges)
ax1.pcolormesh(x, y, H, cmap='cubehelix')
ax1.pcolormesh(x, y+2, H, cmap='cubehelix')
ax1.pcolormesh(x, y-2, H, cmap='cubehelix')
ax1.pcolormesh(x, y+4, H, cmap='cubehelix')
ax1.pcolormesh(x, y-4, H, cmap='cubehelix')
ax1.pcolormesh(x, y+6, H, cmap='cubehelix')
ax1.pcolormesh(x, y-6, H, cmap='cubehelix')
ax1.pcolormesh(x, y+8, H, cmap='cubehelix')
ax1.pcolormesh(x, y-8, H, cmap='cubehelix')
ax1.pcolormesh(x, y+10, H, cmap='cubehelix')
ax1.pcolormesh(x, y-10, H, cmap='cubehelix')
ax1.pcolormesh(x, y+12, H, cmap='cubehelix')
ax1.pcolormesh(x, y-12, H, cmap='cubehelix')
ax1.pcolormesh(x, y+14, H, cmap='cubehelix')
ax1.pcolormesh(x, y-14, H, cmap='cubehelix')
ax1.pcolormesh(x, y+16, H, cmap='cubehelix')
ax1.pcolormesh(x, y-16, H, cmap='cubehelix')
ax1.pcolormesh(x, y+18, H, cmap='cubehelix')
ax1.pcolormesh(x, y-18, H, cmap='cubehelix')

#ax2 = ax1.twinx()
ax1.plot(st, hc2/np.pi+10, c='r')
#ax2.plot(nltime[nl1:nl2], nlData[nl1:nl2], c='g')

plt.scatter(sawtimes, np.zeros(sawtimes.shape))

# %%

#sawtimeMed = (sawtimes[1:]+sawtimes[:-1])/2
sawtimeMed = sawtimes
indMed = np.searchsorted(st, sawtimeMed)
window = 1024

indLow = indMed - window/4*5
indHigh = indMed - window/4

timeRange = np.linspace(-1.0, 1.0, num=window)

szr = np.zeros((len(indMed), window), dtype=sz.dtype)
for i in range(len(indMed)):
    szr[i,:] = sz[indLow[i]:indHigh[i]]


def phiVar2(x, t):
    rot = np.exp(-1j*(x[0]*timeRange))
    #rot = np.exp(-1j*(x[0]*timeRange + x[1]*timeRange**2))
    newZ = szr[t,:]*rot
    return -2.0*np.log(np.abs(np.mean(newZ)))

csr = np.zeros(szr.shape[0])
runaway = np.zeros(szr.shape[0])
runCurve = np.zeros(szr.shape[0])


for i in range(szr.shape[0]):
    sol = optimize.minimize(phiVar2, np.array([0.0, 0.0]), args=(i,))
    csr[i] = phiVar2(sol.x, i)
    runaway[i] = sol.x[0]
    runCurve[i] = sol.x[1]
    
runCurve = np.zeros(szr.shape[0])


csra = -2.0*np.log(np.abs(np.mean(szr, axis=-1)))

meanAngle = np.angle(np.mean(szr, axis=-1))
rot = np.exp(-1j*(runaway[:, np.newaxis]*timeRange[np.newaxis, :] + runCurve[:, np.newaxis]*timeRange[np.newaxis,:]**2 + meanAngle[:, np.newaxis]))

szrRot = szr*rot

angles = np.angle(szrRot)

def blockMean(s):
    strim = s
    sre = np.reshape(strim, (-1, 1))
    return np.mean(sre, axis=-1)

f, ax1 = plt.subplots()
ax1.plot(blockMean(sawtimeMed), blockMean(np.std(angles, axis=-1)), marker='.')

#ax2 = ax1.twinx()
#ax2.plot(nltime[nl1:nl2], nlData[nl1:nl2], c='g')

ax3 = ax1.twinx()
ax3.plot(vtime[v1:v2], vData[v1:v2], c='r', marker='.')


# %%

plt.figure()
plt.plot(gpc.dim_of().data(), gpc.data())
plt.scatter(sawtimes, sawData)

plt.figure()
plt.scatter(sawtimes[:-1], np.diff(sawtimes))
