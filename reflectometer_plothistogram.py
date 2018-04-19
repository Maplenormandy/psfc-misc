# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 16:02:10 2017

@author: normandy
"""

import readline
import MDSplus

import numpy as np
import matplotlib.pyplot as plt

from scipy import signal
from scipy import stats
from scipy import optimize

# %%

elecTree = MDSplus.Tree('electrons', 1160506007)

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



b, a = signal.butter(4, 0.001, btype='lowpass')
si2 = signal.filtfilt(b, a, si)
sq2 = signal.filtfilt(b, a, sq)


ci2 = np.mean(si)
cq2 = np.mean(sq)

#si = np.mean(np.reshape(si, (-1, 1000)), 1)
#sq = np.mean(np.reshape(sq, (-1, 1000)), 1)
#st = np.mean(np.reshape(st, (-1, 1000)), 1)



"""
plt.figure()
plt.plot(si-ci, sq-cq)
plt.axis('square')
plt.axhline(ls='--')
plt.axvline(ls='--')
"""

def getPhiWrapped(si, sq, ci, cq):
    phi = np.arctan2(si-ci, sq-cq)

    phidiff = np.diff(phi)
    phidiff1 = phidiff > np.pi
    phidiff2 = phidiff < -np.pi
    
    phidiffsum1 = np.cumsum(phidiff1)*-2*np.pi
    phidiffsum2 = np.cumsum(phidiff2)*2*np.pi
    
    phdiffsum = np.concatenate(([0], phidiffsum1+phidiffsum2))
    return (phi+phdiffsum)

c1 = getPhiWrapped(si2, sq2, ci2, cq2)

"""
plt.figure()
plt.scatter(st, c1/2/np.pi, marker='o', c=np.sqrt((si-ci)**2+(sq-cq)**2), cmap='inferno_r')
#plt.plot(st, ((si-ci)**2 + (sq-cq)**2)*100)
#plt.plot(st, phi+phdiffsum)
plt.axhline(np.pi, ls='--')
plt.axhline(-np.pi, ls='--')
"""
#plt.hist(phi+phdiffsum,bins=50)'



pbins = np.linspace(-1.0,1.0, num=32)
rbins = np.linspace(-0.4,0.4,num=32)


xe=rbins
ye=rbins
h, xe, ye = np.histogram2d(si2-ci2, sq2-cq2, bins=(xe,ye))
xp, yp = np.meshgrid(xe, ye)
plt.figure()
plt.pcolormesh(xp, yp, h, cmap='cubehelix')


"""
plt.figure()
plt.scatter(si-ci, sq-cq, c=st, marker='+', cmap='cubehelix')
"""

plt.axis('square')
plt.xlim([-0.4,0.4])
plt.ylim([-0.4,0.4])
plt.axhline(ls='--')
plt.axvline(ls='--')
plt.title('LOC')

plt.figure()
plt.plot(st, c1/2/np.pi, marker='.')

# %% Higher freq components

shsi = si-si2
shsq = sq-sq2

b, a = signal.butter(4, [0.3, 0.7], btype='bandpass')
#b, a = signal.butter(4, 0.1, btype='highpass')
sshsi = signal.filtfilt(b, a, si)
sshsq = signal.filtfilt(b, a, sq)

t3, t4 = np.searchsorted(st, (0.5,1.5))

hsi = sshsi[t3:t4]
hsq = sshsq[t3:t4]
hst = st[t3:t4]


ci3 = np.mean(hsi)
cq3 = np.mean(hsq)

c2 = getPhiWrapped(hsi, hsq, 0, 0)

pbins = np.linspace(-1.0,1.0, num=32)
rbins = np.linspace(-0.2,0.2,num=32)
xe=rbins
ye=rbins
h, xe, ye = np.histogram2d(hsi, hsq, bins=(xe,ye))
xp, yp = np.meshgrid(xe, ye)
plt.figure()
plt.pcolormesh(xp, yp, h, cmap='cubehelix')
plt.plot(hsi, hsq)
plt.plot(si[t3:t4]-ci3, sq[t3:t4]-cq3)


plt.axis('square')
plt.xlim([-0.2,0.2])
plt.ylim([-0.2,0.2])
plt.axhline(ls='--')
plt.axvline(ls='--')
plt.title('LOC')

phasedrift = (c2[-1]-c2[0]) * (hst-hst[0]) / (hst[-1]-hst[0])

plt.figure()
plt.plot(hst, (c2)/2/np.pi, marker='.')

plt.figure()
c3 = signal.filtfilt(b, a, c2)
plt.plot(hst, c3)


def model(t, coeffs):
    return stats.norm.pdf(t, loc=coeffs[0], scale=coeffs[1])
def residuals(coeffs, y, t):
    return y - model(t, coeffs)


# 1160506007 0.57-0.63 or 0.93-0.99
# 1160506015 0.67-0.73 or 0.87-0.93
t5, t6 = np.searchsorted(hst, (0.57,0.63))
#t5, t6 = np.searchsorted(hst, (0.67,0.73))

c3mean = np.mean(c3[t5:t6])
c3std = np.std(c3[t5:t6])
normx = np.linspace(-3,3)*c3std+c3mean

counts, bins = np.histogram(c3[t5:t6], bins=64, normed=True)
binsPlot = (bins[1:]+bins[:-1])/2.0

plt.figure()
plt.plot(binsPlot, counts, marker='.', c='b')
plt.fill_between(binsPlot, counts, facecolor='blue', alpha=0.5)
#plt.plot(normx, stats.norm.pdf(normx, loc=c3mean, scale=c3std), c='b', lw=2)
#plt.axvline()
#plt.title('SOC')

kap2 = stats.kstat(c3[t5:t6], n=2)
kap4 = stats.kstat(c3[t5:t6], n=4)
print 'SOC', kap2, kap4/(kap2**2)
print stats.kurtosistest(c3[t5:t6]).statistic
print stats.skew(c3[t5:t6])

t5, t6 = np.searchsorted(hst, (0.93,0.99))
#t5, t6 = np.searchsorted(hst, (0.87,0.93))

c3mean = np.mean(c3[t5:t6])
c3std = np.std(c3[t5:t6])
normx = np.linspace(-4,4)*c3std+c3mean
    
counts, bins = np.histogram(c3[t5:t6], bins=64, normed=True)
binsPlot = (bins[1:]+bins[:-1])/2.0

inCounts = np.concatenate((counts[:12], counts[52:]))
inBins = np.concatenate((binsPlot[:12], binsPlot[52:]))
inStats, flag = optimize.leastsq(residuals, np.array([c3mean, c3std]), args=(inCounts, inBins))

outCounts = counts[26:(64-26)]
outBins = binsPlot[26:(64-26)]
outStats, flag = optimize.leastsq(residuals, np.array([c3mean, c3std]), args=(outCounts, outBins))

#plt.figure()
#plt.hist(c3[t5:t6], bins=50, normed=True)
plt.plot(binsPlot, counts, marker='.', c='r')
plt.fill_between(binsPlot, counts, facecolor='red', alpha=0.5)
#plt.plot(normx, stats.norm.pdf(normx, loc=c3mean, scale=c3std), c='r', lw=2)
#plt.axvline()
#plt.title('LOC')

kap2 = stats.kstat(c3[t5:t6], n=2)
kap4 = stats.kstat(c3[t5:t6], n=4)
print 'LOC', kap2, kap4/(kap2**2)
print stats.kurtosistest(c3[t5:t6]).statistic
print stats.skew(c3[t5:t6])

# %% Fitting Gaussian statistics

"""
t5, t6 = np.searchsorted(hst, (0.67,0.73))

counts, bins = np.histogram(c3[t5:t6], bins=64, normed=True)
binsPlot = (bins[1:]+bins[:-1])/2.0

c3mean = np.mean(c3[t5:t6])
c3std = np.std(c3[t5:t6])

def model(t, coeffs):
    return stats.norm.pdf(t, loc=coeffs[0], scale=coeffs[1])
def residuals(coeffs, y, t):
    return y - model(t, coeffs)
    
    
reducedCounts = np.concatenate((counts[:12], counts[52:]))
reducedBins = np.concatenate((binsPlot[:12], binsPlot[52:]))
newStats, flag = optimize.leastsq(residuals, np.array([c3mean, c3std]), args=(counts, binsPlot))

plt.figure()
plt.plot(binsPlot, counts, marker='.')
#plt.semilogy(-binsPlot, counts, marker='.')
plt.plot(binsPlot, stats.norm.pdf(binsPlot, loc=newStats[0], scale=newStats[1]))
#plt.semilogy(reducedBins, reducedCounts, marker='.')
"""

# %% Statistics as a function of time

perSlice = 20000
nc = c3.size/perSlice
cCut = c3[0:(nc*perSlice)]
tCut = st[0:(nc*perSlice)]

cCut = cCut.reshape(nc,-1)
tCut = tCut.reshape(nc,-1)
tCut = np.mean(tCut, axis=1)

k4Cut = np.zeros(nc)
k2Cut = np.zeros(nc)

for i in range(nc):
    k2Cut[i] = stats.kstat(cCut[i,:], n=2)
    k4Cut[i] = stats.kstat(cCut[i,:], n=4)

f, (ax1,ax2) = plt.subplots(2, 1, sharex=True)
ax1.plot(tCut, k2Cut, c='b', marker='.')
ax2.plot(tCut, k4Cut, c='b', marker='.')
plt.axhline(ls='--', c='r')

# %% Coherence

"""
t7 = np.searchsorted(hst, 0.85)
c4u = c3u[t7:t7+2**17]
c4l = c3l[t7:t7+2**17]
print hst[t7+2**17]-hst[t7]
"""
#f, Cxy = signal.coherence(c4u, c4l, fs=2e6)
#plt.figure()
#plt.semilogy(f, Cxy)

# %% Spectrogram

#f, t, Sxx = signal.spectrogram(c3u, fs=2e6, nperseg=4096)
#plt.semilogy(f, Sxx[:,220])
#print t[220]+0.5