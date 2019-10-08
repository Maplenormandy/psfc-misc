# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 13:12:56 2016

@author: normandy
"""

import profiletools
import gptools
import eqtools

import numpy as np
import scipy

import readline
import MDSplus

import matplotlib.pyplot as plt

import sys
sys.path.append('/home/normandy/git/psfc-misc/Common')

import ShotAnalysisTools as sat

import copy

import scipy.optimize as op
from Collisionality import NustarProfile

import matplotlib.cm as cm

from scipy.signal import medfilt

import scipy.io

plt.close("all")


# %%

plt.close("all")


def hysteresisPlot(shot, axarr, prof, color='b', offset=0):
    elecTree = MDSplus.Tree('electrons', shot)
    
    nl04Node = elecTree.getNode('\ELECTRONS::TOP.TCI.RESULTS:NL_04')
    magTree = MDSplus.Tree('magnetics', shot)
    specTree = MDSplus.Tree('spectroscopy', shot)
    anaTree = MDSplus.Tree('analysis', shot)
    btorNode = magTree.getNode(r'\magnetics::btor')
    q95Node = anaTree.getNode(r'\ANALYSIS::EFIT_AEQDSK:QPSIB')
    velNode = specTree.getNode(r'\SPECTROSCOPY::TOP.HIREX_SR.ANALYSIS.W:VEL')
    
    rfTree = MDSplus.Tree('rf', shot)
    rfNode = rfTree.getNode(r'\rf::rf_power_net')
    
    vtime = velNode.dim_of().data()
    btime = btorNode.dim_of().data()
    nltime = nl04Node.dim_of().data()
    
    
    vlow = np.searchsorted(vtime, 0.5)
    vhigh = np.searchsorted(vtime, 1.5)
    
    vtime = vtime[vlow:vhigh]
    vdata = velNode.data()[0]
    vdata = vdata[vlow:vhigh]
    
    magData = np.interp(vtime, btime, btorNode.data())
    nlData = np.interp(vtime, nltime, nl04Node.data())
    q95data = np.interp(vtime, q95Node.dim_of().data(), q95Node.data())
    rfData = np.interp(vtime, rfNode.dim_of().data(), rfNode.data())
    
    #plt.plot(magData*nldata/0.48, vdata, label=str(shot), marker='.')
    #
    #plt.xlabel('nl_04 * q95')
    #plt.ylabel('vtor')
    
    axarr[0].plot(vtime, vdata+offset, label=str(shot), marker='.', color=color)
    axarr[1].plot(vtime, prof.xminTrace2, label=str(shot), color=color)
    axarr[0].set_ylabel('toroidal velocity [km/s]')
    axarr[1].set_ylabel('line integrated density [m^-2]')
    axarr[1].set_xlabel('time [sec]')
    
    plt.figure(2)
    #plt.plot(prof.numinTrace2, vdata+offset, label=str(shot), color=color, marker='.')
    #plt.ylabel('toroidal velocity [km/s]')
    #plt.xlabel('$\\nu^*$')
    
    
#f, axarr = plt.subplots(2, sharex=True)
#hysteresisPlot(1160506015, axarr, profiles[0], 'b')
#hysteresisPlot(1160506008, axarr, profiles[1], 'g')
#hysteresisPlot(1160506024, axarr, profiles[3], 'c', 6)
#hysteresisPlot(1160506025, axarr, profiles[4], 'r', 6)
#hysteresisPlot(1160506013, axarr, profiles[2], 'm')

# %% PCI Stuff

idlsav = scipy.io.readsav('/home/normandy/1160506007_pci.sav')

t = idlsav.spec.t[0]
f = idlsav.spec.f[0]
k = idlsav.spec.k[0]
s = idlsav.spec.spec[0]
shot = idlsav.spec.shot[0]

# %%

kp = np.sum(s[:,:,18:24], axis=2)
kn = np.sum(s[:,:,10:16], axis=2)

pp = np.sum(s[:,:,10:24], axis=2)
total = np.sum(pp[:,:], axis=1)

kall = np.concatenate((kn[:,299:29:-1], kp[:,30:300]), axis=1)
fall = np.concatenate((-f[299:29:-1], f[30:300]))

plt.figure()
plt.pcolormesh(t, fall, np.log(kall.T), cmap='cubehelix')

totalp = np.sum(kp[:,30:300], axis=1)
totaln = np.sum(kn[:,30:300], axis=1)

plt.figure()
#plt.plot(t, totalp)
#plt.plot(t, totaln)
plt.plot(t, total)

# %%

plt.figure()
elecTree = MDSplus.Tree('electrons', shot)
nl04Node = elecTree.getNode('\ELECTRONS::TOP.TCI.RESULTS:NL_04')
nlData = np.interp(t, nl04Node.dim_of().data(), nl04Node.data())
t0, t1 = np.searchsorted(t, (0.8, 1.1))
plt.scatter(nlData[:t0], total[:t0], marker='v', c='b')
plt.scatter(nlData[t0:t1], total[t0:t1], marker='^', c='r')
plt.scatter(nlData[t1:], total[t1:], marker='v', c='b')


# %%

f0, f1, f2, f3 = np.searchsorted(f, (50, 200, 300, 700))

svk = np.sum(s[:,f0:f1,:], axis=1)
svk2 = np.sum(s[:,f2:f3,:], axis=1)

#plt.figure()
#plt.pcolormesh(t, k, svk.T, cmap='cubehelix')
#plt.figure()
#plt.pcolormesh(t, k, svk2.T, cmap='cubehelix')

vkp = np.sum(svk[:,17:23], axis=1)
vkn = np.sum(svk[:,10:16], axis=1)

vkp2 = np.sum(svk2[:,17:23], axis=1)
vkn2 = np.sum(svk2[:,10:16], axis=1)

plt.figure()
plt.plot(t, (vkp-vkn)/(vkp+vkn)*2, marker='.')
plt.plot(t, (vkp2-vkn2)/(vkp2+vkn2)*2, marker='.')
plt.axhline(color='r', ls='--')

plt.figure()
plt.plot(t, (vkp+vkn)/2, marker='.', c='b')

plt.figure()
plt.plot(t, (vkp2+vkn2)/2, marker='.', c='g')

"""
def plotTiRotation(shot, tmin=0.5, tmax=1.45, color='b', delay=5, offset=0):
    specTree = MDSplus.Tree('spectroscopy', shot)
    magTree = MDSplus.Tree('magnetics', shot)
    velNode = specTree.getNode(r'\SPECTROSCOPY::TOP.HIREX_SR.ANALYSIS.W:VEL')
    ipNode = magTree.getNode(r'\magnetics::ip')
    
    vtime = velNode.dim_of().data()
    
    vlow = np.searchsorted(vtime, tmin)
    vhigh = np.searchsorted(vtime, tmax)+2
    
    vtime = vtime[vlow:vhigh]
    vdata = velNode.data()[0]
    vdata = vdata[vlow:vhigh]
    
    tdata = tiNode.data()[0]
    tdata = tdata[vlow:vhigh]
    
    ipData = np.interp(vtime, ipNode.dim_of().data(), ipNode.data())
    
    offset = ((shot%100)-7.0)/(25.0-7.0)*6.0
    print shot, offset
    
    plt.scatter(tdata[:-delay], vdata[delay:]+offset, label=str(shot), c=color, marker='.', lw=0)
    plt.ylabel('toroidal velocity [km/s]')
    plt.xlabel('emissivity-averaged Ti [keV]')
"""
