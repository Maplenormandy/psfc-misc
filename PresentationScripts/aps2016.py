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

# %% Ti vs. Rotation

def plotTiRotation(shot, tmin=0.5, tmax=1.45, color='b', delay=5, offset=0):
    specTree = MDSplus.Tree('spectroscopy', shot)
    magTree = MDSplus.Tree('magnetics', shot)
    velNode = specTree.getNode(r'\SPECTROSCOPY::TOP.HIREX_SR.ANALYSIS.W:VEL')
    tiNode = specTree.getNode(r'\SPECTROSCOPY::TOP.HIREX_SR.ANALYSIS.W:TI')
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
    
    offset = ((shot%100)-7.0)/(25.0-7.0)*5.33
    print shot, offset
    
    plt.scatter(tdata[:-delay], vdata[delay:]+offset, label=str(shot), c=color, marker='.', lw=0)
    plt.ylabel('toroidal velocity [km/s]')
    plt.xlabel('emissivity-averaged Ti [keV]')
    

plt.figure(1)
plotTiRotation(1160506017, 0.5, 1.4, 'b')
plotTiRotation(1160506018, 0.5, 1.4, 'g')
plotTiRotation(1160506023, 0.5, 1.4, 'r', 5)
plotTiRotation(1160506007, 0.6, 1.3, 'c', 5)
plotTiRotation(1160506014, 0.6, 1.3, 'y')
plotTiRotation(1160506015, 0.6, 1.3, 'm')
plotTiRotation(1160506019, 0.6, 1.3, 'k', 5)
plotTiRotation(1160506020, 0.6, 1.3, '0.5', 5)


"""
#plotTiRotation(1160506007, 0.5, 1.45, 'r')
#plotTiRotation(1160506008, 0.5, 1.45, 'g')
plotTiRotation(1160506025, 0.75, 1.45, 'b', 4)
"""

"""
plotTiRotation(1150901016, 0.5, 1.45, 'b')
plotTiRotation(1150901017, 0.5, 1.45, 'g')
plotTiRotation(1150901020, 0.5, 1.45, 'r')
plotTiRotation(1150901021, 0.5, 1.45, 'c')
plotTiRotation(1150901022, 0.5, 1.45, 'y')
plotTiRotation(1150901023, 0.5, 1.45, 'm')
plotTiRotation(1150901024, 0.5, 1.45, 'k')
"""

# %% ne hysteresis

plt.close("all")

def hysteresisPlot(shot, axarr, color='b', offset=0):
    elecTree = MDSplus.Tree('electrons', shot)
    
    nl04Node = elecTree.getNode('\ELECTRONS::TOP.TCI.RESULTS:NL_04')
    nl10Node = elecTree.getNode('\ELECTRONS::TOP.TCI.RESULTS:NL_10')
    nl01Node = elecTree.getNode('\ELECTRONS::TOP.TCI.RESULTS:NL_01')
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
    
    
    vlow = np.searchsorted(vtime, 0.55)
    vhigh = np.searchsorted(vtime, 1.25)+2
    
    vtime = vtime[vlow:vhigh]
    vdata = velNode.data()[0]
    vdata = vdata[vlow:vhigh]
    
    nlDataRaw = nl04Node.data()
    
    nlDataRaw01 = nl01Node.data()
    nlDataRaw10 = nl10Node.data()
    
    magData = np.interp(vtime, btime, btorNode.data())
    nlData = np.interp(vtime, nltime, nlDataRaw)
    
    offset = ((shot%100)-7.0)/(25.0-7.0)*6
    
    nlDataAvg = np.zeros(vdata.shape)
    nlPeaking = np.zeros(vdata.shape)
    for i in range(len(vdata)):
        low = np.searchsorted(nltime, vtime[i]-0.0035)
        high = np.searchsorted(nltime, vtime[i]+0.0035)
        nlDataAvg[i] = np.average(nlDataRaw[low:high])
        nlPeaking[i] = np.average(nlDataRaw01[low:high]) / np.average(nlDataRaw10[low:high])
    
    q95data = np.interp(vtime, q95Node.dim_of().data(), q95Node.data())
    rfData = np.interp(vtime, rfNode.dim_of().data(), rfNode.data())
    
    #plt.plot(magData*nldata/0.48, vdata, label=str(shot), marker='.')
    #
    #plt.xlabel('nl_04 * q95')
    #plt.ylabel('vtor')
    
    axarr[1].plot(vtime, vdata+offset, label=str(shot), marker='.', color=color)
    axarr[0].plot(vtime, nlData/6e19, label=str(shot), color=color)
    axarr[1].set_ylabel('v$\phi$ [km/s]')
    axarr[0].set_ylabel('n$_e$ [10$^{20}$/m$^{3}$]')
    axarr[1].set_xlabel('time [sec]')
    axarr[1].set_xlim([0.58, 1.42])
    
    plt.figure(2)
    plt.plot(nlData/6e19, vdata+offset, label=str(shot), color=color, marker='.')
    #plt.plot(nlData, vdata+offset, label=str(shot), color=color, marker='.')
    plt.ylabel('v$\phi$ [km/s]')
    #plt.xlabel('P$_{RF}$ [MW]')
    plt.xlabel('n$_e$ [10$^{20}$/m$^{3}$]')
    
    print np.trapz(vdata+offset, nlData/6e19/1.1)
    
    #plt.figure(3)
    #plt.plot(rfNode.dim_of().data(), rfNode.data(), label=str(shot), color=color)
    #plt.ylabel('P$_{RF}$ [MW]')
    
    
    #plt.plot(rfData, vdata, label=str(shot))

f, axarr = plt.subplots(2, sharex=True)
#hysteresisPlot(1160506007, axarr, 'b')
#hysteresisPlot(1160506008, axarr, 'g')
#hysteresisPlot(1160506024, axarr, 'r', 7)
#hysteresisPlot(1160506025, axarr, 'c', 6)
#hysteresisPlot(1160506013, axarr, 'm')

#hysteresisPlot(1160506009, axarr, 'b')
#hysteresisPlot(1160506010, axarr, 'g')

hysteresisPlot(1160506007, axarr, 'b')
#hysteresisPlot(1160506008, axarr, 'g')
#hysteresisPlot(1160506024, axarr, 'r')

#hysteresisPlot(1160506015, axarr, 'b')

# %% Calculating collisionalities

shots = [1160506007, 1160506009, 1160506015]
profiles = map(lambda x: NustarProfile(x, 0.4, 1.6), shots)

tnefit = np.arange(0.5, 1.5, 0.04)
tpfit = np.arange(0.5035, 1.5, 0.01)

def calcNuminMedTrace(p):
    def unpack(f):
        return lambda x: f(x)[0]

    p.numinTrace2 = np.zeros(len(p.tfits))
    p.xminTrace2 = np.zeros(len(p.tfits))        
        
    for j in range(len(p.tfits)):
        res = op.minimize_scalar(unpack(p.collMed[j]), bounds=[0.1, 0.95], method='bounded')
        p.numinTrace2[j] = res.fun
        p.xminTrace2[j] = res.x

for p in profiles:
    p.fitNe(tnefit)
    p.evalProfile(tpfit)
    p.calcMinTrace()



for p in profiles:
    calcNuminMedTrace(p)

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
    
    
f, axarr = plt.subplots(2, sharex=True)
hysteresisPlot(1160506015, axarr, profiles[0], 'b')
#hysteresisPlot(1160506008, axarr, profiles[1], 'g')
#hysteresisPlot(1160506024, axarr, profiles[3], 'c', 6)
#hysteresisPlot(1160506025, axarr, profiles[4], 'r', 6)
#hysteresisPlot(1160506013, axarr, profiles[2], 'm')

# %% PCI Stuff

idlsav = scipy.io.readsav('/home/normandy/1160506015_pci.sav')

t = idlsav.spec.t[0]
f = idlsav.spec.f[0]
k = idlsav.spec.k[0]
s = idlsav.spec.spec[0]
shot = idlsav.spec.shot[0]

# %%

kp = np.sum(s[:,:,18:24], axis=2)
kn = np.sum(s[:,:,10:16], axis=2)

kall = np.concatenate((kn[:,299:29:-1], kp[:,30:300]), axis=1)
fall = np.concatenate((-f[299:29:-1], f[30:300]))

plt.figure()
plt.pcolormesh(t, fall, np.log(kall.T), cmap='cubehelix')

totalp = np.sum(kp[:,30:300], axis=1)
totaln = np.sum(kn[:,30:300], axis=1)

plt.figure()
plt.plot(t, totalp)
plt.plot(t, totaln)

# %%

f0, f1, f2, f3 = np.searchsorted(f, (100, 200, 300, 700))

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
plt.plot(t, (vkp-vkn)/(vkp+vkn)/2, marker='.')
plt.plot(t, (vkp2-vkn2)/(vkp2+vkn2)/2, marker='.')
plt.axhline(color='r', ls='--')

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
