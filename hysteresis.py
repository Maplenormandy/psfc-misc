# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 13:31:51 2016

@author: normandy
"""

import numpy as np

import matplotlib.pyplot as plt
import matplotlib

import readline
import MDSplus


def plotBtHysteresis(shot):
    elecTree = MDSplus.Tree('electrons', shot)
    
    nl04Node = elecTree.getNode('\ELECTRONS::TOP.TCI.RESULTS:NL_04')
    magTree = MDSplus.Tree('magnetics', shot)
    specTree = MDSplus.Tree('spectroscopy', shot)
    btorNode = magTree.getNode(r'\magnetics::btor')
    velNode = specTree.getNode(r'\SPECTROSCOPY::TOP.HIREX_SR.ANALYSIS.W:VEL')
    
    rfTree = MDSplus.Tree('rf', shot)
    rfNode = rfTree.getNode(r'\rf::rf_power_net')
    
    vtime = velNode.dim_of().data()
    btime = btorNode.dim_of().data()
    nltime = nl04Node.dim_of().data()
    
    
    vlow = np.searchsorted(vtime, 0.7)
    vhigh = np.searchsorted(vtime, 1.5)+2
    
    vtime = vtime[vlow:vhigh]
    vdata = velNode.data()[0]
    vdata = vdata[vlow:vhigh]
    
    magData = np.interp(vtime, btime, btorNode.data())
    nldata = np.interp(vtime, nltime, nl04Node.data())
    rfData = np.interp(vtime, rfNode.dim_of().data(), rfNode.data())
    
    #plt.plot(magData*nldata/0.48, vdata, label=str(shot), marker='.')
    plt.plot(magData, vdata, label=str(shot), marker='.')
    #plt.plot(rfData, vdata, label=str(shot))
    
    
def plotBrightness(shot):
    elecTree = MDSplus.Tree('electrons', shot)
    
    nl04Node = elecTree.getNode('\ELECTRONS::TOP.TCI.RESULTS:NL_04')
    magTree = MDSplus.Tree('magnetics', shot)
    specTree = MDSplus.Tree('spectroscopy', shot)
    emissNode = specTree.getNode(r'\SPECTROSCOPY::TOP.HIREX_SR.ANALYSIS.Z:INT')
    
    gpcNode = elecTree.getNode(r'\ELECTRONS::gpc2_te0')
    
    etime = emissNode.dim_of().data()
    
    elow = np.searchsorted(etime, 0.5)
    ehigh = np.searchsorted(etime, 1.5)+2
    
    etime = etime[elow:ehigh]
    edata = emissNode.data()[0]
    edata = edata[elow:ehigh]
    
    nldata = np.interp(etime, nl04Node.dim_of().data(), nl04Node.data())
    
    fig, ax1 = plt.subplots()
    
    ax1.plot(etime, np.log(edata) - np.log(nldata))
    
    ax2 = ax1.twinx()
    
    ax2.plot(gpcNode.dim_of().data(), gpcNode.data())
    
    
    
    
#plotBtHysteresis(1160506007)
plotBtHysteresis(1160512025)
plotBtHysteresis(1160512026)
#plotBtHysteresis(1160506024)
#plotBtHysteresis(1160506025)
#plotBtHysteresis(1160506013)
#plotBtHysteresis(1160506001)
#plotBtHysteresis(1160506007)
#plotBtHysteresis(1160506017)
plt.legend()
    
#plotBrightness(1150903021)
