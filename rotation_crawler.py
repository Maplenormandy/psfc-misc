# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 19:57:23 2016

@author: normandy
"""

import numpy as np

import readline
import MDSplus

import matplotlib.pyplot as plt

from scipy.signal import find_peaks_cwt

readline

#f = open('allshots.txt', 'r')
#allshots = (int(line) for line in f)
allshots = [1120106020]

for shot in allshots:
    specTree = MDSplus.Tree('spectroscopy', shot)

    zvelNode = specTree.getNode('\SPECTROSCOPY::TOP.HIREX_SR.ANALYSIS.Z.VEL')
    zvel = zvelNode.data()[0]
    ztime = zvelNode.dim_of().data()
    zvel = zvel[ztime > 0.65]
    ztime = ztime[ztime > 0.65]
    
    zvel = zvel[~np.isnan(zvel)]    
    
    zlims = np.percentile(zvel, [25, 50, 75])
    zrange = zlims[2] - zlims[0]
    zbins = np.linspace(zlims[0]-zrange/2, zlims[2]+zrange/2, 35)
    plt.hist(zvel, bins=zbins)
    
    zhist = np.histogram(zlims, zbins)[0]
    
    print find_peaks_cwt(zhist, np.arange(3,16))