# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 21:48:02 2016

@author: normandy
"""

import numpy as np
import matplotlib.pyplot as plt

# My python installation yells at me if I don't have this for some reason
import readline
readline
import MDSplus





class ThacoData:
    def __init__(self, node):

        proNode = node.getNode('PRO')
        rhoNode = node.getNode('RHO')
        perrNode = node.getNode('PROERR')

        rpro = proNode.data()
        rrho = rhoNode.data()
        rperr = perrNode.data()
        rtime = rhoNode.dim_of()

        goodTimes = (rtime > 0).sum()

        self.time = rtime.data()[:goodTimes]
        self.rho = rrho[0,:] # Assume unchanging rho bins
        self.pro = rpro[:,:goodTimes,:len(self.rho)]
        self.perr = rperr[:,:goodTimes,:len(self.rho)]


specTree = MDSplus.Tree('spectroscopy', 1101014029)

# Load the nodes associated with inverted profile data
nodeA = specTree.getNode(r'\SPECTROSCOPY::TOP.HIREXSR.ANALYSIS1.HELIKE.PROFILES.Z')
nodeB = specTree.getNode(r'\SPECTROSCOPY::TOP.HIREXSR.ANALYSIS.HLIKE.PROFILES.LYA1')

# Load the actual data
dataA = ThacoData(nodeA)
dataB = ThacoData(nodeB)

t0 = 1.01
# Look for the index corresponding to the given time point
indexA = np.searchsorted(dataA.time, t0)
indexB = np.searchsorted(dataB.time, t0)

# Plot the Ti
plt.figure()
plt.errorbar(dataA.rho, dataA.pro[3,indexA,:], yerr=dataA.perr[3,indexA,:])
plt.errorbar(dataB.rho, dataB.pro[3,indexB,:], yerr=dataB.perr[3,indexB,:])
plt.ylabel('Ti [keV]')
plt.xlabel('normalized poloidal flux')
plt.show()
