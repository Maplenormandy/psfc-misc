# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 21:48:02 2016

@author: normandy
"""

import numpy as np
import matplotlib.pyplot as plt

import readline
import MDSplus

readline


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

nodeA = specTree.getNode('r\SPECTROSCOPY::TOP.HIREXSR.ANALYSIS1.HELIKE.PROFILES.Z')
nodeB = specTree.getNode('r\SPECTROSCOPY::TOP.HIREXSR.ANALYSIS.HLIKE.PROFILES.LYA1')

dataA = ThacoData(nodeA)
dataB = ThacoData(nodeB)

t0 = 1.1

indexA = np.searchsorted(dataA.time, t0)
indexB = np.searchsorted(dataB.time, t0)

plt.figure()
plt.plot(indexA.rho, indexA.pro[3, indexA, :])
plt.plot(indexB.rho, indexB.pro[3, indexB, :])


