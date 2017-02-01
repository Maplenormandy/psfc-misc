# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 18:33:25 2016

@author: normandy
"""

import numpy as np

import readline
import MDSplus

import matplotlib.pyplot as plt


class ThacoData:
    def __init__(self, thtNode, shot=None, tht=None):
        if (shot != None):
            self.shot = shot
            self.specTree = MDSplus.Tree('spectroscopy', shot)

            if (tht == 0):
                self.tht = ''
            else:
                self.tht = str(tht)

            self.thtNode = self.specTree.getNode('\SPECTROSCOPY::TOP.HIREXSR.ANALYSIS' + self.tht + '.HELIKE.PROFILES.Z')
        else:
            self.thtNode = thtNode

        proNode = self.thtNode.getNode('PRO')
        perrNode = self.thtNode.getNode('PROERR')
        rhoNode = self.thtNode.getNode('RHO')

        rpro = proNode.data()
        rperr = perrNode.data()
        rrho = rhoNode.data()
        rtime = rhoNode.dim_of()

        goodTimes = (rtime > 0).sum()

        self.time = rtime.data()[:goodTimes]
        self.rho = rrho[0,:] # Assume unchanging rho bins
        self.pro = rpro[:,:goodTimes,:len(self.rho)]
        self.perr = rperr[:,:goodTimes,:len(self.rho)]
        
        
specTree = MDSplus.Tree('spectroscopy', 1100317005)
rotNode = specTree.getNode('\SPECTROSCOPY::TOP.HIREX_SR.ANALYSIS.Z:VEL')

td = ThacoData(None, 1100317005, 0)

elecTree = MDSplus.Tree('electrons', 1100317005)
tciNode = elecTree.getNode('\ELECTRONS::TOP.TCI.RESULTS')
densNode = tciNode.getNode('nl_04')

f, axarr = plt.subplots(2,1, sharex=True)
axarr[0].plot(rotNode.dim_of().data(), rotNode.data()[0])
axarr[0].scatter(rotNode.dim_of().data()[67], rotNode.data()[0][67], c='b')
axarr[0].scatter(rotNode.dim_of().data()[71], rotNode.data()[0][71], c='g')
axarr[0].scatter(rotNode.dim_of().data()[75], rotNode.data()[0][75], c='r')
axarr[1].plot(densNode.dim_of().data(), densNode.data()/1e20)
axarr[0].axhline(y=0, c='r', ls=':')

axarr[0].set_ylabel('$V_t$ [km/s]')
axarr[1].set_ylabel('nl_04 [$10^{20} m^{-3}$]')

yticks = axarr[1].yaxis.get_major_ticks()
yticks[-1].label1.set_visible(False)

axarr[1].set_xlabel('time [s]')

plt.tight_layout()



plt.figure()
plt.errorbar(np.sqrt(td.rho[1:-4]), td.pro[1,37,1:-4], yerr=td.perr[1,37,1:-4])
plt.errorbar(np.sqrt(td.rho[1:-4]), td.pro[1,41,1:-4], yerr=td.perr[1,41,1:-4])
plt.errorbar(np.sqrt(td.rho[1:-4]), td.pro[1,45,1:-4], yerr=td.perr[1,45,1:-4])



