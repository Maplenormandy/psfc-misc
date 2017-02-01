# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 00:26:05 2016

@author: normandy
"""

import numpy as np

import readline
import MDSplus

import sys

import matplotlib.pyplot as plt
import matplotlib

readline

font = {'family': 'normal', 'size': 18}
matplotlib.rc('font', **font)

class FrceceData:
    def __init__(self, elecTree, numChannels):
        self.time = [None]*numChannels
        self.temp = [None]*numChannels
        self.rmid = [None]*numChannels

        print "Loading ECE channels:",

        for i in range(1, numChannels+1):
            tempNode = elecTree.getNode('\ELECTRONS::TE_HRECE%02d' % i)
            rmidNode = elecTree.getNode('\ELECTRONS::RMID_HRECE%02d' % i)

            rtimes = rmidNode.dim_of().data()
            ttimes = tempNode.dim_of().data()

            self.time[i-1] = ttimes
            self.temp[i-1] = tempNode.data()
            self.rmid[i-1] = np.interp(self.time[i-1], rtimes, rmidNode.data().flatten())

            print i,
            sys.stdout.flush()

        print "done"

        self.time = np.array(self.time)
        self.temp = np.array(self.temp)
        self.rmid = np.array(self.rmid)
        
"""
elecTree = MDSplus.Tree('electrons', 1120106020)
#frc = FrceceData(elecTree, 16)

plt.figure()
plt.plot(frc.time[0,:], frc.temp[0,:], c='b')
plt.plot(frc.time[3,:], frc.temp[3,:], c='b')
plt.plot(frc.time[6,:], frc.temp[6,:], c='b')
plt.plot(frc.time[9,:], frc.temp[9,:], c='b')
plt.plot(frc.time[12,:], frc.temp[12,:], c='b')
plt.plot(frc.time[15,:], frc.temp[15,:], c='b')

plt.ylabel('Te [keV]')
plt.xlabel('time [sec]')
"""

class HirexsrSpec:
    def __init__(self, shot, tht=1):
        if (shot != None):
            self.shot = shot
            self.specTree = MDSplus.Tree('spectroscopy', shot)

            if (tht == 0):
                self.tht = ''
            else:
                self.tht = str(tht)

            self.thtNode = self.specTree.getNode('\SPECTROSCOPY::TOP.HIREXSR.ANALYSIS' + self.tht + '.HELIKE.SPEC')

        specNode = self.thtNode.getNode('SPECBR')
        lamNode = self.thtNode.getNode('LAM')
        rbr = specNode.data()
        rtime = specNode.dim_of(1).data()
        rlam = lamNode.data()

        goodChans = (rbr[0,0,:] > 0).sum()
        goodTimes = (rtime > 0).sum()

        self.time = rtime[:goodTimes]
        self.br = rbr[:,:goodTimes,:goodChans]
        self.lam = rlam[:,:goodTimes,:goodChans]
        """
        self.bg = np.percentile(self.br, 10, axis=0).T

        self.tp = centerScale(self.time)
        self.cp = centerScale(range(self.br.shape[2]))

        self.tplot, self.cplot = np.meshgrid(self.tp, self.cp)

        self.fig = plt.figure(figsize=(22,4))
        gs = gridspec.GridSpec(1,1)
        self.ax1 = self.fig.add_subplot(gs[0])
        self.cax1 = self.ax1.pcolormesh(self.tplot, self.cplot, self.bg, cmap='cubehelix')
        self.fig.colorbar(self.cax1)

        self.fig.suptitle('Shot ' + str(self.shot) + ' B_lambda background')

        self.fig.canvas.draw()
        plt.show(block=False)
        """

"""
hs = HirexsrSpec(1150903021)
print hs.time[7]
plt.semilogy(hs.lam[:,7,36], hs.br[:,7,36], marker='+', label=str(hs.time[7])+'s')
plt.semilogy(hs.lam[:,7,36], hs.br[:,8,36], linestyle='--', label=str(hs.time[8])+'s')
plt.semilogy(hs.lam[:,7,36], hs.br[:,9,36], linestyle='-.', label=str(hs.time[9])+'s')
plt.legend(loc='upper right', fontsize=18)
plt.xlabel('Wavelength [Ang]')
plt.ylabel('$B_\lambda$ [-]')
"""

class ThacoData:
    def __init__(self, heNode, hyNode):
        
        heproNode = heNode.getNode('PRO')
        herhoNode = heNode.getNode('RHO')
        heperrNode = heNode.getNode('PROERR')

        herpro = heproNode.data()
        herrho = herhoNode.data()
        herperr = heperrNode.data()
        hertime = herhoNode.dim_of()

        hegoodTimes = (hertime > 0).sum()

        self.hetime = hertime.data()[:hegoodTimes]
        self.herho = herrho[0,:] # Assume unchanging rho bins
        self.hepro = herpro[:,:hegoodTimes,:len(self.herho)]
        self.heperr = herperr[:,:hegoodTimes,:len(self.herho)]
        
        if hyNode != None:
            hyproNode = hyNode.getNode('PRO')
            hyrhoNode = hyNode.getNode('RHO')
            hyperrNode = hyNode.getNode('PROERR')
    
            hyrpro = hyproNode.data()
            hyrrho = hyrhoNode.data()
            hyrperr = hyperrNode.data()
            hyrtime = hyrhoNode.dim_of()
    
            hygoodTimes = (hyrtime > 0).sum()
    
            self.hytime = hyrtime.data()[:hygoodTimes]
            self.hyrho = hyrrho[0,:] # Assume unchanging rho bins
            self.hypro = hyrpro[:,:hygoodTimes,:len(self.hyrho)]
            self.hyperr = hyrperr[:,:hygoodTimes,:len(self.hyrho)]
            
            self.hashy = True
        else:
            self.hashy = False
            
            
specTree = MDSplus.Tree('spectroscopy', 1150903021)
heNode = specTree.getNode('\SPECTROSCOPY::TOP.HIREXSR.ANALYSIS1.HELIKE.PROFILES.Z')
td = ThacoData(heNode, None)


f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex=True, sharey='row')

ax1.errorbar(np.sqrt(td.herho), td.hepro[3,7,:], yerr=td.heperr[3,7,:], fmt='-+', label=str(td.hetime[7])+'s')
ax1.errorbar(np.sqrt(td.herho), td.hepro[3,8,:], yerr=td.heperr[3,8,:], fmt='--', label=str(td.hetime[8])+'s')
ax1.errorbar(np.sqrt(td.herho), td.hepro[3,9,:], yerr=td.heperr[3,9,:], fmt='-.', label=str(td.hetime[9])+'s')
ax1.legend(loc='lower left', fontsize=18)
ax1.set_ylabel('Ion Temp [keV]')

ax3.errorbar(np.sqrt(td.herho), td.hepro[1,7,:], yerr=td.heperr[1,7,:], fmt='-+', label=str(td.hetime[7])+'s')
ax3.errorbar(np.sqrt(td.herho), td.hepro[1,8,:], yerr=td.heperr[1,8,:], fmt='--', label=str(td.hetime[8])+'s')
ax3.errorbar(np.sqrt(td.herho), td.hepro[1,9,:], yerr=td.heperr[1,9,:], fmt='-.', label=str(td.hetime[9])+'s')

ax2.errorbar(np.sqrt(td.herho), td.hepro[3,47,:], yerr=td.heperr[3,47,:], fmt='-+', label=str(td.hetime[47])+'s')
ax2.errorbar(np.sqrt(td.herho), td.hepro[3,48,:], yerr=td.heperr[3,48,:], fmt='--', label=str(td.hetime[48])+'s')
ax2.errorbar(np.sqrt(td.herho), td.hepro[3,50,:], yerr=td.heperr[3,50,:], fmt='-.', label=str(td.hetime[50])+'s')
ax2.legend(loc='lower left', fontsize=18)

ax4.errorbar(np.sqrt(td.herho), td.hepro[1,47,:], yerr=td.heperr[1,47,:], fmt='-+', label=str(td.hetime[47])+'s')
ax4.errorbar(np.sqrt(td.herho), td.hepro[1,48,:], yerr=td.heperr[1,48,:], fmt='--', label=str(td.hetime[48])+'s')
ax4.errorbar(np.sqrt(td.herho), td.hepro[1,50,:], yerr=td.heperr[1,50,:], fmt='-.', label=str(td.hetime[50])+'s')

ax3.set_ylim([-11, 11])
ax1.set_ylim([0.01, 1.6])

