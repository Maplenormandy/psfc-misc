# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 10:20:54 2016

@author: normandy
"""

import numpy as np

import readline
import MDSplus

import matplotlib.pyplot as plt
import matplotlib

# %% Data loading

class ThacoData:
    def __init__(self, thtNode, shot=None, tht=None, path='.HELIKE.PROFILES.Z'):
        if (shot != None):
            self.shot = shot
            self.specTree = MDSplus.Tree('spectroscopy', shot)

            if (tht == 0):
                self.tht = ''
            else:
                self.tht = str(tht)

            self.thtNode = self.specTree.getNode('\SPECTROSCOPY::TOP.HIREXSR.ANALYSIS' + self.tht + path)
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



td = ThacoData(None, 1110105031, 8)
#tdai = ThacoData(None, 111, 0, '.HLIKE.PROFILES.LYA1')

ei = eqtools.CModEFITTree(1110105031)

# %% Plotting

plt.close("all")

font = {'family' : 'normal',
        'size'   : 18}

matplotlib.rc('font', **font)



tli0, tli1 = np.searchsorted(td.time, (0.9, 1.0))
tmi0, tmi1 = np.searchsorted(td.time, (1.3, 1.4))

rzli = ei.psinorm2roa(td.rho, td.time[tli0])
rzmi = ei.psinorm2roa(td.rho, td.time[tmi0])

#plt.errorbar(np.average(rzli, axis=0), -np.average(tdzi.pro[1,tli0:tli1,:],axis=0)*2*np.pi*0.67, yerr=np.average(tdzi.perr[1,tli0:tli1,:],axis=0)*2*np.pi*0.67/2, fmt='b.', label='L-mode, Ar16+')

#plt.errorbar(np.average(rzmi[:,rzm0:], axis=0), -np.average(tdzi.pro[1,tmi0:tmi1,rzm0:], axis=0)*2*np.pi*0.67, yerr=np.average(tdzi.perr[1,tmi0:tmi1,rzm0:], axis=0)*2*np.pi*0.67, fmt='r.', label='I-mode, Ar16+')
#plt.errorbar(np.average(rami[:,:ram1], axis=0), -np.average(tdai.pro[1,tmi0:tmi1,:ram1], axis=0)*2*np.pi*0.67, yerr=np.average(tdai.perr[1,tmi0:tmi1,:ram1], axis=0)*2*np.pi*0.67, fmt='rx', label='I-mode, Ar17+')

plt.figure(1)
plt.errorbar(rzli,td.pro[1,tli0,:]*2*np.pi*0.67,yerr=td.perr[1,tli0,:]*2*np.pi*0.67, label='LOC', color='r')
plt.errorbar(rzmi,td.pro[1,tmi0,:]*2*np.pi*0.67,yerr=td.perr[1,tmi0,:]*2*np.pi*0.67, label='SOC', color='b')

plt.legend(loc='lower right')
plt.xlim([-0.01, 1.0])
plt.xlabel('r/a')
plt.ylabel('Rotation Velocity [km/s]')

plt.figure(2)
plt.errorbar(rzli,td.pro[3,tli0,:],yerr=td.perr[3,tli0,:], label='LOC', color='r')
plt.errorbar(rzmi,td.pro[3,tmi0,:],yerr=td.perr[3,tmi0,:], label='SOC', color='b')

plt.legend(loc='lower left')
plt.xlim([-0.01, 1.0])
plt.xlabel('r/a')
plt.ylabel('Ion Temperature [keV]')

# %% More plotting

plt.close("all")

tdzh = ThacoData(None, 1160725018, 0)
tdah = ThacoData(None, 1160725018, 0, '.HLIKE.PROFILES.LYA1')

eh = eqtools.CModEFITTree(1160725018)

tlh0, tlh1 = np.searchsorted(tdzh.time, (0.52, 0.62))
tmh0, tmh1 = np.searchsorted(tdzh.time, (0.7, 0.8))

rzlh = eh.psinorm2roa(tdzh.rho, tdzh.time[tlh0:tlh1])

rzmh = eh.psinorm2roa(tdzh.rho, tdzh.time[tmh0:tmh1])
ramh = eh.psinorm2roa(tdah.rho, tdah.time[tmh0:tmh1])

rzm0 = np.searchsorted(rzmh[0,:], 0.35)
ram1 = np.searchsorted(ramh[0,:], 0.65)

plt.figure(2)
plt.errorbar(np.average(rzlh, axis=0), np.average(tdzh.pro[1,tlh0:tlh1,:],axis=0)*2*np.pi*0.67, yerr=np.average(tdzh.perr[1,tlh0:tlh1,:],axis=0)*2*np.pi*0.67/2, fmt='b.', label='L-mode, Ar16+')

plt.errorbar(np.average(rzmh[:,rzm0:], axis=0), np.average(tdzh.pro[1,tmh0:tmh1,rzm0:], axis=0)*2*np.pi*0.67, yerr=np.average(tdzh.perr[1,tmh0:tmh1,rzm0:], axis=0)*2*np.pi*0.67/2, fmt='r.', label='H-mode, Ar16+')
plt.errorbar(np.average(ramh[:,:ram1], axis=0), np.average(tdah.pro[1,tmh0:tmh1,:ram1], axis=0)*2*np.pi*0.67, yerr=np.average(tdah.perr[1,tmh0:tmh1,:ram1], axis=0)*2*np.pi*0.67/2, fmt='rx', label='H-mode, Ar17+')

#plt.scatter(rzlh.flatten(), tdzh.pro[1,tlh0:tlh1,:].flatten()*2*np.pi*0.67, color='b', marker=',', label='L-mode, Ar16+')

#plt.scatter(rzmh[:,rzm0:].flatten(), tdzh.pro[1,tmh0:tmh1,rzm0:].flatten()*2*np.pi*0.67, color='r', marker='.', label='H-mode, Ar16+')
#plt.scatter(ramh[:,:ram1].flatten(), tdah.pro[1,tmh0:tmh1,:ram1].flatten()*2*np.pi*0.67, color='r', marker='x', label='H-mode, Ar17+')

plt.legend(loc='upper right')
plt.xlim([-0.01, 1.0])
plt.ylim([-40, 100])
plt.xlabel('r/a')
plt.ylabel('Rotation Velocity [km/s]')