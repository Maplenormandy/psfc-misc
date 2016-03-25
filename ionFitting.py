# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 21:48:02 2016

@author: normandy
"""

import numpy as np
import matplotlib.pyplot as plt

from matplotlib import gridspec
#import matplotlib.colors as colors
    
import readline
import MDSplus

from scipy.interpolate import SmoothBivariateSpline

#import shotAnalysisTools as sat

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
        

specTree = MDSplus.Tree('spectroscopy', 1120607008)

anaNode = specTree.getNode('\SPECTROSCOPY::TOP.HIREXSR.ANALYSIS')
heNode = anaNode.getNode('HELIKE.PROFILES.Z')
td = ThacoData(heNode)

t0 = 0.9-0.011
t1 = 0.9+0.031

i0 = np.searchsorted(td.time, t0)
i1 = np.searchsorted(td.time, t1)

rp, tp = np.meshgrid(td.rho, td.time[i0:i1])
tip = td.pro[3,i0:i1,:]
wtip = 1.0 / td.perr[3,i0:i1,:]
vtorp = td.pro[1,i0:i1,:]
wvtorp = 1.0 / td.perr[1,i0:i1,:]

tifit = SmoothBivariateSpline(rp.flatten(), tp.flatten(), tip.flatten(), w=wtip.flatten(), kx=5, ky=5)
vtorfit = SmoothBivariateSpline(rp.flatten(), tp.flatten(), vtorp.flatten(), w=wvtorp.flatten(), kx=5, ky=5)




"""
rplot, tplot = np.meshgrid(td.rho, td.time)
tiplot = td.pro[3,:,:-1]
vtorplot = td.pro[1,:,:-1]
"""

rplot, tplot = np.meshgrid(np.linspace(0.00,0.9), np.linspace(t0, t1))
tiplot = tifit(np.linspace(0.05,1.0), np.linspace(t0, t1)).T
vtorplot = vtorfit(np.linspace(0.05,1.0), np.linspace(t0, t1)).T




fig = plt.figure(figsize=(22,6))
gs = gridspec.GridSpec(2,1)
ax1 = fig.add_subplot(gs[0])
cax1 = ax1.pcolormesh(tplot, rplot, tiplot, cmap='cubehelix', vmin=0.3, vmax=1.9)
fig.colorbar(cax1)

fig.suptitle('Shot ' + str(1120607008) + ' Ion Temp, Toroidal Velocity (Bspline)')

ax2 = fig.add_subplot(gs[1], sharex=ax1, sharey=ax1)
cax2 = ax2.pcolormesh(tplot, rplot, vtorplot, cmap='BrBG', vmin=-20, vmax=20)
fig.colorbar(cax2)

plt.tight_layout()
plt.plot()

