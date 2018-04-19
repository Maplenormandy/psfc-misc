# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 14:50:40 2016

@author: normandy
"""

import readline
import MDSplus

import mcRaytracing

import numpy as np
import matplotlib.pyplot as plt


from mpl_toolkits.mplot3d import Axes3D

# %% Raytrace for all modules

mcRaytracing.calculateInstrumentals(1160503008, 'MOD1')
mcRaytracing.calculateInstrumentals(1160503008, 'MOD2')

# %% Get results

specTree = MDSplus.Tree('spectroscopy', 1160503008)
instNode = specTree.getNode(r'\SPECTROSCOPY::TOP.HIREXSR.CALIB.MOD1:INST')
plt.pcolor(instNode.data())


# %% Things
"""
mirrorHe, det1 = mcRaytracing.getInfoFile(1160506007, 'MOD1')
mirrorHe, det2 = mcRaytracing.getInfoFile(1160506007, 'MOD2')
"""
# %%
"""
xi, zeta = np.meshgrid(np.linspace(0,1,25), np.linspace(0,1,25))
xi = xi.flatten()
zeta = zeta.flatten()

def plotDet(det, ax, c):
    zetax = det.x1 - det.x0
    xix = det.x2 - det.x0
    
    zetax = zetax / np.linalg.norm(zetax)
    xix = xix / np.linalg.norm(xix)
    
    detpts = det.x0[:,np.newaxis] + np.outer(zetax,zeta)*det.size[1]*det.n_zeta + np.outer(xix,xi)*det.size[0]*det.n_xi
    
    if c == 'b':
        cbar = ax.scatter(detpts[0,:], detpts[1,:], detpts[2:], c=zeta, cmap='gray')
    else:
        cbar = ax.scatter(detpts[0,:], detpts[1,:], detpts[2:], c=c)
    return cbar
    
fig = plt.figure(1)
ax = fig.add_subplot(111, projection='3d')

plotDet(det1, ax, 'r')
cbar = plotDet(det2, ax, 'b')

ax.axis('square')
ax.set_aspect('equal')

fig.colorbar(cbar)
"""