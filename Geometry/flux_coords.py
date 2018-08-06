# -*- coding: utf-8 -*-
"""
Created on Fri Aug  3 17:17:22 2018

@author: normandy
"""

import numpy as np
import matplotlib.pyplot as plt

import scipy.interpolate
from scipy.integrate import ode

import readline
import MDSplus

import eqtools

# %%
t0 = 0.6

e = eqtools.CModEFITTree(1160506007)

time = e.getTimeBase()
tind = np.searchsorted(time, t0)
tb = time[tind]

# %% Calculate orthogonal straight field line coordinates
 
rmid = e.getRmidPsi()[tind,:]
rgrid = e.getRGrid()
zgrid = e.getZGrid()
# [nt, nz, nr]
fluxgrid = e.getFluxGrid()[tind,:,:]

psiAxis = e.getFluxAxis()[tind]
psiLCFS = e.getFluxLCFS()[tind]

kappa = e.getElongation()[tind]
aout = e.getAOut()[tind]
rmag = e.getMagR()[tind]

r = rmid-rmag

## Start with a single field line

theta0 = np.linspace(0, 2*np.pi, 32)
x = np.array([rmag + r[16]*np.cos(theta0), r[16]*np.sin(theta0)])
v = np.zeros(x.shape)
# Flattens row major, so fastest is point, then r/z, then x/v
state = np.array([x, v])
statevec = np.flatten(state)


rplot, zplot = np.meshgrid(rgrid, zgrid)

#plt.contour(rgrid, zgrid, fluxgrid)
#plt.plot(R0, Z0)
#plt.axis('equal')

flux = scipy.interpolate.RectBivariateSpline(rgrid, zgrid, fluxgrid.T)
psi = flux.ev(rmid, np.zeros(rmid.shape))
psi0 = psi[16]

# %% Start evolving points
k_flux = 1.0
def deriv(t, y, psi0):
    s = np.reshape(y, (state.shape))
    r = s[0,0,:]
    z = s[0,1,:]

    # Restoring force towards correct flux surface
    delta_flux = (psi0 - flux.ev(r, z))
    dir_flux = np.array([flux.ev(r, z, dx=1), flux.ev(r, z, dy=1)])
    f_flux = k_flux * delta_flux * dir_flux

    # Force between
    pass
