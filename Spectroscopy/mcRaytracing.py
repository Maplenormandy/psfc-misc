# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 14:42:33 2016

@author: normandy
"""

import numpy as np
import matplotlib.pyplot as plt

from collections import namedtuple

import eqtools

# %% Info file
InfoBragg = namedtuple('InfoBragg', 'iref rwid twod'.split())
InfoMirror = namedtuple('InfoMirror', 'vec rot size bragg rad'.split())
InfoDetector = namedtuple('InfoDetector', 'x0 x1 x2 xi zeta size n_xi n_zeta'.split())


# %% Set up geometric parameters

mirrorHe = InfoMirror(np.array([3.687, 0, -0.0068]), # [R, theta, Z] of crystal location [m]
                      np.array([0.0, 0.029, 2.03]), # [alpha, beta, gamma] rotation angles [rad]
                      np.array([6.4, 2.7]), # [y, z] size of crystal [cm]
                      InfoBragg(1.0, 0.0, 4.56216), # [integrated reflectivity, rocking curve, 2d spacing [Ang]]
                      1.442) # radius of crystal curvature
det1 = InfoDetector(np.array([109.95, -60.12, -4.19]), # position of origin of det system
                    np.array([109.95, -60.12, 4.15]), # position of point along zeta axis
                    np.array([109.95, -60.12, -4.19]), # position of point along xi axis
                    -1, # xi values of pixel centers, unused
                    -1, # zeta values of pixel centers, unused
                    np.array([0.0172, 0.0172]), # [xi, zeta] size of pixels [cm]
                    195, # number of xi channels
                    487) # number of zeta channels


# Spectral lines
lamz = 3.9941451
lamw = 3.9490665


# %% Calculate the transfer function

m = mirrorHe
det = det1

shot = 1150903021

e = eqtools.CModEFITTree(shot)
rgrid = e.getRGrid()
zgrid = e.getZGrid()

flux = e.getFluxGrid()
flcfs = e.getFluxLCFS()
levels = np.linspace(np.min(flux[35,:,:]), flcfs[35], 8)


# Plot the magnetic flux
rplot, zplot = np.meshgrid(rgrid, zgrid)
plt.contour(rplot, zplot, flux[35,:,:], levels=levels)
plt.axis('equal')

# Plot a couple sightlines from the detector to the plasma

# Unit vector from detector to center of mirror, in mirror coordinates
l_dm_m = det1.x0 / np.linalg.norm(det1.x0)
# Unit vector from center of mirror to plasma, in mirror coordinates
n_m = np.array([1.0, 0.0, 0.0])
l_mp_m = l_dm_m - 2*np.dot(n_m, l_dm_m)*n_m # reflection r=d-2(d.n)n
