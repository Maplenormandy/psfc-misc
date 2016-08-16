# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 14:42:33 2016

@author: normandy
"""

import numpy as np
import matplotlib.pyplot as plt

from collections import namedtuple

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







# Crystal parameters
spacing = 4.56225 # m_twod
crystalRadius = 1443.0 # m_rad
crystalWidth = 64.0 # m_y * 10
crystalHeight = 27.0 # m_z * 10

crystalExtent = crystalWidth/crystalRadius

# Detector parameters
detPixWidth = 195
#detPixHeight = 487 # Unused for now
detPixSize = 0.172

detWidth = detPixSize*detPixWidth


braggz = np.arcsin(lamz/spacing) # This is going to be the focused line
braggw = np.arcsin(lamw/spacing)
# calculate the bragg peak location on the Rowland circle
zpeakX = -crystalRadius / 2.0 * np.sin(2*braggz)
zpeakY = zpeakX * np.tan(braggz)



# How far along the width to anchor the detector, 0.0 is at the right
detAnchorX = 0.2

# How far to move the detector anchor point from the ideal peak location
detTransX = 0.0
detTransY = 0.0

# How much to rotate the detector from tangency
detRotateZ = 0.0


plasmaRmagx = 670.0 # Approx. plasma magnetic axis radius
plasmaAout = 210.0 # Approx plasma radius at midplane



torCenterX = 3.687 * np.sin(2*braggz)
torCenterY = -torCenterX * np.tan(braggz)



# %% Plot the setup

# Crystal plot
crth = np.linspace(-crystalExtent, crystalExtent)
crX = np.sin(crth) * crystalRadius
crY = (np.cos(crth) - 1.0)  * crystalRadius

# Rowland circle plot
rcth = np.linspace(-np.pi, np.pi, 512)
rcX = np.sin(rcth)  * crystalRadius / 2.0
rcY = (np.cos(rcth) - 1.0)  * crystalRadius / 2.0

# Ideal Bragg angle plot
brX = [zpeakX, 0.0, -zpeakX]
brY = [zpeakY, 0.0, zpeakY]

# Detector plot
dets = np.array([detAnchorX - 1.0, detAnchorX])*detWidth
detX = zpeakX + detTransX + dets*-np.cos(2*braggz+detRotateZ)
detY = zpeakY + detTransY + dets*-np.sin(2*braggz+detRotateZ)

# Plasma plot


plt.figure()
plt.plot(crX, crY)
plt.plot(rcX, rcY, c='r', ls=':')
plt.plot(detX, detY, c='g')
plt.axis('equal')

# %% Focus calculations

fm = crystalRadius * np.sin(braggz)
fs = fm / (2 * np.sin(braggz)**2 - 1)

# %% Raytracing from detector to mirror

# %% Raytracing from mirror to plasma




