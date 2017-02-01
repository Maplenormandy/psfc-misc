# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 14:42:33 2016

@author: normandy
"""

import numpy as np
import matplotlib.pyplot as plt

from collections import namedtuple

import eqtools

import itertools

# %% Info file
InfoBragg = namedtuple('InfoBragg', 'iref rwid twod'.split())
InfoMirror = namedtuple('InfoMirror', 'vec rot size bragg rad'.split())
InfoDetector = namedtuple('InfoDetector', 'x0 x1 x2 xi zeta size n_xi n_zeta'.split())

# %% Helper functions
def rotx(theta):
    return np.array([[1, 0, 0], [0, np.cos(theta), -np.sin(theta)], [0, np.sin(theta), np.cos(theta)]])
    
def roty(theta):
    return np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])

def rotz(theta):
    return np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])

def cartToCyl(vec):
    # [x,y,z] -> [r,theta,z]
    return np.array([np.sqrt(vec[0]**2 + vec[1]**2), np.arctan2(vec[1], vec[0]), vec[2]])
    
def cylToCart(vec):
    return np.array([vec[0]*np.cos(vec[1]), vec[0]*np.sin(vec[1]), vec[2]])

def cylRay(x1, x2, l):
    r1 = x1[0]
    r2 = x2[0]
    th1 = x1[1]
    th2 = x2[1]
    z1 = x1[2]
    z2 = x2[2]
    
    x = np.sqrt(r1**2 * (1-2*l) + l**2 * (r1**2 + r2**2) - 2*l*(l-1)*r1*r2*np.cos(th2-th1))
    th = np.arctan2(np.sin(th1) + l*(r2/r1*np.sin(th2)-np.sin(th1)), np.cos(th1)+l*(r2/r1*np.cos(th2)-np.cos(th1)))
    z = z1 + l*(z2-z1)
    
    return np.array([x, th, z])

# %% Set up geometric parameters

# This info file is from 1150903028

mirrorHe = InfoMirror(np.array([3.687, 0, 0.0004]), # [R, theta, Z] of crystal location [m]
                      np.array([0.0, -0.00174533, 2.025]), # [alpha, beta, gamma] rotation angles [rad]
                      np.array([0.064, 0.027]), # [y, z] size of crystal [m]
                      InfoBragg(1.0, 0.0, 4.56216), # [integrated reflectivity, rocking curve, 2d spacing [Ang]]
                      1.442) # radius of crystal curvature
det2 = InfoDetector(np.array([111.082, -60.73, -3.0108]), # position of origin of det system
                    np.array([111.083, -60.7308, 5.36560]), # position of point along zeta axis
                    np.array([109.396, -63.6292, -3.01080]), # position of point along xi axis
                    -1, # xi values of pixel centers, unused
                    -1, # zeta values of pixel centers, unused
                    np.array([0.0172, 0.0172]), # [xi, zeta] size of pixels [cm]
                    195, # number of xi channels
                    487) # number of zeta channels

det1 = InfoDetector(np.array([111.196, -60.7343, 5.68298]), # position of origin of det system
                    np.array([111.202, -59.5147, 13.9701]), # position of point along zeta axis
                    np.array([109.532, -63.6150, 6.10806]), # position of point along xi axis
                    -1, # xi values of pixel centers, unused
                    -1, # zeta values of pixel centers, unused
                    np.array([0.0172, 0.0172]), # [xi, zeta] size of pixels [cm]
                    195, # number of xi channels
                    487) # number of zeta channels


# Spectral lines
lamz = 3.9941451
lamw = 3.9490665

# %% Plot tokamak and such

shot = 1150903021

e = eqtools.CModEFITTree(shot)
rgrid = e.getRGrid()
zgrid = e.getZGrid()

plt.close("all")

flux = e.getFluxGrid()
flcfs = e.getFluxLCFS()
levels = np.linspace(np.min(flux[35,:,:]), flcfs[35], 8)


plt.figure()
# Plot the magnetic flux
rplot, zplot = np.meshgrid(rgrid, zgrid)
plt.contour(rplot, zplot, flux[35,:,:], levels=levels)
plt.colorbar()
plt.axis('equal')

plt.figure()
circlex = np.cos(np.linspace(0, 2*np.pi, 256))
circley = np.sin(np.linspace(0, 2*np.pi, 256))
plt.plot(0.44*circlex, 0.44*circley, c='#550000')
plt.plot(0.89*circlex, 0.89*circley, c='#550000')
plt.axis('equal')

# %% Calculate the transfer function

def raytrace(m, det, m_y, m_z, det_xi, det_zeta, generateRays=False):
    """
    m_y, m_z # flat (y,z) position of mirror elements to integrate over (in m)
    Note that this isn't normalized in case of circular mirror
    
    det_xi, det_zeta # xi (wavelength axis), zeta (spatial axis) in normalized
    pixel coodinates, i.e. 1.0 = one pixel, should also be flat
    
    Will return len(det_xi) x len(m_y)
    """
    
    # Rotation matrices from mirror (m) <> tokamak cartesian (t) coordinates
    # Note the coordinate system: In tokamak cylindrical (k), theta=0 is aligned with X_t
    # gamma=0 corresponds to y_m = X_t, not x_m = X_t unfortunately. Also the rotations
    # are in a clockwise sense.
    #R_mt = rotz(-np.pi/2.0).dot(rotx(-m.rot[0]).dot(roty(-m.rot[1]).dot(rotz(-m.rot[2]))))
    #R_tm = np.linalg.inv(R_mt)
    
    # (y,z) position of mirror element (in m)
    twod_pos_m = np.array([m_y, m_z])
    
    # real position of mirror [cm]
    z_m = np.sqrt(m.rad**2-np.einsum('i...,i...',twod_pos_m,twod_pos_m))
    pos_m = np.array([m.rad-z_m, twod_pos_m[0,:], twod_pos_m[1,:]]) * 100.0
    
    # mirror normal, mirror coordinates
    n_m = -(pos_m - np.array([m.rad*100, 0, 0])[:,np.newaxis])
    n_m = n_m / np.sqrt(np.einsum('i...,i...', n_m, n_m))[np.newaxis, :]
    
    # area element (i.e. jacobian) of mirror at pos_m
    area_m = m.rad/z_m
    area_m = np.array([area_m]*len(det_xi)) # resize array to proper size
    
    # Unit translation vectors along the xi and zeta axes, mirror coordinates
    xi_m = (det.x2 - det.x0) / np.linalg.norm(det.x2 - det.x0) * det.size[1]
    zeta_m = (det.x1 - det.x0) / np.linalg.norm(det.x1 - det.x0) * det.size[1]
    
    # Vector from detector (d) to center of mirror, in mirror coordinates
    r_dm0_m = -det.x0[:, np.newaxis] - np.outer(xi_m,det_xi) - np.outer(zeta_m,det_zeta)
    
    # Vector from detector to actual point on mirror
    r_dm_m = r_dm0_m[:,:,np.newaxis] + pos_m[:,np.newaxis,:]
    
    # Squared distance from detector to actual point on mirror
    s2_dm = np.einsum('i...,i...', r_dm_m, r_dm_m)
    
    # Unit vector from detector to point on mirror
    l_dm_m = r_dm_m / np.sqrt(s2_dm)[np.newaxis, :, :]
    
    # Total weight of integration point (i.e. jacobian)
    weight = area_m * -np.einsum('i...,i...', n_m, l_dm_m) / s2_dm
    
    if generateRays:
        """
        Not entirely implemented yet in the general binned case
        
        # Unit vector from center of mirror to plasma, in mirror coordinates
        l_mp_m = l_dm_m - 2*np.dot(n_m, l_dm_m)*n_m # reflection r=d-2(d.n)n
        # Now in tokamak cartesian coordinates
        l_mp_t = R_mt.dot(l_mp_m)
        # Mirror element displacement in tokamak coordinates [m]
        d_m_t = R_mt.dot(pos_m/100.0)
        # Mirror element position
        pos_m_t = m.vec + d_m_t
        # position of point along ray
        r_mp_t = pos_m_t + l_mp_t
        # Convert to tokamak cylindrical coordinates (k)
        pos_m_k = cartToCyl(pos_m_t)
        r_mp_k = cartToCyl(r_mp_t) 
        
        # Now, using formula C.9, draw the sightline
        ray = cylRay(pos_m_k, r_mp_k, np.linspace(0, 4))
        """
    
    # Lambda of each integration point
    lam = -np.einsum('i...,i...', n_m, l_dm_m)*m.bragg.twod

    return lam, weight
    
def momentAnalysis(lam, weight, lam0):
    mAr = 37211326.1 # Mass of argon in keV
    c = 2.998e+5 # speed of light in km/s
    
    w = lam0**2 / mAr # width of 1 keV line
    
    m0 = np.sum(weight)    
    m1 = np.average(lam.flatten(), weights=weight.flatten())
    m2 = np.average((lam.flatten() - m1)**2, weights=weight.flatten())
    m3 = np.average((lam.flatten() - m1)**3, weights=weight.flatten())
    
    return m0, m1, m2, m3, (m1-lam0)/lam0*c/np.sin(8/180.0*np.pi), m2/w*1e3


# %% Central wavelength calculation
m_y = np.array([0.0])
m_z = np.array([0.0])

det_xi_grid, det_zeta_grid = np.meshgrid(np.array(range(195))+0.5, np.array(range(487))+0.5)

lam1, weight1 = raytrace(mirrorHe, det1, m_y, m_z, det_xi_grid.flatten(), det_zeta_grid.flatten())
lam2, weight2 = raytrace(mirrorHe, det2, m_y, m_z, det_xi_grid.flatten(), det_zeta_grid.flatten())

# %% Central wavelength plotting
lam1_grid = lam1.reshape(det_xi_grid.shape)
lam2_grid = lam2.reshape(det_xi_grid.shape)

plt.figure()
plt.pcolormesh(det_xi_grid, det_zeta_grid, lam2_grid)
plt.colorbar()

# Calculate z line, w line pixels
z_pix1 = np.abs(lam1_grid-lamz).argmin(axis=1)
z_pix2 = np.abs(lam2_grid-lamz).argmin(axis=1)
w_pix1 = np.abs(lam1_grid-lamw).argmin(axis=1)
w_pix2 = np.abs(lam2_grid-lamw).argmin(axis=1)

# %% Instrument function calculation, single pixels
m_range = np.linspace(-0.5, 0.5, 64)
m_x_grid, m_y_grid = np.meshgrid(m_range*mirrorHe.size[0], m_range*mirrorHe.size[1])

p_range = np.linspace(0, 1, 16)
z_xi1 = p_range + z_pix1[350]
z_zeta1 = p_range + 350
w_xi1 = p_range + w_pix1[350]
w_zeta1 = p_range + 350
z_xi2 = p_range + z_pix2[250]
z_zeta2 = p_range + 250
w_xi2 = p_range + w_pix2[250]
w_zeta2 = p_range + 250

z_lam1, z_weight1 = raytrace(mirrorHe, det1, m_x_grid.flatten(), m_y_grid.flatten(), z_xi1, z_zeta1)
w_lam1, w_weight1 = raytrace(mirrorHe, det1, m_x_grid.flatten(), m_y_grid.flatten(), w_xi1, w_zeta1)
z_lam2, z_weight2 = raytrace(mirrorHe, det2, m_x_grid.flatten(), m_y_grid.flatten(), z_xi2, z_zeta2)
w_lam2, w_weight2 = raytrace(mirrorHe, det2, m_x_grid.flatten(), m_y_grid.flatten(), w_xi2, w_zeta2)

# %% Plot instrument function

#plt.figure()
#plt.hist(z_lam2.flatten()-lamz, bins=25, weights=z_weight2.flatten())

print momentAnalysis(z_lam1, z_weight1, lamz)
print momentAnalysis(w_lam1, w_weight1, lamw)
print momentAnalysis(z_lam2, z_weight2, lamz)
print momentAnalysis(w_lam2, w_weight2, lamw)

# %% Instrument function for entire grid

m_range = np.linspace(-0.5, 0.5, 64)
m_x_grid, m_y_grid = np.meshgrid(m_range*mirrorHe.size[0], m_range*mirrorHe.size[1])

m0_grid = np.zeros((487, 195, 2))
m1_grid = np.zeros((487, 195, 2))
inst_temp_grid = np.zeros((487, 195, 2))
m2_grid = np.zeros((487, 195, 2))
m3_grid = np.zeros((487, 195, 2))

p_range = np.linspace(0, 1, 16)

for row in range(487):
    for col in range(195):
        plam1, pweight1 = raytrace(mirrorHe, det1, m_x_grid.flatten(), m_y_grid.flatten(), p_range+col, p_range+row)
        plam2, pweight2 = raytrace(mirrorHe, det2, m_x_grid.flatten(), m_y_grid.flatten(), p_range+col, p_range+row)
        
        m0, m1, m2, m3, inst_vel, inst_temp = momentAnalysis(plam1, pweight1, lamz)
        m0_grid[row, col, 0] = m0
        m1_grid[row, col, 0] = m1
        m2_grid[row, col, 0] = m2
        m3_grid[row, col, 0] = m3
        inst_temp_grid[row, col, 0] = inst_temp
        m0, m1, m2, m3, inst_vel, inst_temp = momentAnalysis(plam2, pweight2, lamz)
        m0_grid[row, col, 1] = m0
        m1_grid[row, col, 1] = m1
        m2_grid[row, col, 1] = m2
        m3_grid[row, col, 1] = m3
        inst_temp_grid[row, col, 1] = inst_temp
        
# %% Whole-detector plots

plt.figure()
plt.pcolormesh(m1_grid[:,:,0], det_zeta_grid, (m1_grid[:,:,0]-lam1_grid)*1e3)
plt.colorbar()
plt.figure()
plt.pcolormesh(m1_grid[:,:,0], det_zeta_grid, (m1_grid[:,:,1]-lam2_grid)*1e3)
plt.colorbar()


plt.figure()
plt.pcolormesh(m1_grid[:,:,0], det_zeta_grid, inst_temp_grid[:,:,0])
plt.axvline(lamz)
plt.axvline(lamw)
plt.colorbar()
plt.figure()
plt.pcolormesh(m1_grid[:,:,1], det_zeta_grid, inst_temp_grid[:,:,1])
plt.axvline(lamz)
plt.axvline(lamw)
plt.colorbar()

# %% Single line plots

itemp1 = inst_temp_grid[range(487), z_pix1, 0]
itemp2 = inst_temp_grid[range(487), z_pix2, 1]

plt.figure()
plt.plot(np.array(range(487))+487, itemp1)
plt.plot(np.array(range(487)), itemp2)
plt.xlabel('Spatial channel')
plt.ylabel('Instrumental Temperature [keV]')


# %% Playing with instrument functions

mAr = 37211326.1 # Mass of argon in keV

def singleGaussian(lam0, te):
    w = lam0**2 * te / mAr
    a = 1 / np.sqrt(w * 2 * np.pi)
    return lambda x: a * np.exp(-np.square(x - lam0) / 2 / w)
    
def instrumentFunction(lam, weight):
    hist, inst_lam = np.histogram(lam, bins=63, weights=weight, density=True)
    inst_f = np.zeros(inst_lam.shape)
    inst_f[1:] += hist/2.0
    inst_f[:-1] += hist/2.0
    
    return inst_f, inst_lam
    
def evalInstrumentFunction(f, inst_f, inst_lam):
    dlam = np.median(np.diff(inst_lam))
    return inst_f.dot(f(inst_lam)) * dlam

# %% Evaluate instrument function

lam0 = lamw
lam0_grid = lam1_grid

f = singleGaussian(lam0, 1.16)

row = 50
col0 = w_pix2[row]
colrange = range(col0-15, col0+10)

ishift = m1_grid[row, col0, 0] - lam0_grid[row, col0]
itemp = m2_grid[row, col0, 0] / (lam0**2 / mAr)

g = singleGaussian(lam0-ishift, 1.16+itemp)

syntheticLine = f(lam0_grid[row, colrange])
convolvedLine = g(lam0_grid[row, colrange])
instrumentLine = np.zeros(len(colrange))

m_range = np.linspace(-0.5, 0.5, 64)
m_x_grid, m_y_grid = np.meshgrid(m_range*mirrorHe.size[0], m_range*mirrorHe.size[1])

p_range = np.linspace(0, 1, 16)

j = 0
for col in colrange:
    lam, weight = raytrace(mirrorHe, det1, m_x_grid.flatten(), m_y_grid.flatten(), p_range+col, p_range+row)
    inst_f, inst_lam = instrumentFunction(lam.flatten(), weight.flatten())
    instrumentLine[j] = evalInstrumentFunction(f, inst_f, inst_lam)
    j += 1
    
plt.figure()
plt.plot(colrange, syntheticLine, marker='.', c='g', label='Synthetic W line')
plt.plot(colrange, convolvedLine, marker='.', c='r', label='Gaussian Convolution')
plt.plot(colrange, instrumentLine, marker='+', c='b', label='Instrument Function')
plt.xlabel('Pixels [+'+str(col0)+']')
plt.ylabel('Counts [A.U.]')
plt.legend(loc='upper left')
    