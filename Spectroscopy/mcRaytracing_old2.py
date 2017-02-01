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

m = mirrorHe
det = det2



# Plot a couple sightlines from the detector to the plasma

# Rotation matrices from mirror (m) <> tokamak cartesian (t) coordinates
# Note the coordinate system: In tokamak cylindrical (k), theta=0 is aligned with X_t
# gamma=0 corresponds to y_m = X_t, not x_m = X_t unfortunately. Also the rotations
# are in a clockwise sense.
R_mt = rotz(-np.pi/2.0).dot(rotx(-m.rot[0]).dot(roty(-m.rot[1]).dot(rotz(-m.rot[2]))))
R_tm = np.linalg.inv(R_mt)

# number of bins
n_i_m = 128
n_j_m = 54
wavebin = np.zeros((9, n_i_m, n_j_m))
waveweights = np.zeros((9, n_i_m, n_j_m))


for i_m, j_m in itertools.product(range(n_i_m), range(n_j_m)):
    xi_m = np.linspace(-1, 1, n_i_m)
    zeta_m = np.linspace(-1, 1, n_j_m)
    
    # (y,z) position of mirror element (in m)
    twod_pos_m = [xi_m[i_m], xi_m[j_m]] * m.size / 2.0
    # position of mirror [cm]
    pos_m = np.array([m.rad-np.sqrt(m.rad**2-np.sum(np.square(twod_pos_m))), twod_pos_m[0], twod_pos_m[1]]) * 100.0
    
    # Jacobian of area element of mirror at pos_m
    area_m = np.sqrt(m.rad**2/(m.rad**2-np.sum(np.square(twod_pos_m))))
    
    # Mirror normal, mirror coordinates
    n_m = -(pos_m - [m.rad*100, 0, 0])/np.linalg.norm(pos_m - [m.rad*100, 0, 0])
        
    k_det = 0
    for i_det, j_det in itertools.izip([160, 161, 162], [250, 250, 250]):
        # Vector from detector (d) to center of mirror, in mirror coordinates
        r_dm0_m = -det.x0
        # 1 is zeta, 2 is xi
        r_dm0_m -= (det.x2 - det.x0) / np.linalg.norm(det.x2 - det.x0) * det.size[1] * i_det
        r_dm0_m -= (det.x1 - det.x0) / np.linalg.norm(det.x1 - det.x0) * det.size[1] * j_det
        
        # Vector from detector to actual point on mirror
        r_dm_m = r_dm0_m + pos_m
        # Unit vector from detector to point on mirror
        l_dm_m = r_dm_m / np.linalg.norm(r_dm_m)
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
        plt.figure(1)
        plt.plot(ray[0], ray[2], c=((mxi+1.0)/2.0, (mzeta+1.0)/2.0, 0.0))
        plt.figure(2)
        plt.plot(ray[0]*np.cos(ray[1]), ray[0]*np.sin(ray[1]), c=((mxi+1.0)/2.0, (mzeta+1.0)/2.0, 0.0))
        """
        
        # Calculate wavelength of bragg angle
        wavebin[k_det, i_m, j_m] = -l_dm_m.dot(n_m)*m.bragg.twod
        waveweights[k_det, i_m, j_m] = area_m * -l_dm_m.dot(n_m)
        
        #print k_det, i_det, j_det
        k_det += 1

wavebin2 = wavebin.reshape((wavebin.shape[0], -1))
waveweights2 = waveweights.reshape((waveweights.shape[0], -1))

# %% Histogram stuff

plt.hist(wavebin2[0,:], bins=50, weights=waveweights2[0,:], color='b', alpha=0.5)
plt.hist(wavebin2[1,:], bins=50, weights=waveweights2[1,:], color='g', alpha=0.5)
plt.hist(wavebin2[2,:], bins=50, weights=waveweights2[2,:], color='r', alpha=0.5)

# %% Calculate central wavelengths

i_det_grid, j_det_grid = np.meshgrid(np.array(range(195))+0.5, np.array(range(487))+0.5)
i_det = i_det_grid.flatten()
j_det = j_det_grid.flatten()

xi_v = (det.x2 - det.x0) / np.linalg.norm(det.x2 - det.x0) * det.size[1]
zeta_v = (det.x1 - det.x0) / np.linalg.norm(det.x1 - det.x0) * det.size[1]

twod_pos_m = [0.0, 0.0] * m.size / 2.0
pos_m = np.array([m.rad-np.sqrt(m.rad**2-np.sum(np.square(twod_pos_m))), twod_pos_m[0], twod_pos_m[1]]) * 100.0
n_m = -(pos_m - [m.rad*100, 0, 0])/np.linalg.norm(pos_m - [m.rad*100, 0, 0])

r_dm0_m = -det.x0[:, np.newaxis] - np.outer(xi_v,i_det) - np.outer(zeta_v,j_det)
r_dm_m = r_dm0_m + pos_m[:, np.newaxis]
l_dm_m = r_dm_m / np.sqrt(np.einsum('i...,i...', r_dm_m, r_dm_m))[np.newaxis, :]
#l_mp_m = l_dm_m - np.outer(n_m, 2*np.dot(n_m, l_dm_m))
#l_mp_t = R_mt.dot(l_mp_m)

lam0 = -n_m.dot(l_dm_m)*m.bragg.twod
lam0_grid = lam0.reshape(i_det_grid.shape)

# %% Plot central wavelengths

plt.figure()
plt.scatter(range(195), lam0_grid[486,:])
plt.axhline(lamz)
plt.axhline(lamw)

# %% Plot mirror wavelengths (Johann error)

m_x_grid, m_y_grid = np.meshgrid(np.linspace(-1, 1)*m.size[0]/2.0, np.linspace(-1, 1)*m.size[1]/2.0)
twod_pos_m = np.array([m_x_grid.flatten(), m_y_grid.flatten()])
pos_m = np.array([m.rad-np.sqrt(m.rad**2-np.einsum('i...,i...',twod_pos_m,twod_pos_m)), twod_pos_m[0,:], twod_pos_m[1,:]]) * 100.0
n_m = -(pos_m - np.array([m.rad*100, 0, 0])[:,np.newaxis])
n_m = n_m / np.sqrt(np.einsum('i...,i...', n_m, n_m))[np.newaxis, :]

i_det_grid, j_det_grid = np.meshgrid([29, 177], [250])
i_det = i_det_grid.flatten()
j_det = j_det_grid.flatten()

xi_v = (det.x2 - det.x0) / np.linalg.norm(det.x2 - det.x0) * det.size[1]
zeta_v = (det.x1 - det.x0) / np.linalg.norm(det.x1 - det.x0) * det.size[1]

r_dm0_m = -det.x0[:, np.newaxis] - np.outer(xi_v,i_det) - np.outer(zeta_v,j_det)
r_dm_m = r_dm0_m[:,:,np.newaxis] + pos_m[:,np.newaxis,:]
l_dm_m = r_dm_m / np.sqrt(np.einsum('i...,i...', r_dm_m, r_dm_m))[np.newaxis, :, :]

mirror_lam = -np.einsum('i...,i...', n_m, l_dm_m)*m.bragg.twod

mlam_grid_0 = mirror_lam[0,:].reshape(m_x_grid.shape) - lam0_grid[250, 29]
mlam_grid_1 = mirror_lam[1,:].reshape(m_x_grid.shape) - lam0_grid[250, 177]

# %% Plot mirror

plt.figure()
plt.pcolormesh(m_x_grid, m_y_grid, mlam_grid_1)
plt.axis('equal')
plt.colorbar()

# %% Plot mean, instrumental temperature as function of vignetting

means = [np.mean(mlam_grid_1[:,:i]) for i in range(5, 50)]
var = [np.var(mlam_grid_1[:,:i]) for i in range(5, 50)]

plt.plot(range(5, 50), var)
plt.plot(range(5, 50), means)

plt.hist(mlam_grid_1.flatten(), bins=50)

# %% Plot wavelengths

plt.scatter(mirror_lam[1,:], m_x_grid)

# %% Calculate single pixel instrument function

detc = 177
detr = 250

m_x_grid, m_y_grid = np.meshgrid(np.linspace(-1, 1, 64)*m.size[0]/2.0, np.linspace(-1, 1, 64)*m.size[1]/2.0)
twod_pos_m = np.array([m_x_grid.flatten(), m_y_grid.flatten()])
z_m = np.sqrt(m.rad**2-np.einsum('i...,i...',twod_pos_m,twod_pos_m))
pos_m = np.array([m.rad-z_m, twod_pos_m[0,:], twod_pos_m[1,:]]) * 100.0
n_m = -(pos_m - np.array([m.rad*100, 0, 0])[:,np.newaxis])
n_m = n_m / np.sqrt(np.einsum('i...,i...', n_m, n_m))[np.newaxis, :]

area_m = m.rad/z_m

i_det_grid, j_det_grid = np.meshgrid(np.linspace(0.0, 1.0, 8)+detc, np.linspace(0.0, 1.0, 8)+detr)
i_det = i_det_grid.flatten()
j_det = j_det_grid.flatten()

area_m = np.array([area_m]*len(i_det))

xi_v = (det.x2 - det.x0) / np.linalg.norm(det.x2 - det.x0) * det.size[1]
zeta_v = (det.x1 - det.x0) / np.linalg.norm(det.x1 - det.x0) * det.size[1]

r_dm0_m = -det.x0[:, np.newaxis] - np.outer(xi_v,i_det) - np.outer(zeta_v,j_det)
r_dm_m = r_dm0_m[:,:,np.newaxis] + pos_m[:,np.newaxis,:]
s2_dm_m = np.einsum('i...,i...', r_dm_m, r_dm_m)
l_dm_m = r_dm_m / np.sqrt(s2_dm_m)[np.newaxis, :, :]

area_m *= -np.einsum('i...,i...', n_m, l_dm_m) / s2_dm_m

inst_lam_1 = -np.einsum('i...,i...', n_m, l_dm_m)*m.bragg.twod

# %% Instrument function

mean = np.average(inst_lam_1.flatten(), weights=area_m.flatten())
var = np.average((inst_lam_1.flatten() - mean)**2, weights=area_m.flatten())

print var/4.29e-7 * 1000

laml = lamw

hist, bin_edges = np.histogram(inst_lam_1.flatten(), bins=63, weights=area_m.flatten(), density=True)
inst_eval = np.zeros(bin_edges.shape)
inst_eval[1:] += hist / 2.0
inst_eval[:-1] += hist/2.0

mAr = 37211326.1 # Mass of argon in keV

def singleGaussian(lam0, te):
    w = lam0**2 * te / mAr
    a = 1 / np.sqrt(w * 2 * np.pi)
    return lambda x: a * np.exp(-np.square(x - lam0) / 2 / w)

#def generateSpectra(f, lam0_eval, inst_lam, inst_height, inst_lam0):

f = singleGaussian(laml, 1.16)
lam0_eval = lam0_grid[detr,detc-15:detc+15]
inst_lam0 = lam0_grid[detr,detc]
inst_lam = bin_edges
inst_height = inst_eval

lams1, lams2 = np.meshgrid(lam0_eval, inst_lam)
lam_eval = lams2 - inst_lam0 + lams1
f_evaled = f(lam_eval)
result = inst_height.dot(f_evaled) * np.median(np.diff(inst_lam))

g = singleGaussian(laml-mean+inst_lam0, 1.16+var/(laml*laml/mAr))

plt.figure()
plt.plot(f(lam0_eval), marker='.', c='g', label='Synthetic Z line')
plt.plot(g(lam0_eval), marker='.', c='r', label='Gaussian Convolution')
plt.plot(result, marker='+', c='b', label='Instrument Function')
plt.xlabel('Pixels [+160]')
plt.ylabel('Counts [A.U.]')
plt.legend(loc='upper left')