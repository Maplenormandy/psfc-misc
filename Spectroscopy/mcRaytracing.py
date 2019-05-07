# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 14:42:33 2016

@author: normandy
"""

import numpy as np
import matplotlib.pyplot as plt

from collections import namedtuple


import readline
import MDSplus

import eqtools

import itertools

plotting = False

# %% Info file named tuple definitions, for convenience
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

# Ray in cylindrical coordinates given two points in cylindrical coordinates
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

def getInfoFile(shot, module):
    specTree = MDSplus.Tree('spectroscopy', shot)
    modNode = specTree.getNode(r'\SPECTROSCOPY::TOP.HIREXSR.INFO.' + module)
    mirrorNode = modNode.getNode('MIRROR')
    braggNode = mirrorNode.getNode('BRAGG')
    detNode = modNode.getNode('DET')
    
    # Load info file from the tree
    mirror = InfoMirror(mirrorNode.getNode('vec').data(), # [R, theta, Z] of crystal location [m]
                        mirrorNode.getNode('rot').data(), # [alpha, beta, gamma] rotation angles [rad]
                        mirrorNode.getNode('size').data(), # [y, z] size of crystal [m]
                        InfoBragg(braggNode.getNode('iref').data(),
                                  braggNode.getNode('rwid').data(),
                                  braggNode.getNode('twod').data()
                                  ), # [integrated reflectivity, rocking curve, 2d spacing [Ang]]
                        mirrorNode.getNode('rad').data() # radius of crystal curvature [m]
                        )
    
    
    det = InfoDetector(detNode.getNode('x0').data(), # position of origin of det system [m]
                       detNode.getNode('x1').data(), # position of point along zeta axis [m]
                       detNode.getNode('x2').data(), # position of point along xi axis [m]
                       detNode.getNode('XI_O').data(), # xi values of pixel centers, unused
                       detNode.getNode('ZETA_O').data(), # zeta values of pixel centers, unused
                       np.array([detNode.getNode('det_xi').data(),detNode.getNode('det_zeta').data()]), # [xi, zeta] size of pixels [m]
                       detNode.getNode('n_xi').data(), # number of xi channels
                       detNode.getNode('n_zeta').data() # number of zeta channels
                       )
                       
    return mirror, det

# This info file is from 1150903028
"""
mirrorHe = InfoMirror(np.array([3.687, 0, 0.0004]), # [R, theta, Z] of crystal location [m]
                      np.array([0.0, -0.00174533, 2.025]), # [alpha, beta, gamma] rotation angles [rad]
                      np.array([0.064, 0.027]), # [y, z] size of crystal [m]
                      InfoBragg(1.0, 0.0, 4.56216), # [integrated reflectivity, rocking curve, 2d spacing [Ang]]
                      1.442) # radius of crystal curvature
det2 = InfoDetector(np.array([111.082, -60.73, -3.0108])/100.0, # position of origin of det system
                    np.array([111.083, -60.7308, 5.36560])/100.0, # position of point along zeta axis
                    np.array([109.396, -63.6292, -3.01080])/100.0, # position of point along xi axis
                    -1, # xi values of pixel centers, unused
                    -1, # zeta values of pixel centers, unused
                    np.array([0.0172, 0.0172])/100.0, # [xi, zeta] size of pixels [m]
                    195, # number of xi channels
                    487) # number of zeta channels

det1 = InfoDetector(np.array([111.196, -60.7343, 5.68298])/100.0, # position of origin of det system
                    np.array([111.202, -59.5147, 13.9701])/100.0, # position of point along zeta axis
                    np.array([109.532, -63.6150, 6.10806])/100.0, # position of point along xi axis
                    -1, # xi values of pixel centers, unused
                    -1, # zeta values of pixel centers, unused
                    np.array([0.0172, 0.0172])/100.0, # [xi, zeta] size of pixels [m]
                    195, # number of xi channels
                    487) # number of zeta channels
"""

# Spectral lines
lamz = 3.9941451
lamw = 3.9490665

# %% Get the magnetic geometry

"""
e = eqtools.CModEFITTree(shot)
rgrid = e.getRGrid()
zgrid = e.getZGrid()

plt.close("all")

flux = e.getFluxGrid()
flcfs = e.getFluxLCFS()
levels = np.linspace(np.min(flux[35,:,:]), flcfs[35], 8)

if plotting:
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
"""

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
    pos_m = np.array([m.rad-z_m, twod_pos_m[0,:], twod_pos_m[1,:]])
    
    # mirror normal, mirror coordinates
    n_m = -(pos_m - np.array([m.rad, 0, 0])[:,np.newaxis])
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
        d_m_t = R_mt.dot(pos_m)
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


# %% Full integrated code

def calculateWavelengths(shot, module, plotting=True):
    mirror, det = getInfoFile(shot, module)
    m_y = np.array([0.0])
    m_z = np.array([0.0])
    
    det_xi_grid, det_zeta_grid = np.meshgrid(np.array(range(det.n_xi)), np.array(range(det.n_zeta)))
    lam1, weight1 = raytrace(mirror, det, m_y, m_z, det_xi_grid.flatten(), det_zeta_grid.flatten())
    lam1_grid = lam1.reshape(det_xi_grid.shape)
    
    if plotting:
        plt.figure()
        plt.pcolormesh(det_xi_grid, det_zeta_grid, lam1_grid)
        plt.colorbar()
        
    return np.flipud(lam1_grid.T)

def calculateInstrumentals(shot, module, plotting=True, writeToTree=False):
    mirror, det = getInfoFile(shot, module)
    m_y = np.array([0.0])
    m_z = np.array([0.0])
    
    det_xi_grid, det_zeta_grid = np.meshgrid(np.array(range(det.n_xi)), np.array(range(det.n_zeta)))
    lam1, weight1 = raytrace(mirror, det, m_y, m_z, det_xi_grid.flatten(), det_zeta_grid.flatten())
    lam1_grid = lam1.reshape(det_xi_grid.shape)
    
    if plotting:
        plt.figure()
        plt.pcolormesh(det_xi_grid, det_zeta_grid, lam1_grid)
        plt.colorbar()
    
    
    # Use n-th order Gauss-Legendre quadrature to integrate over mirror
    m_range, m_weights = np.polynomial.legendre.leggauss(3)
    m_x_grid, m_y_grid = np.meshgrid(m_range*mirror.size[0]/2.0, m_range*mirror.size[1]/2.0)
    m_grid_weights = np.outer(m_weights, m_weights).flatten()
    
    m0_grid = np.zeros((487, 195))
    m1_grid = np.zeros((487, 195))
    inst_temp_grid = np.zeros((487, 195))
    m2_grid = np.zeros((487, 195))
    m3_grid = np.zeros((487, 195))
    
    # Use n-th order Gauss-Legendre quadrature when integrating over pixel as well
    p_range, p_weights = np.polynomial.legendre.leggauss(3)
    p_range = p_range/2.0 + 0.5
    p_weights = p_weights * 0.5
    
    for row in range(487):
        if row%10 == 0:
            print row
        for col in range(195):
            plam1, pweight1 = raytrace(mirror, det, m_x_grid.flatten(), m_y_grid.flatten(), p_range+col, p_range+row)
            
            m0, m1, m2, m3, inst_vel, inst_temp = momentAnalysis(plam1, pweight1*m_grid_weights*p_weights[:,np.newaxis], lamz)
            m0_grid[row, col] = m0
            m1_grid[row, col] = m1
            m2_grid[row, col] = m2
            m3_grid[row, col] = m3
            inst_temp_grid[row, col] = inst_temp
    
    if plotting:
        plt.figure()
        plt.pcolormesh(m1_grid[:,:], det_zeta_grid, (m1_grid[:,:]-lam1_grid)*1e3)
        plt.colorbar()
        
        
        plt.figure()
        plt.pcolormesh(m1_grid[:,:], det_zeta_grid, np.sqrt(m2_grid)[:,:])
        plt.axvline(lamz)
        plt.axvline(lamw)
        plt.colorbar()

    ishift = np.zeros((det.n_xi, det.n_zeta))
    itemp = np.sqrt(m2_grid.T)
    # For whatever reason the indices are backwards in THACO
    itemp = np.flipud(itemp)
    #itemp = np.fliplr(itemp)
    print "flipped"
    
    if writeToTree:
        # Note this is taken from Mat Reinke's code directly
        newInst = MDSplus.Data.compile('build_signal(build_with_units($1,"Ang"),*,build_with_units($2,"Ang"))', itemp, ishift)
        specTree = MDSplus.Tree('spectroscopy', shot)
        instNode = specTree.getNode(r'\SPECTROSCOPY::TOP.HIREXSR.CALIB.'+module+':INST')
        instNode.putData(newInst)
    
    return lam1_grid, np.flipud(m1_grid.T), np.flipud(m2_grid.T)

def binInstrumentals(shot, module, tht):
    lamw = 3.94912
    mAr = 39.948
    #c = 2.998e+5
    
    lam1, m1, m2 = calculateInstrumentals(shot, module)
    itemp = m2 * mAr / lamw**2 * 1e6

# %% Playing with instrument functions
"""
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
"""
# %% Evaluate instrument function
"""
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
"""