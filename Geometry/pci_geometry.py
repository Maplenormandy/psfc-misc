# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 15:49:59 2020

@author: normandy
"""

import numpy as np
import matplotlib.pyplot as plt

import scipy.interpolate

import eqtools

import neotools

# %%

eg = neotools.EquilibriumGeometry(1160506007, (0.57,0.63), plotting=True)

# %% Set up grids to evaluate 

# Grid on which to evaluate the interpolation.
thetagridEv = np.linspace(-np.pi*1.25, np.pi*1.25, 128)

# Interpolation functions from (psi,theta) coordinates
nu_grid = [None] * len(eg.psigridEv)
cR_grid = [None] * len(eg.psigridEv)
cZ_grid = [None] * len(eg.psigridEv)

def extend_data(x, offset=0):
    x_ex = np.zeros(len(x)*3)
    x_ex[:len(x)] = x - offset
    x_ex[len(x):len(x)*2] = x
    x_ex[len(x)*2:] = x + offset
    
    return x_ex

qfunc = scipy.interpolate.interp1d(eg.psigrid, eg.e.getQProfile()[eg.tind])

for i in range(len(eg.psigridEv)):    
    theta = eg.psicontours_rtheta[i][1,:]
    theta_ex = extend_data(theta, 2*np.pi)
    
    q = qfunc(eg.psigridEv[i])
    
    nu_grid[i] = scipy.interpolate.interp1d(theta_ex, extend_data(eg.psicontours_nu[i], 2*np.pi*q))(thetagridEv)
    cR_grid[i] = scipy.interpolate.interp1d(theta_ex, extend_data(eg.psicontours[i][0,:]))(thetagridEv)
    cZ_grid[i] = scipy.interpolate.interp1d(theta_ex, extend_data(eg.psicontours[i][1,:]))(thetagridEv)
    
nu_grid = np.array(nu_grid)
cR_grid = np.array(cR_grid)
cZ_grid = np.array(cZ_grid)

nu_func = scipy.interpolate.RectBivariateSpline(eg.psigridEv, thetagridEv, nu_grid, kx=2, ky=2)
cR_func = scipy.interpolate.RectBivariateSpline(eg.psigridEv, thetagridEv, cR_grid, kx=2, ky=2)
cZ_func = scipy.interpolate.RectBivariateSpline(eg.psigridEv, thetagridEv, cZ_grid, kx=2, ky=2)

# %% Test geometry capability

def getEikonal(psinorm):

    thetaEv = np.linspace(-np.pi*0.75, np.pi*0.75, 128)
    psiEv = np.ones(thetaEv.shape) * psinorm
    
    dnu_dpsi = nu_func.ev(psiEv, thetaEv, dx=1)
    dnu_dtheta = nu_func.ev(psiEv, thetaEv, dy=1)
    
    dR_dtheta = cR_func.ev(psiEv, thetaEv, dy=1)
    dZ_dtheta = cZ_func.ev(psiEv, thetaEv, dy=1)
    ds2_dtheta = dR_dtheta**2 + dZ_dtheta**2
    
    dtheta_dR = dR_dtheta / ds2_dtheta
    dtheta_dZ = dZ_dtheta / ds2_dtheta
    
    cREv = cR_func.ev(psiEv, thetaEv)
    cZEv = cZ_func.ev(psiEv, thetaEv)
    
    dpsi_dR = eg.psinormfunc.ev(cREv, cZEv, dx=1)
    dpsi_dZ = eg.psinormfunc.ev(cREv, cZEv, dy=1)
    
    kR = ( dnu_dpsi * dpsi_dR + dnu_dtheta * dtheta_dR ) / 100.0 # in cm^-1
    kZ = ( dnu_dpsi * dpsi_dZ + dnu_dtheta * dtheta_dZ ) / 100.0 # in cm^-1
    
    return cREv, cZEv, kR, kZ


#plt.quiver(cREv, cZEv, kR, kZ)
#plt.axis('equal')

# R = 0.675-0.705 m [High-k configuration] of PCI

# %% Load data

cgyro_data = np.load(file('/home/normandy/git/psfc-misc/PresentationScripts/hysteresis_pop/1120216_cgyro_data.npz'))

case1_transp = scipy.io.netcdf.netcdf_file('/home/normandy/hysteresis_transp/12345B05/12345B05.CDF')
case2_transp = scipy.io.netcdf.netcdf_file('/home/normandy/hysteresis_transp/12345A05/12345A05.CDF')

# %% Plot synthetic spectra

ky_max = 16

omega_csoa = cgyro_data['omega'][1,1,:ky_max]
roa_plot = cgyro_data['radii'][1,1]


ky_rhos = cgyro_data['ky'][0,:ky_max]

r_plot = roa_plot*21.878 # r in cm

transp=case2_transp
t_plot = 1.21

time = transp.variables['TIME'].data
tind = np.searchsorted(time, t_plot)
roa = transp.variables['RMNMP'].data[tind,:] / 21.878

rot_omega = transp.variables['OMEGA'].data[tind,:]

#shear = transp.variables['SREXB_NCL'].data
te = transp.variables['TE'].data[tind,:]

cs = np.sqrt(te) * 692056.111 # c_s in cm/s
gyfreq = 47.8941661e6 * 5.4 # gyrofrequency in Hz

rhos = cs / gyfreq # rho_s in cm
csoa = cs / 21.878 / 1e3 # c_s / a in kHz

rhos_plot = scipy.interpolate.interp1d(roa, rhos)(roa_plot)
csoa_plot = scipy.interpolate.interp1d(roa, csoa)(roa_plot)

qfunc = scipy.interpolate.interp1d(eg.psigrid, eg.e.getQProfile()[eg.tind])
psi_plot = eg.e.roa2psinorm(roa_plot, t_plot)
q_plot = qfunc(psi_plot)

rot_omega_plot = scipy.interpolate.interp1d(roa, rot_omega)(roa_plot)

plt.figure()

cR, cZ, kR, kZ = getEikonal(psi_plot)
eik_plot = np.logical_and(cR<0.705+0.05, cR>0.675+0.05)
#eik_plot = np.logical_and(cR<0.735, cR>0.645)

kR_visible = kR[eik_plot]

for i in range(ky_max):
    # ky = n * q / r, n = r * ky / q
    n_plot = ( ky_rhos[i] / rhos_plot ) * r_plot / q_plot
    
    omega_plot = omega_csoa[i] * csoa_plot + n_plot * (rot_omega_plot / 1e3)
    kR_plot = n_plot * kR_visible * np.sign(omega_plot)    
    freq_plot = np.ones(kR_plot.shape)*np.abs(omega_plot/2/np.pi)
    
    
    if omega_csoa[i] < 0:
        plt.scatter(kR_plot, freq_plot, marker='v', c='b')
    else:
        plt.scatter(kR_plot, freq_plot, marker='^', c='r')
        
plt.xlim([-25,25])
plt.ylim([0, 750])