# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 15:49:59 2020

@author: normandy
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

import scipy.interpolate

import eqtools

import neotools

# %%

shot = 1160506009
eg = neotools.EquilibriumGeometry(shot, (0.57,0.63), plotting=True)

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

theta0 = 0.0

for i in range(len(eg.psigridEv)):    
    theta = eg.psicontours_rtheta[i][1,:]
    theta_ex = extend_data(theta, 2*np.pi)
    
    q = qfunc(eg.psigridEv[i])
    
    nu_grid[i] = scipy.interpolate.interp1d(theta_ex, extend_data(eg.psicontours_nu[i]+theta0*q, 2*np.pi*q))(thetagridEv)
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

case07_transp = scipy.io.netcdf.netcdf_file('/home/normandy/hysteresis_transp/12345B05/12345B05.CDF')
case09_transp = scipy.io.netcdf.netcdf_file('/home/normandy/hysteresis_transp/12345A05/12345A05.CDF')

pci_detector_response = np.loadtxt(open('./pci_detector_response.csv'), delimiter=',')
pci_resp_func = scipy.interpolate.interp1d(pci_detector_response[:,0], pci_detector_response[:,1])


ky_max = 14

pos_power = np.zeros((2,6,ky_max))
neg_power = np.zeros((2,6,ky_max))

# %% Plot PCI plots


plt.figure()

case='1.1 MA SOC'

if case == '0.8 MA LOC':
    t_pci = 0.96
    t_synth = 0.95
    transp = case07_transp
    cgyro_ind=0
    loc_ind = 1
elif case == '0.8 MA SOC':
    t_pci = 0.6
    t_synth = 1.37
    transp = case07_transp
    cgyro_ind=0
    loc_ind = 0
if case == '1.1 MA LOC':
    t_pci = 0.92
    t_synth = 1.25
    transp = case09_transp
    cgyro_ind=1
    loc_ind = 1
elif case == '1.1 MA SOC':
    t_pci = 0.72
    t_synth = 1.43
    transp = case09_transp
    cgyro_ind=1
    loc_ind = 0

def shiftByHalf(x):
    y = np.zeros(len(x)+1)
    y[1:-1] = (x[1:] + x[:-1])/2
    y[0] = 2*x[0] - y[1]
    y[-1] = 2*x[-1] - y[-2]

    return y
    
    
idlsav = scipy.io.readsav('/home/normandy/%d_pci.sav'%shot)

t = idlsav.spec.t[0]
f = idlsav.spec.f[0]
k = idlsav.spec.k[0]
s = idlsav.spec.spec[0]
#shot = idlsav.spec.shot[0]

t0 = np.searchsorted(t, t_pci)

kplot = shiftByHalf(k)
#kplot = np.concatenate((k, [-k[0]]))
fplot = shiftByHalf(f)

plt.pcolormesh(kplot, fplot, s[t0,:,:], cmap='cubehelix', norm=mpl.colors.LogNorm(vmin=1e-6, vmax=1e-1), rasterized=True)


spec_func = scipy.interpolate.RectBivariateSpline(k, f, s[t0,:,:].T, kx=2, ky=2)



for r in range(6):
    
    omega_csoa = cgyro_data['omega'][cgyro_ind,r,:ky_max]
    roa_plot = cgyro_data['radii'][cgyro_ind,r]
    
    
    ky_rhos = cgyro_data['ky'][0,:ky_max]
    
    r_plot = roa_plot*21.878 # r in cm
    
    
    
    
    time = transp.variables['TIME'].data
    tind = np.searchsorted(time, t_synth)
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
    psi_plot = eg.e.roa2psinorm(roa_plot, t_synth)
    q_plot = qfunc(psi_plot)
    
    rot_omega_plot = scipy.interpolate.interp1d(roa, rot_omega)(roa_plot)
    
    
    
    cR, cZ, kR, kZ = getEikonal(psi_plot)
    eik_plot = np.logical_and(cR<0.705+0.04, cR>0.675+0.04)
    #eik_plot = np.logical_and(cR<0.705, cR>0.695)
    #eik_plot = np.logical_and(cR<0.735, cR>0.645)
    
    kR_visible = kR[eik_plot]
    
    for i in range(ky_max):
        # ky = n * q / r, n = r * ky / q
        n_plot = ( ky_rhos[i] / rhos_plot ) * r_plot / q_plot
        
        omega_plot = omega_csoa[i] * csoa_plot - n_plot * (rot_omega_plot / 1e3)
        kR_plot = n_plot * kR_visible * np.sign(omega_plot)    
        freq_plot = np.ones(kR_plot.shape)*np.abs(omega_plot/2/np.pi)    
        
        if omega_csoa[i] < 0:
            plt.scatter(kR_plot[kR_plot>0], freq_plot[kR_plot>0], marker='^', c='b')
            plt.scatter(kR_plot[kR_plot<0], freq_plot[kR_plot<0], marker='v', c='b')
        else:
            plt.scatter(kR_plot[kR_plot>0], freq_plot[kR_plot>0], marker='^', c='r')
            plt.scatter(kR_plot[kR_plot<0], freq_plot[kR_plot<0], marker='v', c='r')
            
        pos_power[loc_ind,r,i] = np.sum(spec_func(kR_plot[kR_plot>0], freq_plot[kR_plot>0], grid=False)/pci_resp_func(freq_plot[kR_plot>0]))
        neg_power[loc_ind,r,i] = np.sum(spec_func(kR_plot[kR_plot<0], freq_plot[kR_plot<0], grid=False)/pci_resp_func(freq_plot[kR_plot<0]))

        
plt.xlim([-30,30])
plt.ylim([0, 1250])
plt.title(case)

# %%

plt.figure()
plt.loglog(ky_rhos[:ky_max], pos_power[0,2,:]*ky_rhos[:ky_max], c='b')
plt.loglog(ky_rhos[:ky_max], neg_power[0,2,:]*ky_rhos[:ky_max], c='b', ls='--')
plt.loglog(ky_rhos[:ky_max], pos_power[1,2,:]*ky_rhos[:ky_max], c='r')
plt.loglog(ky_rhos[:ky_max], neg_power[1,2,:]*ky_rhos[:ky_max], c='r', ls='--')
