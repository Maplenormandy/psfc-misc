# -*- coding: utf-8 -*-
"""
Created on Fri Aug  3 17:17:22 2018

@author: normandy
"""

import numpy as np
import matplotlib.pyplot as plt

import scipy.interpolate
import scipy.integrate

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
r_index = 24
num_points = 32
theta0 = np.linspace(0, 2*np.pi, num_points, endpoint=False)
# Theta coordinates of the dual grid
theta0d = np.linspace(0, 2*np.pi, num_points+1, endpoint=True) + np.pi/num_points
# Used for constructing a periodic interpolation function
theta0x = np.linspace(-2*np.pi/num_points, 2*np.pi, num_points+2, endpoint=True)
x = np.array([rmag + r[r_index]*np.cos(theta0), r[r_index]*np.sin(theta0)])
v = np.zeros(x.shape)
# Flattens row major, so fastest is point, then r/z, then x/v
state = np.array([x, v])
statevec = state.flatten()


rplot, zplot = np.meshgrid(rgrid, zgrid)

#plt.contour(rgrid, zgrid, fluxgrid)
#plt.plot(R0, Z0)
#plt.axis('equal')

flux = scipy.interpolate.RectBivariateSpline(rgrid, zgrid, fluxgrid.T, kx=2, ky=2, s=0)
psi = flux.ev(rmid, np.zeros(rmid.shape))
psi0 = psi[r_index]

# %% Start evolving points
k_flux = 100.0
k_straight = 0.02
b_drag = 10.0
def deriv(y, t):
    s = np.reshape(y, state.shape)
    r = s[0,0,:]
    z = s[0,1,:]
    
    # Enforce periodicity the lazy way
    rx = np.concatenate(([r[-1]], r, [r[0]]))
    zx = np.concatenate(([z[-1]], z, [z[0]]))

    # Restoring force towards correct flux surface
    delta_flux = (psi0 - flux.ev(r, z))
    dir_flux = np.array([flux.ev(r, z, dx=1), flux.ev(r, z, dy=1)])
    f_flux = k_flux * delta_flux[np.newaxis,:] * dir_flux

    # Force between points to enforce straight field line coordinates
    rspline = scipy.interpolate.UnivariateSpline(theta0x, rx, k=2, s=0)
    zspline = scipy.interpolate.UnivariateSpline(theta0x, zx, k=2, s=0)
    
    rd = rspline(theta0d)
    zd = zspline(theta0d)
    deltar = rspline(theta0, nu=1)
    deltaz = zspline(theta0, nu=1)
    dir_theta = np.array([deltar, deltaz])
    # Calculate b_theta * nabla_theta on the dual grid
    dtheta2 = rspline(theta0d, nu=1)**2 + zspline(theta0d, nu=1)**2
    bpold2r2 = (flux.ev(rd, zd, dx=1)**2 + flux.ev(rd, zd, dy=1)**2)
    bntheta = np.sqrt(bpold2r2 / dtheta2)
    
    delta_straight = (bntheta[1:] - bntheta[:-1])
    
    f_straight = - k_straight * delta_straight[np.newaxis,:] * dir_theta
    
    f_drag = -b_drag * s[1,:,:]
    
    ds = np.zeros(state.shape)
    ds[0,:,:] = s[1,:,:]
    ds[1,:,:] = f_flux + f_straight + f_drag

    return ds.flatten()
    
    

plt.close('all')

t = np.linspace(0,100)
system = scipy.integrate.odeint(deriv, statevec, t)



#system = ode(deriv).set_integrator('lsoda', method='bdf')
#system.set_initial_value(statevec, 0)

plt.figure()

system.integrate(10)

t1 = 20
dt = 0.1
while system.successful() and system.t < t1:
    y = system.integrate(system.t+dt)
    print system.t
    sy = np.reshape(y, state.shape)
    
    s = np.reshape(y, state.shape)
    r = s[0,0,:]
    z = s[0,1,:]
    
    # Enforce periodicity the lazy way
    rx = np.concatenate(([r[-1]], r, [r[0]]))
    zx = np.concatenate(([z[-1]], z, [z[0]]))
    
    # Restoring force towards correct flux surface
    delta_flux = (psi0 - flux.ev(r, z))
    dir_flux = np.array([flux.ev(r, z, dx=1), flux.ev(r, z, dy=1)])
    f_flux = k_flux * delta_flux[np.newaxis,:] * dir_flux
    
    # Force between points to enforce straight field line coordinates
    rspline = scipy.interpolate.UnivariateSpline(theta0x, rx, k=2, s=0)
    zspline = scipy.interpolate.UnivariateSpline(theta0x, zx, k=2, s=0)
    
    rd = rspline(theta0d)
    zd = zspline(theta0d)
    deltar = rspline(theta0, nu=1)
    deltaz = zspline(theta0, nu=1)
    dir_theta = np.array([deltar, deltaz])
    # Calculate b_theta * nabla_theta on the dual grid
    dtheta2 = rspline(theta0d, nu=1)**2 + zspline(theta0d, nu=1)**2
    bpold2r2 = (flux.ev(rd, zd, dx=1)**2 + flux.ev(rd, zd, dy=1)**2)
    bntheta = np.sqrt(bpold2r2 / dtheta2)
    
    delta_straight = (bntheta[1:] - bntheta[:-1])
    
    f_straight = - k_straight * delta_straight[np.newaxis,:] * dir_theta
    
    f_drag = -b_drag * s[1,:,:]
    
    ds = np.zeros(state.shape)
    ds[0,:,:] = s[1,:,:]
    ds[1,:,:] = f_flux + f_straight + f_drag
    
    plt.plot(bntheta)
    #plt.figure()
    #plt.pcolormesh(rgrid, zgrid, fluxgrid)
    #plt.plot(sy[0,0,:], sy[0,1,:], marker='o')
    #plt.axis('equal')
    #plt.show()
    
# %%
    
plt.figure()
plt.contour(rgrid, zgrid, fluxgrid, 32)
plt.plot(sy[0,0,:], sy[0,1,:], marker='o')
plt.axis('equal')
plt.show()
    
# %%


