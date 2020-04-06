# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 14:25:14 2019

@author: normandy
"""

import numpy as np
import matplotlib.pyplot as plt

import scipy.interpolate
import scipy.integrate

import readline
readline
import MDSplus

import eqtools

import sys
sys.path.append('/home/normandy/git/psfc-misc/Fitting')

from profiles_fits import get_ne_fit

from neotools import EquilibriumGeometry


import cPickle as pkl


# %%

shot=1160902018

#tifit = scipy.io.readsav('/home/normandy/fits/tifit_%d_THT0.dat'%shot)
omfit = scipy.io.readsav('/home/normandy/fits/omfit_%d_THT0.dat'%shot)

try:
    nefits = pkl.load(file('/home/normandy/git/psfc-misc/Geometry/angular_momentum_fits/ne_fits_%d.pkl'%shot))
except:
    nefits = [None]*50
    
    for i in range(50):
        t0 = 0.5+i*0.02
        t1 = 0.5+(i+1)*0.02
        nefits[i] = get_ne_fit(shot=shot, t_min=t0, t_max=t1, x0_mean=0.95, dst='/home/normandy/git/psfc-misc/Geometry/angular_momentum_fits', plot=False)
    
    pkl.dump(nefits, file('/home/normandy/git/psfc-misc/Geometry/angular_momentum_fits/ne_fits_%d.pkl'%shot, 'w'))


# %%

# %%

e = eqtools.CModEFITTree(shot)
mD = 3.345e-27

max_t = 34

angular_momentum_density = [None]*max_t

for i in range(max_t):
    t0 = 0.5+i*0.02
    t1 = 0.5+(i+1)*0.02
    t = (t0+t1)/2
    print i
    
    eg = EquilibriumGeometry(shot, (t0, t1), e=e)
    
    omfitSlice = omfit.bsom[i][0][0][0]
    
    ne = scipy.interpolate.interp1d(e.roa2psinorm(nefits[i]['X'], t), nefits[i]['y'])(eg.psigridEv)
    omegator = scipy.interpolate.interp1d(eg.e.roa2psinorm(omfitSlice[1], t), omfitSlice[0], fill_value='extrapolate')(eg.psigridEv)*2*np.pi
    
    moment = eg.fs_integrate(lambda R,Z,psi: R**2)*2*np.pi
    
    angular_momentum_density[i] = ne*1e20*mD*moment*omegator*1e3
    
l_density = np.array(angular_momentum_density)
# %%
    
l_total = np.sum(eg.dpsigridEv[np.newaxis,:]*l_density, axis=1)

time = np.linspace(0.51, 1.49, 50)
time = time[:max_t]

plt.figure()
plt.plot(time, l_total, marker='.')
plt.xlabel('time [sec]')
plt.ylabel('Total Angular Momentum [N m s]')

plt.yticks(np.arange(-0.035, 0.015, 0.005))
plt.xticks(np.arange(0.4, 1.65, 0.1))
plt.axes().yaxis.grid()
plt.axes().xaxis.grid()
plt.title(str(shot))

# %%
plt.figure()
plt.plot(eg.psigridEv, l_density[1,:])
plt.xlabel(r'$\psi_{\mathrm{norm}}$')
plt.ylabel(r"Angular Momentum Density (kind of) $dL/d\psi_{\mathrm{norm}}$")
plt.title(str(shot) + ' t=' + str(time[1]))


# %%

i=25
omfitSlice = omfit.bsom[i][0][0][0]
t0 = 0.5+i*0.02
t1 = 0.5+(i+1)*0.02
t = (t0+t1)/2

moment = eg.fs_integrate(lambda R,Z,psi: R**2)*2*np.pi

volumeTrue = np.sum(eg.fs_integrate(lambda R,Z,psi: np.ones(R.shape))*eg.dpsigridEv)*2*np.pi
volumeEst = np.pi * 0.21**2 * 1.8 * 2 * np.pi * 0.67


eg = EquilibriumGeometry(shot, (t0, t1), e=e)
plt.figure()
plt.plot(eg.e.roa2psinorm(omfitSlice[1], t), omfitSlice[0]*2*np.pi)

# %%
"""
# fs_integrate calculates the integral over poloidal angle, and 2 * np.pi calculates the integral over toroidal angle
# Also, units have been converted into m^{-3} by the normalization of the flux coordinate
moment = eg.fs_integrate(lambda R,Z,psi: R**2)*2*np.pi
ne = scipy.interpolate.interp1d(eg.e.roa2psinorm(ne_fit['X'], result['time'][i]), ne_fit['y'])(eg.psigridEv)
omegator = scipy.interpolate.interp1d(eg.e.roa2psinorm(omegator_fit['X'], result['time'][i]), omegator_fit['y'])(eg.psigridEv)

mD = 3.345e-27

plt.plot(eg.psigridEv, ne*1e20*mD*moment*omegator*1e3)
"""

# %% Save data to csv file

np.savetxt('/home/normandy/%d_angular_momentum.csv'%shot, np.array([time, l_total]).T, delimiter=',', header='time [s],angular_momentum [N m s]')