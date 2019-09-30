# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 14:43:57 2019

@author: normandy
"""

import numpy as np
import sys

sys.path.append('/home/normandy/git/psfc-misc/Fitting')
import profiles_fits

import matplotlib.pyplot as plt
from matplotlib import cm

import cPickle as pkl

# %%

dt = 0.06

def fit_shot(shot):
    fits = {}
    times = np.arange(0.5, 1.5, dt)
    fits['time'] = (times[1:] + times[:-1])/2
    fits['ne_fits'] = [None] * len(fits['time'])
    fits['te_fits'] = [None] * len(fits['time'])
    fits['ti_fits'] = [None] * len(fits['time'])
    fits['vtor_fits'] = [None] * len(fits['time'])
    
    for i in range(len(fits['time'])):
        print '=== Fitting: ' + str(fits['time'][i]) + ' ==='
        fits['ne_fits'][i] = profiles_fits.get_ne_fit(shot=shot, t_min=times[i], t_max=times[i+1], plot=False, x0_mean=0.45)
        fits['te_fits'][i] = profiles_fits.get_te_fit(shot=shot, t_min=times[i], t_max=times[i+1], plot=False, x0_mean=0.45)
        fits['ti_fits'][i] = profiles_fits.get_ti_fit(shot=shot, t_min=times[i], t_max=times[i+1], plot=False, x0_mean=0.45, te_fit=fits['te_fits'][i])
        fits['vtor_fits'][i] = profiles_fits.get_vtor_fit(shot=shot, t_min=times[i], t_max=times[i+1], plot=False, x0_mean=0.45)
        
    return fits
        
shot = 1120216017
result = fit_shot(shot)

# %%

plt.close('all')


plt.figure()
plt.axhline(c='k', ls='--')
for i in range(len(result['time'])):
    plt.plot(result['ne_fits'][i]['X'], result['ne_fits'][i]['y'], c=cm.plasma(result['time'][i]-0.5), label=result['time'][i])
plt.legend()

plt.figure()
plt.axhline(c='k', ls='--')
for i in range(len(result['time'])):
    plt.plot(result['te_fits'][i]['X'], result['te_fits'][i]['y'], c=cm.plasma(result['time'][i]-0.5), label=result['time'][i])
plt.legend()

plt.figure()
plt.axhline(c='k', ls='--')
for i in range(len(result['time'])):
    plt.plot(result['ti_fits'][i]['X'], result['ti_fits'][i]['y'], c=cm.plasma(result['time'][i]-0.5), label=result['time'][i])
plt.legend()

plt.figure()
plt.axhline(c='k', ls='--')
for i in range(len(result['time'])):
    plt.plot(result['vtor_fits'][i]['X'], result['vtor_fits'][i]['y'], c=cm.plasma(result['time'][i]-0.5), label=result['time'][i])
plt.legend()

# %%

pkl.dump(result, open('/home/normandy/git/psfc-misc/Fitting/result_%d.pkl'%shot, 'w'), protocol=pkl.HIGHEST_PROTOCOL)