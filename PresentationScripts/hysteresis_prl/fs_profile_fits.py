# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 18:47:09 2018

@author: normandy
"""

import numpy as np

import matplotlib.pyplot as plt
import matplotlib as mpl

# %%

data = np.load('C:/Users/maple/git/psfc-misc/Fitting/profiles.npz')
# %%

max_ind = np.searchsorted(data['loc_ne_x'], 1.0)

plt.figure()
plt.errorbar(data['loc_ne_x'][:max_ind], data['loc_ne_a_Ly'][:max_ind], yerr=data['loc_ne_err_a_Ly'][:max_ind], c='r')
plt.errorbar(data['soc_ne_x'][:max_ind], data['soc_ne_a_Ly'][:max_ind], yerr=data['soc_ne_err_a_Ly'][:max_ind], c='b')
plt.errorbar(data['old_ne_x'][:max_ind], data['old_ne_a_Ly'][:max_ind], yerr=data['old_ne_err_a_Ly'][:max_ind], c='g')

plt.figure()
plt.errorbar(data['loc_ne_x'][:max_ind], data['loc_ne_y'][:max_ind], yerr=data['loc_ne_err_y'][:max_ind], c='r')
plt.errorbar(data['soc_ne_x'][:max_ind], data['soc_ne_y'][:max_ind], yerr=data['soc_ne_err_y'][:max_ind], c='b')
plt.errorbar(data['old_ne_x'][:max_ind], data['old_ne_y'][:max_ind], yerr=data['old_ne_err_y'][:max_ind], c='g')