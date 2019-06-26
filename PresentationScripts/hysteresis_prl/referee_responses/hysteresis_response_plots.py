# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 14:44:53 2019

@author: normandy
"""

import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.append('/home/sciortino/ML/machinelearnt')
sys.path.append('/home/sciortino/ML')
sys.path.append('/home/sciortino/shot_analysis_tools')

import cPickle as pkl

from transport_classes import prof_object
from advanced_fitting import profile_fit_fs

"""
# %%

shot=1160506007
data_soc = np.load('/home/normandy/git/bsfc/bsfc_fits/fit_data/mf_%d_nl_nh%d_t%d.npz'%(shot,3,10))
data_loc = np.load('/home/normandy/git/bsfc/bsfc_fits/fit_data/mf_%d_nl_nh%d_t%d.npz'%(shot,3,46))

# %%

chans = range(data_soc['meas_avg'].shape[1])

#data_soc['meas_avg'][0,8] = np.nan
#data_loc['meas_avg'][0,8] = np.nan

bsoc = data_soc['meas_avg'][0,:]
bloc = data_loc['meas_avg'][0,:]
bsoc[8] = np.nan
bloc[8] = np.nan

ns = np.nansum(bsoc)
nl = np.nansum(bloc)

plt.figure()
plt.errorbar(chans, bsoc/ns, yerr=data_soc['meas_std'][0,:]/ns, c='b', label='SOC, t=0.6s')
plt.errorbar(chans, bloc/nl, yerr=data_loc['meas_std'][0,:]/nl, c='r', label='LOC, t=0.96s')
plt.xlabel('Spatial Channel #')
plt.ylabel('Normalized Line-Integrated Emissivity')
plt.legend(loc='upper left')
"""

# %%

with open('/home/normandy/git/psfc-misc/Fitting/mcmc/averaging/ne_prof_1160506007_FS.pkl') as f:
    ne_fit = pkl.load(f)