# -*- coding: utf-8 -*-
"""
Created on Sat Jun 29 12:36:45 2019

@author: maple
"""

import numpy as np
import matplotlib.pyplot as plt

# %%

fig3_profs = np.load('fig3_data1.npy')
fig3_nedata_loc = np.load('fig3_data2a.npy')
fig3_nedata_soc = np.load('fig3_data2b.npy')
fig3_tedata = np.load('fig3_data3.npy')

fig3_fs = np.load('C:/Users/maple/git/psfc-misc/Fitting/profiles.npz')

#ax310.errorbar(r3, fig3_profs[5,:], yerr=2*fig3_profs[6,:], c='r', linewidth=0.8)
#ax310.errorbar(r3, fig3_profs[7,:], yerr=2*fig3_profs[8,:], c='b', linewidth=0.8)

#ax310.errorbar(fig3_fs['loc_ne_x'][0:max_ind:8], fig3_fs['loc_ne_a_Ly'][0:max_ind:8], yerr=fig3_fs['loc_ne_err_a_Ly'][0:max_ind:8], c='r', linewidth=0.8)
#ax310.errorbar(fig3_fs['soc_ne_x'][0:max_ind:8], fig3_fs['soc_ne_a_Ly'][0:max_ind:8], yerr=fig3_fs['soc_ne_err_a_Ly'][0:max_ind:8], c='b', linewidth=0.8)


r3 = fig3_profs[0,:]
max_ind = np.searchsorted(fig3_fs['loc_ne_x'], 1.0)

plt.plot(r3, fig3_profs[6,:]/fig3_profs[5,:])
plt.plot(fig3_fs['loc_ne_x'][0:max_ind:8], fig3_fs['loc_ne_err_a_Ly'][0:max_ind:8]/fig3_fs['loc_ne_a_Ly'][0:max_ind:8])
plt.ylim([0.0, 1.0])