# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 16:39:48 2019

@author: maple
"""

import matplotlib as mpl
import matplotlib.pyplot as plt

import numpy as np

# %%

cgyro_freqs = np.load('./cgyro_outputs/all_freqs.npz')


for foldb in reversed(cgyro_freqs['folders']):
    fold = foldb.decode("utf-8")
    data = cgyro_freqs[fold]
    
    ky = data[0,:]
    #kmax = np.searchsorted(ky, 1.42)
    kmax=len(ky)
    kmin=0
    #kmin = np.searchsorted(ky, 1.42)
    omega = data[1,:]
    gamma = data[2,:]

    c = (0.7, 0.7, 0.7)
    alpha = 1.0
    if fold == 'soc_mid':
        c = 'b'
        alpha = 1.0
    elif fold == 'loc_mid':
        c = 'r'
        alpha = 0.0
        continue
    elif fold == 'soc_low':
        c = (0.5, 0.5, 1.0)
        alpha = 0.0
        continue
    elif fold == 'loc_hig':
        c = (1.0, 0.5, 0.5)
        alpha = 0.0
        continue
    else:
        continue

    #plt.plot(omega[kmin:kmax-1], np.diff(omega[kmin:kmax])/np.diff(ky[kmin:kmax]), marker='.', c=c)
    plt.plot((ky[kmin:kmax-1] + ky[kmin+1:kmax])/2.0, np.diff(omega[kmin:kmax])/np.diff(ky[kmin:kmax]), marker='.', c=c, label='$v_{gr}$')
    plt.plot(ky[kmin:kmax], omega[kmin:kmax]/ky[kmin:kmax], marker='.', c=c, ls='--', label='$v_{res}$')
    #plt.plot(ky[kmin:kmax], omega[kmin:kmax], marker='.', c=c)
    #plt.plot(ky[kmin:kmax], gamma[kmin:kmax], marker='.', c=c)

plt.xlabel(r'$k_y \rho_s$')
plt.ylabel(r'$c_s \rho_s / a$')
plt.legend()