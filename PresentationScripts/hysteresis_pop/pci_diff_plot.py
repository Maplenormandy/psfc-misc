# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 13:58:02 2020

Note that the order of figures in the code does not necessarily reflect the order
of figures in the paper

@author: normandy
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import numpy as np
import scipy.io, scipy.signal

import cPickle as pkl

import MDSplus

# %% Get PCI data


idlsav = scipy.io.readsav('/home/normandy/1160506007_pci.sav')


t = idlsav.spec.t[0]
f = idlsav.spec.f[0]
k = idlsav.spec.k[0]
s = idlsav.spec.spec[0]

t_loc = 0.96
t_soc = 0.6

#t_loc = 0.92
#t_soc = 0.72

t0, t1 = np.searchsorted(t, (t_soc, t_loc))

def shiftByHalf(x):
    y = np.zeros(len(x)+1)
    y[1:-1] = (x[1:] + x[:-1])/2
    y[0] = 2*x[0] - y[1]
    y[-1] = 2*x[-1] - y[-2]

    return y

kplot = shiftByHalf(k)
#kplot = np.concatenate((k, [-k[0]]))
fplot = shiftByHalf(f)

# Calculate conditional spectra
s_soc = s[t0,:,:]
sc_soc = s_soc / np.sum(s_soc, axis=1)[:,np.newaxis]
s_loc = s[t1,:,:]
sc_loc = s_loc / np.sum(s_loc, axis=1)[:,np.newaxis]

# Plot difference in conditional spectra
plt.figure()
plt.pcolormesh(kplot, fplot, sc_loc-sc_soc, cmap='PRGn', vmin=-0.05, vmax=0.05)
plt.colorbar()
plt.draw()
