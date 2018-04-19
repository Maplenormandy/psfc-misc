# -*- coding: utf-8 -*-
"""
Created on Thu May 25 17:36:53 2017

@author: normandy
"""

import numpy as np
import scipy.io
import matplotlib.pyplot as plt

import readline
import MDSplus

import eqtools

# %%

# Load the fit files; ideally we'd like to output files like this in the end
tifit = scipy.io.readsav('/home/normandy/fits/tifit_1120914036_THT1.dat')
omfit = scipy.io.readsav('/home/normandy/fits/omfit_1120914036_THT1.dat')

# Which time slice you want
tind = 7

print "plotting time", tifit.time[tind]

# Fitted Ti
tifitSlice = tifit.bsti[tind][0][0][0]
# Inverted Ti; this is the data the above fit comes from
tirawSlice = tifit.bsti[tind][0][1][0]

# I think the indices are [time][always 0][fit, raw][always 0][data type]
# tirawSlice[3] gives the type of data used (inverted vs. line-integrated, used at the edge)


plt.figure()
plt.errorbar(tifitSlice[1], tifitSlice[0], yerr=tifitSlice[2])
plt.errorbar(tirawSlice[2], tirawSlice[0], yerr=tirawSlice[1], fmt='o')
plt.xlabel('r/a (LFS midplane)')
plt.ylabel('Ti [keV]')


# %% Plot rotation just for fun
e = eqtools.CModEFITTree(1150903021)

# Fitted rotation
omfitSlice = omfit.bsom[tind][0][0][0]

toplot = e.rho2rho('r/a', 'phinorm', omfitSlice[1], tifit.time[tind])

plt.figure()
plt.errorbar(np.sqrt(toplot), omfitSlice[0], yerr=omfitSlice[2])
plt.xlabel('square root normalized toroidal flux')
plt.ylabel('omega_tor [kHz (note: kHz not krad/s, i.e. real frequency)]')