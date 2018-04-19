# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 14:50:40 2016

@author: normandy
"""

import numpy as np

import readline
import MDSplus

import matplotlib.pyplot as plt

# %% load spectroscopic data

specTree = MDSplus.Tree('spectroscopy', 1121002022)
#mod1Node = specTree.getNode(r'\SPECTROSCOPY::TOP.HIREX_SR.RAW_DATA:MOD1')
mod2Node = specTree.getNode(r'\SPECTROSCOPY::TOP.HIREX_SR.RAW_DATA:MOD4')
#mod3Node = specTree.getNode(r'\SPECTROSCOPY::TOP.HIREX_SR.RAW_DATA:MOD3')
#mod4Node = specTree.getNode(r'\SPECTROSCOPY::TOP.HIREX_SR.RAW_DATA:MOD4')
#mod1Data = mod1Node.data()
mod2Data = mod2Node.data()
#mod3Data = mod3Node.data()
#mod4Data = mod4Node.data()

time = int(mod2Data.shape[0]/2)

plt.pcolormesh(mod2Data[time,:,:], vmax=500)
plt.colorbar()
plt.title(str(time))

#, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
#ax1.pcolormesh(mod1Data[time,:,:], vmax=500)
#ax2.pcolormesh(mod2Data[time,:,:], vmax=500)
#ax3.pcolormesh(mod3Data[time,:,:], vmax=500)
#ax4.pcolormesh(mod4Data[time,:,:], vmax=500)

# %% Generate instrumentals

mAr = 37211326.1 # Mass of argon in keV
c = 2.998e+5 # speed of light in km/s

lamz = 3.9941451e-3
lamw = 3.9490665e-3

lam0 = lamw
w = lam0**2 / mAr * 0.5 # width of X keV line
instT = np.sqrt(w)
print instT

# %% check instrumentals

"""
inst = specTree.getNode(r'\SPECTROSCOPY::TOP.HIREXSR.ANALYSIS7.HELIKE.MOMENTS.Z:INST')

iwidth = np.ones(96)*instT
ishift = np.zeros(96)
tpos = 0

newInst = MDSplus.Data.compile('build_signal(build_with_units($1,"Ang"),*,build_with_units($2,"Ang"),build_with_units($3,"sec"))'
,iwidth,ishift,tpos)
"""

#moms = specTree.getNode(r'\SPECTROSCOPY::TOP.HIREXSR.ANALYSIS7.HELIKE.MOMENTS.Z:MOM')

# %% Apologies to all invovled, try and get line-integrated positions

mom = specTree.getNode(r'\SPECTROSCOPY::TOP.HIREXSR.ANALYSIS.HELIKE.MOMENTS.Z:MOM')

