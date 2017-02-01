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

specTree = MDSplus.Tree('spectroscopy', 1101123023)
mod2Node = specTree.getNode(r'\SPECTROSCOPY::TOP.HIREX_SR.RAW_DATA:MOD2')
mod2Data = mod2Node.data()


plt.pcolormesh(mod2Data[50,:,:], vmax=500)
plt.colorbar()