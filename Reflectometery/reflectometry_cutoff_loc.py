# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 15:02:54 2018

@author: normandy
"""

from __future__ import division

import profiletools
import gptools
import eqtools

import numpy as np
import scipy

import readline
import MDSplus

import matplotlib.pyplot as plt

import sys
sys.path.append('/home/normandy/git/psfc-misc')
sys.path.append('/home/normandy/git/psfc-misc/Common')

import shotAnalysisTools as sat
from Collisionality import NustarProfile

import copy

import scipy.optimize as op

readline

# %%

nustar = NustarProfile(1160506007, 0.4, 1.6)
nustar.fitNe(np.linspace(0.5, 1.5, 20))

# %%

nustar.evalProfile(np.linspace(0.5, 1.5, 20))


# %%

cutoffs = np.zeros(16)

for i in range(16):
    cutoffs[i] = op.bisect(lambda r: nustar.neFit[i](r)[0]-0.971, 0.0, 1.0)
    
plt.figure()
t0 = np.linspace(0.5, 1.5, 20)
t0 = t0[:16]
plt.plot(t0, cutoffs)