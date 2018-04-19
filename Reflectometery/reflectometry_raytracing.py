# -*- coding: utf-8 -*-
"""
Created on Thu May 11 17:55:13 2017

@author: normandy
"""

import numpy as np

import pickle
import dill

import eqtools
import gptools
import profiletools

import sys
from os import path
sys.path.append('/home/normandy/git/psfc-misc/Common')
from Collisionality import NustarProfile

# %% Fit the density profile

shot = 1160506007

e = eqtools.CModEFITTree(shot)
nustar = NustarProfile(shot, 0.5, 1.5)
nustar.fitNe([0.6])

# %% 