# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 17:15:04 2016

@author: normandy
"""

from __future__ import division

import profiletools
import gptools
import eqtools

import numpy as np
import scipy

import matplotlib.pyplot as plt

# %% ne Profile Fitting

shot = 1160506007

p_ne = profiletools.ne(shot, abscissa='r/a', t_min=0.72, t_max=0.82, include=['CTS', 'ETS'])

p_ne.create_gp(k='SE', constrain_at_limiter=False)

p_ne.gp.k.hyperprior = (
                gptools.UniformJointPrior([(0, 4)]) *
                gptools.UniformJointPrior([(0, 0.2)]) *
                gptools.GammaJointPrior([1 + 1 * 5], [5])
            )

sol = p_ne.find_gp_MAP_estimate()
#p_ne.plot_gp()

print sol


# %% ne Profile Plotting

e = eqtools.CModEFITTree(shot)

t = 0.77

a = np.vstack((np.ones(50)*t, np.linspace(0,1))).T

mean, std = p_ne.smooth(a)

plt.errorbar(np.linspace(0,1), mean, yerr=std)

# %% Te Profile Fitting

p_Te = profiletools.Te(shot, abscissa='r/a', t_min=0.72, t_max=0.82, include=['GPC', 'GPC2'])

p_Te.create_gp(k='SE', constrain_at_limiter=False)

p_Te.gp.k.hyperprior = (
                gptools.UniformJointPrior([(0, 8)]) *
                gptools.UniformJointPrior([(0, 0.2)]) *
                gptools.GammaJointPrior([1 + 1 * 5], [5])
            )

solT = p_Te.find_gp_MAP_estimate()
#p_ne.plot_gp()

print solT