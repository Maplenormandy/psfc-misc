# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 16:36:26 2016

@author: normandy
"""

from scipy.stats import levy_stable, linregress
import numpy as np

import matplotlib.pyplot as plt

#x = levy_stable.rvs(alpha=0.4, beta=1.0, size=100000)

def rollingMean(x, size):
    return np.convolve(x, np.ones((size,))/size, mode='valid')

#spaces = np.logspace(1,4,25 , dtype=np.int)
#means = np.array([np.median(rollingMean(x,s)) for s in spaces])

#plt.loglog(spaces, means)

def getSlope(alpha):
    x = levy_stable.rvs(alpha=alpha, beta=0.0, size=100000)
    cutoffs = np.logspace(-1,4)
    means = np.array([np.std(np.clip(x,-c,c)) for c in cutoffs])
    slope, intercept, r_value, p_value, std_err = linregress(np.log(cutoffs), np.log(means))
    plt.loglog(cutoffs, means)
    return slope


alphs = np.linspace(0.01, 2.0, 21)
slopes = np.array([getSlope(a) for a in alphs])

#plt.plot(alphs, slopes)


#getSlope(1.6)
#plt.semilogy(levy_stable.rvs(alpha=1.1, beta=1.0, size=500000))

#plt.hist(levy_stable.rvs(alpha=1.1, beta=1.0, size=100000), bins=np.logspace(-4,10))
#plt.gca().set_xscale('log')