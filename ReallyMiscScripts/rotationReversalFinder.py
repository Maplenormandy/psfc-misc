# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 18:06:23 2016

@author: normandy
"""

import numpy as np

import readline
import MDSplus

import matplotlib.pyplot as plt

import emcee

import scipy.optimize as op

readline

# %% Load rotation data

shot = 1160506007
specTree = MDSplus.Tree('spectroscopy', shot)
rotNode = specTree.getNode(r'\SPECTROSCOPY::TOP.HIREX_SR.ANALYSIS.W:VEL')
tiNode = specTree.getNode(r'\SPECTROSCOPY::TOP.HIREX_SR.ANALYSIS.W:TI')
countNode = specTree.getNode(r'\SPECTROSCOPY::TOP.HIREX_SR.ANALYSIS.W:INT')

ti = tiNode.data()[0]
count = countNode.data()[0]

y = rotNode.data()[0]
t = rotNode.dim_of().data()

times = np.logical_and(t >= 0.4, t < 1.6)

y = y[times]
t = t[times]

ti = ti[times]
count = count[times]

# %% Calculate uncertainty on velocity measurement

mAr = 3.7211e7 # Atomic mass of argon in keV
lightspeed = 3e5 # speed of light in km/s
viewAngle = 8.0 / 180.0 * np.pi

# Not sure what happened to the viewing angle factor. Oh well.
yerr = np.sqrt(ti / mAr / (count - 1)) * lightspeed


# %% MLE estimate

testTimes = np.logical_and(t >= 0.6, t < 0.82)
sx = t[testTimes]
sy = y[testTimes]
syerr = yerr[testTimes]

def lnlike(theta, x, y, yerr):
    v0, b0, tInv, bInv, lnf = theta # initial rotation v, initial velocity drift, time of inversion, slope, and variance of fluctuations
    model = v0 + (x - tInv) * np.where(x > tInv, bInv+b0, b0)
    inv_sigma2 = 1.0/(yerr**2 + np.exp(2*lnf))
    return -0.5*(np.sum((y-model)**2*inv_sigma2 - np.log(inv_sigma2)))
    
nll = lambda *args: -lnlike(*args)
result = op.minimize(nll, [-20.0, 0.0, 0.77, 100.0, 0.0], args=(sx, sy, syerr))
#result = op.minimize(nll, [-15.0, 0.0, 0.75, 132.0], args=(sx, sy, syerr))

theta0 = result["x"]
print result["x"]

plt.figure()
plt.errorbar(sx, sy, yerr=syerr)
plt.plot(sx, theta0[0] + (sx - theta0[2]) * np.where(sx > theta0[2], theta0[3]+theta0[1], theta0[1]))

# %% Emcee time

    
def lnprior(theta):
    v0, b0, tInv, bInv, lnf = theta
    if -30.0 < v0 < 30.0 and -3.0 < lnf < 3.0 and -50.0 < b0 < 50.0:
        if 0.72 < tInv < 0.82 and 50 < bInv < 250:
            return 0.0
        else:
            return -np.inf
    else:
        return -np.inf
        
def lnprob(theta, x, y, yerr):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, x, y, yerr)
    
ndim, nwalkers = 5, 100
pos = [result["x"] + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]

sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(sx, sy, syerr))
sampler.run_mcmc(pos, 500)

samples = sampler.chain[:, 50:, :].reshape((-1, ndim))

# %% Plot emcee results

f, axarr = plt.subplots(5)
axarr[0].hist(samples[:,0], bins=50)
axarr[1].hist(samples[:,1], bins=50)
axarr[2].hist(samples[:,2], bins=50)
axarr[3].hist(samples[:,3], bins=50)
axarr[4].hist(samples[:,4], bins=50)