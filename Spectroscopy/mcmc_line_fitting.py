# -*- coding: utf-8 -*-
"""
Try and improve line fitting in THACO

Created on Fri Jan 26 16:43:26 2018

@author: normandy
"""

import numpy as np
import matplotlib.pyplot as plt

import readline
import MDSplus

import eqtools

import mcRaytracing as mcr

from scipy import stats
import scipy.optimize as op

import emcee
import corner

import scipy

# %% Load data

shot = 1120914036

specTree = MDSplus.Tree('spectroscopy', shot)

# Indices are [lambda, time, channel]
specBr_all = specTree.getNode(r'\SPECTROSCOPY::TOP.HIREXSR.ANALYSIS1.HELIKE.SPEC:SPECBR').data()
sig_all = specTree.getNode(r'\SPECTROSCOPY::TOP.HIREXSR.ANALYSIS1.HELIKE.SPEC:SIG').data()
lam_all = specTree.getNode(r'\SPECTROSCOPY::TOP.HIREXSR.ANALYSIS1.HELIKE.SPEC:LAM').data()

# %% Load spectral lines

with open('/home/normandy/idl/HIREXSR/hirexsr_wavelengths.csv', 'r') as f:
    lineData = [s.strip().split(',') for s in f.readlines()]
    lineLam = np.array([float(ld[1]) for ld in lineData[2:]])
    lineZ = np.array([int(ld[2]) for ld in lineData[2:]])
    lineName = np.array([ld[3] for ld in lineData[2:]])
    
with open('/home/normandy/git/psfc-misc/ReallyMiscScripts/atomic_data.csv', 'r') as f:
    atomData = [s.strip().split(',') for s in f.readlines()]
    atomSymbol = np.array([ad[1] for ad in atomData[1:84]])
    atomAmu = np.array([float(ad[3]) for ad in atomData[1:84]])

# %%

tbin = 5
chbin = 17

wlim = (3.725, 3.743)
w0, w1 = np.searchsorted(lam_all[:,tbin,chbin], wlim)

lam = lam_all[w0:w1,tbin,chbin]
specBr = specBr_all[w0:w1,tbin,chbin]
sig = sig_all[w0:w1,tbin,chbin]

plt.errorbar(lam, specBr, yerr=sig, fmt='.')

lineInd = np.logical_and(lineLam>wlim[0], lineLam<wlim[1])

print 'fitting:', lineName[lineInd]
fitName = lineName[lineInd]
fitLam = lineLam[lineInd]
fitZ = lineZ[lineInd]
fitAmu = atomAmu[fitZ-1]

for l in range(len(fitLam)):
    plt.axvline(fitLam[l], ls='--')
    plt.text(fitLam[l], np.max(specBr), fitName[l])

def estimateNoise(specBr):
    return np.array(np.percentile(specBr, 1))

def estimateLines(noise, lam, specBr):
    
    gCenter = np.zeros(len(fitLam))
    gScale = np.zeros(len(fitLam))
    gHerm0 = np.zeros(len(fitLam))
    
    for l in range(len(fitLam)):
        l0 = np.searchsorted(lam, fitLam[l])
        lamFit = lam[l0-4:l0+5]
        specFit = specBr[l0-4:l0+5]-noise

        gCenter[l] = np.average(lamFit, weights=specFit)
        gScale[l] = np.sqrt(np.average((lamFit-gCenter[l])**2, weights=specFit))*1e4
        gHerm0[l] = np.max(specBr[l0]-noise, 0)
    
    return gHerm0, gCenter, gScale
    
noise0 = estimateNoise(specBr)
gHerm00, gCenter0, gScale0 = estimateLines(noise0, lam, specBr)

plt.plot(lam, noise0+np.sum(gHerm00*np.exp(-(lam[:, np.newaxis]-gCenter0)**2/2/(gScale0*1e-4)**2), axis=1))

# %% Now, begin the nonlinear optimization; "first step" nonlinear fit

lam0 = fitLam[0]
amu0 = fitAmu[0]
gHerm10 = np.zeros(gHerm00.shape)
gHerm20 = np.zeros(gHerm00.shape)
theta0 = np.hstack([noise0, (gCenter0[0]-lam0)*1e4, gScale0[0], gHerm00, gHerm10, gHerm20])

fitAmuRatio = np.sqrt(fitAmu/amu0)
fitLamOffset = fitLam - lam0
nfit = len(fitLam)

lamEdge = np.zeros(len(lam)+1)
lamEdge[1:-1] = (lam[1:] + lam[:-1]) / 2
lamEdge[-1] = 2 * lamEdge[-2] - lamEdge[-3]
lamEdge[0] = 2 * lamEdge[1] - lamEdge[2]

def modelLine(theta, lam, specBr, sig):
    # elements of theta:
    # noise, mean, variance, brightnesses
    gCenter = theta[1]*1e-4 + fitLam
    gScale = (theta[2]/fitAmuRatio)*1e-4
    gHerm0 = theta[3:3+nfit]
    gHerm1 = theta[3+nfit:3+nfit*2]
    gHerm2 = theta[3+nfit*2:3+nfit*3]
    
    
    
    lamEv = (lam[:, np.newaxis]-gCenter)/gScale
    lamEvEdge = (lamEdge[:, np.newaxis]-gCenter)/gScale
    
    gauss = np.exp(-lamEv**2 / 2)
    gaussEdge = np.exp(-lamEvEdge**2 / 2)
    
    lineH0Edge = gHerm0 * gaussEdge
    lineH0 = (4 * gHerm0 * gauss + lineH0Edge[1:] + lineH0Edge[-1:]) / 6
    
    lineH1Edge = gHerm1 * lamEvEdge * gaussEdge
    lineH1 = (4 * gHerm1 * lamEv * gauss + lineH1Edge[1:] + lineH1Edge[-1:]) / 6
    
    lineH2Edge = gHerm2 * (lamEvEdge**2 - 1) * gaussEdge
    lineH2 = (4 * gHerm2 * (lamEv**2 - 1) * gauss + lineH2Edge[1:] + lineH2Edge[-1:]) / 6
    
    pred = theta[0] + np.sum(lineH0 + lineH1 + lineH2, axis=1)
    
    return pred


def lnlike1(theta, lam, specBr, sig):
    pred = modelLine(theta, lam, specBr, sig)
    return -np.sum((specBr-pred)**2/sig**2)
    
    
def h0cnst(theta, n):
    return theta[3+n]
    
def h1cnst(theta, n):
    return theta[3+n]-np.abs(10*theta[3+n+nfit])

def h2cnst(theta, n):
    return theta[3+n]-np.abs(10*theta[3+n+2*nfit])


nll1 = lambda *args: -lnlike1(*args)

constraints = []
for i in range(nfit):
    constraints.append({
        'type': 'ineq',
        'fun': h0cnst,
        'args': [i]
    })
    constraints.append({
        'type': 'ineq',
        'fun': h1cnst,
        'args': [i]
    })
    constraints.append({
        'type': 'ineq',
        'fun': h2cnst,
        'args': [i]
    })

result = op.minimize(nll1, theta0, args=(lam, specBr, sig), tol=1e-8, constraints = constraints)

# %%

    
def modelLineDecompose(theta, lam, specBr, sig, order, lines=None):
    gCenter = theta[1]*1e-4 + fitLam
    gScale = (theta[2]/fitAmuRatio)*1e-4
    gHerm0 = theta[3:3+nfit]
    gHerm1 = theta[3+nfit:3+nfit*2]
    gHerm2 = theta[3+nfit*2:3+nfit*3]
    
    if lines==None:
        lines = range(nfit)
    
    lamEv = (lam[:, np.newaxis]-gCenter)/gScale
    lamEvEdge = (lamEdge[:, np.newaxis]-gCenter)/gScale
    
    gauss = np.exp(-lamEv**2 / 2)
    gaussEdge = np.exp(-lamEvEdge**2 / 2)
    
    lineH0Edge = gHerm0 * gaussEdge
    lineH0 = (4 * gHerm0 * gauss + lineH0Edge[1:] + lineH0Edge[-1:]) / 6
    
    if order >= 1:
        lineH1Edge = gHerm1 * lamEvEdge * gaussEdge
        lineH1 = (4 * gHerm1 * lamEv * gauss + lineH1Edge[1:] + lineH1Edge[-1:]) / 6
    else:
        lineH1 = np.zeros(lineH0.shape)
    
    if order >= 2:
        lineH2Edge = gHerm2 * (lamEvEdge**2 - 1) * gaussEdge
        lineH2 = (4 * gHerm2 * (lamEv**2 - 1) * gauss + lineH2Edge[1:] + lineH2Edge[-1:]) / 6
    else:
        lineH2 = np.zeros(lineH0.shape)
    
    pred = theta[0] + np.sum(lineH0[:,lines] + lineH1[:,lines] + lineH2[:,lines], axis=1)
    
    return pred
    
    
theta1 = result['x']
print theta1
pred1 = modelLine(theta1, lam, specBr, sig)
pred1H0 = modelLineDecompose(theta1, lam, specBr, sig, 0)
plt.plot(lam, pred1)
plt.plot(lam, pred1H0)

plt.errorbar(lam, specBr, yerr=sig, fmt='.')
plt.scatter(lam, pred1-specBr)

for i in range(nfit):
    pred1line = modelLineDecompose(theta1, lam, specBr, sig, 2, [i])
    plt.plot(lam, pred1line, c='c')
    
# %% emcee
    
amuToKeV = 931494.095 # amu in keV
speedOfLight = 2.998e+5 # speed of light in km/s
w0 = np.sqrt(lam0**2 / (amu0 * amuToKeV)) * 15 # 15 keV line width
    
def lnprior(theta):
    #specBrMax = np.percentile(specBr, 99) * 2
    
    #gCenter = theta[1]*1e-4 + fitLam
    #gScale = (theta[2]/fitAmuRatio)*1e-4
    gHerm0 = theta[3:3+nfit]
    gHerm1 = theta[3+nfit:3+nfit*2]
    gHerm2 = theta[3+nfit*2:3+nfit*3]
    
    if (theta[2] > 0) and (theta[0] > 0) and np.all(gHerm0>0) and np.all(gHerm1>0) and np.all(gHerm2>0):
        return 0.0
    else:
        return -np.inf
        
def lnprob(theta, lam, specBr, sig):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    else:
        return lp+lnlike1(theta, lam, specBr, sig)

ndim, nwalkers = len(theta1), 100
pos = [theta1 + 1e-4 * np.random.randn(ndim) for i in range(nwalkers)]

sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(lam, specBr, sig))
sampler.run_mcmc(pos, 500)

samples = sampler.chain[:, 50:, :].reshape((-1, ndim))

# %% Convert to moments

def momentConvert(thetas, line=0):
    if len(thetas.shape) < 2:
        thetas = np.array([thetas])
    
    gScale = (thetas[:,2]/fitAmuRatio[line])*1e-4
    normFactor = np.sqrt(2*np.pi)*gScale
    m0 = normFactor*thetas[:,3+line]
    m1 = thetas[:,1]*1e-4 + normFactor*thetas[:,3+nfit+line]*gScale/m0
    m2 = gScale**2 + normFactor*thetas[:,3 + 2*nfit+line]*2*gScale**2/m0
    if len(m0) > 1:
        return m0, m1*1e3, m2*1e6
    else:
        return m0[0], m1[0]*1e3, m2[0]*1e6
        
def numMomentConvert(theta, line=0):
    pred = modelLineDecompose(theta, lam, specBr, sig, 2, [line]) - theta[0]
    m0 = scipy.integrate.simps(pred, lam)
    m1 = scipy.integrate.simps(pred*lam, lam)/m0 - fitLam[line]
    m2 = scipy.integrate.simps(pred*(lam-fitLam[line]-m1)**2, lam)/m0
    
    return m0, m1*1e3, m2*1e6
    
m0, m1, m2 = momentConvert(samples)

