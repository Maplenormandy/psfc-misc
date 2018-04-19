# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 17:14:02 2017

@author: normandy
"""

import numpy as np

import readline
import MDSplus

import matplotlib.pyplot as plt

import scipy.optimize as op
import scipy.stats

# %% Load spectroscopic data

mod='MOD1'

specTree = MDSplus.Tree('spectroscopy', 1100811015)
modNode = specTree.getNode(r'\SPECTROSCOPY::TOP.HIREX_SR.RAW_DATA:' + mod)
modRawData = modNode.data()
modTime = modNode.dim_of(2).data()

lamNode = specTree.getNode(r'\SPECTROSCOPY::TOP.HIREXSR.CALIB.'+mod+':LAMBDA')
lamData = lamNode.data()

# %%

# Integrate from lower to upper time limits
t_low, t_high = 0.85, 1.45
ind_low, ind_high = np.searchsorted(modTime, (t_low, t_high))

modData = np.sum(modRawData[ind_low:ind_high,:,:], axis=0)

# Look for dead pixels and other outliers

# If a pixel is reporting a value before the shot begins, mark it as dead
ind_start = np.searchsorted(modTime, 0.0)
modPreData = np.sum(modRawData[:ind_start,:,:], axis=0)
goodPixels = (modPreData<2)

# If a pixel is out more than 3 empty histogram bins, remove it
histData, binEdges = np.histogram(modData[goodPixels], bins=32)
numZeroBins = np.cumsum(histData==0)
thirdZeroBin = np.searchsorted(numZeroBins, 3)

goodPixels = np.logical_and(goodPixels, modData<binEdges[thirdZeroBin])


# %% Plot spectra

plt.figure()
plt.pcolormesh(range(487), lamData, np.log10(modData*goodPixels), cmap='cubehelix', vmin = 2)
plt.colorbar()

# %%

plt.figure()
plt.plot(lamData[:,300], (modData*goodPixels)[:,300], marker='.')

# Peak is around pixel 168 for row 200

# 162 to 191 for now

    

# %% Find the peaks

def lnlike_peaks(theta, xi, counts):
    """
    Calculates the log likelihood of seeing the observed spectra given xi and counts
    """
    # peak, first, second moments of the line, then noise level
    peak, m1, m2, noise = theta
    
    # Calculate the expected counts for the given
    modelCounts = peak * np.exp(-(xi-m1)**2/m2) + noise
    return -0.5 * np.sum((counts-modelCounts)**2/modelCounts)

def findOptimalPeak(modData, goodPixels, row, guess, width=-1, noise=-1):
    spectra = modData[:,row]
    xi = np.array(range(len(spectra)))
    # Take only the good pixels
    gSpectra = spectra[goodPixels[:,row]]
    gXi = xi[goodPixels[:,row]]
    
    # We want to optimize the best range to use for the model, which we do by maximizing the log likelihood over all models
    # Restrict the spectra to a certain range
    xi_low, xi_high = guess-5, guess+9
    # Whether or not going lower/higher improved the result
    rGoingLower = True
    rGoingHigher = True
    # Whether or not we went higher last time
    rWentHigher = False
    
    if width < 0:
        width = 11
    if noise < 0:
        noise = np.percentile(gSpectra, 5)
        
    lastGuess = [spectra[int(guess)], guess, 11, np.percentile(gSpectra, 5)]
    lastResults = None
    
    loops = 0
    
    while True:
        loops += 1
        # Find the indices for restricting the spectra
        r_low, r_high = np.searchsorted(gXi, [xi_low, xi_high])
        rSpectra = gSpectra[r_low:r_high]
        rXi = gXi[r_low:r_high]
    
        nll = lambda *args: -lnlike_peaks(*args)
        result = op.minimize(nll, lastGuess, args=(rXi, rSpectra))
        
        #print xi_low, xi_high, result.fun, result.x
        
        if scipy.stats.chi2.cdf(result.fun, len(rXi)-1) < 0.95:
            if rWentHigher and rGoingLower:
                rWentHigher = False
                xi_low -= 1
            elif rGoingHigher:
                rWentHigher = True
                xi_high += 1
            else:
                print "something went wrong"
                return lastResults
        else:
            if rWentHigher and rGoingHigher:
                rGoingHigher = False
                xi_high -= 2
            elif (not rWentHigher) and rGoingLower:
                rGoingLower = False
                xi_low += 2
            
            if not (rGoingHigher or rGoingLower):
                return lastResults
                
        lastResults = (result, rXi, rSpectra)
        
        #print xi_low, xi_high, result.fun, result.x
            
        if loops > 5:
            return lastResults

# %% Get all optimal peaks

row = 200
guess = 168
width = -1
noise = -1

plt.figure()
plt.pcolormesh(modData*goodPixels, cmap='cubehelix')

while row > 0:
    result, rXi, rSpectra = findOptimalPeak(modData, goodPixels, row, guess, width, noise)
    if np.isfinite(result.fun):
        plt.scatter(row, result.x[1])
        guess = result.x[1]
        width = result.x[2]
        noise = result.x[3]
    
    if row%10 == 0:
        print row
    row -= 1
    
row = 201
guess = 168
width = -1
noise = -1

while row < modData.shape[1]:
    result, rXi, rSpectra = findOptimalPeak(modData, goodPixels, row, guess, width, noise)
    if np.isfinite(result.fun):
        plt.scatter(row, result.x[1])
        guess = result.x[1]
        width = result.x[2]
        noise = result.x[3]
    
    if row%10 == 0:
        print row
    row += 1