# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 13:51:24 2016

@author: normandy
"""

import numpy as np
from scipy.signal import medfilt, find_peaks_cwt
import matplotlib.pyplot as plt


    
import readline
import MDSplus

def rephaseToMax(peaks, data, indMin, indMax):
    bounds = [0] * (len(peaks) + 1)
    
    for i in range(len(peaks)):
        bounds[i] += peaks[i]
        bounds[i+1] += peaks[i]
    
    for i in range(len(peaks)):
        bounds[i] = bounds[i] / 2
            
    bounds[0] = indMin
    bounds[-1] = indMax
    
    peaksPhased = [0] * len(peaks)
    
    for i in range(len(peaks)):
        peaksPhased[i] = data[bounds[i]:bounds[i+1]].argmax() + bounds[i]
        # Reject values that end up on one of the boundaries
        if peaksPhased[i] - bounds[i] < 4 or  bounds[i+1]-peaksPhased[i] < 4:
            peaksPhased[i] = -1
    
    #peaksPhased =     
        
    return filter(lambda x: x > 0, peaksPhased)
    
def rephaseToMin(peaks, data, indMin, indMax):
    bounds = [0] * (len(peaks) + 1)
    
    for i in range(len(peaks)):
        bounds[i] += peaks[i]
        bounds[i+1] += peaks[i]
    
    for i in range(len(peaks)):
        bounds[i] = bounds[i] / 2
            
    bounds[0] = indMin
    bounds[-1] = indMax
    
    peaksPhased = [0] * len(peaks)
    
    for i in range(len(peaks)):
        peaksPhased[i] = data[bounds[i]:bounds[i+1]].argmin() + bounds[i]
        # Reject values that end up on one of the boundaries
        if peaksPhased[i] - bounds[i] < 4 or  bounds[i+1]-peaksPhased[i] < 4:
            peaksPhased[i] = -1
    
    #peaksPhased =     
        
    return filter(lambda x: x > 0, peaksPhased)

def rephaseToNearbyMax(peaks, data, radius):
    peaksPhased = [0] * len(peaks)
    
    for i in range(len(peaks)):
        lbound = peaks[i]-radius
        ubound = peaks[i]+radius
        peaksPhased[i] = data[lbound:ubound+1].argmax() + lbound
        
    return peaksPhased

def rephaseToNearbyMin(peaks, data, radius):
    peaksPhased = [0] * len(peaks)
    
    for i in range(len(peaks)):
        lbound = peaks[i]-radius
        ubound = peaks[i]+radius
        peaksPhased[i] = data[lbound:ubound+1].argmin() + lbound
        
    return peaksPhased

# Note that while the array is called "peaks", they're actually troughs
def findPeakCrest(peaks, data):
    crests = [0] * len(peaks)
    
    for i in range(len(peaks)):
        maxVal = data[peaks[i]]
        #print maxVal
        maxInd = peaks[i]
        for k in range(1,10):
            #print data[peaks[i]-k]
            if data[peaks[i]-k] > maxVal:
                maxInd = peaks[i]-k
                maxVal = data[peaks[i]-k]
            else:
                break
            
        crests[i] = maxInd
        
    return crests
            

def findColdPulses(shot):
    transTree = MDSplus.Tree('transport', shot)
    try:
        injNode = transTree.getNode('\\top.imp_inj.dt196.input_10')
        
        inj = injNode.data()
        time = injNode.dim_of().data()[:-1]
        
        peaks = medfilt(inj, 5) - np.median(inj) > 0.1
        
        return np.floor(time[np.diff(peaks*1) > 0] * 100) / 100
    except:
        return np.array([])
        
    
def findSawteethCwt(time, te, tmin, tmax):
    """
    Generally use GPC_T0
    """
    
    #elecTree = MDSplus.Tree('electrons', shot)
    #teNode = elecTree.getNode('\ELECTRONS::TE_HRECE15')
    
    teFilt = medfilt(te, 7)
    
    indMin = time.searchsorted(tmin)
    indMax = time.searchsorted(tmax)+1

    # Use black magic to find sawteeth peaks. Usually they're around 0.0015s
    # wide, but they can get shorter. Each frame is 5e-5s long. May want to
    # consider using an asymmetric wavelet in the future
    teDiff = np.abs(np.diff(np.diff(teFilt[indMin-1:indMax+1])))
    #plt.plot(teDiff)
    peaks = find_peaks_cwt(teDiff, np.arange(7,23))
    peaks = [p + indMin for p in peaks]
    
    return rephaseToMin(peaks, te, indMin, indMax)
    #return peaks
    
def sawtoothMedian(peaks, time, te):
    newTemps = np.array([0.0] * (len(peaks)-1))
    newTimes = np.array([0.0] * (len(peaks)-1))
    
    for i in range(len(peaks)-1):
        newTemps[i] = np.median(te[peaks[i]:peaks[i+1]])
        newTimes[i] = np.median(time[peaks[i]:peaks[i+1]])
        
    return newTimes, newTemps

def findSawteeth(time, te, tmin, tmax, sep=8, threshold=0.0):
    indMin = time.searchsorted(tmin)-sep
    indMax = time.searchsorted(tmax)+sep+1
    
    teDiff = (te[indMin+sep:indMax] - te[indMin:indMax-sep]) < threshold
    teFront = np.diff(teDiff*1.0)
    peaks = np.where(teFront > 0)[0]
    peaks = [p + indMin + sep for p in peaks]
    
    # Note that it looks like peaks but it's actually troughs.
    return rephaseToNearbyMin(peaks, te, 4)
    #return peaks
