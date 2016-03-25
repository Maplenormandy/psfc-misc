# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 20:06:23 2016

@author: normandy
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from scipy.linalg import lstsq
    
import readline
import MDSplus

import shotAnalysisTools as sat

class ThacoData:
    def __init__(self, heNode, hyNode):
        
        heproNode = heNode.getNode('PRO')
        herhoNode = heNode.getNode('RHO')
        heperrNode = heNode.getNode('PROERR')

        herpro = heproNode.data()
        herrho = herhoNode.data()
        herperr = heperrNode.data()
        hertime = herhoNode.dim_of()

        hegoodTimes = (hertime > 0).sum()

        self.hetime = hertime.data()[:hegoodTimes]
        self.herho = herrho[0,:] # Assume unchanging rho bins
        self.hepro = herpro[:,:hegoodTimes,:len(self.herho)]
        self.heperr = herperr[:,:hegoodTimes,:len(self.herho)]
        
        
        hyproNode = hyNode.getNode('PRO')
        hyrhoNode = hyNode.getNode('RHO')
        hyperrNode = hyNode.getNode('PROERR')

        hyrpro = hyproNode.data()
        hyrrho = hyrhoNode.data()
        hyrperr = hyperrNode.data()
        hyrtime = hyrhoNode.dim_of()

        hygoodTimes = (hyrtime > 0).sum()

        self.hytime = hyrtime.data()[:hygoodTimes]
        self.hyrho = hyrrho[0,:] # Assume unchanging rho bins
        self.hypro = hyrpro[:,:hygoodTimes,:len(self.hyrho)]
        self.hyperr = hyrperr[:,:hygoodTimes,:len(self.hyrho)]
        
        # Assume same times and rhos
        self.time = self.hetime
        self.rho = self.herho
        
        self.pro = np.copy(self.hepro)
        self.perr = np.copy(self.heperr)
        
        for j in range(self.hypro.shape[1]):
            takingHy = False
            for k in reversed(range(self.hypro.shape[2])):
                if self.perr[3,j,k] > self.hyperr[3,j,k]:
                    takingHy = True
                    
                if takingHy:
                    self.pro[:,j,k] = self.hypro[:,j,k]
                    self.perr[:,j,k] = self.hyperr[:,j,k]
                    


#specTree = MDSplus.Tree('spectroscopy', 1120221032)
specTree = MDSplus.Tree('spectroscopy', 1120221032)#1120106032
heNode = specTree.getNode('\SPECTROSCOPY::TOP.HIREXSR.ANALYSIS.HELIKE.PROFILES.Z')
hyNode = specTree.getNode('\SPECTROSCOPY::TOP.HIREXSR.ANALYSIS.HLIKE.PROFILES.LYA1')

td = ThacoData(heNode, hyNode)


#elecTree = MDSplus.Tree('electrons', 1120221032)
elecTree = MDSplus.Tree('electrons', 1120221032)#1120106021
gpc = elecTree.getNode('\gpc_t0')

#0.58 - 1.49
saw = sat.findSawteeth(gpc.dim_of().data(), gpc.data(), 0.56, 1.51)
#oldSaw = saw
#saw = np.linspace(0, 1, 45) * (saw[-1] - saw[0]) + saw[0]
#saw = [int(s) for s in saw]
sawtimes = gpc.dim_of().data()[saw]
time = gpc.dim_of().data()
te = gpc.data()

"""
indMin = time.searchsorted(0.56)
indMax = time.searchsorted(1.51)+1
    
#sawmin = sat.rephaseToMin(saw, te, indMin, indMax)
sawcrest = sat.findPeakCrest(saw, te)
plt.figure()
plt.plot(time, te)
plt.scatter(time[saw], te[saw])
plt.scatter(time[sawcrest], te[sawcrest])

sawCrash = time[sawcrest[1:]] - time[saw[:-1]]
crashAmp = sat.sawtoothMedian(saw, time, te)[1]
plt.figure()
plt.scatter(sawCrash, crashAmp / np.sqrt(sawCrash))
plt.xlabel('sawtooth growth time')
plt.ylabel('peak Te / growth time')
"""


def getPhaseVector(sawtimes, tbin, numBasis):
    t0 = tbin - (0.006 - 0.003) / 2 + 0.003
    t1 = tbin + (0.006 - 0.003) / 2 + 0.003
   
    if t0 < 0.57 or t1 > 1.49:
        return None

    i0 = np.searchsorted(sawtimes, t0)
    i1 = np.searchsorted(sawtimes, t1)

    ph0 = (t0 - sawtimes[i0-1]) / (sawtimes[i0] - sawtimes[i0-1])
    ph1 = (t1 - sawtimes[i1-1]) / (sawtimes[i1] - sawtimes[i1-1])

    phBasis = np.linspace(0, 1, numBasis, False)
    phBDiff = np.median(np.diff(phBasis))

    ib0 = np.searchsorted(phBasis, ph0)
    ib1 = np.searchsorted(phBasis, ph1)
    
    # Fourier basis functions
    
    """  
    # Quadratic Basis Functions
    # Note this assumes that the length of the window < half sawtooth period
    if ib1 > ib0:
        phVector = np.zeros(numBasis)
        for i in range(ib0+1, ib1-1):
            phVector[i] = 1.0
        
        d0 = (phBasis[ib0] - ph0) / phBDiff
        phVector[ib0-1] = 0.5 * (d0 * d0)
        phVector[ib0%numBasis] = 1.0 - 0.5 * ((1-d0) * (1-d0))
        
        d1 = (phBasis[ib1-1] + phBDiff - ph1) / phBDiff
        phVector[ib1-1] = 1.0 - 0.5 * (d1 * d1)
        phVector[ib1%numBasis] = 0.5 * ((1-d1) * (1-d1))
    else:
        phVector = np.ones(numBasis)
        for i in range(ib1+1, ib0-1):
            phVector[i] = 0.0
            
        d0 = (phBasis[ib0-1] + phBDiff - ph0) / phBDiff
        phVector[ib0-1] = 0.5 * (d0 * d0)
        phVector[ib0%numBasis] = 1.0 - 0.5 * ((1-d0) * (1-d0))
        
        if ib1 >= len(phBasis):
            d1 = (phBasis[ib1-1] + phBDiff - ph1) / phBDiff
        else:
            d1 = (phBasis[ib1] - ph1) / phBDiff
        phVector[ib1-1] = 1.0 - 0.5 * (d1 * d1)
        phVector[ib1%numBasis] = 0.5 * ((1-d1) * (1-d1))
    """
    
    # Linear Basis Functions
    if ib1 > ib0:
        phVector = np.zeros(numBasis)
        for i in range(ib0+1, ib1-1):
            phVector[i] = 1.0
        
        d0 = (phBasis[ib0] - ph0) / phBDiff
        if d0 >= 0.5:
            phVector[ib0] = 1.5 - d0
        else:
            phVector[ib0] = 1.0
            phVector[ib0-1] = 0.5 - d0
            
        d1 = (phBasis[ib1-1] + phBDiff - ph1) / phBDiff
        if d1 > 0.5:
            phVector[ib1-1] = 1.0
            phVector[ib1%numBasis] = d1 - 0.5
    else:
        phVector = np.zeros(numBasis)
        for i in range(0, ib1-1):
            phVector[i] = 1.0
        for i in range(ib0+1,numBasis):
            phVector[i] = 1.0
            
        d0 = (phBasis[ib0-1] + phBDiff - ph0) / phBDiff
        if d0 >= 0.5:
            phVector[ib0%numBasis] = 1.5 - d0
        else:
            phVector[ib0%numBasis] = 1.0
            phVector[ib0-1] = 0.5 - d0
            
        d1 = (phBasis[ib1] - ph1) / phBDiff
        if d1 > 0.5:
            phVector[ib1-1] = 1.0
            phVector[ib1%numBasis] = d1 - 0.5
        
    """
    # ZOH
    if ib1 > ib0:
        phVector = np.zeros(numBasis)
        for i in range(ib0-1, ib1):
            phVector[i] = 1.0
    else:
        phVector = np.ones(numBasis)
        for i in range(ib1+1, ib0-1):
            phVector[i] = 0.0
    """    
    phVector = phVector / np.sum(phVector)
    return phVector


numBasis = 12
phBasis = np.linspace(0, 1, numBasis, False)

elecMeans = np.zeros(len(td.time))
elecData = gpc.data()
elecTime = gpc.dim_of().data()

# Verification against electron data
for i in range(len(td.time)):
    t0 = td.time[i] - (0.006 - 0.003) / 2 + 0.006
    t1 = td.time[i] + (0.006 - 0.003) / 2 + 0.006
    
    i0 = np.searchsorted(elecTime, t0)-1
    i1 = np.searchsorted(elecTime, t1)
    
    elecMeans[i] = np.mean(elecData[i0:i1])
   
yraw = [getPhaseVector(sawtimes, t, numBasis) for t in td.time]

traw = []



for k in range(len(td.rho)-5):
    Araw = []
    braw = []
    xraw = []
    wraw = []
    for i in range(len(yraw)):
        if yraw[i] != None:
            Araw.append(yraw[i])
            braw.append(td.pro[1,i,k])
            wraw.append(1.0 / (td.perr[1,i,k]**2))
            #wraw.append(1.0)
    
    #wraw = np.diag(wraw)
    AArr = np.dot(wraw, np.array(Araw))
    bArr = np.dot(wraw, np.array(braw))
    AArr = np.array(Araw)
    bArr = np.array(braw)
    
    diffCond = np.zeros(numBasis)
    diffCond[0] = 2.0 / numBasis / numBasis
    diffCond[1] = -1.0 / numBasis / numBasis
    diffCond[-1] = -1.0 / numBasis / numBasis
    diffCond = diffCond*1.0
    for i in range(numBasis):
        AArr = np.vstack((AArr, np.roll(diffCond, i).T))
        
    bArr = np.hstack((bArr, np.zeros(numBasis)))
    
    
    #wArr = np.diag(td.perr[3,:,3])
    
    #AArr = np.dot(wArr, AArr)
    #bArr = np.dot(wArr, bArr)
    
    xArr = lstsq(AArr, bArr)
    traw.append(xArr[0])


def centerScale(vals):
    """
    Maps 1,2,3,4 -> 0.5,1.5,2.5,3.5,4.5
    Only works for flat arrays
    """

    newVals = np.append(vals, 2*vals[-1]-vals[-2])
    valDiffs = np.ediff1d(newVals, to_end=0)
    valDiffs[-1] = valDiffs[2]
    newVals -= valDiffs / 2

    return newVals
    
fracscale = centerScale(np.linspace(0, 1, numBasis))
rhoscale = centerScale(np.linspace(0, 1, len(td.rho))[:-5])

plt.figure()
xp, yp = np.meshgrid(fracscale, rhoscale)
plt.pcolor(xp, yp, np.array(traw), cmap='cubehelix')
plt.colorbar(label='toroidal rotation rate [kHz]')
plt.xlabel('sawtooth fraction')
plt.ylabel('r/a')
plt.title('Ti')


braw = []
for i in range(len(yraw)):
    if yraw[i] != None:
        braw.append(elecMeans[i])
        
bArr = np.array(braw)
bArr = np.hstack((bArr, np.zeros(numBasis)))

xArr = lstsq(AArr, bArr)         
plt.figure()
plt.plot(phBasis, np.array(traw[0]))
plt.plot(phBasis, xArr[0])
plt.xlabel('sawtooth fraction')
plt.ylabel('temperature [keV]')
plt.legend(['HIREX Ti upsample','GPC Te "upsample"'], loc='upper left')
