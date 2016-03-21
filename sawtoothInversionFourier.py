# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 20:06:23 2016

@author: normandy
"""

import numpy as np
import matplotlib.pyplot as plt

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
                    


"""
specTree = MDSplus.Tree('spectroscopy', 1120221032)
heNode = specTree.getNode('\SPECTROSCOPY::TOP.HIREXSR.ANALYSIS1.HELIKE.PROFILES.Z')
hyNode = specTree.getNode('\SPECTROSCOPY::TOP.HIREXSR.ANALYSIS1.HLIKE.PROFILES.LYA1')

td = ThacoData(heNode, hyNode)


elecTree = MDSplus.Tree('electrons', 1120221032)
gpc = elecTree.getNode('\gpc_t0')

#0.58 - 1.49
saw = sat.findSawteeth(gpc.dim_of().data(), gpc.data(), 0.56, 1.51)
sawtimes = gpc.dim_of().data()[saw]
time = gpc.dim_of().data()
te = gpc.data()
"""

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
    t0 = tbin - (0.006 - 0.003) / 2 + 0.006
    t1 = tbin + (0.006 - 0.003) / 2 + 0.006
   
    if t0 < 0.57 or t1 > 1.49:
        return None

    i0 = np.searchsorted(sawtimes, t0)
    i1 = np.searchsorted(sawtimes, t1)

    ph0 = (t0 - sawtimes[i0-1]) / (sawtimes[i0] - sawtimes[i0-1])
    ph1 = (t1 - sawtimes[i1-1]) / (sawtimes[i1] - sawtimes[i1-1])

    
    phVector = np.zeros(2*numBasis+1)
    
    # Fourier basis functions
    # Note this assumes that the length of the window < half sawtooth period
    if ph1 > ph0:
        phVector[0] = 2.0*(ph1-ph0)
        
        
    else:
        phVector[0] = 2.0+2.0*(ph1-ph0)

    for i in range(numBasis):
        n=i+1
        phVector[2*i+1] = (-np.sin(2*ph0*n*np.pi) + np.sin(2*ph1*n*np.pi)) / n / np.pi * np.sinc(1.0*n/(numBasis+2))**0
        phVector[2*i+2] = (np.cos(2*ph0*n*np.pi) - np.cos(2*ph1*n*np.pi)) / n / np.pi * np.sinc(1.0*n/(numBasis+2))**0
            
    return phVector

def transformEq(numPoints, coefs, r):
    r=0.6
    plotBasis = np.linspace(0,1,numPoints)
    func = np.ones(numPoints) * coefs[0] * 0.5
    m = len(coefs)/2
    for i in range(m):
        n=i+1
        func += np.cos(2*n*np.pi*plotBasis)*coefs[2*i+1] * np.sinc(1.0*n/(m+1))**1 * r**n
        func += np.sin(2*n*np.pi*plotBasis)*coefs[2*i+2] * np.sinc(1.0*n/(m+1))**1 * r**n
        
    return func

numBasis = 10

elecMeans = np.zeros(len(td.time))
elecData = gpc.data()
elecTime = gpc.dim_of().data()

plotBasis = np.linspace(0,1,50)


# Verification against electron data
for i in range(len(td.time)):
    t0 = td.time[i] - (0.006 - 0.003) / 2 + 0.006
    t1 = td.time[i] + (0.006 - 0.003) / 2 + 0.006
    
    i0 = np.searchsorted(elecTime, t0)-1
    i1 = np.searchsorted(elecTime, t1)
    
    elecMeans[i] = np.mean(elecData[i0:i1])
   
yraw = [getPhaseVector(sawtimes, t, numBasis) for t in td.time]

traw = []

for k in range(len(td.rho)):
    Araw = []
    braw = []
    xraw = []
    wraw = []
    for i in range(len(yraw)):
        if yraw[i] != None:
            Araw.append(yraw[i])
            braw.append(td.pro[3,i,k])
            wraw.append(1.0 / (td.perr[3,i,k]**2))
    
    #wraw = np.diag(wraw)
    #AArr = np.dot(wraw, np.array(Araw))
    #bArr = np.dot(wraw, np.array(braw))
    AArr = np.array(Araw)
    bArr = np.array(braw)
    
    
    #wArr = np.diag(td.perr[3,:,3])
    
    #AArr = np.dot(wArr, AArr)
    #bArr = np.dot(wArr, bArr)
    
    xArr = lstsq(AArr, bArr)
    vals = transformEq(50, xArr[0], 0.4)
    traw.append(vals)


plt.figure()

plt.pcolor(np.array(traw), cmap='cubehelix')
plt.colorbar()


braw = []
for i in range(len(yraw)):
    if yraw[i] != None:
        braw.append(elecMeans[i])
        
bArr = np.array(braw)

xArr = lstsq(AArr, bArr)         
plt.figure()
plt.plot(np.array(traw[0]))
plt.plot(transformEq(50, xArr[0], 0.4))
