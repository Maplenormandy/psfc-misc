# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 16:57:13 2016

@author: normandy
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

psin = np.linspace(0, 1, 250)
#psin = psin[1:]

meanPsi = 0.1
sigmaPsi = 0.4

brightness = np.exp(-( (psin-meanPsi)**2 / (2*sigmaPsi**2) ))
#brightness = np.exp(-psin**2/sigmaPsi)

plt.close("all")

collBright = np.sqrt(psin) * brightness
intBright = integrate.cumtrapz(collBright, psin, initial=0)
intBright = intBright / np.max(intBright)
plt.figure()
plt.plot(psin, intBright)
plt.plot(psin, brightness)

npts = 16
samplePts = np.linspace(0, 1, npts+1)
samplePts = samplePts[:-1]

bestBin = np.interp(samplePts**2, intBright, psin**2)
print "psin", ",".join(map(str,np.round(bestBin, 4)))
print "r/a", ",".join(map(str,np.round(np.sqrt(bestBin), 4)))
print "psin2", ",".join(map(str,np.round(bestBin**2, 4)))

plt.figure()
plt.scatter(range(npts), bestBin)
