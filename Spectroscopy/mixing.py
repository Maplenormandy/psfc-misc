# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 18:05:11 2016

@author: normandy
"""

import numpy as np

import matplotlib.pyplot as plt
import matplotlib

import scipy.integrate as integrate
import scipy.special as special
from scipy.ndimage.filters import gaussian_filter1d

astar = 0
bstar = 0
cstar = 0.6
dstar = -0.2
estar = 0

e_trans = 3.104 # in keV

zgk = 16-0.5 # Z - screening factor

csfte = np.array([0.5,1,1.5,2,5,7,10,15,23,32,52,74,100,165,235,310,390,475,655,845,1000,1441,1925,2454,3030,3655,4331,5060,5844,6685,7585,8546,10000,20000,50000,100000])
ar16csf = np.array([0,0,0,0,0,0,0,0,0,0,1.058e-36,2.458e-24,7.623e-17,2.1438e-8,0.00014053,0.014225,0.15973,0.46114,0.79789,0.88031,0.88385,0.7487,0.50779,0.29588,0.16162,0.088679,0.050593,0.030089,0.018794,0.012152,0.008169,0.0056789,0.0035697,0.00052215,5.8474e-5,1.0888e-5])


def omegastar(y):
    return astar + (bstar * y - cstar * y**2 + dstar * y**3 + estar) * \
            np.exp(y) * special.exp1(y) + (cstar + dstar) * y - dstar * y**2
            
            
def excitationRate(te):
    y = e_trans/te
    return 8.62e-6 * (omegastar(y) / zgk**2) * np.sqrt(te*1000) * np.exp(-y)
    
def emissivity(te):
    csf = np.interp(np.log(te*1000), np.log(csfte), ar16csf)
    return excitationRate(te)*csf
    



def waveDist(x):
    return 1.0/np.sqrt(1-x**2)/ np.pi
    
def waveIntegral(te0, tet):
    tePlot = np.linspace(te0-tet+0.01, te0+tet-0.01)
    r1 = integrate.quad(lambda te: waveDist((te-te0)/tet)/tet*emissivity(te)*(te-1)*te, te0-tet+0.01, te0+tet-0.01)
    r2 = integrate.quad(lambda te: waveDist((te-te0)/tet)/tet*emissivity(te)*(te-1), te0-tet+0.01, te0+tet-0.01)
    plt.plot(tePlot, waveDist((tePlot-te0)/tet)/tet*emissivity(tePlot)*(tePlot-1))
    return r1[0]/r2[0]
    
#print waveIntegral(1.8, 0.4)


import readline
import MDSplus

def getPoh(efitTree):
    ssibryNode = efitTree.getNode(r'\EFIT01::TOP.RESULTS.G_EQDSK:SSIBRY')
    cpasmaNode = efitTree.getNode(r'\EFIT01::TOP.RESULTS.A_EQDSK:CPASMA')
    liNode = efitTree.getNode(r'\EFIT01::TOP.RESULTS.A_EQDSK:ALI')
    rmagxNode = efitTree.getNode(r'\EFIT01::TOP.RESULTS.A_EQDSK:RMAGX')
    L = liNode.data() * 6.28 * rmagxNode.data() * 1e-9
    
    vsurf = gaussian_filter1d(ssibryNode.data(), 15, order=1) / np.median(np.diff(ssibryNode.dim_of().data())) * 6.28
    didt = gaussian_filter1d(np.abs(cpasmaNode.data()), 15, order=1) / np.median(np.diff(cpasmaNode.dim_of().data()))
    #vsurf = np.gradient(ssibryNode.data()) / np.median(np.diff(ssibryNode.dim_of().data())) * 6.28
    #didt = np.gradient(np.abs(cpasmaNode.data())) / np.median(np.diff(cpasmaNode.dim_of().data()))
    
    #vsurf = np.ediff1d(ssibryNode.data(), to_end=0) / np.median(np.diff(ssibryNode.dim_of().data())) * 6.28
    #didt = np.ediff1d(np.abs(cpasmaNode.data()), to_end=0) / np.median(np.diff(cpasmaNode.dim_of().data()))
    
    vi = L * np.interp(liNode.dim_of().data(), cpasmaNode.dim_of().data(), didt)
    ip = np.interp(liNode.dim_of().data(), cpasmaNode.dim_of().data(), np.abs(cpasmaNode.data()))
    return liNode.dim_of().data(), ip*(vsurf-vi)/1e6
    
def getInductiveEnergy(efitTree):
    cpasmaNode = efitTree.getNode(r'\EFIT01::TOP.RESULTS.A_EQDSK:CPASMA')
    liNode = efitTree.getNode(r'\EFIT01::TOP.RESULTS.A_EQDSK:ALI')
    rmagxNode = efitTree.getNode(r'\EFIT01::TOP.RESULTS.A_EQDSK:RMAGX')
    
    L = liNode.data() * 6.28 * rmagxNode.data() * 1e-9
    
    return liNode.dim_of().data(), L
    
    
efitTree = MDSplus.Tree('efit01', 1160503012)
a, b = getPoh(efitTree)
plt.plot(a,b)