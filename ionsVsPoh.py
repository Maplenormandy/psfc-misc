# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 03:26:18 2016

@author: normandy
"""

from scipy.ndimage.filters import gaussian_filter1d
from scipy.interpolate import UnivariateSpline

from scipy.integrate import quadrature

import numpy as np

import matplotlib.pyplot as plt
import matplotlib

import readline
import MDSplus

from scipy.stats import linregress

import pandas as pd

readline

font = {'family': 'normal', 'size': 24}
matplotlib.rc('font', **font)

def getVals(p, t, x, slope=True):
    i = np.searchsorted(t, p)
    if i == len(t):
        return None
    elif slope:
        j = np.searchsorted(t, p-0.061)
        
        tfit = t[j:i] - p
        xfit = x[j:i]
        
        slope, intercept, r_value, p_value, std_err = linregress(tfit, xfit)
        
        return slope, intercept
    else:
        return x[i]
        
def getPeaks(p, t, x, x0, delay=0.061):
    i = np.searchsorted(t, p)
    j = np.searchsorted(t, p+delay)
    xslice = x[i:j+1]
    
    imax = np.argmax(xslice)
    imin = np.argmin(xslice)
    
    return t[imax+i]-p, xslice[imax]-x0, t[imin+i]-p, xslice[imin]-x0

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
        
        if hyNode != None:
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
            
            self.hashy = True
        else:
            self.hashy = False
        
    def fitSplines(self):
        self.splines = [None] * len(self.hetime)
        for i in range(len(self.hetime)):
            rho = self.herho
            pro = self.hepro[3,i,:]
            perr = self.heperr[3,i,:]
            
            self.splines[i] = UnivariateSpline(np.sqrt(rho), pro, w=1.0/perr, k=5)
        

class ThomsonCoreData:
    def __init__(self, yagNode):
        self.yagNode = yagNode

        densNode = self.yagNode.getNode('NE_RZ')
        rmidNode = self.yagNode.getNode('R_MID_T')
        derrNode = self.yagNode.getNode('NE_ERR')

        rdens = densNode.data()
        rrmid = rmidNode.data()
        rderr = derrNode.data()
        rtime = densNode.dim_of().data()

        goodTimes = rrmid[0] > 0

        self.dens = np.array(rdens[:,goodTimes])
        self.rmid = np.array(rrmid[:,goodTimes])
        self.time = np.array(rtime[goodTimes])
        self.derr = np.array(rderr[:,goodTimes])
        
    def remapRadius(self, anaTree):
        rmagxNode = anaTree.getNode('\\analysis::efit_aeqdsk:rmagx')
        aoutNode = anaTree.getNode('\\analysis::efit_aeqdsk:aout')
        
        rmagxSampled = np.interp(self.time, rmagxNode.dim_of().data(), rmagxNode.data())
        aoutSampled = np.interp(self.time, aoutNode.dim_of().data(), aoutNode.data())
        
        self.rho = (self.rmid*100 - rmagxSampled) / aoutSampled
        
    def fitSplines(self, timebase):
        self.splines = [None] * len(timebase)
        for i in range(len(timebase)):
            densData = np.array([np.interp(timebase[i], self.time, self.dens[j,:]) for j in range(self.dens.shape[0])])
            rhoData = np.array([np.interp(timebase[i], self.time, self.rho[j,:]) for j in range(self.rho.shape[0])])
            derrData = np.array([np.interp(timebase[i], self.time, self.derr[j,:]) for j in range(self.derr.shape[0])])
            
            densData = np.append(densData[::-1], densData)
            rhoData = np.append(-rhoData[::-1], rhoData)
            derrData = np.append(derrData[::-1], derrData)
            self.splines[i] = UnivariateSpline(rhoData, densData, w=1.0/derrData, k=5)
        
        
def getPoh(anaTree):
    ssibryNode = anaTree.getNode('\\analysis::efit_ssibry')
    cpasmaNode = anaTree.getNode('\\analysis::efit_aeqdsk:cpasma')
    liNode = anaTree.getNode('\\analysis::efit_aeqdsk:ali')
    L = liNode.data() * 6.28 * 67 * 1e-9
    
    #vsurf = gaussian_filter1d(ssibryNode.data(), 1, order=1, truncate=1.0) / np.median(np.diff(ssibryNode.dim_of().data())) * 6.28
    #didt = gaussian_filter1d(np.abs(cpasmaNode.data()), 1, order=1, truncate=1.0) / np.median(np.diff(cpasmaNode.dim_of().data()))
    timax / pohmax
    vsurf = np.gradient(ssibryNode.data()) / np.median(np.diff(ssibryNode.dim_of().data())) * 6.28
    didt = np.gradient(np.abs(cpasmaNode.data())) / np.median(np.diff(cpasmaNode.dim_of().data()))
    
    #vsurf = np.ediff1d(ssibryNode.data(), to_end=0) / np.median(np.diff(ssibryNode.dim_of().data())) * 6.28
    #didt = np.ediff1d(np.abs(cpasmaNode.data()), to_end=0) / np.median(np.diff(cpasmaNode.dim_of().data()))
    
    vi = L * np.interp(liNode.dim_of().data(), cpasmaNode.dim_of().data(), didt)
    ip = np.interp(liNode.dim_of().data(), cpasmaNode.dim_of().data(), np.abs(cpasmaNode.data()))
    return liNode.dim_of().data(), ip*(vsurf-vi)/1e6
    

def calcStoredEnergy(minRadius, maxRadius, td, nl04Node, anaTree):
    rmagxNode = anaTree.getNode('\\analysis::efit_aeqdsk:rmagx')
    aoutNode = anaTree.getNode('\\analysis::efit_aeqdsk:aout')
    
    rmagxSampled = np.interp(td.hetime, rmagxNode.dim_of().data(), rmagxNode.data())
    aoutSampled = np.interp(td.hetime, aoutNode.dim_of().data(), aoutNode.data())
    nl04Sampled = np.interp(td.hetime, nl04Node.dim_of().data(), nl04Node.data())
    
    outputs = np.zeros(len(td.hetime))
    
    for i in range(len(td.hetime)):
        #func = lambda rho: td.splines[i](rho)*rho*aoutSampled[i]*6.28*1.5*6.28*rmagxSampled[i]
        func = lambda rho: 6.28*td.splines[i](rho)*rho/(3.14*(maxRadius**2-minRadius**2))
        func(0.2)
        outputs[i], _ = quadrature(func, minRadius, maxRadius)      
        
    return outputs
    

#anaTree = MDSplus.Tree('analysis', 1150903021)
#plt.plot(*getPoh(anaTree))


plt.close('all')



"""
data = pd.read_csv('pulsesTrawled_hirexonly.csv')
shot = -1


for i, row in data.iterrows():
    if row['Shot Number'] != shot:
        shot = int(row['Shot Number'])
        print shot
        specTree = MDSplus.Tree('spectroscopy', shot)
        try:
            if shot < 1150000000:
                heNode = specTree.getNode('\SPECTROSCOPY::TOP.HIREXSR.ANALYSIS.HELIKE.PROFILES.Z')
            else:
                heNode = specTree.getNode('\SPECTROSCOPY::TOP.HIREXSR.ANALYSIS1.HELIKE.PROFILES.Z')
        except:
            heNode = specTree.getNode('\SPECTROSCOPY::TOP.HIREXSR.ANALYSIS.HELIKE.PROFILES.Z')
            
        #if np.median(heNode.getNode('pro').dim_of().data()) > 0.021:
        #    heNode = specTree.getNode('\SPECTROSCOPY::TOP.HIREXSR.ANALYSIS.HELIKE.PROFILES.Z')
            
        #hyNode = specTree.getNode('\SPECTROSCOPY::TOP.HIREXSR.ANALYSIS1.HLIKE.PROFILES.LYA1')
        
        try:
            td = ThacoData(heNode, None)
        except:
            shot = -1
            continue
        
        td.fitSplines()
        #plt.plot(td.herho, td.hepro[3,12,:])
        #plt.plot(np.linspace(0,1), td.splines[12](np.sqrt(np.linspace(0,1))))
        elecTree = MDSplus.Tree('electrons', shot)
        anaTree = MDSplus.Tree('analysis', shot)
        
        nl04Node = elecTree.getNode('\ELECTRONS::TOP.TCI.RESULTS:NL_04')
        
        #tsc = ThomsonCoreData(elecTree.getNode('\ELECTRONS::TOP.YAG_NEW.RESULTS.PROFILES'))
        #tsc.remapRadius(anaTree)
        #tsc.fitSplines(td.hetime)
        #plt.plot(np.linspace(0,0.8), tsc.splines[19](np.linspace(0,0.8)))
        
        outs = calcStoredEnergy(0, 0.8, td, nl04Node, anaTree)
        pohTime, poh = getPoh(anaTree)
        
        
        #fig, ax1 = plt.subplots()
        #ax1.plot(td.hetime, outs, label='Ion Temperature [keV]')
        #ax2 = ax1.twinx()
        #ax2.[1.1611,1.39972]plot(pohTime + 0.02, poh, c='r', label='Ohmic Power [MW]')
        #ax1.set_xlabel('time [sec]')
        #ax1.set_ylabel('Ion Temperature [keV] (blue)')
        #ax2.set_ylabel('Ohmic Power [MW] (red)')
        #ax1.set_title(str(shot))
        
    
    p = row['Pulse Time']
    
    
    
    
    poh0 = getVals(p, pohTime + 0.04, poh, slope=False)
    pohttmax, pohmax, a, b = getPeaks(p, pohTime+0.04, poh, poh0)
    slope0, ti0 = getVals(p, td.hetime, outs, slope=True)
    tittmax, timax, a, timin = getPeaks(p, td.hetime, outs, ti0)
    
    if pohmax < 0.01:
        pohmax = np.nan
    if timax-timin > 0.3:
        tijump = np.nan
        timax = np.nan
        ti0 = np.nan
    else:
        tijump = timax-timin
        
    
    data.set_value(i, 'poh max', pohmax)
    data.set_value(i, 'ti jump', tijump)
    data.set_value(i, 'poh abs max', pohmax+poh0)
    data.set_value(i, 'ti abs max', timax+ti0)
    data.set_value(i, 'poh0', poh0)
    data.set_value(i, 'ti0', ti0)
"""

#data2 = data
#data = data[data['HIREX Time'] > 8]

plt.figure()

currShots = data[np.abs(np.abs(data['Ip'])-0.8) < 0.1]
#currShots = currShots[currShots['Shot Number'] < 1120217000]
currShots = currShots[currShots['RF']<0.1]
currShots = currShots[np.isfinite(currShots['ti jump'])]
currShots = currShots[np.isfinite(currShots['poh max'])]
regressShots = currShots[currShots['poh max'] < 0.5]
slope, intercept, r_value, p_value, std_err = linregress(regressShots['poh max'], regressShots['ti jump'])
plt.scatter(currShots['nl_04']/0.6, (currShots['ti jump']) / currShots['poh max'], c='b', marker='o', label='0.8 MA')
#plt.figure()
#plt.scatter(currShots['poh max'], currShots['ti jump'])
#plt.plot(np.linspace(0,0.5), slope*np.linspace(0,0.5)+intercept, c='r')
#plt.xlabel('Ohmic Power Jump [MW]')
#plt.ylabel('Ti Jump [keV]')
#plt.xlim([0.0, 0.8])
#plt.ylim([0.0, 0.3])
#plt.title('0.8 MA')
plt.tight_layout()

currShots = data[np.abs(np.abs(data['Ip'])-1.1) < 0.1]
#currShots = currShots[currShots['Shot Number'] < 1120217000]
currShots = currShots[currShots['RF']<0.1]
currShots = currShots[np.isfinite(currShots['ti jump'])]
currShots = currShots[np.isfinite(currShots['poh max'])]
slope, intercept, r_value, p_value, std_err = linregress(currShots['poh max'], currShots['ti jump'])
plt.scatter(currShots['nl_04']/0.6, (currShots['ti jump']) / currShots['poh max'], c='g', marker='^', label='1.1 MA')
#plt.figure()
#plt.scatter(currShots['poh max'], currShots['ti jump'])
#plt.plot(np.linspace(0,0.6), slope*np.linspace(0,0.6)+intercept, c='r')
#plt.xlabel('Ohmic Power Jump [MW]')
#plt.ylabel('Ti Jump [keV]')
#plt.xlim([0.0, 0.8])
#plt.ylim([0.0, 0.3])
#plt.title('1.1 MA')
plt.tight_layout()

currShots = data[np.abs(np.abs(data['Ip'])-0.55) < 0.1]
#currShots = currShots[currShots['Shot Number'] < 1120217000]
currShots = currShots[currShots['RF']<0.1]
currShots = currShots[np.isfinite(currShots['ti jump'])]
currShots = currShots[np.isfinite(currShots['poh max'])]
slope, intercept, r_value, p_value, std_err = linregress(currShots['poh max'], currShots['ti jump'])
plt.scatter(currShots['nl_04']/0.6, (currShots['ti jump']) / currShots['poh max'], c='r', marker='D', label='0.55 MA')
#plt.figure()
#plt.scatter(currShots['poh max'], currShots['ti jump'])
#plt.plot(np.linspace(0,0.6), slope*np.linspace(0,0.6)+intercept, c='r')
#plt.xlabel('Ohmic Power Jump [MW]')
#plt.ylabel('Ti Jump [keV]')
#plt.xlim([0.0, 0.8])
#plt.ylim([0.0, 0.3])
#plt.title('0.55 MA')

plt.xlabel('Line Average Density [$10^{20} m^{-3}$]')
plt.ylabel('Stiffness [$\Delta keV / \Delta MW$]')
plt.legend(loc='upper right')

plt.tight_layout()

"""
fig, ax1 = plt.subplots()
ax1.plot(td.hetime, outs, label='Ion Temperature [keV]')
ax2 = ax1.twinx()
ax2.plot(pohTime + 0.02, poh, c='r', label='Ohmic Power [MW]')
ax1.set_xlabel('time [sec]')
ax1.set_ylabel('Ion Temperature [keV] (blue)')
ax2.set_ylabel('Ohmic Power [MW] (red)')
ax1.set_title(str(shot))
"""