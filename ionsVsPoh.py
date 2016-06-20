# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 03:26:18 2016

@author: normandy
"""

from scipy.interpolate import UnivariateSpline

from scipy.integrate import quadrature

import numpy as np

import matplotlib.pyplot as plt
import matplotlib

import readline
import MDSplus

from scipy.stats import linregress

import pandas as pd

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
    vsurf = np.gradient(ssibryNode.data()) / np.median(np.diff(ssibryNode.dim_of().data())) * 6.28
    didt = np.gradient(np.abs(cpasmaNode.data())) / np.median(np.diff(cpasmaNode.dim_of().data()))
    
    #vsurf = np.ediff1d(ssibryNode.data(), to_end=0) / np.median(np.diff(ssibryNode.dim_of().data())) * 6.28
    #didt = np.ediff1d(np.abs(cpasmaNode.data()), to_end=0) / np.median(np.diff(cpasmaNode.dim_of().data()))
    
    vi = L * np.interp(liNode.dim_of().data(), cpasmaNode.dim_of().data(), didt)
    ip = np.interp(liNode.dim_of().data(), cpasmaNode.dim_of().data(), np.abs(cpasmaNode.data()))
    return liNode.dim_of().data(), ip*(vsurf-vi)/1e6
    
def getPohFull(anaTree):
    ssibryNode = anaTree.getNode('\\analysis::efit_ssibry')
    cpasmaNode = anaTree.getNode('\\analysis::efit_aeqdsk:cpasma')
    liNode = anaTree.getNode('\\analysis::efit_aeqdsk:ali')
    L = liNode.data() * 6.28 * 67 * 1e-9
    
    vsurf = np.gradient(ssibryNode.data()) / np.median(np.diff(ssibryNode.dim_of().data())) * 6.28
    didt = np.gradient(np.abs(cpasmaNode.data())) / np.median(np.diff(cpasmaNode.dim_of().data()))
    
    vi = L * np.interp(liNode.dim_of().data(), cpasmaNode.dim_of().data(), didt)
    
    return ssibryNode.dim_of().data(), vsurf, liNode.dim_of().data(), liNode.data(), cpasmaNode.dim_of().data(), np.abs(cpasmaNode.data()), liNode.dim_of().data(), vi
    

def calcStoredEnergy(minRadius, maxRadius, td, nl04Node, anaTree):
    #rmagxNode = anaTree.getNode('\\analysis::efit_aeqdsk:rmagx')
    #aoutNode = anaTree.getNode('\\analysis::efit_aeqdsk:aout')
    
    #rmagxSampled = np.interp(td.hetime, rmagxNode.dim_of().data(), rmagxNode.data())
    #aoutSampled = np.interp(td.hetime, aoutNode.dim_of().data(), aoutNode.data())
    #nl04Sampled = np.interp(td.hetime, nl04Node.dim_of().data(), nl04Node.data())
    
    outputs = np.zeros(len(td.hetime))
    
    for i in range(len(td.hetime)):
        #func = lambda rho: td.splines[i](rho)*rho*aoutSampled[i]*6.28*1.5*6.28*rmagxSampled[i]
        func = lambda rho: 6.28*td.splines[i](rho)*rho/(3.14*(maxRadius**2-minRadius**2))
        func(0.2)
        outputs[i], _ = quadrature(func, minRadius, maxRadius)      
        
    return outputs

"""
anaTree = MDSplus.Tree('analysis', 1120106020)
magTree = MDSplus.Tree('magnetics', 1120106020)

a,b,c,d,e,f,g,h = getPohFull(anaTree)

plt.figure()

ax1 = plt.plot(a,b, label='surface', marker='.')
plt.plot(g,h, label='self-induction', marker='+')
plt.plot(g,b-h, label='total', marker='x')
plt.legend(fontsize=18, loc='lower right')
plt.title('Voltages [V]')
plt.xlabel('Time [sec]')
plt.xlim([0.71, 1.49])
plt.autoscale(True, 'y')
plt.tight_layout()

plt.figure()
plt.plot(c,d)
plt.title('Inductance [H]')
plt.xlabel('Time [sec]')
plt.xlim([0.71, 1.49])
plt.autoscale(True, 'y')
plt.tight_layout()

ipNode = magTree.getNode('\magnetics::ip')
vcurNode = anaTree.getNode('\\analysis::efit_fitout:sumif')
plt.figure()
#plt.plot(ipNode.dim_of().data(),np.abs(ipNode.data())/1e6, label='Plasma')
plt.plot(vcurNode.dim_of().data(),np.abs(vcurNode.data())/1e6, label='Vessel', marker='+')
plt.title('Vessel Current [MA]')
plt.xlabel('Time [sec]')
plt.xlim([0.71, 1.49])
plt.autoscale(True, 'y')
plt.tight_layout()
"""



#plt.close('all')


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



#data2 = data
#data = data[data['HIREX Time'] > 8]
"""



#plt.figure()
f, (ax1, ax2) = plt.subplots(2,1, sharex=True)

currShots = data[np.abs(np.abs(data['Ip'])-1.1) < 0.1]
#currShots = currShots[currShots['Shot Number'] < 1120217000]
currShots = currShots[currShots['RF']<0.1]
currShots = currShots[np.isfinite(currShots['ti jump'])]
currShots = currShots[np.isfinite(currShots['poh max'])]
currShots = currShots[currShots['HIREX Time'] > 8]
slope, intercept, r_value, p_value, std_err = linregress(currShots['poh max'], currShots['ti jump'])
#plt.scatter(currShots['nl_04']/0.6, currShots['ti jump']/currShots['poh max'], c='g', marker='^', label='1.1 MA')
ax1.scatter(currShots['nl_04']/0.6, currShots['ti jump'], c='g', marker='^', label='1.1 MA')
ax2.scatter(currShots['nl_04']/0.6, currShots['poh max'], c='g', marker='^', label='1.1 MA')
#plt.figure()
#plt.scatter(currShots['poh max'], currShots['ti jump'])
#plt.plot(np.linspace(0,0.6), slope*np.linspace(0,0.6)+intercept, c='r')
#plt.xlabel('Ohmic Power Jump [MW]')
#plt.ylabel('Ti Jump [keV]')
#plt.xlim([0.0, 0.8])
#plt.ylim([0.0, 0.3])
#plt.title('1.1 MA')
plt.tight_layout()


currShots = data[np.abs(np.abs(data['Ip'])-0.8) < 0.1]
#currShots = currShots[currShots['Shot Number'] < 1120217000]
currShots = currShots[currShots['RF']<0.1]
currShots = currShots[np.isfinite(currShots['ti jump'])]
currShots = currShots[np.isfinite(currShots['poh max'])]
currShots = currShots[currShots['HIREX Time'] > 8]
regressShots = currShots[currShots['poh max'] < 0.5]
slope, intercept, r_value, p_value, std_err = linregress(regressShots['poh max'], regressShots['ti jump'])
#plt.scatter(currShots['nl_04']/0.6, currShots['ti jump']/currShots['poh max'], c='b', marker='o', label='0.8 MA')
ax1.scatter(currShots['nl_04']/0.6, currShots['ti jump'], c='b', marker='o', label='0.8 MA')
ax2.scatter(currShots['nl_04']/0.6, currShots['poh max'], c='b', marker='o', label='0.8 MA')
#plt.plot(np.linspace(0,0.5), slope*np.linspace(0,0.5)+intercept, c='r')
#plt.xlabel('Ohmic Power Jump [MW]')
#plt.ylabel('Ti Jump [keV]')
#plt.xlim([0.0, 0.8])
#plt.ylim([0.0, 0.3])
#plt.title('0.8 MA')
#plt.tight_layout()


currShots = data[np.abs(np.abs(data['Ip'])-0.55) < 0.1]
#currShots = currShots[currShots['Shot Number'] < 1120217000]
currShots = currShots[currShots['RF']<0.1]
currShots = currShots[np.isfinite(currShots['ti jump'])]
currShots = currShots[np.isfinite(currShots['poh max'])]
slope, intercept, r_value, p_value, std_err = linregress(currShots['poh max'], currShots['ti jump'])
#plt.scatter(currShots['nl_04']/0.6, currShots['ti jump']/currShots['poh max'], c='r', marker='D', label='0.55 MA')
ax1.scatter(currShots['nl_04']/0.6, currShots['ti jump'], c='r', marker='D', label='0.55 MA')
ax2.scatter(currShots['nl_04']/0.6, currShots['poh max'], c='r', marker='D', label='0.55 MA')
#plt.figure()
#plt.scatter(currShots['poh max'], currShots['ti jump'])
#plt.plot(np.linspace(0,0.6), slope*np.linspace(0,0.6)+intercept, c='r')
#plt.xlabel('Ohmic Power Jump [MW]')
#plt.ylabel('Ti Jump [keV]')
#plt.xlim([0.0, 0.8])
#plt.ylim([0.0, 0.3])
#plt.title('0.55 MA')


"""
currShots = data[np.abs(np.abs(data['Ip'])-0.8) < 0.1]
#currShots = currShots[currShots['Shot Number'] < 1120217000]
currShots = currShots[currShots['RF']>0.5]
currShots = currShots[currShots['RF']<0.7]
currShots = currShots[np.isfinite(currShots['ti jump'])]
currShots = currShots[np.isfinite(currShots['poh max'])]
currShots = currShots[currShots['HIREX Time'] > 8]
slope, intercept, r_value, p_value, std_err = linregress(currShots['poh max'], currShots['ti jump'])
#plt.figure()
plt.scatter(currShots['poh max'], currShots['ti jump'], c='c', marker='o')
#plt.plot(np.linspace(0,0.5), slope*np.linspace(0,0.5)+intercept, c='r')
plt.xlabel('Ohmic Power Jump [MW]')
plt.ylabel('Ti Jump [keV]')
plt.xlim([0.0, 0.8])
plt.ylim([0.0, 0.3])
plt.title('0.6 MW ICRH')
#plt.scatter(currShots['nl_04']/0.6, currShots['ti jump']/currShots['poh max'], c='c', marker='o', label='0.6 MW ICRH')

currShots = data[np.abs(np.abs(data['Ip'])-0.8) < 0.1]
#currShots = currShots[currShots['Shot Number'] < 1120217000]
currShots = currShots[currShots['RF']>1.1]
currShots = currShots[currShots['RF']<1.3]
currShots = currShots[np.isfinite(currShots['ti jump'])]
currShots = currShots[np.isfinite(currShots['poh max'])]
currShots = currShots[currShots['HIREX Time'] > 8]
slope, intercept, r_value, p_value, std_err = linregress(currShots['poh max'], currShots['ti jump'])
plt.scatter(currShots['poh max'], currShots['ti jump'], c='m', marker='o')
#plt.plot(np.linspace(0,0.5), slope*np.linspace(0,0.5)+intercept, c='r')
plt.xlabel('Ohmic Power Jump [MW]')
plt.ylabel('Ti Jump [keV]')
plt.xlim([0.0, 0.8])
plt.ylim([0.0, 0.3])
plt.title('Ohmic vs. ICRH')
#plt.scatter(currShots['nl_04']/0.6, currShots['ti jump']/currShots['poh max'], c='m', marker='o', label='1.2 MW ICRH')
"""

"""
#currShots = currShots[np.isfinite(currShots['ti jump'])]
#currShots = currShots[np.isfinite(currShots['poh max'])]

#plt.xlabel('Line Average Density [$10^{20} m^{-3}$]')
#plt.ylabel('Response [$\Delta keV / \Delta MW$]')
ax1.legend(loc='upper right', fontsize=18)

ax1.axvline(x=0.6, c='r', ls='--')
ax1.axvline(x=0.75, c='b', ls='--')
ax1.axvline(x=1.1, c='g', ls='--')
ax2.axvline(x=0.6, c='r', ls='--')
ax2.axvline(x=0.75, c='b', ls='--')
ax2.axvline(x=1.1, c='g', ls='--')

ax1.set_ylabel('Ion Temp Jump [keV]')
ax2.set_ylabel('Power Jump [MW]')
ax2.set_xlabel('Line-Average Density [$10^{20} m^{-3}$]')

plt.tight_layout()
"""

#f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2, sharex='col', sharey='row')
"""
plt.figure()
ax1 = plt.subplot(2, 2, 1)
ax2 = plt.subplot(2, 2, 2, sharey=ax1)
ax2.yaxis.set_visible(False)
ax3 = plt.subplot(2, 2, 3, sharex=ax1)
ax3.xaxis.set_visible(False)
ax1.xaxis.tick_top()
ax2.xaxis.tick_top()

currShots = data[np.abs(np.abs(data['Ip'])-1.1) < 0.1]
#currShots = currShots[currShots['Shot Number'] < 1120217000]
currShots = currShots[np.isfinite(currShots['ti jump'])]
currShots = currShots[np.isfinite(currShots['poh max'])]
ax1.scatter(currShots['nl_04 peak']/currShots['nl_04']*100, currShots['poh max']/currShots['poh0']*100, c='g', marker=6)
ax2.scatter(currShots['Ip peak']/currShots['Ip']*100, currShots['poh max']/currShots['poh0']*100, c='g', marker=6)
ax3.scatter(currShots['nl_04 peak']/currShots['nl_04']*100, currShots['Ip peak']/currShots['Ip']*100, c='g', marker=6, label='1.1 MA')


currShots = data[np.abs(np.abs(data['Ip'])-0.8) < 0.1]
#currShots = currShots[currShots['Shot Number'] < 1120217000]
currShots = currShots[np.isfinite(currShots['ti jump'])]
currShots = currShots[np.isfinite(currShots['poh max'])]
ax1.scatter(currShots['nl_04 peak']/currShots['nl_04']*100, currShots['poh max']/currShots['poh0']*100, c='b', marker='.')
ax2.scatter(currShots['Ip peak']/currShots['Ip']*100, currShots['poh max']/currShots['poh0']*100, c='b', marker='.')
ax3.scatter(currShots['nl_04 peak']/currShots['nl_04']*100, currShots['Ip peak']/currShots['Ip']*100, c='b', marker='.', label='0.8 MA')

currShots = data[np.abs(np.abs(data['Ip'])-0.55) < 0.1]
#currShots = currShots[currShots['Shot Number'] < 1120217000]
currShots = currShots[np.isfinite(currShots['ti jump'])]
currShots = currShots[np.isfinite(currShots['poh max'])]
ax1.scatter(currShots['nl_04 peak']/currShots['nl_04']*100, currShots['poh max']/currShots['poh0']*100, c='r', marker='x')
ax2.scatter(currShots['Ip peak']/currShots['Ip']*100, currShots['poh max']/currShots['poh0']*100, c='r', marker='x')
ax3.scatter(currShots['nl_04 peak']/currShots['nl_04']*100, currShots['Ip peak']/currShots['Ip']*100, c='r', marker='x', label='0.55 MA')

leg = ax3.legend(fontsize=18)
leg.draggable()
"""


"""
shot = -1

for i, row in data.iterrows():
    if row['Shot Number'] != shot:
        print shot
        shot = row['Shot Number']
        anaTree = MDSplus.Tree('analysis', shot)
        
    a, b, litime, li, c, d, e, f = getPohFull(anaTree)
    p = row['Pulse Time']
    
    li0 = getVals(p, litime, li, slope=False)
    littmax, limax, a, b = getPeaks(p, litime, li, li0)
    
    data.set_value(i, 'li0', li0)
    data.set_value(i, 'li0 peak', limax)
"""

"""
plt.figure()
plt.scatter(np.abs(data['Ip peak']/data['Ip']), data['li0 peak'] / data['li0'])
plt.plot([0, 0.035], [0, 0.07])
"""


"""
specTree = MDSplus.Tree('spectroscopy', 1120106020)
heNode = specTree.getNode('\SPECTROSCOPY::TOP.HIREXSR.ANALYSIS.HELIKE.PROFILES.Z')
td = ThacoData(heNode, None)
td.fitSplines()
anaTree = MDSplus.Tree('analysis', 1120106020)
pohTime, poh = getPoh(anaTree)
outs = calcStoredEnergy(0, 0.8, td, None, anaTree)


fig, ax1 = plt.subplots()
ax1.plot(td.hetime, outs, label='Ion Temperature')
#ax1.plot(0, 0, c='r', marker='+', label='Ohmic Power')
ax2 = ax1.twinx()
ax2.plot(0, 0, label='Ion Temperature')
ax2.plot(pohTime, poh, c='r', marker='+', label='Ohmic Power')
ax1.set_xlabel('time [sec]')
ax1.set_ylabel('Ion Temperature [keV]')
ax2.set_ylabel('Ohmic Power [MW]')
ax2.legend(loc='upper left', fontsize=18)
"""
