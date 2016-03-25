# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 19:02:17 2016

@author: normandy
"""

import numpy as np

import readline
import MDSplus

import matplotlib.pyplot as plt

import shotAnalysisTools as sat

from scipy.stats import linregress

import pandas as pd

import time

readline

def runs(day):
    return range(day*1000+1, day*1000+40)

shotList = runs(1120106) + runs(1120216) + runs(1120605) + runs(1120607) + runs(1120620) + runs(1120720) + runs(1120919) + runs(1120926) + runs(1140327) + runs(1140328) + runs(1140402) + runs(1150901) + runs(1150903)

def unpack(node):
    return node.dim_of().data(), node.data()
    
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
    
    

headers = ['Shot Number',
           'Pulse Time',
           'Ip','Ip Slope',
           'Bt','Bt Slope',
           'nl_04','nl_04 Slope',
           'RF',
           'LSN/USN',
           'HIREX Time',
           'Zvel','Zvel Slope',
           'Zint','Zint Slope',
           'Zti', 'Zti Slope',
           'Avel','Avel Slope',
           'Aint','Aint Slope',
           'Ati', 'Ati Slope',
           'nl_04 peak','nl_04 tt peak',
           'Ip peak', 'Ip tt peak',
           'Zint max', 'Zint tt max',
           'Zvel max', 'Zvel tt max',
           'Zti max', 'Zti tt max',
           'Zint min', 'Zint tt min',
           'Zvel min', 'Zvel tt min',
           'Zti min', 'Zti tt min',
           'Aint max', 'Aint tt max',
           'Avel max', 'Avel tt max',
           'Ati max', 'Ati tt max',
           'Aint min', 'Aint tt min',
           'Avel min', 'Avel tt min',
           'Ati min', 'Ati tt min']
    
df = pd.DataFrame([], columns=headers)    
    
for shot in shotList:
    try:
        pulses = sat.findColdPulses(shot)
    except:
        print shot, "error finding pulses"
        continue
    
    if len(pulses) == 0:
        print shot, "no pulses"
        continue
    else:
        print shot, len(pulses), "cold pulses"
        
        elecTree = MDSplus.Tree('electrons', shot)
        magTree = MDSplus.Tree('magnetics', shot)
        rfTree = MDSplus.Tree('rf', shot)
        specTree = MDSplus.Tree('spectroscopy', shot)
        anaTree = MDSplus.Tree('analysis', shot)
        
        try:
            ssepNode = anaTree.getNode('\\analysis::efit_aeqdsk:ssep') # USN>0, LSN<0
            densNode = elecTree.getNode('\ELECTRONS::TOP.TCI.RESULTS:nl_04')
            gpcNode = elecTree.getNode('\gpc_t0')
            ipNode = magTree.getNode('\magnetics::ip')
            q95Node = anaTree.getNode('\\analysis::efit_aeqdsk:qpsib')
            rfNode = rfTree.getNode('\\rf::rf_power_net')
            btorNode = magTree.getNode('\magnetics::btor')
            
            ssept, ssep = unpack(ssepNode)
            denst, dens = unpack(densNode)
            gpct, gpc = unpack(gpcNode)
            ipt, ip = unpack(ipNode)
            q95t, q95 = unpack(q95Node)
            rft, rf = unpack(rfNode)
            btort, btor = unpack(btorNode)
        except:
            print shot, "unpack error"
        

        
        
        
        try:
            zintNode = specTree.getNode('\SPECTROSCOPY::TOP.HIREX_SR.ANALYSIS.Z:INT')
            zvelNode = specTree.getNode('\SPECTROSCOPY::TOP.HIREX_SR.ANALYSIS.Z:VEL')
            ztiNode = specTree.getNode('\SPECTROSCOPY::TOP.HIREX_SR.ANALYSIS.Z:TI')
        except:
            zintNode = None
            
        try:
            aintNode = specTree.getNode('\SPECTROSCOPY::TOP.HIREX_SR.ANALYSIS.A:INT')
            avelNode = specTree.getNode('\SPECTROSCOPY::TOP.HIREX_SR.ANALYSIS.A:VEL')
            atiNode = specTree.getNode('\SPECTROSCOPY::TOP.HIREX_SR.ANALYSIS.A:TI')
        except:
            aintNode = None
        
        # HIREX time rate
        if np.any(zintNode != None):
            zintt = zintNode.dim_of().data()
            zint = zintNode.data()[0]
            zvelt = zvelNode.dim_of().data()
            zvel = zvelNode.data()[0]
            ztit = ztiNode.dim_of().data()
            zti = ztiNode.data()[0]
            hirexRate = int(np.round(np.mean(np.diff(zintt)) * 1000))
        else:
            hirexRate = 0
            
        if np.any(aintNode != None):
            aintt = aintNode.dim_of().data()
            aint = aintNode.data()
            avelt = avelNode.dim_of().data()
            avel = avelNode.data()
            atit = atiNode.dim_of().data()
            ati = atiNode.data()
        
        for p in pulses:
            row = {'Shot Number': shot,
                   'Pulse Time': p }
            
            ssepp = getVals(p, ssept, ssep, False)
        
            if ssepp != None:
                
                
                if ssepp < -0.5:
                    row['LSN/USN'] = 'LSN'
                elif ssepp > 0.5:
                    row['LSN/USN'] = 'USN'
                else:
                    row['LSN/USN'] = 'DN'
            else:
                continue
            
            row['HIREX Time'] = hirexRate
            row['RF'] = getVals(p, rft, rf, False)
                
                
            row['nl_04 Slope'], row['nl_04'] = getVals(p, denst, dens/1e20)
            row['nl_04 tt peak'], row['nl_04 peak'], d1, d2 = getPeaks(p, denst, dens/1e20, row['nl_04'], 0.04)
            
            row['Ip Slope'], row['Ip'] = getVals(p, ipt, ip/1e6)
            d1, d2, row['Ip tt peak'], row['Ip peak'] = getPeaks(p, ipt, np.abs(ip)/1e6, np.abs(row['Ip']), 0.04)
            
            row['Bt Slope'], row['Bt'] = getVals(p, btort, btor)
            
            if np.any(zintNode != None):
                row['Zint Slope'], row['Zint'] = getVals(p, zintt, zint)
                row['Zint tt max'], row['Zint max'], row['Zint tt min'], row['Zint min'] = getPeaks(p, zintt, zint, row['Zint'])
                row['Zvel Slope'], row['Zvel'] = getVals(p, zvelt, zvel)
                row['Zvel tt max'], row['Zvel max'], row['Zvel tt min'], row['Zvel min'] = getPeaks(p, zvelt, zvel, row['Zvel'])
                row['Zti Slope'], row['Zti'] = getVals(p, ztit, zti)
                row['Zti tt max'], row['Zti max'], row['Zti tt min'], row['Zti min'] = getPeaks(p, ztit, zti, row['Zti'])
                
            if np.any(aintNode != None):
                row['Aint Slope'], row['Aint'] = getVals(p, aintt, aint)
                row['Aint tt max'], row['Aint max'], row['Aint tt min'], row['Aint min'] = getPeaks(p, aintt, aint, row['Aint'])
                row['Avel Slope'], row['Avel'] = getVals(p, avelt, avel)
                row['Avel tt max'], row['Avel max'], row['Avel tt min'], row['Avel min'] = getPeaks(p, avelt, avel, row['Avel'])
                row['Ati Slope'], row['Ati'] = getVals(p, atit, ati)
                row['Ati tt max'], row['Ati max'], row['Ati tt min'], row['Ati min'] = getPeaks(p, atit, ati, row['Ati'])
            
            df = df.append(row, ignore_index=True)

    
    time.sleep(0.1)