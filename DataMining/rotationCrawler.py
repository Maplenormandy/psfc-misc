# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 19:57:23 2016

@author: normandy
"""

import numpy as np

import readline
import MDSplus

import matplotlib.pyplot as plt
import time
import itertools

import pandas as pd

import sys
sys.path.append('/home/normandy/git/psfc-misc')

import shotAnalysisTools as sat

readline

#f = open('allshots.txt', 'r')
#allshots = (int(line) for line in f)
def runs(day):
    return range(day*1000+1, day*1000+40)

with open('_rotationRunList.txt', 'r') as f:
    shotList = list(itertools.chain.from_iterable(map(lambda x: runs(int(x)), f.readlines())))

print shotList

def getValAndRange(data, maxtime, threshold=-1.0):
    t, y = data
    good = (t >= 0.5) & (t <= maxtime)
    above = None
    if threshold > 0.0:
        above = y > threshold
        good = good & above
        
        if np.sum(good) > 0.5:        
            a, b, c = np.nanpercentile(y[good], [5, 50, 95])
            return b, c-a, np.trapz(above[good]*1.0, t[good])
        else:
            return 0.0, 0.0, 0.0
    else:
        a, b, c = np.nanpercentile(y[good], [5, 50, 95])
        return b, c-a

        
    
def unpack(node):
    return node.dim_of().data(), node.data()
    

headers = [
    'shot',
    'ar_signal',
    'tmax',
    
    'nl_04',
    'nl_04_r',
    
    'bt',
    'bt_r',
    
    'ip',
    'ip_r',
    
    'p_rf',
    'p_rf_r',
    'p_rf_t',
    
    'ssep',
    'ssep_r',
    
    'vel',
    'vel_r',
    
    'lbo',
    'cryopump'
    ]

df = pd.DataFrame([], columns=headers)


for shot in shotList:
    time.sleep(0.2)
    
    try:
        rfTree = MDSplus.Tree('rf', shot)
        specTree = MDSplus.Tree('spectroscopy', shot)
        elecTree = MDSplus.Tree('electrons', shot)
        magTree = MDSplus.Tree('magnetics', shot)
        anaTree = MDSplus.Tree('analysis', shot)
        edgeTree = MDSplus.Tree('edge', shot)
        
        ssepNode = anaTree.getNode(r'\analysis::efit_aeqdsk:ssep') # USN>0, LSN<0
        densNode = elecTree.getNode('\ELECTRONS::TOP.TCI.RESULTS:nl_04')
        ipNode = magTree.getNode('\magnetics::ip')
        rfNode = rfTree.getNode(r'\rf::rf_power_net')
        btorNode = magTree.getNode('\magnetics::btor')
        
        if shot > 1160000000:
            intNode = specTree.getNode('\SPECTROSCOPY::TOP.HIREX_SR.ANALYSIS.W:INT')
            rotNode = specTree.getNode('\SPECTROSCOPY::TOP.HIREX_SR.ANALYSIS.W:VEL')
        else:
            intNode = specTree.getNode('\SPECTROSCOPY::TOP.HIREX_SR.ANALYSIS.Z:INT')
            rotNode = specTree.getNode('\SPECTROSCOPY::TOP.HIREX_SR.ANALYSIS.Z:VEL')
            
        cryoNode = edgeTree.getNode(r'\F_CRYO_MKS')
    except:
        print shot, "Exception during node load"
        continue
    
    row = {'shot': shot}
    
    time.sleep(0.2)
    
    try:
        artime = intNode.dim_of().data()
        arint = intNode.data()[0]
        
        argood = arint > 0
        artime = artime[argood]
        arint = arint[argood]
        
        if len(arint) < 10:
            print shot, "Not enough data points"
            continue
        
        tmax = np.max(artime)
        tmax = min(1.5, tmax)
        if tmax <= 0.5:
            print shot, "Startup disruption"
            continue
        
        row['tmax'] = tmax
        row['ar_signal'], throwaway = getValAndRange((artime, arint), tmax)
        
        if row['ar_signal'] < 50:
            print shot, "Not enough argon signal"
            continue
        
        ar_signal = row['ar_signal']
        
        row['nl_04'], row['nl_04_r'] = getValAndRange(unpack(densNode), tmax)
        row['bt'], row['bt_r'] = getValAndRange(unpack(btorNode), tmax)
        row['ip'], row['ip_r'] = getValAndRange(unpack(ipNode), tmax)
        row['p_rf'], row['p_rf_r'], row['p_rf_t'] = getValAndRange(unpack(rfNode), tmax, 0.1)
        row['ssep'], row['ssep_r'] = getValAndRange(unpack(ssepNode), tmax)
        
        row['vel'], row['vel_r'] = getValAndRange((rotNode.dim_of().data(), rotNode.data()[0]), tmax)
        
        row['cryopump'] = np.median(cryoNode.data())
            
    except:
        print shot, "Exception during data load"
        continue
    
    try:
        pulses = sat.findColdPulses(shot)
        row['lbo'] = len(pulses)
    except:
        row['lbo'] = 0    
    
    df = df.append(row, ignore_index=True)
    print shot, ar_signal
    
    time.sleep(0.1)
    
df['ip'] /= 1e6
df['ip_r'] /= 1e6
