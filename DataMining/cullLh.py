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
    
df = pd.read_csv('__toCull.csv')

for i, row in df.iterrows():
    try:
        lhTree = MDSplus.Tree('lh', row['shot'])
        lhNode = lhTree.getNode(r'\LH::TOP.RESULTS:NETPOW')
        
        df.loc[i, 'p_lh'], df.loc[i, 'p_lh_r'], df.loc[i, 'p_lh_t'] = getValAndRange(unpack(lhNode), row['tmax'], 50)
    except:
        df.loc[i, 'p_lh'] = 0.0
        df.loc[i, 'p_lh_r'] = 0.0
        df.loc[i, 'p_lh_t'] = 0.0
    