# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 13:52:05 2018

@author: normandy
"""

import numpy as np

import MDSplus


import cPickle as pkl

import matplotlib.pyplot as plt
from matplotlib import cm

import sys
sys.path.append('/home/normandy/git/psfc-misc/Geometry')
import neotools

# %% General function definitions

rotdb_csv = np.loadtxt('rotdb.csv', delimiter=',', skiprows=1)


def loadData(shot, tree, node, t0, t1):
    node = MDSplus.Tree(tree, shot).getNode(node)
    ind_t0, ind_t1 = np.searchsorted(node.dim_of().data(), (t0, t1))
    return  np.abs(np.mean(node.data()[ind_t0:ind_t1]))
    
shot_pkls = []
shot_ips = []
shot_q95s = []
shot_dens = []

for r in range(rotdb_csv.shape[0]):
    shot = int(rotdb_csv[r, 0])
    sl = int(rotdb_csv[r, 1])
    time = float(rotdb_csv[r, 2])
    window = float(rotdb_csv[r, 3])
    
    t0 = time-window
    t1 = time+window
    
    try:
        shot_pkl = pkl.load(open('/home/normandy/git/psfc-misc/DataMining/CmodRotdb/fit_%d_%d.pkl'%(shot, sl), 'r'))
        
        #eg = neotools.EquilibriumGeometry(shot, (t0, t1))
        #zeff = eg.calculate_zeff_neo(shot_pkl['ne_fit'], shot_pkl['te_fit'], shot_pkl['ti_fit'], ft_method='lin-liu95')
        
        #shot_pkl['zeff'] = zeff
        
    except:
        continue
    
    
    
    
    q95 = loadData(shot, 'analysis', r'\ANALYSIS::EFIT_AEQDSK:QPSIB', t0, t1)
    ip = loadData(shot, 'magnetics', r'\MAGNETICS::IP', t0, t1)
    
    shot_pkls.append(shot_pkl)
    shot_ips.append(ip)
    shot_q95s.append(q95)
    shot_dens.append(np.percentile(shot_pkl['ne_fit']['y'], 95))
    

# %%
    
shot_zeffs = []
shot_te = []

for shot_pkl in shot_pkls:
    shot_zeffs.append(shot_pkl['zeff'])
    shot_te.append(np.percentile(shot_pkl['te_fit']['y'], 95))

# %%

shot_zeffs = np.array(shot_zeffs)
shot_te = np.array(shot_te)
shot_ips = np.array(shot_ips)
shot_q95s = np.array(shot_q95s)
shot_dens = np.array(shot_dens)

#%%

plt.figure()

plt.scatter(shot_dens, shot_zeffs/shot_te**2, c=cm.plasma((shot_q95s - 2.0) / 5.0))
plt.axhline(1.0, ls='--', c='k')
#plt.scatter(shot_dens, 1.5/shot_dens**2 + 1, c='b')
    