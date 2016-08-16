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

import scipy

import sys
sys.path.append('/home/normandy/git/psfc-misc/Common')

import ShotAnalysisTools as sat
from Collisionality import NustarProfile

import dill
import pickle

import traceback

readline

df_source = pd.read_csv('_all_reversals.csv')

df_source_revs = df_source.dropna(subset=['reversals'])

# %% Trawl the code
headers = [
    'shot',
    'time',
    'mods',
    
    'nl_04',    
    'bt',
    'ip',
    'p_rf',
    'ssep',
    'zave',
    'q95',
    
    'numin',
    'xmin',
    
    'Lne',
    'LTe',
    
    'a',
    'R'
    ]

df = pd.DataFrame([], columns=headers)

def smoothedFunction(node):
    t = node.dim_of().data()
    y = node.data()
    
    def smoothed(t_eval):
        i0, i1 = np.searchsorted(t, [t_eval-0.01, t_eval+0.01])
        return np.median(y[i0:i1])
        
    return smoothed

for j, shotRow in df_source_revs.iterrows():
    shot = shotRow['shot']
    
    if shotRow['p_lh_t'] > 0:
        print shotRow['shot'], "LH on"
        continue
    else:
        reversals = [x.strip() for x in shotRow['reversals'].split(',')]
        print shotRow['shot'], len(reversals), 'reversals'
        
            
        try:
            rfTree = MDSplus.Tree('rf', shot)
            elecTree = MDSplus.Tree('electrons', shot)
            magTree = MDSplus.Tree('magnetics', shot)
            anaTree = MDSplus.Tree('analysis', shot)
            edgeTree = MDSplus.Tree('edge', shot)            
                
            ssepNode = anaTree.getNode(r'\analysis::efit_aeqdsk:ssep') # USN>0, LSN<0
            q95Node = anaTree.getNode(r'\analysis::efit_aeqdsk:qpsib')
            densNode = elecTree.getNode('\ELECTRONS::TOP.TCI.RESULTS:nl_04')
            ipNode = magTree.getNode('\magnetics::ip')
            rfNode = rfTree.getNode(r'\rf::rf_power_net')
            btorNode = magTree.getNode('\magnetics::btor')
            
            ssep = smoothedFunction(ssepNode)
            q95 = smoothedFunction(q95Node)
            dens = smoothedFunction(densNode)
            ip = smoothedFunction(ipNode)
            rf = smoothedFunction(rfNode)
            btor = smoothedFunction(btorNode)
        except:
            traceback.print_exc()
            continue
        
        rtimes = np.zeros(len(reversals))
            
        for i in range(len(reversals)):
            revt = ''.join([x for x in reversals[i] if not str.isalpha(x) and x!='?'])
            rtimes[i] = float(revt)
            
        try:
            if shotRow['tmax'] + 1e-3 >= 1.5:
                nustar = NustarProfile(shotRow['shot'], 0.4, 1.6)
            else:
                nustar = NustarProfile(shotRow['shot'], 0.4, shotRow['tmax']-0.1)
            nustar.fitNe(rtimes)
            nustar.evalProfile(rtimes)
            nustar.calcMinTrace()
            
            with open('__nustar_' + str(shot) + '.p', 'w') as f:
                pickle.dump(nustar, f, -1)
        except:
            traceback.print_exc()
            continue
            
        for i in range(len(reversals)):
            revt = rtimes[i]
            row = { 'shot': shotRow['shot'] }
            mods = ''.join([x for x in reversals[i] if str.isalpha(x) or x=='?'])
            
            
            row['time'] = revt
            row['mods'] = mods
            
            row['nl_04'] = dens(revt)
            row['bt'] = btor(revt)
            row['ip'] = ip(revt) / 1e6
            row['p_rf'] = rf(revt)
            row['ssep'] = ssep(revt)
            row['q95'] = q95(revt)
            row['zave'] = float(nustar.zefff(revt))
            row['R'] = float(nustar.magRf(revt))
            row['a'] = float(nustar.magaf(revt))
            
            row['numin'] = nustar.numinTrace[i]
            xmin = nustar.xminTrace[i]
            row['xmin'] = xmin
            
            row['Lne'] = nustar.neFit[i](xmin) / nustar.dne[i](xmin)
            row['Lne'] = -row['Lne'][0]
            row['LTe'] = nustar.TeCrash[i](xmin) / nustar.dTeCrash[i](xmin)
            row['LTe'] = -row['LTe'][0]
            
            df = df.append(row, ignore_index=True)
    
# %% Separate them

df.to_csv('__nustar_output.csv')


# %%

def runfilt(x):
    run = int(x['shot']/1000)
    return run not in [1140327, 1140328, 1140402, 1150616, 1150617, 1160617, 1140415, 1150630, 1120210, 1120222]
    #return run in [1160506]

df_filt = df[np.abs(np.abs(df['bt'])-5.4)<0.2]
df_norf = df_filt[df_filt['p_rf'] < 0.1]

df_filt2 = df_filt[df_filt.apply(lambda x: runfilt(x), axis=1)]
df_norf2 = df_filt2[df_filt2['p_rf'] < 0.1]

df_trim = df[(df['xmin'] < 0.5) & (df['xmin'] > 0.33)]

dn = df_trim[df_trim['mods'] == 'd']
up = df_trim[df_trim['mods'] == 'u']

fwd = dn[np.abs(dn['bt'] + 5.4) < 0.1]
rev = up[np.abs(up['bt'] + 5.4) < 0.1]

fwd_norf = fwd[fwd['p_rf'] < 0.1]
rev_norf = rev[rev['p_rf'] < 0.1]

#fwd_mid = fwd[(fwd['shot'] > 1160506000) & (fwd['shot'] < 1160507000)]
#rev_mid = rev[(rev['shot'] > 1160506000) & (rev['shot'] < 1160507000)]

#fwd_mid = fwd_norf[np.abs(fwd_norf['ip'] + 0.8) < 0.1]
#rev_mid = rev_norf[np.abs(rev_norf['ip'] + 0.8) < 0.1]

# %% Plot them

#plt.scatter(fwd_mid['nl_04'] * fwd_mid['q95'], fwd_mid['shot'], c='b', marker='v')
#plt.scatter(rev_mid['nl_04'] * rev_mid['q95'], rev_mid['shot'], c='r', marker='^')

#plt.scatter(fwd_mid['nl_04'], fwd_mid['q95'], c='b', marker='v')
#plt.scatter(rev_mid['numin'], rev_mid['q95'], c='r', marker='^')

#plt.scatter(fwd_norf['q95'], fwd_norf['xmin'], c='b', marker='v')
#plt.scatter(rev_norf['q95'], rev_norf['xmin'], c='r', marker='^')

#plt.scatter(fwd_norf['q95']*fwd_norf['nl_04'], fwd_norf['numin'], c='b', marker='v')
#plt.scatter(rev_norf['q95']*rev_norf['nl_04'], rev_norf['numin'], c='r', marker='^')

#plt.scatter(fwd_norf['q95'], fwd_norf['nl_04'], c='b', marker='v')
#plt.scatter(rev_norf['q95'], rev_norf['nl_04'], c='r', marker='^')

#plt.scatter(df[df['p_rf']<0.1]['nl_04'] * df[df['p_rf']<0.1]['q95'], df[df['p_rf']<0.1]['numin'])

#plt.scatter(df_norf[df_norf['mods']=='d']['q95'], df_norf[df_norf['mods']=='d']['nl_04'], c='b', marker='v')
#plt.scatter(df_norf[df_norf['mods']=='u']['q95'], df_norf[df_norf['mods']=='u']['nl_04'], c='r', marker='^')

#plt.scatter(df_norf['q95'], df_norf['nl_04'], c='g')
#plt.scatter(df_norf2['q95'], df_norf2['nl_04'], c='r')

plt.scatter(df_norf['q95'], df_norf['nl_04'])
plt.scatter(df_norf2['q95'], df_norf2['nl_04'], c='g')

m, b, r, p, sy = scipy.stats.linregress(df_norf['q95'], df_norf['nl_04'])

pivot = 3*m+b

pivoted = df_norf[(df_norf['nl_04'] < pivot) & (df_norf['q95'] < 3.5)]
print map(int, pivoted['shot'])
