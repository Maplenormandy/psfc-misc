# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 13:52:05 2018

@author: normandy
"""

import numpy as np

import MDSplus

import eqtools

import sys

sys.path.append('/home/normandy/git/psfc-misc/Fitting')
import profiles_fits
sys.path.append('/home/normandy/git/psfc-misc/Geometry')
import neotools

import cPickle as pkl

# %% General function definitions

class ThacoData:
    def __init__(self, thtNode, shot=None, tht=None, path='.HELIKE.PROFILES.Z', time=0.95):
        if (thtNode == None):
            self.shot = shot
            self.specTree = MDSplus.Tree('spectroscopy', shot)

            if (tht == 0):
                self.tht = ''
            else:
                self.tht = str(tht)

            self.thtNode = self.specTree.getNode('\SPECTROSCOPY::TOP.HIREXSR.ANALYSIS' + self.tht + path)
        else:
            self.thtNode = thtNode

        e = eqtools.CModEFITTree(shot)

        proNode = self.thtNode.getNode('PRO')
        perrNode = self.thtNode.getNode('PROERR')
        rhoNode = self.thtNode.getNode('RHO')

        rpro = proNode.data()
        rperr = perrNode.data()
        rrho = rhoNode.data()
        rtime = rhoNode.dim_of()

        goodTimes = (rtime > 0).sum()

        self.time = rtime.data()[:goodTimes]
        self.rho = rrho[0,:] # Assume unchanging rho bins
        self.roa = e.psinorm2roa(self.rho, time)
        self.pro = rpro[:,:goodTimes,:len(self.rho)]
        self.perr = rperr[:,:goodTimes,:len(self.rho)]

def trimNodeData(node, t0=0.5, t1=1.5):
    time = node.dim_of().data()
    data = node.data()
    i0, i1 = np.searchsorted(time, (t0, t1))
    return time[i0:i1+1], data[i0:i1+1]
def trimData(time, data, t0=0.5, t1=1.5):
    i0, i1 = np.searchsorted(time, (t0, t1))
    return time[i0:i1+1], data[i0:i1+1]

# %% Data loading

#reload(profiles_fits)

rotdb_csv = np.loadtxt('rotdb.csv', delimiter=',', skiprows=1)
failed = []

for r in range(rotdb_csv.shape[0]):
    if r <= 43:
        continue
    
    shot = int(rotdb_csv[r, 0])
    sl = int(rotdb_csv[r, 1])
    time = float(rotdb_csv[r, 2])
    window = float(rotdb_csv[r, 3])
    
    t0 = time-window
    t1 = time+window
    
    print '=== %d ==='%shot
    
    q95Node = MDSplus.Tree('analysis', shot).getNode(r'\ANALYSIS::EFIT_AEQDSK:QPSIB')
    q95_t0, q95_t1 = np.searchsorted(q95Node.dim_of().data(), (t0, t1))
    q95 = np.abs(np.mean(q95Node.data()[q95_t0:q95_t1]))
    
    st_estimate = (1.0/q95-0.17)/(0.3-0.17) * (0.45-0.3) + 0.3
    #print 'ip',ip
    #print st_estimate
    
    shot_pkl = {}
    
    try:
        shot_pkl['ne_fit'] = profiles_fits.get_ne_fit(shot=shot, t_min=t0, t_max=t1, plot=False, x0_mean=st_estimate)
        shot_pkl['te_fit'] = profiles_fits.get_te_fit(shot=shot, t_min=t0, t_max=t1, plot=False, x0_mean=st_estimate)
        shot_pkl['ti_fit'] = profiles_fits.get_ti_fit(shot=shot, t_min=t0, t_max=t1, plot=False, x0_mean=st_estimate, te_fit=shot_pkl['te_fit'], THT=8)
        shot_pkl['vtor_fit'] = profiles_fits.get_vtor_fit(shot=shot, t_min=t0, t_max=t1, plot=False, x0_mean=st_estimate, THT=8)
    
    
        eg = neotools.EquilibriumGeometry(shot, (t0, t1), e=profiles_fits.e)
        zeff = eg.calculate_zeff_neo(shot_pkl['ne_fit'], shot_pkl['te_fit'], shot_pkl['ti_fit'], ft_method='lin-liu95')
        
        shot_pkl['zeff'] = zeff
    
    except:
        failed.append(r)
        continue
    
    print 'q95', q95
    print 'zeff', zeff

    pkl.dump(shot_pkl, open('/home/normandy/git/psfc-misc/DataMining/CmodRotdb/fit_%d_%d.pkl'%(shot, sl), 'w'), protocol=pkl.HIGHEST_PROTOCOL)