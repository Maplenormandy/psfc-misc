# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 11:19:59 2016

@author: normandy
"""

import numpy as np

import matplotlib.pyplot as plt
import matplotlib

import readline
import MDSplus

from scipy.ndimage.filters import gaussian_filter1d

def getPohFull(efitTree):
    ssibryNode = efitTree.getNode(r'\efit01::efit_ssibry')
    cpasmaNode = efitTree.getNode(r'\efit01::efit_aeqdsk:cpasma')
    liNode = efitTree.getNode(r'\EFIT01::TOP.RESULTS.A_EQDSK:ALI')
    L = liNode.data() * 6.28 * 67 * 1e-9
    
    vsurf = gaussian_filter1d(ssibryNode.data(), 1, order=1, truncate=2.0) / np.median(np.diff(ssibryNode.dim_of().data())) * 6.28
    didt = gaussian_filter1d(np.abs(cpasmaNode.data()), 1, order=1, truncate=2.0) / np.median(np.diff(cpasmaNode.dim_of().data()))
    #vsurf = np.gradient(ssibryNode.data()) / np.median(np.diff(ssibryNode.dim_of().data())) * 6.28
    #didt = np.gradient(np.abs(cpasmaNode.data())) / np.median(np.diff(cpasmaNode.dim_of().data()))
    
    #vsurf = np.ediff1d(ssibryNode.data(), to_end=0) / np.median(np.diff(ssibryNode.dim_of().data())) * 6.28
    #didt = np.ediff1d(np.abs(cpasmaNode.data()), to_end=0) / np.median(np.diff(cpasmaNode.dim_of().data()))
    
    vi = L * np.interp(liNode.dim_of().data(), cpasmaNode.dim_of().data(), didt)
    ip = np.interp(liNode.dim_of().data(), cpasmaNode.dim_of().data(), np.abs(cpasmaNode.data()))
    return liNode.dim_of().data(), ip*(vsurf-vi)/1e6
    
efitTree = MDSplus.Tree('efit01', 1160720015)
a, b = getPohFull(efitTree)

plt.plot(a,b)

node = efitTree.getNode(r'\efit01::TOP.RESULTS.A_EQDSK:aaq2')
plt.plot(node.dim_of().data(), node.data())