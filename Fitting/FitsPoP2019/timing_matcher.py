# -*- coding: utf-8 -*-
"""
Utility script to find the right time matching windows for fitting using GPC

Created on Mon Aug  5 13:15:42 2019

@author: normandy
"""

import numpy as np
import matplotlib.pyplot as plt

import MDSplus

# %%

class EceData:
    def __init__(self, shot):
        elecTree = MDSplus.Tree('electrons', shot)
        
        gpc_te = []
        
        gpc_r = []
        
        gpc_t = []
        
        for i in range(1,9):
            teNode = elecTree.getNode(r'\electrons::gpc_te'+str(i))
            rNode = elecTree.getNode(r'\ELECTRONS::gpc_r'+str(i))
            
            gpc_te.append(teNode.data())
            gpc_r.append(rNode.data())
            gpc_t = teNode.dim_of().data()
            
        self.gpc_te = np.array(gpc_te)
        self.gpc_r = np.array(gpc_r)
        self.gpc_t = gpc_t
            
d = EceData(1160506009)

# %%

i1, i2 = np.searchsorted(d.gpc_t, (0.93, 0.99))
i3, i4 = np.searchsorted(d.gpc_t, (0.61, 0.67))

plt.scatter(range(d.gpc_te.shape[0]), np.average(d.gpc_te[:,i1:i2], axis=-1), c='r')
plt.scatter(range(d.gpc_te.shape[0]), np.average(d.gpc_te[:,i3:i4], axis=-1), c='b')