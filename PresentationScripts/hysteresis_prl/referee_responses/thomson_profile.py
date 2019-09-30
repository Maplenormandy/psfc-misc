# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 14:07:00 2019

@author: normandy
"""

import numpy as np
import matplotlib.pyplot as plt

import MDSplus

# %%


elecTree1 = MDSplus.Tree('electrons', 1160506005)
neNode1 = elecTree1.getNode(r'\ELECTRONS::TOP.YAG_NEW.RESULTS.PROFILES:NE_RZ')
neErrNode1 = elecTree1.getNode(r'\ELECTRONS::TOP.YAG_NEW.RESULTS.PROFILES:NE_ERR')
rNode1 = elecTree1.getNode(r'\ELECTRONS::TOP.YAG_NEW.RESULTS.PROFILES:R_MID_T')

neeNode1 = elecTree1.getNode(r'\ELECTRONS::TOP.YAG_EDGETS.RESULTS:NE')

time1 = neNode1.dim_of().data()
ne1 = neNode1.data()/1e20
neErr1 = neErrNode1.data()/1e20
r1 = rNode1.data()

# %%

i1,i2 = np.searchsorted(time1, (0.7, 1.15))

ne1_red = ne1[:,i1:i2]
r1_red = r1[:,i1:i2]
neErr1_red = neErr1[:,i1:i2]

ne1_ma = np.ma.masked_less(ne1_red, 0.1)

#avg_ma1 = np.ma.median(ne1_ma, axis=-1)
#avg_ne1 = np.ma.median(ne1_ma)

avg_ma1 = np.ma.average(ne1_ma, axis=-1, weights=1/neErr1_red**2)
avg_ne1 = np.ma.average(ne1_ma, weights=1/neErr1_red**2)

avg_r1 = np.average(r1_red, axis=-1)

pfactor = avg_ne1 / avg_ma1

plt.errorbar(np.ravel(r1_red), np.ravel(ne1_red), c='r', fmt='o')
plt.plot(avg_r1, avg_ma1)

# %%

elecTree = MDSplus.Tree('electrons', 1120216012)
neNode = elecTree.getNode(r'\ELECTRONS::TOP.YAG_NEW.RESULTS.PROFILES:NE_RZ')
rNode = elecTree.getNode(r'\ELECTRONS::TOP.YAG_NEW.RESULTS.PROFILES:R_MID_T')

time = neNode.dim_of().data()
ne = neNode.data()
r = rNode.data()

#pfactor[9] = 1
# %%

i1,i2 = np.searchsorted(time, (0.95, 1.1))

ne_red = ne[:,i1:i2]
r_red = r[:,i1:i2]

ne_ma = np.ma.masked_less(ne_red, 1e19)
avg_ma = np.ma.median(ne_ma, axis=-1)
avg_ne = np.ma.median(ne_ma)
avg_r = np.average(r_red, axis=-1)

plt.scatter((np.ravel(r_red)-0.67)/0.22, np.ravel(ne_red), c='r')