# -*- coding: utf-8 -*-
"""
Script to find and apply twiddle factors to Thomson ne data and save it

Created on Thu Aug  1 14:07:00 2019

@author: normandy
"""

import numpy as np
import matplotlib.pyplot as plt

import MDSplus

import pickle

import eqtools

import scipy.interpolate

# %%


elecTree1 = MDSplus.Tree('electrons', 1160506005)
neNode1 = elecTree1.getNode(r'\ELECTRONS::TOP.YAG_NEW.RESULTS.PROFILES:NE_RZ')
neErrNode1 = elecTree1.getNode(r'\ELECTRONS::TOP.YAG_NEW.RESULTS.PROFILES:NE_ERR')
rNode1 = elecTree1.getNode(r'\ELECTRONS::TOP.YAG_NEW.RESULTS.PROFILES:R_MID_T')

nl04Node = elecTree1.getNode('\ELECTRONS::TOP.TCI.RESULTS:NL_04')

neEdgeNode1 = elecTree1.getNode(r'\ELECTRONS::TOP.YAG_EDGETS.RESULTS:NE')
rEdgeNode1 = elecTree1.getNode(r'\ELECTRONS::TOP.YAG_EDGETS.RESULTS:RMID')

nebarNode = elecTree1.getNode(r'\ELECTRONS::TOP.TCI.RESULTS.INVERSION:NEBAR_EFIT')

time1 = neNode1.dim_of().data()
ne1 = neNode1.data()/1e20
nee1 = neEdgeNode1.data()/1e20
neErr1 = neErrNode1.data()/1e20
r1 = rNode1.data()
re1 = rEdgeNode1.data()

nl041 = nl04Node.data()/1e20
nl04t = nl04Node.dim_of().data()

referenceFit = pickle.load(open('/home/normandy/git/psfc-misc/Fitting/ne_dict_fit_1160725013.pkl'))
refFit = scipy.interpolate.interp1d(referenceFit['X'], referenceFit['y'])

e = eqtools.CModEFITTree(1160506005)

# %%

# To future generations: At one point I wanted to remove the density perturbation
# at t=0.9s, but it looks like it performs better if you keep it in
i1,i2 = np.searchsorted(time1, (0.7, 0.9))
i3,i4 = np.searchsorted(time1, (0.9, 1.15))

i5,i6 = np.searchsorted(nebarNode.dim_of().data(), (0.75, 1.15))

ne1_red = np.concatenate((ne1[:,i1:i2], ne1[:,i3:i4]), axis=-1)
nee1_red = np.concatenate((nee1[:,i1:i2], nee1[:,i3:i4]), axis=-1)
r1_red = np.concatenate((r1[:,i1:i2], r1[:,i3:i4]), axis=-1)
re1_red = np.concatenate((re1[:,i1:i2], re1[:,i3:i4]), axis=-1)
time1_red = np.concatenate((time1[i1:i2], time1[i3:i4]))
t1_red = np.tile(time1_red, (r1_red.shape[0], 1))
te1_red = np.tile(time1_red, (re1_red.shape[0], 1))

roa1_red = e.rmid2roa(r1_red, t1_red, each_t=False)
roae1_red = e.rmid2roa(re1_red, te1_red, each_t=False)

ne1_ma = np.ma.masked_outside(ne1_red, 0.1, 1.5)
nee1_ma = np.ma.masked_outside(nee1_red, 0.1, 1.5)

pfactor = np.ma.average(refFit(roa1_red) / ne1_ma, axis=-1)
efactor = np.ma.average(refFit(roae1_red) / nee1_ma, axis=-1)

ne1_ma = np.ma.masked_outside(ne1_red, 0.1, 1.5)
nee1_ma = np.ma.masked_outside(nee1_red, 0.1, 1.5)

#avg_ma1 = np.ma.median(ne1_ma, axis=-1)
#avg_ne1 = np.ma.median(ne1_ma)
#avg_mae1 = np.ma.median(nee1_ma, axis=-1)

avg_ma1 = np.ma.average(ne1_ma, axis=-1)
avg_ne1 = np.average(nebarNode.data()[i5:i6]/1e20)
avg_mae1 = np.ma.average(nee1_ma, axis=-1)

avg_r1 = np.average(r1_red, axis=-1)
avg_re1 = np.average(re1_red, axis=-1)

#pfactor = avg_ne1 / avg_ma1
#efactor = avg_ne1 / avg_mae1

padd = avg_ne1 - avg_ma1
eadd = avg_ne1 - avg_mae1

plt.figure()
plt.errorbar(np.ravel(roa1_red), np.ravel(ne1_red), c='r', fmt='o', label='Thomson')
plt.errorbar(np.ravel(roae1_red), np.ravel(nee1_red), c='r', fmt='o')
plt.plot(referenceFit['X'], referenceFit['y'], c='b')
#plt.plot(avg_r1, avg_ma1, c='b', label='TS average')
#plt.plot(avg_re1, avg_mae1, c='b')
plt.axhline(avg_ne1, c='g', ls='--', label='TCI nebar')
plt.legend(loc='lower right')
plt.xlabel('Rmid [m]')
plt.ylabel('density [1e20 m^-3]')
plt.title('Locked mode 1160506005')

# %%

pfactor1 = pfactor
efactor1 = efactor

#pfactor1[0] = np.ma.mean(pfactor)
#pfactor[2] = 1
pfactor1[9] = 1

padd1 = padd
eadd1 = eadd

padd[0] = 0
padd[9] = 0

#pfactor1 = np.sqrt(pfactor1)
#efactor1 = np.sqrt(efactor1)

# %%

f, axes = plt.subplots(3, 1, sharex=True, sharey=True)

elecTree = MDSplus.Tree('electrons', 1160506007)
neNode = elecTree.getNode(r'\ELECTRONS::TOP.YAG_NEW.RESULTS.PROFILES:NE_RZ')
rNode = elecTree.getNode(r'\ELECTRONS::TOP.YAG_NEW.RESULTS.PROFILES:R_MID_T')

neEdgeNode = elecTree.getNode(r'\ELECTRONS::TOP.YAG_EDGETS.RESULTS:NE')
rEdgeNode = elecTree.getNode(r'\ELECTRONS::TOP.YAG_EDGETS.RESULTS:RMID')

time = neNode.dim_of().data()
ne = neNode.data()
r = rNode.data()

nee = neEdgeNode.data()
re = rEdgeNode.data()

i1,i2 = np.searchsorted(time, (0.57, 0.63))

ne_red = ne[:,i1:i2]
r_red = r[:,i1:i2]
nee_red = nee[:,i1:i2]
re_red = re[:,i1:i2]

axes[0].scatter((np.ravel(r_red)-0.67)/0.22, np.ravel(ne_red), c='b')
axes[0].scatter((np.ravel(re_red)-0.67)/0.22, np.ravel(nee_red), c='b')

axes[1].scatter((np.ravel(r_red)-0.67)/0.22, np.ravel(ne_red*pfactor1[:,np.newaxis]), c='b')
axes[1].scatter((np.ravel(re_red)-0.67)/0.22, np.ravel(nee_red*efactor1[:,np.newaxis]), c='b')
#plt.scatter((np.ravel(r_red)-0.67)/0.22, np.ravel(ne_red+padd1[:,np.newaxis]), c='b')
#plt.scatter((np.ravel(re_red)-0.67)/0.22, np.ravel(nee_red+eadd1[:,np.newaxis]), c='b')



i1,i2 = np.searchsorted(time, (0.93, 0.99))

ne_red = ne[:,i1:i2]
r_red = r[:,i1:i2]
nee_red = nee[:,i1:i2]
re_red = re[:,i1:i2]

axes[0].scatter((np.ravel(r_red)-0.67)/0.22, np.ravel(ne_red), c='r')
axes[0].scatter((np.ravel(re_red)-0.67)/0.22, np.ravel(nee_red), c='r')

axes[1].scatter((np.ravel(r_red)-0.67)/0.22, np.ravel(ne_red*pfactor1[:,np.newaxis]), c='r')
axes[1].scatter((np.ravel(re_red)-0.67)/0.22, np.ravel(nee_red*efactor1[:,np.newaxis]), c='r')
#plt.scatter((np.ravel(r_red)-0.67)/0.22, np.ravel(ne_red+padd1[:,np.newaxis]), c='r')
#plt.scatter((np.ravel(re_red)-0.67)/0.22, np.ravel(nee_red+eadd1[:,np.newaxis]), c='r')

elecTree = MDSplus.Tree('electrons', 1120216017)
neNode = elecTree.getNode(r'\ELECTRONS::TOP.YAG_NEW.RESULTS.PROFILES:NE_RZ')
rNode = elecTree.getNode(r'\ELECTRONS::TOP.YAG_NEW.RESULTS.PROFILES:R_MID_T')

neEdgeNode = elecTree.getNode(r'\ELECTRONS::TOP.YAG_EDGETS.RESULTS:NE')
rEdgeNode = elecTree.getNode(r'\ELECTRONS::TOP.YAG_EDGETS.RESULTS:RMID')

time = neNode.dim_of().data()
ne = neNode.data()
r = rNode.data()

nee = neEdgeNode.data()
re = rEdgeNode.data()

i1,i2 = np.searchsorted(time, (0.99, 1.05))

ne_red = ne[:,i1:i2]
r_red = r[:,i1:i2]
nee_red = nee[:,i1:i2]
re_red = re[:,i1:i2]

axes[2].scatter((np.ravel(r_red)-0.67)/0.22, np.ravel(ne_red), c='b')
axes[2].scatter((np.ravel(re_red)-0.67)/0.22, np.ravel(nee_red), c='b')


axes[0].axvline(1.0, ls='--')
axes[1].axvline(1.0, ls='--')
axes[2].axvline(1.0, ls='--')

axes[2].set_xlabel('r/a')