# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 14:44:53 2019

@author: normandy
"""

import matplotlib.pyplot as plt
import numpy as np

import MDSplus

loc_specTree = MDSplus.Tree('spectroscopy', 1120106016)
soc_specTree = MDSplus.Tree('spectroscopy', 1120106012)

lvnode = loc_specTree.getNode(r'\SPECTROSCOPY::TOP.HIREX_SR.ANALYSIS.Z:VEL')
svnode = soc_specTree.getNode(r'\SPECTROSCOPY::TOP.HIREX_SR.ANALYSIS.Z:VEL')

def trimData(time, data, t0=0.5, t1=1.5):
    i0, i1 = np.searchsorted(time, (t0, t1))
    return time[i0:i1+1], data[i0:i1+1]

lvtime, lvdata = trimData(lvnode.dim_of().data(), lvnode.data()[0], 0.35, 0.75)
#svtime, svdata = trimData(svnode.dim_of().data(), svnode.data()[0], 0.35, 0.75)
plt.plot(lvtime, lvdata, c='r', label='LOC', marker='.')
plt.axhline(ls='--', c='k')
#plt.plot(svtime, svdata, c='b', label='SOC', marker='.')
#plt.legend(loc='lower right')

plt.xlabel('Time [sec]')
plt.ylabel('Line-Average Toroidal Velocity [km/s]')
plt.show()
