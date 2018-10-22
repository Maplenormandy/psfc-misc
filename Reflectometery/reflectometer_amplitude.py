# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 16:02:10 2017

@author: normandy
"""

import readline
import MDSplus

readline

import numpy as np
import matplotlib.pyplot as plt
#import matplotlib as mpl

from scipy import signal
from scipy import stats
#from scipy import optimize

# %%

elecTree = MDSplus.Tree('electrons', 1160506007)

# 9 and 10 are normally the best
sig88ui = elecTree.getNode('\ELECTRONS::TOP.REFLECT.CPCI:DT132_1:INPUT_07').data()
sig88uq = elecTree.getNode('\ELECTRONS::TOP.REFLECT.CPCI:DT132_1:INPUT_08').data()
sig88ut = elecTree.getNode('\ELECTRONS::TOP.REFLECT.CPCI:DT132_1:INPUT_07').dim_of().data()

ci = np.mean(sig88ui)
cq = np.mean(sig88uq)

nl04Node = elecTree.getNode('\ELECTRONS::TOP.TCI.RESULTS:NL_04')

# %%
t1, t2 = np.searchsorted(sig88ut, (0.4,1.6))
#0.5944-0.5954
#0.9625-0.9650

# Get raw signals over proper time window
si = sig88ui[t1:t2]
sq = sig88uq[t1:t2]
st = sig88ut[t1:t2]

ci = np.median(si)
cq = np.median(sq)

sz = si-ci + 1j*(sq-cq)

# Filter signals
b, a = signal.butter(4, 1e-4, btype='highpass')
filt_si = signal.filtfilt(b, a, si-ci)
filt_sq = signal.filtfilt(b, a, sq-cq)


t3, t4 = np.searchsorted(st, (0.5,1.5))

hsi = filt_si[t3:t4]
hsq = filt_sq[t3:t4]
hst = st[t3:t4]
hsz = hsi+1j*hsq

# %%

#rt1, rt2, rt3, rt4 = np.searchsorted(hst, (0.5944, 0.5954, 0.9625, 0.9650))
#dc3 = np.abs(hsz)

rt1, rt2, rt3, rt4 = np.searchsorted(st, (0.5944, 0.5954, 0.9625, 0.9650))
#rt1, rt2, rt3, rt4 = np.searchsorted(st, (0.5849, 0.6049, 0.9537, 0.9737))
dc3 = np.abs(sz)

df1, db1 = np.histogram(dc3[rt1:rt2], bins=64, density=True)
df2, db2 = np.histogram(dc3[rt3:rt4], bins=64, density=True)

dbc1 = (db1[1:] + db1[:-1])/2
dbc2 = (db2[1:] + db2[:-1])/2

m1 = np.average(dbc1, weights=df1)
m2 = np.average(dbc2, weights=df2)

var1 = np.average((dbc1-m1)**2, weights=df1)
var2 = np.average((dbc2-m2)**2, weights=df2)

plt.figure()
plt.fill_between(dbc1, df1*1.0, 0, facecolor='red', alpha=0.5, label='LOC')
plt.fill_between(dbc2, df2*1.0, 0, facecolor='blue', alpha=0.5, label='SOC')

plt.xlabel('|z|')
plt.ylabel('probability of observing')
plt.legend()
plt.show()

print stats.kstat(dc3[rt1:rt2], n=4) / stats.kstat(dc3[rt1:rt2], n=2)**2
print stats.kstat(dc3[rt3:rt4], n=4) / stats.kstat(dc3[rt3:rt4], n=2)**2

"""
# %%

H, xedges, yedges = np.histogram2d(st, np.abs(sz), bins=(100,128))
H = H.T

plt.figure()
x,y = np.meshgrid(xedges, yedges)
plt.pcolormesh(x, y, np.clip(H, np.min(H), np.percentile(H, 99)), cmap='cubehelix')
plt.show()
"""
