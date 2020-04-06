# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 15:59:30 2019

@author: normandy
"""

import numpy as np
import scipy.io
import matplotlib.pyplot as plt

# %%

data = scipy.io.netcdf.netcdf_file('/home/normandy/hysteresis_transp/12345A05/12345A05.CDF')

roa = data.variables['RMNMP'].data/21.878
time = data.variables['TIME'].data

tind = np.searchsorted(time, 1.25)

volts = data.variables['V'].data

eta_snc = data.variables['ETA_SNC'].data
eta_sps = data.variables['ETA_SPS'].data
ft = data.variables['NCFT'].data
ftl = data.variables['NCFTMINUS'].data
ftu = data.variables['NCFTPLUS'].data
zeffc = data.variables['ZEFFC'].data
zeffp = data.variables['ZEFFP'].data

nuste = data.variables['NUSTE'].data
nusti = data.variables['NUSTI'].data

cloge = data.variables['CLOGE'].data
clogi = data.variables['CLOGI'].data

te = data.variables['TE'].data
ti = data.variables['TI'].data
ne = data.variables['NE'].data

pcond = data.variables['PCOND'].data
pconv = data.variables['PCONV'].data

pcnde = data.variables['PCNDE'].data
pcnve = data.variables['PCNVE'].data

# %%

plt.figure()
#plt.plot(roa[tind,:], pcond[tind,:]+pconv[tind,:], c='b')
#plt.plot(roa[tind,:], pcnde[tind,:]+pcnve[tind,:], c='g')
plt.plot(roa[tind,:], ne[tind-1,:], c='b', ls='--')
plt.plot(roa[tind,:], ne[tind,:], c='b')
plt.plot(roa[tind,:], ne[tind+1,:], c='b', ls=':')

# %%

plt.figure()
plt.plot(roa[tind,:], ti[tind-1,:], c='b', ls='--')
plt.plot(roa[tind,:], ti[tind,:], c='b')
plt.plot(roa[tind,:], ti[tind+1,:], c='b', ls=':')
plt.plot(roa[tind,:], te[tind-1,:], c='g', ls='--')
plt.plot(roa[tind,:], te[tind,:], c='g')
plt.plot(roa[tind,:], te[tind+1,:], c='g', ls=':')

plt.show()
