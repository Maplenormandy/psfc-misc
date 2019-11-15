# -*- coding: utf-8 -*-
"""
Created on Thu May  4 10:18:35 2017

@author: normandy
"""


import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from matplotlib import cm

# %% open tree

data = scipy.io.netcdf.netcdf_file('/home/pablorf/Cao_transp/12345A05/12345A05.CDF')

roa = data.variables['RMNMP'].data/21.878
plflx = data.variables['PLFLX'].data
time = data.variables['TIME'].data
#omega = data.variables['VRPOT'].data
omega = data.variables['OMEGA_NC'].data
q = data.variables['Q'].data
xb = data.variables['XB'].data
shear = data.variables['SREXB_NCL'].data

te = data.variables['TE'].data
ti = data.variables['TI'].data
ne = data.variables['NE'].data

def psigrad(z,x,n=1):
    if n > 1:
        return psigrad(psigrad(z,x,n-1), x, 1)
    elif n == 1:
        return np.gradient(z) / np.gradient(x)

t0, t1 = np.searchsorted(time, (1.25, 1.43)) #A05
#t0, t1 = np.searchsorted(time, (0.95, 1.47)) #B05

# %%

om_min = np.min(omega[:,0])
om_max = np.max(omega[:,0])

colors = cm.viridis((omega[:,0]-om_min)/(om_max-om_min))

"""
plt.figure()
for i in range(len(time)):
    cs = np.sqrt(te[i,:]) * 6920.56111 # c_s in m/s
    csoa = cs / 0.21878
    #plt.plot(roa[i,:], psigrad(omega[i,:], roa[i,:])*roa[i,:]/q[i,:] / csoa, c=colors[i-t0])
    plt.plot(roa[i,:], shear[i,:], c=colors[i])

plt.plot(time, omega[:,0])
"""

plt.plot(roa[t0-1,:], shear[t0-1,:], c='r')
plt.plot(roa[t1-1,:], shear[t1-1,:], c='b')