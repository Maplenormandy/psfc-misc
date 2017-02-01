# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 15:54:28 2016

@author: normandy
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

#%%

N = 20000
L = 1
pth = 0.01
pmean = 1
m = 1

x0 = np.random.rand(N)*L
p0 = np.random.randn(N)*pth


nt = 1500
t = np.linspace(0, 200, nt)
dt = np.median(np.diff(t))

x = np.zeros((nt, N))
p = np.zeros((nt, N))

x[0,:] = x0
p[0,:] = p0

wallamp = 0.05
wallom = 1
wallx = np.ones(nt)*L + wallamp*signal.sawtooth(wallom*t, 0.5)
wallv = np.gradient(wallx)

for i in range(nt-1):
    x[i+1,:] = x[i,:] + p[i,:]/m*dt
    
    p[i+1,:] = np.where(x[i+1,:] > wallx[i+1], -p[i,:]+2*wallv[i+1], p[i,:])
    x[i+1,:] = np.where(x[i+1,:] > wallx[i+1], 2*wallx[i+1]-x[i+1,:], x[i+1,:])
    
    p[i+1,:] = np.where(x[i+1,:] < 0, -p[i+1,:], p[i+1,:])
    x[i+1,:] = np.where(x[i+1,:] < 0, -x[i+1,:], x[i+1,:])

"""
th0 = x0 * np.sign(p0) / L * np.pi
om0 = np.abs(p0) / L / m * 2 * np.pi



td = t[1:]

om = np.outer(td, om0)
th_c = om+th0[np.newaxis,:]
th = th_c % (2*np.pi)
th = np.where(th > np.pi, th-2*np.pi, th)

p = om * L * m / 2 / np.pi * np.sign(th)
x = np.abs(th) / np.pi * L
"""

bins = np.linspace(0, L*0.85, 50)
expectation = N / np.size(bins)
xbins = np.array([np.histogram(x[i,:], bins=bins, density=True)[0] for i in range(x.shape[0])])

plt.figure()
plt.pcolormesh(bins, t, xbins, cmap='cubehelix')