# -*- coding: utf-8 -*-
"""
Created on Tue May 31 10:50:22 2016

@author: normandy
"""

import numpy as np
from scipy.integrate import ode
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def f(t, y):
    return [10*(y[1]-y[0]), y[0]*(28.0-y[2])-y[1], y[0]*y[1]-8.0*y[2]/3.0]
    
def jac(t, y):
    return [[-10, 10, 0], [(28-y[2]), -1, -y[0]], [y[1], y[0], -8.0/3.0]]
    
    


def getSol(y0, maxInd=5000, maxTime=50.0):
    r = ode(f, jac).set_integrator('vode', method='adams')
    r.set_initial_value(y0, 0)
    
    sol = np.zeros((3, maxInd))
    tsol = np.zeros(maxInd)
    ind = 1
    
    
    sol[:,0] = y0
    tsol[0] = 0
    
    while r.successful and ind < maxInd:
        sol[:, ind] = r.integrate(r.t+maxTime/maxInd)
        tsol[ind] = r.t
        ind += 1        
        
    return tsol, sol


y0s = np.random.rand(3, 1000)

sols = [None] * 1000
tsol = []

for i in range(1000):
    tsol, sols[i] = getSol(y0s[:, i])
    print i

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(sols[5][0,:], sols[5][1,:], sols[5][2,:])
ax.plot(sols[6][0,:], sols[6][1,:], sols[6][2,:])

