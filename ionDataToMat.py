# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt
import readline
import MDSplus
from matplotlib.widgets import Slider

from scipy.interpolate import bisplrep, bisplev, splprep, splev
import scipy.io as sio

readline

class ThacoData:
    def __init__(self, thtNode, shot=None, tht=None):
        if (shot != None):
            self.shot = shot
            self.specTree = MDSplus.Tree('spectroscopy', shot)

            if (tht == 0):
                self.tht = ''
            else:
                self.tht = str(tht)

            self.thtNode = self.specTree.getNode('\SPECTROSCOPY::TOP.HIREXSR.ANALYSIS' + self.tht + '.HELIKE.PROFILES.Z')
        else:
            self.thtNode = thtNode

        proNode = self.thtNode.getNode('PRO')
        perrNode = self.thtNode.getNode('PROERR')
        rhoNode = self.thtNode.getNode('RHO')

        rpro = proNode.data()
        rperr = perrNode.data()
        rrho = rhoNode.data()
        rtime = rhoNode.dim_of()

        goodTimes = (rtime > 0).sum()

        self.time = rtime.data()[:goodTimes]
        self.rho = rrho[0,:] # Assume unchanging rho bins
        self.pro = rpro[:,:goodTimes,:len(self.rho)]
        self.perr = rperr[:,:goodTimes,:len(self.rho)]

#td = ThacoData(None, 1150901020, 1)

shotList = [
        1150901005,
        1150901006,
        1150901007,
        1150901008,
        1150901009,
        1150901010,
        1150901011,
        1150901013,
        1150901014,
        1150901015,
        1150901016,
        1150901017,
        1150901018,
        1150901020,
        1150901021,
        1150901022,
        1150901023,
        1150901024,
        1150903019,
        1150903021,
        1150903022,
        1150903023,
        1150903024,
        1150903025,
        1150903026,
        1150903028
        ]
shotDict = {}

for shot in shotList:
    print shot
    td = ThacoData(None, shot, 1)
    shotDict['shot' + str(shot)] = {
            'time': td.time,
            'rho': td.rho,
            'pro': td.pro,
            'perr': td.perr
            }
    print td.pro.shape, td.time.shape, td.rho.shape

sio.savemat('mp793shots.mat', shotDict)

"""
# Get B-Spline fit
tgrid, rgrid = np.meshgrid(td.time, td.rho)
btck = bisplrep(tgrid.flatten(), rgrid.flatten(), td.pro[3,:,:].T.flatten(), \
                             w = 1.0 / td.perr[3,:,:].flatten(), s=len(tgrid.flatten())**1.3, \
                             kx = 3, ky = 3)

stck, u = splprep([td.pro[3,5,:]], w = 1.0 / td.perr[3,5,:], \
                      u = td.rho, k = 3, s = len(td.rho)-np.sqrt(2*len(td.rho)))

rplot = np.linspace(0.0,1.0)

timeVar = 15

tb = bisplev(td.time[timeVar], rplot, btck)
ts = splev(rplot, stck)

plt.figure()
line, (bottoms, tops), verts = plt.errorbar(td.rho, td.pro[3,timeVar,:], yerr=td.perr[3,timeVar,:], fmt='.')
axb = plt.plot(rplot, tb)
axs = plt.plot(rplot, ts[0])

axtime = plt.axes([0.25, 0.01, 0.65, 0.03])
stime = Slider(axtime, 'Time', td.time.min(), td.time.max(), valinit = td.time[timeVar])

def update(val):
    global timeVar

    idx = np.abs(td.time - val).argmin()
    if idx != timeVar:
        timeVar = idx

    pass

plt.xlabel('r/a')
plt.ylabel('T [keV]')
plt.show()

print "Knot Vector:", stck[0]
print "B-Spline Coefs:", stck[1]
print "Degree:", stck[2]
"""
