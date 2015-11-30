# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt
import readline
import MDSplus

shot = 1150901023
tree = MDSplus.Tree('electrons', shot)
proNode = tree.getNode('\ELECTRONS::TOP.YAG_NEW.RESULTS.PROFILES:NE_RZ')
rhoNode = tree.getNode('\ELECTRONS::TOP.YAG_NEW.RESULTS.PROFILES:R_MID_T')

rpro = proNode.data()
rrho = rhoNode.data()
rtime = rhoNode.dim_of().data()

goodTimes = np.logical_and(rtime < 0.65, rtime > 0.55)
timeC = rtime[goodTimes]
rhoC = rrho[:,goodTimes]
pro = rpro[:,goodTimes]

timeD = np.array([timeC] * 11)

plt.pcolormesh(timeD, rhoC, pro, cmap='cubehelix')
plt.show()

"""
tt, rr = np.meshgrid(time, rho)

ti = pro[3,:,:-1].transpose()
vtor = pro[1,:,:-1].transpose()

plt.figure()
plt.pcolormesh(tt, rr, ti, cmap='afmhot', vmin=0.3, vmax=1.8)
plt.colorbar()
plt.title('Ti, Shot ' + str(shot))

plt.figure()
plt.pcolormesh(tt, rr, vtor, cmap='BrBG', vmin=-10, vmax=10)
plt.colorbar()
plt.title('Vtor, Shot ' + str(shot))
plt.show()
"""
