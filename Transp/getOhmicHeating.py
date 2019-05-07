# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 16:19:44 2017

@author: normandy
"""

import numpy as np
import matplotlib.pyplot as plt

import readline
import MDSplus

import eqtools

# %% open tree

loc=False

if loc:
    transpShot = 89670
    t0 = 0.96
else:
    transpShot = 89398
    t0 = 0.6

ttree = MDSplus.Tree('transp', transpShot)


# %% equilibrium, gyro-bohm normalizations

e = eqtools.CModEFITTree(1160506007)


surfNode = ttree.getNode('\TRANSP::TOP.OUTPUTS.TWO_D:SURF')

pcond = ttree.getNode('\TRANSP::TOP.OUTPUTS.TWO_D:PCOND').data()
pconv = ttree.getNode('\TRANSP::TOP.OUTPUTS.TWO_D:PCONV').data()

pcnde = ttree.getNode('\TRANSP::TOP.OUTPUTS.TWO_D:PCNDE').data()
pcnve = ttree.getNode('\TRANSP::TOP.OUTPUTS.TWO_D:PCNVE').data()

qie = ttree.getNode('\TRANSP::TOP.OUTPUTS.TWO_D:QIE').data()

#divfdNode = ttree.getNode('\TRANSP::TOP.OUTPUTS.TWO_D:DIVFD')
#divfeNode = ttree.getNode('\TRANSP::TOP.OUTPUTS.TWO_D:DIVFE')

pohNode = ttree.getNode('\TRANSP::TOP.OUTPUTS.TWO_D:POH')
rad = pohNode.dim_of(0).data()
time = pohNode.dim_of(1).data()

tind = np.searchsorted(time, t0)
poh = pohNode.data()
surf = surfNode.data()

# %%
roa= e.rho2rho('sqrtphinorm', 'r/a', rad[tind], time[tind])
plt.plot(roa, (pcond[tind,:]+pconv[tind,:])*surf[tind,:])
plt.plot(roa, (qie[tind,:])*surf[tind,:])
#plt.plot(roa, (poh[tind,:])*surf[tind,:])
