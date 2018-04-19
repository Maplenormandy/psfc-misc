# -*- coding: utf-8 -*-
"""
Created on Thu May  4 10:18:35 2017

@author: normandy
"""


import numpy as np
import matplotlib.pyplot as plt

import readline
import MDSplus

# %% open tree

ttree = MDSplus.Tree('transp', 89398)

veldNode = ttree.getNode('\TRANSP::TOP.OUTPUTS.TWO_D:VELD')

veld = veldNode.data()

vpolNode = ttree.getNode('\TRANSP::TOP.OUTPUTS.TWO_D:VPOLD_NC')
#vpolNode = ttree.getNode('\TRANSP::TOP.OUTPUTS.ONE_D:OMEGA')

# %%

vpol = vpolNode.data()

vpolR = vpolNode.dim_of(0).data()
vpolt = vpolNode.dim_of(1).data()

timeRow = np.ones(vpolR.shape[1])

tplot = np.outer(vpolt, timeRow)

plt.figure()
plt.pcolormesh(vpolR, vpolt, vpol)

plt.colorbar()