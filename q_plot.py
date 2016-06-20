# -*- coding: utf-8 -*-
"""
Created on Wed May  4 16:40:49 2016

@author: normandy
"""

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.widgets import Slider, Button, RadioButtons


import readline
import MDSplus


anaTree = MDSplus.Tree('efit01', 1160427006)
qNode = anaTree.getNode(r'\EFIT01::TOP.RESULTS.FITOUT:QPSI')
#anaTree = MDSplus.Tree('analysis', 1150903021)
#qNode = anaTree.getNode(r'\analysis::efit_fitout:QPSI')

class TimeSpacePlotter:
    def __init__(self, qNode):
        self.parseQNode(qNode)
        self.fig, self.ax = plt.subplots()
        
        self.fig2, self.sax = plt.subplots()
        
        self.dims = Slider(self.sax, "Time", 0.0, 1.0)
        
        self.pax, = self.ax.plot(self.rdim, self.data[:,50])
        
        self.dims.on_changed(self.updateSlider)
        
    def parseQNode(self, qNode):
        self.data = qNode.data()
        self.rdim = np.sqrt(qNode.dim_of(1).data())
        self.tdim = qNode.dim_of(0).data()
        
    def updateSlider(self, val):
        tind = int(val*len(self.tdim))
        print self.tdim[tind]
        self.pax.set_ydata(np.gradient(self.data[:,tind])*self.rdim)
        self.fig.canvas.draw_idle()
    
plt.close("all")
tsp = TimeSpacePlotter(qNode)