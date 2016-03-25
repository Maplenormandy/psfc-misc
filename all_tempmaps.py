# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 20:06:23 2016

@author: normandy
"""

import numpy as np
import matplotlib.pyplot as plt

from matplotlib import gridspec
#import matplotlib.colors as colors
    
import readline
import MDSplus

#import shotAnalysisTools as sat

readline

class ThacoData:
    def __init__(self, node):
        
        proNode = node.getNode('PRO')
        rhoNode = node.getNode('RHO')
        perrNode = node.getNode('PROERR')

        rpro = proNode.data()
        rrho = rhoNode.data()
        rperr = perrNode.data()
        rtime = rhoNode.dim_of()

        goodTimes = (rtime > 0).sum()

        self.time = rtime.data()[:goodTimes]
        self.rho = rrho[0,:] # Assume unchanging rho bins
        self.pro = rpro[:,:goodTimes,:len(self.rho)]
        self.perr = rperr[:,:goodTimes,:len(self.rho)]
        
        

with open('goodshots.txt', 'r') as f:
    shotList = [int(s) for s in f]

plt.ioff()

for shot in shotList:
    print shot
    specTree = MDSplus.Tree('spectroscopy', shot)

    for tht in ['', '1', '2']:
        try:
            anaNode = specTree.getNode('\SPECTROSCOPY::TOP.HIREXSR.ANALYSIS'+tht)
        except:
            continue
        
        try:
            heNode = anaNode.getNode('HELIKE.PROFILES.Z')
            td = ThacoData(heNode)
            
            rplot, tplot = np.meshgrid(td.rho, td.time)
            tiplot = td.pro[3,:,:-1]
            vtorplot = td.pro[1,:,:-1]
            
            fig = plt.figure(figsize=(22,6))
            gs = gridspec.GridSpec(2,1)
            ax1 = fig.add_subplot(gs[0])
            cax1 = ax1.pcolormesh(tplot, rplot, tiplot, cmap='cubehelix', vmin=0.3, vmax=1.9)
            fig.colorbar(cax1)
    
            fig.suptitle('Shot ' + str(shot) + ' Ion Temp, Toroidal Velocity')
    
            ax2 = fig.add_subplot(gs[1], sharex=ax1, sharey=ax1)
            cax2 = ax2.pcolormesh(tplot, rplot, vtorplot, cmap='BrBG', vmin=-20, vmax=20)
            fig.colorbar(cax2)
            
            plt.tight_layout()
            plt.savefig('AllMapsOutput/' + str(shot) + '_he' + tht)
        except:
            pass
        
        try:
            hyNode = anaNode.getNode('HLIKE.PROFILES.LYA1')
            
            td = ThacoData(hyNode)
            
            rplot, tplot = np.meshgrid(td.rho, td.time)
            tiplot = td.pro[3,:,:-1]
            vtorplot = td.pro[1,:,:-1]
            
            fig = plt.figure(figsize=(22,6))
            gs = gridspec.GridSpec(2,1)
            ax1 = fig.add_subplot(gs[0])
            cax1 = ax1.pcolormesh(tplot, rplot, tiplot, cmap='cubehelix', vmin=0.3, vmax=1.9)
            fig.colorbar(cax1)
    
            fig.suptitle('Shot ' + str(shot) + ' Ion Temp, Toroidal Velocity')
    
            ax2 = fig.add_subplot(gs[1], sharex=ax1, sharey=ax1)
            cax2 = ax2.pcolormesh(tplot, rplot, vtorplot, cmap='BrBG', vmin=-20, vmax=20)
            fig.colorbar(cax2)
            
            plt.tight_layout()
            plt.savefig('AllMapsOutput/' + str(shot) + '_hy' + tht)
        except:
            pass