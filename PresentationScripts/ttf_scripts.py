# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 00:26:05 2016

@author: normandy
"""

import numpy as np

import readline
import MDSplus

import sys

import matplotlib.pyplot as plt

readline

class FrceceData:
    def __init__(self, elecTree, numChannels):
        self.time = [None]*numChannels
        self.temp = [None]*numChannels
        self.rmid = [None]*numChannels

        print "Loading ECE channels:",

        for i in range(1, numChannels+1):
            tempNode = elecTree.getNode('\ELECTRONS::TE_HRECE%02d' % i)
            rmidNode = elecTree.getNode('\ELECTRONS::RMID_HRECE%02d' % i)

            rtimes = rmidNode.dim_of().data()
            ttimes = tempNode.dim_of().data()

            self.time[i-1] = ttimes
            self.temp[i-1] = tempNode.data()
            self.rmid[i-1] = np.interp(self.time[i-1], rtimes, rmidNode.data().flatten())

            print i,
            sys.stdout.flush()

        print "done"

        self.time = np.array(self.time)
        self.temp = np.array(self.temp)
        self.rmid = np.array(self.rmid)
        
"""
elecTree = MDSplus.Tree('electrons', 1120106020)
#frc = FrceceData(elecTree, 16)

plt.figure()
plt.plot(frc.time[0,:], frc.temp[0,:], c='b')
plt.plot(frc.time[3,:], frc.temp[3,:], c='b')
plt.plot(frc.time[6,:], frc.temp[6,:], c='b')
plt.plot(frc.time[9,:], frc.temp[9,:], c='b')
plt.plot(frc.time[12,:], frc.temp[12,:], c='b')
plt.plot(frc.time[15,:], frc.temp[15,:], c='b')

plt.ylabel('Te [keV]')
plt.xlabel('time [sec]')
"""

