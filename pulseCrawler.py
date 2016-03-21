# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 19:02:17 2016

@author: normandy
"""

import numpy as np

import readline
import MDSplus

import matplotlib.pyplot as plt

import shotAnalysisTools as sat

shotList = range(1120605000, 1120605040)

for shot in shotList:
    try:
        pulses = sat.findColdPulses(shot)
    except:
        continue
    
    if len(pulses) > 0:
        pass
    
    elecTree = MDSplus.Tree('electrons', shot)
    specTree = MDSplus.Tree('spectroscopy', shot)
    magTree = MDSplus.Tree('magnetics', shot)
    anaTree = MDSplus.Tree('analysis', shot)
    rfTree = MDSplus.Tree('rf', shot)
    
    tciNode = elecTree.getNode('\ELECTRONS::TOP.TCI.RESULTS')
    densNode = tciNode.getNode('nl_04')