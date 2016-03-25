# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 20:02:05 2016

@author: normandy
"""

import numpy as np

import readline
import MDSplus

import matplotlib.pyplot as plt

import shotAnalysisTools as sat

import pandas as pd

readline

shot = 1150903021
elecTree = MDSplus.Tree('electrons', shot)
teNode = elecTree.getNode('\gpc_t0')
    
te = teNode.data()
time = teNode.dim_of().data()
peaks = sat.findSawteeth(time, te, 0.57, 1.43)