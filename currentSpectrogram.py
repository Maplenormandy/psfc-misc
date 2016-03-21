# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 18:15:43 2016

@author: normandy
"""

import numpy as np

import readline
import MDSplus

import matplotlib.pyplot as plt

from scipy import signal

readline

magTree = MDSplus.Tree('magnetics', 1150903021)
ipNode = magTree.getNode('\magnetics::ip')
f, t, Sxx = signal.spectrogram(ipNode.data(), fs)
plt.pcolormesh(t, f, np.log(Sxx))