# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 19:26:20 2020

@author: normandy
"""

import numpy as np

import readline
import MDSplus

import matplotlib.pyplot as plt

import eqtools

# %%

specTree = MDSplus.Tree('spectroscopy', 1120917011)
pos = specTree.getNode(r'\SPECTROSCOPY::TOP.HIREXSR.CALIB.MOD1:POS').data()
lam = specTree.getNode(r'\SPECTROSCOPY::TOP.HIREXSR.CALIB.MOD1:LAMBDA').data()

# %%

momNode = specTree.getNode(r'\SPECTROSCOPY::TOP.HIREXSR.ANALYSIS1.HLIKE.MOMENTS.LYA1:MOM')
psin_raw = momNode.dim_of(0).data()
time = momNode.dim_of(1).data()

t0 = time[13]
psin = psin_raw[13,:16]
e = eqtools.CModEFITTree(1120917011)

roa = e.psinorm2roa(psin, t0)