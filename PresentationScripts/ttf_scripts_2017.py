# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 00:26:05 2016

@author: normandy
"""

import numpy as np

import readline
import MDSplus

import matplotlib.pyplot as plt
import matplotlib as mpl

readline

import eqtools

import scipy

font = {'family': 'normal', 'size': 18}
mpl.rc('font', **font)

# %%

def loadIonData(shot):
    e = eqtools.CModEFITTree(shot)
    specTree = MDSplus.Tree('spectroscopy', shot)
    heNode = specTree.getNode('\SPECTROSCOPY::TOP.HIREXSR.ANALYSIS1.HELIKE.PROFILES.Z')
    heproNode = heNode.getNode('PRO')
    herhoNode = heNode.getNode('RHO')
    heperrNode = heNode.getNode('PROERR')
    
    herpro = heproNode.data()
    herrho = herhoNode.data()
    herperr = heperrNode.data()
    hertime = herhoNode.dim_of()

    hegoodTimes = (hertime > 0).sum()
    
    hetime = hertime.data()[:hegoodTimes]
    herho = herrho[0,:] # Assume unchanging rho bins
    hepro = herpro[:,:hegoodTimes,:len(herho)]
    heperr = herperr[:,:hegoodTimes,:len(herho)]
    heroa = e.psinorm2roa(herho, 0.95)
    
    return hetime, heroa, hepro, heperr
    
t1, r1, p1, pe1 = loadIonData(1120106012)
t2, r2, p2, pe2 = loadIonData(1120106016)

# %% Plot raw inverted data

tind = 10

plt.figure()
plt.errorbar(r1, p1[1,tind,:], yerr=pe1[1,tind,:], c='b')
plt.errorbar(r2, p2[1,tind,:], yerr=pe2[1,tind,:], c='r')
plt.xlabel('r/a')

# %% Plot spline fitted data

tifit1 = scipy.io.readsav(r'/home/normandy/fits/tifit_1120106012_THT1.dat')
tifit2 = scipy.io.readsav(r'/home/normandy/fits/tifit_1120106016_THT1.dat')

# %%

tind = 9
data1 = tifit1['bsti'][tind][0][0][0]
data2 = tifit2['bsti'][tind][0][0][0]

f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
ax1.errorbar(data1[1], data1[0], yerr=data1[2], c='b')
ax2.errorbar(data1[1], -data1[3], yerr=data1[4], c='b')
ax2.set_xlabel('r/a')

ax1.errorbar(data2[1], data2[0], yerr=data2[2], c='r')
ax2.errorbar(data2[1], -data2[3], yerr=data2[4], c='r')