# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 13:52:05 2018

@author: normandy
"""

import matplotlib as mpl
import matplotlib.pyplot as plt

import numpy as np
#import scipy.interpolate

font = {'family' : 'serif',
        'serif': ['Computer Modern'],
        'size'   : 9}

mpl.rc('font', **font)

# %%

data1_soc = np.loadtxt('soc_mid_input_1.txt', skiprows=1)
data2_soc = np.loadtxt('soc_mid_input_2.txt', skiprows=1)

data1_loc = np.loadtxt('loc_mid_input_1.txt', skiprows=1)
data2_loc = np.loadtxt('loc_mid_input_2.txt', skiprows=1)

# %%

def calculateShear(data1, data2):
    roa = data1[1:-1,0]
    r = data1[:,1]
    q = data1[1:-1,3]
    omega0 = data1[:,4]
    te = data2[1:-1,1]

    domega0dr = (omega0[2:] - omega0[:-2]) / (r[2:] - r[:-2]) # 1/m/s

    mD = 1876544.16 # mass of Deuterium in keV
    c = 2.998e8
    cs = np.sqrt(te/mD) * c # sound speed in m/s

    gammae = -r[1:-1] * domega0dr / q
    a = 0.218

    gammae_norm = gammae * (a/cs)
    return roa, gammae_norm

roa_soc, gammae_soc = calculateShear(data1_soc, data2_soc)
roa_loc, gammae_loc = calculateShear(data1_loc, data2_loc)

rgam = np.array([ 0.35  , 0.425 , 0.5  ,  0.575 , 0.65 ,  0.725 , 0.8  ])
ggam = np.array([ 0.027435 , 0.076341,  0.12212 ,  0.16693 ,  0.20676 ,  0.22962 ,  0.52844  ])


plt.figure(figsize=(3.375*0.9, 3.375*0.75*0.8))
ax = plt.subplot(111)
plt.plot(roa_soc[:], -gammae_soc[:], c='b')
plt.plot(roa_loc[:], -gammae_loc[:], c='r')
plt.plot(rgam[:5], ggam[:5], marker='o', c='b')
plt.axhline(ls='--', c='k')
plt.ylabel(r'$[a/c_s]$')
plt.xlabel('r/a')
plt.xlim([0.25, 0.75])
#ax3i.set_xlim([0.25, 0.75])
plt.ylim([-.05, .19])
plt.tight_layout()

plt.text(0.2, 0.65, '$\gamma_{\mathrm{max}}$', transform=ax.transAxes)
plt.text(0.5, 0.26, '$\gamma_{E}$', transform=ax.transAxes)


plt.savefig('shear_locsoc.pdf', format='pdf', dpi=1200, facecolor='white')
