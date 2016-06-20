# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 17:15:04 2016

@author: normandy
"""

import numpy as np
from scipy.io import readsav
import matplotlib.pyplot as plt

import readline
import MDSplus

readline

import sys
sys.path.append('/home/normandy/git/psfc-misc')
import nonlocalAnalysis as nla


# load the various savefiles and combine them
load1 = readsav('/home/normandy/git/psfc-misc/PresentationScripts/1160506_zeff_neo_1.sav')

times = load1['time_array']
zeffs = load1['zeff_array']
zeffs_qfit = load1['zeff_qfit_array']
shots = load1['shot_array']

#plt.plot(times[6], zeffs_qfit[6])



print shots[6]
tcd = nla.ThomsonCoreData(None, 1160506007)
fed = nla.FrceceData(None, 31, 1160506007)

anaTree = MDSplus.Tree('efit01', 1160506007)
qNode = anaTree.getNode(r'\EFIT01::TOP.RESULTS.FITOUT:QPSI')

qdata = qNode.data()
qrdim = np.sqrt(qNode.dim_of(1).data())
qtdim = qNode.dim_of(0).data()

rmagxNode = anaTree.getNode(r'\EFIT01::EFIT_AEQDSK.RMAGX')
aoutNode = anaTree.getNode(r'\EFIT01::EFIT_AEQDSK.AOUT')

rData = rmagxNode.data()
aData = aoutNode.data()


def getCollisionality(t1, p=None):
    tcdInd1 = np.searchsorted(tcd.time, t1-0.01)
    
    rmid1 = tcd.rmid[:,tcdInd1]
    
    ne1 = tcd.dens[:,tcdInd1]
    
    sort1 = np.argsort(rmid1)
    
    rmid1 = rmid1[sort1]
    
    ne1 = ne1[sort1]
    
    fedInd1 = np.searchsorted(fed.time[0,:], [t1-0.01, t1+0.01])
    
    fedRmid1 = np.flipud(np.mean(fed.rmid[:,fedInd1[0]:fedInd1[1]], axis=1))
    
    #fedTemp1 = np.flipud(np.mean(fed.temp[:,fedInd1[0]:fedInd1[1]], axis=1))
    if p==None:
        fedTemp1 = np.flipud(np.mean(fed.temp[:,fedInd1[0]:fedInd1[1]], axis=1))
    else:
        fedTemp1 = np.flipud(np.percentile(fed.temp[:,fedInd1[0]:fedInd1[1]], p, axis=1))
    
    fedGood1 = fedTemp1>0
    fedTemp1 = fedTemp1[fedGood1]
    fedRmid1 = fedRmid1[fedGood1]
    
    te1 = np.interp(rmid1, fedRmid1, fedTemp1)
    
    zeff1 = np.interp(t1, times[6,:], zeffs_qfit[6,:])
    
    qInd1 = np.searchsorted(qtdim, [t1-0.01, t1+0.01])
    
    r1 = np.mean(rData[qInd1[0]:qInd1[1]]) / 100.0
    a1 = np.mean(aData[qInd1[0]:qInd1[1]]) / 100.0
    
    qrmid1 = (np.sqrt(qrdim)*a1 + r1)
    
    qd1 = np.mean(qdata[:,qInd1[0]:qInd1[1]], axis=1)
    
    q1 = np.interp(rmid1, qrmid1, qd1)
    
    return (rmid1-r1)/a1, q1, ne1, zeff1, te1, a1/r1

r11, q11, ne11, zeff11, te11, eps11 = getCollisionality(0.73, 1)
r12, q12, ne12, zeff12, te12, eps12 = getCollisionality(0.75, 1)
r13, q13, ne13, zeff13, te13, eps13 = getCollisionality(0.77, 1)

r21, q21, ne21, zeff21, te21, eps21 = getCollisionality(1.13, 99)
r22, q22, ne22, zeff22, te22, eps22 = getCollisionality(1.15, 99)
r23, q23, ne23, zeff23, te23, eps23 = getCollisionality(1.17, 99)


"""
plt.plot((rmid1-r1)/a1, q1*ne1*zeff1/te1**2/1e20)
plt.plot((rmid2-r2)/a2, q2*ne2*zeff2/te2**2/1e20)
"""

plt.semilogy(r11, 0.0118/(eps11**1.5)*q11*ne11*zeff11/(te11**2)/1e20, c='#ff0000', label='0.73s')
plt.semilogy(r12, 0.0118/(eps12**1.5)*q12*ne12*zeff12/(te12**2)/1e20, c='#aa0000', label='0.75s')
plt.semilogy(r13, 0.0118/(eps13**1.5)*q13*ne13*zeff13/(te13**2)/1e20, c='#550000', label='0.77s')

plt.semilogy(r21, 0.0118/(eps21**1.5)*q21*ne21*zeff21/(te21**2)/1e20, c='#00ff00', label='1.13s')
plt.semilogy(r22, 0.0118/(eps22**1.5)*q22*ne22*zeff22/(te22**2)/1e20, c='#00aa00', label='1.15s')
plt.semilogy(r23, 0.0118/(eps23**1.5)*q23*ne23*zeff23/(te23**2)/1e20, c='#005500', label='1.17s')

plt.legend(loc='upper left')

plt.xlabel('r/a')
plt.ylabel('0.0118 q ne zeff / te^2 / eps^1.5')
plt.title('1160506007 collisionality profile, sawtooth 1st/99th')


#for i in range(3,7):
    #print shots[7]
    #plt.plot(times[i,:], zeffs[i,:])
#plt.plot(times[7,:], zeffs_qfit[7,:])

