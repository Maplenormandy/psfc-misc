# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 18:38:22 2016

@author: normandy
"""

import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import medfilt

from nonlocalAnalysis import ShotManager
import shotAnalysisTools as sat

def sawtoothAverage(data, time, teethTime):
    newTime = (teethTime[1:] + teethTime[:-1]) / 2.0
    newData = np.zeros((len(teethTime)-1))
    
    for i in range(len(teethTime)-1):
        tl, tr = timeSlice(time, teethTime[i], teethTime[i+1])
        newData[i] = np.median(data[tl:tr])
        
    print newTime.shape, newData.shape
    return newTime, newData

def medianValue(data, time, a, b):
    tl, tr = timeSlice(time, a, b)
    return np.median(data[tl:tr])

def timeSlice(time, a, b):
    return np.searchsorted(time, a), np.searchsorted(time, b, side='right')

def trophyPlot(sm1, sm2):
    f, axarr = plt.subplots(3,1, sharex=True)
    #f.subplots_adjust(hspace=0, wspace=0)
    
    
    
    a = 22
    
    eceTime1 = sm1.frceceData.time[14,:]
    minTime1, maxTime1 = timeSlice(eceTime1, 0.9, 1.5)
    
    eceTime1 = eceTime1[minTime1:maxTime1]
    
    eceTime2 = sm2.frceceData.time[15,:]
    minTime2, maxTime2 = timeSlice(eceTime2, 0.9, 1.5)
    
    eceTime2 = eceTime2[minTime2:maxTime2]
    
    rmids1 = sm1.frceceData.rmid[14,minTime1:maxTime1]
    rmids2 = sm2.frceceData.rmid[15,minTime2:maxTime2]
    
    magtime1 = sm1.rmagxNode.dim_of().data()
    lmt1, rmt1 = timeSlice(magtime1, 0.9, 1.5)
    rmagx1 = sm1.rmagxNode.data()[lmt1:rmt1]
    ra1 = (np.mean(rmids1)*100 - np.mean(rmagx1)) / a
    
    print "r/a ECE 1:", ra1
    
    rho1 = ra1*ra1
    
    print rho1
    
    magtime2 = sm2.rmagxNode.dim_of().data()
    lmt2, rmt2 = timeSlice(magtime2, 0.9, 1.5)
    rmagx2 = sm2.rmagxNode.data()[lmt2:rmt2]
    ra2 = (np.mean(rmids2)*100 - np.mean(rmagx2)) / a
    
    print "r/a ECE 2:", ra2
    
    rho2 = ra2*ra2
    
    print rho2
    
    til1, tir1 = timeSlice(sm1.thacoData.time, 0.9, 1.5)
    til2, tir2 = timeSlice(sm1.thacoData.time, 0.9, 1.5)
    
    ti1 = sm1.thacoData.pro[3,til1:tir1,4]
    ti2 = sm2.thacoData.pro[3,til2:tir2,7]
    
    vt1 = sm1.thacoData.pro[1,til1:tir1,4]
    vt2 = sm2.thacoData.pro[1,til2:tir2,7]
    
    print "r/a THACO 1:", np.sqrt(sm1.thacoData.rho[4])
    print "r/a THACO 2:", np.sqrt(sm2.thacoData.rho[7])
    
    
    
    te1 = sm1.frceceData.temp[14,minTime1:maxTime1]
    te2 = sm2.frceceData.temp[14,minTime2:maxTime2]
    
    
    del1, der1 = timeSlice(sm1.tciData.time[3,:], 0.9, 1.5)
    del2, der2 = timeSlice(sm2.tciData.time[3,:], 0.9, 1.5)
    

    
    axarr[0].plot(sm2.thacoData.time[til2:tir2], (ti2 - np.median(ti2)) / np.median(ti2), marker='.', label='"non-local"')
    axarr[0].plot(sm1.thacoData.time[til1:tir1], (ti1 - np.median(ti1)) / np.median(ti1), marker='.', label='"local"')
    
    axarr[0].axvline(1.0, c='r', ls='--')
    axarr[0].axvline(1.2, c='r', ls='--')
    axarr[0].axvline(1.4, c='r', ls='--')
    
    axarr[0].set_ylim(-0.1, 0.15)
    
    
    
    ne1 = np.median(sm1.tciData.dens[3,del1:der1])
    ne2 = np.median(sm2.tciData.dens[3,del2:der2])
    
    gpc1 = sm1.elecTree.getNode('\gpc_t0')
    gpc2 = sm2.elecTree.getNode('\gpc_t0')
    
    saw1 = sat.findSawteeth(gpc1.dim_of().data(), gpc1.data(), 0.9, 1.5)
    saw2 = sat.findSawteeth(gpc2.dim_of().data(), gpc2.data(), 0.9, 1.5)
    st1 = gpc1.dim_of().data()[saw1]
    st2 = gpc2.dim_of().data()[saw2]
    
    tet1, tes1 = sawtoothAverage(te1, eceTime1, st1)
    tet2, tes2 = sawtoothAverage(te2, eceTime2, st2)
    
    
    axarr[1].plot(tet2, (tes2 - np.median(tes2)) / np.median(tes2), marker='.', label='"non-local"')
    axarr[1].plot(tet1, (tes1 - np.median(tes1)) / np.median(tes1), marker='.', label='"local"')
    
    axarr[1].axvline(1.0, c='r', ls='--')
    axarr[1].axvline(1.2, c='r', ls='--')
    axarr[1].axvline(1.4, c='r', ls='--')
    
    axarr[1].set_ylim(-0.1, 0.15)
    
    axarr[2].plot(sm2.thacoData.time[til2:tir2], vt2, marker='.', label='"non-local"')
    axarr[2].plot(sm1.thacoData.time[til1:tir1], vt1, marker='.', label='"local"')
    
    axarr[2].axvline(1.0, c='r', ls='--')
    axarr[2].axvline(1.2, c='r', ls='--')
    axarr[2].axvline(1.4, c='r', ls='--')
    
    print "ne1", ne1
    print "ne2", ne2
    
    print "Ti1", np.median(ti1)
    print "Ti2", np.median(ti2)
    
    print "Te1", np.median(te1)
    print "Te2", np.median(te2)
    
    print "Ip1", medianValue(sm1.ipNode.data(), sm1.ipNode.dim_of().data(), 0.9, 1.5)
    print "Ip2", medianValue(sm2.ipNode.data(), sm2.ipNode.dim_of().data(), 0.9, 1.5)
    
    btorNode1 = sm1.magTree.getNode('\magnetics::btor')
    btorNode2 = sm2.magTree.getNode('\magnetics::btor')
    print "Bt1", medianValue(btorNode1.data(), btorNode1.dim_of().data(), 0.9, 1.5)
    print "Bt2", medianValue(btorNode2.data(), btorNode2.dim_of().data(), 0.9, 1.5)
    
    
    axarr[0].set_title('Ip = 0.8MA, Bt = 5.4T, r/a = 0.42', fontsize=14)
    axarr[0].set_ylabel('% Ti Change')
    axarr[1].set_ylabel('% Te Change (Sawtooth Median)')
    axarr[2].set_ylabel('$\omega_{torr}$')
    
    axarr[1].legend(loc='upper right', 
          ncol=3)
    
    axarr[2].set_xlabel('time [sec]')
    
    
    plt.tight_layout()
    #plt.subplots_adjust(top=0.90)
    
    #axarr[2].plot(sm1.tciData.time[3,:], ne1)
    #axarr[2].plot(sm2.tciData.time[3,:], ne2)

sm1 = ShotManager(1120106012, 0, 16)
sm2 = ShotManager(1120106020, 0, 16)

plt.close("all")
trophyPlot(sm1, sm2)