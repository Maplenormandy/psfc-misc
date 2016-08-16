# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 13:31:51 2016

@author: normandy
"""

import numpy as np

import matplotlib.pyplot as plt
import matplotlib

import readline
import MDSplus

import sys
sys.path.append('/home/normandy/bin')

import fftcece8 as fftcece

import eqtools

# %%

def plotBtHysteresis(shot):
    elecTree = MDSplus.Tree('electrons', shot)
    
    nl04Node = elecTree.getNode('\ELECTRONS::TOP.TCI.RESULTS:NL_04')
    magTree = MDSplus.Tree('magnetics', shot)
    specTree = MDSplus.Tree('spectroscopy', shot)
    btorNode = magTree.getNode(r'\magnetics::btor')
    velNode = specTree.getNode(r'\SPECTROSCOPY::TOP.HIREX_SR.ANALYSIS.W:VEL')
    
    rfTree = MDSplus.Tree('rf', shot)
    rfNode = rfTree.getNode(r'\rf::rf_power_net')
    
    vtime = velNode.dim_of().data()
    btime = btorNode.dim_of().data()
    nltime = nl04Node.dim_of().data()
    
    
    vlow = np.searchsorted(vtime, 0.7)
    vhigh = np.searchsorted(vtime, 1.5)+2
    
    vtime = vtime[vlow:vhigh]
    vdata = velNode.data()[0]
    vdata = vdata[vlow:vhigh]
    
    magData = np.interp(vtime, btime, btorNode.data())
    nlData = np.interp(vtime, nltime, nl04Node.data())
    rfData = np.interp(vtime, rfNode.dim_of().data(), rfNode.data())
    
    #plt.plot(magData*nldata/0.48, vdata, label=str(shot), marker='.')
    plt.plot(nlData, vdata, label=str(shot), marker='.')
    #plt.plot(rfData, vdata, label=str(shot))
    
    
def plotBrightness(shot):
    elecTree = MDSplus.Tree('electrons', shot)
    
    nl04Node = elecTree.getNode('\ELECTRONS::TOP.TCI.RESULTS:NL_04')
    magTree = MDSplus.Tree('magnetics', shot)
    specTree = MDSplus.Tree('spectroscopy', shot)
    emissNode = specTree.getNode(r'\SPECTROSCOPY::TOP.HIREX_SR.ANALYSIS.Z:INT')
    
    gpcNode = elecTree.getNode(r'\ELECTRONS::gpc2_te0')
    
    etime = emissNode.dim_of().data()
    
    elow = np.searchsorted(etime, 0.5)
    ehigh = np.searchsorted(etime, 1.5)+2
    
    etime = etime[elow:ehigh]
    edata = emissNode.data()[0]
    edata = edata[elow:ehigh]
    
    nldata = np.interp(etime, nl04Node.dim_of().data(), nl04Node.data())
    
    fig, ax1 = plt.subplots()
    
    ax1.plot(etime, np.log(edata) - np.log(nldata))
    
    ax2 = ax1.twinx()
    
    ax2.plot(gpcNode.dim_of().data(), gpcNode.data())
    

# %% 0.8 MA
plt.figure()
plotBtHysteresis(1160506007)
plotBtHysteresis(1160506008)
plotBtHysteresis(1160506024)
plotBtHysteresis(1160506025)
#plotBtHysteresis(1160506006)
#plotBtHysteresis(1160506013)

plt.legend()

# %% 1.1 MA
plt.figure()
#plotBtHysteresis(1160506009)
#plotBtHysteresis(1160506010)
plotBtHysteresis(1160506011)
plotBtHysteresis(1160506012)


plt.legend()
    
#plotBrightness(1150903021)

# %% Calc data

shot = 1160506007


e = eqtools.CModEFITTree(shot)
wmhd, taumhd, ping, wbdot, wpdot = e.getEnergy()
mhdt = e.getTimeBase()


def smoothedFunction(node, y=None):
    if y == None:
        t = node.dim_of().data()
        y = node.data()
    else:
        t = node
    
    def smoothed(t_eval):
        i0, i1 = np.searchsorted(t, [t_eval-0.005, t_eval+0.005])
        if i0 == i1:
            return y[i0]
        else:
            return np.median(y[i0:i1])
        #return y[i0]
        
        
    def smoothedArr(t_eval):
        return np.array([smoothed(x) for x in t_eval])
        
    return smoothedArr

elecTree = MDSplus.Tree('electrons', shot)

nl01Node = elecTree.getNode(r'\ELECTRONS::TOP.TCI.RESULTS:NL_01')
nl04Node = elecTree.getNode(r'\ELECTRONS::TOP.TCI.RESULTS:NL_04')
nl10Node = elecTree.getNode(r'\ELECTRONS::TOP.TCI.RESULTS:NL_10')
ne0Node = elecTree.getNode(r'\thom_midpln:ne_t')

ne0t = ne0Node.dim_of().data()
ne0 = ne0Node.data()
negood = (ne0t > 0.5) & (ne0t < 1.5) & (ne0 > 0.1)

nl04 = smoothedFunction(nl04Node)
nl01 = smoothedFunction(nl01Node)
nl10 = smoothedFunction(nl10Node)

specTree = MDSplus.Tree('spectroscopy', shot)
velNode = specTree.getNode(r'\SPECTROSCOPY::TOP.HIREX_SR.ANALYSIS.W:VEL')

vtime = velNode.dim_of().data()
vg = (vtime > 0.5) & (vtime < 1.5)
vdata = velNode.data()[0]

vel = smoothedFunction(vtime, vdata)

tg = ne0t[negood]
ne0g = ne0[negood]

tau = smoothedFunction(mhdt, taumhd)

vt = vtime[vg]
vy = vdata[vg]


cdata = fftcece.cecedata()
cdata.fftBin = 512
cdata.lagWindow = 0
cdata.lowpassf = 1.5e6
ttflower = 0
ttfupper = 200*1000
cdata.readMDS(shot,4)
cdata.lowpass()

hanningBase = np.r_[0:(cdata.samplingRate):(float(cdata.samplingRate)/
                                                         cdata.fftBin)]
                                                         
f0 = ttflower    #Hz
f1 = ttfupper    #Hz

#Convert frequencies to indices
f0index = int(f0*(float(cdata.fftBin)/cdata.samplingRate))
f1index = int(f1*(float(cdata.fftBin)/cdata.samplingRate))    

     
allcdata = np.zeros((len(vt), len(hanningBase), 2))
allcstat = np.zeros((len(vt), len(hanningBase), 2))
allcerr = np.zeros((len(vt), len(hanningBase), 2))

for i in range(len(vt)):
    t = vt[i]
    print "cece", t
    cdata.timeBegin = t-0.025
    cdata.timeEnd = t+0.025
    
    cdata.calcAutoOverlap()
    cdata.calcCrossOverlap()
    cdata.calcCoherence()
    cdata.calcCrossCorr()
    
    cdata.coherence[0,:] = 0
    allcdata[i,:,0] = cdata.coherence[:,0]
    allcdata[i,:,1] = cdata.coherence[:,5]
    
    allcerr[i,:,0] = cdata.coherVar[:,0]
    allcerr[i,:,1] = cdata.coherVar[:,5]
    
    allcstat[i,:,0] = cdata.statlimit[:,0]
    allcstat[i,:,1] = cdata.statlimit[:,5]

    


allcdata2 = allcdata[:,f0index:f1index,:]
allcstat2 = allcstat[:,f0index:f1index,:]

summed = np.trapz(allcdata2, axis=1)

#plt.figure()
#plt.scatter(vy, summed[:,0], cmap='BrBG', c=vt)

#plt.figure()
#plt.scatter(vy, summed[:,1], cmap='BrBG', c=vt)

# %% Plotting CECE

f2index = int(700*1000*(float(cdata.fftBin)/cdata.samplingRate))

X, Y = np.meshgrid(vt, hanningBase[f0index:f1index])
toPlot = allcdata[:,f0index:f1index,:] - allcstat[:,f0index:f1index,:]

plt.figure()
plt.pcolormesh(X, Y/1000, toPlot[:,:,1].T, cmap='cubehelix')
plt.xlabel('time [sec]')
plt.ylabel('freq [kHz]')

"""
plt.scatter(vel(tg), ne0g/nl04(tg)*0.6, cmap='RdGy', c=nl04(tg))
plt.xlabel('toroidal velocity [km/s]')
plt.ylabel('ne0 / (nl_04 / 0.6)')

clb = plt.colorbar()
clb.ax.set_ylabel('nl_04')

plt.title('1.1MA, 1160506009')
"""