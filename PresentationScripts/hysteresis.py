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
    anaTree = MDSplus.Tree('analysis', shot)
    btorNode = magTree.getNode(r'\magnetics::btor')
    q95Node = anaTree.getNode(r'\ANALYSIS::EFIT_AEQDSK:QPSIB')
    velNode = specTree.getNode(r'\SPECTROSCOPY::TOP.HIREX_SR.ANALYSIS.W:VEL')
    
    rfTree = MDSplus.Tree('rf', shot)
    rfNode = rfTree.getNode(r'\rf::rf_power_net')
    
    vtime = velNode.dim_of().data()
    btime = btorNode.dim_of().data()
    nltime = nl04Node.dim_of().data()
    
    
    vlow = np.searchsorted(vtime, 0.4)
    vhigh = np.searchsorted(vtime, 1.4)+2
    
    vtime = vtime[vlow:vhigh]
    vdata = velNode.data()[0]
    vdata = vdata[vlow:vhigh]
    
    magData = np.interp(vtime, btime, btorNode.data())
    nlData = np.interp(vtime, nltime, nl04Node.data())
    q95data = np.interp(vtime, q95Node.dim_of().data(), q95Node.data())
    rfData = np.interp(vtime, rfNode.dim_of().data(), rfNode.data())
    
    plt.plot(nlData, vdata, label=str(shot), marker='.')
    #plt.plot(magData, vdata, label=str(shot), marker='.')
    #plt.plot(nlData*q95data, vdata, label=str(shot), marker='.')
    plt.xlabel('ne')
    plt.ylabel('vtor')
    #plt.plot(vtime, vdata, label=str(shot), marker='.')
    #plt.ylabel('A - vel [km/s]')
    #plt.xlabel('time [sec]')
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
plotBtHysteresis(1160506019)
#plotBtHysteresis(1160512025)
#plotBtHysteresis(1160506024)
#plotBtHysteresis(1160506025)
#plotBtHysteresis(1160506006)
#plotBtHysteresis(1160506013)

plt.legend(loc='lower right')

# %% 1.1 MA
plt.figure()
#plotBtHysteresis(1160506009)
#plotBtHysteresis(1160506010)
plotBtHysteresis(1160506011)
plotBtHysteresis(1160506012)


plt.legend()
    
#plotBrightness(1150903021)

# %% Calc data

shotlist = [1160506001, 1160506002, 1160506003, 1160506007, 1160506008, 1160506009, 1160506010, 1160506013, 1160506024, 1160506025]
shotdata = [None]*len(shotlist)

# 0.8 MA - 0, 3, 4, 7, 8, 9
# 1.1 MA - 1, 2, 5, 6

for j in range(len(shotlist)):
    shot = shotlist[j]
    """
    e = eqtools.CModEFITTree(shot)
    wmhd, taumhd, ping, wbdot, wpdot = e.getEnergy()
    mhdt = e.getTimeBase()
    """
    
    
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
                return np.mean(y[i0:i1])
            #return y[i0]
            
            
        def smoothedArr(t_eval):
            return np.array([smoothed(x) for x in t_eval])
            
        return smoothedArr
    
    elecTree = MDSplus.Tree('electrons', shot)
    
    nl04Node = elecTree.getNode(r'\ELECTRONS::TOP.TCI.RESULTS:NL_04')
    
    nl04 = smoothedFunction(nl04Node)
    
    specTree = MDSplus.Tree('spectroscopy', shot)
    velNode = specTree.getNode(r'\SPECTROSCOPY::TOP.HIREX_SR.ANALYSIS.W:VEL')
    
    vtime = velNode.dim_of().data()
    vg = (vtime > 0.5) & (vtime < 1.5)
    vdata = velNode.data()[0]
    
    vel = smoothedFunction(vtime, vdata)
    
    #tau = smoothedFunction(mhdt, taumhd)
    
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
    
    allttcross = np.zeros((len(vt), 2))
    allttcoh = np.zeros((len(vt), 2))
    alltttime = np.zeros((len(vt), 2))
    
    for i in range(len(vt)):
        t = vt[i]
        print "cece", t
        cdata.timeBegin = t-0.075
        cdata.timeEnd = t+0.075
        
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
    
        """
        T~/T Calculations, from Creely's code
        """
    
        #Calculate Te~ over Te
        Bif = 100000000.0      #Hz
        Bvid = cdata.lowpassf       #Hz
        time = cdata.analyzeTime      #s
    
        #normalization factor required for cross-power calculations
        normal = (((2*np.pi)/((cdata.fftBin**2)*cdata.numOverBlocks)))
    
        #Signal bandwidth
        Bsig = f1-f0
    
        #Sensitivity limit calculated from the various bandwidths
        senlimit = np.sqrt((1/np.sqrt(2*Bvid*time))*((2*Bsig)/Bif))
        
        #Calculate the Bias Subtractoin based on Bendat and Piersol page 333
        biasSub = np.sqrt((Bsig/Bif)*np.sum(cdata.statlimit[f0index:f1index,0])
                              /(f1index-f0index))
                              
        #Cross power calculation method
        allttcross[i,0] = np.sqrt(np.divide((Bsig*
                (np.sum(np.abs(
                        cdata.overcross[f0index:f1index,0]))) ),
                (Bif*np.sqrt(
                    (normal*np.sum(np.square(np.abs(cdata.overfft[:,
                                                    f0index:f1index,0]))))*
                    (normal*np.sum(np.square(np.abs(cdata.overfft[:,
                                                    f0index:f1index,1])))) )) ))

        #Coherence integration
        allttcoh[i,0] = np.sqrt((Bsig/Bif)*
                            np.sum(cdata.coherence[f0index:f1index,0])
                              /(f1index-f0index))

        #time lag cross correlation method
        alltttime[i,0] = np.sqrt(2*Bsig*np.mean(cdata.timelag
                [cdata.fftBin-5:cdata.fftBin+5,0])/Bif)
                
        # Repeat for other channels
        #Repeat for channels 15 and 16
        allttcross[i,1] = np.sqrt(np.divide((Bsig*
                (np.sum(np.abs(
                        cdata.overcross[f0index:f1index,5]))) ),
                (Bif*np.sqrt(
                    (normal*np.sum(np.square(np.abs(cdata.overfft[:,
                                                    f0index:f1index,2]))))*
                    (normal*np.sum(np.square(np.abs(cdata.overfft[:,
                                                    f0index:f1index,3])))) )) ))

        allttcoh[i,1] = np.sqrt((Bsig/Bif)*
                            np.sum(cdata.coherence[f0index:f1index,5])
                              /(f1index-f0index))

        alltttime[i,1] = np.sqrt(2*Bsig*np.mean(cdata.timelag
                [cdata.fftBin-5:cdata.fftBin+5,5])/Bif)
    
    
    allcdata2 = allcdata[:,f0index:f1index,:]
    allcstat2 = allcstat[:,f0index:f1index,:]
    
    summed = np.trapz(allcdata2, axis=1)
    sstat = np.trapz(allcstat2, axis=1)
    
    floored = np.trapz(np.clip(allcdata2-allcstat2, 0, 1000), axis=1)
    
    shotdata[j] = (allcdata, allcstat, floored, vt, vy, nl04, allttcross, allttcoh, alltttime)

#plt.figure()
#plt.scatter(nl04(vt), floored[:,0], cmap='BrBG', c=vy)

#plt.figure()
#plt.scatter(nl04(vt), summed[:,1], cmap='BrBG', c=vy)

# %% Storing CECE

ceceDataL07 = (allcdata, allcstat, summed, vt, vy, nl04)

# %% Plotting CECE

f2index = int(700*1000*(float(cdata.fftBin)/cdata.samplingRate))

print shotlist[4]
data = shotdata[4]

X, Y = np.meshgrid(vt, hanningBase[f0index:f1index])
toPlot = np.clip(data[0][:,f0index:f1index,:] - data[1][:,f0index:f1index,:], 0, 1000)

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

# %% Other cece plots

ceceData = [shotdata[6]]

#plt.figure()
for data in ceceData:
    
    #summed = np.trapz(data[0][:,f0index:f1index,:], axis=1)
    vmid = (np.max(data[4]) + np.min(data[4]))/2.0
    vp = data[4]>vmid
    vn = data[4]<vmid
    plt.scatter(data[5](vt[vp]), data[6][vp,0], marker='+', c='r', label='')
    plt.scatter(data[5](vt[vn]), data[6][vn,0], marker='.', c='b', label='')
    #plt.plot(vt, data[2][:,0], c='0.75')
    #plt.scatter(vt, data[2][:,0], cmap='BrBG', c=data[4], marker='+')