# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt
import readline
import MDSplus

readline

plt.close("all")

class ThacoData:
    def __init__(self, thtNode, shot=None, tht=None):
        if (shot != None):
            self.shot = shot
            self.specTree = MDSplus.Tree('spectroscopy', shot)

            if (tht == 0):
                self.tht = ''
            else:
                self.tht = str(tht)

            self.thtNode = self.specTree.getNode('\SPECTROSCOPY::TOP.HIREXSR.ANALYSIS' + self.tht + '.HELIKE.PROFILES.Z')
        else:
            self.thtNode = thtNode

        proNode = self.thtNode.getNode('PRO')
        perrNode = self.thtNode.getNode('PROERR')
        rhoNode = self.thtNode.getNode('RHO')

        rpro = proNode.data()
        rperr = perrNode.data()
        rrho = rhoNode.data()
        rtime = rhoNode.dim_of()

        goodTimes = (rtime > 0).sum()

        self.time = rtime.data()[:goodTimes]
        self.rho = rrho[0,:] # Assume unchanging rho bins
        self.pro = rpro[:,:goodTimes,:len(self.rho)]
        self.perr = rperr[:,:goodTimes,:len(self.rho)]


class HirexsrSpecData:
    def __init__(self, thtNode, shot=None, tht=None):
        if (shot != None):
            self.shot = shot
            self.specTree = MDSplus.Tree('spectroscopy', shot)

            if (tht == 0):
                self.tht = ''
            else:
                self.tht = str(tht)

            self.thtNode = self.specTree.getNode('\SPECTROSCOPY::TOP.HIREXSR.ANALYSIS' + self.tht + '.HELIKE.SPEC')
        else:
            self.thtNode = thtNode

        specNode = self.thtNode.getNode('SPECBR')
        rbr = specNode.data()
        rtime = specNode.dim_of(1).data()
        
        goodChans = (rbr[0,0,:] > 0).sum()
        goodTimes = (rtime > 0).sum()
        
        self.time = rtime[:goodTimes]
        self.br = rbr[:,:goodTimes,:goodChans]
        
    def backgroundLevel(self):
        return np.percentile(self.br, 10, axis=0)
        

class TciData:
    def __init__(self, tciNode):
        self.tciNode = tciNode

        self.rmid  = self.tciNode.getNode('rad').data()
        self.time = [None]*10
        self.dens = [None]*10

        for i in range(1,11):
            dnode = self.tciNode.getNode('nl_%02d' % i)
            self.time[i-1] = dnode.dim_of().data()
            self.dens[i-1] = dnode.data()

        self.time = np.array(self.time)
        self.dens = np.array(self.dens)


shotList = [
        1150901005,
        1150901006,
        1150901007,
        1150901008,
        1150901009,
        1150901010,
        1150901011,
        1150901013,
        1150901014,
        1150901015,
        1150901016,
        1150901017,
        1150901018,
        1150901020,
        1150901021,
        1150901022,
        1150901023,
        1150901024,
        1150903019,
        1150903021,
        1150903022,
        1150903023,
        1150903024,
        1150903025,
        1150903026,
        1150903028,
        1120216006,
        1120216007,
        1120216008,
        1120216009,
        1120216010,
        1120216011,
        1120216012,
        1120216013,
        1120216014,
        1120216017,
        1120216020,
        1120216021,
        1120216023,
        1120216025,
        1120216026,
        1120216028,
        1120216030,
        1120216031,
        1120106010,
        1120106011,
        1120106012,
        1120106015,
        1120106016,
        1120106017,
        1120106020,
        1120106021,
        1120106022,
        1120106025,
        1120106026,
        1120106027,
        1120106028,
        1120106030,
        1120106031,
        1120106032
        
        
        ]
        
shotDict = {}

#f0 = plt.figure()
#ax0 = f0.add_axes([0.1, 0.1, 0.8, 0.8])

"""
f1 = plt.figure()
f2 = plt.figure()
f3 = plt.figure()
f4 = plt.figure()

ax1 = f1.add_axes([0.1, 0.1, 0.8, 0.8])
ax2 = f2.add_axes([0.1, 0.1, 0.8, 0.8])
ax3 = f3.add_axes([0.1, 0.1, 0.8, 0.8])
ax4 = f4.add_axes([0.1, 0.1, 0.8, 0.8])
"""

f5 = plt.figure()
ax5 = f5.add_axes([0.1, 0.1, 0.8, 0.8])

#f6 = plt.figure()
#ax6 = f6.add_axes([0.1, 0.1, 0.8, 0.8])

labela = '1.2MW'
labelb = '0.6MW'
labelc = '0.0MW'

rho1 = []
rho2 = []
rho3 = []
rho4 = []


for shot in shotList:
    if shot < 1150000000:
        continue
    
    #print shot
    try:
        td = ThacoData(None, shot, 1)
        sd = HirexsrSpecData(None, shot, 1)
    except:
        try:
            td = ThacoData(None, shot, 0)
            sd = HirexsrSpecData(None, shot, 0)
        except:
            continue
    rfTree = MDSplus.Tree('rf', shot)
    rfNode = rfTree.getNode('\\rf::rf_power_net')
    
    elecTree = MDSplus.Tree('electrons', shot)
    tciNode = elecTree.getNode('\ELECTRONS::TOP.TCI.RESULTS')
    
    magTree = MDSplus.Tree('magnetics', shot)
    ipNode = magTree.getNode('\magnetics::ip')
    
    tci = TciData(tciNode)

    rfTime = rfNode.dim_of().data()
    inds = np.all([rfTime > td.time[0], rfTime < td.time[-1]], axis=0)
    rfGood = rfNode.data()[inds]
    
    rfMed = np.array([np.median(rfGood)] * len(td.rho))
    
    
    temps = td.pro[3,:,:]
    tempMean = np.mean(temps, axis=0)
    tempStd = np.std(temps, axis=0)
    percentDev = tempStd / tempMean
    
    
    rots = td.pro[1,:,:]
    rotsLaplacian = np.sqrt(np.sum(np.diff(rots, n=2, axis=0) ** 2, axis=0))
    
    try:
        rotMeans = np.average(rots, axis=1, weights=td.pro[0,:,:])
    except:
        continue
    
    rotsDev = (rots - np.array([rotMeans] * len(td.rho)).T) / td.perr[1,:,:]
    
    label = ''
    
    if rfMed[0] > 0.9:
        mark = '*'
        col = 'r'
        label=labela
        labela = ''
    elif rfMed[0] > 0.3:
        mark = '^'
        col = 'g'
        label=labelb
        labelb = ''
    else:
        mark = 'o'
        col = 'b'
        label=labelc
        labelc = ''
    
    
    
    #if tempMean[0] > 0 and tempMean[12] > 0:
    if True:
        #ax0.scatter(tempMean, percentDev, c=col, marker=mark, label=label)
        
        
        """
        idx1 = 0
        rho1.append(td.rho[idx1])
        idx2 = (np.abs(td.rho-0.03)).argmin()
        rho2.append(td.rho[idx2])
        idx3 = (np.abs(td.rho-0.12)).argmin()
        rho3.append(td.rho[idx3])
        idx4 = (np.abs(td.rho-0.5)).argmin()
        rho4.append(td.rho[idx4])
        ax1.scatter(tempMean[idx1], percentDev[idx1], c=col, marker=mark, label=label)
        ax2.scatter(tempMean[idx2], percentDev[idx2], c=col, marker=mark, label=label)
        ax3.scatter(tempMean[idx3], percentDev[idx3], c=col, marker=mark, label=label)
        ax4.scatter(tempMean[idx4], percentDev[idx4], c=col, marker=mark, label=label)
        """

        #ax5.scatter(td.rho, rotsLaplacian, c=col, marker=mark, label=label)
        #ax5.scatter(np.array([td.rho] * rots.shape[0]).flatten(), (rotsDev).flatten(), c=col, marker=mark, label=label)
        
        #ax6.scatter(rotMeans, rots[:,0], c=col, marker=mark, label=label)

"""
ax0.set_title('% RMS Ti Deviation from mean, whole profile')
ax0.set_ylabel('RMS(Ti - Mean[Ti]) / Mean[Ti]')
ax0.set_xlabel('Mean[Ti] [keV]')
ax0.legend(loc='upper left')
"""

"""
ax1.set_title('% RMS Ti Deviation from mean, r/a=' + str(np.mean(rho1)))
ax2.set_title('% RMS Ti Deviation from mean, r/a=' + str(np.mean(rho2)))
ax3.set_title('% RMS Ti Deviation from mean, r/a=' + str(np.mean(rho3)))
ax4.set_title('% RMS Ti Deviation from mean, r/a=' + str(np.mean(rho4)))

ax1.set_xlabel('Mean[Ti] [keV]')
ax2.set_xlabel('Mean[Ti] [keV]')
ax3.set_xlabel('Mean[Ti] [keV]')
ax4.set_xlabel('Mean[Ti] [keV]')

#ax1.set_ylim([0.02, 0.21])
#ax2.set_ylim([0.02, 0.21])

ax1.set_ylabel('RMS(Ti - Mean[Ti]) / Mean[Ti]')
ax2.set_ylabel('RMS(Ti - Mean[Ti]) / Mean[Ti]')
ax3.set_ylabel('RMS(Ti - Mean[Ti]) / Mean[Ti]')
ax4.set_ylabel('RMS(Ti - Mean[Ti]) / Mean[Ti]')

ax1.legend(loc='upper left')
ax2.legend(loc='upper left')
ax3.legend(loc='upper left')
ax4.legend(loc='upper left')
"""

"""
ax5.set_title('RMS Vtor second derivative')
ax5.set_ylabel('RMS(d^2/dx^2[Vtor]) [kHz]')
ax5.set_xlabel('r/a')
ax5.legend(loc='upper right')
"""

ax5.set_title('Vtor fluctuation contribution over entire profile')
ax5.set_ylabel('Normalized Vtor - Emissivity-averaged Vtor [kHz]')
ax5.set_xlabel('r/a')
ax5.legend(loc='upper right')

"""
ax6.plot([-5, 15], [-5, 15], 'r-')
ax6.set_title('Noise in Vtor')
ax6.set_ylabel('Core Vtor [kHz]')
ax6.set_xlabel('Emissivity-Averaged Vtor [kHz]')
ax6.legend(loc='upper left')
"""

plt.show()