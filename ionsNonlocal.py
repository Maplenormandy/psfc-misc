# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt
import readline
import MDSplus

import shotAnalysisTools as sat

import sys

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
        

class FrceceData:
    def __init__(self, elecTree, numChannels, startChannel=1):
        self.time = [None]*numChannels
        self.temp = [None]*numChannels
        self.rmid = [None]*numChannels

        print "Loading ECE channels:",

        for i in range(startChannel, numChannels+startChannel):
            tempNode = elecTree.getNode('\ELECTRONS::TE_HRECE%02d' % i)
            rmidNode = elecTree.getNode('\ELECTRONS::RMID_HRECE%02d' % i)

            rtimes = rmidNode.dim_of().data()
            ttimes = tempNode.dim_of().data()

            self.time[i-startChannel] = ttimes
            self.temp[i-startChannel] = tempNode.data()
            self.rmid[i-startChannel] = np.interp(self.time[i-startChannel], rtimes, rmidNode.data().flatten())

            print i,
            sys.stdout.flush()

        print "done"

        self.time = np.array(self.time)
        self.temp = np.array(self.temp)
        self.rmid = np.array(self.rmid)

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


# Non-RF shots; need to invert H-like data for RF shots
shotList = [
        1150901017,
        1150901020,
        1150901021,
        1150901022,
        1150901023,
        1150901024,
        1150903019,
        1150903021,
        1150903022,
        1150903023,
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
        #1120106022,
        1120106025,
        1120106026,
        1120106027,
        1120106028,
        1120106030,
        1120106031,
        1120106032
        
        
        ]
"""
shotList = [
        1150903021,
        1150903023,
        1150903022]
"""

labela='Pulse, Forward Rotation'
labelb='Pulse, Reverse Rotation'
labelc='No Pulse'

labela='0.6 MA'
labelb='0.8 MA'
labelc='1.1 MA'
labeld='Control'

f0 = plt.figure()
ax0 = f0.add_axes([0.1, 0.1, 0.8, 0.8])

f1 = plt.figure()
ax1 = f1.add_axes([0.1, 0.1, 0.8, 0.8])

f2 = plt.figure()
ax2 = f1.add_axes([0.1, 0.1, 0.8, 0.8])

ionTemps = [[], [], []]
elecTemps = [[], [], []]

for shot in shotList:
    print shot
    
    #print shot
    try:
        td = ThacoData(None, shot, 1)
    except:
        try:
            td = ThacoData(None, shot, 0)
        except:
            continue

    elecTree = MDSplus.Tree('electrons', shot)
    
    if shot < 1130000000:
        fd = FrceceData(elecTree, 4, 13)
    else:
        fd = FrceceData(elecTree, 4, 21)
    
    # TODO: replace with line-integrated data
    temps = td.pro[3,:,:]
    rots = td.pro[1,:,:]
    
    pulses = sat.findColdPulses(shot)
    
    tciNode = elecTree.getNode('\ELECTRONS::TOP.TCI.RESULTS:NL_04')
    dens = tciNode.data()
    dtime = tciNode.dim_of().data()
    
    try:
        rotMeans = np.average(rots, axis=1, weights=td.pro[0,:,:])
        #tempMeans = np.average(rots, axis=1, weights=td.pro[0,:,:])
    except:
        continue
    
    label = ''
    
    times = np.linspace(0.6, 1.4, 17)
    
    afterPulse = False
    
    magTree = MDSplus.Tree('magnetics', shot)
    ipNode = magTree.getNode('\magnetics::ip')
    ipTime = ipNode.dim_of().data()
    ip = ipNode.data()
        
    
    for p in times:
        if afterPulse:
            afterPulse = False
            continue
        
        pdens = dens[np.searchsorted(dtime, p)]
        
        # Need to move since HIREXSR times are the middle of the collection times
        pulseFrame = np.searchsorted(td.time, p-0.01)
        
        if pulseFrame >= len(td.time):
            continue
        
        
            
        preFrame = np.searchsorted(td.time, p-0.03)
        postFrame = np.searchsorted(td.time, p+0.05)
        
        if preFrame == pulseFrame:
            continue
        
        if postFrame >= len(td.time):
            continue
        
        maxTempBefore = np.percentile(temps[preFrame:pulseFrame,:], 95)
        maxTempAfter = np.percentile(temps[pulseFrame:postFrame,:], 95)
        #maxTempBefore = np.max(temps[preFrame:pulseFrame,:])
        #maxTempAfter = np.max(temps[pulseFrame:postFrame,:])
        
        tePulseFrame = np.searchsorted(fd.time[0], p)
        
        tePreFrame = np.searchsorted(fd.time[0], p-0.02)
        tePostFrame = np.searchsorted(fd.time[0], p+0.04)
        
        maxTeBefore = np.percentile(fd.temp[:,tePreFrame:tePulseFrame], 99)
        maxTeAfter = np.percentile(fd.temp[:,tePulseFrame:tePostFrame], 99)
        #maxTeBefore = np.max(te[tePreFrame:tePulseFrame])
        #maxTeAfter = np.max(te[tePulseFrame:tePostFrame])
        
        
        
        """
        if np.any(np.abs(pulses - p) < 0.01):
            afterPulse = True
            if rotMeans[pulseFrame] > 0:
                mark = '4'
                col = 'r'
                label=labela
                labela = ''
                
                ionTemps[0].append(maxTempAfter - maxTempBefore)
                elecTemps[0].append(maxTeAfter - maxTeBefore)
            else:
                mark = '3'
                col = 'g'
                label=labelb
                labelb = ''
                
                ionTemps[1].append(maxTempAfter - maxTempBefore)
                elecTemps[1].append(maxTeAfter - maxTeBefore)
        else:
            mark = '.'
            col = 'b'
            label=labelc
            labelc=''
        
            ionTemps[2].append(maxTempAfter - maxTempBefore)
            elecTemps[2].append(maxTeAfter - maxTeBefore)
        """
        
        pIp = ip[np.searchsorted(ipTime, p)]
        
        if np.any(np.abs(pulses - p) < 0.01):
            afterPulse = True
            if pIp > -7e5:
                mark = 'x'
                col = 'r'
                label=labela
                labela = ''
            elif pIp > -1e6:
                mark = 'x'
                col = 'g'
                label=labelb
                labelb = ''
            else:
                mark = 'x'
                col = 'm'
                label=labelc
                labelc = ''
        else:
            mark = '.'
            col = 'b'
            label=labeld
            labeld=''
                
        
        ax0.scatter(pdens, maxTempAfter - maxTempBefore, c=col, marker=mark, label=label)
        
        ax1.scatter(pdens, maxTeAfter - maxTeBefore, c=col, marker=mark, label=label)
        
        ax2.scatter(maxTeAfter - maxTeBefore, maxTempAfter - maxTempBefore, c=col, marker=mark, label=label)

ax0.set_title('Ion Pulse 95th Percentile')
ax0.set_ylabel('Post-Pulse Ti - Pre-Pulse Ti [KeV]')
ax0.set_xlabel('TCI nl_04 [m^-3]')
ax0.legend(loc='upper right')

ax0.axhline(0)

ax1.set_title('Electron Pulse, 99th Percentile')
ax1.set_ylabel('Post-Pulse Te - Pre-Pulse Te [KeV]')
ax1.set_xlabel('TCI nl_04 [m^-3]')
ax1.legend(loc='upper right')

ax1.axhline(0)


ax2.set_title('Electron Pulse, 99th Percentile')
ax2.set_ylabel('Post-Pulse Te - Pre-Pulse Te [KeV]')
ax2.set_xlabel('TCI nl_04 [m^-3]')
ax2.legend(loc='upper right')

plt.show()