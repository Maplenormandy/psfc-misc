import numpy as np

import readline
import MDSplus

import sys
import traceback

import matplotlib.pyplot as plt
import nlplots

readline

def fitProfile(rmid, temp):
    # Does a least squares fit of a quadratic on the log of the data, i.e. fit a gaussian curve
    x = np.array([np.ones(rmid.size), rmid, rmid**2])
    coefs = np.linalg.lstsq(x.T, np.log(temp))[0]

    diffLen = np.sqrt(-1/coefs[2])
    center = diffLen*diffLen*coefs[1]/2
    tempPeak = np.exp(coefs[0] + (center*center/diffLen/diffLen))

    profFunc = lambda r: np.exp(coefs[0] + coefs[1]*r + coefs[2]*r*r)

    return (profFunc, diffLen, center, tempPeak)

def centerScale(vals):
    """
    Maps 1,2,3,4 -> 0.5,1.5,2.5,3.5,4.5
    Only works for flat arrays
    """

    newVals = np.append(vals, 2*vals[-1]-vals[-2])
    valDiffs = np.ediff1d(newVals, to_end=0)
    valDiffs[-1] = valDiffs[2]
    newVals -= valDiffs / 2

    return newVals

class ThacoData:
    def __init__(self, thtNode):
        self.thtNode = thtNode

        proNode = thtNode.getNode('PRO')
        rhoNode = thtNode.getNode('RHO')

        rpro = proNode.data()
        rrho = rhoNode.data()
        rtime = rhoNode.dim_of()

        goodTimes = (rtime > 0).sum()

        self.time = rtime.data()[:goodTimes]
        self.rho = rrho[0,:] # Assume unchanging rho bins
        self.pro = rpro[:,:goodTimes,:len(self.rho)]

class FrceceData:
    def __init__(self, elecTree, numChannels):
        self.time = [None]*numChannels
        self.temp = [None]*numChannels
        self.rmid = [None]*numChannels

        print "Loading ECE channels:",

        for i in range(1, numChannels+1):
            tempNode = elecTree.getNode('\ELECTRONS::TE_HRECE%02d' % i)
            rmidNode = elecTree.getNode('\ELECTRONS::RMID_HRECE%02d' % i)

            rtimes = rmidNode.dim_of().data()
            ttimes = tempNode.dim_of().data()

            self.time[i-1] = ttimes
            self.temp[i-1] = tempNode.data()
            self.rmid[i-1] = np.interp(self.time[i-1], rtimes, rmidNode.data().flatten())

            print i,
            sys.stdout.flush()

        print "done"

        self.time = np.array(self.time)
        self.temp = np.array(self.temp)
        self.rmid = np.array(self.rmid)

class ThomsonCoreData:
    def __init__(self, yagNode):
        self.yagNode = yagNode

        densNode = self.yagNode.getNode('NE_RZ')
        rmidNode = self.yagNode.getNode('R_MID_T')

        rdens = densNode.data()
        rrmid = rmidNode.data()
        rtime = densNode.dim_of().data()

        goodTimes = rrmid[0] > 0

        self.dens = np.array(rdens[:,goodTimes])
        self.rmid = np.array(rrmid[:,goodTimes])
        self.time = np.array(rtime[goodTimes])

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

class ShotManager:
    allShots = []

    def __init__(self, shot, tht=1, eceChannels=24):
        self.shot = shot

        self.eceChannels = eceChannels

        self.elecTree = MDSplus.Tree('electrons', shot)
        self.specTree = MDSplus.Tree('spectroscopy', shot)
        self.magTree = MDSplus.Tree('magnetics', shot)
        self.anaTree = MDSplus.Tree('analysis', shot)
        self.rfTree = MDSplus.Tree('rf', shot)

        if (tht == 0):
            self.tht = ''
        else:
            self.tht = str(tht)

        self.thtNode = self.specTree.getNode('\SPECTROSCOPY::TOP.HIREXSR.ANALYSIS'
                + self.tht + '.HELIKE.PROFILES.Z')
        self.yagNode = self.elecTree.getNode('\ELECTRONS::TOP.YAG_NEW.RESULTS.PROFILES')
        self.tciNode = self.elecTree.getNode('\ELECTRONS::TOP.TCI.RESULTS')

        self.thacoData = ThacoData(self.thtNode)
        self.frceceData = FrceceData(self.elecTree, eceChannels)
        self.tscData = ThomsonCoreData(self.yagNode)
        self.tciData = TciData(self.tciNode)

        self.ipNode = self.magTree.getNode('\magnetics::ip')
        self.q95Node = self.anaTree.getNode('\\analysis::efit_aeqdsk:qpsib')
        self.rfNode = self.rfTree.getNode('\\rf::rf_power_net')

        ShotManager.allShots.append(self)

print "Try typing 'shot 1150903021'"

line = False


while True:
    line = raw_input().lower()

    if line == 'exit':
        sys.exit(0)
        break

    toks = line.split(' ')

    try:
        if toks[0] == 'shot':
            sm = None
            if len(toks) == 2:
                sm = ShotManager(int(toks[1]))
            elif len(toks) == 3:
                sm = ShotManager(int(toks[1]), int(toks[2]))
            elif len(toks) == 4:
                sm = ShotManager(int(toks[1]), int(toks[2]), int(toks[3]))
            else:
                print "Syntax is 'shot shotNumber [numChannels] [THT]'"

            if sm != None:
                nlplots.plot(sm)

        if toks[0] == 'reload':
            if toks[1] == 'plots':
                reload(nlplots)
                for figNum in plt.get_fignums():
                    fig = plt.close(plt.figure(figNum))

                for sm in ShotManager.allShots:
                    nlplots.plot(sm)

        if toks[0] == 'save':
            if toks[1] == 'plots':
                i = 0
                for figNum in plt.get_fignums():
                    plt.figure(figNum).savefig(str(i) + '.png')
                    print "Saved " + str(i) + ".png"
                    i += 1

    except:
        traceback.print_exc()
