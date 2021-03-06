import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import gridspec
import matplotlib.colors as colors
from matplotlib.cm import ScalarMappable

import readline
import MDSplus

import sys
import traceback

readline

font = {'family': 'normal', 'size': 18}
matplotlib.rc('font', **font)

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


class FrceceMap:
    allMaps = []

    def __init__(self, shot, numChannels=24):
        FrceceMap.allMaps.append(self)

        self.shot = shot
        self.tree = MDSplus.Tree('electrons', shot)
        self.anaTree = MDSplus.Tree('analysis', shot)
        self.numChannels = numChannels

        self.times = [None]*(numChannels+1)
        self.temps = [None]*numChannels
        self.rmids = [None]*numChannels

        self.tmeans = np.zeros(numChannels)
        self.rmeans = np.zeros(numChannels)

        rmagxNode = self.anaTree.getNode('\\analysis::efit_aeqdsk:rmagx')
        aoutNode = self.anaTree.getNode('\\analysis::efit_aeqdsk:aout')

        print "Loaded channels:",

        for i in range(1, numChannels+1):
            tempNode = self.tree.getNode('\ELECTRONS::TE_HRECE%02d' % i)
            rmidNode = self.tree.getNode('\ELECTRONS::RMID_HRECE%02d' % i)

            rtimes = rmidNode.dim_of().data()
            ttimes = tempNode.dim_of().data()

            self.times[i-1] = ttimes
            self.temps[i-1] = tempNode.data()[:-1]
            self.rmids[i-1] = np.interp(self.times[i-1], rtimes, rmidNode.data().flatten())
            rmagxSampled = np.interp(self.times[i-1], rmagxNode.dim_of().data(), rmagxNode.data())
            aoutSampled = np.interp(self.times[i-1], aoutNode.dim_of().data(), aoutNode.data())
            self.rmids[i-1] = (self.rmids[i-1]*100 - rmagxSampled) / aoutSampled

            if np.any(self.temps[i-1] < -1):
                self.temps[i-1] = np.ones(self.temps[i-1].shape)

            print i,
            sys.stdout.flush()


        print "done, v2"





        self.times[numChannels] = self.times[numChannels-1]

        self.times = np.array(self.times)
        self.temps = np.array(self.temps)
        self.rmids = np.array(self.rmids)

        tmeans = self.temps.mean(1, keepdims = True)
        #tmeans[18] = (2*tmeans[17]+1*tmeans[20])/3
        #tmeans[19] = (1*tmeans[17]+2*tmeans[20])/3

        rmeans = self.rmids.mean(1)
        prof = fitProfile(rmeans[5:], tmeans.flatten()[5:])
        print "Gaussian fit params:", prof[1], prof[2], prof[3]

        self.tempPlot = (self.temps - tmeans)
        #self.tempPlot[18].fill(0)
        #self.tempPlot[19].fill(0)

        self.rmidPlot = np.zeros(self.times.shape)

        self.rmidPlot[:-1,:] = self.rmids / 2.0
        self.rmidPlot[1:,:] += self.rmids / 2.0


        self.rmidPlot[0,:] = 2*self.rmidPlot[1,:] - self.rmidPlot[2,:]
        self.rmidPlot[numChannels,:] = 2*self.rmidPlot[numChannels-1,:] - self.rmidPlot[numChannels-2,:]

        self.fig = plt.figure(figsize=(22,5))
        self.gs = gridspec.GridSpec(1,2, width_ratios=[5,1])

        self.axTmap = self.fig.add_subplot(self.gs[0])
        self.axTpro = self.fig.add_subplot(self.gs[1], sharey=self.axTmap)

        self.norm = colors.SymLogNorm(linthresh=0.1, vmin=-0.5, vmax=0.5)

        self.axTpro.scatter(tmeans.flatten(), rmeans)
        rline = np.linspace(rmeans.min(), rmeans.max())
        self.axTpro.plot(prof[0](rline), rline)

        norm = colors.SymLogNorm(linthresh = prof[3]/20, vmin = -prof[3]/4, vmax = prof[3]/4)

        self.caxTmap = self.axTmap.pcolormesh(self.times, self.rmidPlot, self.tempPlot, norm=norm, cmap='spectral')
        self.axTmap.set_ylim([0.15, 0.95])
        self.fig.colorbar(self.caxTmap)

        self.axTmap.callbacks.connect('xlim_changed', self.zoomFunc)
        self.axTmap.set_xlabel('Time [sec]')
        self.axTmap.set_ylabel('r/a')
        #self.axTmap.set_title('Shot ' + str(self.shot) + ' Te Diff from Mean')
        self.axTmap.set_title('Te Diff from Mean [keV]')

        self.axTpro.set_xlabel('avg Te')
        self.axTpro.axes.get_yaxis().set_visible(False)
        self.fig.subplots_adjust(wspace=0)
        #self.axTpro.set_title('Time-Avg Profile')

        self.fig.canvas.draw()
        plt.show(block=False)

        self.pnum = 0

    def testFunc(self, ax):
        print ax

    def zoomFunc(self, ax):
        tlower = ax.viewLim.bounds[0]
        tupper = tlower + ax.viewLim.bounds[2]

        newInd = self.times[0].searchsorted([tlower, tupper])
        if newInd[1] == len(self.times[0]):
            newInd[1] -= 2

        tmeans = self.temps[:,newInd[0]:newInd[1]].mean(1, keepdims=True)
        #tmeans[18] = (2*tmeans[17]+1*tmeans[20])/3
        #tmeans[19] = (1*tmeans[17]+2*tmeans[20])/3


        self.axTpro.clear()


        rmeans = self.rmids[:,newInd[0]:newInd[1]].mean(1)
        prof = fitProfile(rmeans[5:], tmeans.flatten()[5:])
        print "Gaussian fit params:", prof[1], prof[2], prof[3]
        self.axTpro.scatter(tmeans.flatten(), rmeans)
        rline = np.linspace(rmeans.min(), rmeans.max())
        self.axTpro.plot(prof[0](rline), rline)

        self.tempPlot[:,newInd[0]:newInd[1]] = (self.temps[:,newInd[0]:newInd[1]] - tmeans)
        #self.tempPlot[18].fill(0)
        #self.tempPlot[19].fill(0)
        self.caxTmap.set_array(self.tempPlot.ravel())

        #norm = colors.SymLogNorm(linthresh = prof[3]/20, vmin = -prof[3]/4, vmax = prof[3]/4)
        norm = colors.SymLogNorm(linthresh = 0.1, vmin = -0.3, vmax = 0.3)
        self.caxTmap.set_norm(norm)

        self.axTpro.set_xlabel('avg Te')
        self.axTpro.set_title('Time-Average Profile')

        self.fig.canvas.draw()

    def set_xlim(self, xlims):
        self.axTmap.set_xlim(xlims)
        self.fig.canvas.draw()

    def set_ylim(self, ylims):
        self.axTmap.set_ylim(ylims)
        self.fig.canvas.draw()

class ThacoMap:
    def __init__(self, shot, tht=1, line='Z'):
        self.shot = shot
        self.tree = MDSplus.Tree('spectroscopy', shot)

        if (tht == 0):
            self.tht = ''
        else:
            self.tht = str(tht)

        heNode = self.tree.getNode('\SPECTROSCOPY::TOP.HIREXSR.ANALYSIS'
                + self.tht + '.HELIKE.PROFILES.' + line)
        """
        hyNode = self.tree.getNode('\SPECTROSCOPY::TOP.HIREXSR.ANALYSIS'
                + self.tht + '.HLIKE.PROFILES.LYA1')
        """

        """
        rpro = self.proNode.data()
        rrho = self.rhoNode.data()
        rtime = self.rhoNode.dim_of()

        goodTimes = (rtime > 0).sum()

        self.time = centerScale(rtime.data()[:goodTimes])
        self.rho = rrho[0,:] # Assume unchanging rho bins
        self.pro = rpro[:,:goodTimes,:len(self.rho)]
        """

        heproNode = heNode.getNode('PRO')
        herhoNode = heNode.getNode('RHO')
        heperrNode = heNode.getNode('PROERR')

        herpro = heproNode.data()
        herrho = herhoNode.data()
        herperr = heperrNode.data()
        hertime = herhoNode.dim_of()

        hegoodTimes = (hertime > 0).sum()

        self.hetime = hertime.data()[:hegoodTimes]
        self.herho = herrho[0,:] # Assume unchanging rho bins
        self.hepro = herpro[:,:hegoodTimes,:len(self.herho)]
        self.heperr = herperr[:,:hegoodTimes,:len(self.herho)]

        """
        hyproNode = hyNode.getNode('PRO')
        hyrhoNode = hyNode.getNode('RHO')
        hyperrNode = hyNode.getNode('PROERR')

        hyrpro = hyproNode.data()
        hyrrho = hyrhoNode.data()
        hyrperr = hyperrNode.data()
        hyrtime = hyrhoNode.dim_of()

        hygoodTimes = (hyrtime > 0).sum()

        self.hytime = hyrtime.data()[:hygoodTimes]
        self.hyrho = hyrrho[0,:] # Assume unchanging rho bins
        self.hypro = hyrpro[:,:hygoodTimes,:len(self.hyrho)]
        self.hyperr = hyrperr[:,:hygoodTimes,:len(self.hyrho)]
        """

        # Assume same times and rhos
        self.time = self.hetime
        self.rho = self.herho

        self.pro = np.copy(self.hepro)
        self.perr = np.copy(self.heperr)

        """
        for j in range(self.hypro.shape[1]):
            takingHy = False
            for k in reversed(range(self.hypro.shape[2])):
                if self.perr[3,j,k] > self.hyperr[3,j,k]:
                    takingHy = True

                if takingHy:
                    self.pro[:,j,k] = self.hypro[:,j,k]
                    self.perr[:,j,k] = self.hyperr[:,j,k]
        """

        self.rplot, self.tplot = np.meshgrid(np.sqrt(self.rho), self.time)
        self.tiplot = self.pro[3,:,:-1]
        self.vtorplot = self.pro[1,:,:-1]


        self.fig = plt.figure(figsize=(22,6))
        self.ax = self.fig.add_subplot(111)
        gs = gridspec.GridSpec(2,1)
        self.ax1 = self.fig.add_subplot(gs[0])
        self.cax1 = self.ax1.pcolormesh(self.tplot, self.rplot, self.tiplot, cmap='cubehelix', vmin=0.3, vmax=3.0)
        #self.cax1.ax.set_ylabel('Ti [keV]', rotation=270)
        self.fig.colorbar(self.cax1)

        #self.fig.suptitle('Shot ' + str(self.shot) + ' Ion Temp [keV], Toroidal Velocity [kHz]')
        self.ax1.set_title('Ion Temp [keV], Toroidal Velocity [kHz]')

        self.ax2 = self.fig.add_subplot(gs[1], sharex=self.ax1, sharey=self.ax1)
        self.cax2 = self.ax2.pcolormesh(self.tplot, self.rplot, self.vtorplot, cmap='BrBG', vmin=-20, vmax=20)
        #self.cax2.ax.set_ylabel('$\omega_t$ [kHz]', rotation=270)
        self.fig.colorbar(self.cax2)

        self.ax.spines['top'].set_color('none')
        self.ax.spines['bottom'].set_color('none')
        self.ax.spines['left'].set_color('none')
        self.ax.spines['right'].set_color('none')
        self.ax.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')
        self.ax.set_ylabel('r/a')
        self.ax2.set_xlabel('Time [sec]')

        self.ax1.xaxis.set_visible(False)

        self.fig.canvas.draw()
        plt.show(block=False)

class HirexsrSpecMap:
    def __init__(self, shot, tht=1):
        if (shot != None):
            self.shot = shot
            self.specTree = MDSplus.Tree('spectroscopy', shot)

            if (tht == 0):
                self.tht = ''
            else:
                self.tht = str(tht)

            self.thtNode = self.specTree.getNode('\SPECTROSCOPY::TOP.HIREXSR.ANALYSIS' + self.tht + '.HELIKE.SPEC')

        specNode = self.thtNode.getNode('SPECBR')
        rbr = specNode.data()
        rtime = specNode.dim_of(1).data()

        goodChans = (rbr[0,0,:] > 0).sum()
        goodTimes = (rtime > 0).sum()

        self.time = rtime[:goodTimes]
        self.br = rbr[:,:goodTimes,:goodChans]

        self.bg = np.percentile(self.br, 10, axis=0).T

        self.tp = centerScale(self.time)
        self.cp = centerScale(range(self.br.shape[2]))

        self.tplot, self.cplot = np.meshgrid(self.tp, self.cp)

        self.fig = plt.figure(figsize=(22,4))
        gs = gridspec.GridSpec(1,1)
        self.ax1 = self.fig.add_subplot(gs[0])
        self.cax1 = self.ax1.pcolormesh(self.tplot, self.cplot, self.bg, cmap='cubehelix')
        self.fig.colorbar(self.cax1)

        self.fig.suptitle('Shot ' + str(self.shot) + ' B_lambda background')

        self.fig.canvas.draw()
        plt.show(block=False)

class ElecTraceMap:
    def __init__(self, shot):
        self.shot = shot

        self.elecTree = MDSplus.Tree('electrons', shot)
        self.magTree = MDSplus.Tree('magnetics', shot)
        self.anaTree = MDSplus.Tree('analysis', shot)
        self.rfTree = MDSplus.Tree('rf', shot)

        self.fig = plt.figure(figsize=(22,15))
        self.gs = gridspec.GridSpec(6,1)
        self.fig.suptitle('Shot ' + str(self.shot))

        self.ax0 = self.fig.add_subplot(self.gs[0])
        self.ax1 = self.fig.add_subplot(self.gs[1], sharex=self.ax0)
        self.ax2 = self.fig.add_subplot(self.gs[2], sharex=self.ax0)
        self.ax3 = self.fig.add_subplot(self.gs[3], sharex=self.ax0)
        self.ax4 = self.fig.add_subplot(self.gs[4], sharex=self.ax0)
        self.ax5 = self.fig.add_subplot(self.gs[5], sharex=self.ax0)


        self.plotTciData(self.ax0)
        self.plotThomsonCoreData(self.ax1)
        self.plotThomsonEdgeData(self.ax2)

        self.plotOtherData(self.ax3, self.ax4, self.ax5)

        self.fig.subplots_adjust(wspace=0, hspace=0)

        self.fig.canvas.draw()
        plt.show(block=False)

    def plotThomsonCoreData(self, ax):
        proNode = self.elecTree.getNode('\ELECTRONS::TOP.YAG_NEW.RESULTS.PROFILES:NE_RZ')
        rhoNode = self.elecTree.getNode('\ELECTRONS::TOP.YAG_NEW.RESULTS.PROFILES:R_MID_T')

        rpro = proNode.data()
        rrho = rhoNode.data()
        rtime = proNode.dim_of().data()

        goodTimes = rrho[0] > 0

        pro = rpro[:,goodTimes]
        rho = rrho[:,goodTimes]
        time = rtime[goodTimes]

        sm = ScalarMappable()
        rhoColor = sm.to_rgba(-rho)

        for i in range(rpro.shape[0]):
            ax.plot(time, pro[i], c=np.mean(rhoColor[i],axis=0))

        ax.set_ylabel('ne')

    def plotThomsonEdgeData(self, ax):
        proNode = self.elecTree.getNode('\ELECTRONS::TOP.YAG_EDGETS.RESULTS:NE')
        rhoNode = self.elecTree.getNode('\ELECTRONS::TOP.YAG_EDGETS.RESULTS:RMID')

        rpro = proNode.data()
        rrho = rhoNode.data()
        rtime = proNode.dim_of().data()

        goodTimes = rrho[0] > 0

        pro = rpro[:,goodTimes]
        rho = rrho[:,goodTimes]
        time = rtime[goodTimes]

        sm = ScalarMappable()
        rhoColor = sm.to_rgba(-rho)

        for i in range(rpro.shape[0]):
            ax.plot(time, pro[i], c=np.mean(rhoColor[i],axis=0))

        ax.set_ylabel('ne')

    def plotTciData(self, ax):
        rdata = self.elecTree.getNode('\electrons::top.tci.results:rad').data()
        sm = ScalarMappable()
        rhoColor = sm.to_rgba(-rdata)

        for i in range(1,11):
            pnode = self.elecTree.getNode('\electrons::top.tci.results:nl_%02d' % i)
            ax.plot(pnode.dim_of().data(), pnode.data(), c = rhoColor[i-1])

        ax.set_ylabel('ne')

    def plotOtherData(self, axA, axB, axC):
        ipNode = self.magTree.getNode('\magnetics::ip')
        q95Node = self.anaTree.getNode('\\analysis::efit_aeqdsk:qpsib')
        rfNode = self.rfTree.getNode('\\rf::rf_power_net')

        axA.plot(ipNode.dim_of().data(), ipNode.data())
        axA.set_ylabel('Ip')
        axB.plot(q95Node.dim_of().data(), q95Node.data())
        axB.set_ylabel('q95')
        axC.plot(rfNode.dim_of().data(), rfNode.data())
        axC.set_ylabel('Net RF power')

"""
# Test temperatures
temps = [[np.linspace(0.0, 1.0)]*6, np.random.rand(6,50)+[np.linspace(0.0,12.0)]*6, [[0.0]*50, [0.2]*50, [0.4]*50, [0.6]*50, [0.8]*50, [1.0]*50]]
"""

print "Try typing 'frcece 1150903023' or 'thaco 1150903023'"

line = False

while True:
    line = raw_input().lower()

    if line == 'exit':
        sys.exit(0)
        break

    toks = line.split(' ')

    try:
        if toks[0] == 'frcece':
            if len(toks) == 2:
                FrceceMap(int(toks[1]))
            elif len(toks) == 3:
                FrceceMap(int(toks[1]), int(toks[2]))
            else:
                print "Syntax is 'frcece shotNumber [numChannels]'"

        if toks[0] == 'thaco':
            if len(toks) == 2:
                ThacoMap(int(toks[1]))
            elif len(toks) == 3:
                ThacoMap(int(toks[1]), int(toks[2]))
            elif len(toks) == 4:
                ThacoMap(int(toks[1]), int(toks[2]), toks[3])
            else:
                print "Syntax is 'thaco shotNumber [THT] [line]'"

        if toks[0] == 'hirexsr_bg':
            if len(toks) == 2:
                HirexsrSpecMap(int(toks[1]))
            elif len(toks) == 3:
                HirexsrSpecMap(int(toks[1]), int(toks[2]))
            else:
                print "Syntax is 'thaco shotNumber [THT]'"

        if toks[0] == 'edens':
            if len(toks) == 2:
                ElecTraceMap(int(toks[1]))
            else:
                print "Syntax is 'edens shotNumber'"

        if toks[0] == 'xlim':
            if len(toks) == 3:
                for figNum in plt.get_fignums():
                    fig = plt.figure(figNum)
                    ax = fig.get_axes()
                    ax[0].set_xlim([float(toks[1]), float(toks[2])])
                    fig.canvas.draw()
            elif len(toks) > 3:
                for i in range(3,len(toks)):
                    figNum = int(toks[i])
                    fig = plt.figure(figNum)
                    ax = fig.get_axes()
                    ax[0].set_xlim([float(toks[1]), float(toks[2])])
                    fig.canvas.draw()
            else:
                print "Syntax is 'xlim lower upper [figureNumbers...]'"

        if toks[0] == 'ylim':
            if len(toks) == 3:
                for figNum in plt.get_fignums():
                    fig = plt.figure(figNum)
                    ax = fig.get_axes()
                    ax[0].set_ylim([float(toks[1]), float(toks[2])])
                    fig.canvas.draw()
            elif len(toks) > 3:
                for i in range(3,len(toks)):
                    figNum = int(toks[i])
                    fig = plt.figure(figNum)
                    ax = fig.get_axes()
                    ax[0].set_ylim([float(toks[1]), float(toks[2])])
                    fig.canvas.draw()
            else:
                print "Syntax is 'ylim lower upper [figureNumbers...]'"

    except:
        traceback.print_exc()
