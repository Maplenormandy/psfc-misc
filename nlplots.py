import matplotlib.pyplot as plt
from matplotlib import gridspec
#import matplotlib.colors as colors
from matplotlib.cm import ScalarMappable
import numpy as np

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

class TimePlots:
    rootPlot = None
    crossFig = None
    crossGs = None
    numCrossPlots = 4
    crossAxes = [None] * numCrossPlots

    def initCross():
        rootShareys = [None]*TimePlots.numCrossPlots
        crossShareys = [None]*TimePlots.numCrossPlots

        TimePlots.crossFig = plt.figure(figsize=(22,3*TimePlots.numCrossPlots))
        TimePlots.crossGs = gridspec.GridSpec(TimePlots.numCrossPlots,1)

        for i in range(TimePlots.numCrossPlots):
            if rootShareys != None:
                TimePlots.crossAxes[i] = \
                        TimePlots.crossFig.add_subplot(TimePlots.crossGs[i],
                                sharey=TimePlots.rootPlot.axes[rootShareys[i]],
                                sharex=TimePlots.rootPlot.axes[i])
            elif crossShareys != None:
                TimePlots.crossAxes[i] = \
                        TimePlots.crossFig.add_subplot(TimePlots.crossGs[i],
                                sharey=TimePlots.crossAxes[crossShareys[i]],
                                sharex=TimePlots.rootPlot.axes[i])
            else:
                TimePlots.crossAxes[i] = \
                        TimePlots.crossFig.add_subplot(TimePlots.crossGs[i],
                                sharex=TimePlots.rootPlot.axes[i])

        TimePlots.crossFig.subplots_adjust(wspace=0, hspace=0)


    def crossPlot(sm):
        if TimePlots.crossFig == None:
            TimePlots.initCross()

    def __init__(self, sm):
        numPlots = 8
        shareys = [None]*numPlots

        self.fig = plt.figure(figsize=(8,numPlots*3))
        self.gs = gridspec.GridSpec(numPlots,1)

        self.fig.suptitle('Shot ' + str(sm.shot))

        self.axes = [None]*numPlots
        self.sm = sm

        for i in range(numPlots):
            if TimePlots.rootPlot == None:
                if i == 0:
                    self.axes[i] = self.fig.add_subplot(self.gs[i])
                elif shareys[i] == None:
                    self.axes[i] = self.fig.add_subplot(self.gs[i], sharex=self.axes[0])
                else:
                    self.axes[i] = self.fig.add_subplot(self.gs[i], sharey=shareys[i],
                            sharex=self.axes[0])

            else:
                self.axes[i] = self.fig.add_subplot(self.gs[i],
                        sharey=TimePlots.rootPlot.axes[i], sharex=TimePlots.rootPlot.axes[i])

        self.fig.subplots_adjust(wspace=0, hspace=0)

        if TimePlots.rootPlot == None:
            TimePlots.rootPlot = self

        self.multiTimeTrace(0, sm.frceceData.time, np.median(sm.frceceData.rmid, axis=-1),
                sm.frceceData.temp, 'Te (keV)')
        self.multiTimeTrace(1, sm.thacoData.time, sm.thacoData.rho,
                sm.thacoData.pro[3,:,:], 'Ti (keV)', True)
        self.multiTimeTrace(2, sm.thacoData.time, sm.thacoData.rho,
                sm.thacoData.pro[1,:,:], 'Vt (km/s)', True)
        self.multiTimeTrace(3, sm.tscData.time, np.median(sm.tscData.rmid, axis=-1),
                sm.tscData.dens/1e20, 'TSC ne (10^20 m^-3)')
        self.multiTimeTrace(4, sm.tciData.time, sm.tciData.rmid,
                sm.tciData.dens/1e20, 'TCI (core) ne (10^20 m^-3)')
        self.timeTrace(5, sm.ipNode.dim_of().data(), sm.ipNode.data()/1e6, 'Ip (MA)')
        self.timeTrace(6, sm.q95Node.dim_of().data(), sm.q95Node.data(), 'q95')
        self.timeTrace(7, sm.rfNode.dim_of().data(), sm.rfNode.data(), 'Net RF (MW)')

        self.fig.canvas.draw()
        plt.show(block=False)

    def timeTrace(self, i, time, data, ylabel):
        self.axes[i].plot(time, data)
        self.axes[i].set_ylabel(ylabel)

    def multiTimeTrace(self, i, time, rhoFlat, data, ylabel, reverse=True):
        if reverse:
            sm = ScalarMappable(cmap='gist_rainbow')
        else:
            sm = ScalarMappable(cmap='gist_rainbow_r')

        rhoColor = sm.to_rgba(rhoFlat)

        for j in range(len(rhoFlat)):
            if len(time.shape) > 1:
                if (data.shape[0] < data.shape[1]):
                    self.axes[i].plot(time[j,:], data[j,:], c=rhoColor[j])
                else:
                    self.axes[i].plot(time[:,j], data[:,j], c=rhoColor[j])
            else:
                if (data.shape[0] < data.shape[1]):
                    self.axes[i].plot(time, data[j,:], c=rhoColor[j])
                else:
                    self.axes[i].plot(time, data[:,j], c=rhoColor[j])

        self.axes[i].set_ylabel(ylabel)


tps = []

def plot(sm):
    tps.append(TimePlots(sm))
