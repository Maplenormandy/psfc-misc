import numpy as np

import readline
import MDSplus
import matplotlib.pyplot as plt


import shotAnalysisTools as sat

from multiprocessing import Pool

"""
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
        1150903028
        ]
"""

shotList = [
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

shotList = [
        1150903021,
        1150903023,
        1150903022]


#shotList = range(1150728016, 1150728029)




def dataFunc(args):
    shot = args
    pulses = sat.findColdPulses(shot)
    
    print shot, "data started"

    elecTree = MDSplus.Tree('electrons', shot)
    teNode = elecTree.getNode('\gpc_t0')
    
    te = teNode.data()
    time = teNode.dim_of().data()
    peaks = sat.findSawteeth(time, te, 0.57, 1.43)
    
    print shot, "data done"
    
    return shot, te, time , peaks, pulses

def plotFunc(args):
    ax, (shot, te, time, peaks, pulses) = args
    
    medTime, medTe = sat.sawtoothMedian(peaks, time, te)
    
    ax.plot(medTime, medTe)
    
    ax.plot(time, te)
    ax.scatter(time[peaks], te[peaks], c='r', marker='^')

    for p in pulses:
        ax.axvline(x=p, c='r', ls='--')

    ax.set_title(str(shot), y=0.8)

if __name__ == '__main__':
    nrows = min(8, len(shotList))
    ncols = ((len(shotList) - 1) / nrows) + 1
    f, axarr = plt.subplots(nrows,ncols, sharex=True, sharey=True)
    
    f.subplots_adjust(hspace=0, wspace=0)
    
    plt.draw()
    
    k = 0
    
    p = Pool(4)
    
    f.subplots_adjust(hspace=0, wspace=0)
    
    plotArgs = [0] * len(shotList)
    axArgs = [0] * len(shotList)

    for j in range(ncols):
        for i in range(nrows):
            if (k >= len(shotList)):
                break
            
            shot = shotList[k]
            
            if ncols > 1:
                ax = axarr[i,j]
            else:
                ax = axarr[i]
                    
            plotArgs[k] = shot
            axArgs[k] = ax
    
            k += 1
            

    try:
        dataArgs = p.map(dataFunc, plotArgs)
        dataArgs = [(axArgs[i], dataArgs[i]) for i in range(len(shotList))]
        map(plotFunc, dataArgs)
    except:
        p.terminate()
        p.join()

    print "done"
    
    plt.show()
