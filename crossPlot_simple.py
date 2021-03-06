import numpy as np

import readline
import MDSplus

import matplotlib.pyplot as plt

from scipy.integrate import simps

import shotAnalysisTools as sat

from scipy.interpolate import splprep, splev


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

"""
shotList = [
        1120216004,
        1120216004,
]
"""

shotList = [
        1150901017,
        1150901020,
        #1150901021,
        #1150901022,
        #1150901023,
        #1150901024,
        1150903019,
        1150903021,
        1150903022,
        #1150903023,
        1150903026,
        #1150903028,
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

#shotList = range(1150728016, 1150728029)

shotList = [1160506007, 1150903021, 1160506008]

nrows = min(8, len(shotList))
ncols = ((len(shotList) - 1) / nrows) + 1
f, axarr = plt.subplots(nrows,ncols, sharex=True, sharey=True)
f.subplots_adjust(hspace=0, wspace=0)

k = 0
for j in range(ncols):
    for i in range(nrows):
        if (k >= len(shotList)):
            break

        shot = shotList[k]


        if ncols > 1:
            ax = axarr[i,j]
        else:
            ax = axarr[i]
            
        electrons = MDSplus.Tree('electrons', shot)
        gpc0 = electrons.getNode(r'\ELECTRONS::GPC2_TE0')
        time = gpc0.dim_of().data()
        te = gpc0.data()

        #ax.plot(rotNode.dim_of().data(), rotNode.data()[0])
        peaks = sat.findSawteeth(time, te, 0.5, 1.5)
        indMin = time.searchsorted(0.5)-8
        indMax = time.searchsorted(1.5)+8+1
        peaks2 = sat.rephaseToNearbyMax(peaks, te, 4)
        #ax.plot(time, te)
        #ax.scatter(time[peaks], te[peaks])
        #ax.scatter(time[peaks2], te[peaks2])
        ax.scatter(time[peaks[:-1]], np.diff(time[peaks]))

        k += 1

f.subplots_adjust(hspace=0, wspace=0)

plt.show()
