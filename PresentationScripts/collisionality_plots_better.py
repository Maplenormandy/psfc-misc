# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 17:15:04 2016

@author: normandy
"""

from __future__ import division

import profiletools
import gptools
import eqtools

import numpy as np
import scipy

import readline
import MDSplus

import matplotlib.pyplot as plt

import sys
sys.path.append('/home/normandy/git/psfc-misc')

import shotAnalysisTools as sat

import copy

import scipy.optimize as op

readline

# %% Initial defines

def downsampleZave(ztime, z_ave, tmin, tmax):
    newtime = np.arange(tmin, tmax, 0.02)
    searchtime = np.append(newtime, tmax + 0.02) - 0.01
    
    #b, a = scipy.signal.butter(2, 0.005)
    #z_avef = scipy.signal.filtfilt(b, a, z_ave, method="gust")
    z_avef = z_ave
    
    newz = np.zeros(newtime.shape)
    
    bounds = np.searchsorted(ztime, searchtime)
    
    for i in range(len(bounds)-1):
        newz[i] = np.median(z_avef[bounds[i]:bounds[i+1]])
    
    return newtime, newz

# %% Te fitting



def getTeRanges(p_Te, tmin, tmax):
    channels = np.unique(p_Te.channels[:,1])
    fitr = np.zeros((len(channels), 3))
    fitTe = np.zeros((len(channels), 3))
    fitTeerr = np.zeros((len(channels)))
    
    t = p_Te.X[:,0]
    
    for i in range(len(channels)):
        validTimes = np.logical_and(t >= tmin, t < tmax)
        valid = np.logical_and(p_Te.channels[:,1] == channels[i], validTimes)
        cr = p_Te.X[valid, 1]
        cTe = p_Te.y[valid]
        
        if len(cTe) < 3:
            continue
        
        sort = cTe.argsort()
        
        n = len(sort)
        
        i0 = [np.floor(n*0.1), np.ceil(n*0.01)]
        i1 = [np.floor(n*0.5), np.ceil(n*0.5)]
        i2 = [np.floor(n*0.9), np.ceil(n*0.99)]
        i2 = map(lambda x: np.clip(x, 0, len(sort)-1), i2)
        
        i0 = map(int, i0)
        i1 = map(int, i1)
        i2 = map(int, i2)
        
        fitTe[i,0] = np.mean(cTe[sort[i0]])
        fitTe[i,1] = np.mean(cTe[sort[i1]])
        fitTe[i,2] = np.mean(cTe[sort[i2]])
        
        fitr[i,0] = np.mean(cr[sort[i0]])
        fitr[i,1] = np.mean(cr[sort[i1]])
        fitr[i,2] = np.mean(cr[sort[i2]])
        
        fitTeerr[i] = np.sqrt(np.var(cTe, ddof=1) + (fitTe[i,1]/10.0)**2)
        
    fitr = fitr[fitTeerr > 0.015, :]
    fitTe = fitTe[fitTeerr > 0.015, :]
    fitTeerr = fitTeerr[fitTeerr > 0.015]
    
        
    return fitr, fitTe, fitTeerr



def sliceTeCrashPoints(p_Te, sawtimes):
    p = copy.deepcopy(p_Te)

    pt = p.X[:,0]    
    
    peakpoints = np.logical_and.reduce([(pt > st+1e-5) | (pt < st-1e-3) for st in sawtimes])
    p.remove_points(peakpoints)
    
    return p


def fitTeBeforeCrash(p_Te, p_Te2, sawtimes):
    p = sliceTeCrashPoints(p_Te, sawtimes)
    
    try:
        p2 = sliceTeCrashPoints(p_Te2, sawtimes)
        
        p.add_profile(p2)
    except:
        pass
    
    
    p.time_average()

    p.err_y = p.y / 10.0
    
    p.create_gp(k='SE', constrain_at_limiter=False, use_hyper_deriv=True)
    
    p.gp.k.hyperprior = (
                    gptools.UniformJointPrior([(0, 8)]) *
                    gptools.GammaJointPrior([1 + 1 * 5], [5])
                )
    
    p.find_gp_MAP_estimate(random_starts=2)
    return p

def fitTe(r, Te, err):
    k0 = gptools.SquaredExponentialKernel()
    k0.hyperprior = (
                    gptools.UniformJointPrior([(0, 8)]) *
                    gptools.GammaJointPrior([1 + 1 * 5], [5])
                )
    gp0 = gptools.GaussianProcess(k0, use_hyper_deriv=True)
    gp0.add_data(r, Te, err_y = err)
    
    gp0.optimize_hyperparameters(random_starts=2)
    
    return gp0


def getTeFit(p_Te, p_Te2, gpc0time, gpc0te, t):
    
    print t
    peaks = sat.findSawteeth(gpc0time, gpc0te, t-0.012, t+0.012)
    
    tmin = gpc0time[peaks[0]]
    tmax = gpc0time[peaks[-1]]
    
    r, Te, sTe = getTeRanges(p_Te, tmin, tmax)
    
    """
    try:
        r1, Te1, sTe1 = getTeRanges(p_Te, tmin, tmax)
        try:
            r2, Te2, sTe2 = getTeRanges(p_Te2, tmin, tmax)
        except:
            r = r1
            Te = Te1
            sTe = sTe1
            
        sTe = np.concatenate((sTe1, sTe2))
        r = np.vstack((r1, r2))
        Te = np.vstack((Te1, Te2))
    except:
        r, Te, sTe = getTeRanges(p_Te2, tmin, tmax)
    """        
    
    
    
    
    gp0 = fitTe(r[:,0], Te[:,0], sTe)
    gp1 = fitTe(r[:,1], Te[:,1], sTe)
    gp2 = fitTe(r[:,2], Te[:,2], sTe)
    
    maxPeaks = sat.rephaseToNearbyMax(peaks, gpc0te, 4)
    
    p_peak = fitTeBeforeCrash(p_Te, p_Te2, gpc0time[maxPeaks])
    
    # Note the standard deviations are meaningless due to the random guess errors
    return (lambda roa: gp0.predict(roa)[0],
            lambda roa: gp1.predict(roa)[0],
            lambda roa: gp2.predict(roa)[0],
            lambda roa: p_peak.smooth(roa)[0],
            lambda roa: p_peak.smooth(roa, n=1)[0])

# %% ne fitting; need to load every time

def fitNe(p_ne, t):
    p = copy.deepcopy(p_ne)
    p.remove_points((p.X[:, 0] < t-0.09) | (p.X[:, 0] > t+0.09))
    p.remove_points(p.y < 0.1)
    p.create_gp(k='SE', constrain_at_limiter=False, use_hyper_deriv=True)

    p.gp.k.hyperprior = (
                    gptools.UniformJointPrior([(0, 5)]) *
                    gptools.UniformJointPrior([(0, 0.1)]) *
                    gptools.GammaJointPrior([1 + 1 * 5], [5])
                )
    
    print p.find_gp_MAP_estimate(random_starts=2)
    
    return p
    
def evalNeFit(p_ne, roa, t, n=0):
    roa = np.array(roa)
    a = np.vstack((np.ones(roa.shape)*t, roa)).T
    return p_ne.smooth(a, n=n)

def getNeFit(p_ne, t):
    return lambda roa: evalNeFit(p_ne, roa, t)[0], lambda roa: evalNeFit(p_ne, roa, t, n=1)[0]


# %% Collisionality class

class NustarProfile:
    def __init__(self, shot, tmin, tmax):
        # Load data
        self.e = eqtools.CModEFITTree(shot)
        
        self.p_ne_master = profiletools.ne(shot, abscissa='r/a', t_min=tmin, t_max=tmax, include=['CTS', 'ETS'], efit_tree=self.e)
        self.p_Te_master = profiletools.Te(shot, abscissa='r/a', t_min=tmin, t_max=tmax, include=['GPC'], efit_tree=self.e)
        self.p_Te2_master = profiletools.Te(shot, abscissa='r/a', t_min=tmin, t_max=tmax, include=['GPC2'], efit_tree=self.e)
        
        specTree = MDSplus.Tree('spectroscopy', shot)
        z_aveNode = specTree.getNode(r'\z_ave')
        z_ave = z_aveNode.data()
        ztime = z_aveNode.dim_of().data()
        
        ztime, zeff = downsampleZave(ztime, z_ave, tmin, tmax)
        
        self.zefff = scipy.interpolate.interp1d(ztime, zeff)
        
        electrons = MDSplus.Tree('electrons', shot)
        gpc0 = electrons.getNode(r'\ELECTRONS::GPC2_TE0')
        self.gpc0time = gpc0.dim_of().data()
        self.gpc0te = gpc0.data()
        
        # Magnetics manipulations
        qp = self.e.getQProfile()
        qt = self.e.getTimeBase()
        psin = np.linspace(0, 1, qp.shape[1])
        
        qfpsi = scipy.interpolate.interp2d(psin, qt, qp)
        
        self.qfroa = lambda t: lambda roa: qfpsi(self.e.roa2psinorm(roa, t, each_t=True), t)            
        magRf = self.e.getMagRSpline(kind='linear')
        magaf = self.e.getAOutSpline(kind='linear')
        self.epsfroa = lambda t: lambda roa: roa * magaf(t) / magRf(t)
        
    def fitNe(self, tnefits):
        self.tnefits = tnefits
        self.p_ne = [None] * len(tnefits)
        for i in range(len(tnefits)):
            self.p_ne[i] = fitNe(self.p_ne_master, tnefits[i])
            
    def evalProfile(self, tfits):
        neFit = [None] * len(tfits)
        dne = [None] * len(tfits)
        TeMin = [None] * len(tfits)    
        TeMed = [None] * len(tfits)
        TeMax = [None] * len(tfits)
        TeCrash = [None] * len(tfits)
        dTeCrash = [None] * len(tfits)
        q = [None] * len(tfits)
        eps = [None] * len(tfits)
        
        self.collMin = [None] * len(tfits)
        self.collMed = [None] * len(tfits)
        self.collMax = [None] * len(tfits)
        
        def coll(eps, q, ne, zeff, Te):
            return lambda roa: 0.0118/(eps(roa)**1.5)*q(roa)*ne(roa)*zeff/(Te(roa)**2)
        
        for j in range(len(tfits)):
            # Calculate the relevant profile fit
            i = np.argmin(np.abs(self.tnefits - tfits[j]))
            
            # Get things as a function of r/a
            neFit[j], dne[j] = getNeFit(self.p_ne[i], tfits[j])
            TeMin[j], TeMed[j], TeMax[j], TeCrash[j], dTeCrash[j] = getTeFit(self.p_Te_master, self.p_Te2_master, self.gpc0time, self.gpc0te, tfits[j])
            q[j] = self.qfroa(tfits[j])
            eps[j] = self.epsfroa(tfits[j])
            
            # Calculate collisionality
            self.collMin[j] = coll(eps[j], q[j], neFit[j], self.zefff(tfits[j]), TeMax[j])
            self.collMed[j] = coll(eps[j], q[j], neFit[j], self.zefff(tfits[j]), TeMed[j])
            self.collMax[j] = coll(eps[j], q[j], neFit[j], self.zefff(tfits[j]), TeMin[j])

        self.neFit = neFit
        self.dne = dne
        self.TeMin = TeMin
        self.TeMed = TeMed
        self.TeMax = TeMax
        self.TeCrash = TeCrash
        self.dTeCrash = dTeCrash
        self.q = q
        self.eps = eps
        self.tfits = tfits
        
    
# %% Temporary construction functions
    
def calcTraces(slf):
    def unpack(f):
        return lambda x: f(x)[0]

    slf.numinTrace = np.zeros(len(slf.tfits))
    slf.xminTrace = np.zeros(len(slf.tfits))        
        
    for j in range(len(slf.tfits)):
        res = op.minimize_scalar(unpack(slf.collMin[j]), bounds=[0.2, 0.8], method='bounded')
        slf.numinTrace[j] = res.fun
        slf.xminTrace[j] = res.x
        
    
    
# %% Collisionality plot

nustar = NustarProfile(1160506015, 0.4, 1.6)
nustar.fitNe([0.77, 1.01])
nustar.evalProfile(np.array([0.77, 1.01]))

rho = np.linspace(0.1,0.9)
plt.figure()
plt.plot(rho, (nustar.neFit[0](rho)))
#plt.plot(rho, (nustar.neFit[1](rho)))

d0 = np.gradient(nustar.neFit[0](rho), np.median(np.diff(rho)))
d1 = np.gradient(nustar.neFit[0](rho), np.median(np.diff(rho)))

plt.figure()
plt.plot(rho, (nustar.neFit[0](rho)/d0))
#plt.plot(rho, (nustar.neFit[1](rho)/d1))

#plt.plot(rho, (nustar.neFit[1](rho)))

#plt.scatter(rtest, tetest)
#plt.scatter(rtest2, tetest2, c='r')



# %% Plot the fits


#plt.errorbar(np.linspace(0,1), ne2, yerr=nestd2)

#ne0 = np.array([ne(np.zeros(1)) for ne in nustar.neFit])
plt.figure()
#temin = np.array([te(np.zeros(1)) for te in nustar.TeMin])
#temed = np.array([te(np.zeros(1)) for te in nustar.TeMed])
#temax = np.array([te(np.zeros(1)) for te in nustar.TeMax])
#plt.plot(nustar.tfits, temin)
#plt.plot(nustar.tfits, temed)
#plt.plot(nustar.tfits, temax)
#plt.plot(nustar.tfits, ne0)
j = np.argmin(np.abs(nustar.tfits - 0.77))
plt.semilogy(np.linspace(0,1), nustar.collMed[0](np.linspace(0,1)), c='#0000ff', label='0.79s, SOC->LOC')
plt.semilogy(np.linspace(0,1), nustar.collMin[0](np.linspace(0,1)), c='#000055')
plt.semilogy(np.linspace(0,1), nustar.collMax[0](np.linspace(0,1)), c='#000055')
j = np.argmin(np.abs(nustar.tfits - 1.01))
plt.semilogy(np.linspace(0,1), nustar.collMed[j](np.linspace(0,1)), c='#ff5500', label='1.01s, LOC->SOC')
plt.semilogy(np.linspace(0,1), nustar.collMin[j](np.linspace(0,1)), c='#553300')
plt.semilogy(np.linspace(0,1), nustar.collMax[j](np.linspace(0,1)), c='#553300')

plt.legend()


# %% Min plots

def unpack(f):
        return lambda x: f(x)[0]

xs = np.zeros(nustar.tfits.shape)
funs = np.zeros(nustar.tfits.shape)

for i in range(len(nustar.tfits)):
    res = op.minimize_scalar(unpack(nustar.collMin[i]), bounds=[0.2, 0.8], method='bounded')
    xs[i] = res.x
    funs[i] = res.fun

plt.figure()
plt.plot(nustar.tfits, funs, label='min nu*')
plt.plot(nustar.tfits, xs, label='argmin nu*')

plt.vlines([0.86, 1.25], 0.0, 1.0, label='reversals')

plt.legend()

# %% ne0, te0, etc...

plt.figure()
ne0 = np.array([ne(0) for ne in nustar.neFit])
plt.plot(nustar.tfits, ne0)

"""
plt.errorbar(np.linspace(0,1), Temean0, yerr=Testd0, c='b')
plt.errorbar(np.linspace(0,1), Temean1, yerr=Testd1, c='g')
plt.errorbar(np.linspace(0,1), Temean2, yerr=Testd2, c='r')

plt.scatter(r1[:,0], Te1[:,0], c='b')
plt.scatter(r2[:,0], Te2[:,0], c='b', marker='^')
plt.scatter(r1[:,1], Te1[:,1], c='g')
plt.scatter(r2[:,1], Te2[:,1], c='g', marker='^')
plt.scatter(r1[:,2], Te1[:,2], c='r')
plt.scatter(r2[:,2], Te2[:,2], c='r', marker='^')
"""
