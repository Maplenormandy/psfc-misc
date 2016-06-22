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

def getTeRanges(t, r, Te, tmin, tmax):
    times = np.logical_and(t >= tmin, t < tmax)
    rr = r[:,times]
    rTe = Te[:,times]
    
    fitTe = np.zeros((rTe.shape[0], 3))
    fitTeerr = np.zeros(rTe.shape[0])
    fitr = np.zeros((rTe.shape[0], 3))
    
    for i in range(rTe.shape[0]):
        cr = rr[i,:]
        cTe = rTe[i,:]
        
        sort = cTe.argsort()
        
        n = len(sort)
        
        i0 = [np.floor(n*0.01), np.ceil(n*0.01)]
        i1 = [np.floor(n*0.5), np.ceil(n*0.5)]
        i2 = [np.floor(n*0.99)-1, np.ceil(n*0.99)-1]
        
        i0 = map(int, i0)
        i1 = map(int, i1)
        i2 = map(int, i2)
        
        fitTe[i,0] = np.mean(cTe[sort[i0]])
        fitTe[i,1] = np.mean(cTe[sort[i1]])
        fitTe[i,2] = np.mean(cTe[sort[i2]])
        
        fitr[i,0] = np.mean(cr[sort[i0]])
        fitr[i,1] = np.mean(cr[sort[i1]])
        fitr[i,2] = np.mean(cr[sort[i2]])
        
        fitTeerr[i] = np.std(cTe, ddof=1)
    
    
    return fitr, fitTe, fitTeerr

def getTeFit(p_Te, p_Te2, gpc0time, gpc0te, t):
    gpcX = p_Te.X.reshape((8, int(p_Te.X.shape[0]/8), 2))
    gpct = gpcX[0,:,0]
    gpcr = gpcX[:,:,1]
    gpcy = p_Te.y.reshape((8, int(p_Te.y.shape[0]/8)))
    
    gpc2X = p_Te2.X.reshape((10, int(p_Te2.X.shape[0]/10), 2))
    gpc2t = gpc2X[0,:,0]
    gpc2r = gpc2X[:,:,1]
    gpc2y = p_Te2.y.reshape((10, int(p_Te2.y.shape[0]/10)))
    
    peaks = sat.findSawteeth(gpc0time, gpc0te, t-0.012, t+0.012)
    
    tmin = gpc0time[peaks[0]]
    tmax = gpc0time[peaks[-1]]
    
    r1, Te1, sTe1 = getTeRanges(gpct, gpcr, gpcy, tmin, tmax)
    r2, Te2, sTe2 = getTeRanges(gpc2t, gpc2r, gpc2y, tmin, tmax)
    
    k0 = gptools.SquaredExponentialKernel()
    k0.hyperprior = (
                    gptools.UniformJointPrior([(0, 8)]) *
                    gptools.GammaJointPrior([1 + 1 * 5], [5])
                )
    gp0 = gptools.GaussianProcess(k0)
    gp0.add_data(r1[:,0], Te1[:,0], err_y = sTe1/np.sqrt(5))
    gp0.add_data(r2[:,0], Te2[:,0], err_y = sTe2*np.sqrt(5))
    
    gp0.optimize_hyperparameters()
    
    k1 = gptools.SquaredExponentialKernel()
    k1.hyperprior = (
                    gptools.UniformJointPrior([(0, 8)]) *
                    gptools.GammaJointPrior([1 + 1 * 5], [5])
                )
    gp1 = gptools.GaussianProcess(k1)
    gp1.add_data(r1[:,1], Te1[:,1], err_y = sTe1/np.sqrt(5))
    gp1.add_data(r2[:,1], Te2[:,1], err_y = sTe2*np.sqrt(5))
    
    gp1.optimize_hyperparameters()
    
    k2 = gptools.SquaredExponentialKernel()
    k2.hyperprior = (
                    gptools.UniformJointPrior([(0, 8)]) *
                    gptools.GammaJointPrior([1 + 1 * 5], [5])
                )
    gp2 = gptools.GaussianProcess(k2)
    gp2.add_data(r1[:,2], Te1[:,2], err_y = sTe1/np.sqrt(5))
    gp2.add_data(r2[:,2], Te2[:,2], err_y = sTe2*np.sqrt(5))
    
    gp2.optimize_hyperparameters()
    
    Temean0, Testd0 = gp0.predict(np.linspace(0,1))
    Temean1, Testd1 = gp1.predict(np.linspace(0,1))
    Temean2, Testd2 = gp2.predict(np.linspace(0,1))
    
    return Temean0, Temean1, Temean2

# %% ne fitting; need to load every time

def fitNe(p_ne, t):
    p = copy.deepcopy(p_ne)
    p.remove_points((p.X[:, 0] < t-0.05) | (p.X[:, 0] > t+0.05))
    p.create_gp(k='SE', constrain_at_limiter=False)

    p.gp.k.hyperprior = (
                    gptools.UniformJointPrior([(0, 4)]) *
                    gptools.UniformJointPrior([(0, 0.2)]) *
                    gptools.GammaJointPrior([1 + 1 * 5], [5])
                )
    
    print p.find_gp_MAP_estimate()
    
    return p
    
def getNeFit(p_ne, t):
    a = np.vstack((np.ones(50)*t, np.linspace(0,1))).T
    return p_ne.smooth(a)

# %% Loading Data

shot = 1160506009

e = eqtools.CModEFITTree(shot)

p_ne = profiletools.ne(shot, abscissa='r/a', t_min=0.5, t_max=1.5, include=['CTS', 'ETS'], efit_tree=e)
p_Te = profiletools.Te(shot, abscissa='r/a', t_min=0.5, t_max=1.5, include=['GPC'], efit_tree=e)
p_Te2 = profiletools.Te(shot, abscissa='r/a', t_min=0.5, t_max=1.5, include=['GPC2'], efit_tree=e)

specTree = MDSplus.Tree('spectroscopy', shot)
z_aveNode = specTree.getNode(r'\z_ave')
z_ave = z_aveNode.data()
ztime = z_aveNode.dim_of().data()
    
ztime, zeff = downsampleZave(ztime, z_ave, 0.49, 1.51)

zefff = scipy.interpolate.interp1d(ztime, zeff)

electrons = MDSplus.Tree('electrons', shot)
gpc0 = electrons.getNode(r'\ELECTRONS::GPC2_TE0')
gpc0time = gpc0.dim_of().data()
gpc0te = gpc0.data()

# %% Magnetics

qp = e.getQProfile()
qt = e.getTimeBase()
psin = np.linspace(0, 1, qp.shape[1])

qfpsi = scipy.interpolate.interp2d(psin, qt, qp)
def qfroa(roa, t, each_t=True):
    psin = e.roa2psinorm(roa, t, each_t=each_t)
    # TODO Add some sort of scalar checking to this
    return qfpsi(psin, t)

magRf = e.getMagRSpline(kind='linear')
magaf = e.getAOutSpline(kind='linear')
def eps(roa, t):
    return roa * magaf(t) / magRf(t)


# %% Profile Fitting

tfit1 = 0.75
tfit2 = 1.05

p1 = fitNe(p_ne, tfit1)
p2 = fitNe(p_ne, tfit2)

# %% Profile evaluation

tfit1 = 0.75
tfit2 = 1.05

ne1, nestd1 = getNeFit(p1, tfit1)
Te10, Te11, Te12 = getTeFit(p_Te, p_Te2, gpc0time, gpc0te, tfit1)

ne2, nestd2 = getNeFit(p2, tfit2)
Te20, Te21, Te22 = getTeFit(p_Te, p_Te2, gpc0time, gpc0te, tfit2)

    
# %% Collisionality plot

q1 = qfroa(np.linspace(0,1), tfit1)
eps1 = eps(np.linspace(0,1), tfit1)
q2 = qfroa(np.linspace(0,1), tfit2)
eps2 = eps(np.linspace(0,1), tfit2)

coll10 = 0.0118/(eps1**1.5)*q1*ne1*zefff(tfit1)/(Te10**2)
coll11 = 0.0118/(eps1**1.5)*q1*ne1*zefff(tfit1)/(Te11**2)
coll12 = 0.0118/(eps1**1.5)*q1*ne1*zefff(tfit1)/(Te12**2)

coll20 = 0.0118/(eps2**1.5)*q2*ne2*zefff(tfit2)/(Te20**2)
coll21 = 0.0118/(eps2**1.5)*q2*ne2*zefff(tfit2)/(Te21**2)
coll22 = 0.0118/(eps2**1.5)*q2*ne2*zefff(tfit2)/(Te22**2)

plt.semilogy(np.linspace(0,1), coll10, c='#550000')
plt.semilogy(np.linspace(0,1), coll11, c='#ff0000')
plt.semilogy(np.linspace(0,1), coll12, c='#550000')
plt.semilogy(np.linspace(0,1), coll20, c='#005500')
plt.semilogy(np.linspace(0,1), coll21, c='#00aa00')
plt.semilogy(np.linspace(0,1), coll22, c='#005500')



# %% Plot the fits

#plt.errorbar(np.linspace(0,1), ne2, yerr=nestd2)

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
