# -*- coding: utf-8 -*-
"""
Defines and implements several classes to do fitting of THACO line data in a
generic way. Uses 2nd order Legendre fitting on noise, and 2nd order Hermite
function fitting on the lines

@author: normandy
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

import readline
import MDSplus

#import eqtools

#from scipy import stats
import scipy.optimize as op

import emcee

#import scipy

from collections import namedtuple

# %%

class BinFit:
    """
    Performs a nonlinear fit and MCMC error estimate of given binned data
    Uses 2nd order Legendre polynomial noise estimate and 2nd order Hermite function line fit
    """
    def __init__(self, lam, specBr, sig, lines, primary_line_ind):
        self.lam = lam
        self.specBr = specBr
        self.sig = sig
        self.lines = lines
        self.pl = primary_line_ind
        
        lamEdge = np.zeros(len(lam)+1)
        lamEdge[1:-1] = (lam[1:] + lam[:-1]) / 2
        lamEdge[-1] = 2 * lamEdge[-2] - lamEdge[-3]
        lamEdge[0] = 2 * lamEdge[1] - lamEdge[2]
        
        # Normalized lambda, for evaluating noise
        self.lamNorm = (lam-np.average(lam))/(np.max(lam)-np.min(lam))*2
        
        # Get the edge of the lambda bins, for integrating over finite pixels
        self.lamEdge = lamEdge
        
        self.nfit = len(lines.names)
        
        # Indices are noise, shift of primary line, scale of primary line, [H0 weights], [H1 weights], [H2 weights]
        # ML is the maximum likelihood theta
        self.result_ml = None
        self.theta_ml = None
        
        self.sampler = None
        self.samples = None
        
        self.good = False
        
    def unpackParams(self, theta):
        noise = theta[0:3]
        
        center = theta[3]*1e-4 + self.lines.lam
        scale = (theta[4]/self.lines.sqrt_m_ratio)*1e-4
        herm0 = theta[5:5+self.nfit]
        herm1 = theta[5+self.nfit:5+self.nfit*2]
        herm2 = theta[5+self.nfit*2:5+self.nfit*3]
        
        return noise, center, scale, herm0, herm1, herm2
        
    def modelPredict(self, theta):
        noise, center, scale, herm0, herm1, herm2 = self.unpackParams(theta)
        
        # Shift and scale lambdas to evaluation points
        lamEv = (self.lam[:,np.newaxis]-center)/scale
        lamEdgeEv = (self.lamEdge[:,np.newaxis]-center)/scale
        
        # Evaluate gaussian functions
        gauss = np.exp(-lamEv**2 / 2)
        gaussEdge = np.exp(-lamEdgeEv**2 / 2)
        
        # Evaluate hermite functions to model lines
        hn = (herm0 + herm1*lamEv + herm2*(lamEv**2-1)) * gauss
        hnEdge = (herm0 + herm1*lamEdgeEv + herm2*(lamEdgeEv**2-1))*gaussEdge
        
        # Compute integral over finite pixel size
        hnEv = (4 * hn + hnEdge[1:] + hnEdge[:-1])/6.0
        
        # Evaluate noise as 2nd order Legendre fit
        noiseEv = noise[0] + noise[1]*self.lamNorm + noise[2]*(3*self.lamNorm**2-1)/2        
        
        # Sum over all lines
        pred = noiseEv + np.sum(hnEv, axis=1)
        
        return pred
        
    def modelNoise(self, theta):
        noise, center, scale, herm0, herm1, herm2 = self.unpackParams(theta)
        
        noiseEv = noise[0] + noise[1]*self.lamNorm + noise[2]*(3*self.lamNorm**2-1)/2
        
        return noiseEv
    
    def modelLine(self, theta, line=-1, order=2):
        if line < 0:
            line = self.pl
        
        noise, center, scale, herm0, herm1, herm2 = self.unpackParams(theta)
        
        # Shift and scale lambdas to evaluation points
        lamEv = (self.lam[:,np.newaxis]-center)/scale
        lamEdgeEv = (self.lamEdge[:,np.newaxis]-center)/scale
        
        # Evaluate gaussian functions
        gauss = np.exp(-lamEv**2 / 2)
        gaussEdge = np.exp(-lamEdgeEv**2 / 2)
        
        # Evaluate hermite function
        if order >= 2:
            hn = (herm0 + herm1*lamEv + herm2*(lamEv**2-1)) * gauss
            hnEdge = (herm0 + herm1*lamEdgeEv + herm2*(lamEdgeEv**2-1))*gaussEdge
        elif order >= 1:
            hn = (herm0 + herm1*lamEv) * gauss
            hnEdge = (herm0 + herm1*lamEdgeEv)*gaussEdge
        else:
            hn = (herm0) * gauss
            hnEdge = (herm0)*gaussEdge
        
        # Compute integral over finite pixel size
        hnEv = (4 * hn + hnEdge[1:] + hnEdge[:-1])/6.0
        
        # Sum over all lines
        pred = hnEv[:,line]
        
        return pred
        
    def guessFit(self):
        # Warning: these shapes are NOT the same as from unpackParams()
        noise = np.percentile(self.specBr, 5)
        center = np.zeros(self.nfit)
        scale = np.zeros(self.nfit)
        herm0 = np.zeros(self.nfit)
        
        for l in range(self.nfit):
            l0 = np.searchsorted(self.lam, self.lines.lam[l])
            lamFit = self.lam[l0-4:l0+5]
            specFit = self.specBr[l0-4:l0+5]-noise
    
            center[l] = np.average(lamFit, weights=specFit)
            scale[l] = np.sqrt(np.average((lamFit-center[l])**2, weights=specFit))*1e4
            herm0[l] = np.max(self.specBr[l0]-noise, 0)
            
        herm1 = np.zeros(self.nfit)
        herm2 = np.zeros(self.nfit)
        
        theta0 = np.hstack(
                (noise, 0, 0, (center[self.pl]-self.lines.lam[self.pl])*1e4, scale[self.pl], herm0, herm1, herm2))
                    
        return theta0
        
    def lnlike(self, theta):
        pred = self.modelPredict(theta)
        return -np.sum((self.specBr-pred)**2/self.sig**2)
        
    def optimizeFit(self, theta0):
        nfit = self.nfit
        
        h0cnstr = lambda theta, n: theta[5+n]
        h1cnstr = lambda theta, n: theta[5+n]-np.abs(10*theta[5+n+nfit])
        h2cnstr = lambda theta, n: theta[5+n]-np.abs(10*theta[5+n+2*nfit])
        
        constraints = []
        for i in range(nfit):
            constraints.append({
                'type': 'ineq',
                'fun': h0cnstr,
                'args': [i]
            })
            constraints.append({
                'type': 'ineq',
                'fun': h1cnstr,
                'args': [i]
            })
            constraints.append({
                'type': 'ineq',
                'fun': h2cnstr,
                'args': [i]
            })
        
        nll = lambda *args: -self.lnlike(*args)
        
        result_ml = op.minimize(nll, theta0, tol=1e-6, constraints = constraints)
        
        return result_ml
        
    def lnprior(self, theta):
        noise, center, scale, herm0, herm1, herm2 = self.unpackParams(theta)
        
        if np.all(noise[0]>0) and np.all(scale>0) and np.all(herm0>0):
            return 0.0
        else:
            return -np.inf

    def lnprob(self, theta):
        lp = self.lnprior(theta)
        if not np.isfinite(lp):
            return -np.inf
        else:
            return lp+self.lnlike(theta)
        
    def mcmcSample(self, theta_ml):
        ndim, nwalkers = len(theta_ml), len(theta_ml)*4
        pos = [theta_ml + 1e-4 * np.random.randn(ndim) for i in range(nwalkers)]
        
        sampler = emcee.EnsembleSampler(nwalkers, ndim, self.lnprob)
        sampler.run_mcmc(pos, 512)
        
        samples = sampler.chain[:, 128:, :].reshape((-1, ndim))
        
        return samples, sampler
        
    def momentConvert(self, theta, line=-1):
        if line<0:
            line = self.pl
            
        noise, center, scale, herm0, herm1, herm2 = self.unpackParams(theta)

        # Calculate the various moments; these come from analytic formulas
        normFactor = np.sqrt(2*np.pi)*scale[line]
        m0 = normFactor*herm0[line]
        m1 = center[line] * m0 + normFactor*herm1[line]*scale[line] - self.lines.lam[line] * m0
        m2 = scale[line]**2 * m0 + normFactor*herm2[line]*2*scale[line]**2
        
        return np.array([m0, m1*1e3, m2*1e6])
    
    def fit(self, mcmc=True):
        theta0 = self.guessFit()
        noise, center, scale, herm0, herm1, herm2 = self.unpackParams(theta0)
        if herm0[self.pl] < noise[0]*0.1:
            # not worh fitting in this case; i.e. the primary line is under the median noise level
            self.m0_ml = 0.0
            self.good = False
            return False
        else:
            self.result_ml = self.optimizeFit(theta0)
            self.theta_ml = self.result_ml['x']
            if mcmc:
                self.samples, self.sampler = self.mcmcSample(self.theta_ml)
            else:
                self.samples = np.array([self.theta_ml]*50)
                
            self.m_samples = np.apply_along_axis(self.momentConvert, axis=1, arr=self.samples)
            self.m_ml = self.momentConvert(self.theta_ml)
            
            self.theta_avg = np.average(self.samples, axis=0)
            self.m_avg = np.average(self.m_samples, axis=0)
            self.m_std = np.std(self.m_samples, axis=0, ddof=len(theta0))
            
            
            self.good = True
            return True
            

# %%

LineInfo = namedtuple('LineInfo', 'lam m_kev names symbol z sqrt_m_ratio'.split())

class MomentFitter:
    def __init__(self, lam_bounds, primary_line, shot, tht, brancha=True):
        self.lines = LineInfo(None, None, None, None, None, None)
        self.lam_bounds = lam_bounds
        self.primary_line = primary_line
        
        amuToKeV = 931494.095 # amu in keV
        #speedOfLight = 2.998e+5 # speed of light in km/s
        
        # Load all wavelength data
        with open('/home/normandy/idl/HIREXSR/hirexsr_wavelengths.csv', 'r') as f:
            lineData = [s.strip().split(',') for s in f.readlines()]
            lineLam = np.array([float(ld[1]) for ld in lineData[2:]])
            lineZ = np.array([int(ld[2]) for ld in lineData[2:]])
            lineName = np.array([ld[3] for ld in lineData[2:]])
        
        # Load atomic data, for calculating line widths, etc...
        with open('/home/normandy/git/psfc-misc/ReallyMiscScripts/atomic_data.csv', 'r') as f:
            atomData = [s.strip().split(',') for s in f.readlines()]
            atomSymbol = np.array([ad[1] for ad in atomData[1:84]])
            atomMass = np.array([float(ad[3]) for ad in atomData[1:84]]) * amuToKeV
            
            
        # Populate the line data
        lineInd = np.logical_and(lineLam>lam_bounds[0], lineLam<lam_bounds[1])
        #satelliteLines = np.array(['s' not in l for l in lineName])
        #lineInd = np.logical_and(satelliteLines, lineInd)
        ln = lineName[lineInd]
        ll = lineLam[lineInd]
        lz = lineZ[lineInd]
        lm = atomMass[lz-1]
        ls = atomSymbol[lz-1]
        
        # Get the index of the primary line
        self.pl = np.where(ln==primary_line)[0][0]
        
        lr = np.sqrt(lm / lm[self.pl])
        
        self.lines = LineInfo(ll, lm, ln, ls, lz, lr)
        
        print 'Fitting:', self.lines.names
        
        self.shot = shot

        self.specTree = MDSplus.Tree('spectroscopy', shot)
        
        ana = '.ANALYSIS'
        if tht > 0:
            ana += str(tht)
        if brancha:
            br = '.HELIKE'
        else:
            br = '.HLIKE'
            
        branchPath = r'\SPECTROSCOPY::TOP.HIREXSR'+ana+br
        
        self.branchNode = self.specTree.getNode(branchPath)
        
        # Indices are [lambda, time, channel]
        self.specBr_all = self.branchNode.getNode('SPEC:SPECBR').data()
        self.sig_all = self.branchNode.getNode('SPEC:SIG').data()
        self.lam_all = self.branchNode.getNode('SPEC:LAM').data()
        
        # Maximum number of channels, time bins
        self.maxChan = np.max(self.branchNode.getNode('BINNING:CHMAP').data())
        self.maxTime = np.max(self.branchNode.getNode('BINNING:TMAP').data())
        
        self.fits = [[None]*self.maxChan]*self.maxTime
        
    def fitSingleBin(self, tbin, chbin):
        w0, w1 = np.searchsorted(self.lam_all[:,tbin,chbin], self.lam_bounds)
        lam = self.lam_all[w0:w1,tbin,chbin]
        specBr = self.specBr_all[w0:w1,tbin,chbin]
        sig = self.sig_all[w0:w1,tbin,chbin]
        
        bf = BinFit(lam, specBr, sig, self.lines, self.pl)
        self.fits[tbin][chbin] = bf
        
        print "Now fitting", tbin, chbin,
        good = bf.fit()
        if not good:
            print "not worth fitting"
        else:
            print "done"
            
    def fitTimeBin(self, tbin):
        for chbin in range(self.maxChan):
            self.fitSingleBin(tbin, chbin)
            plt.close('all')
            self.plotSingleBinFit(tbin, chbin)
        
    def plotSingleBinFit(self, tbin, chbin):
        bf = self.fits[tbin][chbin]
        
        if bf == None:
            return
    
        
        f0, (a0, a1) = plt.subplots(2, 1, sharex=True, gridspec_kw = {'height_ratios': [4,1]})
        a0.errorbar(bf.lam, bf.specBr, yerr=bf.sig, c='m', fmt='.')
        
        if bf.good:
            pred = bf.modelPredict(bf.theta_ml)
            a0.plot(bf.lam, pred, c='r')
            
            
            for samp in range(25):
                theta = bf.samples[np.random.randint(len(bf.samples))]
                noise = bf.modelNoise(theta)
                a0.plot(bf.lam, noise, c='g', alpha=0.04)
                
                for i in range(len(self.lines.names)):
                    line = bf.modelLine(theta, i)
                    a0.plot(bf.lam, line+noise, c='c', alpha=0.04)
            
                    
                
            noise = bf.modelNoise(bf.theta_avg)
            a0.plot(bf.lam, noise, c='g')
            
            for i in range(len(self.lines.names)):
                line = bf.modelLine(bf.theta_avg, i)
                a0.plot(bf.lam, line+noise, c='c')
                
            a1.errorbar(bf.lam, bf.specBr - pred, yerr=bf.sig, c='r', fmt='.')
            a1.axhline(c='m', ls='--')
            
            for i in range(len(self.lines.names)):
                a1.axvline(self.lines.lam[i], c='b', ls='--')
                a0.axvline(self.lines.lam[i], c='b', ls='--')
            
        plt.show()
                    
                    
# %% Test code

#mf = MomentFitter((3.725, 3.747), 'lya1', 1120914036, 1)
#mf = MomentFitter((3.725, 3.747), 'lya1', 1121002022, 0)
mf = MomentFitter((3.172, 3.188), 'w', 1101014030, 0, False)

tbin = 86
mf.fitTimeBin(tbin)
#mf = mf2

# %% Plot stuff

moments = [None] * mf.maxChan
moments_std = [None] * mf.maxChan

for chbin in range(mf.maxChan):
    if mf.fits[tbin][chbin].good:
        moments[chbin] = mf.fits[tbin][chbin].m_avg
        moments_std[chbin] = mf.fits[tbin][chbin].m_std
    else:
        moments[chbin] = np.zeros(3)
        moments_std[chbin] = np.zeros(3)
    
moments = np.array(moments)
moments_std = np.array(moments_std)

f, a = plt.subplots(3, 1, sharex=True)

a[0].errorbar(range(mf.maxChan), moments[:,0], yerr=moments_std[:,0], fmt='.')
a[1].errorbar(range(mf.maxChan), moments[:,1], yerr=moments_std[:,1], fmt='.')
a[2].errorbar(range(mf.maxChan), moments[:,2], yerr=moments_std[:,2], fmt='.')

# %%

mf.plotSingleBinFit(tbin, 28)