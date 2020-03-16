# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 17:14:40 2019

@author: normandy
"""

import numpy as np
import matplotlib.pyplot as plt

import scipy.interpolate
import scipy.integrate

import readline
readline
import MDSplus

import eqtools

from skimage import measure

# %%

class EquilibriumGeometry:
    def __init__(self, shot, t_range, e=None, plotting=False):
        """
        Creates a new equilibrium geometry object. Inputs are shot number, an
        eqtools instance of an EFITTree, and a tuple (t_min, t_max) corresponding
        to the time_averaging window.
        """
        
        self.shot = shot
        
        if e == None:
            e = eqtools.CModEFITTree(shot)
            
        self.e = e
        
        time = e.getTimeBase()
        
        # These indices are the bounds of the window
        self.t0, self.t1 = t_range
        ti0, ti1 = np.searchsorted(time, t_range)
        self.ti0 = ti0
        self.ti1 = ti1
        
        # tind is the default index at which to take data
        self.tind = (ti0+ti1)/2
        self.t = time[self.tind]
        self.time = time[ti0:ti1]
        
        self.plotting = plotting
        
        self.setup_flux_contours()
        
        # Calculates V'(psi), e.g. the normalized flux surface area Note that this is in normalized psi
        self.fs_vprime = self.fs_integrate(lambda R,Z,psi: 2*np.pi*np.ones(R.shape))
        
        self.calculate_ft()
        
    def setup_flux_contours(self):
        """
        Sets up the flux contours in RZ space on which to perform flux surface averaging.
        This is done by finding contours of the poloidal flux using skimage
        """
        e = self.e
        
        self.npsi = len(e.getRmidPsi()[self.tind])
        self.psigrid = np.linspace(0, 1, self.npsi)
        
        refineFactor = 4
        
        # This is the default psigrid to evaluate on
        psigridRefine = np.linspace(0, 1, self.npsi * refineFactor)
        self.psigridEv = (psigridRefine[1:] + psigridRefine[:-1])/2.0
        self.dpsigridEv = np.diff(psigridRefine)
        
        # This is an array of numpy arrays of (R,Z) coordinates which poloidally
        # parameterize the flux surface. Note that the last point is not equal
        # to the first point, so if a closed contour is desired, the last point
        # must be repeated
        self.psicontours = [None]*(len(self.psigridEv))
        
        # The following two are the length element (ds) and the flux surface
        # area element (ds/Bp) for each flux surface element given in psicontours.
        # Note that these are calculated as centered on the given R,Z coordinates.
        self.psicontours_ds = [None]*(len(self.psigridEv))
        # Note an additional very important fact; Bp has been normalized such
        # that psi_axis = 0 and psi_LCFS = 1!!!
        self.psicontours_dsbp = [None]*(len(self.psigridEv))
        
        rgrid = e.getRGrid()
        zgrid = e.getZGrid()
        
        # Begin by loading up basic magnetic geometry information
        self.magR = e.getMagR()[self.tind]
        self.magZ = e.getMagZ()[self.tind]
        self.magRZ = np.array([self.magR, self.magZ])
        
        self.fluxRZ = e.getFluxGrid()[self.tind,:,:]
        
        psiMin = e.getFluxAxis()[self.tind]
        psiMax = e.getFluxLCFS()[self.tind]
        self.psiRange = psiMax-psiMin
        
        self.psinormRZ = (self.fluxRZ-psiMin)/self.psiRange
        
        # Set up interpolation functions for the poloidal flux
        self.psifunc = scipy.interpolate.RectBivariateSpline(rgrid, zgrid, self.fluxRZ.T, kx=2, ky=2)
        self.psinormfunc = scipy.interpolate.RectBivariateSpline(rgrid, zgrid, self.psinormRZ.T, kx=2, ky=2)
        
        # Set up an upscaled version of the poloidal flux, since the conour finder
        # performs better on the upscaled version
        rgrid_upscaled = np.linspace(rgrid[0], rgrid[-1], len(rgrid)*4)
        zgrid_upscaled = np.linspace(zgrid[0], zgrid[-1], len(zgrid)*8)
        rmesh, zmesh = np.meshgrid(rgrid_upscaled, zgrid_upscaled)
        
        rfunc = scipy.interpolate.interp1d(range(len(rgrid_upscaled)), rgrid_upscaled)
        zfunc = scipy.interpolate.interp1d(range(len(zgrid_upscaled)), zgrid_upscaled)
        psinormRZ_upscaled = self.psinormfunc.ev(rmesh, zmesh)
        
        
        # Use skimage to get contours easily for us
        for i in range(len(self.psigridEv)):
            contours = measure.find_contours(psinormRZ_upscaled, self.psigridEv[i])
            if len(contours) == 1:
                self.psicontours[i] = np.array([rfunc(contours[0][:,1]), zfunc(contours[0][:,0])])
            elif len(contours) > 1:
                minDist = np.inf
                for c in contours:
                    rz = np.array([rfunc(c[:,1]), zfunc(c[:,0])])
                    
                    if np.linalg.norm(np.mean(rz, axis=1) - self.magRZ) < minDist:
                        self.psicontours[i] = rz
                        minDist = np.linalg.norm(np.mean(rz, axis=1) - self.magRZ)
            else:
                self.psicontours[i] = np.array([self.magRZ]).T
        
        if self.plotting:
            plt.figure()
        
        # Resample the contours such that the point spacings are of approximately equal arclength
        for i in range(len(self.psigridEv)):
            cRZ = self.psicontours[i]
            
            ctheta = np.unwrap(np.arctan2(cRZ[1] - self.magZ, cRZ[0] - self.magR))
            cr = np.linalg.norm(cRZ - self.magRZ[:,np.newaxis], axis=0)
            
            dtheta = np.unwrap(np.diff(ctheta))
            dr = np.diff(cr)
            r = (cr[1:] + cr[:-1])/2.0
            
            # Calculate total contour arclength using ds**2 = dr**2 + (r*dtheta)**2
            ds = np.sqrt(dr**2 + (r*dtheta)**2)
            cs = np.concatenate(([0], np.cumsum(ds)))
            
            cthetaFunc = scipy.interpolate.interp1d(cs, ctheta)
            crFunc = scipy.interpolate.interp1d(cs, cr, kind='quadratic')
            
            # Resample to an odd number of points
            resamp_theta = cthetaFunc(np.linspace(0, cs[-1], (len(cr)/2)*2+1))
            resamp_r = crFunc(np.linspace(0, cs[-1], (len(cr)/2)*2+1))
            
            dtheta2 = np.diff(resamp_theta[::2])
            dr2 = np.diff(resamp_r[::2])
            
            cr2 = resamp_r[1::2]
            ctheta2 = resamp_theta[1::2]
            
            ds2 = np.sqrt(dr2**2 + (cr2*dtheta2)**2)
            
            resampled_cRZ = np.array([np.cos(ctheta2), np.sin(ctheta2)]*cr2[np.newaxis,:]) + self.magRZ[:,np.newaxis]
            
            if self.plotting:
                plt.plot(resampled_cRZ[0,:], resampled_cRZ[1,:], marker='.')
            
            self.psicontours[i] = resampled_cRZ
            self.psicontours_ds[i] = ds2
            
            # This equation is abs(del(psi) cross del(toroidal angle))
            br_norm = -self.psinormfunc.ev(resampled_cRZ[0,:], resampled_cRZ[1,:], dy=1)
            bz_norm = self.psinormfunc.ev(resampled_cRZ[0,:], resampled_cRZ[1,:], dx=1)
            bp_norm = np.sqrt(br_norm**2 + bz_norm**2)/resampled_cRZ[0,:]
            self.psicontours_dsbp[i] = ds2/bp_norm
            
    def fs_integrate(self, f):
        """
        Takes in a function of (R,Z,psi) and returns an array of the flux-surface
        integrated function evaluated at psinormEv (e.g. calculating int[f(R,Z,psi(R,Z)) ds/Bp])
        """
        results = np.zeros(len(self.psigridEv))
        
        for i in range(len(self.psigridEv)):
            results[i] = np.sum(f(self.psicontours[i][0,:], self.psicontours[i][1,:], self.psigridEv[i])*self.psicontours_dsbp[i])
        
        return results
        
    
    def fs_average(self, f):
        return self.fs_integrate(f)/self.fs_vprime*2*np.pi
    
    def calculate_ft(self):
        """
        Calculates the trapped particle fraction using Lin-Liu and Miller PoP 1995
        """
        
        # Create a function which is able to evaluate B**2
        ffunc = scipy.interpolate.interp1d(self.psigrid, self.e.getF()[self.tind])
        def b2_func(R, Z, psi):
            bt = ffunc(psi)/R
            br = -self.psifunc.ev(R, Z, dy=1)/R
            bz = self.psifunc.ev(R, Z, dx=1)/R
            
            return bt**2 + br**2 + bz**2
    

        def b_bmax2(R,Z,psi):
            b2 = b2_func(R,Z,psi)
            return b2 / np.max(b2)
            
        def b_bmax(R,Z,psi):
            return np.sqrt(b_bmax2(R,Z,psi))
        
        # Evaluate the flux-surface averaged h^2 and h, as required
        fsa_h2 = self.fs_average(b_bmax2)
        fsa_h = self.fs_average(b_bmax)
        
        # This is the function which gets flux-surface averaged in equation (7)
        def ftl_func(R,Z,psi):
            h = b_bmax(R,Z,psi)
            h2 = b_bmax2(R,Z,psi)
            
            return (1 - (np.sqrt(1 - h) * (1 + 0.5 * h)))/h2
        
        
        # Equation 6, 7 in Lin-Liu
        fs_ftu = 1 - fsa_h2 / fsa_h**2 * (1 - np.sqrt(1 - fsa_h) * (1 + 0.5 * fsa_h))
        fs_ftl = 1 - fsa_h2 * self.fs_average(ftl_func)
        # Equation 18, 19 
        om = 0.75
        self.fs_ft = om*fs_ftu + (1-om)*fs_ftl
    
    def calculate_zeff_neo(self, ne_fit, te_fit, ti_fit, ft_method='lin-liu95', return_funcs=False):
        """
        Estimates the Zeff by matching the neoclassical current to the measured current
        """
        e = self.e
        tb = self.t
        
        # Create a function which is able to evaluate B**2
        ffunc = scipy.interpolate.interp1d(self.psigrid, self.e.getF()[self.tind])
        def b2_func(R, Z, psi):
            bt = ffunc(psi)/R
            br = -self.psifunc.ev(R, Z, dy=1)/R
            bz = self.psifunc.ev(R, Z, dx=1)/R
            
            return bt**2 + br**2 + bz**2
        
        # Calculate some terms which will be useful later
        fsa_b2 = self.fs_average(b2_func)
        fsa_r_2 = self.fs_average(lambda R,Z,psi: R**-2)
        fs_f = ffunc(self.psigridEv)
        
        #fsa_r_2 = np.ones(fsa_r_2.shape) * self.magR ** -2
        
        # Calculate the loop voltage by looking at the change in the poloidal flux        
        rgrid = e.getRGrid()
        zgrid = e.getZGrid()
        
        volts = np.zeros(self.psinormRZ.shape)
        fluxAvg = np.zeros(self.psinormRZ.shape)
        fluxTime = e.getFluxGrid()[self.ti0:self.ti1,:,:]
        for i in range(len(rgrid)):
            for j in range(len(zgrid)):
                # Use a linear fit to smooth the derivative
                s = np.polynomial.polynomial.Polynomial.fit(self.time, fluxTime[:,i,j], 1)
                volts[i,j] = s.convert().coef[1]*2*np.pi
                fluxAvg[i,j] = s.convert().coef[0]
                
        vloop_func = scipy.interpolate.RectBivariateSpline(rgrid, zgrid, volts, kx=2, ky=2)
        
        """
        plt.figure()
        plt.contour(rgrid, zgrid, fluxAvg, 32)
        plt.pcolormesh(rgrid, zgrid, volts)
        plt.colorbar()
        plt.axis('equal')
        """
        
        ipNode = MDSplus.Tree('magnetics', self.shot).getNode(r'\magnetics::ip')
        ip_t0, ip_t1 = np.searchsorted(ipNode.dim_of().data(), (self.t0, self.t1))
        s = np.polynomial.polynomial.Polynomial.fit(ipNode.dim_of().data()[ip_t0:ip_t1], ipNode.data()[ip_t0:ip_t1], 1)
        
        ip = s.convert().coef[0]
        #ip_std = np.std(ipNode.data()[ip_t0:ip_t1])
        dip_dt = s.convert().coef[1]
        
        v_inductive = dip_dt * e.getLi()[self.tind] * 2 * np.pi * self.magR * 1e-7
        #print e.getLi()[self.tind]
        
        # For the time being, ignore the inductive voltage
        fsa_vloop = (self.fs_average(lambda R,Z,psi: vloop_func(R,Z, grid=False)) - v_inductive)
        avg_vloop = np.sum(fsa_vloop*self.fs_vprime*self.dpsigridEv)/np.sum(self.fs_vprime*self.dpsigridEv)
        fsa_vloop = np.ones(fsa_vloop.shape)*avg_vloop
        fsa_ellb = fs_f * fsa_vloop * fsa_r_2 / 2 / np.pi
        
        #print fsa_r_2
        
        # Create interpolating functions for the profiles
        ne_func = scipy.interpolate.interp1d(e.roa2psinorm(ne_fit['X'], tb), ne_fit['y'])
        dne_func = scipy.interpolate.interp1d(e.roa2psinorm(ne_fit['X'], tb), ne_fit['dy_dX'])
        te_func = scipy.interpolate.interp1d(e.roa2psinorm(te_fit['X'], tb), te_fit['y'])
        dte_func = scipy.interpolate.interp1d(e.roa2psinorm(te_fit['X'], tb), te_fit['dy_dX'])
        ti_func = scipy.interpolate.interp1d(e.roa2psinorm(ti_fit['X'], tb), ti_fit['y'])
        dti_func = scipy.interpolate.interp1d(e.roa2psinorm(ti_fit['X'], tb), ti_fit['dy_dX'])
        
        #print e.roa2psinorm(ne_fit['X'][1:], tb)
        roa_func = scipy.interpolate.InterpolatedUnivariateSpline(e.roa2psinorm(ne_fit['X'][2:], tb), ne_fit['X'][2:], k=2)
        # Since derivatives are evaluated in r/a, need to convert it to (unnormalized) derivatives in psi
        droa_dpsi = roa_func(self.psigridEv, nu=1) / self.psiRange
        
        fsa_ne = ne_func(self.psigridEv)*1e20 # density in m^-3
        fsa_te = te_func(self.psigridEv)*1e3 # temperature in eV
        fsa_ti = ti_func(self.psigridEv)*1e3 # temperature in eV
        fsa_dne = dne_func(self.psigridEv)*1e20*droa_dpsi
        fsa_dte = dte_func(self.psigridEv)*1e3*droa_dpsi
        fsa_dti = dti_func(self.psigridEv)*1e3*droa_dpsi
        
        fs_q = scipy.interpolate.interp1d(self.psigrid, e.getQProfile()[self.tind])(self.psigridEv)
        fs_eps = e.getAOut()[self.tind] * roa_func(self.psigridEv) / self.magR
        
        ev2J = 1.60218e-19
        
        #fs_ft = np.sqrt(2*fs_eps)
        if ft_method == 'lin-liu95':
            fs_ft = self.fs_ft
        elif ft_method == 'asymptotic':
            fs_ft = np.sqrt(2*fs_eps)
        elif ft_method == 'mlreinke':
            fs_ft = np.sqrt(fs_eps)

        fs_ft_used = fs_ft            
        
        def jllb_func(z, te_unc=1.0, verbose=False, test_vars={}):
            fs_ft = fs_ft_used
            fsa_p = (fsa_ne * (fsa_te*te_unc) + fsa_ne / z * fsa_ti) * ev2J
            fsa_pe = fsa_ne * ((fsa_te*te_unc)) * ev2J
            fsa_dp = (fsa_ne * (fsa_dte + fsa_dti / z) + fsa_dne * ((fsa_te*te_unc) + fsa_ti / z)) * ev2J
        
            # Equation 18d,e
            lnLe = 31.3 - np.log(np.sqrt(fsa_ne) / (fsa_te*te_unc))
            lnLi = 30 - np.log(z**3 * np.sqrt(fsa_ne/z) / fsa_ti**1.5)
            
            if 'lnLe' in test_vars:
                lnLe = test_vars['lnLe']
                
            # Equsation 18a
            nz = 0.58 + 0.74 / (0.76 + z)
            sig_spitzer = 1.9012e4 * (fsa_te*te_unc)**1.5 / z / nz / lnLe
            
            if 'sig_spitzer' in test_vars:
                sig_spitzer = test_vars['sig_spitzer']
            if 'ft' in test_vars:
                fs_ft = test_vars['ft']
            
            
            nustare = 6.921e-18 * fs_q * self.magR * fsa_ne * z * lnLe / (fsa_te*te_unc)**2 / fs_eps**1.5
            nustari = 4.90e-18 * fs_q * self.magR * (fsa_ne/z) * z**4 * lnLi / fsa_ti**2 / fs_eps**1.5
            
            #print sig_spitzer
        
            # equation 13b    
            f33_teff = fs_ft / (1 + (0.55-0.1*fs_ft)*np.sqrt(nustare) + 0.45*(1-fs_ft)*nustare/z**1.5)
            
            # equation 13a
            f33 = 1 - (1+0.36/z)*f33_teff + 0.59/z*f33_teff**2 - 0.23/z*f33_teff**3
            sig_neo = sig_spitzer*f33
            
            #print z,f33_teff/fs_ft
            
            f31_teff = fs_ft / (1 + (1 - 0.1 * fs_ft)*np.sqrt(nustare) + 0.5*(1-fs_ft)*nustare/z)
            l31 = (1 + 1.4/(z+1))*f31_teff - 1.9/(z+1)*f31_teff**2 + 0.3/(z+1)*f31_teff**3 + 0.2/(z+1)*f31_teff**4
            
            f32_ee_teff = fs_ft / (1 + 0.26*(1-fs_ft)*np.sqrt(nustare) + 0.18*(1-0.37*fs_ft)*nustare/np.sqrt(z))
            f32_ei_teff = fs_ft / (1 + (1+0.6*fs_ft)*np.sqrt(nustare) + 0.85*(1-0.37*fs_ft)*nustare*(1+z))
            
            f32_ee = (0.05+0.62*z)/z/(1+0.44*z)*(f32_ee_teff-f32_ee_teff**4) + 1/(1+0.22*z)*(f32_ee_teff**2-f32_ee_teff**4-1.2*(f32_ee_teff**3-f32_ee_teff**4)) + 1.2/(1+0.5*z)*f32_ee_teff**4
            f32_ei = -(0.56+1.93*z)/z/(1+0.44*z)*(f32_ei_teff-f32_ei_teff**4) + 4.95/(1+2.48*z)*(f32_ei_teff**2-f32_ei_teff**4-0.55*(f32_ei_teff**3-f32_ei_teff**4)) - 1.2/(1+0.5*z)*f32_ei_teff**4
            l32 = f32_ee + f32_ei
            
            f34_teff = fs_ft / (1 + (1-0.1*fs_ft)*np.sqrt(nustare) + 0.5*(1-0.5*fs_ft)*nustare/z)
            l34 = (1 + 1.4/(z+1))*f34_teff - 1.9/(z+1)*f34_teff**2 + 0.3/(z+1)*f34_teff**3 + 0.2/(z+1)*f34_teff**4
            
            alpha0 = -1.17*(1-fs_ft)/(1-0.22*fs_ft-0.19*fs_ft**2)
            alpha = ((alpha0 + 0.25*(1-fs_ft**2)*np.sqrt(nustari)) / (1 + 0.5*np.sqrt(nustari)) - 0.315*nustari**2*fs_ft**6) / (1 + 0.15*nustari**2*fs_ft**6)
            
            j_bs = -fs_f * fsa_pe * (l31 * fsa_p/fsa_pe * fsa_dp/fsa_p + l32 * fsa_dte/(fsa_te*te_unc) + l34 * alpha * fsa_dti/fsa_ti)
            
            
            if verbose:
                print 'max f33_teff', np.max(f33_teff)
                print 'min nustare', np.min(nustare)
                print 'r/a min', roa_func(self.psigridEv[np.argmin(nustare)])
            
            test_vars['sig_neo'] = sig_neo            
            
            return -(sig_neo*fsa_ellb*1.07 + j_bs)
            
        
        def ip_func(z, te_unc=1.0, verbose=False, test_vars={}):
            fsa_dp = (fsa_ne * (fsa_dte + fsa_dti / z) + fsa_dne * ((fsa_te*te_unc) + fsa_ti / z)) * ev2J
            
            fsa_jllb = jllb_func(z, te_unc, verbose, test_vars=test_vars)
            fs_k = fsa_jllb / fsa_b2 - fsa_dp * fs_f / fsa_b2
        
            fsa_jt = (fsa_dp + fs_k * fs_f * fsa_r_2)
            fsa_jt = (fsa_dp + fs_k * fs_f * fsa_r_2)
            
            if (verbose):
                pass
                #print fsa_jt*self.fs_vprime*self.dpsigridEv / 2 / np.pi
            
            return np.sum(fsa_jt*self.fs_vprime*self.dpsigridEv / 2 / np.pi)
        
        if return_funcs:
            return (ip_func, roa_func)
        else:
            z_best = scipy.optimize.ridder(lambda z: ip_func(z) - ip, 0.2, 20.0)
            ip_func(z_best, verbose=True)
            return z_best
