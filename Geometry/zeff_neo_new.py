# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 16:43:01 2019

@author: normandy
"""

import numpy as np
import matplotlib.pyplot as plt

import scipy.interpolate
import scipy.optimize

from skimage import measure

import MDSplus
import eqtools

import cPickle as pkl

# %%




# %%
shot = 1120216017
t0, t1 = 0.67, 0.77

e = eqtools.CModEFITTree(shot)

time = e.getTimeBase()
#tind = np.searchsorted(time, 0.6)
ti0, ti1 = np.searchsorted(time, (t0, t1))
tind = (ti0+ti1)/2
tb = time[tind]
time = time[ti0:ti1]

plotting = False

        

# %% Get the flux contours in the poloidal plane

rgrid = e.getRGrid()
zgrid = e.getZGrid()


magR = e.getMagR()[tind]
magZ = e.getMagZ()[tind]
magRZ = np.array([magR, magZ])

fluxRZ = e.getFluxGrid()[tind,:,:]

psiMin = e.getFluxAxis()[tind]
psiMax = e.getFluxLCFS()[tind]
psiRange = psiMax-psiMin

npsi = len(e.getRmidPsi()[tind])
psigrid = np.linspace(0, 1, npsi)
psinormRZ = (fluxRZ-psiMin)/psiRange

psifunc = scipy.interpolate.RectBivariateSpline(rgrid, zgrid, fluxRZ.T, kx=2, ky=2)
psinormfunc = scipy.interpolate.RectBivariateSpline(rgrid, zgrid, psinormRZ.T, kx=2, ky=2)

rgrid_upscaled = np.linspace(rgrid[0], rgrid[-1], len(rgrid)*4)
zgrid_upscaled = np.linspace(zgrid[0], zgrid[-1], len(zgrid)*8)
rmesh, zmesh = np.meshgrid(rgrid_upscaled, zgrid_upscaled)

rfunc = scipy.interpolate.interp1d(range(len(rgrid_upscaled)), rgrid_upscaled)
zfunc = scipy.interpolate.interp1d(range(len(zgrid_upscaled)), zgrid_upscaled)
psinormRZ_upscaled = psinormfunc.ev(rmesh, zmesh)

psigrid2 = (psigrid[1:] + psigrid[:-1])/2.0

if plotting:
    plt.figure()
    plt.contour(rgrid_upscaled, zgrid_upscaled, psinormRZ_upscaled, psigrid2)
    plt.axis('equal')

#plt.plot(psigrid2, vprime)

# %%

psicontours = [None]*(npsi-1)
# Get the raw contours
for i in range(npsi-1):
    contours = measure.find_contours(psinormRZ_upscaled, psigrid2[i])
    if len(contours) == 1:
        psicontours[i] = np.array([rfunc(contours[0][:,1]), zfunc(contours[0][:,0])])
    elif len(contours) > 1:
        minDist = np.inf
        for c in contours:
            rz = np.array([rfunc(c[:,1]), zfunc(c[:,0])])
            
            if np.linalg.norm(np.mean(rz, axis=1) - magRZ) < minDist:
                psicontours[i] = rz
                minDist = np.linalg.norm(np.mean(rz, axis=1) - magRZ)
    else:
        psicontours[i] = np.array([magRZ]).T

if plotting:
    plt.figure()
# Resample the contours such that the point spacings are of approximately equal arclength

psicontours_ds = [None]*(npsi-1)
psicontours_dsbp = [None]*(npsi-1)

for i in range(npsi-1):
    cRZ = psicontours[i]
    
    ctheta = np.unwrap(np.arctan2(cRZ[1] - magZ, cRZ[0] - magR))
    cr = np.linalg.norm(cRZ - magRZ[:,np.newaxis], axis=0)
    
    dtheta = np.unwrap(np.diff(ctheta))
    dr = np.diff(cr)
    r = (cr[1:] + cr[:-1])/2.0
    
    # Calculate total contour arclength
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
    
    resampled_cRZ = np.array([np.cos(ctheta2), np.sin(ctheta2)]*cr2[np.newaxis,:]) + magRZ[:,np.newaxis]
    
    if plotting:
        plt.plot(resampled_cRZ[0,:], resampled_cRZ[1,:], marker='.')
    
    psicontours[i] = resampled_cRZ
    psicontours_ds[i] = ds2
    
    bp = np.sqrt(psinormfunc.ev(resampled_cRZ[0,:], resampled_cRZ[1,:], dy=1)**2 + psinormfunc.ev(resampled_cRZ[0,:], resampled_cRZ[1,:], dx=1)**2)/resampled_cRZ[0,:]
    psicontours_dsbp[i] = ds2/bp
    
    
    #print dtheta
    
rm, zm = np.meshgrid(rgrid, zgrid)
br = -psifunc.ev(rm, zm, dy=1)/rm
bz = psifunc.ev(rm, zm, dx=1)/rm

#plt.quiver(rm, zm, br, bz)

if plotting:
    plt.axis('equal')
    plt.show()
    
# %% Flux surface average

def fs_integrate(f):
    """
    Takes in a function of (R,Z,psi) and returns a function of (psi) that is flux-
    surface averaged.
    """
    results = np.zeros((npsi-1))
    
    for i in range(npsi-1):
        results[i] = np.sum(f(psicontours[i][0,:], psicontours[i][1,:], psigrid2[i])*psicontours_dsbp[i])
    
    return results
    
fs_volume = fs_integrate(lambda R,Z,psi: 2*np.pi*np.ones(R.shape))

def fs_average(f):
    return fs_integrate(f)/fs_volume*2*np.pi

ffunc = scipy.interpolate.interp1d(psigrid, e.getF()[tind])
def b2(R, Z, psi):
    bt = ffunc(psi)/R
    br = -psifunc.ev(R, Z, dy=1)/R
    bz = psifunc.ev(R, Z, dx=1)/R
    
    return bt**2 + br**2 + bz**2
def r_2(R, Z, psi):
    return 1.0/R**2
    
fsa_b2 = fs_average(b2)
fsa_r_2 = fs_average(r_2)
fs_f = ffunc(psigrid2)

# %% Get voltage

volts = np.zeros(psinormRZ.shape)
fluxAvg = np.zeros(psinormRZ.shape)
fluxTime = e.getFluxGrid()[ti0:ti1,:,:]
for i in range(len(rgrid)):
    for j in range(len(zgrid)):
        s = np.polynomial.polynomial.Polynomial.fit(time, fluxTime[:,i,j], 1)
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

ipNode = MDSplus.Tree('magnetics', shot).getNode(r'\magnetics::ip')
ip_t0, ip_t1 = np.searchsorted(ipNode.dim_of().data(), (t0, t1))
s = np.polynomial.polynomial.Polynomial.fit(ipNode.dim_of().data()[ip_t0:ip_t1], ipNode.data()[ip_t0:ip_t1], 1)

ip = s.convert().coef[0]
dip_dt = s.convert().coef[1]

v_inductive = dip_dt * e.getLi()[tind] * 2 * np.pi * magR * 1e-7

# For the time being, ignore the inductive voltage
fsa_vloop = (fs_average(lambda R,Z,psi: vloop_func(R,Z, grid=False)) - v_inductive)
fsa_ellb = fs_f * fsa_vloop * fsa_r_2 / 2 / np.pi
#rmid2 = (e.getRmidPsi()[tind,1:] + e.getRmidPsi()[tind,:-1])/2
#fsa_ellb = fsa_vloop / 2 / np.pi / magR * -5.8

# %% Trapped particle fractions

fs_bmax = np.zeros((npsi-1))

for i in range(npsi-1):
    fs_bmax = np.sqrt(np.max(b2(psicontours[i][0,:], psicontours[i][1,:], psigrid2[i])))

def b_bmax2(R,Z,psi):
    bt = ffunc(psi)/R
    br = -psifunc.ev(R, Z, dy=1)/R
    bz = psifunc.ev(R, Z, dx=1)/R
    
    b2 = bt**2 + br**2 + bz**2
    return b2 / np.max(b2)
def b_bmax(R,Z,psi):
    return np.sqrt(b_bmax2(R,Z,psi))

fsa_h2 = fs_average(b_bmax2)
fsa_h = fs_average(b_bmax)

def ftl_func(R,Z,psi):
    """
    This is the function inside the angle brackets of equation 7
    """
    h = b_bmax(R,Z,psi)
    h2 = b_bmax2(R,Z,psi)
    
    return (1 - (np.sqrt(1 - h) * (1 + 0.5 * h)))/h2


# Equation 6, 7 in Lin-Liu
fs_ftu = 1 - fsa_h2 / fsa_h**2 * (1 - np.sqrt(1 - fsa_h) * (1 + 0.5 * fsa_h))
fs_ftl = 1 - fsa_h2 * fs_average(ftl_func)
# Equation 18, 19 
om = 0.75
fs_ft = om*fs_ftu + (1-om)*fs_ftl


# %%

ne_fit = pkl.load(file('/home/normandy/git/psfc-misc/Fitting/ne_dict_fit_%d.pkl'%shot))
te_fit = pkl.load(file('/home/normandy/git/psfc-misc/Fitting/te_dict_fit_%d.pkl'%shot))
ti_fit = pkl.load(file('/home/normandy/git/psfc-misc/Fitting/ti_dict_fit_%d.pkl'%shot))

ne_func = scipy.interpolate.interp1d(e.roa2psinorm(ne_fit['X'], tb), ne_fit['y'])
dne_func = scipy.interpolate.interp1d(e.roa2psinorm(ne_fit['X'], tb), ne_fit['dy_dX'])
te_func = scipy.interpolate.interp1d(e.roa2psinorm(te_fit['X'], tb), te_fit['y'])
dte_func = scipy.interpolate.interp1d(e.roa2psinorm(te_fit['X'], tb), te_fit['dy_dX'])
ti_func = scipy.interpolate.interp1d(e.roa2psinorm(ti_fit['X'], tb), ti_fit['y'])
dti_func = scipy.interpolate.interp1d(e.roa2psinorm(ti_fit['X'], tb), ti_fit['dy_dX'])

roa_func = scipy.interpolate.InterpolatedUnivariateSpline(e.roa2psinorm(ne_fit['X'][1:], tb), ne_fit['X'][1:], k=2)
droa_dpsi = roa_func(psigrid2, nu=1) / psiRange

# %%


#plt.plot(fs_ftl)
#plt.plot(fs_ftu)
#plt.plot(fs_ft)
#
#plt.plot(np.sqrt(2*eps))

# %%
fsa_ne = ne_func(psigrid2)*1e20 # density in m^-3
fsa_te = te_func(psigrid2)*1e3 # temperature in eV
fsa_ti = ti_func(psigrid2)*1e3 # temperature in eV
fsa_dne = dne_func(psigrid2)*1e19*droa_dpsi
fsa_dte = dte_func(psigrid2)*1e3*droa_dpsi
fsa_dti = dti_func(psigrid2)*1e3*droa_dpsi

fs_q = scipy.interpolate.interp1d(psigrid, e.getQProfile()[tind])(psigrid2)
fs_eps = e.getAOut()[tind] * roa_func(psigrid2) / magR

ev2J = 1.60218e-19

fsa_p = fsa_ne * (fsa_te + fsa_ti) * ev2J
fsa_pe = fsa_ne * (fsa_te) * ev2J
fsa_dp = (fsa_ne * (fsa_dte + fsa_dti) + fsa_dne * (fsa_te + fsa_ti)) * ev2J

#fs_ft = np.sqrt(2*fs_eps)


def jllb_func(z):
    # Equation 18d,e
    lnLe = 31.3 - np.log(np.sqrt(fsa_ne) / fsa_te)
    lnLi = 30 - np.log(z**3 * np.sqrt(fsa_ne) / fsa_ti**1.5)
    
    # Equsation 18a
    nz = 0.58 + 0.74 / (0.76 + z)
    sig_spitzer = 1.9012e4 * fsa_te**1.5 / z / nz / lnLe
    
    nustare = 6.921e-18 * fs_q * magR * fsa_ne * z * lnLe / fsa_te**2 / fs_eps**1.5
    nustari = 4.90e-18 * fs_q * magR * fsa_ne * z**4 * lnLi / fsa_ti**2 / fs_eps**1.5
    
    #print sig_spitzer

    # equation 13b    
    f33_teff = fs_ft / (1 + (0.55-0.1*fs_ft)*np.sqrt(nustare) + 0.45*(1-fs_ft)*nustare/z**1.5)
    # equation 13a
    f33 = 1 - (1+0.36/z)*f33_teff + 0.59/z*f33_teff**2 - 0.23/z*f33_teff**3
    sig_neo = sig_spitzer*f33
    
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
    
    j_bs = -fs_f * fsa_pe * (l31 * fsa_p/fsa_pe * fsa_dp/fsa_p + l32 * fsa_dte/fsa_te + l34 * alpha * fsa_dti/fsa_ti)
    
    #print j_bs*1e-6
    #print z,sig_neo
    
    return -(sig_neo*fsa_ellb + j_bs)

def ip_func(z):
    fsa_jllb = jllb_func(z)
    fs_k = fsa_jllb / fsa_b2 - fsa_dp * fs_f / fsa_b2

    fsa_jt = (fsa_dp + fs_k * fs_f * fsa_r_2)
    fsa_jt = (fsa_dp + fs_k * fs_f * fsa_r_2)
    
    return np.sum(fsa_jt*fs_volume*np.diff(psigrid) / 2 / np.pi)


print 'calculated Z=' + str(scipy.optimize.ridder(lambda z: ip_func(z) - ip, 0.8, 6.0))
