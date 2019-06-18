# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 17:08:20 2019

@author: maple
"""

import numpy as np
import scipy.special
import scipy.optimize
import scipy.integrate
import matplotlib.pyplot as plt

# %%


def dens(z, za, na, zd, tau):
    first = -2*(2.0*za + na * z * zd)
    second = -1j * np.sqrt(2 * np.pi) * (2*z*za + (2-na) * zd + na * z**2 * zd)
    return tau*(first + second*scipy.special.wofz(z/np.sqrt(2.0)))/4

def densprime(z, za, na, zd, tau):
    first = 4 * z *zd - 4*(na - 1)*zd + 2 * na * z**2 * zd
    second = 1j * np.sqrt(2 * np.pi) * (2 * (z**2 - 1) * za + z * (2 + na * (z**2 - 3)) * zd)
    return tau*(first + second*scipy.special.wofz(z/np.sqrt(2.0)))/4
    
def densprime2(z, za, na, zd, tau):
    first = -2 * (2 * (z**2 - 2) * za + z * (2 + na * (z**2 - 5)) * zd)
    second = -1j * np.sqrt(2.0 * np.pi) * (-6 * z * za + 2 * z**3 * za + (3 * na - 2) * zd + (2 - 6*na) * z**2 * zd + na * z**4 * zd)
    return tau*(first + second*scipy.special.wofz(z/np.sqrt(2.0)))/4
    
def dispf(ma, za, na, zd, tau):
    return lambda z: np.sum([dens(z*ma[i], za[i], na[i], zd[i], tau[i]) * za[i] for i in range(len(za))], axis=0)

def dispfprime(ma, za, na, zd, tau):
    return lambda z: np.sum([densprime(z*ma[i], za[i], na[i], zd[i], tau[i]) * za[i] for i in range(len(za))], axis=0)

def dispfprime2(ma, za, na, zd, tau):
    return lambda z: np.sum([densprime2(z*ma[i], za[i], na[i], zd[i], tau[i]) * za[i] for i in range(len(za))], axis=0)

# %%



x = np.linspace(-7, 7, num=255)
y = np.linspace(-7, 7, num=255)
xx, yy = np.meshgrid(x,y)
zz = xx + 1j * yy

#ww = -(zz + 1) / (zz - 1)
#ww = 1.0/zz
ww = zz

f, axes = plt.subplots(2, 3, sharex=True, sharey=True)

def plotdisp(na, ax):
    dispfv = dispf([1.0, 1.0/60.0], [1.0, -1.0], [na, na], [5.0, -5.0/60.0], [1.0, 1.0])
    disp = dispfv(ww)
    ax.contour(xx, yy, np.real(disp), levels=[0], colors='blue')
    ax.contour(xx, yy, np.imag(disp), levels=[0], colors='red')
    #ax.contour(xx, yy, np.real(ww), levels=[10], colors='green')
    ax.axhline(ls='--', c='k')

plotdisp(1.5, axes[0,0])
plotdisp(1.7, axes[0,1])
plotdisp(2.0, axes[0,2])
plotdisp(2.1, axes[1,0])
plotdisp(2.3, axes[1,1])
plotdisp(6.0, axes[1,2])

#plt.contour(xx, yy, np.imag(ww), levels=np.logspace(-1,1,num=5), colors='black')

#plt.axvline(1.0, ls='--', c='k')
#plt.axvline(-1.0, ls='--', c='k')
#plt.axis('equal')

# %%

guess0 = -0.33 + 0.02j
guess1 = 5.8 -0.629j
guess2 = -375.847-1.8833j

guesses = [guess0, guess1, guess2]


naup = np.arange(1.8, 8.0, 0.2)
rootup = np.zeros((len(naup), len(guesses)), dtype=np.complex)
rootup[0,:] = guesses

j = 0

for i in range(len(naup)):
    na = naup[i]
    dispf0 = dispf([1.0, 1.0/60.0], [1.0, -1.0], [na, na], [5.0, -5.0/60.0], [1.0, 1.0])
    
    fmin = lambda x: np.abs(dispf0(x[0]+1j*x[1]))**2
    
    for j in range(len(guesses)):
        if i==0:
            res = scipy.optimize.minimize(fmin, x0=np.array([np.real(rootup[i,j]), np.imag(rootup[i,j])]))
        else:
            res = scipy.optimize.minimize(fmin, x0=np.array([np.real(rootup[i-1,j]), np.imag(rootup[i-1,j])]))
        rootup[i,j] = res.x[0] + 1j*res.x[1]
    
    """
    dispf0prime = dispfprime([1.0, 1.0/60.0], [1.0, -1.0], [na, na], [5.0, -5.0], [1.0, 1.0])
    dispf0prime2 = dispfprime2([1.0, 1.0/60.0], [1.0, -1.0], [na, na], [5.0, -5.0], [1.0, 1.0])
    #for j in range(3):
    if i==0:
        rootup[i,j] = scipy.optimize.newton(dispf0, x0=rootup[i,j], fprime=dispf0prime, fprime2=dispf0prime2, tol=np.abs(rootup[i,j])*1e-4)
    else:
        rootup[i,j] = scipy.optimize.newton(dispf0, x0=rootup[i-1,j], fprime=dispf0prime, fprime2=dispf0prime2, tol=np.abs(rootup[i-1,j])*1e-4)
    """
    
# %%
for j in range(len(guesses)):
    plt.scatter(np.real(rootup[:,j]), np.imag(rootup[:,j]), c=naup, cmap='cubehelix')
plt.axhline(c='k', ls='--')
plt.colorbar()

# %%

def responsef(v, m, z, za, na, zd, tau):
    return (zd * (1 + (v**2/2 + m - 1.5)*na) + v * za) * tau / (z - v) * np.exp(-v**2 / 2.0 -m) / np.sqrt(2.0 * np.pi)
    
def resonantIntegrate(f, z):
    fre = lambda v, m: np.real(f(v, m))
    fim = lambda v, m: np.imag(f(v, m))
    
    re = scipy.integrate.dblquad(fre, 0.0, 25.0, lambda x: -25.0, lambda x: z-0.3, epsabs=1e-4, epsrel=1e-4)[0] + \
         scipy.integrate.dblquad(fre, 0.0, 25.0, lambda x: z-0.3, lambda x: z+0.3, epsabs=1e-4, epsrel=1e-4)[0] + \
         scipy.integrate.dblquad(fre, 0.0, 25.0, lambda x: z+0.3, lambda x: 25.0, epsabs=1e-4, epsrel=1e-4)[0]
    im = scipy.integrate.dblquad(fim, 0.0, 25.0, lambda x: -25.0, lambda x: z-0.3, epsabs=1e-4, epsrel=1e-4)[0] + \
         scipy.integrate.dblquad(fim, 0.0, 25.0, lambda x: z-0.3, lambda x: z+0.3, epsabs=1e-4, epsrel=1e-4)[0] + \
         scipy.integrate.dblquad(fim, 0.0, 25.0, lambda x: z+0.3, lambda x: 25.0, epsabs=1e-4, epsrel=1e-4)[0]
    return re + 1j * im

def densf(z, za, na, zd, tau):
    return resonantIntegrate(lambda v, m: responsef(v,m,z,za,na,zd,tau), z)


def tempvf(z, za, na, zd, tau):
    return resonantIntegrate(lambda v, m: responsef(v,m,z,za,na,zd,tau) * (v**2/2.0 - 0.5), z)
    
def tempmf(z, za, na, zd, tau):
    return resonantIntegrate(lambda v, m: responsef(v,m,z,za,na,zd,tau) * (m - 1.0), z)
    
def vf(z, za, na, zd, tau):
    return resonantIntegrate(lambda v, m: responsef(v,m,z,za,na,zd,tau) * v, z)

# %%

irn = np.zeros(rootup.shape, dtype=np.complex)
irtv = np.zeros(rootup.shape, dtype=np.complex)
irtm = np.zeros(rootup.shape, dtype=np.complex)
irv = np.zeros(rootup.shape, dtype=np.complex)

ertv = np.zeros(rootup.shape, dtype=np.complex)
ertm = np.zeros(rootup.shape, dtype=np.complex)
erv = np.zeros(rootup.shape, dtype=np.complex)

for i in range(len(naup)):
    for j in [1]:
        print i,j
        irn[i,j] = densf(rootup[i,j], 1.0, naup[i], 5.0, 1.0)
        irtv[i,j] = tempvf(rootup[i,j], 1.0, naup[i], 5.0, 1.0)
        irtm[i,j] = tempmf(rootup[i,j], 1.0, naup[i], 5.0, 1.0)
        irv[i,j] = vf(rootup[i,j], 1.0, naup[i], 5.0, 1.0)
        
        ertv[i,j] = tempvf(rootup[i,j]/60.0, -1.0, naup[i], -5.0/60.0, 1.0)
        ertm[i,j] = tempmf(rootup[i,j]/60.0, -1.0, naup[i], -5.0/60.0, 1.0)
        erv[i,j] = vf(rootup[i,j]/60.0, -1.0, naup[i], -5.0/60.0, 1.0)

# %%

j = 1
#plt.scatter(np.imag(rootup[:,j]), -np.imag(irn[:,j]), c=naup, cmap='cubehelix')
#plt.scatter(np.real(irn[:,j]), np.imag(irn[:,j]), c=naup, cmap='cubehelix')
#plt.scatter(np.real((irtv[:,j]+irn[:,j])/irv[:,j]), np.imag((irtv[:,j]+irn[:,j])/irv[:,j]), c=naup, cmap='cubehelix')
#plt.scatter(np.real(irv[:,j]), np.imag(irv[:,j]), c=naup, cmap='cubehelix')
plt.scatter(np.imag(rootup[:,j]), -np.imag(irtm[:,j]+irtv[:,j])-np.imag(ertm[:,j]+ertv[:,j]), c=naup, cmap='cubehelix')
#plt.scatter(np.imag(rootup[:,j]), -np.imag(ertm[:,j]+ertv[:,j]), c=naup, cmap='cubehelix')
plt.axhline(c='k', ls='--')
plt.axvline(c='k', ls='--')
plt.colorbar()

