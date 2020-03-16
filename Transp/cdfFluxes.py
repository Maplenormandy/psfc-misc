# -*- coding: utf-8 -*-
"""
Created on Thu May  4 10:18:35 2017

@author: normandy
"""


import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import matplotlib as mpl
#from matplotlib import cm

# %% open tree

#transp = scipy.io.netcdf.netcdf_file('/home/pablorf/Cao_transp/12345A05/12345A05.CDF')
transp = scipy.io.netcdf.netcdf_file('/home/pablorf/Cao_transp/12345B05/12345B05.CDF')
neo = np.loadtxt('/home/normandy/git/psfc-misc/Transp/out.neo.transport_exp.030')

rmid = transp.variables['RMNMP'].data
time = transp.variables['TIME'].data

t0 = 0.95
tind = np.searchsorted(time, t0)
aout = rmid[tind,-1] # Minor radius in cm
roa = rmid[tind,:]/aout

r0, r1 = np.searchsorted(roa, [0.45, 0.65])
rind = np.searchsorted(roa, 0.6)

surf = transp.variables['SURF'].data
dvol = transp.variables['DVOL'].data

pcond = transp.variables['PCOND'].data
pconv = transp.variables['PCONV'].data
pcnde = transp.variables['PCNDE'].data
pcnve = transp.variables['PCNVE'].data
divfe = transp.variables['DIVFE'].data

ne = transp.variables['NE'].data
ti = transp.variables['TI'].data
te = transp.variables['TE'].data

mnorm = 1.8765e9 # mass of deuterium in eV
c = 3e10 # speed of light in cm/s
eVtoJ = 1.60218e-19 # Joules per eV

btnorm = 5.3252832 # I think this is Bunit? idk in T
omcinorm = 47.894e6 # ion gyrofrequency for 1 T field in Hz

cs = np.sqrt(te / mnorm) * c # Sound speed in cm/s
rhos = cs / (omcinorm * btnorm) # ion sound gyroradius in cm
gbsq = (rhos / aout) ** 2 # Square gyrobohm normalizing unit


#t0 = 1.25

def timeAverage(data):
    minInd, maxInd = np.searchsorted(time, (t0-0.03, t0+0.03))
    return np.average(data[minInd:maxInd,:], axis=0)

def calcFlux(div):
    return np.cumsum(timeAverage(div*dvol)) / timeAverage(surf)

fnorm = timeAverage(ne * cs * gbsq) * 1e4 / 1e19 # Particle flux normalization in e19 /m^2/s
qnorm = timeAverage(ne * cs * te * eVtoJ * gbsq) * 1e4 # Heat flux normalization in W/m^2

qitot = calcFlux(pcond+pconv)*1e4 # bring to W/m^2
qetot = calcFlux(pcnde+pcnve)*1e4 # bring to W/m^2
fetot = calcFlux(divfe)*1e4/1e19 # bring to e19 /m^2/s

neo_roa = neo[:,0]/aout*100.0
neo_fe = neo[:,5+8*3]
neo_qe = neo[:,6+8*3]
neo_qi = neo[:,6+8*0]

qi_anom = (np.interp(neo_roa, roa, qitot) - neo_qi) / np.interp(neo_roa, roa, qnorm)
qe_anom = (np.interp(neo_roa, roa, qetot) - neo_qe) / np.interp(neo_roa, roa, qnorm)
fe_anom = (np.interp(neo_roa, roa, fetot) - neo_fe) / np.interp(neo_roa, roa, fnorm)


def clipRadius(data):
    return data[r0:r1]


plt.plot(neo_roa, qi_anom, marker='.', c=mpl.cm.PiYG(0.0), label='Qi')
plt.plot(neo_roa, qe_anom, marker='.', c=mpl.cm.PiYG(1.0), label='Qe')
plt.plot(neo_roa, fe_anom, marker='.', c=(0.5,0.0,1.0), label=r'$\Gamma$e')

"""
plt.plot(clipRadius(roa[tind,:]), clipRadius(qitot/qnorm[tind,:]), marker='.', c=mpl.cm.PiYG(0.0), label='Qi')
plt.plot(clipRadius(roa[tind,:]), clipRadius(qetot/qnorm[tind,:]), marker='.', c=mpl.cm.PiYG(1.0), label='Qe')
plt.plot(clipRadius(roa[tind,:]), clipRadius(fetot/fnorm[tind,:]), marker='.', c=(0.5,0.0,1.0), label=r'$\Gamma$e')
"""

tosave = {
        'roa': neo_roa,
        'qi_anom': qi_anom,
        'qe_anom': qe_anom,
        'fe_anom': fe_anom
        }
np.savez('/home/normandy/git/psfc-misc/PresentationScripts/hysteresis_pop/fluxes_030.npz', **tosave)

plt.legend()
plt.show()
