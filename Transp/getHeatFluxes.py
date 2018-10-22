# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 16:19:44 2017

@author: normandy
"""

import numpy as np
import matplotlib.pyplot as plt

import readline
import MDSplus

import eqtools

# %% open tree

loc=False

if loc:
    transpShot = 89670
    t0 = 0.96
else:
    transpShot = 89398
    t0 = 0.6

ttree = MDSplus.Tree('transp', 89670)

iheatNode = ttree.getNode('\TRANSP::TOP.OUTPUTS.TWO_D:EHEAT')
dvolNode = ttree.getNode('\TRANSP::TOP.OUTPUTS.TWO_D:DVOL')
surfNode = ttree.getNode('\TRANSP::TOP.OUTPUTS.TWO_D:SURF')

pcondNode = ttree.getNode('\TRANSP::TOP.OUTPUTS.TWO_D:PCOND')
pconvNode = ttree.getNode('\TRANSP::TOP.OUTPUTS.TWO_D:PCONV')

pcndeNode = ttree.getNode('\TRANSP::TOP.OUTPUTS.TWO_D:PCNDE')
pcnveNode = ttree.getNode('\TRANSP::TOP.OUTPUTS.TWO_D:PCNVE')

divfdNode = ttree.getNode('\TRANSP::TOP.OUTPUTS.TWO_D:DIVFD')
divfeNode = ttree.getNode('\TRANSP::TOP.OUTPUTS.TWO_D:DIVFE')

iheat = iheatNode.data()
dvol = dvolNode.data()
surf = surfNode.data()

pcond = pcondNode.data()
pconv = pconvNode.data()

pcnde = pcndeNode.data()
pcnve = pcnveNode.data()

divfd = divfdNode.data()
divfe = divfeNode.data()

rad = iheatNode.dim_of(0).data()
time = iheatNode.dim_of(1).data()

# %% equilibrium, gyro-bohm normalizations

e = eqtools.CModEFITTree(1160506007)

niNode = ttree.getNode('\TRANSP::TOP.OUTPUTS.TWO_D:ND') # in cm^-3
tiNode = ttree.getNode('\TRANSP::TOP.OUTPUTS.TWO_D:TI') # in eV
rmin = 21.8 # minor radius in cm
mnorm = 1.8765e9 # mass of deuterium in eV
c = 3e10 # speed of light in cm/s
eVtoJ = 1.60218e-19 # Joules per eV

neo_vnorm_raw = np.sqrt(tiNode.data() / mnorm) * c # vti in cm/s
neo_fnorm_raw = neo_vnorm_raw * niNode.data() # 1/cm^2/s
neo_qnorm_raw = neo_fnorm_raw * tiNode.data() * eVtoJ # 

teNode = ttree.getNode('\TRANSP::TOP.OUTPUTS.TWO_D:TE')
neNode = ttree.getNode('\TRANSP::TOP.OUTPUTS.TWO_D:NE') # in cm^-3

btnorm = 5.3252832 # I think this is Bunit? idk in T
omcinorm = 47.894e6 # ion gyrofrequency for 1 T field in Hz

cs_raw = np.sqrt(teNode.data() / mnorm) * c
rhos_raw = cs_raw / (omcinorm * btnorm) # ion sound gyroradius in cm
gbsq = (rhos_raw / rmin) ** 2 # Square gyrobohm normalizing unit

gyro_fnorm_raw = neNode.data() * cs_raw * gbsq * 1e4 / 1e19 # bring to e19 /m^2/s
gyro_qnorm_raw = neNode.data() * cs_raw * teNode.data() * eVtoJ * gbsq * 1e4 # bring to W/m^2
gyro_pnorm_raw = neNode.data() * 0.218 * teNode.data() * eVtoJ * gbsq * 1e4 # bring to N/m

# TODO: figure out gyro-bohm normalization

te_raw = teNode.data()* eVtoJ*1e19 # te used for conversion

# %% plot data

def timeAverage(data):
    minInd, maxInd = np.searchsorted(time, (t0-0.03, t0+0.03))
    return np.average(data[minInd:maxInd,:40], axis=0)

def calcFlux(div):
    return np.cumsum(timeAverage(div*dvol)) / timeAverage(surf)


#vnorm = timeAverage(vnorm_raw)
fnorm = timeAverage(gyro_fnorm_raw)
qnorm = timeAverage(gyro_qnorm_raw)
pnorm = timeAverage(gyro_pnorm_raw)

qitot = calcFlux(pcond+pconv)*1e4 # bring to W/m^2
qetot = calcFlux(pcnde+pcnve)*1e4 # bring to W/m^2
fetot = calcFlux(divfe)*1e4/1e19 # bring to e19 /m^2/s


plt.figure()

roa = e.phinorm2roa(timeAverage(rad)**2, t0)
r0, r1 = np.searchsorted(roa, [0.35, 0.8])

def clipRadius(data):
    return data[r0:r1]

plt.plot(clipRadius(roa), clipRadius(fetot/fnorm))
plt.plot(clipRadius(roa), clipRadius(qetot/qnorm))
plt.plot(clipRadius(roa), clipRadius(qitot/qnorm))

#plt.plot(roa, timeAverage(rhos_raw))

#plt.plot(roa, fnorm)

# %% Load neoclassical data

loc_mid_neo = np.loadtxt('/home/normandy/git/psfc-misc/Transp/loc_mid.out.neo.transport_exp')
soc_mid_neo = np.loadtxt('/home/normandy/git/psfc-misc/Transp/soc_mid.out.neo.transport_exp')

if loc:
    mid_neo = loc_mid_neo
else:
    mid_neo = soc_mid_neo

neo_roa = mid_neo[:,0]/0.218
neo_fetot = mid_neo[:,5+8*3]
neo_qetot = mid_neo[:,6+8*3]
neo_qitot = mid_neo[:,6+8*0]
neo_stresstot = mid_neo[:,7+8*2]



qi_anom = (np.interp(neo_roa, roa, qitot) - neo_qitot)
qe_anom = (np.interp(neo_roa, roa, qetot) - neo_qetot)
fe_anom = (np.interp(neo_roa, roa, fetot) - neo_fetot)

qinorm = np.interp(0.575, roa, qnorm)
qenorm = np.interp(0.575, roa, qnorm)
fenorm = np.interp(0.575, roa, fnorm)
pinorm = np.interp(0.575, roa, pnorm)


plt.figure()
plt.plot(neo_roa[0:], qi_anom[0:]/qinorm, marker='.', label='Qi')
plt.plot(neo_roa[0:], qe_anom[0:]/qenorm, marker='.', label='Qe')
plt.plot(neo_roa[0:], fe_anom[0:]/fenorm, marker='.', label='$\Gamma$e')
#plt.plot(neo_roa[0:], neo_stresstot/pinorm, marker='.', label='$\Pi$i')
plt.axhline(ls='--', c='k')
plt.xlabel('r/a')
plt.ylabel('anomalous flux [GB units, r/a=0.575]')
plt.legend()

# %%

plt.figure()
ax = plt.subplot(111)
pos = np.array([2.5, 1.5, 0.5])
vals = [qi_anom[3]/qinorm, qe_anom[3]/qenorm, fe_anom[3]/fenorm]
ax.barh(pos, vals, height=1.0, color=('b', 'g', 'r'), tick_label=('Qi','Qe','$\Gamma$e'), align='center')
ax.axvline(ls='--', c='k')
ax.set_xlim([-0.1, 2.0])

ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')

"""
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.yaxis.set_ticks_position('none')
ax.xaxis.set_ticks_position('none')

ax.xaxis.set_ticks([])
ax.yaxis.set_ticks([])
"""

# %% Plot shear and safety factor

etime = e.getTimeBase()
eind = np.searchsorted(etime, t0)

safety = e.getQProfile()[eind,:]
rmid_safety = e.getRmidPsi()[eind,:]
roa_safety = e.rmid2roa(rmid_safety, t0)

shear = np.ediff1d(np.log(safety)) / np.ediff1d(np.log(roa_safety))
roa_shear = (roa_safety[1:] + roa_safety[:-1])/2.0

plt.figure()

plt.plot(roa_safety, safety)