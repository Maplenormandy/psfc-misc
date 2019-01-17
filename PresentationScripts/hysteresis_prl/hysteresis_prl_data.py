# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 13:52:05 2018

@author: normandy
"""

import numpy as np
import scipy.io, scipy.signal

import MDSplus

import eqtools

# %% General function definitions

class ThacoData:
    def __init__(self, thtNode, shot=None, tht=None, path='.HELIKE.PROFILES.Z', time=0.95):
        if (thtNode == None):
            self.shot = shot
            self.specTree = MDSplus.Tree('spectroscopy', shot)

            if (tht == 0):
                self.tht = ''
            else:
                self.tht = str(tht)

            self.thtNode = self.specTree.getNode('\SPECTROSCOPY::TOP.HIREXSR.ANALYSIS' + self.tht + path)
        else:
            self.thtNode = thtNode

        e = eqtools.CModEFITTree(shot)

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
        self.roa = e.psinorm2roa(self.rho, time)
        self.pro = rpro[:,:goodTimes,:len(self.rho)]
        self.perr = rperr[:,:goodTimes,:len(self.rho)]

def trimNodeData(node, t0=0.5, t1=1.5):
    time = node.dim_of().data()
    data = node.data()
    i0, i1 = np.searchsorted(time, (t0, t1))
    return time[i0:i1+1], data[i0:i1+1]
def trimData(time, data, t0=0.5, t1=1.5):
    i0, i1 = np.searchsorted(time, (t0, t1))
    return time[i0:i1+1], data[i0:i1+1]

# %% Data loading
elecTree = MDSplus.Tree('electrons', 1160506007)
specTree = MDSplus.Tree('spectroscopy', 1160506007)

nl04time, nl04data = trimNodeData(elecTree.getNode(r'\ELECTRONS::TOP.TCI.RESULTS:NL_04'))
vnode = specTree.getNode(r'\SPECTROSCOPY::TOP.HIREX_SR.ANALYSIS.W:VEL')
vtime, vdata = trimData(vnode.dim_of().data(), vnode.data()[0])

# %%

tifit_soc = scipy.io.readsav('/home/normandy/fits/tifit_1120106012_THT0.dat')
omfit_soc = scipy.io.readsav('/home/normandy/fits/omfit_1120106012_THT0.dat')
tifit_loc = scipy.io.readsav('/home/normandy/fits/tifit_1120106016_THT0.dat')
omfit_loc = scipy.io.readsav('/home/normandy/fits/omfit_1120106016_THT0.dat')

thacodata_soc = ThacoData(None, 1120106012, 0)
thacodata_loc = ThacoData(None, 1120106016, 0)


# %% Figure 1: time traces and rotation profiles\

np.savez('figure1_1.npz', nl04time=nl04time, nl04data=nl04data, vtime=vtime, vdata=vdata)

time_index = 20
radial_index = 20

bsom_soc = omfit_soc.bsom[time_index]['fit'][0]
bsom_loc = omfit_loc.bsom[time_index]['fit'][0]

fig1_2 = {}
fig1_2['soc_roa'] = thacodata_soc.roa[:radial_index]
fig1_2['soc_roa'] = thacodata_soc.pro[1,time_index,:radial_index]
fig1_2['soc_roa'] = thacodata_soc.perr[1,time_index,:radial_index]
fig1_2['soc_roa'] = thacodata_loc.roa[:radial_index]
fig1_2['soc_roa'] = thacodata_soc.pro[1,time_index,:radial_index]
fig1_2['soc_roa'] = thacodata_soc.perr[1,time_index,:radial_index]

np.savez('figure1_2.npz', **fig1_2)

# %% Figure 2: Hysteresis plot

def plotHysteresis(shot):
    elecTree = MDSplus.Tree('electrons', shot)

    nl04Node = elecTree.getNode('\ELECTRONS::TOP.TCI.RESULTS:NL_04')
    specTree = MDSplus.Tree('spectroscopy', shot)
    velNode = specTree.getNode(r'\SPECTROSCOPY::TOP.HIREX_SR.ANALYSIS.W:VEL')

    vtime = velNode.dim_of().data()
    nltime = nl04Node.dim_of().data()


    vlow = np.searchsorted(vtime, 0.55)
    vhigh = np.searchsorted(vtime, 1.25)+2

    vtime = vtime[vlow:vhigh]
    vdata = velNode.data()[0]
    vdata = vdata[vlow:vhigh]

    nlData = np.interp(vtime, nltime, nl04Node.data())/1e20/0.6
    offset = ((shot%100)-4.0)/(25.0-7.0)*6

    return np.array([nlData, vdata+offset])

hys01 = plotHysteresis(1160506007)
hys02 = plotHysteresis(1160506008)
hys03 = plotHysteresis(1160506024)


# %% Figure 3: Profile matched plots


time_index = 20
radial_index = 25

bsti_soc = tifit_soc.bsti[time_index][0][0][0]
bsti_loc = tifit_loc.bsti[time_index][0][0][0]

alti_soc = -bsti_soc['dprof']/(bsti_soc['prof']-0.12)
alti_soc_err = np.sqrt(alti_soc**2 * (bsti_soc['err']**2 / (bsti_soc['prof']-0.12)**2 + bsti_soc['derr']**2/bsti_soc['dprof']**2))
alti_loc = -bsti_loc['dprof']/(bsti_loc['prof']-0.12)
alti_loc_err = np.sqrt(alti_loc**2 * (bsti_loc['err']**2 / (bsti_loc['prof']-0.12)**2 + bsti_loc['derr']**2/bsti_loc['dprof']**2))

fig3_2 = {}
fig3_2['loc_rho'] = bsti_loc['rho'][::2]
fig3_2['loc_prof'] = bsti_loc['prof'][::2]-0.12
fig3_2['loc_err'] = bsti_loc['err'][::2]
fig3_2['alti_loc'] = alti_loc[::2]
fig3_2['alti_loc_err'] = alti_loc_err[::2]
fig3_2['soc_rho'] = bsti_soc['rho'][::2]
fig3_2['soc_prof'] = bsti_soc['prof'][::2]-0.12
fig3_2['soc_err'] = bsti_soc['err'][::2]
fig3_2['alti_soc'] = alti_soc[::2]
fig3_2['alti_soc_err'] = alti_soc_err[::2]

np.savez('figure3_2.npy', **fig3_2)

# %% Figure 4: Reflecometer plots

elecTree = MDSplus.Tree('electrons', 1160506007)

# 9 and 10 are normally the best
sig88ui = elecTree.getNode('\ELECTRONS::TOP.REFLECT.CPCI:DT132_1:INPUT_09').data()
sig88uq = elecTree.getNode('\ELECTRONS::TOP.REFLECT.CPCI:DT132_1:INPUT_10').data()
sig88ut = elecTree.getNode('\ELECTRONS::TOP.REFLECT.CPCI:DT132_1:INPUT_09').dim_of().data()

ci = np.mean(sig88ui)
cq = np.mean(sig88uq)



t1, t2 = np.searchsorted(sig88ut, (0.4,1.6))
#0.5944-0.5954
#0.9625-0.9650

si = sig88ui[t1:t2]
sq = sig88uq[t1:t2]
st = sig88ut[t1:t2]
ci = np.median(si)
cq = np.median(sq)
z = (si-ci)+1j*(sq-cq)


# %%
t_soc, t_loc = np.searchsorted(st, (0.5949, 0.9637))

total_samples=4096*2
down_samples =512

z_soc = z[t_soc-total_samples/2:t_soc+total_samples/2]
z_loc = z[t_loc-total_samples/2:t_loc+total_samples/2]

fz_soc = np.fft.fftshift(np.fft.fft(z_soc))/2e3/2e3
fz_loc = np.fft.fftshift(np.fft.fft(z_loc))/2e3/2e3
freqs = np.fft.fftshift(np.fft.fftfreq(total_samples, 1.0/2e6))


Sxx_loc_down = np.real(np.average(np.reshape(fz_loc*np.conjugate(fz_loc), (down_samples, -1)), axis=-1))
Sxx_soc_down = np.real(np.average(np.reshape(fz_soc*np.conjugate(fz_soc), (down_samples, -1)), axis=-1))

f_down = np.average(np.reshape(freqs/1e3, (down_samples, -1)), axis=-1)

np.savez('figure4.npz', f_down=f_down, Sxx_loc_down=Sxx_loc_down, Sxx_soc_down=Sxx_soc_down)

# %% Figure 5: Growth rates / real frequencies.
# Note: The data here is generated from ~/hys2/plot_all_freqs.py

# %% Figure 6: QL weights - need to go on engaging, also what to do with flux data?

