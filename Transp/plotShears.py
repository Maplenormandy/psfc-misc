# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 15:59:30 2019

@author: normandy
"""

import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import cPickle as pkl
import eqtools

# %%


shot=1120216030
result = pkl.load(open('/home/normandy/git/psfc-misc/Fitting/result_%d.pkl'%shot, 'r'))

e = eqtools.CModEFITTree(shot)

if shot==1120216030:
    transp = scipy.io.netcdf.netcdf_file('/home/pablorf/Cao_transp/12345B05/12345B05.CDF')
elif shot==1120216017:
    transp = scipy.io.netcdf.netcdf_file('/home/pablorf/Cao_transp/12345A05/12345A05.CDF')
    
# %%

mD = 1876544.16 # mass of deuterium in keV
c = 299792458

teq = e.getTimeBase()
    
plt.figure()
mean_grad = np.mean(np.array([result['vtor_fits'][j]['dy_dX'] for j in range(11,13)]), axis=0)
for i in range(9,16):
    c_s = np.sqrt(result['te_fits'][i]['y'] / mD) * c
    a_over_c_s = e.getAOut()[np.searchsorted(teq, result['time'][i])] / c_s
    
    if result['vtor_fits'][i]['y'][1] < 0:
        plt.errorbar(result['vtor_fits'][i]['X'], result['vtor_fits'][i]['dy_dX']*a_over_c_s*1e3, result['vtor_fits'][i]['err_dy_dX']*a_over_c_s*1e3, c='g')
        #plt.plot(result['vtor_fits'][i]['X'], result['vtor_fits'][i]['y'], c='b')
    else:
        plt.errorbar(result['vtor_fits'][i]['X'], result['vtor_fits'][i]['dy_dX']*a_over_c_s*1e3, result['vtor_fits'][i]['err_dy_dX']*a_over_c_s*1e3, c='y')
        #plt.plot(result['vtor_fits'][i]['X'], result['vtor_fits'][i]['y'], c='r')