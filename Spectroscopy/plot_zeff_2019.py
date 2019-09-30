
import numpy as np

import matplotlib.pyplot as plt
import matplotlib

import scipy.io

zeffs = scipy.io.readsav('/home/normandy/git/psfc-misc/Spectroscopy/output.sav')

#print zeffs['zeff_array'].shape

plt.plot(zeffs['time_array'][0,:], zeffs['zeff_qfit_array'][0,:], label='1.1 MA')
plt.plot(zeffs['time_array'][1,:], zeffs['zeff_qfit_array'][1,:], label='0.8 MA')
plt.show()
