# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 18:47:09 2018

@author: normandy
"""

import numpy as np
import profiletools

import matplotlib.pyplot as plt
import matplotlib as mpl

# %%

soc_ne = profiletools.ne(1160506007, abscissa='r/a', include=['CTS', 'ETS'], t_min = 0.58, t_max=0.62)
soc_ne.drop_axis(0)
soc_ne.create_gp()
soc_ne.remove_outliers()
soc_ne.find_gp_MAP_estimate()


loc_ne = profiletools.ne(1160506007, abscissa='r/a', include=['CTS', 'ETS'], t_min = 0.94, t_max=0.98)
loc_ne.drop_axis(0)
loc_ne.create_gp()
loc_ne.remove_outliers()
loc_ne.find_gp_MAP_estimate()

# %%

loc_Te = profiletools.Te(1160506007, abscissa='r/a', include=['GPC', 'GPC2'], t_min = 0.9537, t_max=0.9737)
loc_Te.time_average(weighted=True)
loc_Te.create_gp()
#loc_Te.remove_outliers()
loc_Te.find_gp_MAP_estimate()

soc_Te = profiletools.Te(1160506007, abscissa='r/a', include=['GPC', 'GPC2'], t_min = 0.5849, t_max=0.6049)
soc_Te.time_average(weighted=True)
soc_Te.create_gp()
#loc_Te.remove_outliers()
soc_Te.find_gp_MAP_estimate()

# %%

r = np.linspace(0.0, 1.0, 24)



ne_loc, ne_std_loc = loc_ne.smooth(r, 0)
ne_soc, ne_std_soc = soc_ne.smooth(r, 0)

Te_loc, Te_std_loc = loc_Te.smooth(r, 0)
Te_soc, Te_std_soc = soc_Te.smooth(r, 0)

aolne_loc, aolne_std_loc = loc_ne.compute_a_over_L(r)
aolne_soc, aolne_std_soc = soc_ne.compute_a_over_L(r)

aolTe_loc, aolTe_std_loc = loc_Te.compute_a_over_L(r)
aolTe_soc, aolTe_std_soc = soc_Te.compute_a_over_L(r)

arrays1 = [r, ne_loc, ne_std_loc, ne_soc, ne_std_soc, aolne_loc, aolne_std_loc, aolne_soc, aolne_std_soc, Te_loc, Te_std_loc, Te_soc, Te_std_soc, aolTe_loc, aolTe_std_loc, aolTe_soc, aolTe_std_soc]
arrays2a = [loc_ne.X, loc_ne.y, loc_ne.err_y]
arrays2b = [soc_ne.X, soc_ne.y, soc_ne.err_y]
arrays2fa = [np.ravel(a) for a in arrays2a]
arrays2fb = [np.ravel(a) for a in arrays2b]
arrays3 = [loc_Te.X, loc_Te.y, loc_Te.err_y, soc_Te.X, soc_Te.y, soc_Te.err_y]
arrays3f = [np.ravel(a) for a in arrays3]

np.save('fig3_data1', np.array(arrays1))
np.save('fig3_data2a', np.array(arrays2fa))
np.save('fig3_data2b', np.array(arrays2fb))
np.save('fig3_data3', np.array(arrays3f))