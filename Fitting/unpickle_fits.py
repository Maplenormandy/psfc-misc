import cPickle as pkl
import numpy as np

import sys

sys.path.append('/home/sciortino/ML/machinelearnt')
sys.path.append('/home/sciortino/ML')
sys.path.append('/home/sciortino/shot_analysis_tools')

arrays = {}

prefix = 'loc_ne_'
path = '/home/normandy/git/psfc-misc/Fitting/mcmc/'

def load_data(p, prefix):
    with open('/home/normandy/git/psfc-misc/Fitting/mcmc/' + p) as f:
        data = pkl.load(f)
        arrays[prefix+'x'] = data.x
        arrays[prefix+'y'] = data.y
        arrays[prefix+'err_y'] = data.err_y
        arrays[prefix+'a_Ly'] = data.a_Ly
        arrays[prefix+'err_a_Ly'] = data.err_a_Ly

load_data('loc/ne_prof_1160506007_FS.pkl', 'loc_ne_')
load_data('loc/te_prof_1160506007_FS.pkl', 'loc_te_')
load_data('soc/ne_prof_1160506007_FS.pkl', 'soc_ne_')
load_data('soc/te_prof_1160506007_FS.pkl', 'soc_te_')
load_data('old/ne_prof_1150903021_FS.pkl', 'old_ne_')

np.savez('profiles.npz', **arrays)
