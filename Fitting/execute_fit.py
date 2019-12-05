from profiles_fits import get_ne_fit, get_te_fit
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('case', type=int)

args = parser.parse_args()
case = args.case
dst='/home/normandy/git/psfc-misc/Fitting/FitsPoP2019/'
mcmc = True

def f(x):
    if x == 0:
        get_ne_fit(shot=1160506007, t_min=0.57, t_max=0.63, dst=dst+'007_soc', plot=False, use_MCMC=mcmc, x0_mean=0.37)
        get_te_fit(shot=1160506007, t_min=0.57, t_max=0.63, dst=dst+'007_soc', plot=False, use_MCMC=mcmc, x0_mean=0.37)
    #elif x == 1:
        get_ne_fit(shot=1160506007, t_min=0.93, t_max=0.99, dst=dst+'007_loc', plot=False, use_MCMC=mcmc, x0_mean=0.37)
        get_te_fit(shot=1160506007, t_min=0.93, t_max=0.99, dst=dst+'007_loc', plot=False, use_MCMC=mcmc, x0_mean=0.37)

    
    if x == 2:
        get_ne_fit(shot=1160506009, t_min=0.69, t_max=0.75, dst=dst+'009_soc', plot=False, use_MCMC=mcmc, x0_mean=0.43)
        get_te_fit(shot=1160506009, t_min=0.69, t_max=0.75, dst=dst+'009_soc', plot=False, use_MCMC=mcmc, x0_mean=0.43)
    #elif x == 3:
        get_ne_fit(shot=1160506009, t_min=0.89, t_max=0.95, dst=dst+'009_loc', plot=False, use_MCMC=mcmc, x0_mean=0.43)
        get_te_fit(shot=1160506009, t_min=0.89, t_max=0.95, dst=dst+'009_loc', plot=False, use_MCMC=mcmc, x0_mean=0.43)
    
        
        
    if x == 4:
        get_ne_fit(shot=1160506015, t_min=0.65, t_max=0.71, dst=dst+'015_soc', plot=False, use_MCMC=mcmc, x0_mean=0.37)
        get_te_fit(shot=1160506015, t_min=0.65, t_max=0.71, dst=dst+'015_soc', plot=False, use_MCMC=mcmc, x0_mean=0.37)
    #elif x == 5:
        get_ne_fit(shot=1160506015, t_min=0.91, t_max=0.97, dst=dst+'015_loc', plot=False, use_MCMC=mcmc, x0_mean=0.37)
        get_te_fit(shot=1160506015, t_min=0.91, t_max=0.97, dst=dst+'015_loc', plot=False, use_MCMC=mcmc, x0_mean=0.37)

    return 0

f(case)
