from profiles_fits import get_ne_fit, get_te_fit
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('case', type=int)

args = parser.parse_args()
case = args.case
dst='/home/normandy/git/psfc-misc/Fitting/mcmc/'
mcmc = True

def f(x):
    if x == 0:
        get_ne_fit(shot=1160506007, t_min=0.57, t_max=0.63, dst=dst+'soc', plot=False, use_MCMC=mcmc)
    elif x == 1:
        get_te_fit(shot=1160506007, t_min=0.57, t_max=0.63, dst=dst+'soc', plot=False, use_MCMC=mcmc)
    elif x == 2:
        get_ne_fit(shot=1160506007, t_min=0.93, t_max=0.99, dst=dst+'loc', plot=False, use_MCMC=mcmc)
    elif x == 3:
        get_te_fit(shot=1160506007, t_min=0.93, t_max=0.99, dst=dst+'loc', plot=False, use_MCMC=mcmc)
    elif x == 4:
        get_ne_fit(shot=1150903021, t_min=1.01, t_max=1.07, dst=dst+'old', plot=False, use_MCMC=mcmc)

    return 0

f(case)
