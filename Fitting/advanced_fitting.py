# Given (x,y,sigma_y), check statistical consistency of uncertainties and provide an
# improved estimate based on their spatio-temporal scattering using Gaussian Process Regression.
#
# Impose positivity constraint on gradients.
#
# F.Sciortino, 3/31/18
# ==========================================

from __future__ import division
import numpy as np
import gptools
import os
import scipy
import matplotlib.pyplot as plt
import warnings
import multiprocessing
plt.ion()

print "Loaded NC version"

class hyperparams:
    def __init__(self,**kwargs):
        for key, value in kwargs.iteritems():
            setattr(self, key, value)
            #print "Set hparams.%s = %d" %(key, value)


def profile_fit_fs(x, y, err_y=None, optimize=True, grid=None, compute_gradients=False, debug_plots=True, kernel='gibbs', noiseLevel=2., grad_constr=False,low_Te_edge=False, use_MCMC=False, c='b', **kwargs):
    """ Advanced profile fitting for ne, Te and Ti profiles in Alcator C-Mod, based on constrained
    Gaussian Process Regression methods.

    Parameters
    ----------
    x : array of float
        Abscissa of data to be interpolated.
    y : array of float
        Data to be interpolated.
    err_y : array of float, optional
        Uncertainty in `y`. If absent, the data are interpolated.
    optimize : bool, optional
        Specify whether optimization over hyperparameters should occur or not. Default is True.
    debug_plots : bool, optional
        Set to True to plot the data, the smoothed curve (with uncertainty) and
        the location of the peak value.
    noiseLevel : float, optional
        Initial guess for a noise multiplier. Default: 2
    low_Te_edge : bool, optional
         If True, force given quantity (T) to be 0 eV at r/a=1.03. This is imposed as a non-hard constraint,
         creating artificial data of value 0 keV with uncertainty of 0.005 keV
    use_MCMC : bool, optional
         Use MCMC to predict profiles from a GP.
    kwargs : dictionary
        arguments to be passed on to set the hyper-prior bounds for the kernel of choice.
    """
    if type(grid) is not np.ndarray:
        grid = x

    # Create empty object for results:
    res= type('', (), {})()

    hparams = hyperparams(**kwargs);
    
    if kernel=='SE':
        #assert len(kwargs) == 4
        hparams = hyperparams(**kwargs); #hparams.set_kwargs(**kwargs)
        # Defaults:
        if not hasattr(hparams,'sigma_mean'): hparams.sigma_mean = 2.0
        if not hasattr(hparams,'l_mean'): hparams.l_mean = 0.005
        if not hasattr(hparams,'sigma_sd'): hparams.sigma_sd = 10.0
        if not hasattr(hparams,'l_sd'): hparams.l_sd = 0.1
    
        hprior = (
        gptools.GammaJointPriorAlt([hparams.sigma_mean, hparams.l_mean], [hparams.sigma_sd,hparams.l_sd])
        )
        k = gptools.SquaredExponentialKernel(
            #= ====== =======================================================================
            #0 sigma  Amplitude of the covariance function
            #1 l1     Small-X saturation value of the length scale.
            #2 l2     Large-X saturation value of the length scale.
            #= ====== =======================================================================
            # param_bounds=[(0, sigma_max), (0, 2.0)],
            hyperprior=hprior,
            initial_params=[10000.0, 400000.0], # random, doesn't matter because we do random starts anyway
            fixed_params=[False]*2
        )
    elif kernel=='gibbs':
        # Defaults:
        if not hasattr(hparams,'sigma_min'): hparams.sigma_min = 0.0
        if not hasattr(hparams,'sigma_max'): hparams.sigma_max = 10.0
    
        if not hasattr(hparams,'l1_mean'): hparams.l1_mean = 0.3
        if not hasattr(hparams,'l1_sd'): hparams.l1_sd = 0.3
    
        if not hasattr(hparams,'l2_mean'): hparams.l2_mean = 0.5
        if not hasattr(hparams,'l2_sd'): hparams.l2_sd = 0.25
    
        if not hasattr(hparams,'lw_mean'): hparams.lw_mean = 0.0
        if not hasattr(hparams,'lw_sd'): hparams.lw_sd = 0.3
    
        if not hasattr(hparams,'x0_mean'): hparams.x0_mean = 0.0
        if not hasattr(hparams,'x0_sd'): hparams.x0_sd = 0.3
    
        hprior=(
            gptools.UniformJointPrior([(hparams.sigma_min,hparams.sigma_max),])*
            gptools.GammaJointPriorAlt([hparams.l1_mean,hparams.l2_mean,hparams.lw_mean,hparams.x0_mean],
                                       [hparams.l1_sd,hparams.l2_sd,hparams.lw_sd,hparams.x0_sd])
            )
    
        k = gptools.GibbsKernel1dTanh(
            #= ====== =======================================================================
            #0 sigma  Amplitude of the covariance function
            #1 l1     Small-X saturation value of the length scale.
            #2 l2     Large-X saturation value of the length scale.
            #3 lw     Length scale of the transition between the two length scales.
            #4 x0     Location of the center of the transition between the two length scales.
            #= ====== =======================================================================
            initial_params=[2.0, 0.5, 0.05, 0.1, 0.5], # for random_starts!= 0, the initial state of the hyperparameters is not actually used.,
            fixed_params=[False]*5,
            hyperprior=hprior,
            )

    # Create additional noise to optimize over (the first argument is n_dims)
    nk = gptools.DiagonalNoiseKernel(1, n=0, initial_noise=np.mean(err_y)*noiseLevel,
                    fixed_noise=True)#, noise_bound=(np.mean(err_y)*noiseLevel*(4.0/5.0),np.mean(err_y)*noiseLevel*(6.0/5.0)))    #(np.min(err_y), np.max(err_y)*noiseLevel))#, enforce_bounds=True)
    #print "noise_bound= [", np.min(err_y), ",",np.max(err_y)*noiseLevel,"]"

    gp = gptools.GaussianProcess(k, X=x, y=y, err_y=err_y, noise_k=nk)

    # ========= Constraints ============ #
    # Add hard constraint for 0 gradient on axis
    gp.add_data(0, 0, n=1, err_y=0.0)

    # Add non-hard constraint for 0 gradient at the edge
    #gp.add_data(1.2, 0, n=1, err_y=0.05)

    # Add non-hard constraint for gradient in the far edge
    #gp.add_data(1.2, 0, n=0, err_y=0.05)

    if low_Te_edge:
        # Add non-hard constraint for 0 eV value at the edge
        gp.add_data(1.03, 0.0, n=0, err_y=0.005)
        # then hard constraint at 1.05
        gp.add_data(1.05, 0.0, n=0, err_y=0.0001)
        print "####### Applying low_Te_edge constraint!"

    if grad_constr:
        # Impose positive gradients throughout the profile on physical grounds
        con = gptools.Constraint(gp, boundary_val=0.0, n=1, loc='min', type_='gt', bounds=None)

    # =================================== #
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="invalid value encountered in subtract")
        if optimize:
            res_min, ll_trials = gp.optimize_hyperparameters(verbose=False, random_starts=multiprocessing.cpu_count(), num_proc=0)
        else:
            print 'Optimization is turned off. Using initial guesses for hyperparameters!'

    grid2 = np.concatenate((grid, grid))
    derivs2 = np.concatenate((np.zeros(grid.shape), np.ones(grid.shape)))
    m_gp, s_gp = gp.predict(grid2,noise=False,use_MCMC=use_MCMC, n=derivs2, num_proc=1)
    res.grid = grid
    res.m_gp=m_gp[:len(grid)]
    res.s_gp=s_gp[:len(grid)]

    print "Computing Gradients"

    #import pdb
    #pdb.set_trace()
    if compute_gradients:
        res.gm_gp = m_gp[len(grid):]
        res.gs_gp = s_gp[len(grid):]

    res.free_params = gp.free_params[:]
    res.free_param_names = gp.free_param_names[:]
    res.free_param_bounds = gp.free_param_bounds[:]
    print res.free_params

    ###
    sum2_diff = 0
    for i in range(len(y)):
        # Find value of grid that is the closest to x[i]:
        gidx = np.argmin(abs(grid - x[i]))
        sum2_diff = (m_gp[gidx]-y[i])**2

    chi_squared = float(sum2_diff) / len(y)
    num_params = k.num_params
    num_data = len(y)

    res.AIC = chi_squared + 2.0 * num_params
    res.BIC = chi_squared + num_params * scipy.log(num_data)

    # Check percentage of points within 3 sd:
    points_in_1sd=0.0; points_in_2sd=0.0; points_in_3sd=0.0
    for i in range(len(y)):
        # Find value of grid that is the closest to x[i]:
        gidx = np.argmin(abs(grid - x[i]))
        if abs(m_gp[gidx]-y[i]) < s_gp[gidx]:
            points_in_1sd += 1.0
        if abs(m_gp[gidx]- y[i]) > s_gp[gidx] and abs(m_gp[gidx]- y[i]) < 2*s_gp[gidx]:
            points_in_2sd += 1.0
        if abs(m_gp[gidx]- y[i]) > 2*s_gp[gidx] and abs(m_gp[gidx]- y[i]) < 3*s_gp[gidx]:
            points_in_3sd += 1.0

    res.frac_within_1sd=float(points_in_1sd)/ len(y)
    res.frac_within_2sd=float(points_in_2sd)/ len(y)
    res.frac_within_3sd=float(points_in_3sd)/ len(y)

    if debug_plots:
        f = plt.figure(1)
        a = f.add_subplot(1, 1, 1)
        a.errorbar(x, y, yerr=err_y, fmt='.', color=c)
        a.plot(grid, m_gp[:len(grid)], color=c)
        gptools.univariate_envelope_plot(grid, m_gp[:len(grid)], s_gp[:len(grid)], ax=a,label='Inferred', envelopes=[1,3], color=c)

        plt.xlabel('r/a', fontsize=14)
        plt.ylabel('', fontsize=14)
        plt.tick_params(axis='both',which='major', labelsize=14)


    if optimize:
        res.ll = res_min.fun
        res.ll_trials = ll_trials

    return res


