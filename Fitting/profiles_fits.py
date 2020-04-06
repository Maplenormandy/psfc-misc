"""
Script to produce Te, ne and Ti profile fits for transport analysis using TRANSP.

Merge Ti and Te profiles at an edge location in C-Mod where the collisionality is sufficiently high
to reach full species thermal equilibration.

The expected sequence to get data, fit profiles and save them is:
- set shot, times to average through and THT for Hirex-Sr fittings at the beginning of the script
  (also, change the destination path for fit results. Make sure the destination exists!)
- run this script from the command line: python transport_profile_fits.py
- run get_ne_fit(shot=shot, t_min=t_min,t_max=t_max, plot=True, noise_opt=False, dst=dst)
- run get_te_fit(shot=shot, t_min=t_min,t_max=t_max, plot=True, noise_opt=False, dst=dst)
- run get_ti_fit(shot=shot,t_min=t_min,t_max=t_max,THT=THT,merge_point=merge_point,plot=True,noise_opt=False,dst=dst)
- run get_omegator_fit(shot=shot,t_min=t_min,t_max=t_max, THT=THT, plot=True, dst=dst)
- To shift profiles such that the temperature at the LCFS is ~75 eV, also
  run shift_profiles(shot=shot, dst=dst, merge_point = merge_point)

Possibly visualize obtained Te and Ti fits together by running
  overplot_te_ti(shot=shot,dst=dst)

@author: sciortino, 2018-19
"""

from __future__ import division
import sys
sys.path.append('/home/sciortino/ML/machinelearnt')
sys.path.append('/home/sciortino/ML')
sys.path.append('/home/sciortino/shot_analysis_tools')
import gptools
import scipy.special
#import nlopt
from Ti_GPR import Ti_GPR_fitting  # Ti
from profile_unc_estimation import profile_fitting # general, suited for ne,Te
import profiletools

import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import readline
import MDSplus
import cPickle as pkl
import eqtools
import copy
#from transport_classes import prof_object
import advanced_fitting
from advanced_fitting import profile_fit_fs
import pdb
from scipy.interpolate import interp1d, UnivariateSpline
from load_hirex_profs import ThacoData
from profile_unc_estimation import profile_fitting

plt.ion()
plt.close('all')

# =================== USER PARAMETERS ==============================
# For Norman:
shot = 1160506007; t_min = 0.57; t_max = 0.63; THT = 0;
#shot = 1160506007; t_min = 0.93; t_max = 0.99; THT = 0; folder='loc'
#shot = 1150903021; t_min = 1.07; t_max = 1.09; THT = 2; folder=''
#shot = 1120106012; t_min = 0.89; t_max = 0.95; THT = 0
#shot = 1160725013; t_min = 0.85; t_max = 1.45; THT = 0

#shot = 1120216017; t_min = 1.13; t_max = 1.19; THT = 9
#shot = 1150903021; t_min = 0.99; t_max = 1.05; THT = 0

# choose Ti/Te profile merging point
merge_point = 0.95
edge_T = 0.075 # in keV. Profiles will be shifted so that Te is equal to this value at the LCFS
plot=True
noise_opt=False   #keep to False
shift_hirexdata_as_well=True
save_as_dict=True    #to save results as a Python dictionary
edge_focus=False   #plot results for r/a>0.85 only
use_MCMC=False
lastShot = 0
e = None

gibbs_params_global={
            'sigma_min':0.1, 'sigma_max':20.0,
            'l1_mean':1.4, 'l1_sd':0.4,
            'l2_mean':0.8, 'l2_sd':0.3,
            'lw_mean':0.2, 'lw_sd':0.1,
            'x0_mean':0.45, 'x0_sd':0.1
            }
se_params_global={
            'sigma_mean':0.1, 'sigma_sd':1.0,
            'l_mean':0.1, 'l_sd':0.05
            }

if save_as_dict:
    shift_hirexdata_as_well=False # not yet available, likely not useful/necessary to develop

# ====================================================================

dst='/home/normandy/git/psfc-misc/Fitting'
range_1sd = scipy.special.erf(1/np.sqrt(2))
range_2sd = scipy.special.erf(2/np.sqrt(2)) - scipy.special.erf(1/np.sqrt(2))
range_3sd = scipy.special.erf(3/np.sqrt(2)) - scipy.special.erf(2/np.sqrt(2))

class prof_object:
    def __init__(self, x, y, err_y):
        self.x = x
        self.y = y
        self.err_y = err_y

def MSE_Gaussian_loss(x, grad, xx, y, y_unc, params, kernel):
    #assert len(grad) == 0, "grad is not empty, but it should"
    nL = x[0]; print nL
    res_val = profile_fitting(xx, y, err_y=y_unc, optimize=True,
         method='GPR',kernel=kernel,noiseLevel=nL,debug_plots=False, **params)

    frac_within_1sd = res_val.frac_within_1sd
    frac_within_2sd = res_val.frac_within_2sd
    frac_within_3sd = res_val.frac_within_3sd

    beta = 2.0 # try 2
    loss = 0.5 * (range_1sd**(-beta)*(range_1sd - frac_within_1sd)**2 + range_2sd**(-beta)*(range_2sd - frac_within_2sd)**2 + range_3sd**(-beta)*(range_3sd - frac_within_3sd)**2)# + lam * reg
    #print '***************** Validation loss = ', loss, ' ******************'
    return loss


def prof_setup(p, bad_idxs=None):
    ''' Convenience function for ne & Te fits
    '''
    if bad_idxs == None:
        bad_idxs = np.zeros_like(p.y, dtype=bool)

    if p.X.shape[1]>1:
        x = p.X[~bad_idxs,1]
    else:
        x = p.X[~bad_idxs,0]

    y = p.y[~bad_idxs]
    y_unc = p.err_y[~bad_idxs]

    # sort
    sorted_idx = [i[0] for i in sorted(enumerate(x), key = lambda x: x[1])]
    x_n = x[sorted_idx]
    y_n = y[sorted_idx]
    err_y_n = y_unc[sorted_idx]
    prof = prof_object(**{'x':x_n,'y':y_n,'err_y':err_y_n})

    return prof,sorted_idx


def get_ne_fit(shot=shot, t_min=t_min,t_max=t_max, plot=True, noise_opt=False, dst=dst, save_as_dict=save_as_dict, use_MCMC=use_MCMC, c='b', x0_mean=0.45):
    ''' Obtain ne fit either by optimizing over noise or by direct GPR estimate
    '''
    try:
        p_ne=profiletools.ne(shot, include=['CTS','ETS'],abscissa='r/a',t_min=t_min,t_max=t_max)

    except:
        # 1101014030 does not have ETS. Use signals from repeat shot 1101014029
        if shot==1101014030:
            p_ne=profiletools.ne(shot, include=['CTS'],abscissa='r/a',t_min=t_min,t_max=t_max)
            p_ne_ETS=profiletools.ne(1101014029, include=['ETS'],abscissa='r/a',t_min=t_min,t_max=t_max)
            p_ne.add_profile(p_ne_ETS)

        else:
            raise ValueError("Problems fetching TS data!")

    p_ne.remove_points(p_ne.X[:,1]>1.05)  # Thomson definitely not reliable
    if shot==1101014006: p_ne.remove_points(p_ne.err_y>0.25)  # this eliminates one annoying outlier
    p_ne.time_average(weighted=True, y_method='total')
    #p_ne.drop_axis(0)

    # clean up data
    p=copy.deepcopy(p_ne)

    # use convenience function
    ne,sorted_idx = prof_setup(p)
    #time = [p.t_min, p.t_max]

    #gibbs_params={'sigma_min':0.0,'sigma_max':10.0,'l1_mean':1.0,'l2_mean':0.5,'lw_mean':0.01,'x0_mean':1.0,
    #                   'l1_sd':0.3,'l2_sd':0.25,'lw_sd':0.1,'x0_sd':0.05}
    gibbs_params={'sigma_min':0.1,'sigma_max':10.0,'l1_mean':1.0,'l2_mean':0.5,'lw_mean':0.1,'x0_mean':1.0,
                  'l1_sd':3.0,'l2_sd':3.0,'lw_sd':0.1,'x0_sd':0.05}
    #gibbs_params={'sigma_min':0.0,'sigma_max':10.0,'l1_mean':1.0,'l2_mean':0.5,'lw_mean':0.1,'x0_mean':0.33,
        #          'l1_sd':3.0,'l2_sd':3.0,'lw_sd':0.1,'x0_sd':0.05}

    # ONLY TESTING FOR NORMAN !!!!!!!
    gibbs_params=gibbs_params_global
    gibbs_params['x0_mean'] = x0_mean

    res = profile_fit_fs(ne.x, ne.y, err_y= ne.err_y, optimize=True,
                              method='GPR',kernel='gibbs',noiseLevel=0,debug_plots=False,use_MCMC=False,color=c,
                         **gibbs_params)

    deltas = np.abs(res.m_gp - ne.y)/ res.s_gp #ne.err_y
    deltas[ne.err_y == 0] = 0.0
    # pdb.set_trace()
    #bad_idxs = (deltas>=5) | (ne.err_y>0.5)
    bad_idxs = (deltas>=5) | (ne.err_y>0.5)

    # re-sort results from rho=0 outwards
    sorted_idx = [i[0] for i in sorted(enumerate(np.abs(ne.x[~bad_idxs])), key = lambda x: x[1])]

    ne.x = np.abs(ne.x[~bad_idxs])[sorted_idx]
    ne.y = ne.y[~bad_idxs][sorted_idx]
    ne.err_y = ne.err_y[~bad_idxs][sorted_idx]

    # grid to evaluate profiles on --> allow GP to extend well beyond LCFS (but don't use values there!)
    xgrid = np.linspace(0,1.15, 200)

    print "Outliers part done"

    if noise_opt:
        pass
        # Uncertainty quantification and re-fitting:
        #opt = nlopt.opt(nlopt.LN_SBPLX, 1)  # LN_SBPLX
        #opt.set_lower_bounds([1.0,] * opt.get_dimension())
        #opt.set_upper_bounds([4.0,] * opt.get_dimension())
        #opt.set_xtol_abs(0.1)
        #objective = lambda x,grad: MSE_Gaussian_loss(x,grad, ne.x, ne.y, ne.err_y, gibbs_params, kernel='gibbs')
        #opt.set_min_objective(objective)

        ## Launch optimization
        #psi = opt.optimize(np.asarray([2.0]))[0]
        #print " *********** Optimized value of psi: ", psi, ' ****************'

        ## find statistics for optimized result:
        #res = profile_fit_fs(ne.x, ne.y, err_y= ne.err_y, optimize=True, grid=xgrid, compute_gradients=True,
        #                     method='GPR',kernel='gibbs',noiseLevel=psi,debug_plots=plot,
        #                     use_MCMC=use_MCMC,**gibbs_params)

        #print 'Fraction of points within 1 sd for ne: {}'.format(res.frac_within_1sd)
        #print 'Fraction of points within 2 sd for ne: {}'.format(res.frac_within_2sd)
        #print 'Fraction of points within 3 sd for ne: {}'.format(res.frac_within_3sd)

    else:
        res = profile_fit_fs(ne.x, ne.y, err_y= ne.err_y, optimize=True, grid=xgrid, compute_gradients=True,
                             method='GPR',kernel='gibbs',noiseLevel=0,debug_plots=plot, grad_constr=False,
                             use_MCMC=use_MCMC, c=c, **gibbs_params)

    '''
    # get minor radius:
    try:
        t = e.getTimeBase(); a0=np.median(e.getAOut()[(t>t_min)*(t<t_max)])
    except ValueError:
        print "Data retrieval in eqtools failed, fixing a0 to standard value"
        a0 = 0.21989983
    '''

    if save_as_dict:
        ne_prof={};
        ne_prof['X']=res.grid
        ne_prof['y'] = res.m_gp
        ne_prof['err_y']=res.s_gp
        ne_prof['dy_dX'] = res.gm_gp
        ne_prof['err_dy_dX']=res.gs_gp
        ne_prof['a_Ly']=np.abs(res.gm_gp/res.m_gp)
        #ne_prof['a_Ly']=np.abs(a0*res.gm_gp/res.m_gp)
        ne_prof['err_a_Ly']=(1./res.m_gp)*np.sqrt((res.gm_gp/res.m_gp)**2 * res.s_gp**2 + res.gs_gp**2/res.m_gp**2)
        #ne_prof['err_a_Ly']=(a0/res.m_gp)*np.sqrt((res.gm_gp/res.m_gp)**2 * res.s_gp**2 + res.gs_gp**2/res.m_gp**2)
        ne_prof['time'] = (t_min+t_max)/2.0
        
        ne_prof['data_x'] = ne.x
        ne_prof['data_y'] = ne.y
        ne_prof['data_err_y'] = ne.err_y

        if dst!=None:
            with open(dst+'/ne_dict_fit_%d.pkl'%shot,'wb') as f:
                pkl.dump(ne_prof, f, protocol=pkl.HIGHEST_PROTOCOL)
            print 'saved file ' +dst+'/ne_dict_fit_%d.pkl'%shot

    else:
        ne_prof = prof_object()

        # save fitted profile:
        ne_prof.x = res.grid
        ne_prof.y = res.m_gp
        ne_prof.err_y = res.s_gp
        ne_prof.dy_dx = res.gm_gp
        ne_prof.err_dy_dx = res.gs_gp
        ne_prof.a_Ly = np.abs(res.gm_gp/res.m_gp)
        #ne_prof.a_Ly = np.abs(a0*res.gm_gp/res.m_gp)
        ne_prof.err_a_Ly = (1./res.m_gp)*np.sqrt((res.gm_gp/res.m_gp)**2 * res.s_gp**2 + res.gs_gp**2/res.m_gp**2)
        #ne_prof.err_a_Ly = (a0/res.m_gp)*np.sqrt((res.gm_gp/res.m_gp)**2 * res.s_gp**2 + res.gs_gp**2/res.m_gp**2)
        ne_prof.time = (t_min+t_max)/2.0

        if dst!=None:
            with open(dst+'/ne_prof_%d_FS.pkl'%shot,'wb') as f:
                pkl.dump(ne_prof, f, protocol=pkl.HIGHEST_PROTOCOL)
            print 'saved file ' +dst+'/ne_prof_%d_FS.pkl'%shot

    if plot:
        plt.xlabel(r'$r/a$')
        plt.ylabel(r'$n_e [m^{-3}]$')
        if edge_focus: plt.xlim([0.8,1.05])

        plt.figure(2)
        if not save_as_dict:
            plt.errorbar(ne_prof.x, ne_prof.dy_dx, ne_prof.err_dy_dx, fmt='*', color=c)
        else:
            plt.errorbar(ne_prof['X'], ne_prof['dy_dX'], ne_prof['err_dy_dX'], fmt='*', color=c)
        plt.xlabel('r/a'); plt.ylabel('dne/dx')

        plt.figure(3)
        if not save_as_dict:
            plt.errorbar(ne_prof.x, ne_prof.a_Ly, ne_prof.err_a_Ly, fmt='*', color=c)
        else:
            plt.errorbar(ne_prof['X'], ne_prof['a_Ly'], ne_prof['err_a_Ly'], fmt='*', color=c)
        plt.xlabel('r/a'); plt.ylabel('a/Lne'); plt.ylim([-1,5])

        plt.figure(4)
        if not save_as_dict:
            plt.plot(ne_prof.x,ne_prof.err_a_Ly/ne_prof.a_Ly, color=c)
        else:
            plt.plot(ne_prof['X'], ne_prof['err_a_Ly']/ne_prof['a_Ly'], color=c)
        plt.xlabel('r/a'); plt.ylabel('a/Lne fractional uncertainty')
        plt.ylim([0,0.8])

    return ne_prof

def get_te_fit(shot=shot, t_min=t_min,t_max=t_max, plot=True, noise_opt=False, dst=dst, save_as_dict=save_as_dict, use_MCMC=use_MCMC, x0_mean=0.45):
    '''
            Get te fit either by optimizing over noise or by direct GPR fitting
    '''

    # Get ECE data only in the core
    p_Te_ECE=profiletools.Te(shot, include=['GPC','GPC2'],abscissa='r/a',t_min=t_min,t_max=t_max)
    #p_Te_ECE.remove_points(p_Te_ECE.X[:,1]>0.8)

    # Add TS data (core + edge)
    #p_Te=profiletools.Te(shot, include=['CTS','ETS'],abscissa='r/a',t_min=t_min,t_max=t_max)
    p_Te=profiletools.Te(shot, include=['ETS'],abscissa='r/a',t_min=t_min,t_max=t_max)
    p_Te.add_profile(p_Te_ECE)
    p_Te.remove_points(p_Te.y < 0.02)
    #p_Te.remove_points(p_Te.y < p_Te.err_y)
    print "Working"

    """
    except:
        if shot==1101014030:
            # 1101014030 does not have ETS. Use the one from repeat shot 1101014029
            p_Te = profiletools.Te(shot, include=['CTS'],abscissa='r/a',t_min=t_min,t_max=t_max)
            p_Te.add_profile(p_Te_ECE)

            # missing ETS, use repeat shot 1101014029
            p_ETS  = profiletools.Te(1101014029, include=['ETS'],abscissa='r/a',t_min=t_min,t_max=t_max)
            p_ETS.remove_points(p_ETS.X[:,1]<0.6)
            p_Te.add_profile(p_ETS)
        else:
            raise ValueError("Problems fetching TS data!")
    """

    p_Te.time_average(weighted=True, y_method='sample')
    print p_Te.X.shape
    # Remove some problematic points
    if shot%1000 == 1120216000:
        p_Te.remove_points(np.logical_and(p_Te.X[:,0] < 0.466, p_Te.X[:,0] > 0.464))
        p_Te.remove_points(np.logical_and(p_Te.X[:,0] < 0.96, p_Te.X[:,0] > 0.95))

    # clean up data
    p=copy.deepcopy(p_Te)

    # use convenience function
    Te,sorted_idx = prof_setup(p)
    #time = [t_min, t_max]

    #gibbs_params={'sigma_min':0.0,'sigma_max':10.0,'l1_mean':0.5,'l2_mean':0.5,'lw_mean':0.01,'x0_mean':1.0,
        #          'l1_sd':0.02,'l2_sd':0.25,'lw_sd':0.1,'x0_sd':0.05}
    gibbs_params={'sigma_min':0.0,'sigma_max':10.0,'l1_mean':0.5,'l2_mean':0.5,'lw_mean':0.01,'x0_mean':0.33,
                  'l1_sd':0.02,'l2_sd':0.25,'lw_sd':0.1,'x0_sd':0.05}

    gibbs_params=gibbs_params_global
    gibbs_params['x0_mean'] = x0_mean


    res = profile_fit_fs(Te.x, Te.y, err_y= Te.err_y, optimize=True,
                              method='GPR',kernel='gibbs',noiseLevel=1.0,debug_plots=plot,
                         use_MCMC=False, **gibbs_params)

    deltas = np.abs(res.m_gp - Te.y)/ res.s_gp #Te.err_y
    deltas[Te.err_y == 0] = 0.0
    bad_idxs = (deltas>=5) | (Te.err_y>1.5)

    # re-sort results from rho=0 outwards
    sorted_idx = [i[0] for i in sorted(enumerate(np.abs(Te.x[~bad_idxs])), key = lambda x: x[1])]

    Te.x = np.abs(Te.x[~bad_idxs])[sorted_idx]
    Te.y = Te.y[~bad_idxs][sorted_idx]
    Te.err_y = Te.err_y[~bad_idxs][sorted_idx]

    # grid to evaluate profiles on --> allow GP to extend well beyond LCFS (but don't use values there!)
    xgrid = np.linspace(0,1.15, 200)

    print "Outliers done"

    if noise_opt:
        pass
        # Uncertainty quantification and re-fitting:
        #opt = nlopt.opt(nlopt.LN_SBPLX, 1)  # LN_SBPLX
        #opt.set_lower_bounds([1.0,] * opt.get_dimension())
        #opt.set_upper_bounds([4.0,] * opt.get_dimension())
        #opt.set_xtol_abs(0.1)
        #objective = lambda x,grad: MSE_Gaussian_loss(x,grad, Te.x, Te.y, Te.err_y, gibbs_params, kernel='gibbs')
        #opt.set_min_objective(objective)

        ## Launch optimization
        #psi = opt.optimize(np.asarray([2.0]))[0]
        #print " *********** Optimized value of psi: ", psi, ' ****************'

        ## find statistics for optimized result:
        #res = profile_fit_fs(Te.x, Te.y, err_y= Te.err_y, optimize=True, grid=xgrid, compute_gradients=True,
        #                          method='GPR',kernel='gibbs',noiseLevel=psi,debug_plots=plot,
        #                     use_MCMC=use_MCMC,**gibbs_params)

        #print 'Fraction of points within 1 sd for ne: {}'.format(res.frac_within_1sd)
        #print 'Fraction of points within 2 sd for ne: {}'.format(res.frac_within_2sd)
        #print 'Fraction of points within 3 sd for ne: {}'.format(res.frac_within_3sd)

    else:
        res = profile_fit_fs(Te.x, Te.y, err_y= Te.err_y, optimize=True,grid=xgrid, compute_gradients=True,
                                  method='GPR',kernel='gibbs',noiseLevel=1.0,debug_plots=plot,
                             use_MCMC=use_MCMC, **gibbs_params)

    # get minor radius:
    #t = e.getTimeBase(); a0=np.median(e.getAOut()[(t>t_min)*(t<t_max)])

    if save_as_dict:
        te_prof={};
        te_prof['X']=res.grid
        te_prof['y'] = res.m_gp
        te_prof['err_y']=res.s_gp
        te_prof['dy_dX'] = res.gm_gp
        te_prof['err_dy_dX']=res.gs_gp
        te_prof['a_Ly']=np.abs(res.gm_gp/res.m_gp)
        #te_prof['a_Ly']=np.abs(a0*res.gm_gp/res.m_gp)
        te_prof['err_a_Ly']=(1./res.m_gp)*np.sqrt((res.gm_gp/res.m_gp)**2 * res.s_gp**2 + res.gs_gp**2/res.m_gp**2)
        #te_prof['err_a_Ly']=(a0/res.m_gp)*np.sqrt((res.gm_gp/res.m_gp)**2 * res.s_gp**2 + res.gs_gp**2/res.m_gp**2)
        te_prof['time'] = (t_min+t_max)/2.0
        
        te_prof['data_x'] = Te.x
        te_prof['data_y'] = Te.y
        te_prof['data_err_y'] = Te.err_y

        if dst!=None:
            with open(dst+'/te_dict_fit_%d.pkl'%shot,'wb') as f:
                pkl.dump(te_prof, f, protocol=pkl.HIGHEST_PROTOCOL)
            print 'saved file ' +dst+'/te_dict_fit_%d.pkl'%shot

    else:
        te_prof = prof_object()

        # save fitted profile:
        te_prof.x = res.grid
        te_prof.y = res.m_gp
        te_prof.err_y = res.s_gp
        te_prof.dy_dx = res.gm_gp
        te_prof.err_dy_dx = res.gs_gp
        te_prof.a_Ly = np.abs(res.gm_gp/res.m_gp)
        #te_prof.a_Ly = np.abs(a0*res.gm_gp/res.m_gp)
        te_prof.err_a_Ly = (1./res.m_gp)*np.sqrt((res.gm_gp/res.m_gp)**2 * res.s_gp**2 + res.gs_gp**2/res.m_gp**2)
        #te_prof.err_a_Ly = (a0/res.m_gp)*np.sqrt((res.gm_gp/res.m_gp)**2 * res.s_gp**2 + res.gs_gp**2/res.m_gp**2)
        te_prof.time = (t_min+t_max)/2.0

        if dst!=None:
            with open(dst+'/te_prof_%d_FS.pkl'%shot,'wb') as f:
                pkl.dump(te_prof, f, protocol=pkl.HIGHEST_PROTOCOL)
            print 'saved file ' +dst+'/te_prof_%d_FS.pkl'%shot

    if plot:
        plt.xlabel(r'$r/a$')
        plt.ylabel(r'$T_e$ [keV]')
        if edge_focus: plt.xlim([0.8,1.05])

        plt.figure()
        if not save_as_dict:
            plt.errorbar(te_prof.x, te_prof.dy_dx, te_prof.err_dy_dx, fmt='*')
        else:
            plt.errorbar(te_prof['X'], te_prof['dy_dX'], te_prof['err_dy_dX'], fmt='*')
        plt.xlabel('r/a'); plt.ylabel('dTe/dx')

        plt.figure()
        if not save_as_dict:
            plt.errorbar(te_prof.x, te_prof.a_Ly, te_prof.err_a_Ly, fmt='*')
        else:
            plt.errorbar(te_prof['X'], te_prof['a_Ly'], te_prof['err_a_Ly'], fmt='*')
        plt.xlabel('r/a'); plt.ylabel('a/LTe'); plt.ylim([-1,20])

        plt.figure()
        if not save_as_dict:
            plt.plot(te_prof.x,te_prof.err_a_Ly/te_prof.a_Ly)
        else:
            plt.plot(te_prof['X'], te_prof['err_a_Ly']/te_prof['a_Ly'])
        plt.xlabel('r/a'); plt.ylabel('a/LTe fractional uncertainty')
        plt.ylim([0,0.8])

    return te_prof




def get_ti_fit(shot=shot,t_min=t_min,t_max=t_max, THT=THT, merge_point= merge_point,
               plot=True, noise_opt=False, dst=dst, save_as_dict=save_as_dict, te_fit=None, t_inst=0.12, x0_mean=0.45):
    global lastShot, e
    '''
    Get ti profile by merging with te fit
    '''
    
    if shot != lastShot:
        e = eqtools.CModEFITTree(shot) #,tree='EFIT20')
        lastShot= shot

    if shot==1101014029:
        # in this shot, Ar was burnt in the core. A merged profile of Ti was obtained by combining
        # Ar and Ca-injection data -- likely not to high accuracy
        with open('/home/sciortino/fits/tifit_%d_bmix.pkl'%shot,'rb') as f:
            ti_x, ti_y, ti_err_y, (t_min_fit,t_max_fit) = pkl.load(f)

    elif shot==1101014030:
        raise ValueError("FS: I never managed to get a decent Ti fit for this shot and used the fit from repeat-shot 1101014029")
    else:

        specTree = MDSplus.Tree('spectroscopy', shot)
        ana = '.ANALYSIS'
        if THT > 0:
            ana += str(THT)
        rootPath = r'\SPECTROSCOPY::TOP.HIREXSR'+ana

        try:
            branchnode =  specTree.getNode(rootPath+'.HELIKE.PROFILES.Z')
            data = ThacoData(branchnode)
            print 'loaded helike'
        except:
            branchnode =  specTree.getNode(rootPath+'.HLIKE.PROFILES.LYA1')
            data = ThacoData(branchnode)

        ti_x = e.rho2rho('psinorm', 'r/a', data.rho, (t_min+t_max)/2.0)

        # time range of interest
        tt1, tt2 = np.searchsorted(data.time, (t_min, t_max))

        ti_y = np.mean(data.pro[3,tt1:tt2,:],axis=0)-t_inst
        
        # use LoTV for uncertainties
        ti_err_y = np.sqrt(np.var(data.pro[3,tt1:tt2,:],axis=0) + np.mean(data.perr[3, tt1:tt2,:]**2,axis=0))
        
        print 
        plt.errorbar(ti_x, ti_y, ti_err_y)

        # eliminate obvious outliers
        ti_y[(ti_y<0) | (ti_err_y>0.8)] = np.nan
        ti_err_y[(ti_y<0) | (ti_err_y>0.8)] = np.nan

    
    # load te if the fit isn't provided as an argument
    if te_fit == None:
        if save_as_dict:
            with open(dst+'/te_dict_fit_%d.pkl'%shot, 'rb') as f:
                te=pkl.load(f)
        else:
            if shot==1120106012:  # Norman's case
                with open('/home/sciortino/shot_1160506007/transport_profiles/te_prof_1160506007_FS.pkl', 'rb') as f:
                    te=pkl.load(f)
            else:
                with open(dst+'/te_prof_%d_FS.pkl'%shot, 'rb') as f:
                    te=pkl.load(f)
    else:
        te = te_fit

    try:
        te_x = te.x
        te_y = te.y
        te_err_y = te.err_y
    except:
        te_x = te['X']
        te_y = te['y']
        te_err_y = te['err_y']

    # merging position
    edge_idx_te=np.argmin(np.abs(te_x-merge_point))
    edge_idx_ti=np.argmin(np.abs(ti_x-merge_point))

    # merge
    x = ti_x[:edge_idx_ti]
    y = ti_y[:edge_idx_ti]
    err_y = ti_err_y[:edge_idx_ti]

    # find instrumental function offset
    a0,a1 = np.polyfit(x[-3:],y[-3:],deg=1)
    b0,b1 = np.polyfit(te_x[edge_idx_te:edge_idx_te+3], te_y[edge_idx_te:edge_idx_te+3], deg=1)
    offset = (a0*merge_point + a1) - (b0*merge_point + b1)

    ti_x = np.concatenate((x, te_x[edge_idx_te:]), axis=0)
    ti_y = np.concatenate((y - offset, te_y[edge_idx_te:]), axis=0)
    ti_err_y = np.concatenate((err_y, te_err_y[edge_idx_te:]), axis=0)

    # grid to evaluate profiles on (same as for Te) --> allow GP to extend well beyond LCFS (but don't use values there!)
    xgrid = np.linspace(0,1.15, 200)

    ti = prof_object(**{'x':ti_x,'y':ti_y,'err_y':ti_err_y})

    ########
    # prior hyperparameters (making priors wider doesn't seem to matter!)
    gibbs_params={'sigma_min':0.0,'sigma_max':10.0,'l1_mean':0.5,'l2_mean':0.5,'lw_mean':0.01,'x0_mean':1.0,
                  'l1_sd':0.02,'l2_sd':0.25,'lw_sd':0.1,'x0_sd':0.05}

    gibbs_params=gibbs_params_global
    gibbs_params['x0_mean'] = x0_mean
    
    



    # temporary TEST FOR NORMAN
    #gibbs_params={'sigma_min':0.0,'sigma_max':10.0,'l1_mean':0.5,'l2_mean':0.5,'lw_mean':0.01,'x0_mean':1.0,
        #          'l1_sd':20.,'l2_sd':20.0,'lw_sd':0.1,'x0_sd':0.05}

    #########################
    if noise_opt:
        pass
        # Uncertainty quantification and re-fitting:
        #opt = nlopt.opt(nlopt.LN_SBPLX, 1)  # LN_SBPLX
        #opt.set_lower_bounds([1.0,] * opt.get_dimension())
        #opt.set_upper_bounds([4.0,] * opt.get_dimension())
        #opt.set_xtol_abs(0.1)
        #objective = lambda x,grad: MSE_Gaussian_loss(x,grad,ti.x, ti.y, ti.err_y, gibbs_params, kernel='gibbs')
        #opt.set_min_objective(objective)

        ## Launch optimization
        #psi = opt.optimize(np.asarray([2.0]))[0]
        #print " *********** Optimized value of psi: ", psi, ' ****************'

        ## find statistics for optimized result:
        #res = profile_fit_fs(ti.x,ti.y, err_y=ti.err_y, optimize=True, grid=xgrid, compute_gradients=True,
        #             kernel='gibbs', noiseLevel=psi,debug_plots=plot, use_MCMC=use_MCMC, **gibbs_params)

        #plt.xlabel(r'$r/a$', fontsize=16)
        #plt.ylabel(r'$T_i [m^{-3}]$', fontsize=16)
        #print 'Fraction of points within 1 sd for ne: {}'.format(res.frac_within_1sd)
        #print 'Fraction of points within 2 sd for ne: {}'.format(res.frac_within_2sd)
        #print 'Fraction of points within 3 sd for ne: {}'.format(res.frac_within_3sd)

    else:
        # direct estimate for given noise:
        res = profile_fit_fs(ti.x, ti.y, err_y=ti.err_y, optimize=True, grid=xgrid, compute_gradients=True,
                             kernel='gibbs', noiseLevel=2.0,debug_plots=plot, use_MCMC=use_MCMC, **gibbs_params)

    # get minor radius:
    #t = e.getTimeBase(); a0=np.median(e.getAOut()[(t>t_min)*(t<t_max)])

    if save_as_dict:
        ti_prof={};
        ti_prof['X']= res.grid
        ti_prof['y'] = res.m_gp
        ti_prof['err_y']=res.s_gp
        ti_prof['dy_dX'] = res.gm_gp
        ti_prof['err_dy_dX']=res.gs_gp
        ti_prof['a_Ly']=np.abs(res.gm_gp/res.m_gp)
        #ti_prof['a_Ly']=np.abs(a0*res.gm_gp/res.m_gp)
        ti_prof['err_a_Ly']=(1./res.m_gp)*np.sqrt((res.gm_gp/res.m_gp)**2 * res.s_gp**2 + res.gs_gp**2/res.m_gp**2)
        #ti_prof['err_a_Ly']=(a0/res.m_gp)*np.sqrt((res.gm_gp/res.m_gp)**2 * res.s_gp**2 + res.gs_gp**2/res.m_gp**2)
        ti_prof['time'] = (t_min+t_max)/2.0

        if dst!=None:
            with open(dst+'/ti_dict_fit_%d.pkl'%shot,'wb') as f:
                pkl.dump(ti_prof, f, protocol=pkl.HIGHEST_PROTOCOL)
            print 'saved file ' +dst+'/ti_dict_fit_%d.pkl'%shot

    else:
        ti_prof = prof_object()

        # save fitted profile:
        ti_prof.x = res.grid
        ti_prof.y = res.m_gp
        ti_prof.err_y = res.s_gp
        ti_prof.dy_dx = res.gm_gp
        ti_prof.err_dy_dx = res.gs_gp
        ti_prof.a_Ly = np.abs(res.gm_gp/res.m_gp)
        #ti_prof.a_Ly = np.abs(a0*res.gm_gp/res.m_gp)
        ti_prof.err_a_Ly = (1./res.m_gp)*np.sqrt((res.gm_gp/res.m_gp)**2 * res.s_gp**2 + res.gs_gp**2/res.m_gp**2)
        #ti_prof.err_a_Ly = (a0/res.m_gp)*np.sqrt((res.gm_gp/res.m_gp)**2 * res.s_gp**2 + res.gs_gp**2/res.m_gp**2)
        ti_prof.time = (t_min+t_max)/2.0

        # save updated/combined Ti profile:
        if dst!=None:
            with open(dst+'/ti_prof_%d_FS.pkl'%shot,'wb') as f:
                pkl.dump(ti_prof, f, protocol=pkl.HIGHEST_PROTOCOL)
            print 'saved file ' +dst+'/ti_prof_%d_FS.pkl'%shot

    if plot:
        plt.figure()
        if not save_as_dict:
            plt.errorbar(ti_prof.x, ti_prof.y, ti_prof.err_y,  fmt='x')
        else:
            plt.errorbar(ti_prof['X'], ti_prof['y'], ti_prof['err_y'], fmt='x')
        plt.errorbar(te_x, te_y, te_err_y, fmt='o')
        plt.xlabel('r/a');

        plt.figure()
        if not save_as_dict:
            plt.errorbar(ti_prof.x, ti_prof.dy_dx, ti_prof.err_dy_dx, fmt='*')
        else:
            plt.errorbar(ti_prof['X'], ti_prof['dy_dX'], ti_prof['err_dy_dX'], fmt='*')
        plt.xlabel('r/a'); plt.ylabel('dTi/dx')

        plt.figure()
        if not save_as_dict:
            plt.errorbar(ti_prof.x, ti_prof.a_Ly, ti_prof.err_a_Ly, fmt='*')
        else:
            plt.errorbar(ti_prof['X'], ti_prof['a_Ly'], ti_prof['err_a_Ly'], fmt='*')
        plt.xlabel('r/a'); plt.ylabel('a/LTi'); plt.ylim([-1,5])

        plt.figure()
        if not save_as_dict:
            plt.plot(ti_prof.x,ti_prof.err_a_Ly/ti_prof.a_Ly)
        else:
            plt.plot(ti_prof['X'], ti_prof['err_a_Ly']/ti_prof['a_Ly'])
        plt.xlabel('r/a'); plt.ylabel('a/LTi fractional uncertainty')
        plt.ylim([0,0.8])

    return ti_prof






def overplot_te_ti(shot=shot,dst=dst):
    with open(dst+'/omegator_%d_FS.pkl'%shot, 'rb') as f:
        ti=pkl.load(f)
    with open(dst+'/te_prof_%d_FS.pkl'%shot, 'rb') as f:
        te=pkl.load(f)

    plt.figure()
    plt.errorbar(ti.x, ti.y, ti.err_y,  fmt='x')
    plt.errorbar(te.x, te.y, te.err_y, fmt='o')




def get_omegator_fit(shot=shot,t_min=t_min,t_max=t_max, THT=THT, plot=True, dst=dst, x0_mean=0.45):
    global lastShot, e
    ''' Obtain toroidal rotation frequency from saved THACO data.
    Note that this requires Norman to have already saved an ftor BS fit in THACO.

    NOT YET UPDATED TO USE PYTHON DICTIONARIES
    '''

    if shot != lastShot:
        e = eqtools.CModEFITTree(shot) #,tree='EFIT20')
        lastShot= shot

    if shot==1101014029:
        # in this shot, Ar was burnt in the core. A merged profile of ftor was obtained by combining
        # Ar and Ca-injection data -- likely not to high accuracy
        with open('/home/sciortino/fits/ftor_fit_%d_bmix.pkl'%shot,'rb') as f:
            omegator_x, ftor_y, ftor_err_y, (t_min_fit,t_max_fit) = pkl.load(f)

    elif shot==1101014030:
        raise ValueError("FS: this shot had burnt Ar. Better to use the fit from repeat-shot 1101014029")

    else:

        specTree = MDSplus.Tree('spectroscopy', shot)
        ana = '.ANALYSIS'
        if THT > 0:
            ana += str(THT)
        rootPath = r'\SPECTROSCOPY::TOP.HIREXSR'+ana

        try:
            branchnode =  specTree.getNode(rootPath+'.HELIKE.PROFILES.Z')
            data = ThacoData(branchnode)
        except:
            branchnode =  specTree.getNode(rootPath+'.HLIKE.PROFILES.LYA1')
            data = ThacoData(branchnode)

        omegator_x = e.rho2rho('psinorm', 'r/a', data.rho, (t_min+t_max)/2.0)

        # time range of interest
        tt1, tt2 = np.searchsorted(data.time, (t_min, t_max))

        ftor_y = np.mean(data.pro[1,tt1:tt2,:],axis=0)
        # use LoTV for uncertainties
        ftor_err_y = np.sqrt(np.var(data.pro[1,tt1:tt2,:],axis=0) + np.mean(data.perr[1, tt1:tt2,:]**2,axis=0))

    # Change ftor into omega_tor
    #Rmaj = e.rho2rho('r/a', 'Rmid', omegator_x, (t_min+t_max)/2.0)
    y = 2 * np.pi * ftor_y
    err_y = 2 * np.pi * ftor_err_y

    # eliminate obvious outliers
    y[err_y>50.0] = np.nan
    err_y[ err_y>50.0] = np.nan

    xgrid = np.linspace(0,1.15, 200)

    gibbs_params=gibbs_params_global
    gibbs_params['sigma_max'] = 100.0
    se_params={
            'sigma_mean':10.0, 'sigma_sd':10.0,
            'l_mean':0.05, 'l_sd':0.05
            }
    gibbs_params['x0_mean'] = x0_mean

    res = profile_fit_fs(omegator_x, y, err_y=err_y, optimize=True, grid=xgrid, compute_gradients=True,
                         kernel='gibbs', noiseLevel=1.0,debug_plots=plot, use_MCMC=use_MCMC, **gibbs_params)
    #res = profile_fitting(omegator_x, y, err_y, optimize=True, method='GPR', kernel='SE', noise_level=2.0)
    print res.free_params
    print res.free_param_names
    y=res.m_gp
    err_y = res.s_gp



    if save_as_dict:
        omegator={};
        omegator['X']=res.grid
        omegator['y'] = res.m_gp
        omegator['err_y']=res.s_gp
        omegator['dy_dX'] = res.gm_gp
        omegator['err_dy_dX']=res.gs_gp
        omegator['a_Ly']=np.abs(res.gm_gp/res.m_gp)
        #omegator['a_Ly']=np.abs(a0*res.gm_gp/res.m_gp)
        omegator['err_a_Ly']=(1./res.m_gp)*np.sqrt((res.gm_gp/res.m_gp)**2 * res.s_gp**2 + res.gs_gp**2/res.m_gp**2)
        #omegator['err_a_Ly']=(a0/res.m_gp)*np.sqrt((res.gm_gp/res.m_gp)**2 * res.s_gp**2 + res.gs_gp**2/res.m_gp**2)
        omegator['time'] = (t_min+t_max)/2.0

        if dst!=None:
            with open(dst+'/ne_dict_fit_%d.pkl'%shot,'wb') as f:
                pkl.dump(omegator, f, protocol=pkl.HIGHEST_PROTOCOL)
            print 'saved file ' +dst+'/ne_dict_fit_%d.pkl'%shot

    else:
        omegator = prof_object()

        # save fitted profile:
        omegator.x = res.grid
        omegator.y = res.m_gp
        omegator.err_y = res.s_gp
        omegator.dy_dx = res.gm_gp
        omegator.err_dy_dx = res.gs_gp
        omegator.a_Ly = np.abs(res.gm_gp/res.m_gp)
        #omegator.a_Ly = np.abs(a0*res.gm_gp/res.m_gp)
        omegator.err_a_Ly = (1./res.m_gp)*np.sqrt((res.gm_gp/res.m_gp)**2 * res.s_gp**2 + res.gs_gp**2/res.m_gp**2)
        #omegator.err_a_Ly = (a0/res.m_gp)*np.sqrt((res.gm_gp/res.m_gp)**2 * res.s_gp**2 + res.gs_gp**2/res.m_gp**2)
        omegator.time = (t_min+t_max)/2.0

        if dst!=None:
            with open(dst+'/omegator_%d_FS.pkl'%shot,'wb') as f:
                pkl.dump(omegator, f, protocol=pkl.HIGHEST_PROTOCOL)
            print 'saved file ' +dst+'/omegator_%d_FS.pkl'%shot

    if dst!=None:
        with open(dst+'/omegator_prof_%d_FS.pkl'%shot,'wb') as f:
            pkl.dump(omegator, f, protocol=pkl.HIGHEST_PROTOCOL)
        print 'saved file ' +dst+'/omegator_prof_%d_FS.pkl'%shot



    if plot:
        plt.figure()
        if not save_as_dict:
            plt.errorbar(omegator.x, omegator.dy_dx, omegator.err_dy_dx, fmt='*')
        else:
            plt.errorbar(omegator['X'], omegator['dy_dX'], omegator['err_dy_dX'], fmt='*')
        plt.xlabel('r/a'); plt.ylabel('domegator/dx')


    return omegator




def overplot_te_ti_shifted(shot=shot,dst=dst):
    with open(dst+'/omegator_%d_FS_shifted.pkl'%shot, 'rb') as f:
        ti=pkl.load(f)
    with open(dst+'/te_prof_%d_FS_shifted.pkl'%shot, 'rb') as f:
        te=pkl.load(f)

    plt.figure()
    plt.errorbar(ti.x, ti.y, ti.err_y,  fmt='x')
    plt.errorbar(te.x, te.y, te.err_y, fmt='o')


def shift_profiles(shot=shot, dst=dst, merge_point = merge_point, shift_hirexdata_as_well=shift_hirexdata_as_well,
                   save_as_dict=save_as_dict):
    '''
    This function shifts all profiles to match a stated condition of Te at the LCFS.

    Note that currently this only shifts x,y and err_y, and not gradient variables!

    '''

    if save_as_dict:
        # assume data was also saved as Python dictionaries
        with open(dst+'/te_dict_fit_%d.pkl'%shot, 'rb') as f:
            te=pkl.load(f)
        with open(dst+'/ne_dict_fit_%d.pkl'%shot, 'rb') as f:
            ne=pkl.load(f)

        if shift_hirexdata_as_well:
            with open(dst+'/ti_dict_fit_%d.pkl'%shot, 'rb') as f:
                ti=pkl.load(f)
            with open(dst+'/omegator_dict_fit_%d.pkl'%shot, 'rb') as f:
                omegator=pkl.load(f)
    else:
        with open(dst+'/te_prof_%d_FS.pkl'%shot, 'rb') as f:
            te=pkl.load(f)
        with open(dst+'/omegator_%d_FS.pkl'%shot, 'rb') as f:
            ne=pkl.load(f)

        if shift_hirexdata_as_well:
            with open(dst+'/omegator_%d_FS.pkl'%shot, 'rb') as f:
                ti=pkl.load(f)
            with open(dst+'/omegator_prof_%d_FS.pkl'%shot, 'rb') as f:
                omegator=pkl.load(f)

    try:   #or if not save_as_dict
        te_x = te.x; ne_x = ne.x;
        if shift_hirexdata_as_well: ti_x = ti.x; omegator_x = omegator.x
        te_y = te.y; ne_y = ne.y;
        if shift_hirexdata_as_well: ti_y = ti.y; omegator_y = omegator.y
        te_err_y = te.err_y; ne_err_y = ne.err_y;
        if shift_hirexdata_as_well: ti_err_y = ti.err_y; omegator_err_y = omegator.err_y
    except:
        # use dictionaries
        te_x = te['X']; ne_x = ne['X'];
        if shift_hirexdata_as_well: ti_x = ti['X']; omegator_x = omegator['X']

        '''# if using new dictionary form, shift all fields (includes gradients)
        for key in te.keys():
            if key!='X' and key!='time':
                eval('te_'+key+'= te['+key+']')
        '''
        te_y = te['y']; ne_y = ne['y'];
        if shift_hirexdata_as_well: ti_y = ti['y']; omegator_y = omegator['y']
        te_err_y = te['err_y']; ne_err_y = ne['err_y'];
        if shift_hirexdata_as_well: ti_err_y = ti['err_y']; omegator_err_y = omegator['err_y']

    # te_y should probably be interpolated on a denser grid first
    x_dense=np.linspace(te_x.min(),te_x.max(), 1000)

    f=interp1d(te_x,te_y, kind='cubic')
    te_y_dense = f(x_dense)

    edge_idx=np.argmin(np.abs(te_y_dense-edge_T))
    shift = x_dense[edge_idx] - 1.0
    print "Applied shift is \Delta(r/a) = ", shift

    te_x = te_x - shift
    te_y = te_y[te_x>0]
    te_err_y = te_err_y[te_x>0]
    te_x = te_x[te_x>0] # redefinition

    ne_x = ne_x - shift
    ne_y = ne_y[ne_x>0]
    ne_err_y = ne_err_y[ne_x>0]
    ne_x = ne_x[ne_x>0] # redefinition

    if shift_hirexdata_as_well:
        ti_x = ti_x - shift
        ti_y = ti_y[ti_x>0]
        ti_err_y = ti_err_y[ti_x>0]
        ti_x = ti_x[ti_x>0]  # redefinition

        omegator_x = omegator_x - shift
        omegator_y = omegator_y[omegator_x>0]
        omegator_err_y = omegator_err_y[omegator_x>0]
        omegator_x = omegator_x[omegator_x>0]  # redefinition

    # Ensure that first radial point is 0 (accuracy not significantly reduced, but interpolation may also be done...)
    te_x[0] = 0.0; ne_x[0] = 0.0;
    if shift_hirexdata_as_well: ti_x[0] = 0.0; omegator_x[0]=0.0

    if save_as_dict:
        te_prof={}
        te_prof['X']=te_x; te_prof['y'] = te_y; te_prof['err_y'] = te_err_y; te_prof['time'] = te['time']

        omegator={}
        omegator['X']=ne_x; omegator['y'] = ne_y; omegator['err_y'] = ne_err_y; omegator['time'] = ne['time']

        # save dictionaries
        with open(dst+'/te_dict_fit_%d_shifted.pkl'%shot, 'wb') as f:
            pkl.dump(te_prof,f,protocol=pkl.HIGHEST_PROTOCOL)
        with open(dst+'/ne_dict_fit_%d_shifted.pkl'%shot, 'wb') as f:
            pkl.dump(omegator,f,protocol=pkl.HIGHEST_PROTOCOL)

        if shift_hirexdata_as_well:
            omegator={}
            omegator['X']=ti_x; omegator['y'] = ti_y; omegator['err_y'] = ti_err_y; omegator['time'] = ti['time']

            omegator_prof={}
            omegator_prof['X']=omegator_x; omegator_prof['y'] = omegator_y; omegator_prof['err_y'] = omegator_err_y; omegator_prof['time'] = omegator['time']

            with open(dst+'/ti_dict_fit_%d_shifted.pkl'%shot, 'wb') as f:
                pkl.dump(omegator,f,protocol=pkl.HIGHEST_PROTOCOL)
            with open(dst+'/omegator_dict_fit_%d_shifted.pkl'%shot, 'wb') as f:
                pkl.dump(omegator_prof,f,protocol=pkl.HIGHEST_PROTOCOL)


    else:
        # use prof_object format
        te = prof_object(**{'x':te_x,'y':te_y,'err_y':te_err_y, 'time': te.time})
        ne = prof_object(**{'x':ne_x,'y':ne_y,'err_y':ne_err_y, 'time': ne.time})

        if shift_hirexdata_as_well:
            ti = prof_object(**{'x':ti_x,'y':ti_y,'err_y':ti_err_y, 'time': ti.time})
            omegator = prof_object(**{'x':omegator_x,'y':omegator_y,'err_y':omegator_err_y, 'time': omegator.time})


        # plotting currently only for non-dict case...
        plt.rc('xtick',labelsize=16)
        plt.rc('ytick',labelsize=16)
        plt.figure()
        if shift_hirexdata_as_well: plt.errorbar(ti.x, ti.y, ti.err_y,  fmt='x-', label=r'$T_i$ [keV]')
        plt.errorbar(te.x, te.y, te.err_y, fmt='o-',label=r'$T_e$ [keV]')
        plt.xlabel('r/a', fontsize=18); plt.legend(fontsize=18);

        plt.figure()
        plt.errorbar(ne.x,ne.y, ne.err_y, fmt='o-', color ='red')
        plt.xlabel('r/a', fontsize=18); plt.ylabel(r'$n_e$ $[\times 10^{20} m^{-3}]$', fontsize=18)

        with open(dst+'/te_prof_%d_FS_shifted.pkl'%shot, 'wb') as f:
            pkl.dump(te,f,protocol=pkl.HIGHEST_PROTOCOL)
        with open(dst+'/omegator_%d_FS_shifted.pkl'%shot, 'wb') as f:
            pkl.dump(ne,f,protocol=pkl.HIGHEST_PROTOCOL)

        if shift_hirexdata_as_well:
            plt.figure()
            plt.errorbar(omegator.x,omegator.y, omegator.err_y, fmt='o-', color ='purple')
            plt.xlabel('r/a', fontsize=18); plt.ylabel(r'$v_{tor}$ [m/s]', fontsize=18)

            with open(dst+'/omegator_%d_FS_shifted.pkl'%shot, 'wb') as f:
                pkl.dump(ti,f,protocol=pkl.HIGHEST_PROTOCOL)

            with open(dst+'/omegator_prof_%d_FS_shifted.pkl'%shot, 'wb') as f:
                pkl.dump(omegator,f,protocol=pkl.HIGHEST_PROTOCOL)

        print "Shifted profiles and saved them in " + dst

