import matplotlib as mpl
import matplotlib.pyplot as plt

import numpy as np

font = {'family' : 'serif',
        'serif': ['Computer Modern'],
        'size'   : 9}

mpl.rc('font', **font)
fig6 = plt.figure(6, figsize=(3.375*1.5,3.375*0.75))
gs6 = mpl.gridspec.GridSpec(1, 3, width_ratios=[4,4,1])

ax60 = plt.subplot(gs6[0])
ax6s = plt.subplot(gs6[1])
ax61 = plt.subplot(gs6[2])


def plotQlWeights(folder, ax, extras=False):
    radii = np.loadtxt(folder + 'input.norm.radii')
    ky = np.loadtxt(folder + 'input.norm.ky')

    flux_avg = np.load(folder + 'out.norm.flux_avg.npy')
    #flux_std = np.load(folder + 'out.norm.flux_std.npy')

    particle_weights = np.append([0.0, 0.0, 0.0, 1.0], np.zeros(8))
    qe_weights = np.zeros(12)
    qe_weights[7] = 1.0
    qi_weights = np.zeros(12)
    qi_weights[4] = 1.0
#qi_weights[5] = 1.0
#qi_weights[6] = 1.0
    mom_weights = np.zeros(12)
    mom_weights[8] = 1.0
#mom_weights[9] = 1.0
#mom_weights[10] = 1.0

    pflux = np.einsum('i,ijk',particle_weights, flux_avg)
    #pflux_std = np.sqrt(np.einsum('i,ijk',particle_weights, flux_std**2))

    qeflux = np.einsum('i,ijk',qe_weights, flux_avg)
    #qeflux_std = np.sqrt(np.einsum('i,ijk',qe_weights, flux_std**2))

    qiflux = np.einsum('i,ijk',qi_weights, flux_avg)
    #qiflux_std = np.sqrt(np.einsum('i,ijk',qi_weights, flux_std**2))

    #momflux = np.einsum('i,ijk',mom_weights, flux_avg)
    #momflux_std = np.sqrt(np.einsum('i,ijk',mom_weights, flux_std**2))

    rad_ind = np.searchsorted(radii, 0.57)

    ax.axvline(0.465, 0, 0.85, ls=':', c=(0.7, 0.7, 0.7))
    ax.axvline(1.25, 0, 0.95, c=(0.7, 0.7, 0.7))
    ax.axvline(2.6, 0, 0.95, c=(0.7, 0.7, 0.7))

    ax.plot(ky, qiflux[rad_ind,:], marker='.', label='$W {Q_i}$', c='b')
    ax.plot(ky, qeflux[rad_ind,:], marker='.', label='$W {Q_e}$', c='g')
    ax.plot(ky, pflux[rad_ind,:], marker='.', label='$W {\Gamma_e}$', c='r')

    ax.set_xscale('log')
    ax.set_ylim([-1.0,3.6])
    ax.set_xlim([0.1,25])
    ax.set_xlabel(r'$k_y \rho_s$')

    ax.axhline(0.0, ls='--', c='black')

    if extras:
        ax.text(0.16, 3.1, 'Ion-Scale')
        ax.text(2.7, 3.1, '$e^-$-Scale')
        ax.set_title('LOC')
    else:
        ax.set_title('SOC')

    ax.text(0.2, 2.5, 'Ia')
    ax.text(0.6, 2.5, 'Ib')
    ax.text(1.6, 2.5, 'II')
    ax.text(6.0, 2.5, 'III')

plotQlWeights('./cgyro_outputs/loc/', ax60, True)
plotQlWeights('./cgyro_outputs/', ax6s)
ax60.set_ylabel('(GB units)')

#plt.title('r/a = ' + str(radii[rad_ind]))
pos = np.array([0.5, 1.5, 2.5])
vals = [2.0, 3.0, 0.01]
ax61.bar(pos, vals, width=1.0, color=('b', 'g', 'r'), tick_label=('$Q_i$','$Q_e$','$\Gamma_e$'), align='center')

ax61.yaxis.tick_right()
ax61.xaxis.set_ticks_position('bottom')
#ax61.set_xlabel('Anomalous Flux (GB units)')

plt.tight_layout(w_pad=0.2)
plt.tight_layout(w_pad=0.2)

plt.savefig('qlweights_locsoc.pdf', format='pdf', dpi=1200, facecolor='white')
