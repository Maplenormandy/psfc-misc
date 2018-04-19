import matplotlib.pyplot as plt
import numpy as np

# %%

radii = np.loadtxt('./input.norm.radii')
ky = np.loadtxt('./input.norm.ky')

if np.size(radii) == 1:
    radii = np.array([radii])

cases = ["cg%02d_%03d" % (int(np.round(radii[i/len(ky)]*100)), int(np.round(ky[i%len(ky)]*10))) for i in range(len(radii)*len(ky))]

flux_avg = np.load('out.norm.flux_avg.npy')
flux_std = np.load('out.norm.flux_std.npy')

particle_weights = np.append([0.0, 0.0, 0.0, 1.0], np.zeros(8))
qe_weights = np.zeros(12)
qe_weights[7] = 1.0
qi_weights = np.zeros(12)
qi_weights[4] = 1.0
mom_weights = np.zeros(12)
mom_weights[8] = 1.0

pflux = np.einsum('i,ijk',particle_weights, flux_avg)
pflux_std = np.sqrt(np.einsum('i,ijk',particle_weights, flux_std**2))

qeflux = np.einsum('i,ijk',qe_weights, flux_avg)
qeflux_std = np.sqrt(np.einsum('i,ijk',qe_weights, flux_std**2))

qiflux = np.einsum('i,ijk',qi_weights, flux_avg)
qiflux_std = np.sqrt(np.einsum('i,ijk',qi_weights, flux_std**2))

momflux = np.einsum('i,ijk',mom_weights, flux_avg)
momflux_std = np.sqrt(np.einsum('i,ijk',mom_weights, flux_std**2))


plt.figure()
"""
plt.errorbar(range(len(ky)), flux_avg[0,:], yerr=flux_std[0,:], marker='.', label='D')
plt.errorbar(range(len(ky)), flux_avg[1,:], yerr=flux_std[1,:], marker='.', label='Ar')
plt.errorbar(range(len(ky)), flux_avg[2,:], yerr=flux_std[2,:], marker='.', label='Lumped')
plt.errorbar(range(len(ky)), flux_avg[3,:], yerr=flux_std[3,:], marker='.', label='e')
plt.legend()
"""


 # %%
rad_ind = 3

#plt.errorbar(ky, pflux[rad_ind,:], yerr=pflux_std[rad_ind,:], marker='.', label='$\Gamma$e')
#plt.errorbar(ky, qeflux[rad_ind,:], yerr=qeflux_std[rad_ind,:], marker='.', label='Qe')
#plt.errorbar(ky, qiflux[rad_ind,:], yerr=qiflux_std[rad_ind,:], marker='.', label='Qi')
#plt.errorbar(ky, momflux[rad_ind,:], yerr=momflux_std[rad_ind,:], marker='.', label='Momentum')

plt.errorbar(ky, qiflux[rad_ind,:], yerr=qiflux_std[rad_ind,:], marker='x', label='W,Qi')
plt.errorbar(ky, qeflux[rad_ind,:], yerr=qeflux_std[rad_ind,:], marker='x', label='W,Qe')
plt.errorbar(ky, pflux[rad_ind,:], yerr=pflux_std[rad_ind,:], marker='.', label='W,$\Gamma$e')
plt.errorbar(ky, momflux[rad_ind,:], yerr=momflux_std[rad_ind,:], marker='.', label='W,$\Pi$')

plt.xscale('log')
plt.ylim([-1.6,3.5])
plt.xlim([0.1,25])
plt.ylabel('Quasilinear Weight [GB units]')

plt.axhline(0.0, ls='--', c='black')

plt.title('r/a = ' + str(radii[rad_ind]))

plt.xlabel('k_y rho_s')

plt.legend()
plt.show()
