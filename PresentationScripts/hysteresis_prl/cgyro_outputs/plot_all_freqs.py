import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

font = {'family': 'normal', 'size': 18}
mpl.rc('font', **font)

folders = ['soc_mid',
           'loc_mid',
           'soc_low',
           'loc_hig',
           'scans/inc_lte',
           'scans/inc_lti',
           'scans/inc_ln',
           'scans/dec_lte',
           'scans/dec_lti',
           'scans/dec_ln']

freq_dict = {}
ky_dict = {}

for fold in folders:

    radii = np.loadtxt(fold+'/input.norm.radii')
    ky = np.loadtxt(fold+'/input.norm.ky')

    if np.size(radii) == 1:
        radii = np.array([radii])

    cases = [fold+"/cg%02d_%03d" % (int(np.round(radii[i/len(ky)]*100)), int(np.round(ky[i%len(ky)]*10))) for i in range(len(radii)*len(ky))]

    omega = np.zeros(ky.size)
    gamma = np.zeros(ky.size)

    for i in range(len(cases)):
        # Get indices into radius and k
        rj = i/len(ky)
        kj = i%len(ky)
        if radii[rj] > 0.6 or radii[rj] < 0.5:
            continue

        with open(cases[i] + '/out.cgyro.freq', 'r') as f:
            content = f.readlines()
            vals = np.array(map(float, content))
            freqs = vals.reshape((2,-1), order='F')
            freqs_cut = freqs[:,len(freqs)/2:]
            freqs_mean = np.average(freqs_cut, axis=1)
            omega[kj] = freqs_mean[0]
            gamma[kj] = freqs_mean[1]

    freq_dict[fold] = np.array([omega, gamma])
    ky_dict[fold] = ky

f, (ax1, ax2) = plt.subplots(2, sharex=True)

all_freqs = dict()
all_freqs['folders'] = np.array(folders)

for fold in reversed(folders):
    ky = ky_dict[fold]
    #kmax = np.searchsorted(ky, 1.42)
    kmax=len(ky)
    kmin=0
    #kmin = np.searchsorted(ky, 1.42)
    omega = freq_dict[fold][0,:]
    gamma = freq_dict[fold][1,:]

    c = (0.8, 0.8, 0.8)
    alpha = 1.0
    if fold == 'soc_mid':
        c = 'b'
        alpha = 1.0
    elif fold == 'loc_mid':
        c = 'r'
        alpha = 1.0
    elif fold == 'soc_low':
        c = (0.5, 0.5, 1.0)
        alpha = 1.0
    elif fold == 'loc_hig':
        c = (1.0, 0.5, 0.5)
        alpha = 1.0

    all_freqs[fold] = np.array([ky, omega, gamma])

    ax1.plot(ky[kmin:kmax], omega[kmin:kmax], marker='.', c=c, alpha=alpha)
    ax2.plot(ky[kmin:kmax], gamma[kmin:kmax], marker='.', c=c, alpha=alpha)

np.savez(file('all_freqs.npz', 'w'), **all_freqs)

#ax1.set_ylabel('omega [c_s/a]')
#ax2.set_ylabel('gamma [c_s/a]')

"""
ax1.axhline(0.0, linestyle='--', c='black')
ax2.axhline(0.0, linestyle='--', c='black')
"""

#ax2.set_xlabel('k_y rho_s')
#ax1.set_xscale('log')
#ax2.set_xscale('log')
#ax1.set_yscale('log')
#ax2.set_yscale('log')

"""
ax1.set_ylim([-0.6, 1.0])
ax2.set_ylim([-0.1, 1.0])
"""

#ax2.set_xlim([2.4, 25.0])

ax1.set_ylim([-0.6, 0.6])
ax2.set_ylim([-0.08, 0.22])

ax2.set_xlim([0.1, 1.42])


f.subplots_adjust(hspace=0)
plt.show()
