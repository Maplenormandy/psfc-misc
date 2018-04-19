# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 14:13:37 2018

Checks if the line-integrated data looks good for use

@author: normandy
"""

import numpy as np
import matplotlib.pyplot as plt

import readline
import MDSplus


# %% 

plt.close('all')

# %% Check if a thing looks good

runday = 1120917

num_shots = 0

for i in range(0,40):
    shot = runday*1000 + i
    print shot
    try:
        specTree = MDSplus.Tree('spectroscopy', shot)
        
        foundTht = -1
        
        for tht in range(9):
            ana = 'ANALYSIS'
            if tht > 0:
                ana += str(tht)
            
            try:
                comment = specTree.getNode(r'\SPECTROSCOPY::TOP.HIREXSR.'+ana+':COMMENT').data()
                if 'mdsplus' in comment:
                    continue
                else:
                    print comment
                    foundTht = tht
                    num_shots += 1
                    
                    f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
                    
                    momentsNode = specTree.getNode(r'\SPECTROSCOPY::TOP.HIREXSR.ANALYSIS.HELIKE.MOMENTS.Z')
                    mom = momentsNode.getNode('mom').data()
                    pos = momentsNode.getNode('pos').data()
                    err = momentsNode.getNode('err').data()
                    
                    tgood = np.sum(mom[0,:,0] > 0)
                    chgood = np.sum(mom[0,0,:] > 0)
                    
                    m0 = mom[0,int(tgood/2)+1,:chgood]
                    m0_err = mom[0,int(tgood/2)+1,:chgood]
                    m2 = mom[2,int(tgood/2)+1,:chgood]
                    m2_err = mom[2,int(tgood/2)+1,:chgood]
                    rho = pos[:chgood,3]
                    
                    ax2.errorbar(rho[rho>=0], m2[rho>=0]/m0[rho>=0]*1e6, fmt='^', c='r')
                    ax2.errorbar(np.abs(rho[rho<0]), m2[rho<0]/m0[rho<0]*1e6, fmt='v', c='b')
                    ax2.set_ylim([0, 2])
                    
                    ax1.errorbar(rho[rho>=0], m0[rho>=0], fmt='^', c='r')
                    ax1.errorbar(np.abs(rho[rho<0]), m0[rho<0], fmt='v', c='b')
                    
                    ax1.set_title(str(shot))
                    
                    plt.show()
                    
                    break
            except:
                continue
            
        if foundTht >= 0:
            print 'found', foundTht
    except:
        print "no data"
        
print num_shots, 'total run'