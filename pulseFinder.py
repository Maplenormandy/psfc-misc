# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 13:51:24 2016

@author: normandy
"""

import numpy as np
from scipy.signal import medfilt
import matplotlib.pyplot as plt


def findColdPulses(shot):
    transTree = MDSplus.Tree('transport', shot)
    try:
        injNode = transTree.getNode('\\top.imp_inj.dt196.input_10')
        
        inj = injNode.data()
        time = injNode.dim_of().data()
        
        peaks = medfilt(inj, 5) - np.median(inj) > 0.1
        
        return time[np.diff(peaks*1) > 0]
    except:
        return np.array([])
        
    
    
        
        

if __name__ == "__main__":
    
    
    import readline
    import MDSplus
    
    
    shotList = [
        1150901005,
        1150901006,
        1150901007,
        1150901008,
        1150901009,
        1150901010,
        1150901011,
        1150901013,
        1150901014,
        1150901015,
        1150901016,
        1150901017,
        1150901018,
        1150901020,
        1150901021,
        1150901022,
        1150901023,
        1150901024,
        1150903019,
        1150903021,
        1150903022,
        1150903023,
        1150903024,
        1150903025,
        1150903026,
        1150903028,
        1120216006,
        1120216007,
        1120216008,
        1120216009,
        1120216010,
        1120216011,
        1120216012,
        1120216013,
        1120216014,
        1120216017,
        1120216020,
        1120216021,
        1120216023,
        1120216025,
        1120216026,
        1120216028,
        1120216030,
        1120216031,
        1120106010,
        1120106011,
        1120106012,
        1120106015,
        1120106016,
        1120106017,
        1120106020,
        1120106021,
        1120106022,
        1120106025,
        1120106026,
        1120106027,
        1120106028,
        1120106030,
        1120106031,
        1120106032
        ]     
        
    for shot in shotList:
        
        
        
        print shot, findColdPulses(shot)