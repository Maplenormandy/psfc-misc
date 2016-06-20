# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 16:13:12 2016

@author: normandy
"""

import numpy as np

a = 0.0
b = -1.0

mat = np.identity(3)*(a-b) + b*np.ones((3,3))

print np.linalg.eig(mat)