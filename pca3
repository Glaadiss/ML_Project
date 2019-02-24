#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 12:48:09 2019

@author: bartlomiejgladys
"""

# exercise 2.2.4

# (requires data structures from ex. 2.2.1)
from data import *
import matplotlib.pyplot as plt

from scipy.linalg import svd

Y = (smallPca - np.ones((N,1))*smallPca.mean(axis=0)) / (np.ones((N,1))*smallPca.std(axis=0))

U,S,V = svd(Y,full_matrices=False)
V=V.T
N,M = smallPca.shape

# We saw in 2.1.3 that the first 3 components explaiend more than 90
# percent of the variance. Let's look at their coefficients:
pcs = [0,1,2]
legendStrs = ['PC'+str(e+1) for e in pcs]
c = ['r','g','b']
bw = .2
r = np.arange(1,M+1)
for i in pcs:    
    plt.bar(r+i*bw, V[:,i], width=bw)
plt.xticks(r+bw, cityNames)
plt.xlabel('Attributes')
plt.ylabel('Component coefficients')
plt.legend(legendStrs)
plt.grid()
plt.title('NanoNose: PCA Component Coefficients')
plt.show()
