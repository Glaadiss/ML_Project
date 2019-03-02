#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 12:48:09 2019

@author: bartlomiejgladys
"""

# exercise 2.2.4

# (requires data structures from ex. 2.2.1)
from data import smallPca, N, np, pcaNames
import matplotlib.pyplot as plt

from scipy.linalg import svd
# Subtract mean value from data
Y = (smallPca - np.ones((N,1))*smallPca.mean(axis=0)) / (np.ones((N,1))*smallPca.std(axis=0))

# PCA by computing SVD of Y
U,S,V = svd(Y,full_matrices=False)
pcaShow = [ [pcaNames[i]] + V.T[i].tolist() for i in range(len(pcaNames)) ]


V=V.T
N,M = smallPca.shape

# We saw in 2.1.3 that the first 3 components explaiend more than 90
# percent of the variance. Let's look at their coefficients:
# PC as legend
pcs = [0, 1, 2, 3, 4, 5]

"""
legendStrs = ['PC'+str(e+1) for e in pcs]
c = ['r','g','b']
bw = 0.2
r = np.arange(1,M+1)
for i in pcs:    
    plt.bar(r+i*bw, V[:,i], width=bw, )
plt.xticks(r+bw, pcaNames)
plt.xlabel('Attributes')
plt.ylabel('Component coefficients')
plt.legend(legendStrs)
plt.grid()
plt.title('PCA Component Coefficients')
plt.show()
"""

attributes = ['PC'+str(e+1) for e in pcs]
# attributes as legend
legendStrs = pcaNames
c = ['r','g','b']
bw = 0.1
r = np.arange(1,M+1)
for i, val in enumerate(pcaNames):    
    plt.bar(r+i*bw, V[:,i], width=bw, )
plt.xticks(r+bw, attributes)
plt.xlabel('Attributes')
plt.ylabel('Component coefficients')
plt.legend(legendStrs)
plt.grid()
plt.title('PCA Component Coefficients')
plt.show()

