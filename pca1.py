#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 11:10:59 2019

@author: bartlomiejgladys
"""

from data import pcaNames, smallPca, N, np
import matplotlib.pyplot as plt
import numpy


"""
The diagonal values in the Sigma matrix are known as the singular values of the original matrix A. 
The columns of the U matrix are called the left-singular vectors of A, 
and the columns of V are called the right-singular vectors of A.
"""



# Subtract mean value from data
Y = (smallPca - np.ones((N,1))*smallPca.mean(axis=0)) / (np.ones((N,1))*smallPca.std(axis=0))



# PCA by computing SVD of Y
U,S,V = numpy.linalg.svd(Y,full_matrices=False)

pcaShow = [ [pcaNames[i]] + V.T[i].tolist() for i in range(len(pcaNames)) ]

# Compute variance explained by principal components
rho = (S*S) / (S*S).sum() 

threshold = 0.9

# Plot variance explained
plt.figure()
plt.plot(range(1,len(rho)+1),rho,'x-')
plt.plot(range(1,len(rho)+1),np.cumsum(rho),'o-')
plt.plot([1,len(rho)],[threshold, threshold],'k--')
plt.title('Variance explained by principal components');
plt.xlabel('Principal component');
plt.ylabel('Variance explained');
plt.legend(['Individual','Cumulative','Threshold'])
plt.grid()
plt.show()

print('Ran Exercise 2.1.3')

