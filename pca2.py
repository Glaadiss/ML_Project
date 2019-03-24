#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 11:10:59 2019

@author: bartlomiejgladys
"""


from data import *
from matplotlib.pyplot import figure, plot, title, xlabel, ylabel, show, legend, subplots
from scipy.linalg import svd

# Subtract mean value from data
Y = (smallPca - np.ones((N,1))*smallPca.mean(axis=0)) / (np.ones((N,1))*smallPca.std(axis=0))

# PCA by computing SVD of Y
U,S,V = svd(Y,full_matrices=False)

# Compute variance explained by principal components
rho = (S*S) / (S*S).sum() 

threshold = 0.9


# 2

V = V.T
# Project the centered data onto principal component space
Z = Y @ (V.T)


# Indices of the principal components to be plotted
i = 0
j = 1


fig, ax = subplots()
N = len(Z)


ax.grid(color='white', linestyle='solid')


# Plot PCA of the data
f = figure()
title('PCA ')
#Z = array(Z)
for c in range(3):
    # select indices belonging to class c:
    class_mask = cityY==c
    plot(Z[class_mask,i], Z[class_mask,j], 'o', alpha=.3)
    
legend(cityNames)
xlabel('PC{0}'.format(i+1))
ylabel('PC{0}'.format(j+1))

# Output result to screen


print('Ran Exercise 2.1.4')