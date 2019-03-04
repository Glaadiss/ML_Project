#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 14:02:27 2019

@author: bartlomiejgladys
"""
from data import *
from matplotlib.pyplot import figure, plot, title, xlabel, ylabel, show, legend, subplots, cm
from scipy.linalg import svd
import mpld3
# Subtract mean value from data
Y = (smallPca - np.ones((N,1))*smallPca.mean(axis=0)) / (np.ones((N,1))*smallPca.std(axis=0))

# PCA by computing SVD of Y
U,S,V = svd(Y,full_matrices=False)

# Compute variance explained by principal components
rho = (S*S) / (S*S).sum() 

threshold = 0.9
i = 0
j = 1

# 2

V = V.T
# Project the centered data onto principal component space
Z = Y @ (V.T)


fig, ax = subplots(subplot_kw=dict(facecolor='#EEEEEE'))

                                   
classMask = [ el == 0 or el == 1 or el == 2 for el in cityY ] 
        
scatter = ax.scatter(Z[classMask, i],
                     Z[classMask, j],
                     c=cityY,
                     s=100,
                     alpha=0.3
                     )
ax.grid(color='white', linestyle='solid')

ax.set_title("Scatter Plot (with tooltips!)", size=20)

def printData(data):
    return ', '.join(pcaNames[x] + ':' + str(round(data[x], 2))  for x in range(6)) 

labels = [printData(Z[i]) for i in range(len(Z))]


tooltip = mpld3.plugins.PointLabelTooltip(scatter, labels=labels)
mpld3.plugins.connect(fig, tooltip)

mpld3.show()