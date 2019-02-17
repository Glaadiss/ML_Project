#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 18:05:27 2019

@author: bartlomiejgladys
"""

from matplotlib.pyplot import plot, title, legend, xlabel, ylabel, show
from data import X,  attributeNames, np, ageNames, C, ageY

X = np.array(X) #Try to uncomment this line
title('')

for c in range(C):
    # select indices belonging to class c:
    class_mask = ageY==c
    plot(X[class_mask,6], X[class_mask,11], 'o',alpha=.3)

legend(ageNames, loc='upper right')

xlabel(attributeNames[6])
ylabel(attributeNames[11])
show()
