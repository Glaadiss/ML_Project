# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 19:56:14 2019

@author: s133016
"""

from Project1 import *
from matplotlib.pyplot import figure, plot, title, legend, xlabel, ylabel, hist, boxplot
import seaborn as sns

#figure()
#title('NanoNose data')

#for c in range(C):
    # select indices belonging to class c:
 #   class_mask = Gender_y==c
  #  plot(X[class_mask,3], X[class_mask,11], 's',alpha=.3)

#legend(GenderNames, loc = 'upper right')
#xlabel(attributeNames[3])
#ylabel(attributeNames[11])



#Density plot of purchase to see distribution
figure()
sns.distplot(X[:,11], hist = True, kde = True, bins = 20, color = 'darkblue', hist_kws = {'edgecolor':'black'}, kde_kws = {'linewidth': 4})
xlabel(attributeNames[11])
ylabel('Density')
title('Density Plot and Histogram of Purchase')

#Box plot for different age groups in x axis, and purchase on y
figure()
sns.boxplot(x = X[:,3],y = X[:,11], palette = 'Set3')
xlabel(attributeNames[3])
ylabel(attributeNames[11])
legend(AgeNames)
title('Purchase versus age Group')

#Try a pair plot

sns.pairplot(df, kind = "reg", diag_kind="kde", palette="husl")
