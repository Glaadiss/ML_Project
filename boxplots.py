#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 18:59:18 2019

@author: bartlomiejgladys
"""
from data import *
from matplotlib.pyplot import figure, plot, title, legend, xlabel, ylabel, hist, boxplot, subplot, suptitle
import seaborn as sns



#Density plot of purchase to see distribution
#figure()
#sns.distplot(X2[:,6], hist = True, kde = True, bins = 20, color = 'darkblue', hist_kws = {'edgecolor':'black'}, kde_kws = {'linewidth': 4})
#xlabel(attributeNames[11])
#ylabel('Density')
#title('Density Plot and Histogram of Purchase')

#Box plot for different age groups in x axis, and purchase on y

"""

#Try a pair plot
figure()
sns.swarmplot(x = X2[:,3], y=X2[:,6], palette = 'Set2')
xlabel(attributeNames[5])
ylabel(attributeNames[11])
legend(cityNames)


figure()
sns.distplot(X2[:, 1],hist = True, kde = True, bins = 20, color = 'darkblue', hist_kws = {'edgecolor':'black'}, kde_kws = {'linewidth': 4})
xlabel(attributeNames[3])
ylabel('Density')

figure()
sns.distplot(X2[:, 3],hist = True, kde = True, bins = 20, color = 'darkblue', hist_kws = {'edgecolor':'black'}, kde_kws = {'linewidth': 4})
xlabel(attributeNames[5])
ylabel('Density')

#subplot for density plots
figure()
sns.distplot(X[:, 4],hist = True, kde = True, bins = 20, color = 'darkblue', hist_kws = {'edgecolor':'black'}, kde_kws = {'linewidth': 4})
xlabel(attributeNames[4])
ylabel('Density')

#---------------------------------------------------------------
#subplots of density
subplot(2,3,1)
sns.distplot(X2[:, 6],hist = True, kde = True, bins = 20, color = 'darkblue', hist_kws = {'edgecolor':'black'}, kde_kws = {'linewidth': 4})
xlabel(attributeNames[11], fontsize = 12,fontweight = 'bold')
ylabel('Density')


subplot(2,3,2)
sns.distplot(X[:, 5],hist = True, kde = True, bins = 20, color = 'darkblue', hist_kws = {'edgecolor':'black'}, kde_kws = {'linewidth': 4})
xlabel(attributeNames[4], fontsize = 12, fontweight = 'bold')
ylabel('Density')

subplot(2,3,3)
sns.distplot(X2[:, 1],hist = True, kde = True, bins = 20, color = 'darkblue', hist_kws = {'edgecolor':'black'}, kde_kws = {'linewidth': 4})
xlabel(attributeNames[3],fontsize = 12,fontweight = 'bold')
ylabel('Density')

#product cat 1
subplot(2,3,4)
sns.distplot(X[:,11], hist = True, kde = True, bins = 20, color = 'darkblue', hist_kws = {'edgecolor':'black'}, kde_kws = {'linewidth': 4})
xlabel(attributeNames[8],fontsize = 12,fontweight = 'bold')
ylabel('Density')

#product cat 2
subplot(2,3,5)
sns.distplot(X[:,12], hist = True, kde = True, bins = 20, color = 'darkblue', hist_kws = {'edgecolor':'black'}, kde_kws = {'linewidth': 4})
xlabel(attributeNames[9],fontsize = 12,fontweight = 'bold')
ylabel('Density')

#product cat 3
subplot(2,3,6)
sns.distplot(X[:,13], hist = True, kde = True, bins = 20, color = 'darkblue', hist_kws = {'edgecolor':'black'}, kde_kws = {'linewidth': 4})
xlabel(attributeNames[10],fontsize = 12,fontweight = 'bold')
ylabel('Density')

suptitle('Density Plot', fontsize = 16, fontweight = 'bold')
"""
# --------------------Boxplots----------------
sns.boxplot(x = doc.col_values(2, startingFromRow, rowsCount),y = X2[:,6] , palette = 'Set2')
xlabel(attributeNames[2])
ylabel(attributeNames[11])
title('Purchasing age Group')



#sns.boxplot(x = "age", y = "purchase",hue = 'age', data = df)
#title('Purchasing age Group')
#sns.boxplot(x = "age", y = "purchase",hue = 'age', data = df)




#Purchasing versus city category
#subplot(2,2,2)
#sns.boxplot(x = X2[:,3], y=X2[:,6], palette = 'Set2' )
#xlabel(attributeNames[5])
#ylabel(attributeNames[11])


"""
subplot(2,2,3)
sns.boxplot(x = X2[:,0], y = X2[:,6], palette = 'Set2')
xlabel(attributeNames[2])
ylabel(attributeNames[11])

"""


#sns.boxplot(x = "gender", y = "purchase",hue = 'gender', data = df)