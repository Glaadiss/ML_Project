# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 11:37:50 2019

@author: s133016
"""

import numpy as np
import pandas as pd
import xlrd


#import the excel sheet
doc = xlrd.open_workbook(r'C:\Users\danie\Dropbox\3 semester\Intro to Machine learning 02450\02450Toolbox_Python\Data\BlackFriday2.xlsx').sheet_by_index(0)
#BlackFriday = pd.read_excel(r'C:\Users\danie\Dropbox\3 semester\Intro to Machine learning 02450\02450Toolbox_Python\Data\BlackFriday2.xlsx', sheet_name = 0)

colsCount = 12
#rowsCount = 500
startingFromRow = 2
rowsCount = 1000
#rowsCount = len(doc.col_values(0,startingFromRow))

matrix_dim = rowsCount - startingFromRow 

#starts at the second row, first column up in a total of 12 columsn
attributeNames = doc.row_values(1,0,colsCount)

#Transforming the data: Gender, Age, CitybyCategory, and Stay_in_Current
AgeLabels = doc.col_values(3, startingFromRow, rowsCount)
AgeNames = sorted(set(AgeLabels))
AgeDict = dict(zip(AgeNames, range(7)))

GenderLabels = doc.col_values(2, startingFromRow, rowsCount)
GenderNames = sorted(set(GenderLabels))
GenderDict = dict(zip(GenderNames, range(2)))

CityLabels = doc.col_values(5, startingFromRow, rowsCount)
CityNames = sorted(set(CityLabels))
CityDict = dict(zip(CityNames, range(3)))

StayLabels = doc.col_values(6, startingFromRow, rowsCount)


#Create a vector containing the transformed information
Age_y = np.asarray([AgeDict[value] for value in AgeLabels])
Gender_y = np.asarray([GenderDict[value] for value in GenderLabels])
City_y = np.asarray([CityDict[value] for value in CityLabels])
Stay_y = np.asarray([5 if '4+' ==  value else value for value in StayLabels])

#Create the Matrix for the stored data (100-12=88 and 12 columns so 88x12)
X = np.empty((matrix_dim,colsCount), dtype = int)

#This is for the productID, since we don't use it we need to fill the columns
zeros = np.zeros(matrix_dim)
#This is Customer ID
X[:, 0] = doc.col_values(0, startingFromRow, rowsCount)
X[:, 1] = zeros
X[:, 2] = Gender_y
X[:, 3] = Age_y
X[:, 4] = doc.col_values(4, startingFromRow, rowsCount)
X[:, 5] = City_y
X[:, 6] = Stay_y
X[:, 7] = doc.col_values(7, startingFromRow, rowsCount)

#This is to replace all NaN with 0 in Product_Cat1 and 2
for i, col_id in enumerate(range(8, 11)):
    tempData = doc.col_values(col_id, startingFromRow, rowsCount)
    X[:, col_id] = np.asarray([0 if '' == value else value for value in tempData])

X[:, 11] = np.asarray(doc.col_values(11, startingFromRow, rowsCount))

df = pd.DataFrame(X)


N = len(Age_y)

M = len(AgeNames)

C = len(AgeLabels)



