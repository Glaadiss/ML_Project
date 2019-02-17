# exercise 2.1.1
import numpy as np
import xlrd

# Load xls sheet with data
doc = xlrd.open_workbook('/Users/bartlomiejgladys/Desktop/BlackFriday2.xlsx').sheet_by_index(0)

colsCount = 12
rowsCount = 92
startingFromRow = 2
memSize = rowsCount - startingFromRow 
attributeNames = doc.row_values(1,0, colsCount)

ageLabels = doc.col_values(3, startingFromRow, rowsCount)
ageNames = sorted(set(ageLabels))
ageDict = dict(zip(ageNames, range(7)))
ageY = np.asarray([ageDict[value] for value in ageLabels])


genderLabels = doc.col_values(2, startingFromRow, rowsCount)
genderNames = sorted(set(genderLabels))
genderDict = dict(zip(genderNames, range(2)))
genderY = np.asarray([genderDict[value] for value in genderLabels])

cityLabels = doc.col_values(5, startingFromRow, rowsCount)
cityNames = sorted(set(cityLabels))
cityDict = dict(zip(cityNames, range(3)))
cityY = np.asarray([cityDict[value] for value in cityLabels])

stayInLabels = doc.col_values(6, startingFromRow, rowsCount)
stayInY = np.asarray([5 if '4+' == value else value for value in stayInLabels])


# Extract vector y, convert to NumPy array

# Preallocate memory, then extract excel data to matrix X


X = np.empty((memSize, colsCount), dtype=int)
zeros = np.zeros(memSize)
X[:, 0] = doc.col_values(0, startingFromRow, rowsCount)
X[:, 1] = zeros
X[:, 2] = genderY
X[:, 3] = ageY
X[:, 4] = doc.col_values(4, startingFromRow, rowsCount)
X[:, 5] = cityY
X[:, 6] = stayInY
X[:, 7] = doc.col_values(7, startingFromRow, rowsCount)
for i, col_id in enumerate(range(8, 11)):
    tempData = doc.col_values(col_id, startingFromRow, rowsCount)
    X[:, col_id] = np.asarray([0 if '' == value else value for value in tempData])
X[:, 11] = np.asarray(doc.col_values(11, startingFromRow, rowsCount))



# Compute values of N, M and C.
#N = len(y)
#M = len(attributeNames)
#C = len(classNames)

print('Ran Exercise 2.1.1')
