import numpy as np
import xlrd
import pandas as pd
# Load xls sheet with data
doc = xlrd.open_workbook('./BlackFridayRandom.xlsx').sheet_by_index(0)

def getValue(val): 
    return 60 if '55+' == val else np.asarray(val.split('-'), dtype=int).mean()


colsCount = 15
rowsCount = 500
startingFromRow = 2
memSize = rowsCount - startingFromRow 
attributeNames = doc.row_values(1,0, colsCount)

ageLabels = doc.col_values(3, startingFromRow, rowsCount)
ageNames = sorted(set(ageLabels))
ageDict = dict(zip(ageNames, range(7)))
ageY = np.asarray([getValue(value) for value in ageLabels])


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
X[:, 2] = [1 if value == 0 else 0 for value in genderY]
X[:, 3] = [1 if value == 1 else 0 for value in genderY]
X[:, 4] = ageY
X[:, 5] = doc.col_values(4, startingFromRow, rowsCount)
X[:, 6] = [1 if value == 0 else 0 for value in cityY]
X[:, 7] = [1 if value == 1 else 0 for value in cityY]
X[:, 8] = [1 if value == 2 else 0 for value in cityY]
X[:, 9] = stayInY
X[:, 10] = doc.col_values(7, startingFromRow, rowsCount)
for i, col_id in enumerate(range(11, 14)):
    tempData = doc.col_values(col_id - 3, startingFromRow, rowsCount)
    X[:, col_id] = np.asarray([0 if '' == value else int(float(value)) for value in tempData])
X[:, 14] = np.asarray(doc.col_values(11, startingFromRow, rowsCount))


colsCount2 = 7
X2 = np.empty((memSize, colsCount2), dtype=int)
X2[:, 0] = genderY
X2[:, 1] = ageY
X2[:, 2] = doc.col_values(4, startingFromRow, rowsCount)
X2[:, 3] = cityY
X2[:, 4] = stayInY
X2[:, 5] = doc.col_values(7, startingFromRow, rowsCount)
X2[:, 6] = np.asarray(doc.col_values(11, startingFromRow, rowsCount))

df = pd.DataFrame(X2)
df.columns = ['gender', 'age', 'occupation', 'city', 'stay In Y', 'martial status', 'purchase']     

print(X)
N = len(ageY)
M = len(ageNames)
C = len(ageLabels)



