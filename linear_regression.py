from sklearn import linear_model
from data import smallPca, pcaNames, linearRegressionData
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, explained_variance_score

fakeX = np.array([[0, 0], [1, 1], [2, 2], [4, 4]])
fakeY = np.array([0, 1, 2, 3])

# ['avgAge', 'stayInY', 'isMale', 'isFemale', 'martialStatus', cityCategory, occupation]
realX = linearRegressionData[0:, :]
# purchase
realY = smallPca[0:, -1].squeeze()

isDataFake = 0
if isDataFake is 0:
    X, y = realX, realY
else:
    X, y = fakeX, fakeY

attributeNames = pcaNames[0:-1]
reg = linear_model.LinearRegression()
reg.fit(X, y)
predictions = reg.predict(X)

plt.scatter(y, predictions)
# plt.plot(predictions, predictions, color='red')
plt.xlim(0, 26000)
plt.ylim(0, 26000)
plt.xlabel("Real values")
plt.ylabel("Predictions")
plt.show()
print("Mean squared error: %.2f"
      % mean_squared_error(y, predictions))
print('Variance score: %.2f' % explained_variance_score(y, predictions))

##################

