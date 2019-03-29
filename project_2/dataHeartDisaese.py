import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_excel('./a1.xlsx', index_col=0)

# ['sbp', 'tobacco', 'ldl', 'adiposity', 'famhist', 'typea', 'obesity', 'alcohol', 'age', 'chd']
sns.set(style="whitegrid")

df_norm = (df - df.mean()) / (df.max() - df.min())

yName = 'obesity'

x = np.array(df_norm.loc[:, df.columns != yName])

y = np.array(df_norm.loc[:, df.columns == yName]).squeeze()


# Plot the residuals after fitting a linear model
# sns.residplot(x, y, lowess=True, color="g")
# sns.pairplot(df)
# plt.show()
