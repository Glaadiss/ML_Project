import numpy as np
import seaborn as sns
import pandas as pd
df = pd.read_excel('./a1.xlsx', index_col=0)

# ['sbp', 'tobacco', 'ldl',, 'famhist', 'typea', 'obesity', 'alcohol', 'age', 'chd']
sns.set(style="whitegrid")

df_norm = (df - df.mean()) / (df.max() - df.min())


def get_x_and_y(name, yNominal=False):
    x = np.array(df_norm.loc[:, df.columns != name])
    y = np.array(df_norm.loc[:, df.columns == name]).squeeze()
    if yNominal:
        y = np.array([int(round(el)) for el in y])
    return x, y


x, y = get_x_and_y('adiposity')

x2, y2 = get_x_and_y('chd', yNominal=True)

origX = df_norm.loc[:, :]


# Plot the residuals after fitting a linear model
# sns.residplot(x, y, lowess=True, color="g")
# sns.pairplot(df)
# plt.show()
