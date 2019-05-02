import numpy as np
import seaborn as sns
import pandas as pd

df = pd.read_excel('./a1.xlsx', index_col=0)

# ['sbp', 'tobacco', 'ldl',, 'famhist', 'typea', 'obesity', 'alcohol', 'age', 'chd']
sns.set(style="whitegrid")

df_norm = (df - df.mean()) / (df.max() - df.min())


def get_x_and_y(name, yNominal=False, normalize=False, x_columns=[]):
    data = df_norm if normalize else df
    if x_columns:
        x = np.array(data[x_columns])
    else:
        x = np.array(data.loc[:, df.columns != name])
    y = np.array(data.loc[:, df.columns == name]).squeeze()
    if yNominal:
        y = np.array([int(round(el)) for el in y])
    return x, y


def get_nominal_df(k_count=3):
    if k_count < 1:
        raise Exception("k_count has to be higher than 0")
    divider = round(100 / k_count)
    df_builder = []
    for column in df:
        data = df[column]
        if max(data) == 1:
            df_builder.append((column, lambda d, _min, _max: d, column, 0, 0))
        else:
            percentile = divider
            for i in range(k_count):
                _min = np.percentile(data, percentile - divider)
                _max = np.percentile(data, percentile if i != k_count - 1 else 100)
                new_name = "{0}: {1} - {2}".format(column, round(_min), round(_max))
                func = lambda d, _min, _max: 1 if _min <= d < _max else 0
                df_builder.append((new_name, func, column, _min, _max))
                percentile = percentile + divider

    new_columns_name = [name for (name, _, _, _, _) in df_builder]

    data = np.zeros((len(df), len(df_builder)), dtype=int)
    for i, (_, func, old_name, _min, _max) in enumerate(df_builder):
        data[:, i] = [func(d, _min, _max) for d in df[old_name]]

    transactions = list()
    for i, row in enumerate(data):
        d = [new_columns_name[j] for j, c in enumerate(row) if c == 1]
        transactions.append(d)

    return transactions, pd.DataFrame(data=data, columns=new_columns_name)


x, y = get_x_and_y('adiposity')

x2, y2 = get_x_and_y('chd', yNominal=True, normalize=True)

origX = df_norm.loc[:, :]

# Plot the residuals after fitting a linear model
# sns.residplot(x, y, lowess=True, color="g")
# sns.pairplot(df)
# plt.show()
