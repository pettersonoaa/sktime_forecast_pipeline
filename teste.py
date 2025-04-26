import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta

df = pd.read_csv('data/transactions.csv', usecols=['date', 'transactions'], index_col=0, parse_dates=['date'])
series = pd.Series(df['transactions'], index=df.index)

# series = series.index.max().month
# last_finished_month = series.index.max() - relativedelta(months=1)
# series = series[series.index.month==last_finished_month.month]

series = series.groupby(series.index.month).sum().to_numpy()

print(np.mean(np.abs((series) / series) * 100))
# print(series)