import pandas as pd
from darts import TimeSeries
import matplotlib.pyplot as plt

# train = pd.read_csv("data/train.csv")

# 1. Load and preprocess
def load_series(csv_path, time_col='time', value_col='value', freq='D'):
    df = pd.read_csv(csv_path)
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.groupby(time_col)[value_col].sum().asfreq('D').reset_index()
    return TimeSeries.from_dataframe(df, time_col, value_col, freq=freq)

train = load_series("data/train.csv", time_col='date', value_col='sales')	
test = load_series("data/transactions.csv", time_col='date', value_col='transactions')

# print(test.head())

test.plot()
# train.plot()
plt.show()

# print(pd.read_csv("data/transactions.csv"))