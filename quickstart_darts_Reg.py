from darts.datasets import AirPassengersDataset, MonthlyMilkDataset
from darts.dataprocessing.transformers import Scaler
import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(16, 6))

series_air = AirPassengersDataset().load().astype(np.float32).map(np.log)#.map(np.exp)
series_milk = MonthlyMilkDataset().load().astype(np.float32)

# set aside last 36 months of each series as validation set:
train_air, val_air = series_air[:-36], series_air[-36:]
train_milk, val_milk = series_milk[:-36], series_milk[-36:]

train_air.map(np.exp).plot(label="train (air)")
val_air.map(np.exp).plot(label="val (air)")
train_milk.plot(label="train (milk)")
val_milk.plot(label="val (milk)")


from darts import concatenate
from darts.utils.timeseries_generation import datetime_attribute_timeseries as dt_attr
air_covs = concatenate(
    [
        dt_attr(series_air, "month", dtype=np.float32),
        dt_attr(series_air, "year", dtype=np.float32),
    ],
    axis="component",
)
milk_covs = concatenate(
    [
        dt_attr(series_milk, "month", dtype=np.float32),
        dt_attr(series_milk, "year", dtype=np.float32),
    ],
    axis="component",
)

def extract_year(idx):
    """Extract the year each time index entry and normalized it."""
    return (idx.year - 1950) / 50
encoders = {
    "cyclic": {"future": ["month", "week"]},
    "datetime_attribute": {"future": ["dayofyear", "weekofyear", "day", "dayofweek"]},
    # "position": {"past": ["absolute"], "future": ["relative"]},
    # "custom": {"past": [extract_year]},
    "transformer": Scaler(),
}

scaler = Scaler()
train_air_scaled, train_milk_scaled = scaler.fit_transform([train_air, train_milk])
air_covs_scaled, milk_covs_scaled = Scaler().fit_transform([air_covs, milk_covs])


from sklearn.linear_model import BayesianRidge
from darts.models import RegressionModel
model = RegressionModel(lags=72, lags_future_covariates=[-6, 0], model=BayesianRidge())
model.fit(
    [train_air_scaled, train_milk_scaled],
    future_covariates=[air_covs_scaled, milk_covs_scaled],
)
pred_air, pred_milk = model.predict(
    series=[train_air_scaled, train_milk_scaled],
    future_covariates=[air_covs_scaled, milk_covs_scaled],
    n=36,
)


pred_air, pred_milk = scaler.inverse_transform([pred_air, pred_milk])

from darts.metrics import mape
print(mape([series_air, series_milk], [pred_air, pred_milk]))



pred_air.map(np.exp).plot(label="forecast (air)")
pred_milk.plot(label="forecast (milk)")

plt.show()