from darts.datasets import AirPassengersDataset
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


plt.figure(figsize=(16, 8))
series = AirPassengersDataset().load().map(np.log)#.map(np.exp)
# series = series.map(np.log).plot()
train, val = series.split_before(pd.Timestamp("19580101"))
train.plot(label="training")
val.plot(label="validation")
# series.add_holidays("BR").plot()


from darts.utils.statistics import check_seasonality, plot_acf
# plot_acf(train, m=12, alpha=0.05, max_lag=24)
for m in range(2, 25):
    is_seasonal, period = check_seasonality(train, m=m, alpha=0.05)
    if is_seasonal:
        print(f"There is seasonality of order {period}.")



from darts.models import NaiveSeasonal, NaiveDrift
naive_model = NaiveSeasonal(K=1)
naive_model.fit(train)
naive_forecast = naive_model.predict(len(val))
# series.plot(label="actual")
# naive_forecast.plot(label="naive forecast (K=1)")

seasonal_model = NaiveSeasonal(K=12)
seasonal_model.fit(train)
seasonal_forecast = seasonal_model.predict(len(val))
# seasonal_forecast.plot(label="naive forecast (K=12)")

drift_model = NaiveDrift()
drift_model.fit(train)
drift_forecast = drift_model.predict(len(val))
# drift_forecast.plot(label="drift")

from darts.metrics import mape
combined_forecast = drift_forecast + seasonal_forecast - train.last_value()
combined_forecast.plot(label=f"Naive() MAPE: {mape(val.map(np.exp), combined_forecast.map(np.exp)):.2f}%.")


from darts.models import AutoARIMA, ExponentialSmoothing, Theta
def eval_model(model):
    model.fit(train)
    forecast = model.predict(len(val))
    forecast.plot(label=f"{model} MAPE: {mape(val.map(np.exp), forecast.map(np.exp)):.2f}%")

# models_config = {
#     'AutoARIMA': AutoARIMA(seasonal=True, m=12),
#     'ExponentialSmoothing': ExponentialSmoothing(seasonal_periods=12, seasonal="add"),
#     'Theta': Theta(seasonal_periods=12, seasonal="add"),
# }

models_config = {
    'AutoARIMA': AutoARIMA(),
    'ExponentialSmoothing': ExponentialSmoothing(),
    'Theta': Theta(),
}
for model_config in models_config:
    eval_model(models_config[model_config])
    

# Backtesting
hfc_params = {
    "series": series,
    "start": pd.Timestamp(
        "1958-01-01"
    ),  # can also be a float for the fraction of the series to start at
    "forecast_horizon": 3,
    "verbose": True,
}

# Search for the best theta parameter, by trying 50 different values
thetas = 2 - np.linspace(-10, 10, 50)
print(f"number iter. {len(thetas)}")

best_mape = float("inf")
best_theta = 0
for theta in thetas:
    model = Theta(theta)
    model.fit(train)
    pred_theta = model.predict(len(val))
    res = mape(val, pred_theta)

    if res < best_mape:
        best_mape = res
        best_theta = theta

best_theta_model = Theta(best_theta)
best_theta_model.fit(train)
pred_best_theta = best_theta_model.predict(len(val))
pred_best_theta.plot(label=f"B. Theta({best_theta:.2f}) MAPE: {mape(val.map(np.exp), pred_best_theta.map(np.exp)):.2f}%")
historical_fcast_theta = best_theta_model.historical_forecasts(last_points_only=True, **hfc_params)
historical_fcast_theta.plot(label=f"BT B. Theta({best_theta:.2f}) MAPE = {mape(series.map(np.exp), historical_fcast_theta.map(np.exp)):.2f}%")


# Forecasting with ExponentialSmoothing
best_model_es = ExponentialSmoothing(seasonal_periods=12)
best_model_es.fit(train)
pred_best_es = best_theta_model.predict(len(val))
pred_best_es.plot(label=f"B. ES(seas=12) MAPE: {mape(val.map(np.exp), pred_best_es.map(np.exp)):.2f}%")
historical_fcast_es = best_model_es.historical_forecasts(last_points_only=True, **hfc_params)
historical_fcast_es.plot(label=f"BT B. ES(seas=12) MAPE = {mape(series.map(np.exp), historical_fcast_es.map(np.exp)):.2f}%")


# Residuals analysis
from darts.utils.statistics import plot_residuals_analysis
plot_residuals_analysis(best_theta_model.residuals(series))
plot_residuals_analysis(best_model_es.residuals(series))

plt.show()