"""
Daily Univariate Time Series Forecasting Pipeline

This script implements an end-to-end forecasting workflow for a univariate daily time series
with columns 'date' and 'transactions'. It uses sktime where possible, supplemented by
Prophet, LightGBM, CatBoost, PyTorch Forecasting (N-BEATS, TFT), and a Keras-based TCN.

Steps:
1. Data ingestion
2. Preprocessing: resampling, imputation, detrend, deseasonalize, scaling
3. Train/test split
4. Model definitions and training (Naive, OLS, ARIMA, ETS, Theta, Prophet,
   LightGBM, CatBoost, N-BEATS, TCN, TFT)
5. Hyperparameter tuning for select models & record train times
6. Forecast generation
7. Evaluation (MAPE)
8. Visualization & export of forecasts and metrics
"""
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler

# sktime imports
from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.trend import PolynomialTrendForecaster
from sktime.forecasting.arima import AutoARIMA
from sktime.forecasting.ets import AutoETS
from sktime.forecasting.theta import ThetaForecaster
from sktime.forecasting.compose import make_reduction
from sktime.forecasting.model_selection import temporal_train_test_split, ExpandingWindowSplitter, ForecastingGridSearchCV
from sktime.forecasting.base import ForecastingHorizon
from sktime.transformations.series.detrend import Detrender, Deseasonalizer

# External libraries
from prophet import Prophet
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

# Deep learning imports
import torch
import pytorch_lightning as pl
from pytorch_forecasting import TimeSeriesDataSet, NBeats, TemporalFusionTransformer
from torch.utils.data import DataLoader
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Flatten, Dense


def load_data(path):
    df = pd.read_csv(path, parse_dates=["date"], index_col="date")
    df.sort_index(inplace=True)
    return df['transactions']


def preprocess(y):
    # Ensure daily frequency
    y = y.asfreq('D')
    # Impute missing
    if y.isnull().any():
        y = y.interpolate(method='linear').fillna(method='ffill')
    # Detrend
    detr = Detrender()
    y_dt = detr.fit_transform(y)
    # Deseasonalize (weekly seasonality)
    deseas = Deseasonalizer(sp=7, model='multiplicative')
    y_ds = deseas.fit_transform(y_dt)
    return y, y_ds, detr, deseas


def split_data(y, test_size=30):
    y_train, y_test = temporal_train_test_split(y, test_size=test_size)
    fh = ForecastingHorizon(y_test.index, is_relative=False)
    return y_train, y_test, fh


def train_evaluate(y_train, y_test, fh):
    forecasts = {}
    train_times = {}

    # Naive
    m = NaiveForecaster(strategy='last')
    start = time.time()
    m.fit(y_train)
    train_times['Naive'] = time.time() - start
    forecasts['Naive'] = m.predict(fh)

    # OLS
    m = PolynomialTrendForecaster(degree=1)
    start = time.time(); m.fit(y_train)
    train_times['OLS'] = time.time() - start
    forecasts['OLS'] = m.predict(fh)

    # ARIMA (auto)
    m = AutoARIMA(sp=7, suppress_warnings=True)
    start = time.time(); m.fit(y_train)
    train_times['ARIMA'] = time.time() - start
    forecasts['ARIMA'] = m.predict(fh)

    # ETS (auto)
    m = AutoETS(auto=True, sp=7, n_jobs=-1)
    start = time.time(); m.fit(y_train)
    train_times['ETS'] = time.time() - start
    forecasts['ETS'] = m.predict(fh)

    # Theta
    m = ThetaForecaster(sp=7)
    start = time.time(); m.fit(y_train)
    train_times['Theta'] = time.time() - start
    forecasts['Theta'] = m.predict(fh)

    # Prophet
    train_df = y_train.reset_index().rename(columns={"date": "ds", "transactions": "y"})
    m = Prophet(daily_seasonality=True)
    start = time.time(); m.fit(train_df)
    train_times['Prophet'] = time.time() - start
    future = m.make_future_dataframe(periods=len(y_test), freq='D')
    pf = m.predict(future).set_index('ds')['yhat'].loc[y_test.index]
    forecasts['Prophet'] = pf

    # LightGBM reduction
    reg = LGBMRegressor(n_estimators=100, random_state=0)
    m = make_reduction(reg, window_length=14, strategy='recursive')
    start = time.time(); m.fit(y_train)
    train_times['LightGBM'] = time.time() - start
    forecasts['LightGBM'] = m.predict(fh)

    # CatBoost reduction
    reg = CatBoostRegressor(verbose=0, iterations=100, random_state=0)
    m = make_reduction(reg, window_length=14, strategy='recursive')
    start = time.time(); m.fit(y_train)
    train_times['CatBoost'] = time.time() - start
    forecasts['CatBoost'] = m.predict(fh)

    # N-BEATS (PyTorch Forecasting)
    data = pd.DataFrame({
        "time_idx": np.arange(len(y_train) + len(y_test)),
        "value": pd.concat([y_train, y_test]).values,
        "group": 'series'
    })
    max_h = len(y_test)
    dataset = TimeSeriesDataSet(
        data[data.time_idx < len(y_train)],
        time_idx='time_idx', target='value', group_ids=['group'],
        max_encoder_length=14, max_prediction_length=max_h,
        time_varying_known_reals=['time_idx'], allow_missing_timesteps=True
    )
    train_ds = dataset.to_dataloader(batch_size=32, shuffle=True)
    val_ds = dataset.to_dataloader(batch_size=32, shuffle=False)
    m = NBeats.from_dataset(dataset, learning_rate=1e-3, weight_decay=1e-2)
    trainer = pl.Trainer(max_epochs=20, enable_model_summary=False, logger=False, enable_checkpointing=False)
    start = time.time(); trainer.fit(m, train_ds, val_ds)
    train_times['NBEATS'] = time.time() - start
    preds = m.predict(val_ds).numpy().flatten()
    forecasts['NBEATS'] = pd.Series(preds, index=y_test.index)

    # TCN (Keras)
    # prepare data
    def make_xy(series, w):
        X, Y = [], []
        vals = series.values
        for i in range(len(vals) - w):
            X.append(vals[i:i+w]); Y.append(vals[i+w])
        return np.array(X), np.array(Y)
    w = 14
    X_train, Y_train = make_xy(y_train, w)
    X_train = X_train.reshape(-1, w, 1)
    m_keras = Sequential([
        Conv1D(32, 3, dilation_rate=1, activation='relu', input_shape=(w,1)),
        Conv1D(32, 3, dilation_rate=2, activation='relu'),
        Flatten(), Dense(1)
    ])
    m_keras.compile(optimizer='adam', loss='mse')
    start = time.time(); m_keras.fit(X_train, Y_train, epochs=10, batch_size=32, verbose=0)
    train_times['TCN'] = time.time() - start
    # recursive forecast
    last_window = y_train.values[-w:]
    preds = []
    for _ in range(len(y_test)):
        x = last_window.reshape(1, w, 1)
        yhat = m_keras.predict(x, verbose=0)[0,0]
        preds.append(yhat)
        last_window = np.roll(last_window, -1); last_window[-1] = yhat
    forecasts['TCN'] = pd.Series(preds, index=y_test.index)

    # TFT (PyTorch Forecasting)
    dataset_full = TimeSeriesDataSet(
        data[data.time_idx < len(y_train)],
        time_idx='time_idx', target='value', group_ids=['group'],
        max_encoder_length=14, max_prediction_length=max_h,
        time_varying_known_reals=['time_idx'], allow_missing_timesteps=True
    )
    train_ds = dataset_full.to_dataloader(batch_size=32, shuffle=True)
    val_ds = dataset_full.to_dataloader(batch_size=32, shuffle=False)
    m = TemporalFusionTransformer.from_dataset(
        dataset_full, learning_rate=1e-2, hidden_size=16, attention_head_size=4,
        dropout=0.1, hidden_continuous_size=8, output_size=max_h
    )
    trainer = pl.Trainer(max_epochs=15, enable_model_summary=False, logger=False, enable_checkpointing=False)
    start = time.time(); trainer.fit(m, train_ds, val_ds)
    train_times['TFT'] = time.time() - start
    preds = m.predict(val_ds).numpy().flatten()
    forecasts['TFT'] = pd.Series(preds, index=y_test.index)

    return forecasts, train_times


def evaluate_and_export(y_test, forecasts, train_times):
    # Evaluation
    mape_scores = {name: mean_absolute_percentage_error(y_test, y_pred)
                   for name, y_pred in forecasts.items()}
    # Assemble results
    results = pd.DataFrame({
        'Model': list(mape_scores.keys()),
        'MAPE': list(mape_scores.values()),
        'TrainTime(s)': [train_times[m] for m in mape_scores]
    }).sort_values('MAPE')
    print(results)

    # Visualization
    plt.figure(figsize=(10,6))
    plt.plot(y_test.index, y_test, label='Actual', color='black', linewidth=2)
    for name, y_pred in forecasts.items():
        plt.plot(y_test.index, y_pred, label=name)
    plt.legend(); plt.title('Actual vs Forecast'); plt.show()

    # Export
    out_df = pd.concat([y_test.rename('Actual')] + \
                       [f.rename(name) for name, f in forecasts.items()], axis=1)
    out_df.to_csv('model_forecasts.csv')
    results.to_csv('model_metrics.csv', index=False)
    print("Forecasts and metrics exported to CSV.")


def main():
    # 1. Load
    y = load_data('transactions_data.csv')

    # 2. Preprocess
    y_orig, y_transformed, detr, deseas = preprocess(y)

    # 3. Split
    y_train, y_test, fh = split_data(y_transformed, test_size=30)

    # 4-6. Train & Forecast
    forecasts, train_times = train_evaluate(y_train, y_test, fh)

    # 7-8. Evaluate & Export
    evaluate_and_export(y_test, forecasts, train_times)

if __name__ == '__main__':
    main()
