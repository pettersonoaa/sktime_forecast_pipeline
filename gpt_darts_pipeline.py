# pipeline_darts.py

"""
Pipeline for time series forecasting with Darts.

Steps:
1. Data ingestion from CSV
2. Pre-processing: resampling, missing-value handling
3. Train/test split or backtesting setup
4. Model definition: TFT, NBEATS, TCN, LightGBM, CatBoost, Prophet, Theta, ETS, ARIMA, OLS, NAIVE
5. Hyperparameter tuning
6. Forecast generation
7. Evaluation metrics: MAPE
8. Visualization, save and export
9. Dependency check and tests
"""

import os
import sys
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import joblib

import warnings
# Suppress feature name mismatch warnings from LightGBM
warnings.filterwarnings("ignore", message=".*valid feature names.*", category=UserWarning)
# Suppress statsmodels ConvergenceWarning during ETS fitting
from statsmodels.tools.sm_exceptions import ConvergenceWarning as StatsmodelsConvergenceWarning
warnings.filterwarnings("ignore", category=StatsmodelsConvergenceWarning)
# Suppress SARIMAX non-invertible starting MA parameters warning
warnings.filterwarnings("ignore", message="Non-invertible starting MA parameters found.*", category=UserWarning)
# Suppress numpy RuntimeWarning for mean of empty slice in Darts metrics
warnings.filterwarnings("ignore", message="Mean of empty slice.*", category=RuntimeWarning)
# Suppress numpy RuntimeWarning for overflow encountered in statsmodels Holt-Winters
warnings.filterwarnings("ignore", message="overflow encountered.*", category=RuntimeWarning)
# Suppress numpy RuntimeWarning for invalid value encountered in divide in statsmodels Holt-Winters
warnings.filterwarnings("ignore", message="invalid value encountered in divide.*", category=RuntimeWarning)

from darts import TimeSeries
from darts.dataprocessing.transformers import MissingValuesFiller
from darts.metrics import mape
from darts.models import (
    NaiveSeasonal,
    NaiveDrift,
    ExponentialSmoothing,
    ARIMA,
    Theta,
    Prophet,
    NBEATSModel,
    TCNModel,
    TFTModel,
    LightGBMModel,
    CatBoostModel,
    RegressionModel
)
from darts.utils.utils import ModelMode, SeasonalityMode
from darts.utils.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import ParameterGrid

# 1. Load and preprocess
def load_series(csv_path, time_col='time', value_col='value', freq='D'):
    df = pd.read_csv(csv_path)
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.groupby(time_col)[value_col].sum().asfreq('D').reset_index()
    return TimeSeries.from_dataframe(df, time_col, value_col, freq=freq)

def preprocess(series, freq='D'):
    # Resample to uniform frequency and fill missing values
    series = series.resample(freq)
    filler = MissingValuesFiller()
    return filler.transform(series)

# 2. Backtesting splitter
def backtesting_splitter(series, val_size=0.2):
    """
    Create a single train/validation split using train_test_split.
    Returns a list of (train, val) pairs for compatibility with tuner.
    """
    train, val = train_test_split(series, test_size=val_size)
    return [(train, val)]

# 3. Model definitions and parameter grids
model_classes = {
    'Naive Trend': (NaiveDrift, {}),
    'Naive Level': (NaiveSeasonal, {'K': [1]}),
    'Naive Seas007': (NaiveSeasonal, {'K': [7]}),
    'Naive Seas028': (NaiveSeasonal, {'K': [28]}),
    'Naive Seas364': (NaiveSeasonal, {'K': [364]}),
    'OLS': (RegressionModel, {'model':[LinearRegression()], 'lags':[12]}),
    'ETS': (
        ExponentialSmoothing,
        {
            'trend': [ModelMode.ADDITIVE, ModelMode.MULTIPLICATIVE, None],
            'seasonal': [SeasonalityMode.ADDITIVE, SeasonalityMode.MULTIPLICATIVE, None]
        }
    ),
    'ARIMA': (ARIMA, {'p':[1,2], 'd':[0,1], 'q':[1,2]}),
    'Theta': (Theta, {}),
    'Prophet': (Prophet, {}),
    'LightGBM': (LightGBMModel, {'lags':[12,30], 'output_chunk_length':[7]}),
    'CatBoost': (CatBoostModel, {'lags':[12,30], 'output_chunk_length':[7]}),
    'TCN': (TCNModel, {'input_chunk_length':[30], 'output_chunk_length':[14], 'n_epochs':[100]}),
    'NBEATS': (NBEATSModel, {'input_chunk_length':[30], 'output_chunk_length':[14], 'n_epochs':[100]}),
    'TFT': (TFTModel, {'input_chunk_length':[30], 'output_chunk_length':[14], 'n_epochs':[50], 'add_relative_index': [True]}),
}

# 4. Hyperparameter tuning
def tune_model(name, cls, grid, splits):
    """
    Tune model hyperparameters given predefined train/val splits.
    """
    best_score, best_params = float('inf'), None
    for params in ParameterGrid(grid):
        scores = []
        for train, val in splits:
            model = cls(**params)
            model.fit(train)
            pred = model.predict(len(val))
            scores.append(mape(val, pred))
        score = np.mean(scores)
        if score < best_score:
            best_score, best_params = score, params
    print(f'Best {name} params: {best_params} (MAPE={best_score:.2f})')
    return cls(**best_params)

# 5. Forecast generation with timing
def generate_forecasts(models, train, horizon):
    forecasts = {}
    training_times = {}
    for name, model in models.items():
        start_time = time.perf_counter()
        fitted = model.fit(train)
        end_time = time.perf_counter()
        training_times[name] = end_time - start_time
        forecasts[name] = fitted.predict(horizon)
    return forecasts, training_times

# 6. Evaluation
def evaluate(models, train, test):
    return {name: mape(test, model.fit(train).predict(len(test))) for name, model in models.items()}

# 7. Visualization
def plot_forecasts(train, test, forecasts):
    plt.figure(figsize=(16,6))
    train.plot(label='train')
    test.plot(label='test')
    for name, fc in forecasts.items():
        fc.plot(label=name)
    plt.legend()
    plt.show()

# 8. Main pipeline
def main():
    series = load_series("data/transactions.csv", time_col='date', value_col='transactions')
    series = preprocess(series)
    splits = backtesting_splitter(series, val_size=0.2)
    tuned_models = {name: tune_model(name, cls, grid, splits) for name, (cls, grid) in model_classes.items()}

    # Save best models for later use
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_dir = os.path.join('models', f'gpt_darts_pipeline_{timestamp}')
    os.makedirs(model_dir, exist_ok=True)
    for name, model in tuned_models.items():
        model_filename = f"{name}_{timestamp}.pth"
        model_path = os.path.join(model_dir, model_filename)
        try:
            model.save(model_path)
            print(f"Saved {name} model to {model_path}")
        except AttributeError:
            # some models might not support save
            joblib_filename = model_filename.replace('.pth', '.joblib')
            joblib_path = os.path.join(model_dir, joblib_filename)
            joblib.dump(model, joblib_path)
            print(f"Joblib-saved {name} model to {model_path.replace('.pth', '.joblib')}")

    train, test = splits[0]
    forecasts, times = generate_forecasts(tuned_models, train, len(test))
    results = evaluate(tuned_models, train, test)
    print('MAPE results:', results)
    print('Training times (seconds):')
    for name, t in times.items():
        print(f"  {name}: {t:.2f}s")

    plot_forecasts(train, test, forecasts)

    # Save outputs to CSV
    forecast_dir = os.path.join('forecast', f'gpt_darts_pipeline_{timestamp}')
    os.makedirs(forecast_dir, exist_ok=True)
    for name, fc in forecasts.items():
        fc.to_dataframe().to_csv(f'{forecast_dir}/forecast_{name}.csv')

    evaluation_dir = os.path.join('evaluation', f'gpt_darts_pipeline_{timestamp}')
    os.makedirs(evaluation_dir, exist_ok=True)
    pd.DataFrame(results, index=['MAPE']).to_csv(f'{evaluation_dir}/evaluation.csv')

# 9. Test cases
import pytest

def test_load_series(tmp_path):
    tmp_file = tmp_path / 'tmp.csv'
    pd.DataFrame({'time': pd.date_range('2021-01-01', periods=3), 'value': [1,2,3]}).to_csv(tmp_file, index=False)
    series = load_series(str(tmp_file))
    assert hasattr(series, 'time_index'), 'load_series should return a TimeSeries object'


def test_preprocess():
    idx = pd.date_range('2021-01-01', periods=5)
    series = TimeSeries.from_times_and_values(idx, [1, None, 3, None, 5])
    filled = preprocess(series)
    values = np.array(filled.values()).flatten()
    assert not np.isnan(values).any(), 'preprocess should fill missing values'

if __name__ == '__main__':
    main()
