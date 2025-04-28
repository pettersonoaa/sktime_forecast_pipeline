import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import joblib
import os
from datetime import datetime
import seaborn as sns
import torch
import torch.nn as nn
from holidays import country_holidays, financial_holidays

from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor

from sktime.split import SlidingWindowSplitter
from sktime.utils.plotting import plot_series, plot_windows
from sktime.performance_metrics.forecasting import mean_absolute_percentage_error
from sktime.transformations.compose import OptionalPassthrough
from sktime.transformations.series.adapt import TabularToSeriesAdaptor
from sktime.transformations.series.detrend import Deseasonalizer, Detrender
from sktime.transformations.series.difference import Differencer
from sktime.transformations.series.boxcox import LogTransformer
from sktime.transformations.series.holiday import HolidayFeatures
from sktime.transformations.series.date import DateTimeFeatures
from sktime.transformations.series.dummies import SeasonalDummiesOneHot
from sktime.forecasting.model_selection import temporal_train_test_split, ForecastingGridSearchCV
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.compose import AutoEnsembleForecaster, EnsembleForecaster, TransformedTargetForecaster, make_reduction
from sktime.forecasting.neuralforecast import NeuralForecastLSTM, NeuralForecastTCN
from sktime.forecasting.pytorchforecasting import PytorchForecastingTFT, PytorchForecastingNBeats, PytorchForecastingNHiTS, PytorchForecastingDeepAR
from sktime.forecasting.time_llm import TimeLLMForecaster
from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.trend import PolynomialTrendForecaster
from sktime.forecasting.darts import DartsRegressionModel, DartsXGBModel, DartsLinearRegressionModel
from sktime.forecasting.fbprophet import Prophet
from sktime.forecasting.statsforecast import (
    StatsForecastAutoARIMA, 
    StatsForecastAutoTBATS, 
    StatsForecastAutoETS, 
    StatsForecastAutoTheta, 
    StatsForecastMSTL,
    StatsForecastAutoCES
)

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
# # Suppress warnings from TBATS
warnings.filterwarnings("ignore", message=".*Data contains zero.*", category=FutureWarning)  #RuntimeWarning, UserWarning, FutureWarning
warnings.filterwarnings("ignore", message=".*force_all_finite.*", category=UserWarning)  #RuntimeWarning, UserWarning, FutureWarning
# # Suppress warnings from ARIMA
warnings.filterwarnings("ignore", message=".*possible convergence problem.*", category=UserWarning)  #RuntimeWarning, UserWarning, FutureWarning




def load_series(csv_path, time_col='date', value_col='value', freq='D'):
    df = pd.read_csv(csv_path, usecols=[time_col, value_col], index_col=0, parse_dates=[time_col])
    df = df.groupby(time_col)[value_col].sum().asfreq(freq)
    series = pd.Series(df, index=df.index, name='y')
    series = series.interpolate(method='time')
    return series

def holidays_features(data, country='BR', horizon_years=5):
    ch = country_holidays(country, years=list(range(data.index.year.min(), data.index.year.max()+horizon_years)))
    series = pd.Series(ch)
    df = pd.DataFrame({'ds': series.index, 'holiday': series.values})
    df['lower_window'] = -5
    df['upper_window'] = 5
    df['ds'] = pd.to_datetime(df['ds'])
    return df

def mape_metric(y_true, y_pred, month_transform=True):
    if month_transform:
        y_true = y_true.groupby(y_true.index.month).sum()
        y_pred = y_pred.groupby(y_pred.index.month).sum()
    return np.mean(np.abs((y_true - y_pred) / y_true) * 100)

def create_model_configs(y_train):
    """Step 2: Define models and their parameter grids."""
    return [

        ##  stats family
        {
            "name": "PTN",
            "forecaster": PolynomialTrendForecaster(degree=2),  # Quadratic trend
            "params": {
                "forecaster__degree": [1, 2, 3],  # Linear, quadratic, cubic

                "scaler__passthrough": [True, False],
                "deseasonalize_7__passthrough": [False],
                "deseasonalize_365__passthrough": [False],
                "detrend__passthrough": [True],
                "ln__passthrough": [True, False],
            },
            "family": 'stats',
        },
        {
            "name": "OLS",
            "forecaster": make_reduction(
                LinearRegression(),
                strategy="recursive"
            ),
            "params": {
                "forecaster__window_length": [7, 28, 30, 364, 365],
                "forecaster__estimator__fit_intercept": [True, False],

                "scaler__passthrough": [True, False],
                "deseasonalize_7__passthrough": [True, False],
                "deseasonalize_365__passthrough": [True, False],
                "detrend__passthrough": [True, False],
                "ln__passthrough": [True, False],
            },
            "family": 'stats',
        },
        {
            "name": "Theta",
            "forecaster": StatsForecastAutoTheta(), 
            "params": {
                "forecaster__season_length": [7, 28],
                "forecaster__decomposition_type": ['multiplicative', 'additive'],

                "scaler__passthrough": [True, False],
                "deseasonalize_7__passthrough": [True],
                "deseasonalize_365__passthrough": [True],
                "detrend__passthrough": [True],
                "ln__passthrough": [True, False],
            },
            "family": 'stats',
        },
        {
            "name": "ETS",
            "forecaster": StatsForecastAutoETS(),
            "params": {
                "forecaster__season_length": [7, 28],

                "scaler__passthrough": [True, False],
                "deseasonalize_7__passthrough": [True],
                "deseasonalize_365__passthrough": [True],
                "detrend__passthrough": [True],
                "ln__passthrough": [True, False],
            },
            "family": 'stats',
        },
        {
            "name": "Prophet",
            "forecaster": Prophet(holidays=holidays_features(y_train)),
            "params": {
                "forecaster__uncertainty_samples": [50],
                "forecaster__changepoint_range": [round(365/int(len(y_train))), 0.8],  # =0.8 Proportion of history in which trend changepoints will be estimated
                "forecaster__changepoint_prior_scale": [0.03, 0.05, 0.08],  # =0.05 Flexibility of trend - Large values will allow many changepoints, small values will allow few changepoints
                "forecaster__seasonality_mode": ['multiplicative', 'additive'], # ='additive'
                "forecaster__seasonality_prior_scale": [6, 10, 12],   # =10 Flexibility of seasonality 

                "scaler__passthrough": [True],
                "deseasonalize_7__passthrough": [True],
                "deseasonalize_365__passthrough": [True],
                "detrend__passthrough": [True],
                "ln__passthrough": [True, False],
            },
            "family": 'stats',
        },
        {
            "name": "CES",
            "forecaster": StatsForecastAutoCES(),
            "params": {
                "forecaster__season_length": [7, 28],

                "scaler__passthrough": [True, False],
                "deseasonalize_7__passthrough": [True],
                "deseasonalize_365__passthrough": [True],
                "detrend__passthrough": [True],
                "ln__passthrough": [True, False],
            },
            "family": 'stats',
        },
        {
            "name": "ARIMA",
            "forecaster": StatsForecastAutoARIMA(),
            "params": {
                "forecaster__sp": [7],
                "forecaster__seasonal": [True],
                "forecaster__trend": [True],
                "forecaster__with_intercept": [True],
                "forecaster__method": ['lbfgs'],
                "forecaster__stepwise": [True],
                # "forecaster__trace": [True],

                "scaler__passthrough": [True, False],
                "deseasonalize_7__passthrough": [True, False],
                "deseasonalize_365__passthrough": [True, False],
                "detrend__passthrough": [True, False],
                "ln__passthrough": [True, False],
            },
            "family": 'stats',
        },
        {
            "name": "TBATS",
            "forecaster": StatsForecastAutoTBATS(seasonal_periods=7), 
            "params": {
                "forecaster__seasonal_periods": [7, 28],

                "scaler__passthrough": [True, False],
                "deseasonalize_7__passthrough": [True],
                "deseasonalize_365__passthrough": [True],
                "detrend__passthrough": [True],
                "ln__passthrough": [True, False],
            },
            "family": 'stats',
        },
        {
            "name": "Ensemble_Stats",
            "forecaster": AutoEnsembleForecaster(
                forecasters=[
                    ('Prophet', Prophet(holidays=holidays_features(y_train), uncertainty_samples=50, seasonality_mode='multiplicative')),
                    # ('Theta', StatsForecastAutoTheta(season_length=28, decomposition_type='multiplicative')),
                    # ('TBATS', StatsForecastAutoTBATS(seasonal_periods=28)),
                    ('PTN', PolynomialTrendForecaster(degree=3))
                ],
                # test_size=0.9,
                random_state=42,
                n_jobs=-1
            ),
            "params": {
                "scaler__passthrough": [True, False],
                "deseasonalize_7__passthrough": [True],
                "deseasonalize_365__passthrough": [True],
                "detrend__passthrough": [True],
                "ln__passthrough": [True, False],
            },
            "family": 'stats',
        },



        ##  deep learning family
        {
            'name': 'TCN',
            'forecaster': NeuralForecastTCN(
                input_size=365,  # default -1 (all history)
                local_scaler_type='robust',
                scaler_type='robust',
                context_size=7, # default 10
                decoder_layers=3, # default 2
                max_steps=1000, # default 1000
                batch_size=32, # default 32
                learning_rate=0.01, # default 0.001   between 0 and 1
                random_seed=42
            ),
            "params": {},
            "family": 'deep learning',
        },
        {
            'name': 'LSTM',
            'forecaster': NeuralForecastLSTM(
                input_size=365,
                local_scaler_type='robust',
                scaler_type='robust',
                # futr_exog_list=['dayofweek', 'month', 'year'],
                max_steps=50,
                batch_size=32,
                # early_stop_patience_steps=100,
                # val_check_steps=10,
                random_seed=42
            ),
            "params": {},
            "family": 'deep learning',
        },
        {
            'name': 'TimeLLM',
            'forecaster': TimeLLMForecaster(
                task_name='long_term_forecast', #  default='long_term_forecast'    'short_term_forecast'
                pred_len=28+7, # default=24    Forecast horizon - number of time steps to predict.
                seq_len=96, # default=96     Length of input sequence.
                llm_model='GPT2', #[‘GPT2’, ‘LLAMA’, ‘BERT’]
                llm_layers=3, # default=3    Number of transformer layers to use from LLM.
                patch_len=16, # default=16   Length of patches for patch embedding.
                stride=8, # default=8        Stride between patches.
                d_model=128, # default=128   Model dimension.
                d_ff=128, # default=128      Feed-forward dimension.
                n_heads=4, # default=4       Number of attention heads.
                dropout=0.1, # default=0.1    Dropout rate
                device='cuda' # default='cuda' if available else 'cpu'
            ),
            "params": {},
            "family": 'deep learning',
        },
        {
            'name': 'TFT',
            'forecaster': PytorchForecastingTFT(
                trainer_params={
                    "max_epochs": 50,  # for quick test
                    "limit_train_batches": 20,  # for quick test
                }
            ),
            "params": {},
            "family": 'deep learning',
        },
        {
            'name': 'NBeats',
            'forecaster': PytorchForecastingNBeats(
                trainer_params={
                    "max_epochs": 50,  # for quick test
                    "limit_train_batches": 20,  # for quick test
                }
            ),
            "params": {},
            "family": 'deep learning',
        },
        {
            'name': 'NHiTS',
            'forecaster': PytorchForecastingNHiTS(
                trainer_params={
                    "max_epochs": 50,  # for quick test
                    "limit_train_batches": 20,  # for quick test
                }
            ),
            "params": {},
            "family": 'deep learning',
        },
        {
            'name': 'DeepAR',
            'forecaster': PytorchForecastingDeepAR(
                trainer_params={
                    "max_epochs": 50,  # for quick test
                    "limit_train_batches": 20,  # for quick test
                }
            ),
            "params": {},
            "family": 'deep learning',
        },
        
    ]

def find_best_models(y_train, y_test, models):
    """Step 3: Find best parameters for each model using grid search."""
    best_models = []
    
    # Forecast Horizon
    fh = ForecastingHorizon(y_test.index, is_relative=False)

    # Create cross-validation strategy
    cv = SlidingWindowSplitter(
        initial_window=28*12*4,    # 4 years training
        window_length=28*12*3,     # 3 years validation
        step_length=int(len(y_test)),        # 2 month step
        fh=list(range(1, int(len(y_test))))
    )
    # # Visualize the CV splits
    # fig, ax = plt.subplots(figsize=(10, 6))
    # plt.subplots(figsize=(10, 6))
    # plot_windows(cv, y_train, title="Sliding Window Cross-validation")

    
    
    for model in models:
        print(f"\nEvaluating {model['name']}...")

        # Start timing
        start_time = time.time()

        if 'stats' in model['family']:
            # Create pipeline
            pipe = TransformedTargetForecaster([
                ("ln", OptionalPassthrough(LogTransformer())),
                # ("diff", OptionalPassthrough(Differencer(na_handling='drop_na'))),
                ("deseasonalize_7", OptionalPassthrough(Deseasonalizer(sp=7))),
                ("deseasonalize_365", OptionalPassthrough(Deseasonalizer(sp=365))),
                ("detrend", OptionalPassthrough(Detrender())),
                ("scaler", OptionalPassthrough(TabularToSeriesAdaptor(RobustScaler()))),
                # ("seas", OptionalPassthrough(SeasonalDummiesOneHot())),
                # ("holidays", OptionalPassthrough(TabularToSeriesAdaptor(HolidayFeatures(
                #     calendar=country_holidays(country="BR"),
                #     holiday_windows={
                #         "Christmas": (5, 3), 
                #         "New Year": (2, 5), 
                #         "Carnival": (3, 3), 
                #         "Good Friday": (2, 2),
                #         "Tiradentes' Day": (2, 2),
                #         "Worker's Day": (2, 2),
                #         "Independence Day": (2, 2),
                #         "Our Lady of Aparecida": (2, 2),
                #         "All Souls' Day": (2, 2),
                #         "Republic Proclamation Day": (2, 2),
                #         "National Day of Zumbi and Black Awareness": (2, 2),
                #     }
                # )))),
                # ("calendar", OptionalPassthrough(DateTimeFeatures(ts_freq="D", manual_selection=[
                #     "month_of_year", 
                #     "day_of_week", 
                #     "day_of_month", 
                #     "week_of_year", 
                #     "day_of_year", 
                #     'is_weekend'
                # ]))),	
                ("forecaster", model["forecaster"])
            ])
            
            # Perform grid search
            gscv = ForecastingGridSearchCV(
                forecaster=pipe,
                param_grid=[model["params"]],
                cv=cv,
                return_n_best_forecasters=3,
                backend="loky",  # Parallel backend
                backend_params={"n_jobs": -1},  # Number of parallel jobs
            )
            
            try:
                # fitting
                gscv.fit(y=y_train)
                y_pred = gscv.predict(fh=fh)

                best_forecaster = gscv.best_forecaster_
                best_params = gscv.best_params_
                # best_score = gscv.best_score_
                best_score = mape_metric(y_test, y_pred)
                n_best_score = gscv.n_best_scores_
                n_best_forecasters = gscv.n_best_forecasters_
            
            except Exception as e:
                print(f"Error with {model['name']}: {str(e)}")
                computation_time = time.time() - start_time
                print(f"Failed after: {computation_time:.2f} seconds")
                continue
            

        else:
            

            try:
                # fitting
                model['forecaster'].fit(y=y_train, fh=fh)
                y_pred = model['forecaster'].predict(fh=fh)

                best_forecaster = model['forecaster']
                best_params = model['forecaster'].get_fitted_params()
                best_score = mape_metric(y_test, y_pred)
                n_best_score = [best_score]
                n_best_forecasters = [('1', best_params)]

            except Exception as e:
                print(f"Error with {model['name']}: {str(e)}")
                computation_time = time.time() - start_time
                print(f"Failed after: {computation_time:.2f} seconds")
                continue


        # Add debug print
        print(f"Model: {model['name']}")
        print(f"Best parameters: {best_params}")
        print(f"Best score: {best_score:.2f}%")
        print(f'n Best scores: {n_best_score}')
        [print(f"{rank_model[0]}# best forecaster: {rank_model[1]}") for rank_model in n_best_forecasters]

        # Calculate computation time
        computation_time = time.time() - start_time
        
        # Store best model and timing
        best_models.append({
            "name": model["name"],
            "model": best_forecaster,
            "params": best_params,
            "score": best_score,
            "test": y_test,
            "predictions": y_pred,
            "computation_time": computation_time,
            "family": model["family"],
            
        })

        print(f"Computation time: {computation_time:.2f} seconds")
        # print(f"Computation time: {gscv.get("computation_time", 0):.2f} seconds")

        
        
    return best_models

def evaluate_stability(best_models, y, periods=6):
    """Step 4: Calculate MAPE over multiple periods for stability."""
    stability_results = []

    from dateutil.relativedelta import relativedelta
    cutoffs =[]
    for i in range(periods, 0, -1):
        cutoff = y.index.max() - relativedelta(months=i)  # Last month of data minus periods
        cutoffs.append(cutoff)
    
    for model in best_models:
        print(f"\nEvaluating stability of {model['name']}...")
        mape_scores = []
        test_values = []  # Store actual test values
        pred_values = []  # Store predictions
        test_indices = []  # Store test period indices
        
        for cutoff in cutoffs:

            test_size = 7*(4+4+1) # 2 meses (7 dias * (4 semanas + 4 semanas + 1 semana)) = 63 dias
            train_start_date = cutoff - relativedelta(years=3, days=test_size) 
            y_period = y[train_start_date:cutoff]
            train_size = int(len(y_period)) - test_size
            y_train, y_test = y_period[:train_size], y_period[train_size:]
            fh = ForecastingHorizon(y_test.index, is_relative=False)
            
            try:
                # Fit and predict
                if 'stats' in model['family']:
                    model['model'].fit(y_train)
                else:
                    model['model'].fit(y_train, fh=fh)
                
                y_pred = model['model'].predict(fh)
                mape = mape_metric(y_test, y_pred)
                mape_scores.append(mape)

                # Store actual and predicted values
                test_values.append(y_test)
                pred_values.append(y_pred)
                test_indices.append(y_test.index)

                print(f"Period {cutoff.date()} MAPE: {mape:.2f}%")
            except Exception as e:
                print(f"Error in period {cutoff.date()}: {str(e)}")
                mape_scores.append(np.nan)
                test_values.append(None)
                pred_values.append(None)
                test_indices.append(None)
        
        stability_results.append({
            "name": model["name"],
            "mape_scores": mape_scores,
            "mape_mean": np.nanmean(mape_scores),
            "mape_std": np.nanstd(mape_scores),
            "test_values": test_values,
            "pred_values": pred_values,
            "test_indices": test_indices,
            "computation_time": model.get("computation_time", 0)
        })
    
    return stability_results

def plot_results(best_models, stability_results):
    """Step 5: Create enhanced visualizations of results."""
    # Set style
    sns.set_style("whitegrid")
    sns.set_context("paper")
    
    # Create figure with adjusted size and DPI
    fig = plt.figure(figsize=(17, 9), dpi=90)
    
    # Color palette
    colors = plt.cm.Set2(np.linspace(0, 1, len(stability_results)))
    
    # Plot 1: Stability over time with enhanced styling
    ax1 = plt.subplot(2, 2, 1)
    periods = range(1, len(stability_results[0]["mape_scores"]) + 1)
    
    for idx, result in enumerate(stability_results):
        ax1.plot(periods, result["mape_scores"], 
                marker='o', 
                linestyle='-',
                linewidth=2,
                markersize=8,
                color=colors[idx],
                label=f"{result['name']}\nμ={result['mape_mean']:.1f}%, σ={result['mape_std']:.1f}%")
        
        # Add error bands
        mean = result['mape_mean']
        std = result['mape_std']
        ax1.fill_between(periods, 
                        [mean - std] * len(periods), 
                        [mean + std] * len(periods), 
                        color=colors[idx], 
                        alpha=0.2)
    
    ax1.set_xlabel('Update Period', fontsize=12)
    ax1.set_ylabel('MAPE (%)', fontsize=12)
    ax1.set_title('Model Stability Over Time', fontsize=14, pad=20)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    

    # Plot 2: Actual vs Predicted with enhanced styling
    ax2 = plt.subplot(2, 1, 2)

    just_once = True
    for idx, result in enumerate(best_models):
        
        # Plot actual values
        if just_once:
            ax2.plot(result["test"].index, 
                    result["test"], 
                    'o-',
                    linewidth=4,
                    markersize=8,
                    color='black',
                    label='Actual Values')
            just_once = False
        
        # Plot predictions with uncertainty band
        y_pred = result["predictions"]
        ax2.plot(result["test"].index, 
                y_pred,
                '--',
                linewidth=2,
                color=colors[idx],
                label=f'{result["name"]} (Predicted)')
        
        # Add error bands (using MAPE as uncertainty)
        mape = result['score'] / 100
        ax2.fill_between(result["test"].index,
                        y_pred * (1 - mape),
                        y_pred * (1 + mape),
                        color=colors[idx],
                        alpha=0.2)
            
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_ylabel('Value', fontsize=12)
    ax2.set_title('Actual vs Predicted Values', fontsize=14, pad=20)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    
    # Rotate x-axis labels for better readability
    plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
    
    # Adjust layout to prevent overlapping
    plt.tight_layout()
    
    # Add a suptitle
    fig.suptitle('Time Series Forecasting Model Comparison', fontsize=16, y=1.02)
    
    

    #### Add computation time analysis
    timing_data = {
        'Model': [],
        'Time (seconds)': [],
        'Est. Cost/365 runs ($)': []  # Assuming ml.m5.xlarge instance
    }
    
    # AWS SageMaker ml.m5.xlarge cost per second (approximate)
    SAGEMAKER_COST_PER_SECOND = 0.0000642  # $0.23 per hour
    TIME_SPENT_TO_SET_UP_SAGEMAKER = 60 * 5  # 5 minutes
    TIME_SPENT_TO_SET_UP_VENV = 60 * 5  # 5 minutes
    setup_time = TIME_SPENT_TO_SET_UP_SAGEMAKER + TIME_SPENT_TO_SET_UP_VENV

    for result in stability_results:
        model_name = result['name']
        computation_time = result.get('computation_time', 0) 
        # Estimate cost for 365 runs
        estimated_cost = ((computation_time + setup_time) * SAGEMAKER_COST_PER_SECOND * 365)
        
        timing_data['Model'].append(model_name)
        timing_data['Time (seconds)'].append(f"{computation_time:.2f}")
        timing_data['Est. Cost/365 runs ($)'].append(f"{estimated_cost:.2f}")
    
    # Create and display timing analysis
    timing_df = pd.DataFrame(timing_data)
    print("\n\nComputation Time and Cost Analysis (ml.m5.xlarge):\n")
    print(timing_df.to_string(index=False))
    
    # # Add computation time plot
    # plt.figure(figsize=(12, 6))
    # times = [float(t) for t in timing_df['Time (seconds)'].values]
    # costs = [float(c) for c in timing_df['Est. Cost/365 runs ($)'].values]
    
    # # Bar plot for computation times
    # plt.subplot(1, 2, 1)
    # bars = plt.bar(timing_df['Model'], times)
    # plt.xticks(rotation=45, ha='right')
    # plt.ylabel('Computation Time (seconds)')
    # plt.title('Model Training Time Comparison')
    
    # # Add value labels on bars
    # for bar in bars:
    #     height = bar.get_height()
    #     plt.text(bar.get_x() + bar.get_width()/2., height,
    #             f'{height:.1f}s',
    #             ha='center', va='bottom')
    
    # # Bar plot for estimated costs
    # plt.subplot(1, 2, 2)
    # bars = plt.bar(timing_df['Model'], costs)
    # plt.xticks(rotation=45, ha='right')
    # plt.ylabel('Estimated Cost per Year runs ($)')
    # plt.title('Cost Projection (AWS SageMaker)')
    
    # # Add value labels on bars
    # for bar in bars:
    #     height = bar.get_height()
    #     plt.text(bar.get_x() + bar.get_width()/2., height,
    #             f'${height:.2f}',
    #             ha='center', va='bottom')
    
    
    #### Create a third figure for Cost vs MAPE scatter plot
    # plt.figure(figsize=(17, 9), dpi=90)
    # fig3 = plt.figure(figsize=(14, 8), dpi=90)
    # fig3, ax3 = plt.subplots(1, 1)
    ax3 = plt.subplot(2, 2, 2)
    
    # Extract data for scatter plot
    costs = []
    mapes = []
    names = []
    color = []
    for idx, result in enumerate(stability_results):
    # for result in stability_results:
        computation_time = result.get('computation_time', 0)
        estimated_cost = ((computation_time + setup_time) * SAGEMAKER_COST_PER_SECOND * 365)
        costs.append(estimated_cost)
        mapes.append(100-result['mape_mean']-result['mape_std'])
        names.append(result['name'])
        color.append(colors[idx])
    
    # Create scatter plot
    ax3.scatter(costs, mapes, c=color, s=100)
    
    # Add labels for each point
    for idx, name in enumerate(names):
        ax3.annotate(name, 
                    (costs[idx], mapes[idx]),
                    xytext=(5, 5),
                    textcoords='offset points',
                    fontsize=10,
                    color=colors[idx],
                    bbox=dict(facecolor='white', edgecolor='none', alpha=0.01))
    
    ax3.set_xlabel('Estimated Annual Cost ($)', fontsize=12)
    ax3.set_ylabel('Performance [100% - mean(MAPE) - std(MAPE)] (%)', fontsize=12)
    ax3.set_title('Performance-Cost Trade-off', fontsize=14, pad=20)
    ax3.grid(True, linestyle='--', alpha=0.7)
    

    # set x axis to log
    ax3.set_xscale('log')
   
    # Normalize and plot reference line
    ax3.plot([min(costs), max(costs)], [min(mapes), max(mapes)], 
             'r--', alpha=0.3, label='Trade-off Reference')
    
    # Add legend
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    
    # Adjust layout
    plt.tight_layout()
    
    # Show plot
    plt.show()
    
    # Print summary statistics in a formatted table
    print("\n\nStability Analysis Results:")
    summary_data = {
        'Model': [],
        'Mean MAPE (%)': [],
        'Std Dev (%)': [],
        'CV (%)': []
    }
    
    for result in stability_results:
        summary_data['Model'].append(result['name'])
        summary_data['Mean MAPE (%)'].append(f"{result['mape_mean']:.2f}")
        summary_data['Std Dev (%)'].append(f"{result['mape_std']:.2f}")
        cv = (result['mape_std']/result['mape_mean']*100)
        summary_data['CV (%)'].append(f"{cv:.2f}")
    
    summary_df = pd.DataFrame(summary_data)
    print("\n", summary_df.to_string(index=False))



def save_models(best_models, stability_results, base_path="models"):
    """Save best models, their parameters, and performance metrics."""
    # Create models directory if it doesn't exist
    os.makedirs(base_path, exist_ok=True)
    
    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_path = os.path.join(base_path, f"run_{timestamp}")
    os.makedirs(run_path, exist_ok=True)
    
    saved_models = []
    for model_dict in best_models:
        model_name = model_dict["name"]
        model_path = os.path.join(run_path, f"{model_name}")
        os.makedirs(model_path, exist_ok=True)
        
        # Save the model
        joblib.dump(model_dict["model"], 
                   os.path.join(model_path, "model.joblib"))
        
        # Save metadata
        metadata = {
            "name": model_name,
            "params": model_dict["params"],
            "score": model_dict["score"],
            "computation_time": model_dict["computation_time"]
        }
        joblib.dump(metadata, 
                   os.path.join(model_path, "metadata.joblib"))
        
        saved_models.append({
            "name": model_name,
            "path": model_path
        })
    
    # Save stability results
    joblib.dump(stability_results, 
                os.path.join(run_path, "stability_results.joblib"))
    
    # Save model paths index
    joblib.dump(saved_models, 
                os.path.join(run_path, "model_index.joblib"))
    
    print(f"\nModels saved in: {run_path}")
    return run_path

def load_models(run_path):
    """Load saved models and their metadata."""
    # Load model index
    model_index = joblib.load(os.path.join(run_path, "model_index.joblib"))
    
    loaded_models = []
    for model_info in model_index:
        model_path = model_info["path"]
        
        # Load model and metadata
        model = joblib.load(os.path.join(model_path, "model.joblib"))
        metadata = joblib.load(os.path.join(model_path, "metadata.joblib"))
        
        loaded_models.append({
            "name": metadata["name"],
            "model": model,
            "params": metadata["params"],
            "score": metadata["score"],
            "computation_time": metadata["computation_time"]
        })
    
    # Load stability results
    stability_results = joblib.load(
        os.path.join(run_path, "stability_results.joblib"))
    
    return loaded_models, stability_results

def main():
    # Step 1: Load data
    # y = create_daily_data()
    y = load_series(csv_path='data/transactions.csv', time_col='date', value_col='transactions')
    y_train, y_test = temporal_train_test_split(y, test_size=7*(4+4+1))
    
    # Step 2: Create model configurations
    models = create_model_configs(y_train)
    
    # Step 3: Find best models
    best_models = find_best_models(y_train, y_test, models)
    
    # Step 4: Evaluate stability
    stability_results = evaluate_stability(best_models, y_train, periods=2)
   
    # Step 5: Plot and print results
    plot_results(best_models, stability_results)

    # Step 6: Save models and results
    run_path = save_models(best_models, stability_results)
    
    # Example of loading models later
    print("\nLoading saved models for verification...")
    loaded_models, loaded_results = load_models(run_path)
    print(f"Loaded {len(loaded_models)} models from {run_path}")

if __name__ == "__main__":
    main()