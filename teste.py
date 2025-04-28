import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sktime.forecasting.compose import AutoEnsembleForecaster, TransformedTargetForecaster, make_reduction
from sktime.transformations.series.holiday import HolidayFeatures
from sktime.forecasting.trend import PolynomialTrendForecaster
from sktime.transformations.series.detrend import Detrender, Deseasonalizer
from sktime.transformations.series.boxcox import LogTransformer
from sklearn.preprocessing import MinMaxScaler, PowerTransformer, RobustScaler
from sktime.utils.plotting import plot_series
from sktime.transformations.series.adapt import TabularToSeriesAdaptor
from sktime.transformations.series.dummies import SeasonalDummiesOneHot
from sktime.transformations.series.difference import Differencer
from sktime.forecasting.statsforecast import StatsForecastAutoTheta
from sktime.forecasting.statsforecast import StatsForecastAutoARIMA
from sktime.forecasting.naive import NaiveForecaster
from sktime.transformations.series.impute import Imputer
from sktime.forecasting.fbprophet import Prophet
from sklearn.linear_model import LinearRegression
from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.darts import DartsRegressionModel, DartsXGBModel, DartsLinearRegressionModel
from sklearn.linear_model import LinearRegression
from sktime.forecasting.ets import AutoETS
from sktime.forecasting.statsforecast import StatsForecastAutoARIMA, StatsForecastAutoTBATS, StatsForecastAutoETS, StatsForecastMSTL
from sktime.forecasting.tbats import TBATS
from sktime.forecasting.trend import PolynomialTrendForecaster
from sktime.forecasting.statsforecast import StatsForecastAutoTheta, StatsForecastAutoCES
from sktime.forecasting.fbprophet import Prophet
from sktime.forecasting.neuralforecast import NeuralForecastLSTM, NeuralForecastTCN
from sktime.forecasting.compose import make_reduction
from sklearn.neural_network import MLPRegressor
from sktime.forecasting.compose import EnsembleForecaster
from sktime.forecasting.pytorchforecasting import PytorchForecastingTFT, PytorchForecastingNBeats, PytorchForecastingNHiTS, PytorchForecastingDeepAR
from sktime.forecasting.base import ForecastingHorizon
from sklearn.model_selection import train_test_split
from sktime.forecasting.time_llm import TimeLLMForecaster
from sktime.forecasting.compose import StackingForecaster
from sktime.forecasting.neuralforecast import NeuralForecastLSTM, NeuralForecastTCN
from holidays import country_holidays, financial_holidays

import warnings
# Suppress warnings from TBATS
warnings.filterwarnings("ignore", message=".*force_all_finite.*", category=FutureWarning)  #RuntimeWarning, UserWarning, FutureWarning
warnings.filterwarnings("ignore", message=".*Data contains zero.*", category=UserWarning)  #RuntimeWarning, UserWarning, FutureWarning
# Suppress warnings from ARIMA
warnings.filterwarnings("ignore", message=".*possible convergence problem.*", category=UserWarning)  #RuntimeWarning, UserWarning, FutureWarning

def load_series(csv_path, time_col='date', value_col='value', freq='D'):
    df = pd.read_csv(csv_path, usecols=[time_col, value_col], index_col=0, parse_dates=[time_col])
    df = df.groupby(time_col)[value_col].sum().asfreq(freq)
    series = pd.Series(df, index=df.index, name='y')
    series = series.interpolate(method='time')
    return series

def mape_metric(y_true, y_pred, month_transform=True):
    if month_transform:
        y_true = y_true.groupby(y_true.index.month).sum()
        y_pred = y_pred.groupby(y_pred.index.month).sum()
    return np.mean(np.abs((y_true - y_pred) / y_true) * 100)

# y = load_series(csv_path='data/groupby_train.csv', time_col='date', value_col='sales')
y = load_series(csv_path='data/transactions.csv', time_col='date', value_col='transactions')

# Create a Pandas Series from the holidays
brazil_holidays = country_holidays('BR', years=list(range(y.index.year.min(), y.index.year.max()+5)))
holidays_series = pd.Series(brazil_holidays)
holidays_df = pd.DataFrame({'ds': holidays_series.index, 'holiday': holidays_series.values})
holidays_df['lower_window'] = -5
holidays_df['upper_window'] = 5
holidays_df['ds'] = pd.to_datetime(holidays_df['ds'])
# print(f'\n\n{holidays_df}')






# y_train=y
# transformers = [
#     ("fillna",Imputer()),
#     # ("ln",LogTransformer()),
#     # ("deseas7", Deseasonalizer(sp=7)),
#     # ("deseas365", Deseasonalizer(sp=365)),
#     # ("scaler", TabularToSeriesAdaptor(RobustScaler())),
#     (TabularToSeriesAdaptor(HolidayFeatures(
#                 calendar=country_holidays(country="BR"),
#                 holiday_windows={
#                     "Christmas": (5, 3), 
#                     "New Year": (2, 5), 
#                     "Carnival": (3, 3), 
#                     "Good Friday": (2, 2),
#                     "Tiradentes' Day": (2, 2),
#                     "Worker's Day": (2, 2),
#                     "Independence Day": (2, 2),
#                     "Our Lady of Aparecida": (2, 2),
#                     "All Souls' Day": (2, 2),
#                     "Republic Proclamation Day": (2, 2),
#                     "National Day of Zumbi and Black Awareness": (2, 2),
#                 }
#             )))
# ]
# models = [
#     ("forecast", StatsForecastMSTL(season_length=[7, 364])),
#     ("forecast", StatsForecastAutoETS(season_length=7)),
#     ("forecast", StatsForecastAutoCES(season_length=7)),
#     ("forecast", StatsForecastAutoTheta(
#         season_length=7,
#         decomposition_type='multiplicative', # 'additive' or 'multiplicative'
#     )),
#     ("forecast", StatsForecastAutoARIMA(
#         sp=7,
#         seasonal=True, 
#         trend=True, 
#         with_intercept=True,
#         method='lbfgs', 
#         # trace=True,
#         stepwise=True,
#         # parallel=True,
#     )),
#     ("forecast", StatsForecastAutoTBATS(seasonal_periods=364)), 
#     ("forecast", Prophet(
#         seasonality_mode='multiplicative',  # 'additive' or 'multiplicative'	
#         holidays=holidays_df
#     )),
     
# ]
# results = []
# for model in models:
#     forecaster = TransformedTargetForecaster(transformers+[model])
#     forecaster.fit(y_train)
#     y_pred = forecaster.predict(-np.arange(len(y_train)) )

#     # plot_series(y_train.tail(365*1), y_pred.tail(365*1), labels=["y_train", "y_pred"])
#     results.append([mape_metric(y_train, y_pred), mape_metric(y_train, y_pred, month_transform=False), model[1].__class__.__name__])

#     print(forecaster.get_class_tags())
#     # print(f'\n\n')
#     # print(f'MAPE (month): \033[32m{mape_metric(y_train, y_pred):.2f}%\033[0m, MAPE (day): \033[32m{mape_metric(y_train, y_pred, month_transform=False):.2f}%\033[0m, \033[34m{forecaster.get_fitted_params()['forecaster']}\033[0m')
#     # print(f'{forecaster.get_fitted_params()['steps']}\n')

# print(f'\n\n')
# [print(f'MAPE (month): \033[32m{result[0]:.2f}%\033[0m, MAPE (day): \033[32m{result[1]:.2f}%\033[0m, FORECASTER: \033[34m{result[2]}\033[0m ') for result in results]
# print(f'\n\n')










models = [
    # {
    #     'name': 'RNN',
    #     'forecaster': MLPRegressor(
    #         # hidden_layer_sizes=(50,),
    #         max_iter=50,
    #         early_stopping=True,
    #         batch_size=32,
    #         random_state=42
    #     ),
    # },
    {
        'name': 'TCN',
        'forecaster': NeuralForecastTCN(
            input_size=365,  # default -1 (all history)
            local_scaler_type='robust',
            scaler_type='robust',
            context_size=7, # default 10
            decoder_layers=3, # default 2
            max_steps=50, # default 1000
            batch_size=32, # default 32
            learning_rate=0.01, # default 0.001   between 0 and 1
            random_seed=42
        ),
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

    },
    {
        'name': 'TFT',
        'forecaster': PytorchForecastingTFT(
            trainer_params={
                "max_epochs": 50,  # for quick test
                "limit_train_batches": 20,  # for quick test
            }
        ),

    },
    {
        'name': 'NBeats',
        'forecaster': PytorchForecastingNBeats(
            trainer_params={
                "max_epochs": 50,  # for quick test
                "limit_train_batches": 20,  # for quick test
            }
        ),

    },
    {
        'name': 'NHiTS',
        'forecaster': PytorchForecastingNHiTS(
            trainer_params={
                "max_epochs": 50,  # for quick test
                "limit_train_batches": 20,  # for quick test
            }
        ),

    },
    {
        'name': 'DeepAR',
        'forecaster': PytorchForecastingDeepAR(
            trainer_params={
                "max_epochs": 50,  # for quick test
                "limit_train_batches": 20,  # for quick test
            }
        ),

    },
]

y_train, y_test = train_test_split(y, test_size=0.15, shuffle=False)
# PytorchForecastingTFT     6.731261538425411
# PytorchForecastingNBeats  6.047568925247252
# PytorchForecastingNHiTS   5.123437992171747
# PytorchForecastingDeepAR  4.95484522482381
# TimeLLMForecaster         8.901244655105247
# NeuralForecastLSTM        5.279586575100163
# DartsXGBModel(lags=7, output_chunk_length=7, random_state=42, num_samples=100)
for model in models:
    print(f'Model: {model['name']}')
    forecaster = model['forecaster']
    fh = ForecastingHorizon(range(1, int(len(y_test))), is_relative=True)
    forecaster.fit(y=y_train, fh=fh)
    y_pred = forecaster.predict(fh=fh)

    plot_series(y_test, y_pred, labels=["y_test", f"y_pred - {model['name']}, MAPE: {mape_metric(y_test, y_pred)}"])
    print(f'\n\n\nModel: {model['name']}, MAPE: {mape_metric(y_test, y_pred)}')
# forecaster = NeuralForecastLSTM(
#             # trainer_params={
#             #     "max_epochs": 100,  # for quick test
#             #     "limit_train_batches": 20,  # for quick test
#             # },

#             # pred_len=36,
#             # seq_len=96,
#             # llm_model='GPT2' #[‘GPT2’, ‘LLAMA’, ‘BERT’]

#             input_size=365,
#             local_scaler_type='robust',
#             scaler_type='robust',
#             # futr_exog_list=['dayofweek', 'month', 'year'],
#             max_steps=100,
#             batch_size=32,
#             # early_stop_patience_steps=100,
#             # val_check_steps=10,
#             random_seed=42
#         )


# fh = ForecastingHorizon(range(1, int(len(y_test))), is_relative=True)
# forecaster.fit(y=y_train, fh=fh)
# y_pred = forecaster.predict(fh=fh)
# plot_series(y_test, y_pred, labels=["y_test", "y_pred"])

# print(mape_metric(y_test, y_pred))


plt.show()