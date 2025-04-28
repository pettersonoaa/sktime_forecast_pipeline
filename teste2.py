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


# h = TabularToSeriesAdaptor(HolidayFeatures(
#                 calendar=brazil_holidays,
#                 holiday_windows={
                #     "Christmas Day": (5, 3), 
                #     "Universal Fraternization Day": (2, 5), 
                #     "Carnival": (3, 3), 
                #     "Good Friday": (2, 2),
                #     "Tiradentes' Day": (2, 2),
                #     "Worker's Day": (2, 2),
                #     "Independence Day": (2, 2),
                #     "Our Lady of Aparecida": (2, 2),
                #     "All Souls' Day": (2, 2),
                #     "Republic Proclamation Day": (2, 2),
                # }
#             ),
#             fit_in_transform=True,
#             input_type="pandas")
# h = HolidayFeatures(
#                 calendar=brazil_holidays,
#                 holiday_windows={
#                     "Christmas Day": (5, 3), 
#                     "Universal Fraternization Day": (2, 5), 
#                     "Carnival": (3, 3), 
#                     "Good Friday": (2, 2),
#                     "Tiradentes' Day": (2, 2),
#                     "Worker's Day": (2, 2),
#                     "Independence Day": (2, 2),
#                     "Our Lady of Aparecida": (2, 2),
#                     "All Souls' Day": (2, 2),
#                     "Republic Proclamation Day": (2, 2),
#                 }
#             )
# print(type(y.index))
# print(y.index)
# h.fit_transform(y)  
# print(h)




y_train=y
transformers = [
    ("fillna",Imputer()),
    # ("ln",LogTransformer()),
    # ("deseas7", Deseasonalizer(sp=7)),
    # ("deseas365", Deseasonalizer(sp=365)),
    # ("scaler", TabularToSeriesAdaptor(RobustScaler())),
]
models = [
    ("forecast", StatsForecastMSTL(season_length=[7, 364])),
    ("forecast", StatsForecastAutoETS(season_length=7)),
    ("forecast", StatsForecastAutoCES(season_length=7)),
    ("forecast", StatsForecastAutoTheta(
        season_length=7,
        decomposition_type='multiplicative', # 'additive' or 'multiplicative'
    )),
    ("forecast", StatsForecastAutoARIMA(
        sp=7,
        seasonal=True, 
        trend=True, 
        with_intercept=True,
        method='lbfgs', 
        # trace=True,
        stepwise=True,
        # parallel=True,
    )),
    ("forecast", StatsForecastAutoTBATS(seasonal_periods=364)), 
    # ("forecast", Prophet(
    #     seasonality_mode='multiplicative',  # 'additive' or 'multiplicative'	
    #     holidays=holidays_df
    # )),
     
]
results = []
best_mape_month = float('inf')
best_mape_day = float('inf')
for model in models:
    forecaster = TransformedTargetForecaster(transformers+[model])
    forecaster.fit(y_train)
    y_pred = forecaster.predict(-np.arange(len(y_train)) )
    mape_month = mape_metric(y_train, y_pred, month_transform=True)
    mape_day = mape_metric(y_train, y_pred, month_transform=False)
    model_class = model[1].__class__.__name__
    if mape_day < best_mape_day:
        best_mape_day = mape_day
    if mape_month < best_mape_month:
        best_mape_month = mape_month
    
    results.append([mape_month, mape_day, model_class])
    # plot_series(y_train.tail(365*1), y_pred.tail(365*1), labels=["y_train", "y_pred"])

    print(forecaster.get_class_tags())
    # print(f'\n\n')
    # print(f'MAPE (month): \033[32m{mape_metric(y_train, y_pred):.2f}%\033[0m, MAPE (day): \033[32m{mape_metric(y_train, y_pred, month_transform=False):.2f}%\033[0m, \033[34m{forecaster.get_fitted_params()['forecaster']}\033[0m')
    # print(f'{forecaster.get_fitted_params()['steps']}\n')

print(f'\n\n')
[print(f'MAPE (month): \033[32m{result[0]:.2f}%\033[0m, MAPE (day): \033[32m{result[1]:.2f}%\033[0m, FORECASTER: \033[34m{result[2]}\033[0m ') for result in results]
print(f'\n\n')







plt.show()