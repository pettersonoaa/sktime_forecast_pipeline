from darts.models import (
    LightGBMModel, ARIMA, Prophet, NBEATSModel, CatBoostModel,
    NaiveSeasonal, LinearRegressionModel, ExponentialSmoothing, 
    Theta, TCNModel, TFTModel
)

# Model configurations with grid search parameters
MODELS_CONFIG = {
    'NaiveSeasonal': {
        'model_class': NaiveSeasonal,
        'grid': {"K": [7, 14]},
        'kwargs': {},
        'inference_cost_per_hour': 0.05  # SageMaker ml.t3.medium
    },
    'LinearRegression': {
        'model_class': LinearRegressionModel,
        'grid': {"lags": [7, 14, 28]},
        'kwargs': {},
        'inference_cost_per_hour': 0.05  # SageMaker ml.t3.medium
    },
    # 'ExponentialSmoothing': {
    #     'model_class': ExponentialSmoothing,
    #     'grid': {
    #         "trend": [None, "add", "mul"],
    #         "seasonal": [None, "add", "mul"],
    #         "seasonal_periods": [7, 30, 365]
    #     },
    #     'kwargs': {},
    #     'inference_cost_per_hour': 0.05  # SageMaker ml.t3.medium
    # },
    # 'ARIMA': {
    #     'model_class': ARIMA,
    #     'grid': {"p": [1, 2], "d": [0, 1], "q": [0, 1]},
    #     'kwargs': {},
    #     'inference_cost_per_hour': 0.05  # SageMaker ml.t3.medium
    # },
    'Theta': {
        'model_class': Theta,
        'grid': {},
        'kwargs': {},
        'inference_cost_per_hour': 0.05  # SageMaker ml.t3.medium
    },
    # 'Prophet': { # 12sec
    #     'model_class': Prophet,
    #     'grid': {"seasonality_mode": ["additive", "multiplicative"], 
    #             "yearly_seasonality": [True, False]},
    #     'kwargs': {},
    #     'inference_cost_per_hour': 0.05  # SageMaker ml.t3.medium
    # },
    # 'LightGBM': { # 12sec
    #     'model_class': LightGBMModel,
    #     'grid': {"lags": [7, 28], "output_chunk_length": [7, 28]},
    #     'kwargs': {'random_state': 42},
    #     'inference_cost_per_hour': 0.12  # SageMaker ml.m5.large
    # },
    # 'CatBoost': {
    #     'model_class': CatBoostModel,
    #     'grid': {"lags": [7, 28], "output_chunk_length": [7, 28]},
    #     'kwargs': {'random_state': 42},
    #     'inference_cost_per_hour': 0.12  # SageMaker ml.m5.large
    # },
    # 'N-BEATS': {
    #     'model_class': NBEATSModel,
    #     # 'grid': {"input_chunk_length": [28*13, 28*13*2], "output_chunk_length": [7, 28]},
    #     # 'kwargs': {'random_state': 42, 'n_epochs': 10, 'batch_size': 64}
    #     'grid': {},
    #     'kwargs': {
    #         'random_state': 42, 
    #         'n_epochs': 10, 
    #         'batch_size': 64, 
    #         'input_chunk_length': 28*13, 
    #         'output_chunk_length': 7
    #     },
    #     'inference_cost_per_hour': 0.45  # SageMaker ml.m5.xlarge
    # },
    'TCN': {
        'model_class': TCNModel,
        # 'grid': {
        #     "input_chunk_length": [28*13, 28*13*2],
        #     "output_chunk_length": [7, 28],
        #     "num_filters": [16, 32],
        #     "kernel_size": [3, 5]
        # },
        # 'kwargs': {'random_state': 42, 'n_epochs': 10}
        'grid': {},
        'kwargs': {
            'random_state': 42, 
            'n_epochs': 10,
            'input_chunk_length': 28*13, 
            'output_chunk_length': 7,
            'num_filters': 32,
            'kernel_size': 3
        },
        'inference_cost_per_hour': 0.45  # SageMaker ml.m5.xlarge
    },
    # 'TFT': {
    #     'model_class': TFTModel,
    #     # 'grid': {
    #     #     "input_chunk_length": [28*13, 28*13*2],
    #     #     "output_chunk_length": [7, 28],
    #     #     "hidden_size": [32, 64]
    #     # },
    #     # 'kwargs': {
    #     #     'random_state': 42, 
    #     #     'n_epochs': 10
    #     # }
    #     'grid': {},
    #     'kwargs': {
    #         'random_state': 42, 
    #         'n_epochs': 10,
    #         'input_chunk_length': 28*13, 
    #         'output_chunk_length': 7,
    #         'hidden_size': 32,
    #         'add_relative_index': True
    #     },
    #     'inference_cost_per_hour': 0.45  # SageMaker ml.m5.xlarge
    # }
}