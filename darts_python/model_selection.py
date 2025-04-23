from itertools import product
import numpy as np
from darts.metrics import smape

def rolling_cv(model_class, param_grid, train_series, val_len=28, n_splits=2):
    """Perform time series cross-validation with rolling window."""
    best_score = float('inf')
    best_params = None

    # # Default parameters for specific models
    # default_params = {
    #     'ExponentialSmoothing': {
    #         'seasonal': 'add',
    #         'trend': 'add',
    #         'seasonal_periods': 7
    #     },
    #     'Theta': {
    #         'season_mode': 'multiplicative'
    #     }
    # }

    # # If param_grid is empty, use default parameters
    # if not param_grid and model_class.__name__ in default_params:
    #     return default_params[model_class.__name__], None
    
    for params in (dict(zip(param_grid, v)) for v in product(*param_grid.values())):
        scores = []
        for split in range(n_splits):
            split_point = -val_len * (split + 1)
            tr = train_series[:split_point]
            val = train_series[split_point:split_point + val_len]
            
            min_train_len = max(
                params.get("lags", 1) if isinstance(params.get("lags", 1), int) else max(params.get("lags", [1])),
                params.get("input_chunk_length", 1)
            ) + val_len
            
            if len(tr) < min_train_len or len(val) == 0:
                continue
                
            try:
                model = model_class(**params)
                model.fit(tr)
                pred = model.predict(len(val))
                scores.append(smape(val, pred))
            except Exception:
                continue
                
        if scores:
            avg_score = np.mean(scores)
            if avg_score < best_score:
                best_score = avg_score
                best_params = params
    
    # # If grid search failed, use default parameters if available
    # if best_params is None and model_class.__name__ in default_params:
    #     best_params = default_params[model_class.__name__]
    #     print(f"Warning: Using default parameters for {model_class.__name__}")
    
    return best_params, best_score