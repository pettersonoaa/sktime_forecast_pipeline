import numpy as np
import pandas as pd
from pathlib import Path
import joblib
import os
from darts import TimeSeries
from darts.metrics import smape

def generate_synthetic_data(start_date, end_date, seed=42):
    """Generate synthetic time series data with multiple seasonalities."""
    np.random.seed(seed)
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    trend = np.linspace(10, 100, len(date_range))
    yearly = 10 * np.sin(2 * np.pi * date_range.dayofyear / 365.25)
    monthly = 5 * np.sin(2 * np.pi * date_range.day / 30.44)
    weekly = 3 * np.sin(2 * np.pi * date_range.dayofweek / 7)
    noise = np.random.normal(0, 5, len(date_range))
    
    val_transaction = trend + yearly + monthly + weekly + noise
    return TimeSeries.from_dataframe(
        pd.DataFrame({'date': date_range, 'val_transaction': val_transaction}),
        'date',
        'val_transaction'
    )

def save_model(model, name, save_dir='models'):
    """Save a trained model to disk."""
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    if type(model).__name__ in ['NBEATSModel', 'TCNModel', 'TFTModel']:
        model.save(os.path.join(save_dir, f"{name}_model.pth"))
    else:
        joblib.dump(model, os.path.join(save_dir, f"{name}_model.joblib"))

def load_model(name, model_class, save_dir='models'):
    """Load a trained model from disk."""
    if model_class.__name__ in ['NBEATSModel', 'TCNModel', 'TFTModel']:
        return model_class.load(os.path.join(save_dir, f"{name}_model.pth"))
    else:
        return joblib.load(os.path.join(save_dir, f"{name}_model.joblib"))

def evaluate_stability(model, series, window_size=30):
    """Evaluate model stability using rolling one-step-ahead forecasts."""
    smapes = []
    forecasts = []
    
    for i in range(len(series) - window_size, len(series)):
        train = series[:i]
        actual = series[i:i+1]
        
        try:
            model.fit(train)
            pred = model.predict(1)
            smapes.append(smape(actual, pred))
            forecasts.append(pred)
        except Exception as e:
            print(f"Error in stability evaluation: {str(e)}")
            continue
    
    return np.array(smapes), forecasts