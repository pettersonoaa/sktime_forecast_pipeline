import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from darts import TimeSeries
from darts.models import LightGBMModel
from darts.metrics import mape
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torchmetrics import MeanAbsolutePercentageError
from sklearn.preprocessing import MinMaxScaler as SklearnMinMaxScaler

def generate_synthetic_data(start_date, end_date, seed=42):
    """Generate synthetic time series data with multiple seasonalities and covariates."""
    np.random.seed(seed)
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Target variable (val_transaction)
    trend = np.linspace(10, 100, len(date_range))
    yearly = 10 * np.sin(2 * np.pi * date_range.dayofyear / 365.25)
    monthly = 5 * np.sin(2 * np.pi * date_range.day / 30.44)
    weekly = 3 * np.sin(2 * np.pi * date_range.dayofweek / 7)
    noise = np.random.normal(0, 5, len(date_range))
    val_transaction = trend + yearly + monthly + weekly + noise
    
    # Covariate 1: val_limite (credit limit, correlated with transactions)
    val_limite = (
        trend * 2 +  # Higher trend
        yearly * 1.5 +  # Stronger yearly seasonality
        monthly +  # Same monthly pattern
        np.random.normal(0, 10, len(date_range))  # Different noise
    )
    
    # Covariate 2: qtd_ativos (number of active customers)
    base_ativos = 1000
    qtd_ativos = (
        base_ativos +
        np.linspace(0, 500, len(date_range)) +  # Upward trend
        100 * np.sin(2 * np.pi * date_range.dayofyear / 365.25) +  # Yearly pattern
        50 * np.sin(2 * np.pi * date_range.dayofweek / 7) +  # Weekly pattern
        np.random.normal(0, 20, len(date_range))  # Random variations
    ).astype(int)
    
    # Create DataFrames
    target_df = pd.DataFrame({
        'date': date_range,
        'val_transaction': val_transaction
    })
    
    covariates_df = pd.DataFrame({
        'date': date_range,
        'val_limite': val_limite,
        'qtd_ativos': qtd_ativos
    })
    
    # Convert to Darts TimeSeries
    target_series = TimeSeries.from_dataframe(target_df, 'date', 'val_transaction')
    covariate_series = TimeSeries.from_dataframe(covariates_df, 'date', ['val_limite', 'qtd_ativos'])
    
    return target_series, covariate_series

# Generate data with covariates
y, X = generate_synthetic_data('2022-01-01', '2025-04-09')

# Split data with proper history windows
forecast_horizon = 28
input_chunk_length = 28
output_chunk_length = 7
validation_length = input_chunk_length * 4

# Training data ends before validation
train_end = -validation_length-forecast_horizon
# Validation data includes necessary history
val_end = -forecast_horizon
# Test data needs history from validation period
test_history_start = -forecast_horizon-input_chunk_length

# Split target variable
train = y[:train_end]
val = y[train_end:val_end]
test = y[val_end:]

# Split covariates with overlapping windows for history
train_cov = X[:train_end]
val_cov = X[train_end:val_end]  
# Include more history for test covariates
test_cov = X[test_history_start:]

# Create scalers
scaler = SklearnMinMaxScaler()
cov_scaler = SklearnMinMaxScaler()

# Fit and transform training data
train_scaled = TimeSeries.from_times_and_values(
    train.time_index,
    scaler.fit_transform(train.values().reshape(-1, 1))
)

# Transform validation and test data
val_scaled = TimeSeries.from_times_and_values(
    val.time_index,
    scaler.transform(val.values().reshape(-1, 1))
)
test_scaled = TimeSeries.from_times_and_values(
    test.time_index,
    scaler.transform(test.values().reshape(-1, 1))
)

# Scale covariates with feature names
feature_names = ['val_limite', 'qtd_ativos']
train_cov_scaled = TimeSeries.from_dataframe(
    pd.DataFrame(
        cov_scaler.fit_transform(train_cov.values()),
        columns=feature_names,
        index=train_cov.time_index
    )
)

val_cov_scaled = TimeSeries.from_dataframe(
    pd.DataFrame(
        cov_scaler.transform(val_cov.values()),
        columns=feature_names,
        index=val_cov.time_index
    )
)

test_cov_scaled = TimeSeries.from_dataframe(
    pd.DataFrame(
        cov_scaler.transform(test_cov.values()),
        columns=feature_names,
        index=test_cov.time_index
    )
)

# Verify sizes and dates
print(f"Training: {train.start_time()} to {train.end_time()}, size: {len(train)}")
print(f"Validation: {val.start_time()} to {val.end_time()}, size: {len(val)}")
print(f"Test: {test.start_time()} to {test.end_time()}, size: {len(test)}")
print(f"Test covariates: {test_cov.start_time()} to {test_cov.end_time()}, size: {len(test_cov)}")














# LightGBM specific hyperparameters
lgb_params = {
    'num_leaves': 31,  # Maximum tree leaves for base learners      [15, 31, 63]
    'learning_rate': 0.05,  # Boosting learning rate                [0.01, 0.05, 0.1]
    'n_estimators': 100,  # Number of boosted trees to fit          [50, 100, 200]
    'max_depth': -1,  # Maximum tree depth, -1 means no limit
    'min_child_samples': 20,  # Minimum number of data needed in a leaf
    'subsample': 0.8,  # Subsample ratio of the training instance
    'colsample_bytree': 0.8,  # Subsample ratio of columns when constructing each tree
    'reg_alpha': 0.1,  # L1 regularization
    'reg_lambda': 0.1,  # L2 regularization
    'random_state': 42,
    'verbose': -1,  # Less verbose output
    'n_jobs': -1,  # Use all available cores
}

# Create logger
# logger = TensorBoardLogger("lightning_logs", name="lightgbm_model")

# Create LightGBM model
model = LightGBMModel(
    lags=input_chunk_length,  # Similar to input_chunk_length in other models
    lags_past_covariates=input_chunk_length,
    output_chunk_length=output_chunk_length,
    model_name='lightgbm_model_v1',
    force_reset=True,
    **lgb_params
)

# Create models directory
os.makedirs("models", exist_ok=True)

# Train the model
print("Training LightGBM model...")
model.fit(
    series=train_scaled,
    past_covariates=train_cov_scaled,
    val_series=val_scaled,
    val_past_covariates=val_cov_scaled
)

# Save the trained model
model.save("models/lightgbm_model.pt")

# Create a properly formatted test covariates TimeSeries
test_cov_for_pred = TimeSeries.from_dataframe(
    pd.DataFrame(
        test_cov_scaled.values(),
        columns=feature_names,
        index=test_cov_scaled.time_index
    )
)

# Predict with properly formatted covariates
pred_scaled = model.predict(
    n=forecast_horizon,
    series=val_scaled[-input_chunk_length:],
    past_covariates=test_cov_for_pred,
    show_warnings=False
)

# Convert prediction to numpy array before inverse transform
pred_values = pred_scaled.values().reshape(-1, 1)
pred_inverse = scaler.inverse_transform(pred_values)

# Create new TimeSeries with inverse transformed values
pred = TimeSeries.from_times_and_values(
    pred_scaled.time_index,
    pred_inverse
)

# Visualize results
plt.figure(figsize=(15, 7))
plt.plot(test.time_index, test.values(), 'k-', label='Actual', linewidth=2)
plt.plot(pred.time_index, pred.values(), 'g--', label='LightGBM Forecast', linewidth=2)

# Calculate prediction intervals (Note: LightGBM doesn't provide native uncertainty)
std_dev = pred.values().std()
plt.fill_between(
    pred.time_index, 
    pred.values().flatten() - std_dev,
    pred.values().flatten() + std_dev,
    alpha=0.2, 
    color='g',
    label='±1 std. dev.'
)

plt.title('LightGBM Model Forecast vs Actual (with uncertainty)', fontsize=14)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Value', fontsize=12)
plt.legend(fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.show()

# Print performance metrics
print(f"Test MAPE: {mape(test, pred):.2f}%")