import os
import numpy as np
import pandas as pd
from darts import TimeSeries
from darts.models import TFTModel
from darts.metrics import mape
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torchmetrics import MeanAbsolutePercentageError
from darts.dataprocessing.transformers import Scaler
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

# Scale covariates
train_cov_scaled = TimeSeries.from_times_and_values(
    train_cov.time_index,
    cov_scaler.fit_transform(train_cov.values())
)
val_cov_scaled = TimeSeries.from_times_and_values(
    val_cov.time_index,
    cov_scaler.transform(val_cov.values())
)
test_cov_scaled = TimeSeries.from_times_and_values(
    test_cov.time_index,
    cov_scaler.transform(test_cov.values())
)

# Verify sizes and dates
print(f"Training: {train.start_time()} to {train.end_time()}, size: {len(train)}")
print(f"Validation: {val.start_time()} to {val.end_time()}, size: {len(val)}")
print(f"Test: {test.start_time()} to {test.end_time()}, size: {len(test)}")
print(f"Test covariates: {test_cov.start_time()} to {test_cov.end_time()}, size: {len(test_cov)}")



# Create logger and early stopping
logger = TensorBoardLogger("lightning_logs", name="tft_model")
my_stopper = EarlyStopping(
    monitor="train_MeanAbsolutePercentageError",
    patience=20,
    min_delta=0.001,
    mode='min',
)

# Create and train TFT model with relative index encoding
model = TFTModel(
    input_chunk_length=input_chunk_length,
    output_chunk_length=7,
    hidden_size=256,
    lstm_layers=3,
    num_attention_heads=8,
    dropout=0.1,
    batch_size=32,
    n_epochs=500,
    # learning_rate=0.001,
    add_relative_index=True,
    model_name='tft_model_v1',
    force_reset=True,
    torch_metrics=MeanAbsolutePercentageError(),
    pl_trainer_kwargs={
        "accelerator": "cpu",
        "enable_progress_bar": True,
        "log_every_n_steps": 1,
        "callbacks": [my_stopper],
        "logger": logger 
    },
    
)

# Create models directory
os.makedirs("models", exist_ok=True)

# load the weights from a manual save
model.load_weights("models/tft_model.pt")

# Train with scaled data
model.fit(
    series=train_scaled,
    past_covariates=train_cov_scaled,
    val_series=val_scaled,
    val_past_covariates=val_cov_scaled,
    verbose=True
)

# Save the trained model
model.save("models/tft_model.pt")

# Load model and make predictions with covariates
# loaded_model = TFTModel.load("models/tft_model.pt")

# Predict and inverse transform
pred_scaled = model.predict(
    n=forecast_horizon,
    series=val_scaled[-input_chunk_length:],
    past_covariates=test_cov_scaled,
    verbose=True
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
import matplotlib.pyplot as plt

plt.figure(figsize=(15, 7))
plt.plot(test.time_index, test.values(), 'k-', label='Actual', linewidth=2)
plt.plot(pred.time_index, pred.values(), 'b--', label='Forecast', linewidth=2)

# Calculate prediction intervals
std_dev = pred.values().std()
plt.fill_between(
    pred.time_index, 
    pred.values().flatten() - std_dev,
    pred.values().flatten() + std_dev,
    alpha=0.2, 
    color='b',
    label='±1 std. dev.'
)

plt.title('TFT Model Forecast vs Actual (with uncertainty)', fontsize=14)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Value', fontsize=12)
plt.legend(fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.show()