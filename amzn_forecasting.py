"""
AMZN Stock Price Forecasting
=============================
This script implements and compares multiple forecasting models for Amazon stock prices.

Activities:
1. Baseline Models (Naive & Holt's Linear Trend)
2. ARIMA & SARIMA Models
3. Regression with Lagged Features
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.holtwinters import Holt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error


# ==============================================================================
# DATA LOADING AND PREPARATION
# ==============================================================================

# Load the dataset
df = pd.read_csv("AMZN.csv")

# Clean price data: remove dollar signs and convert to float
df["Close/Last"] = df["Close/Last"].str.replace("$", "").astype(float)

# Convert date column to datetime format
df["Date"] = pd.to_datetime(df["Date"])

# Sort by date (oldest to newest)
df = df.sort_values("Date")

# Aggregate to monthly data using mean of closing prices
monthly = df.resample("M", on="Date")["Close/Last"].mean()

# Split into training and test sets
# We'll forecast 6 months ahead
forecast_horizon = 6
train = monthly[:-forecast_horizon]
test = monthly[-forecast_horizon:]

print(f"Training data: {len(train)} months")
print(f"Test data: {len(test)} months")
print(f"Forecasting {forecast_horizon} months ahead\n")


# ==============================================================================
# ACTIVITY 1: BASELINE MODELS
# ==============================================================================

print("=" * 60)
print("ACTIVITY 1: BASELINE MODELS")
print("=" * 60)

# --- Naive Forecast ---
# Simply uses the last observed value as the forecast
naive_forecast = np.repeat(train.iloc[-1], forecast_horizon)
print("✓ Naive forecast completed")

# --- Holt's Linear Trend ---
# Captures both level and trend in the data
holt_model = Holt(train).fit()
holt_forecast = holt_model.forecast(forecast_horizon)
print("✓ Holt's Linear Trend forecast completed\n")


# ==============================================================================
# ACTIVITY 2: ARIMA & SARIMA MODELS
# ==============================================================================

print("=" * 60)
print("ACTIVITY 2: ARIMA & SARIMA MODELS")
print("=" * 60)

# --- Stationarity Check (ADF Test) ---
# The Augmented Dickey-Fuller test checks if the series is stationary
adf_result = adfuller(monthly)
print(f"ADF Test Statistic: {adf_result[0]:.4f}")
print(f"p-value: {adf_result[1]:.4f}")

if adf_result[1] < 0.05:
    print("→ Series is stationary (p < 0.05)\n")
else:
    print("→ Series is non-stationary (p >= 0.05)\n")

# --- ARIMA Model ---
# ARIMA(1,1,1): p=1 (AR term), d=1 (differencing), q=1 (MA term)
arima_model = ARIMA(train, order=(1, 1, 1)).fit()
arima_forecast = arima_model.forecast(forecast_horizon)
print(f"✓ ARIMA(1,1,1) completed | AIC: {arima_model.aic:.2f}")

# --- SARIMA Model ---
# SARIMA adds seasonal component with 12-month periodicity
sarima_model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 0, 1, 12)).fit()
sarima_forecast = sarima_model.forecast(forecast_horizon)
print(f"✓ SARIMA(1,1,1)(1,0,1,12) completed | AIC: {sarima_model.aic:.2f}\n")


# ==============================================================================
# ACTIVITY 3: REGRESSION WITH LAGS
# ==============================================================================

print("=" * 60)
print("ACTIVITY 3: REGRESSION WITH LAGGED FEATURES")
print("=" * 60)

# Transform the series to log scale for better linearity
log_series = np.log(monthly)

# Create a dataframe with lagged features
# We'll use the previous 2 months (lag1 and lag2) as predictors
regression_df = pd.DataFrame({
    "y": log_series,
    "lag1": log_series.shift(1),  # Previous month
    "lag2": log_series.shift(2)   # Two months ago
}).dropna()

# Split regression data
train_reg = regression_df.iloc[:-forecast_horizon]
test_reg = regression_df.iloc[-forecast_horizon:]

# Prepare features and target variable
X_train = sm.add_constant(train_reg[["lag1", "lag2"]])
y_train = train_reg["y"]

# Fit the regression model
regression_model = sm.OLS(y_train, X_train).fit()

# Make predictions
X_test = sm.add_constant(test_reg[["lag1", "lag2"]])
reg_forecast_log = regression_model.predict(X_test)

# Transform back to original scale
reg_forecast = np.exp(reg_forecast_log)

print(f"✓ Regression model completed | R²: {regression_model.rsquared:.4f}\n")


# ==============================================================================
# MODEL EVALUATION
# ==============================================================================

def evaluate_model(actual, predicted):
    """Calculate MAPE and RMSE for a given forecast"""
    mape = mean_absolute_percentage_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    return {"MAPE": mape, "RMSE": rmse}


# Evaluate all models
results = {
    "Naive": evaluate_model(test, naive_forecast),
    "Holt": evaluate_model(test, holt_forecast),
    "ARIMA": evaluate_model(test, arima_forecast),
    "SARIMA": evaluate_model(test, sarima_forecast),
    "Regression": evaluate_model(test, reg_forecast)
}

# Create comparison table
results_df = pd.DataFrame(results).T
results_df = results_df.sort_values("MAPE")  # Sort by best MAPE

print("=" * 60)
print("MODEL COMPARISON RESULTS")
print("=" * 60)
print(results_df.to_string())
print("\n" + "=" * 60)

# Identify best model
best_model = results_df.index[0]
best_mape = results_df.loc[best_model, "MAPE"]
print(f"🏆 Best Model: {best_model}")
print(f"   MAPE: {best_mape:.4f} ({best_mape*100:.2f}%)")
print("=" * 60 + "\n")


# ==============================================================================
# VISUALIZATION
# ==============================================================================

# Create a comprehensive comparison plot
plt.figure(figsize=(14, 7))

# Plot training data (historical)
plt.plot(train.index, train, label="Training Data", color="lightgray", linewidth=2)

# Plot actual test values
plt.plot(test.index, test, label="Actual", color="black", linewidth=2.5, marker="o", markersize=8)

# Plot all forecasts
plt.plot(test.index, sarima_forecast, label="SARIMA", marker="s", linewidth=2, alpha=0.8)
plt.plot(test.index, reg_forecast, label="Regression", marker="^", linewidth=2, alpha=0.8)
plt.plot(test.index, arima_forecast, label="ARIMA", marker="d", linewidth=1.5, alpha=0.7)
plt.plot(test.index, holt_forecast, label="Holt", marker="v", linewidth=1.5, alpha=0.7)
plt.plot(test.index, naive_forecast, label="Naive", marker="x", linewidth=1.5, alpha=0.7)

# Formatting
plt.title("AMZN Stock Price Forecasting - Model Comparison", fontsize=16, fontweight="bold", pad=20)
plt.xlabel("Date", fontsize=12)
plt.ylabel("Stock Price ($)", fontsize=12)
plt.legend(loc="best", fontsize=10)
plt.grid(True, alpha=0.3, linestyle="--")
plt.tight_layout()

# Save the plot
plt.savefig("forecast_comparison.png", dpi=300, bbox_inches="tight")
print("📊 Visualization saved as 'forecast_comparison.png'")

plt.show()
