import arviz as az
import numpy as np
import pandas as pd
from src.motors.bvar_model import train_bvar_model, forecast_bvar_model


def run_fixed_holdout(y_data, n_lags: int = 14, holdout_days: int = 90):
    """
    Performs a fixed holdout validation on the time series data.
    """
    # 1. Split data
    if len(y_data) <= holdout_days + n_lags:
        raise ValueError("Not enough data for the specified holdout and lag periods.")

    train_data = y_data[:-holdout_days]
    test_data = y_data[-holdout_days:]

    # 2. Fit model on training data
    print("\\n--- Fitting model on training data for holdout validation ---")
    # We create a dummy DataFrame for the train function
    train_df = pd.DataFrame(
        np.exp(train_data), columns=["borrow_volume", "supply_volume"]
    )
    trace, train_y_data, _ = train_bvar_model(train_df, n_lags=n_lags)

    # 3. Forecast for the holdout period
    forecast_samples = forecast_bvar_model(
        trace, train_y_data, n_lags=n_lags, n_forecast_steps=holdout_days
    )

    # 4. Calculate metrics
    # Get median and HDI from forecast samples
    median_forecast = np.median(forecast_samples, axis=0)
    hdi = az.hdi(az.convert_to_inference_data(forecast_samples), hdi_prob=0.9).x.values

    # a. Mean Absolute Error (MAE)
    mae_borrow = np.mean(np.abs(median_forecast[:, 0] - test_data[:, 0]))
    mae_supply = np.mean(np.abs(median_forecast[:, 1] - test_data[:, 1]))

    # b. HDI Coverage Score
    coverage_borrow = np.mean(
        (test_data[:, 0] >= hdi[0, 0]) & (test_data[:, 0] <= hdi[0, 1])
    )
    coverage_supply = np.mean(
        (test_data[:, 1] >= hdi[1, 0]) & (test_data[:, 1] <= hdi[1, 1])
    )

    print("\\n--- Holdout Validation Metrics ---")
    print(f"Log-Borrow MAE: {mae_borrow:.4f}")
    print(f"Log-Supply MAE: {mae_supply:.4f}")
    print(f"Log-Borrow 90% HDI Coverage: {coverage_borrow:.2%}")
    print(f"Log-Supply 90% HDI Coverage: {coverage_supply:.2%}")

    return test_data, median_forecast, forecast_samples

