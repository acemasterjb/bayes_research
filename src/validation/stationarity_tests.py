import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, kpss
import matplotlib.pyplot as plt

def run_stationarity_analysis(prices: np.ndarray, use_returns: bool = False):
    """
    Performs and prints a comprehensive stationarity analysis on a price series.
    Can analyze either log-prices or log-returns.
    """
    if use_returns:
        series_to_test = np.diff(np.log(prices))
        series_name = "Log-Returns"
    else:
        series_to_test = np.log(prices)
        series_name = "Log-Prices"

    print(f"\\n--- Stationarity Analysis on {series_name} ---")

    # --- 1. ACF Plot ---
    print(f"\\n1. Autocorrelation Function (ACF) Plot for {series_name}:")
    fig, ax = plt.subplots()
    sm.graphics.tsa.plot_acf(series_to_test, lags=30, ax=ax)
    ax.set_title(f"Autocorrelation of {series_name}")
    plt.show()
    print("  - An ACF plot that decays to zero very slowly suggests non-stationarity.")
    print("  - A quick decay to zero suggests stationarity.")

    # --- 2. Augmented Dickey-Fuller (ADF) Test ---
    print(f"\\n2. Augmented Dickey-Fuller (ADF) Test on {series_name}:")
    print("   Null Hypothesis (H0): The series has a unit root (it is non-stationary).")
    print("   - A low p-value (< 0.05) means we REJECT H0, indicating the series is likely stationary.")
    
    adf_result = adfuller(series_to_test)
    print(f"   - Test Statistic: {adf_result[0]:.4f}")
    print(f"   - p-value: {adf_result[1]:.4f}")
    print("   - Critical Values:")
    for key, value in adf_result[4].items():
        print(f"     - {key}: {value:.4f}")

    # --- 3. Half-Life of Mean Reversion (only for log-prices) ---
    if not use_returns:
        lag = np.roll(series_to_test, 1)
        delta = series_to_test - lag
        delta = delta[1:]
        lag = lag[1:]
        
        ols_result = sm.OLS(delta, sm.add_constant(lag)).fit()
        lambda_coeff = ols_result.params[1]
        
        if lambda_coeff < 0:
            half_life = -np.log(2) / lambda_coeff
            print(f"\\n3. Half-Life of Mean Reversion: {half_life:.2f} days")
        else:
            print("\\n3. Half-Life of Mean Reversion: Not applicable (process is not mean-reverting).")

    # --- 4. KPSS Test ---
    print(f"\\n4. Kwiatkowski-Phillips-Schmidt-Shin (KPSS) Test on {series_name}:")
    print("   Null Hypothesis (H0): The series is stationary around a constant.")
    print("   - A high p-value (> 0.05) means we FAIL to REJECT H0, indicating the series is likely stationary.")
    
    # Note: KPSS can produce warnings, which is normal.
    kpss_result = kpss(series_to_test, regression='c', nlags="auto")
    print(f"   - Test Statistic: {kpss_result[0]:.4f}")
    print(f"   - p-value: {kpss_result[1]:.4f}")
    print("   - Critical Values:")
    for key, value in kpss_result[3].items():
        print(f"     - {key}: {value:.4f}")
        
    # --- 5. Final Verdict ---
    print(f"\\n--- Final Verdict for {series_name} ---")
    adf_is_stationary = adf_result[1] < 0.05
    kpss_is_stationary = kpss_result[1] > 0.05
    
    if adf_is_stationary and kpss_is_stationary:
        print("  - Strong evidence FOR stationarity.")
    elif not adf_is_stationary and not kpss_is_stationary:
        print("  - Strong evidence AGAINST stationarity.")
    else:
        print("  - Ambiguous results. The tests conflict.")

    if not use_returns:
        if adf_is_stationary and kpss_is_stationary:
            print("  - The Ornstein-Uhlenbeck model is likely appropriate for the price series.")
        else:
            print("  - The Ornstein-Uhlenbeck model is likely INAPPROPRIATE for the price series.")
    else:
        if adf_is_stationary:
             print("  - The returns are stationary, which is typical for financial assets.")
        else:
             print("  - The returns appear non-stationary, which is unusual and may warrant further investigation.")


