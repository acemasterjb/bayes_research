# This file will contain the plotting functions for the GARCH model.
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import pymc as pm
import statsmodels.api as sm


def plot_garch_simple_residuals(trace, returns):
    """
    Plots the simple residuals (observed - predicted mean) for the GARCH model.
    """
    mu_mean = trace.posterior["mu"].mean().item()
    residuals = returns - mu_mean

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("GARCH Model Simple Residuals Analysis", fontsize=16)

    # Simple Residuals vs. Time
    axes[0].plot(residuals)
    axes[0].set_title("Simple Residuals (Returns - mu) vs. Time")
    axes[0].set_xlabel("Time")
    axes[0].set_ylabel("Return")

    # Histogram of Simple Residuals
    axes[1].hist(residuals, bins=40, density=True)
    axes[1].set_title("Histogram of Simple Residuals")
    axes[1].set_xlabel("Return")
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def plot_garch_standardized_residuals_and_acf(trace, returns):
    """
    Performs ACF analysis on the GARCH model's standardized residuals.
    """
    # 1. Reconstruct the historical volatility series from posterior means
    posterior = trace.posterior.stack(sample=("chain", "draw"))
    mu_mean = posterior["mu"].mean().item()
    omega_mean = posterior["omega"].mean().item()
    alpha_mean = posterior["alpha"].mean().item()
    beta_mean = posterior["beta"].mean().item()
    initial_vol_sq_mean = posterior["initial_vol_sq"].mean().item()

    vol_sq_series = np.zeros(len(returns) + 1)
    vol_sq_series[0] = initial_vol_sq_mean
    for t in range(1, len(returns) + 1):
        err_tm1 = returns[t - 1] - mu_mean
        vol_sq_series[t] = (
            omega_mean + alpha_mean * err_tm1**2 + beta_mean * vol_sq_series[t - 1]
        )
    inferred_vol = np.sqrt(vol_sq_series[1:])

    # 2. Calculate Standardized Residuals
    residuals = returns - mu_mean
    standardized_residuals = residuals / inferred_vol

    # 3. Create 1x2 plot for ACF diagnostics
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("ACF of Standardized Residuals", fontsize=16)

    # ACF of Standardized Residuals
    sm.graphics.tsa.plot_acf(standardized_residuals, lags=40, ax=axes[0])
    axes[0].set_title("ACF of Standardized Residuals")

    # ACF of Squared Standardized Residuals
    sm.graphics.tsa.plot_acf(standardized_residuals**2, lags=40, ax=axes[1])
    axes[1].set_title("ACF of Squared Standardized Residuals")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    print("\\n--- Standardized Residual Analysis ---")
    print("  - The 'ACF of Standardized Residuals' should have no significant spikes after lag 0.")
    print("  - Crucially, the 'ACF of Squared Standardized Residuals' should also have no significant spikes, which indicates that the GARCH model has successfully captured the volatility clustering.")



def plot_garch_ppc(model, trace):
    """
    Performs and plots a posterior predictive check for the GARCH model.
    """
    # Slice the trace for efficiency
    minimized_trace = trace.sel(draw=slice(None, None, 50))
    with model:
        ppc = pm.sample_posterior_predictive(minimized_trace, var_names=["obs"])

    az.plot_ppc(ppc, var_names=["obs"], kind="kde")
    plt.suptitle("Posterior Predictive Check for GARCH Model", y=1.02)
    plt.show()


def plot_garch_forecast(prices: np.ndarray, forecasted_returns: np.ndarray):
    """
    Plots the historical price and the forecasted price paths using Plotly.
    """
    # 1. Convert forecasted returns to price paths
    last_price = prices[-1]
    n_samples = forecasted_returns.shape[0]
    n_forecast_steps = forecasted_returns.shape[1]

    # Create price paths starting from the last known price
    # np.exp(np.cumsum(forecasted_returns, axis=1)) gives the cumulative return factor
    price_paths = last_price * np.exp(np.cumsum(forecasted_returns, axis=1))

    # 2. Calculate median and HDI for each time step
    median_path = np.median(price_paths, axis=0)
    hdi = np.array(
        [az.hdi(price_paths[:, t], hdi_prob=0.9) for t in range(n_forecast_steps)]
    )
    hdi_lower = hdi[:, 0]
    hdi_upper = hdi[:, 1]

    # 3. Create time axes
    historical_time = np.arange(len(prices))
    forecast_time = np.arange(len(prices) - 1, len(prices) + n_forecast_steps)

    # 4. Create Plotly figure
    fig = go.Figure()

    # Historical prices
    fig.add_trace(
        go.Scatter(x=historical_time, y=prices, mode="lines", name="Historical Price")
    )

    # Forecasted HDI cone
    fig.add_trace(
        go.Scatter(
            x=np.concatenate([forecast_time, forecast_time[::-1]]),
            y=np.concatenate([hdi_upper, hdi_lower[::-1]]),
            fill="toself",
            fillcolor="rgba(0,100,80,0.2)",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            name="90% HDI Forecast",
        )
    )

    # Median forecast path
    fig.add_trace(
        go.Scatter(
            x=forecast_time,
            y=np.insert(
                median_path, 0, last_price
            ),  # Prepend last price for a continuous line
            mode="lines",
            name="Median Forecast",
            line=dict(dash="dash"),
        )
    )

    fig.update_layout(
        title="GARCH Model Price Forecast",
        xaxis_title="Time Steps",
        yaxis_title="Price",
        showlegend=True,
    )
    fig.show()


def plot_garch_in_sample_fit(
    trace,
    prices: np.ndarray,
    p_jump: float,
    mu_jump: float,
    sigma_jump: float,
    n_paths: int = 10,
    percentile: float = 0.5,
    trim_percentile_lower: float = 0.1,
    trim_percentile_upper: float = 0.9,
):
    """
    Plots the observed price path and superimposes theoretical paths
    simulated using GARCH parameters from a specific posterior percentile.
    """
    # ... (previous code for reconstructing volatility remains the same) ...
    log_returns = np.diff(np.log(prices))
    posterior = trace.posterior.stack(sample=("chain", "draw"))

    mu_p = np.quantile(posterior["mu"], q=percentile)
    omega_p = np.quantile(posterior["omega"], q=percentile)
    alpha_p = np.quantile(posterior["alpha"], q=percentile)
    beta_p = np.quantile(posterior["beta"], q=percentile)
    # nu_p = np.quantile(posterior["nu"], q=percentile)
    initial_vol_sq_p = np.quantile(posterior["initial_vol_sq"], q=percentile)

    vol_sq_series = np.zeros(len(log_returns) + 1)
    vol_sq_series[0] = initial_vol_sq_p
    for t in range(1, len(log_returns) + 1):
        err_tm1 = log_returns[t - 1] - mu_p
        vol_sq_series[t] = (
            omega_p + alpha_p * err_tm1**2 + beta_p * vol_sq_series[t - 1]
        )
    inferred_vol = np.sqrt(vol_sq_series)

    # 2. Generate simulated price paths
    fig = go.Figure()
    all_simulated_paths = np.zeros((n_paths, len(prices)))
    rng = np.random.default_rng(seed=42)

    for i in range(n_paths):
        # ... (simulation loop remains the same) ...
        simulated_returns = np.zeros(len(log_returns))
        for t in range(len(log_returns)):
            if sigma_jump > 0 and rng.random() < p_jump:
                shock = rng.normal(mu_jump, sigma_jump)
            else:
                shock = rng.standard_t(df=3)
            simulated_returns[t] = mu_p + shock * inferred_vol[t + 1]

        simulated_path = np.zeros(len(prices))
        simulated_path[0] = prices[0]
        simulated_path[1:] = prices[0] * np.exp(np.cumsum(simulated_returns))
        all_simulated_paths[i, :] = simulated_path

        if i < 10:
            fig.add_trace(
                go.Scatter(
                    x=np.arange(len(prices)),
                    y=simulated_path,
                    mode="lines",
                    name=f"Simulated Path {i+1}",
                    line=dict(dash="dash", width=1),
                    opacity=0.7,
                )
            )

    # 3. Calculate and Plot Trimmed Mean Path
    trimmed_mean_path = np.zeros(len(prices))
    for t in range(len(prices)):
        paths_at_t = all_simulated_paths[:, t]
        lower_bound = np.quantile(paths_at_t, q=trim_percentile_lower)
        upper_bound = np.quantile(paths_at_t, q=trim_percentile_upper)
        trimmed_paths_at_t = paths_at_t[
            (paths_at_t >= lower_bound) & (paths_at_t <= upper_bound)
        ]
        trimmed_mean_path[t] = np.mean(trimmed_paths_at_t)

    fig.add_trace(
        go.Scatter(
            x=np.arange(len(prices)),
            y=trimmed_mean_path,
            mode="lines",
            name=f"Trimmed Mean ({int(trim_percentile_lower*100)}-{int(trim_percentile_upper*100)}%) Path",
            line=dict(color="red", width=2),
        )
    )

    # 4. Plot Observed Prices
    fig.add_trace(
        go.Scatter(
            x=np.arange(len(prices)),
            y=prices,
            mode="lines",
            name="Observed Price Path",
            line=dict(color="black", width=2),
        )
    )

    fig.update_layout(
        title=f"GARCH In-Sample Fit (Simulated from {int(percentile*100)}th Percentile Parameters)",
        xaxis_title="Time Steps",
        yaxis_title="Price",
        showlegend=True,
    )
    fig.show()


def plot_garch_posteriors(trace):
    """
    Plots the posterior distributions for the main GARCH parameters.
    """

    az.plot_posterior(
        trace,
        var_names=["mu", "omega", "alpha", "beta"],
        hdi_prob=0.95,
    )
    plt.suptitle("Posterior Distributions of GARCH Parameters", y=1.02)
    plt.show()


def plot_garch_volatility(trace, returns):
    """
    Plots the inferred conditional volatility over time.
    """
    # Extract the posterior mean of the conditional volatility
    # Note: `obs` likelihood's sigma corresponds to our volatility.
    # We need to compute it from the trace.

    # We can't directly get the time series of volatility from the trace
    # as it's computed inside scan. A common way to visualize is to
    # re-calculate it using the mean of the posterior parameters.

    posterior = trace.posterior.stack(sample=("chain", "draw"))
    mu = posterior["mu"].mean().item()
    omega = posterior["omega"].mean().item()
    alpha = posterior["alpha"].mean().item()
    beta = posterior["beta"].mean().item()
    initial_vol_sq = posterior["initial_vol_sq"].mean().item()
    vol_forecast = np.zeros(len(returns))
    vol_forecast[0] = initial_vol_sq

    for t in range(1, len(returns)):
        error_tm1 = returns[t - 1] - mu
        vol_forecast[t] = omega + alpha * error_tm1**2 + beta * vol_forecast[t - 1]

    # Plotting
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot returns
    ax.plot(returns, label="Log-Returns", color="gray", alpha=0.7)

    # Plot inferred volatility
    ax.plot(np.sqrt(vol_forecast), label="Inferred Volatility (Mean)", color="C1")
    ax.set_title("Inferred Conditional Volatility vs. Log-Returns")
    ax.set_xlabel("Time")
    ax.set_ylabel("Volatility / Return")
    ax.legend()
    plt.show()
