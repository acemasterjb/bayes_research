import plotly.graph_objects as go
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm

from src.motors.ewma_model import generate_ewma_price_paths


def plot_ewma_posteriors(trace):
    """
    Plots the posterior distributions for the EWMA model parameters.
    """
    az.plot_posterior(trace, var_names=["mu", "lambda", "nu"])


def plot_ewma_volatility(trace, returns):
    """
    Plots the inferred volatility from the EWMA model against the returns.
    """
    posterior = trace.posterior.stack(sample=("chain", "draw"))
    mu_mean = posterior["mu"].mean().item()
    lambda_mean = posterior["lambda"].mean().item()
    alpha_mean = 1 - lambda_mean
    initial_vol_sq_mean = posterior["initial_vol_sq"].mean().item()

    vol_sq_series = np.zeros(len(returns))
    vol_sq_series[0] = initial_vol_sq_mean
    for t in range(1, len(returns)):
        err_tm1 = returns[t - 1] - mu_mean
        vol_sq_series[t] = alpha_mean * err_tm1**2 + lambda_mean * vol_sq_series[t - 1]

    volatility = np.sqrt(vol_sq_series)

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=returns, mode="lines", name="Log Returns"))
    fig.add_trace(go.Scatter(y=volatility, mode="lines", name="Inferred Volatility"))
    fig.update_layout(
        title="EWMA Inferred Volatility",
        xaxis_title="Time",
        yaxis_title="Volatility/Log Returns",
    )
    fig.show()


def plot_ewma_ppc(model, trace):
    """
    Plots the posterior predictive check for the EWMA model.
    """
    # Slice the trace for efficiency
    minimized_trace = trace.sel(draw=slice(None, None, 50))
    with model:
        ppc = pm.sample_posterior_predictive(minimized_trace, var_names=["obs"])

    az.plot_ppc(ppc, var_names=["obs"], kind="kde")
    plt.suptitle("Posterior Predictive Check for GARCH Model", y=1.02)
    plt.show()


def plot_ewma_in_sample_fit(
    trace,
    prices,
    percentile=0.5,
    trim_percentile_lower=0.1,
    trim_percentile_upper=0.9,
):
    """
    Plots the in-sample fit of the EWMA model.
    """
    log_returns = np.diff(np.log(prices))
    posterior = trace.posterior.stack(sample=("chain", "draw"))

    mu_mean = np.quantile(posterior["mu"], percentile)
    lambda_mean = np.quantile(posterior["lambda"], percentile)
    nu_mean = np.quantile(posterior["nu"], percentile)
    initial_vol_sq_mean = np.quantile(posterior["initial_vol_sq"], percentile)

    alpha_mean = 1 - lambda_mean

    vol_sq_series = np.zeros(len(log_returns))
    vol_sq_series[0] = initial_vol_sq_mean
    for t in range(1, len(log_returns)):
        err_tm1 = log_returns[t - 1] - mu_mean
        vol_sq_series[t] = alpha_mean * err_tm1**2 + lambda_mean * vol_sq_series[t - 1]
    last_vol_sq = vol_sq_series[-1]

    simulated_paths = generate_ewma_price_paths(
        s0=prices[-1],
        T=len(prices) - 1,
        N=100,
        mu=mu_mean,
        alpha=alpha_mean,
        beta=lambda_mean,
        nu=nu_mean,
        last_return=log_returns[-1],
        last_vol_sq=last_vol_sq,
    )

    trimmed_mean_path = np.quantile(
        simulated_paths, q=[trim_percentile_lower, trim_percentile_upper], axis=1
    )

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(y=prices, mode="lines", name="Actual Prices", line=dict(color="red"))
    )
    fig.add_trace(
        go.Scatter(
            y=trimmed_mean_path[0],
            fill=None,
            mode="lines",
            line=dict(color="blue", dash="dash"),
            name=f"{trim_percentile_lower*100:.0f}th Percentile",
        )
    )
    fig.add_trace(
        go.Scatter(
            y=trimmed_mean_path[1],
            fill="tonexty",
            mode="lines",
            line=dict(color="blue", dash="dash"),
            name=f"{trim_percentile_upper*100:.0f}th Percentile",
        )
    )

    for i in range(10):
        fig.add_trace(
            go.Scatter(
                y=simulated_paths[:, i],
                mode="lines",
                line=dict(width=0.5),
                showlegend=False,
            )
        )

    fig.update_layout(
        title="EWMA In-Sample Fit", xaxis_title="Time", yaxis_title="Price"
    )
    fig.show()


def plot_ewma_standardized_residuals_and_acf(trace, returns):
    """
    Plots the standardized residuals and their ACF for the EWMA model.
    """
    posterior = trace.posterior.stack(sample=("chain", "draw"))
    mu_mean = posterior["mu"].mean().item()
    lambda_mean = posterior["lambda"].mean().item()
    alpha_mean = 1 - lambda_mean
    initial_vol_sq_mean = posterior["initial_vol_sq"].mean().item()

    vol_sq_series = np.zeros(len(returns))
    vol_sq_series[0] = initial_vol_sq_mean
    for t in range(1, len(returns)):
        err_tm1 = returns[t - 1] - mu_mean
        vol_sq_series[t] = alpha_mean * err_tm1**2 + lambda_mean * vol_sq_series[t - 1]

    std_resid = (returns - mu_mean) / np.sqrt(vol_sq_series)

    fig_resid = go.Figure()
    fig_resid.add_trace(
        go.Scatter(y=std_resid, mode="lines", name="Standardized Residuals")
    )
    fig_resid.update_layout(
        title="EWMA Standardized Residuals", xaxis_title="Time", yaxis_title="Residual"
    )
    fig_resid.show()

    az.plot_acf(std_resid, lags=30)
