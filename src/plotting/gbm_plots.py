import plotly.graph_objects as go
import arviz as az
import numpy as np
from src.motors.gbm_model import generate_gbm_price_paths


def plot_gbm_posteriors(trace):
    """
    Plots the posterior distributions for the GBM model parameters.
    """
    az.plot_posterior(trace, var_names=["mu", "sigma"])


def plot_gbm_in_sample_fit(trace, prices, trim_percentile_lower=0.1, trim_percentile_upper=0.9):
    """
    Plots the in-sample fit of the GBM model by simulating paths.
    """
    posterior = trace.posterior.stack(sample=("chain", "draw"))
    mu_mean = posterior["mu"].mean().item()
    sigma_mean = posterior["sigma"].mean().item()

    simulated_paths = generate_gbm_price_paths(
        s0=prices[0],
        T=len(prices) - 1,
        N=100,
        mu=mu_mean,
        sigma=sigma_mean
    )

    trimmed_mean_path = np.quantile(simulated_paths, q=[trim_percentile_lower, trim_percentile_upper], axis=1)

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=prices, mode='lines', name='Actual Prices', line=dict(color='red', width=2)))
    fig.add_trace(go.Scatter(y=trimmed_mean_path[0], fill=None, mode='lines', line=dict(color='blue', dash='dash'), name=f'{trim_percentile_lower*100:.0f}th Percentile'))
    fig.add_trace(go.Scatter(y=trimmed_mean_path[1], fill='tonexty', mode='lines', line=dict(color='blue', dash='dash'), name=f'{trim_percentile_upper*100:.0f}th Percentile'))

    for i in range(10):
        fig.add_trace(go.Scatter(y=simulated_paths[:, i], mode='lines', line=dict(width=0.5), showlegend=False))
    
    fig.update_layout(title="GBM In-Sample Fit", xaxis_title="Time", yaxis_title="Price")
    fig.show()


def plot_gbm_ppc(model, trace):
    """
    Plots the posterior predictive check for the GBM model.
    """
    az.plot_ppc(trace, model=model, kind='cumulative')


def plot_gbm_standardized_residuals_and_acf(trace, prices):
    """
    Plots the standardized residuals and their ACF for the GBM model.
    """
    log_returns = np.diff(np.log(prices))
    posterior = trace.posterior.stack(sample=("chain", "draw"))
    mu_mean = posterior["mu"].mean().item()
    sigma_mean = posterior["sigma"].mean().item()
    
    std_resid = (log_returns - mu_mean) / sigma_mean

    fig_resid = go.Figure()
    fig_resid.add_trace(go.Scatter(y=std_resid, mode='lines', name='Standardized Residuals'))
    fig_resid.update_layout(title="GBM Standardized Residuals", xaxis_title="Time", yaxis_title="Residual")
    fig_resid.show()

    az.plot_acf(std_resid, lags=30)

