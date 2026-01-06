import arviz as az
import numpy as np
import plotly.graph_objects as go
from matplotlib import pyplot as plt
import pymc as pm
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from scipy import stats

from src.motors.ou_model import generate_ar_paths


def plot_priors_vs_posteriors(trace, m_mu, m_sigma):
    """
    Plots the prior vs. posterior distributions for the main parameters using Plotly.
    """
    var_names = ["phi", "m", "sigma"]
    prior_dists = {
        "phi": stats.beta(a=2, b=2),
        "m": stats.norm(loc=m_mu, scale=m_sigma),
        "sigma": stats.expon(scale=1 / 10.0),
    }

    fig = make_subplots(
        rows=len(var_names),
        cols=1,
        subplot_titles=[f"Distribution for {var}" for var in var_names],
    )

    for i, var in enumerate(var_names):
        posterior_samples = trace.posterior[var].values.flatten()
        prior_samples = prior_dists[var].rvs(size=10000)

        # Create distribution plots
        hist_data = [prior_samples, posterior_samples]
        group_labels = ["Prior", "Posterior"]
        colors = ["#333333", "#3178C6"]

        dist_fig = ff.create_distplot(
            hist_data, group_labels, bin_size=0.05, colors=colors, show_rug=False
        )

        # Add traces to subplot
        for trace_data in dist_fig.data:
            fig.add_trace(trace_data, row=i + 1, col=1)

    fig.update_layout(
        title_text="Prior vs. Posterior Distributions",
        height=300 * len(var_names),
        showlegend=True,
    )
    fig.show()


def plot_posterior_predictive_check(model, trace):
    """
    Performs and plots a posterior predictive check.
    """
    minimized_trace = trace.sel(draw=slice(None, None, 50))
    with model:
        ppc = pm.sample_posterior_predictive(minimized_trace)
    az.plot_ppc(ppc, kind="kde", num_pp_samples=50)
    plt.show()


def plot_residuals(trace, observed_log_prices):
    """
    Analyzes and plots the model's residuals.
    """
    phi_mean = trace.posterior["phi"].mean().item()
    m_mean = trace.posterior["m"].mean().item()

    x_prev = observed_log_prices[:-1]
    predicted_log_prices = m_mean + phi_mean * (x_prev - m_mean)
    residuals = observed_log_prices[1:] - predicted_log_prices

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Residuals vs. Time
    ax1.plot(residuals)
    ax1.set_title("Residuals vs. Time")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Residual")

    # Histogram of Residuals
    ax2.hist(residuals, bins=30, density=True)
    ax2.set_title("Histogram of Residuals")
    ax2.set_xlabel("Residual")
    ax2.set_ylabel("Density")

    plt.tight_layout()
    plt.show()


def plot_posterior(trace):
    """
    Visualizes the posterior distributions of the model parameters.
    """
    az.plot_posterior(trace)
    plt.show()


def plot_ar1_in_sample_fit(
    trace,
    prices: np.ndarray,
    p_jump: float,
    mu_jump: float,
    sigma_jump: float,
    n_paths: int = 100,
    percentile: float = 0.5,
    trim_percentile_lower: float = 0.1,
    trim_percentile_upper: float = 0.9,
):
    """
    Generates and plots in-sample simulated paths for the AR(1) model,
    including a trimmed mean path.
    """
    # 1. Extract parameters from the specified posterior percentile
    posterior = trace.posterior.stack(sample=("chain", "draw"))
    phi_p = np.quantile(posterior["phi"], q=percentile)
    m_p = np.quantile(posterior["m"], q=percentile)
    sigma_p = np.quantile(posterior["sigma"], q=percentile)

    # 2. Generate a large number of simulated paths
    all_simulated_paths = generate_ar_paths(
        current_price=prices[0],
        dt=1.0,  # Assuming daily data
        steps=len(prices),
        n_paths=n_paths,
        phi=phi_p,
        m_log=m_p,
        sigma=sigma_p,
        p_jump=p_jump,
        mu_jump=mu_jump,
        sigma_jump=sigma_jump,
    )
    fig = go.Figure()

    # 3. Plot a subset of simulated paths
    for i in range(min(n_paths, 10)):
        fig.add_trace(
            go.Scatter(
                x=np.arange(len(prices)),
                y=all_simulated_paths[:, i],
                mode="lines",
                name=f"Simulated Path {i+1}",
                line=dict(dash="dash", width=1),
                opacity=0.7,
            )
        )

    # 4. Calculate and Plot Trimmed Mean Path
    trimmed_mean_path = np.zeros(len(prices))
    for t in range(len(prices)):
        paths_at_t = all_simulated_paths[t]
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
            name=f"Trimmed Mean ({int(trim_percentile_lower*100)}-{int(trim_percentile_upper*100)}%)",
            line=dict(color="red", width=2),
        )
    )

    # 5. Plot Observed Prices
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
        title=f"AR(1) In-Sample Fit (Simulated from {int(percentile*100)}th Percentile Parameters)",
        xaxis_title="Time Steps",
        yaxis_title="Price",
        showlegend=True,
    )
    fig.show()


def plot_simulated_paths(prices, simulated_paths, n_paths):
    """
    Plots the simulated price paths. (DEPRECATED by plot_ar1_in_sample_fit)
    """
    fig = go.Figure()
    for i in range(n_paths):
        fig.add_trace(
            go.Scatter(
                x=np.arange(len(prices)),
                y=simulated_paths[:, i],
                mode="lines",
                name=f"Simulated Path {i+1}",
            )
        )
    fig.update_layout(
        title="Simulated Price Paths", xaxis_title="Time", yaxis_title="Price"
    )
    fig.show()


def plot_empirical_vs_simulated(prices, simulated_paths):
    """
    Compares the empirical price path with the mean of the simulated paths. (DEPRECATED by plot_ar1_in_sample_fit)
    """
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=np.arange(len(prices)),
            y=prices,
            mode="lines",
            name="Empirical Prices",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=np.arange(len(prices)),
            y=np.mean(simulated_paths, axis=1),
            mode="lines",
            name="Mean Simulated Price",
        )
    )
    fig.update_layout(
        title="Empirical vs. Simulated Prices",
        xaxis_title="Time",
        yaxis_title="Price",
    )
    fig.show()
