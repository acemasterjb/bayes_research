import arviz as az
import numpy as np
import pymc as pm
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from scipy import stats
import plotly.graph_objects as go
import matplotlib.pyplot as plt


def plot_holdout_forecast(test_data, median_forecast, forecast_samples):
    """
    Plots the holdout forecast against the actual observed data using Plotly.
    Calculates the HDI for each time step for accurate visualization.
    """
    fig = make_subplots(
        rows=2, cols=1, subplot_titles=("Log-Borrow Volume", "Log-Supply Volume")
    )

    time_axis = np.arange(len(test_data))
    n_vars = test_data.shape[1]

    colors = {
        "borrow": {"fill": "rgba(0,100,80,0.2)", "name": "Observed Borrow"},
        "supply": {"fill": "rgba(255,140,0,0.2)", "name": "Observed Supply"},
    }

    for i, var_name in enumerate(["borrow", "supply"]):
        # FIX: Calculate HDI per time step for plotting
        hdi_var = np.array(
            [az.hdi(forecast_samples[:, t, i], hdi_prob=0.9) for t in range(len(test_data))]
        )
        hdi_lower = hdi_var[:, 0]
        hdi_upper = hdi_var[:, 1]

        # --- Plot HDI bounds ---
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([time_axis, time_axis[::-1]]),
                y=np.concatenate([hdi_upper, hdi_lower[::-1]]),
                fill="toself",
                fillcolor=colors[var_name]["fill"],
                line=dict(color="rgba(255,255,255,0)"),
                hoverinfo="skip",
                name="90% HDI",
                showlegend=(i == 0),
            ),
            row=i + 1,
            col=1,
        )

        # --- Plot Observed Data ---
        fig.add_trace(
            go.Scatter(
                x=time_axis, y=test_data[:, i], mode="lines", name=colors[var_name]["name"]
            ),
            row=i + 1,
            col=1,
        )

        # --- Plot Median Forecast ---
        fig.add_trace(
            go.Scatter(
                x=time_axis,
                y=median_forecast[:, i],
                mode="lines",
                name="Median Forecast",
                line=dict(dash="dash"),
                showlegend=(i == 0),
            ),
            row=i + 1,
            col=1,
        )

    fig.update_layout(
        title_text="Holdout Forecast vs. Observed Data (90-Day Forecast)",
        height=600,
    )
    fig.show()



def plot_lfo_cv_results(log_pds, pareto_ks):
    """
    Visualizes the results of the Leave-Future-Out Cross-Validation.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Cumulative ELPD (Expected Log Predictive Density)
    cumulative_elpd = np.cumsum(log_pds)
    axes[0].plot(cumulative_elpd)
    axes[0].set_title("Cumulative ELPD (LFO-CV)")
    axes[0].set_xlabel("Forecast Step")
    axes[0].set_ylabel("Cumulative Log Predictive Score")

    # Plot 2: Pareto K Diagnostics
    axes[1].hist(pareto_ks, bins=20)
    axes[1].set_title("Pareto-K Diagnostics")
    axes[1].set_xlabel("Pareto K Value")
    axes[1].set_ylabel("Count")

    plt.tight_layout()
    plt.show()

    print("\\n--- LFO-CV Summary ---")
    print(f"Mean ELPD: {np.mean(log_pds):.2f}")
    print(
        f"Number of 'bad' or 'very bad' Pareto K values (>0.7): {np.sum(pareto_ks > 0.7)}"
    )


def plot_priors_vs_posteriors_bvar(trace, model):
    """
    Plots prior vs. posterior distributions for key BVAR parameters using Plotly.
    """
    # Focus on key parameters for clarity
    var_map = {
        "alpha_0": {"prior": stats.norm(loc=0, scale=1), "posterior_key": ("alpha", 0)},
        "alpha_1": {"prior": stats.norm(loc=0, scale=1), "posterior_key": ("alpha", 1)},
        "beta_0_borrow_borrow": {
            "prior": stats.norm(loc=0, scale=0.5),
            "posterior_key": ("beta", 0, 0, 0),
        },
        "beta_0_supply_supply": {
            "prior": stats.norm(loc=0, scale=0.5),
            "posterior_key": ("beta", 0, 1, 1),
        },
        # "nu": {"prior": stats.gamma(a=2, scale=1/0.1), "posterior_key": ("nu",)},
    }

    fig = make_subplots(
        rows=len(var_map),
        cols=1,
        subplot_titles=list(var_map.keys()),
    )

    for i, (title, info) in enumerate(var_map.items()):
        key = info["posterior_key"]
        # Access posterior data using .sel() for coordinates or direct key for others
        posterior_data = trace.posterior
        for k in key:
            if isinstance(k, str):
                posterior_data = posterior_data[k]
            else:  # Integer index for dimensions
                posterior_data = posterior_data.isel({posterior_data.dims[-1]: k})

        posterior_samples = posterior_data.stack(sample=("chain", "draw")).values
        prior_samples = info["prior"].rvs(size=len(posterior_samples))

        hist_data = [prior_samples, posterior_samples]
        group_labels = ["Prior", "Posterior"]

        dist_fig = ff.create_distplot(
            hist_data, group_labels, show_rug=False, bin_size=0.05
        )

        for trace_data in dist_fig.data:
            fig.add_trace(trace_data, row=i + 1, col=1)

    fig.update_layout(
        title_text="Prior vs. Posterior Distributions for Key Parameters",
        height=300 * len(var_map),
        showlegend=True,
    )
    fig.show()


def plot_posterior_predictive_check_bvar(model, trace):
    """
    Performs and plots a posterior predictive check for the BVAR model.
    """
    with model:
        ppc = pm.sample_posterior_predictive(trace)
    az.plot_ppc(ppc, var_names=["obs"], kind="kde")
    plt.show()


def _plot_predicted_vs_observed_plotly(y_data, mu_pred, n_lags):
    """Helper to plot observed vs. predicted values using Plotly."""
    fig = make_subplots(
        rows=2, cols=1, subplot_titles=("Log-Borrow Volume", "Log-Supply Volume")
    )

    # The time axis for the plots starts after the initial lag period
    plot_time_axis = np.arange(len(y_data) - n_lags)

    # Log-Borrow Plot
    fig.add_trace(
        go.Scatter(
            x=plot_time_axis,
            y=y_data[n_lags:, 0],
            mode="lines",
            name="Observed Borrow",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=plot_time_axis,
            y=mu_pred[:, 0],
            mode="lines",
            name="Predicted Borrow (Mean)",
            line=dict(dash="dash"),
        ),
        row=1,
        col=1,
    )

    # Log-Supply Plot
    fig.add_trace(
        go.Scatter(
            x=plot_time_axis,
            y=y_data[n_lags:, 1],
            mode="lines",
            name="Observed Supply",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=plot_time_axis,
            y=mu_pred[:, 1],
            mode="lines",
            name="Predicted Supply (Mean)",
            line=dict(dash="dash"),
        ),
        row=2,
        col=1,
    )

    fig.update_layout(
        title_text="Observed vs. Predicted Mean (In-Sample Fit)",
        height=600,
        showlegend=True,
    )
    fig.show()


def plot_residuals_bvar(trace, y_data, n_lags):
    """
    Analyzes and plots the model's residuals for each variable.
    """
    # Calculate posterior mean of parameters
    posterior = trace.posterior.stack(sample=("chain", "draw"))
    alpha_mean = posterior["alpha"].mean(axis=-1).values
    beta_mean = posterior["beta"].mean(axis=-1).values

    # Calculate predicted mean
    mu_pred = alpha_mean
    for i in range(n_lags):
        mu_pred = mu_pred + np.dot(y_data[n_lags - (i + 1) : -(i + 1), :], beta_mean[i])

    residuals = y_data[n_lags:] - mu_pred

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle("Model Residuals Analysis")

    # Residuals vs. Time
    axes[0, 0].plot(residuals[:, 0])
    axes[0, 0].set_title("Log-Borrow Residuals vs. Time")
    axes[1, 0].plot(residuals[:, 1], color="orange")
    axes[1, 0].set_title("Log-Supply Residuals vs. Time")

    # Histogram of Residuals
    axes[0, 1].hist(residuals[:, 0], bins=30, density=True)
    axes[0, 1].set_title("Log-Borrow Residuals Distribution")
    axes[1, 1].hist(residuals[:, 1], bins=30, density=True, color="orange")
    axes[1, 1].set_title("Log-Supply Residuals Distribution")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

    # Also show the interactive observed vs. predicted plot
    _plot_predicted_vs_observed_plotly(y_data, mu_pred, n_lags)
