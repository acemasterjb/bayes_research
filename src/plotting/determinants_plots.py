import arviz as az
import matplotlib.pyplot as plt
import pymc as pm
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from scipy import stats


def plot_priors_vs_posteriors_determinants(trace):
    """
    Plots prior vs. posterior distributions for the determinant model parameters.
    """
    # Define the priors directly, mirroring the model definition
    priors = {
        "alpha": stats.norm(loc=0, scale=1),
        "beta_mean_returns": stats.norm(loc=0, scale=1),
        "beta_std_returns": stats.expon(scale=1 / 0.5),
        "beta_log_volume": stats.norm(loc=0, scale=10),
        "beta_supply_incentives": stats.norm(loc=0, scale=1),
        "beta_borrow_incentives": stats.norm(loc=0, scale=1),
        "sigma": stats.expon(scale=1 / 0.5),
    }
    var_names = list(priors.keys())

    fig = make_subplots(
        rows=len(var_names),
        cols=1,
        subplot_titles=var_names,
    )

    for i, var in enumerate(var_names):
        prior_samples = priors[var].rvs(size=10000)
        posterior_samples = trace.posterior[var].stack(sample=("chain", "draw")).values

        hist_data = [prior_samples, posterior_samples]
        group_labels = ["Prior", "Posterior"]

        dist_fig = ff.create_distplot(hist_data, group_labels, show_rug=False)

        for trace_data in dist_fig.data:
            fig.add_trace(trace_data, row=i + 1, col=1)

    fig.update_layout(
        title_text="Prior vs. Posterior Distributions",
        height=300 * len(var_names),
        showlegend=True,
    )
    fig.show()


def plot_posterior_predictive_check_determinants(model, trace):
    """
    Performs and plots a posterior predictive check for the determinants model.
    """

    minimized_trace = trace.sel(draw=slice(None, None, 1000))
    with model:
        ppc = pm.sample_posterior_predictive(minimized_trace, var_names=["delta_s"])

    az.plot_ppc(ppc, var_names=["delta_s"], kind="kde")
    plt.show()
