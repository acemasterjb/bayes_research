# This file will contain the plotting functions for the risk metrics.
import arviz as az
import matplotlib.pyplot as plt

def plot_risk_distributions(var_dist, cvar_dist):
    """
    Plots the posterior distributions of VaR and CVaR.
    """
    az.plot_posterior(
        {
            "Value at Risk (VaR)": var_dist,
            "Conditional VaR (CVaR)": cvar_dist,
        },
        hdi_prob=0.9,
    )
    plt.suptitle("Posterior Distributions of Risk Metrics", y=1.02)
    plt.show()

