# src/examples/2_6.py

import os
import sys

import plotly.graph_objects as go
import pyro
import pyro.distributions as dist
import torch
from pyro.distributions import Binomial, Uniform

# Add the root of the project to the Python path
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from src.motors import get_quad_approx


def model(p, W, L):
    """
    Pyro model for binomial likelihood with a uniform prior on p.

    Args:
        p (torch.Tensor): The probability of success.
        W (float): The number of successes (wins).
        L (float): The number of failures (losses).
    """
    N = torch.tensor(W + L, dtype=torch.float)
    # Uniform prior for p
    pyro.sample("p_prior", Uniform(0.0, 1.0), obs=p)
    # Binomial likelihood for the observed data W
    pyro.sample("W", Binomial(N, p), obs=torch.tensor(W, dtype=torch.float))


def main():
    """
    Calculates and plots the quadratic approximation of a posterior distribution
    using the centralized `get_quad_approx` function and compares it to the
    true Beta posterior.
    """
    W, L = 6.0, 3.0
    weights = [W, L]

    # --- 1. Get the quadratic approximation from the motors module ---
    mu, std_dev, estimates = get_quad_approx(model, weights)

    print(f"MAP (mu): {mu:.4f}, Std Dev: {std_dev:.4f}")

    # --- 2. Plot the distributions ---
    p_range = torch.linspace(0.001, 0.999, 100)

    # Quadratic approximation (Normal distribution)
    quad_approx_dist = dist.Normal(mu, std_dev)
    quad_approx_pdf = torch.exp(quad_approx_dist.log_prob(p_range))

    # True posterior (Beta distribution)
    # For a Binomial likelihood and Uniform prior, the posterior is Beta(W+1, L+1)
    beta_dist = dist.Beta(1 + W, 1 + L)
    beta_pdf = torch.exp(beta_dist.log_prob(p_range))

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=p_range,
            y=quad_approx_pdf,
            mode="lines",
            name="Quadratic Approximation (Normal)",
        )
    )
    fig.add_trace(
        go.Scatter(x=p_range, y=beta_pdf, mode="lines", name="True Posterior (Beta)")
    )
    fig.update_layout(
        title="Quadratic Approximation vs. True Posterior",
        xaxis_title="Parameter Value (p)",
        yaxis_title="Density",
    )
    fig.show()

    # --- 3. Plot the estimates ---
    fig_estimates = go.Figure()
    fig_estimates.add_trace(
        go.Scatter(
            x=list(range(len(estimates))),
            y=estimates,
            mode='lines',
            name='Estimates'
        )
    )
    fig_estimates.update_layout(
        title='Estimates Progression',
        xaxis_title='Steps',
        yaxis_title='*p*'
    )
    fig_estimates.show()


if __name__ == "__main__":
    main()
