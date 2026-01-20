import pymc as pm
import numpy as np


def run_gbm_model(prices: np.ndarray):
    """
    Implements a Bayesian Geometric Brownian Motion (GBM) model.
    """
    log_returns = np.diff(np.log(prices))

    with pm.Model() as gbm_model:
        # --- Priors ---
        # Drift (mu)
        mu = pm.Normal("mu", mu=log_returns.mean(), sigma=log_returns.std())
        # Volatility (sigma)
        sigma = pm.HalfNormal("sigma", sigma=log_returns.std())

        # --- Likelihood ---
        # The likelihood of log-returns in GBM is a Normal distribution
        likelihood = pm.Normal(
            "obs", mu=mu, sigma=sigma, observed=log_returns
        )

        # --- Sampling ---
        trace = pm.sample(int(1e3), tune=1000, target_accept=0.95)

    return trace, gbm_model


def generate_gbm_price_paths(s0, T, N, mu, sigma):
    """
    Generates N price paths over T steps for a single set of GBM parameters.
    """
    dt = 1  # Assuming daily steps
    log_returns = np.random.normal(mu - 0.5 * sigma**2, sigma, (T, N))
    price_paths = np.exp(np.log(s0) + np.cumsum(log_returns, axis=0))
    price_paths = np.vstack([np.full(N, s0), price_paths])
    return price_paths
