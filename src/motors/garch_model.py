# This file will contain the Bayesian GARCH(1,1)-T model.
import pymc as pm
import pytensor.tensor as pt
import pytensor
import numpy as np


def forecast_garch_model(trace, last_return, last_vol_sq, n_forecast_steps: int = 30):
    """
    Forecasts future returns and volatility using the trained GARCH model.
    """
    # Robustly extract posterior samples
    posterior = trace.posterior.stack(sample=("chain", "draw"))
    mu = np.moveaxis(posterior["mu"].values, -1, 0)
    omega = np.moveaxis(posterior["omega"].values, -1, 0)
    alpha = np.moveaxis(posterior["alpha"].values, -1, 0)
    beta = np.moveaxis(posterior["beta"].values, -1, 0)
    # nu = np.moveaxis(posterior["nu"].values, -1, 0)
    n_samples = len(mu)

    # Initialize arrays
    forecasted_returns = np.zeros((n_samples, n_forecast_steps))
    forecasted_vol_sq = np.zeros((n_samples, n_forecast_steps))

    # Get last known values to start the recursion
    current_return = np.full(n_samples, last_return)
    current_vol_sq = np.full(n_samples, last_vol_sq)

    # Iteratively forecast each step for each sample
    for t in range(n_forecast_steps):
        # Calculate error and next variance
        error_t = current_return - mu
        next_vol_sq = omega + alpha * error_t**2 + beta * current_vol_sq

        # Draw next return from the Student-T distribution
        # Student-T noise = Normal noise / sqrt(Chi2/nu)
        random_shock = np.random.standard_normal(n_samples)
        chi2 = np.random.chisquare(df=3, size=n_samples)
        student_t_shock = random_shock * np.sqrt(1 / chi2)

        next_return = mu + np.sqrt(next_vol_sq) * student_t_shock

        # Store results and update current state
        forecasted_returns[:, t] = next_return
        forecasted_vol_sq[:, t] = next_vol_sq
        current_return = next_return
        current_vol_sq = next_vol_sq

    return forecasted_returns


def run_garch_model(returns: np.ndarray):
    """
    Implements a Bayesian GARCH(1,1) model with a Student-T distribution.
    """
    with pm.Model() as garch_model:
        # --- Priors ---
        mu = pm.Normal("mu", mu=0, sigma=0.1)
        omega = pm.HalfNormal("omega", sigma=0.1)
        phi = pm.Beta("phi", alpha=5, beta=1.5)
        rho = pm.Beta("rho", alpha=2, beta=10)

        alpha = pm.Deterministic("alpha", phi * rho)
        beta = pm.Deterministic("beta", phi * (1 - rho))
        # nu = pm.Gamma("nu", alpha=2, beta=0.1)

        # --- GARCH(1,1) Recursion using pytensor.scan ---
        # The GARCH model requires a recursive function to compute the variance at each time step.
        # `scan` is the tool for this in PyTensor/PyMC.

        # The inner function for one step of the recursion.
        # It takes the PREVIOUS return (from sequences), the PREVIOUS variance (from outputs_info),
        # and the non-sequences.
        def garch_step(return_tm1, sigma_sq_tm1, mu, omega, alpha, beta):
            # Error term at time t-1 (the innovation)
            error_tm1 = return_tm1 - mu
            # GARCH(1,1) variance equation for time t
            sigma_sq_t = omega + alpha * error_tm1**2 + beta * sigma_sq_tm1
            return sigma_sq_t

        # Initialize the variance. A common approach is to use the unconditional variance of the series.
        initial_vol_sq = pm.Exponential("initial_vol_sq", 1 / np.std(returns) ** 2)

        # The `scan` function itself.
        # It iterates over the returns from t=0 to t=T-1.
        # At each step `t`, `garch_step` receives `returns[t-1]` and `sigma_sq[t-1]`
        # to calculate `sigma_sq[t]`.
        sigma_sq, _ = pytensor.scan(
            fn=garch_step,
            sequences=[returns[:-1]],  # Iterate over returns from t=0..T-1
            outputs_info=[initial_vol_sq],  # Initial state for sigma_sq at t=0
            non_sequences=[mu, omega, alpha, beta],
            strict=True,
        )

        # We now have the conditional variance for t=1..T.
        # The full variance series must be prepended with the initial variance for t=0.
        full_sigma_sq = pt.concatenate([[initial_vol_sq], sigma_sq])

        # --- Likelihood ---
        # The likelihood is a Student-T distribution, where the volatility (sigma)
        # at each time step `t` is determined by the GARCH recursion.
        likelihood = pm.Normal(
            "obs", mu=mu, sigma=pt.sqrt(full_sigma_sq), observed=returns
        )

        # --- Sampling ---
        trace = pm.sample(int(1e3), tune=1000, target_accept=0.95)

    return trace, garch_model
