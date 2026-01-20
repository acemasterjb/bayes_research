# This file will contain the logic for calculating VaR and CVaR distributions.
import arviz as az
import numpy as np

from src.motors.ou_model import generate_ar_paths
from src.motors.garch_model import generate_garch_price_paths


def calculate_risk_metrics(price_paths, initial_price, alpha=0.05):
    """
    Calculates Value at Risk (VaR) and Conditional Value at Risk (CVaR).
    """
    final_prices = price_paths[-1, :]
    total_returns = (final_prices - initial_price) / initial_price
    sorted_returns = np.sort(total_returns)

    var_idx = int(alpha * len(sorted_returns))
    var_value = sorted_returns[var_idx]
    cvar_value = np.mean(sorted_returns[:var_idx])

    return var_value, cvar_value


def get_ar1_risk_distribution(trace, s0, T, N_per_sample, alpha=0.05):
    """
    Generates a distribution of VaR and CVaR estimates for an AR(1) model.
    """
    # Use xarray's stack and sample methods for robust sampling
    thinned_samples = az.extract(trace, num_samples=100)

    var_dist = []
    cvar_dist = []

    for i in range(len(thinned_samples.sample)):
        sample = thinned_samples.isel(sample=i)
        paths = generate_ar_paths(
            current_price=s0,
            dt=1.0,
            steps=T,
            n_paths=N_per_sample,
            phi=sample["phi"].item(),
            m_log=sample["m"].item(),
            sigma=sample["sigma"].item(),
        )

        var, cvar = calculate_risk_metrics(paths, s0, alpha)
        var_dist.append(var)
        cvar_dist.append(cvar)

    return np.array(var_dist), np.array(cvar_dist)


def get_garch_risk_distribution(
    trace,
    returns: np.ndarray,
    s0: float,
    T: int,
    N_per_sample: int,
    p_jump: float,
    mu_jump: float,
    sigma_jump: float,
    alpha: float = 0.05,
):
    """
    Generates a distribution of VaR and CVaR estimates for a GARCH model by
    iterating through posterior samples.
    """
    # ... (code to thin trace and calculate last_vol_sq remains the same) ...
    posterior = trace.posterior.stack(sample=("chain", "draw"))
    thinned_samples = az.extract(trace, num_samples=100)

    mu_mean = posterior["mu"].mean().item()
    omega_mean = posterior["omega"].mean().item()
    alpha_mean = posterior["alpha"].mean().item()
    beta_mean = posterior["beta"].mean().item()
    initial_vol_sq_mean = posterior["initial_vol_sq"].mean().item()

    vol_sq_series = np.zeros(len(returns))
    vol_sq_series[0] = initial_vol_sq_mean
    for t in range(1, len(returns)):
        err_tm1 = returns[t - 1] - mu_mean
        vol_sq_series[t] = (
            omega_mean + alpha_mean * err_tm1**2 + beta_mean * vol_sq_series[t - 1]
        )
    last_vol_sq = vol_sq_series[-1]
    last_return = returns[-1]

    var_dist = []
    cvar_dist = []

    for i in range(len(thinned_samples.sample)):
        sample = thinned_samples.isel(sample=i)

        paths = generate_garch_price_paths(
            s0=s0,
            T=T,
            N=N_per_sample,
            mu=sample["mu"].item(),
            omega=sample["omega"].item(),
            alpha=sample["alpha"].item(),
            beta=sample["beta"].item(),
            last_return=last_return,
            last_vol_sq=last_vol_sq,
            p_jump=p_jump,
            mu_jump=mu_jump,
            sigma_jump=sigma_jump,
        )

        var, cvar = calculate_risk_metrics(paths, s0, alpha)
        var_dist.append(var)
        cvar_dist.append(cvar)

    return np.array(var_dist), np.array(cvar_dist)
