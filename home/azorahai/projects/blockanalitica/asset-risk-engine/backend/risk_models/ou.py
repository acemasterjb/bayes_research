import numpy as np
import pymc as pm

N_SAMPLES = int(1e4)

def run_ar1_model(log_prices: np.ndarray):
    """
    Fits an AR(1) model to a log-price series.
    """
    with pm.Model() as ar1_model:
        m_mu = np.percentile(log_prices, 50)
        m_sigma = np.std(log_prices)
        m = pm.Normal("m", mu=m_mu, sigma=m_sigma)
        phi = pm.Beta("phi", alpha=2, beta=2)
        sigma = pm.HalfNormal("sigma", sigma=1.0)

        x_prev = log_prices[:-1]
        mu_t = m + phi * (x_prev - m)
        
        pm.Normal("obs", mu=mu_t, sigma=sigma, observed=log_prices[1:])

        trace = pm.sample(N_SAMPLES, tune=1000, target_accept=0.95, progressbar=False)

    return trace

def derive_ou_parameters(trace, percentile: float = 0.5) -> dict:
    """
    Extracts the final OU/AR(1) parameters from a fitted trace.
    """
    posterior = trace.posterior.stack(sample=("chain", "draw"))
    
    return {
        "phi_persistence": np.quantile(posterior["phi"], q=percentile),
        "mean_log_price": np.quantile(posterior["m"], q=percentile),
        "volatility": np.quantile(posterior["sigma"], q=percentile),
    }
