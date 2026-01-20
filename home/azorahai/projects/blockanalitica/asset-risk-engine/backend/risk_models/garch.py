import numpy as np
import pymc as pm
import pytensor

def run_garch_model(returns: np.ndarray):
    """
    Implements a Bayesian GARCH(1,1) model with a Student-T distribution.
    """
    with pm.Model() as garch_model:
        mu = pm.Normal("mu", mu=0, sigma=0.1)
        omega = pm.HalfNormal("omega", sigma=0.1)
        alpha_garch = pm.HalfNormal("alpha", sigma=0.3)
        beta_garch = pm.HalfNormal("beta", sigma=0.3)
        nu = pm.Gamma("nu", alpha=2, beta=0.1)

        def garch_step(return_tm1, sigma_sq_tm1, mu, omega, alpha, beta):
            error_tm1 = return_tm1 - mu
            sigma_sq_t = omega + alpha * error_tm1**2 + beta * sigma_sq_tm1
            return sigma_sq_t

        initial_vol_sq = pm.Exponential("initial_vol_sq", 1 / np.std(returns) ** 2)

        sigma_sq, _ = pytensor.scan(
            fn=garch_step,
            sequences=[returns[:-1]],
            outputs_info=[initial_vol_sq],
            non_sequences=[mu, omega, alpha_garch, beta_garch],
            strict=True,
        )
        
        full_sigma_sq = pm.pytensor.tensor.concatenate([[initial_vol_sq], sigma_sq])

        pm.StudentT("obs", nu=nu, mu=mu, sigma=pm.pytensor.tensor.sqrt(full_sigma_sq), observed=returns)

        trace = pm.sample(1000, tune=1000, target_accept=0.95, progressbar=False)

    return trace

def derive_garch_parameters(trace, percentile: float = 0.5) -> dict:
    """
    Extracts the final GARCH parameters from a fitted trace.
    """
    posterior = trace.posterior.stack(sample=("chain", "draw"))
    
    return {
        "mu_mean_return": np.quantile(posterior["mu"], q=percentile),
        "omega_variance_intercept": np.quantile(posterior["omega"], q=percentile),
        "alpha_arch_term": np.quantile(posterior["alpha"], q=percentile),
        "beta_garch_term": np.quantile(posterior["beta"], q=percentile),
        "nu_degrees_of_freedom": np.quantile(posterior["nu"], q=percentile),
    }
