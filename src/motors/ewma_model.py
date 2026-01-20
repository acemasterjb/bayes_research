import pymc as pm
import pytensor.tensor as pt
import pytensor
import numpy as np


def run_ewma_model(returns: np.ndarray):
    """
    Implements a Bayesian EWMA model, which is a special case of GARCH(1,1).
    """
    with pm.Model() as ewma_model:
        # --- Priors ---
        mu = pm.Normal("mu", mu=0, sigma=0.1)
        # In EWMA, alpha + beta = 1. We model lambda (beta) and alpha is 1-lambda.
        # A tight prior on lambda is common, often around 0.94 for daily data.
        lam = pm.Beta("lambda", alpha=94, beta=6)

        alpha = pm.Deterministic("alpha", 1 - lam)
        beta = pm.Deterministic("beta", lam)
        nu = pm.Gamma("nu", alpha=2, beta=0.1)

        # --- EWMA Recursion using pytensor.scan ---
        def ewma_step(return_tm1, sigma_sq_tm1, mu, alpha, beta):
            error_tm1 = return_tm1 - mu
            sigma_sq_t = alpha * error_tm1**2 + beta * sigma_sq_tm1
            return sigma_sq_t

        initial_vol_sq = pm.Exponential("initial_vol_sq", 1 / np.std(returns) ** 2)

        sigma_sq, _ = pytensor.scan(
            fn=ewma_step,
            sequences=[returns[:-1]],
            outputs_info=[initial_vol_sq],
            non_sequences=[mu, alpha, beta],
            strict=True,
        )

        full_sigma_sq = pt.concatenate([[initial_vol_sq], sigma_sq])

        # --- Likelihood ---
        likelihood = pm.StudentT(
            "obs", nu=nu, mu=mu, sigma=pt.sqrt(full_sigma_sq), observed=returns
        )

        # --- Sampling ---
        trace = pm.sample(int(1e3), tune=1000, target_accept=0.95)

    return trace, ewma_model


def generate_ewma_price_paths(
    s0: float,
    T: int,
    N: int,
    mu: float,
    alpha: float,
    beta: float,
    nu: float,
    last_return: float,
    last_vol_sq: float,
):
    """
    Generates N price paths over T steps for a single set of EWMA parameters.
    """
    returns_matrix = np.zeros((T, N))

    current_returns = np.full(N, last_return)
    current_vol_sqs = np.full(N, last_vol_sq)

    rng = np.random.default_rng()

    for t in range(T):
        errors = current_returns - mu
        next_vol_sqs = alpha * errors**2 + beta * current_vol_sqs

        # Student-T shocks
        shocks = rng.standard_t(df=nu, size=N)

        next_returns = mu + np.sqrt(next_vol_sqs) * shocks

        returns_matrix[t, :] = next_returns
        current_returns = next_returns
        current_vol_sqs = next_vol_sqs

    price_paths = s0 * np.exp(np.cumsum(returns_matrix, axis=0))
    price_paths = np.vstack([np.full(N, s0), price_paths])

    return price_paths

