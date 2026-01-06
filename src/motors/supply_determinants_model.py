import pymc as pm
import pandas as pd
import numpy as np

N_SAMPLES = int(1e6)


def run_supply_determinants_model(
    X_aggregate: pd.DataFrame, y_aggregate: pd.Series, category_indices: np.ndarray
):
    """
    Runs the Bayesian regression on aggregated data from multiple markets.
    """
    with pm.Model() as model:
        # --- 2. Priors ---
        alpha = pm.Normal("alpha", mu=10, sigma=0.5)
        beta_0 = pm.Normal("beta_mean_returns", mu=0, sigma=1)
        beta_1 = pm.Exponential("beta_std_returns", lam=2)
        beta_2 = pm.Normal(
            "beta_log_volume",
            mu=X_aggregate["log_volume"].quantile(0.85),
            sigma=X_aggregate["log_volume"].std(),
        )
        beta_3 = pm.Normal("beta_classification", mu=0, sigma=1, shape=4)
        beta_supply_incentives = pm.Normal("beta_supply_incentives", mu=0, sigma=1)
        beta_borrow_incentives = pm.Normal("beta_borrow_incentives", mu=0, sigma=1)
        sigma = pm.Exponential("sigma", lam=1 / y_aggregate.std())

        # --- 3. Linear Model (mu) ---
        mu = (
            alpha
            + X_aggregate["mean_returns_14d"].values * beta_0
            + X_aggregate["std_returns_14d"].values * beta_1
            + X_aggregate["log_volume"].values * beta_2
            + beta_3[category_indices]  # Advanced indexing for categorical feature
            + X_aggregate["isIncentivisedSupply"].values * beta_supply_incentives
            + X_aggregate["isIncentivisedBorrow"].values * beta_borrow_incentives
        )

        # --- 4. Likelihood ---
        delta_s = pm.Normal("delta_s", mu=mu, sigma=sigma, observed=y_aggregate.values)  # noqa: F841

        # --- 5. Sampling ---
        trace = pm.sample(N_SAMPLES, tune=2000, target_accept=0.95, progressbar=True)

    return trace, model
