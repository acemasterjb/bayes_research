import arviz as az
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import numpy as np

N_SAMPLES = int(2e3)


def run_lfo_cv(y_data, n_lags: int = 2, min_train_points: int = 50):
    """
    Performs Pareto-Smoothed-Importance-Sampling Leave-Future-Out Cross-Validation.
    """
    n_vars = y_data.shape[1]
    n_total_points = y_data.shape[0]

    # We will compute the log predictive density for each point from min_train_points onwards
    log_pds = []
    pareto_ks = []

    for t in range(min_train_points, n_total_points):
        # 1. Define the model using data up to time t-1
        train_data = y_data[:t]

        with pm.Model() as refit_model:
            chol, _, _ = pm.LKJCholeskyCov(
                "noise_chol", n=n_vars, eta=2.0, sd_dist=pm.Exponential.dist(1.0)
            )
            beta = pm.Normal("beta", mu=0, sigma=0.05, shape=(n_lags, n_vars, n_vars))
            alpha = pm.Normal("alpha", mu=0, sigma=0.05, shape=n_vars)

            mu = alpha
            for i in range(n_lags):
                mu = mu + pt.dot(train_data[len(train_data) - (i + 1)], beta[i])

            # This is the key: we define the likelihood but don't observe it yet
            obs_dist = pm.MvNormal.dist(mu=mu, chol=chol)

            # This is the likelihood of the training data
            pm.MvNormal("obs_train", mu=mu, chol=chol, observed=train_data[n_lags:])

            # 2. Fit the model
            trace = pm.sample(
                N_SAMPLES,
                return_inferencedata=True,
                target_accept=0.95,
                progressbar=False,
            )

        # 3. Calculate the log-likelihood of the NEXT unseen data point
        # The point to predict is y_data[t]
        point_to_predict = y_data[t]

        # We need to evaluate the logp of obs_dist for each posterior sample
        logp_vals_fn = pm.compile_fn(
            inputs=list(refit_model.free_RVs),
            outs=pm.logp(obs_dist, pt.as_tensor(point_to_predict)),
            random_seed=True,
        )

        log_weights = np.array(
            [
                logp_vals_fn(
                    *[trace.posterior[var].values[c, d] for var in refit_model.free_RVs]
                )
                for c in range(trace.posterior.chain.size)
                for d in range(trace.posterior.draw.size)
            ]
        )

        # 4. Use ArviZ for Pareto-Smoothing
        psis_result = az.psislw(log_weights)
        log_pds.append(psis_result[0])
        pareto_ks.append(psis_result[1])

        print(
            f"CV Step {t - min_train_points + 1}/{n_total_points - min_train_points} complete."
        )

    return np.array(log_pds), np.array(pareto_ks)


def train_bvar_model(market_data: pd.DataFrame, n_lags: int = 2):
    """
    Trains the Bayesian Vector Autoregression model on historical market data.
    """
    # Data Preparation: Log-transform your Supply and Borrow volumes
    y_data = np.log(market_data[["borrow_volume", "supply_volume"]].values)
    n_vars = y_data.shape[1]  # Dynamically get number of variables (should be 2)

    with pm.Model() as bvar_model:  # noqa: F841
        # --- 1. Priors for Correlation ---
        chol, _, _ = pm.LKJCholeskyCov(
            "noise_chol",
            n=n_vars,  # Use n_vars for robustness
            eta=2.0,
            sd_dist=pm.Exponential.dist(1.0),
            compute_corr=True,
        )

        # --- 2. Priors for Autoregressive Coefficients ---
        beta = pm.Normal("beta", mu=0, sigma=0.05, shape=(n_lags, n_vars, n_vars))
        alpha = pm.Normal("alpha", mu=0, sigma=0.05, shape=n_vars)

        # --- 3. Construct the Expected Mean ---
        mu = alpha
        for i in range(n_lags):
            mu = mu + pt.dot(y_data[n_lags - (i + 1) : -(i + 1), :], beta[i])

        # --- 4. Likelihood ---
        target_data = y_data[n_lags:]
        # nu = pm.Gamma("nu", alpha=7, beta=0.05)
        obs = pm.MvNormal("obs", mu=mu, chol=chol, observed=target_data)  # noqa: F841

        # --- 5. Sampling ---
        trace = pm.sample(N_SAMPLES, return_inferencedata=True, target_accept=0.95)

    return trace, y_data, bvar_model


def forecast_bvar_model(trace, y_data, n_lags: int = 2, n_forecast_steps: int = 1):
    """
    Forecasts future values using the trained BVAR model trace.
    """
    # 1. Extract Posteriors Robustly
    stacked_posterior = trace.posterior.stack(sample=("chain", "draw"))

    beta_samples = np.moveaxis(stacked_posterior["beta"].values, -1, 0)
    alpha_samples = np.moveaxis(stacked_posterior["alpha"].values, -1, 0)
    chol_samples = np.moveaxis(stacked_posterior["noise_chol"].values, -1, 0)

    n_samples = beta_samples.shape[0]
    n_vars = y_data.shape[1]

    # 2. Initialize Forecast Container
    forecasts = np.zeros((n_samples, n_forecast_steps, n_vars))
    initial_history = y_data[-n_lags:]

    print(
        f"Generating {n_forecast_steps} steps of forecast for {n_samples} posterior samples..."
    )

    # 4. Simulation Loop (Vectorized where possible, loop over samples)
    for i in range(n_samples):
        b = beta_samples[i]
        a = alpha_samples[i]

        L_packed = chol_samples[i]
        L = np.zeros((n_vars, n_vars))
        L[np.tril_indices(n_vars)] = L_packed

        curr_hist = initial_history.copy()

        for t in range(n_forecast_steps):
            # A. Calculate Mean
            mu = a.copy()
            for lag in range(n_lags):
                mu += np.dot(curr_hist[-(lag + 1)], b[lag])

            # B. Add Noise
            print(n_vars)
            white_noise = np.random.normal(0, 1, size=n_vars)
            shock = np.dot(L, white_noise)

            # C. Combine
            y_next = mu + shock
            forecasts[i, t, :] = y_next
            curr_hist = np.vstack([curr_hist, y_next])

    return forecasts
