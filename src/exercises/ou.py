import argparse
import asyncio
import os
from datetime import datetime, timedelta
from src.motors.garch_model import run_garch_model, forecast_garch_model
from src.plotting.garch_plots import (
    plot_garch_posteriors,
    plot_garch_volatility,
    plot_garch_forecast,
    plot_garch_in_sample_fit,
    plot_garch_ppc,
    plot_garch_simple_residuals,
    plot_garch_standardized_residuals_and_acf,
)


import cloudpickle
import numpy as np
import pandas as pd

from src.mobula.client import MobulaClient
from src.motors.ou_model import (
    estimate_ar1_params,
    get_jump_prior_stats,
    generate_ar_paths,
)
from src.plotting.ou_plots import (
    plot_posterior,
    plot_ar1_in_sample_fit,
    plot_priors_vs_posteriors,
    plot_posterior_predictive_check,
    plot_residuals,
)

from src.validation.stationarity_tests import run_stationarity_analysis


async def run(
    asset_address: str,
    chain_id: str,
    percentile: float = 0.5,
    current_price: float = 1.0,
    n_paths: int = 10,
    plot_jumps: bool = False,
    run_stationarity_tests: bool = False,
    test_returns: bool = False,
    model_type: str = "ar1",
    forecast_days: int = 30,
    trim_lower: float = 0.1,
    trim_upper: float = 0.9,
):
    """
    Main function to run the Ornstein-Uhlenbeck process analysis.
    """
    # --- Caching Setup ---
    today = datetime.now().strftime("%Y-%m-%d")
    cache_dir = "./data"
    os.makedirs(cache_dir, exist_ok=True)

    prices_cache_path = os.path.join(
        cache_dir, f"prices_{chain_id}_{asset_address}_{today}.npy"
    )
    results_cache_path = os.path.join(
        cache_dir, f"model_results_{chain_id}_{asset_address}_{today}.pkl"
    )

    # --- Step 1: Load or Fetch Prices ---
    if os.path.exists(prices_cache_path):
        print("Loading prices from cache...")
        prices = np.load(prices_cache_path)
    else:
        print("Fetching prices from Mobula API...")
        mobula_client = MobulaClient()
        to_timestamp = int(datetime.now().timestamp() * 1000)
        from_timestamp = int((datetime.now() - timedelta(days=334)).timestamp() * 1000)
        market_history = await mobula_client.get_market_history(
            asset_address=asset_address,
            chain_id=chain_id,
            from_timestamp=from_timestamp,
            to_timestamp=to_timestamp,
        )
        prices = np.array([item[1] for item in market_history["data"]["priceHistory"]])

        # Handle NaN values and save to cache
        prices_series = pd.Series(prices)
        prices_series.interpolate(method="linear", inplace=True)
        prices_series.bfill(inplace=True)
        prices_series.ffill(inplace=True)
        prices = prices_series.to_numpy()
        np.save(prices_cache_path, prices)

    if np.isnan(prices).all():
        raise ValueError("All price data is NaN after cleaning. Cannot proceed.")

    if run_stationarity_tests:
        run_stationarity_analysis(prices, use_returns=test_returns)
        return

    # --- Step 2: Model Selection and Execution ---
    if model_type.lower() == "garch":
        print("\\n--- Running GARCH(1,1)-T Model ---")

        garch_results_cache_path = os.path.join(
            cache_dir,
            f"garch_model_results_{chain_id}_{asset_address}_{today}.pkl",
        )
        if os.path.exists(garch_results_cache_path):
            print("Loading GARCH model results from cache...")
            with open(garch_results_cache_path, "rb") as f:
                cached_results = cloudpickle.load(f)
            trace = cached_results["trace"]
            model = cached_results["model"]
        else:
            print("Running MCMC simulation for GARCH model...")
            log_returns = np.diff(np.log(prices))
            trace, model = run_garch_model(log_returns)
            with open(garch_results_cache_path, "wb") as f:
                cloudpickle.dump({"trace": trace, "model": model}, f)

        log_returns = np.diff(np.log(prices))
        p_est, mu_est, sigma_est = get_jump_prior_stats(prices)

        print("Visualizing GARCH results...")
        # plot_garch_posteriors(trace)
        # plot_garch_volatility(trace, log_returns)
        plot_garch_in_sample_fit(
            trace,
            prices,
            p_est,
            mu_est,
            sigma_est,
            percentile=percentile,
            trim_percentile_lower=trim_lower,
            trim_percentile_upper=trim_upper,
        )
        plot_garch_ppc(model, trace)
        plot_garch_simple_residuals(trace, log_returns)
        plot_garch_standardized_residuals_and_acf(trace, log_returns)

        # Get optimal parameters
        omega_opt = np.quantile(trace.posterior["omega"], percentile)
        alpha_opt = np.quantile(trace.posterior["alpha"], percentile)
        beta_opt = np.quantile(trace.posterior["beta"], percentile)

        print("Optimal OU Process Parameters:")
        print(f"    Omega: {omega_opt}")
        print(f"    Alpha: {alpha_opt}")
        print(f"    Beta: {beta_opt}")
        print("Estimated Jump Params:")
        print(f"    p_jump: {p_est}, mu: {mu_est}, sigma: {sigma_est}")

        # --- GARCH Forecasting ---
        if forecast_days > 0:
            print(f"\\n--- Generating GARCH Forecast for {forecast_days} days ---")
            # We need the last return and the last inferred volatility to start the forecast
            posterior = trace.posterior.stack(sample=("chain", "draw"))
            mu_mean = posterior["mu"].mean().item()
            omega_mean = posterior["omega"].mean().item()
            alpha_mean = posterior["alpha"].mean().item()
            beta_mean = posterior["beta"].mean().item()
            initial_vol_sq_mean = posterior["initial_vol_sq"].mean().item()

            # Reconstruct the last volatility
            vol_sq_series = np.zeros(len(log_returns))
            vol_sq_series[0] = initial_vol_sq_mean
            for t in range(1, len(log_returns)):
                err_tm1 = log_returns[t - 1] - mu_mean
                vol_sq_series[t] = (
                    omega_mean
                    + alpha_mean * err_tm1**2
                    + beta_mean * vol_sq_series[t - 1]
                )

            last_vol_sq = vol_sq_series[-1]
            last_return = log_returns[-1]

            forecasted_returns = forecast_garch_model(
                trace, last_return, last_vol_sq, n_forecast_steps=forecast_days
            )
            plot_garch_forecast(prices, forecasted_returns)

        return

    # --- Step 3: AR(1) Model Execution ---
    print("\\n--- Running AR(1) Model ---")
    if os.path.exists(results_cache_path):
        print("Loading AR(1) model results from cache...")
        with open(results_cache_path, "rb") as f:
            cached_results = cloudpickle.load(f)
        trace = cached_results["trace"]
        model = cached_results["model"]
        m_mu = cached_results["m_mu"]
        m_sigma = cached_results["m_sigma"]
    else:
        print("Running MCMC simulation for AR(1) model...")
        N_SAMPLES = int(1e4)
        p_est, mu_est, sigma_est = 0, 0, 0
        trace, model, m_mu, m_sigma = estimate_ar1_params(
            np.log(prices), p_est, mu_est, sigma_est, N_SAMPLES, False
        )
        with open(results_cache_path, "wb") as f:
            cloudpickle.dump(
                {"trace": trace, "model": model, "m_mu": m_mu, "m_sigma": m_sigma}, f
            )

    # --- Step 4: Analysis and Plotting for AR(1) ---
    print("Visualizing AR(1) results...")
    p_est, mu_est, sigma_est = get_jump_prior_stats(prices)

    # Visualize the posteriors
    # plot_posterior(trace)
    # plot_priors_vs_posteriors(trace, m_mu, m_sigma)
    plot_posterior_predictive_check(model, trace)
    plot_residuals(trace, np.log(prices))

    # --- Step 5: Generate and Plot In-Sample Fit ---
    print("Generating AR(1) in-sample fit plot...")
    plot_ar1_in_sample_fit(
        trace,
        prices,
        p_jump=p_est,
        mu_jump=abs(mu_est),
        sigma_jump=sigma_est,
        n_paths=100,
        percentile=percentile,
        trim_percentile_lower=trim_lower,
        trim_percentile_upper=trim_upper,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Ornstein-Uhlenbeck process analysis."
    )
    parser.add_argument(
        "--asset_address", type=str, required=True, help="Asset address"
    )
    parser.add_argument("--chain_id", type=str, required=True, help="Chain ID")
    parser.add_argument(
        "--percentile",
        type=float,
        default=0.5,
        help="Percentile for optimal parameters",
    )
    parser.add_argument(
        "--current_price", type=float, default=1.0, help="Current price for simulation"
    )
    parser.add_argument(
        "--n_paths", type=int, default=10, help="Number of simulated paths"
    )
    parser.add_argument(
        "--plot_jumps",
        action="store_true",
        help="Whether or not to plot path with jumps",
    )
    parser.add_argument(
        "--run-stationarity-tests",
        action="store_true",
        help="Run stationarity tests on the price series and exit.",
    )
    parser.add_argument(
        "--test-returns",
        action="store_true",
        help="If running stationarity tests, use log-returns instead of log-prices.",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="ar1",
        choices=["ar1", "garch"],
        help="Specify the model type to run ('ar1' or 'garch').",
    )
    parser.add_argument(
        "--forecast-days",
        type=int,
        default=30,
        help="Number of days to forecast into the future for the GARCH model.",
    )
    parser.add_argument(
        "--trim-lower",
        type=float,
        default=0.1,
        help="Lower percentile for trimmed mean calculation in in-sample plot.",
    )
    parser.add_argument(
        "--trim-upper",
        type=float,
        default=0.9,
        help="Upper percentile for trimmed mean calculation in in-sample plot.",
    )

    args = parser.parse_args()
    asyncio.run(
        run(
            asset_address=args.asset_address,
            chain_id=args.chain_id,
            percentile=args.percentile,
            current_price=args.current_price,
            n_paths=args.n_paths,
            plot_jumps=args.plot_jumps,
            run_stationarity_tests=args.run_stationarity_tests,
            test_returns=args.test_returns,
            model_type=args.model_type,
            forecast_days=args.forecast_days,
            trim_lower=args.trim_lower,
            trim_upper=args.trim_upper,
        )
    )
