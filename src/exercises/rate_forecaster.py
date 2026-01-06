import argparse
import asyncio
import arviz as az
import numpy as np
import pandas as pd
import os
import cloudpickle as pickle
from matplotlib import pyplot as plt

from src.protocols.morpho_client import MorphoClient
from src.motors.bvar_model import train_bvar_model, forecast_bvar_model, run_lfo_cv
from src.protocols.rate_calculators import calculate_morpho_rate_stateless
from src.plotting.bvar_plots import (
    plot_priors_vs_posteriors_bvar,
    plot_posterior_predictive_check_bvar,
    plot_residuals_bvar,
    plot_lfo_cv_results,
    plot_holdout_forecast,
)
from src.validation.holdout import run_fixed_holdout



async def main():
    """
    Main entrypoint for the rate forecasting script.
    """
    parser = argparse.ArgumentParser(
        description="Forecast borrow rates for DeFi lending markets."
    )
    parser.add_argument(
        "--chain_ids",
        type=int,
        nargs="+",
        required=True,
        help="List of chain IDs to query.",
    )
    parser.add_argument(
        "--market_ids",
        type=str,
        nargs="+",
        required=True,
        help="List of market IDs to query.",
    )
    parser.add_argument(
        "--start_timestamp",
        type=int,
        default=None,
        help="Optional start timestamp (Unix seconds) for historical data.",
    )
    parser.add_argument(
        "--end_timestamp",
        type=int,
        default=None,
        help="Optional end timestamp (Unix seconds) for historical data.",
    )
    parser.add_argument(
        "--run-cv",
        action="store_true",
        help="Run computationally intensive Leave-Future-Out Cross-Validation.",
    )
    parser.add_argument(
        "--run-holdout",
        action="store_true",
        help="Run fixed holdout validation against the last 90 days of data.",
    )
    args = parser.parse_args()

    # --- Caching Setup ---
    today = pd.Timestamp.now().strftime("%Y-%m-%d")
    cache_dir = "./data"
    os.makedirs(cache_dir, exist_ok=True)

    market_hash = abs(hash(tuple(sorted(args.market_ids))))
    data_cache_path = os.path.join(cache_dir, f"morpho_data_{market_hash}_{today}.pkl")

    # --- 1. Load or Fetch Data ---
    if os.path.exists(data_cache_path):
        print("Loading market data from cache...")
        with open(data_cache_path, "rb") as f:
            all_market_data = pickle.load(f)
    else:
        print("Fetching market data from Morpho API...")
        morpho_client = MorphoClient()
        all_market_data = await morpho_client.get_market_data(
            args.chain_ids,
            args.market_ids,
            start_timestamp=args.start_timestamp,
            end_timestamp=args.end_timestamp,
        )
        with open(data_cache_path, "wb") as f:
            pickle.dump(all_market_data, f)

    if not all_market_data:
        print("No data returned. Exiting.")
        return

    # --- 2. Run Model and Analyze for Each Market ---
    for market_name, market_df in all_market_data.items():
        print(f"\\n--- Analyzing Market: {market_name} ---")

        if market_df.empty:
            print(f"Skipping {market_name} due to empty data.")
            continue

        # --- Data Cleaning ---
        # Replace inf with NaN, then forward-fill and back-fill
        market_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        market_df.ffill(inplace=True)
        market_df.bfill(inplace=True)
        market_df.dropna(inplace=True) # Drop any remaining NaNs if all values were NaN initially

        if market_df.empty:
            print(f"Skipping {market_name} after cleaning resulted in empty data.")
            continue


        n_lags = 2
        model_cache_path = os.path.join(
            cache_dir, f"bvar_model_{market_name.replace('/', '_')}_{today}.pkl"
        )

        if os.path.exists(model_cache_path):
            print("Loading trained model from cache...")
            with open(model_cache_path, "rb") as f:
                cached_model = pickle.load(f)
            trace = cached_model["trace"]
            y_data = cached_model["y_data"]
            bvar_model = cached_model["model"]
        else:
            print("Training Bayesian Vector Autoregression model...")
            trace, y_data, bvar_model = train_bvar_model(market_df, n_lags=n_lags)
            with open(model_cache_path, "wb") as f:
                pickle.dump({"trace": trace, "y_data": y_data, "model": bvar_model}, f)

        # --- b. Diagnostic Plots ---
        print("Generating diagnostic plots...")
        plot_priors_vs_posteriors_bvar(trace, bvar_model)
        plot_posterior_predictive_check_bvar(bvar_model, trace)
        plot_residuals_bvar(trace, y_data, n_lags=n_lags)

        # --- c. Optional: Run and Plot Cross-Validation ---
        if args.run_cv:
            print("\\n--- Running Leave-Future-Out Cross-Validation ---")
            log_pds, pareto_ks = run_lfo_cv(y_data, n_lags=n_lags)
            plot_lfo_cv_results(log_pds, pareto_ks)

        # --- d. Optional: Run Fixed Holdout Validation ---
        if args.run_holdout:
            test_data, median_forecast, forecast_samples_holdout = run_fixed_holdout(
                y_data, n_lags=n_lags
            )
            plot_holdout_forecast(
                test_data, median_forecast, forecast_samples_holdout
            )

        # --- e. Final Forecast for Future Rate ---
        print("\\n--- Generating Final 1-Day Ahead Rate Forecast ---")
        forecast_samples = forecast_bvar_model(
            trace, y_data, n_lags=n_lags, n_forecast_steps=1
        )
        log_borrow_samples = forecast_samples[:, 0, 0]
        log_supply_samples = forecast_samples[:, 0, 1]
        borrow_vol = np.exp(log_borrow_samples)
        supply_vol = np.exp(log_supply_samples)
        utilization_dist = borrow_vol / supply_vol

        current_time = (market_df.index[-1] + pd.Timedelta(days=1)).timestamp()
        last_update_time = market_df.index[-1].timestamp()

        rate_dist = calculate_morpho_rate_stateless(
            utilization_dist,
            current_time=current_time,
            last_update_time=last_update_time,
        )

        print("Plotting final forecasted interest rate distribution...")
        az.plot_posterior(
            {"rate": rate_dist},
            var_names=["rate"],
            hdi_prob=0.9,
            point_estimate="mean",
        )
        plt.show()



if __name__ == "__main__":
    asyncio.run(main())
