import argparse
import asyncio
from datetime import datetime, timedelta, timezone
import pandas as pd
import numpy as np
import os
import cloudpickle as pickle

from src.protocols.morpho_client import MorphoClient
from src.mobula.client import MobulaClient
from src.features.build_market_features import build_features_for_markets
from src.motors.supply_determinants_model import run_supply_determinants_model
from src.plotting.determinants_plots import (
    plot_priors_vs_posteriors_determinants,
    plot_posterior_predictive_check_determinants,
)

async def main():
    """
    Main entrypoint for the supply determinants analysis script.
    """
    parser = argparse.ArgumentParser(description="Find determinants of supply in lending markets.")
    parser.add_argument(
        "--chain_id",
        type=str,
        required=True,
        help="Chain to analyze (e.g., 'ethereum').",
    )
    args = parser.parse_args()

    # --- Caching Setup ---
    cache_dir = "./data"
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"enriched_markets_{args.chain_id}.pkl")
    
    enriched_markets = None
    
    # Define time range for data fetching
    now_utc = datetime.now(timezone.utc)
    current_midnight_ts = int(now_utc.replace(hour=0, minute=0, second=0, microsecond=0).timestamp())
    from_ts = int((datetime.fromtimestamp(current_midnight_ts) - timedelta(days=365)).timestamp())

    # --- 1. Check Cache ---
    if os.path.exists(cache_path):
        last_mod_time = os.path.getmtime(cache_path)
        # Invalidate if older than 2 days (172800 seconds)
        if (current_midnight_ts - last_mod_time) < 172800:
            print("Loading enriched market data from cache...")
            with open(cache_path, "rb") as f:
                enriched_markets = pickle.load(f)
        else:
            print("Cache is older than two days, refetching data...")

    # --- 2. If Cache Miss, Fetch and Build Features ---
    if enriched_markets is None:
        print(f"Fetching all whitelisted markets for {args.chain_id}...")
        morpho_client = MorphoClient()
        all_markets = await morpho_client.get_all_markets(args.chain_id, start_timestamp=from_ts)

        if not all_markets:
            print("No markets found. Exiting.")
            return

        print("Fetching OHLCV data for all loan assets...")
        mobula_client = MobulaClient()
        loan_asset_addresses = list(set([m["loanAsset"]["address"] for m in all_markets]))
        
        ohlcv_requests = [
            {"address": addr, "chainId": args.chain_id, "period": "1d", "from": from_ts * 1000, "to": current_midnight_ts * 1000}
            for addr in loan_asset_addresses
        ]
        ohlcv_data = await mobula_client.get_ohlcv_history(ohlcv_requests)

        print("Enriching markets with features...")
        enriched_markets = await build_features_for_markets(all_markets, ohlcv_data, args.chain_id, from_ts, current_midnight_ts)
        
        # Save to cache
        with open(cache_path, "wb") as f:
            pickle.dump(enriched_markets, f)

    if not enriched_markets:
        print("No enriched market data to process. Exiting.")
        return


    # --- 3. Aggregate Data and Run Model ---
    print("\\n--- Aggregating data from all markets for modeling ---")
    all_X_dfs = []
    all_y_series = []
    all_cat_indices = []

    features = [
        "mean_returns_14d",
        "std_returns_14d",
        "log_volume",
        "isIncentivisedSupply",
        "isIncentivisedBorrow",
    ]

    for market in enriched_markets:
        df = market["features_df"].copy()
        
        # Prepare data for this market
        df["delta_supply_usd"] = df["supplyAssetsUsd"].diff().fillna(0)
        df["log_volume"] = np.log(df["volume"] + 1)
        df["isIncentivisedSupply"] = df["isIncentivisedSupply"].astype(int)
        df["isIncentivisedBorrow"] = df["isIncentivisedBorrow"].astype(int)
        
        df.dropna(subset=["delta_supply_usd"] + features, inplace=True)
        
        if df.empty:
            continue

        # Append this market's data to the aggregate lists
        all_X_dfs.append(df[features])
        all_y_series.append(df["delta_supply_usd"])
        
        # Create an array for the categorical index, one entry per row
        cat_index_for_market = np.full(len(df), market["classification"])
        all_cat_indices.append(cat_index_for_market)

    if not all_X_dfs:
        print("No valid data available for modeling after cleaning. Exiting.")
        return

    # Concatenate all data into single structures
    X_aggregate = pd.concat(all_X_dfs, ignore_index=True)
    y_aggregate = pd.concat(all_y_series, ignore_index=True)
    category_indices = np.concatenate(all_cat_indices)

    # --- 5. Run the Bayesian Model Once ---
    print("\\n--- Running Aggregate Bayesian Model ---")
    try:
        trace, model = run_supply_determinants_model(
            X_aggregate, y_aggregate, category_indices
        )

        # --- 6. Generate Diagnostic Plots ---
        print("\\n--- Generating Diagnostic Plots for Aggregate Model ---")
        plot_priors_vs_posteriors_determinants(trace)
        plot_posterior_predictive_check_determinants(model, trace)

    except Exception as e:
        print(f"Failed to run the aggregate model. Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())

