import pandas as pd
import numpy as np
from typing import Dict, List
from src.mobula.client import MobulaClient

MODEL_ASSETS = {
    "usd": {"ethereum": "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48"},
    "eth": {"ethereum": "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2"},
    "btc": {"ethereum": "0x2260fac5e5542a773aa44fbcfedf7c193bc2c599"},
}

CLASSIFICATION_MAP = {"usd-like": 0, "blue-chip": 1, "alts": 2, "pendle-like": 3}


def _calculate_returns_and_stats(df):
    df["returns"] = df["close"].pct_change().fillna(0)
    df["mean_returns_14d"] = df["returns"].rolling(window=14).mean().fillna(0)
    df["std_returns_14d"] = df["returns"].rolling(window=14).std().fillna(0)
    # Backfill gaps after the initial 14-day period
    if len(df) > 14:
        df["mean_returns_14d"].iloc[14:] = df["mean_returns_14d"].iloc[14:].bfill()
        df["std_returns_14d"].iloc[14:] = df["std_returns_14d"].iloc[14:].bfill()
    return df


async def _get_model_asset_stats(chain_id, from_timestamp, to_timestamp):
    mobula_client = MobulaClient()
    stats = {}
    for name, addresses in MODEL_ASSETS.items():
        if chain_id in addresses:
            history = await mobula_client.get_market_history(
                chain_id,
                addresses[chain_id],
                from_timestamp * 1000,
                to_timestamp * 1000,
            )
            prices = [
                item[1] for item in history.get("data", {}).get("priceHistory", [])
            ]
            if prices:
                returns = pd.Series(prices).pct_change().dropna()
                stats[name] = {"mean": returns.mean(), "std": returns.std()}
    return stats


def _classify_asset(loan_symbol, returns_mean, returns_std, model_stats):
    if loan_symbol.startswith("pt-"):
        return CLASSIFICATION_MAP["pendle-like"]

    if not model_stats:
        raise Exception
        return CLASSIFICATION_MAP["alts"]  # Default if no models available

    distances = {}
    loan_vec = np.array([returns_mean, returns_std])
    for name, stats in model_stats.items():
        model_vec = np.array([stats["mean"], stats["std"]])
        distances[name] = np.linalg.norm(loan_vec - model_vec)

    closest_model = min(distances, key=distances.get)
    if closest_model == "usd":
        return CLASSIFICATION_MAP["usd-like"]
    elif closest_model in ["eth", "btc"]:
        return CLASSIFICATION_MAP["blue-chip"]
    else:
        return CLASSIFICATION_MAP["alts"]


async def build_features_for_markets(
    all_market_data: List[Dict],
    ohlcv_data: List[Dict],
    chain_id: str,
    from_ts: int,
    to_ts: int,
) -> List[Dict]:
    """
    Main function to enrich market data with calculated features.
    """
    enriched_markets = []

    # --- Prepare lookup for OHLCV data ---
    ohlcv_lookup = {}
    for item in ohlcv_data:
        address = item.get("address")
        if address:
            df = pd.DataFrame(item.get("ohlcv", []))
            if not df.empty:
                df["timestamp"] = pd.to_datetime(df["t"], unit="ms").dt.floor("D")
                df.rename(columns={"c": "close", "v": "volume"}, inplace=True)
                ohlcv_lookup[address.lower()] = df[["timestamp", "close", "volume"]]

    # --- Get stats for model assets ---
    model_stats = await _get_model_asset_stats(chain_id, from_ts, to_ts)

    # --- Process each market ---
    for market in all_market_data:
        market_name = (
            f"{market['collateralAsset']['symbol']}/{market['loanAsset']['symbol']}"
        )
        print(f"Building features for market: {market_name}")
        loan_address = market["loanAsset"]["address"].lower()

        if loan_address not in ohlcv_lookup:
            print("  - Skipping, no OHLCV data found for loan asset.")
            continue

        # Convert historical data to DataFrames
        df_dict = {}
        for key, value in market["historicalState"].items():
            df = pd.DataFrame(value)
            df["x"] = pd.to_datetime(df["x"], unit="s").dt.floor("D")
            df.rename(columns={"x": "timestamp", "y": key}, inplace=True)
            df_dict[key] = df

        # Merge all time series data
        base_df = df_dict.pop(next(iter(df_dict)))
        for key, df in df_dict.items():
            base_df = pd.merge(base_df, df, on="timestamp", how="outer")

        # Merge with OHLCV data
        enriched_df = pd.merge(
            base_df, ohlcv_lookup[loan_address], on="timestamp", how="left"
        )
        enriched_df.sort_values("timestamp", inplace=True)
        enriched_df.set_index("timestamp", inplace=True)

        # 3. Calculate returns and rolling stats
        enriched_df = _calculate_returns_and_stats(enriched_df)

        # 4. Flag incentives
        supply_diff = enriched_df["netSupplyApy"] - enriched_df["supplyApy"]
        borrow_diff = enriched_df["netBorrowApy"] - enriched_df["borrowApy"]
        enriched_df["isIncentivisedSupply"] = supply_diff >= 0.0001
        enriched_df["isIncentivisedBorrow"] = borrow_diff >= 0.0001

        market["features_df"] = enriched_df.reset_index()

        # 5. Classify asset
        loan_returns = enriched_df["returns"].dropna()
        market["classification"] = _classify_asset(
            market["loanAsset"]["symbol"],
            loan_returns.mean(),
            loan_returns.std(),
            model_stats,
        )

        enriched_markets.append(market)

    return enriched_markets
