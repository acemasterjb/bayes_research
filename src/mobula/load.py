import os
import pandas as pd
from datetime import datetime, timezone
from scripts.mobula.extract import extract_data
from scripts.mobula.transform import transform_price_history_data

CACHE_DIR = "data"

chain_map = {
    "eth": "ethereum",
    "arb": "arbitrum",
}


def get_cache_filepath(chain_id: str, token_address: str, date_str: str) -> str:
    """
    Constructs the filepath for a cached data file.
    """
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)
    return os.path.join(CACHE_DIR, f"mobula_{chain_id}_{token_address}_{date_str}.csv")


def load_from_cache(filepath: str) -> pd.DataFrame:
    """
    Loads data from a cache file if it exists.
    """
    if os.path.exists(filepath):
        df = pd.read_csv(filepath)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df
    return None


def save_to_cache(filepath: str, df: pd.DataFrame):
    """
    Saves a DataFrame to a cache file.
    """
    df.to_csv(filepath, index=False)


def get_asset_price_data(chain_id: str, token_address: str) -> pd.DataFrame:
    """
    Main function to get asset price data, using cache if available.
    """
    chain_id = chain_map.get(chain_id, chain_id)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    cache_filepath = get_cache_filepath(chain_id, token_address, today)

    # Try to load from cache first
    cached_data = load_from_cache(cache_filepath)
    if cached_data is not None:
        print("Loading data from cache.")
        return cached_data

    # If not in cache, extract, transform, and then cache the data
    print("Fetching new data from the API.")
    raw_data = extract_data(chain_id, token_address)

    if not raw_data:
        print(
            f"No data found for the given token and chain: ({token_address}, {chain_id})"
        )
        return pd.DataFrame()

    transformed_data = transform_price_history_data(raw_data)

    if not transformed_data.empty:
        save_to_cache(cache_filepath, transformed_data)
        print(f"Data saved to cache at {cache_filepath}")

    return transformed_data


if __name__ == "__main__":
    # Example usage:
    # This is a placeholder, please replace with actual values for testing
    chain_id_id_example = "solana"
    token_address_example = "3vz82EWYv8xnc7Cm7qSgERcpMeqw92PcX8PBz88npump"

    df = get_asset_price_data(chain_id_id_example, token_address_example)
    if not df.empty:
        print("Successfully retrieved asset price data:")
        print(df.head())
    else:
        print("Could not retrieve asset price data.")
