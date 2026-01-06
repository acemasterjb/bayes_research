from typing import List, Any
import pandas as pd


def transform_price_history_data(
    price_history: List[List[Any]],
) -> pd.DataFrame:
    """
    Transforms a raw price history list into a structured Pandas DataFrame.
    """
    if not price_history:
        return pd.DataFrame()

    columns = ["timestamp", "price"]
    df = pd.DataFrame(price_history, columns=columns)

    # Convert timestamp to datetime objects (it's in milliseconds)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

    df.rename(columns={"price": "median_close"}, inplace=True)

    # Sort by timestamp
    df.sort_values(by="timestamp", inplace=True)

    # Reset index
    df.reset_index(drop=True, inplace=True)

    return df
