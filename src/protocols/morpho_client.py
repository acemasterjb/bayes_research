import asyncio
import httpx
import pandas as pd
from datetime import datetime, timedelta, timezone
from typing import Dict, List


class MorphoClient:
    """
    A client for interacting with the Morpho Blue GraphQL API.
    """

    _chain_id_map = {
        "ethereum": 1,
        # Add other chain mappings as needed
    }

    def __init__(self, graphql_url: str = "https://blue-api.morpho.org/graphql"):
        self.graphql_url = graphql_url

    async def get_market_data(
        self,
        chain_ids: List[int],
        market_ids: List[str],
        start_timestamp: int = None,
        end_timestamp: int = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetches and processes historical supply and borrow data for specified markets.
        """
        # --- Set default timestamps if not provided ---
        if end_timestamp is None:
            # Default to today at midnight UTC
            now_utc = datetime.now(timezone.utc)
            end_timestamp = int(
                now_utc.replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
            )
        if start_timestamp is None:
            # Default to 365 days before the end timestamp
            end_dt = datetime.fromtimestamp(end_timestamp, tz=timezone.utc)
            start_timestamp = int((end_dt - timedelta(days=365)).timestamp())

        query = """
        query Items($first: Int, $skip: Int, $where: MarketFilters, $options: TimeseriesOptions) {
          markets(first: $first, skip: $skip, where: $where) {
            items {
              historicalState {
                borrowAssets(options: $options) { x y }
                supplyAssets(options: $options) { x y }
              }
              loanAsset { symbol }
              collateralAsset { symbol }
            }
            pageInfo {
              count
              countTotal
            }
          }
        }
        """
        all_market_data = {}
        first = 25
        skip = 0
        total_fetched = 0
        total_to_fetch = float("inf")

        async with httpx.AsyncClient(timeout=60.0) as client:
            while total_fetched < total_to_fetch:
                variables = {
                    "first": first,
                    "skip": skip,
                    "where": {"chainId_in": chain_ids, "uniqueKey_in": market_ids},
                    "options": {
                        "startTimestamp": start_timestamp,
                        "interval": "DAY",
                        "endTimestamp": end_timestamp,
                    },
                }
                payload = {"query": query, "variables": variables}

                try:
                    response = await client.post(self.graphql_url, json=payload)
                    response.raise_for_status()
                    data = response.json()

                    markets = data["data"]["markets"]["items"]
                    page_info = data["data"]["markets"]["pageInfo"]
                    total_to_fetch = page_info["countTotal"]

                    for market in markets:
                        market_name = (
                            f"{market['collateralAsset']['symbol']}/"
                            f"{market['loanAsset']['symbol']}"
                        )

                        supply_df = pd.DataFrame(
                            market["historicalState"]["supplyAssets"]
                        )
                        borrow_df = pd.DataFrame(
                            market["historicalState"]["borrowAssets"]
                        )

                        # Rename columns and merge
                        supply_df.rename(
                            columns={"x": "timestamp", "y": "supply_volume"},
                            inplace=True,
                        )
                        borrow_df.rename(
                            columns={"x": "timestamp", "y": "borrow_volume"},
                            inplace=True,
                        )

                        # Convert timestamp to datetime and set as index
                        supply_df["timestamp"] = pd.to_datetime(
                            supply_df["timestamp"], unit="s"
                        )
                        borrow_df["timestamp"] = pd.to_datetime(
                            borrow_df["timestamp"], unit="s"
                        )

                        # Merge into a single DataFrame
                        merged_df = pd.merge(
                            supply_df, borrow_df, on="timestamp", how="outer"
                        ).sort_values("timestamp")
                        merged_df.set_index("timestamp", inplace=True)
                        merged_df.interpolate(method="time", inplace=True)
                        merged_df.dropna(inplace=True)

                        all_market_data[market_name] = merged_df

                    skip += first
                    total_fetched += page_info["count"]

                    if total_fetched >= total_to_fetch:
                        break

                except (httpx.RequestError, httpx.HTTPStatusError) as e:
                    print(f"An error occurred while fetching data: {e}")
                    break
                await asyncio.sleep(0.5)  # Rate limiting

        return all_market_data

    async def get_all_markets(
        self,
        chain_id_str: str,
        start_timestamp: int = None,
        end_timestamp: int = None,
        limit: int = 115,
    ) -> List[Dict]:
        """
        Fetches all whitelisted markets for a given chain with their historical data.
        """
        if end_timestamp is None:
            now_utc = datetime.now(timezone.utc)
            end_timestamp = int(
                now_utc.replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
            )
        if start_timestamp is None:
            end_dt = datetime.fromtimestamp(end_timestamp, tz=timezone.utc)
            start_timestamp = int((end_dt - timedelta(days=365)).timestamp())

        query = """
        query Markets($first: Int, $skip: Int, $where: MarketFilters, $options: TimeseriesOptions, $orderBy: MarketOrderBy) {
        markets(first: $first, skip: $skip, where: $where, orderBy: $orderBy) {
            items {
            uniqueKey
            collateralAsset { symbol address }
            loanAsset { symbol address }
            historicalState {
                borrowAssetsUsd(options: $options) { x y }
                supplyAssetsUsd(options: $options) { x y }
                supplyApy(options: $options) { x y }
                borrowApy(options: $options) { x y }
                netBorrowApy(options: $options) { x y }
                netSupplyApy(options: $options) { x y }
            }
            }
            pageInfo { count countTotal }
        }
        }
        """
        all_items = []
        first = 15
        skip = 0

        numerical_chain_id = self._chain_id_map.get(chain_id_str.lower())
        if numerical_chain_id is None:
            raise ValueError(f"Unknown chain_id: {chain_id_str}")

        async with httpx.AsyncClient(timeout=60.0) as client:
            while True:
                variables = {
                    "first": first,
                    "skip": skip,
                    "where": {
                        "chainId_in": [numerical_chain_id],
                        "whitelisted": True,
                    },
                    "options": {
                        "startTimestamp": start_timestamp,
                        "interval": "DAY",
                    },
                    "orderBy": "BorrowAssetsUsd",
                }
                payload = {"query": query, "variables": variables}

                try:
                    response = await client.post(self.graphql_url, json=payload)
                    response.raise_for_status()
                    data = response.json()

                    items = data["data"]["markets"]["items"]
                    all_items.extend(items)

                    skip += first
                    if not items or len(all_items) >= limit:
                        break

                except (httpx.RequestError, httpx.HTTPStatusError) as e:
                    print(f"An error occurred while fetching markets: {e}")
                    break
                await asyncio.sleep(0.5)

        return all_items[:limit]
