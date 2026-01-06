import asyncio
import os
import httpx
import pandas as pd
from typing import Dict, Any, List, Optional


class MobulaClient:
    """
    A client for interacting with the Mobula API.
    """

    def __init__(self, base_url: str = "https://api.mobula.io/api"):
        self.base_url = base_url
        self.api_key = os.getenv("MOBULA_API_KEY")
        if not self.api_key:
            raise ValueError("MOBULA_API_KEY environment variable not set")
        self.headers = {"Authorization": self.api_key}
        self.cache_path = "./data/mobula_token_details_cache.parquet"

    def clear_token_details_cache(self):
        """Deletes the token details cache file if it exists."""
        if os.path.exists(self.cache_path):
            os.remove(self.cache_path)
            print(f"Cache cleared at {self.cache_path}")
        else:
            print("No cache file to clear.")

    async def _get(
        self, endpoint: str, params: Optional[Dict[str, Any]] = None
    ) -> httpx.Response:
        """
        Makes a GET request to the specified endpoint.
        """
        url = f"{self.base_url}{endpoint}"
        retries = 3
        async with httpx.AsyncClient() as client:
            while True:
                try:
                    await asyncio.sleep(0.4)
                    response = await client.get(
                        url, headers=self.headers, params=params, timeout=60.0
                    )
                    response.raise_for_status()
                    return response
                except (httpx.RequestError, httpx.HTTPStatusError) as e:
                    retries -= 1
                    await asyncio.sleep(5)
                    if retries > 0:
                        print(f"Error calling Mobula API: {params.get("address")}")
                    else:
                        return e.response

    async def _post(self, endpoint: str, body: List[Dict[str, Any]]) -> httpx.Response:
        """
        Makes a POST request to the specified endpoint.
        """
        url = f"{self.base_url}{endpoint}"
        retries = 3
        async with httpx.AsyncClient() as client:
            while True:
                try:
                    await asyncio.sleep(0.4)
                    response = await client.post(
                        url, headers=self.headers, json=body, timeout=60.0
                    )
                    response.raise_for_status()
                    return response
                except (httpx.RequestError, httpx.HTTPStatusError) as e:
                    retries -= 1
                    await asyncio.sleep(5)
                    if retries > 0:
                        print(f"Error calling Mobula API POST endpoint: {endpoint}")
                    else:
                        return e.response

    async def get_ohlcv_history(
        self,
        requests: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Fetches OHLCV history for multiple tokens in batches.
        """
        all_results = []
        batch_size = 10
        for i in range(0, len(requests), batch_size):
            batch = requests[i : i + batch_size]
            print(f"Fetching OHLCV data for batch {i//batch_size + 1}...")
            endpoint = "/2/token/ohlcv-history"
            response = await self._post(endpoint, body=batch)
            if response.is_success:
                all_results.extend(response.json().get("data", []))
            else:
                print(
                    f"Failed to fetch batch {i//batch_size + 1}. Status: {response.status_code}"
                )
        return all_results

    async def get_token_details(
        self, chain_id: str, token_address: str
    ) -> Dict[str, Any]:
        """
        Fetches token details for a given chain and token address, using a persistent cache.
        """
        # 1. Check cache first
        if os.path.exists(self.cache_path):
            try:
                filters = [
                    ("chain_id", "==", chain_id),
                    ("token_address", "==", token_address),
                ]
                cached_df = pd.read_parquet(self.cache_path, filters=filters)
                if not cached_df.empty:
                    # Convert DataFrame row back to the expected JSON structure
                    return cached_df.to_dict("records")[0]["response"]
            except Exception as e:
                print(f"Error reading from cache: {e}")

        # 2. If not in cache, fetch from API
        params = {"blockchain": chain_id, "address": token_address}
        endpoint = "/2/token/details"
        response = await self._get(endpoint, params=params)
        response_json = response.json()

        # 3. Save the new result to the cache
        if response.is_success:
            try:
                new_data = {
                    "chain_id": chain_id,
                    "token_address": token_address,
                    "response": response_json,
                }
                new_df = pd.DataFrame([new_data])

                if os.path.exists(self.cache_path):
                    existing_df = pd.read_parquet(self.cache_path)
                    combined_df = pd.concat([existing_df, new_df], ignore_index=True)
                else:
                    combined_df = new_df

                combined_df.to_parquet(self.cache_path, index=False)
            except Exception as e:
                print(f"Error writing to cache: {e}")

        return response_json

    async def get_market_history(
        self, chain_id: str, asset_address: str, from_timestamp: int, to_timestamp: int
    ) -> Dict[str, Any]:
        """
        Fetches market history for a given asset.
        """
        params = {
            "chainId": chain_id,
            "address": asset_address,
            "period": "1d",
            "from": from_timestamp,
            "to": to_timestamp,
        }
        endpoint = "/2/asset/price-history"
        response = await self._get(endpoint, params=params)
        return response.json()

    async def get_swap_quote(
        self, chain_id: str, sell_token: str, buy_token: str, raw_input: int
    ):
        params = {
            "chainId": chain_id,
            "tokenIn": sell_token,
            "tokenOut": buy_token,
            "amountRaw": raw_input,
            "walletAddress": "0x6810e776880c02933d47db1b9fc05908e5386b96",
        }

        endpoint = "/2/swap/quoting"
        response = await self._get(endpoint, params)
        return response.json()

    async def get_holders(
        self,
        chain_id: str,
        token_address: str,
        limit: int = 100,
        force_fresh: bool = False,
    ) -> List[str]:
        params = {
            "blockchain": chain_id,
            "address": token_address,
            "limit": limit,
            "force": force_fresh,
        }

        endpoint = "/2/token/holder-positions"
        response = await self._get(endpoint, params)
        return response.json()
