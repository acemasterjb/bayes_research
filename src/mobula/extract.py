from datetime import datetime, timezone
from typing import Any, Dict, List

from scripts.mobula.client import MobulaClient


def get_token_id_and_creation_timestamp(
    client: MobulaClient, chain_id: str, token_address: str
) -> (str, int):
    """
    Retrieves the token id and creation timestamp for a given token.
    """
    token_details = client.get_token_details(
        chain_id=chain_id, token_address=token_address
    )
    token_id = token_details.get("data", {}).get("id")

    created_at_str = token_details.get("data", {}).get("createdAt")
    if created_at_str:
        # The timestamp is in ISO 8601 format with milliseconds and 'Z' for UTC.
        # Python's fromisoformat can handle this directly if we remove the 'Z'
        # and replace it with '+00:00'. However, a simpler approach is to
        # use strptime and define the format.
        dt_object = datetime.strptime(created_at_str, "%Y-%m-%dT%H:%M:%S.%fZ")
        # Ensure the datetime object is timezone-aware (UTC)
        dt_object = dt_object.replace(tzinfo=timezone.utc)
        created_at_timestamp = int(dt_object.timestamp() * 1000)  # to milliseconds
    else:
        created_at_timestamp = None

    return token_id, created_at_timestamp


def get_price_history_data(
    client: MobulaClient,
    chain_id: str,
    token_id: str,
    from_timestamp: int,
) -> List[Dict[str, Any]]:
    """
    Fetches price history data for a given token.
    """
    to_timestamp = int(datetime.now(timezone.utc).timestamp() * 1000)
    market_history = client.get_market_history(
        chain_id=chain_id,
        asset_id=token_id,
        from_timestamp=from_timestamp,
        to_timestamp=to_timestamp,
    )

    return market_history.get("data", {}).get("price_history", [])


def extract_data(chain_id: str, token_address: str) -> List[Dict[str, Any]]:
    """
    Main extraction function to get all price history data for a token.
    """
    client = MobulaClient()
    token_id, from_timestamp = get_token_id_and_creation_timestamp(
        client, chain_id, token_address
    )

    if not token_id or not from_timestamp:
        print(
            f"Could not retrieve token id or creation timestamp for the given token and chain: ({token_address}, {chain_id})"
        )
        return []

    price_history_data = get_price_history_data(
        client, chain_id, token_id, from_timestamp
    )
    return price_history_data
