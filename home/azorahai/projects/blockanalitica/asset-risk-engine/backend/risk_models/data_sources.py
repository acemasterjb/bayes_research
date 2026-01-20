# Abstract Base Class for price data sources
from abc import ABC, abstractmethod
import numpy as np

class PriceSource(ABC):
    @abstractmethod
    def get_price_history(self, asset_address: str, chain_id: str) -> np.ndarray:
        """
        Fetches historical price data for a given asset and chain.
        Returns a numpy array of prices.
        """
        pass
