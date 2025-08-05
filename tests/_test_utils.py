"""Shared test utilities for brain-go-brrr tests."""

import numpy as np
from typing import List, Optional, Any


class FakeAutoReject:
    """Mock AutoReject object for testing without the real dependency."""
    
    def __init__(
        self, 
        thresholds: Optional[np.ndarray] = None,
        consensus: Optional[List[float]] = None,
        n_interpolate: Optional[List[int]] = None,
        picks: Optional[List[int]] = None
    ):
        """Initialize fake AutoReject with parameters.
        
        Args:
            thresholds: Channel thresholds array
            consensus: Consensus values
            n_interpolate: Interpolation parameters
            picks: Channel picks
        """
        self.threshes_ = thresholds if thresholds is not None else np.random.rand(19, 10)
        self.consensus_ = consensus if consensus is not None else [0.1]
        self.n_interpolate_ = n_interpolate if n_interpolate is not None else [1, 4]
        self.picks_ = picks if picks is not None else list(range(19))
    
    @classmethod
    def from_params(cls, params: dict[str, Any]) -> "FakeAutoReject":
        """Create from parameter dictionary.
        
        Args:
            params: Dictionary with keys 'thresholds', 'consensus', etc.
            
        Returns:
            FakeAutoReject instance
        """
        return cls(
            thresholds=params.get("thresholds"),
            consensus=params.get("consensus"),
            n_interpolate=params.get("n_interpolate"),
            picks=params.get("picks")
        )