"""
Configuration settings for HiPoSa.

This module contains default parameters and configuration options
for the HiPoSa library components.
"""

from typing import Dict, Any
from dataclasses import dataclass

@dataclass
class SamplingConfig:
    """Configuration for Poisson disk sampling."""
    
    # Default sampling parameters
    DEFAULT_K: int = 60
    DEFAULT_K_SECOND_PASS: int = 80
    DEFAULT_EPSILON: float = 1e-10
    
    # Update frequency for KDTree
    KDTree_UPDATE_FREQUENCY: int = 10
    
    # Optimization parameters
    MINIMIZATION_XATOL: float = 1e-10
    MINIMIZATION_FATOL: float = 1e-10
    INVARIANT_POINT_ATTEMPTS: int = 10

@dataclass
class TilingConfig:
    """Configuration for Poisson tiling."""
    
    # Minimum tile size factor
    MIN_TILE_FACTOR: float = 2.0
    
    # Default number of processes for parallel processing
    DEFAULT_N_PROCESSES: int = None
    
    # Whether to add corner points by default
    DEFAULT_ADD_CORNERS: bool = True

@dataclass
class PointSelectorConfig:
    """Configuration for point selection."""
    
    # Default threshold parameters
    DEFAULT_TAU: float = 75.0
    DEFAULT_SIGN: int = 1
    DEFAULT_EPS: float = 0.0
    DEFAULT_START_LEVEL: int = 0
    DEFAULT_SET_ASIDE: float = 0.5
    
    # Quantile calculation parameters
    DEFAULT_LOWER_TAU_QUANTILE: float = 1.0
    DEFAULT_N_SPLITS_QUANTILE: int = 10
    
    # Border parameters
    DEFAULT_BORDER: int = 4
    DEFAULT_BORDER_BIAS: float = 0.5
    DEFAULT_RADIUS_SCALE: float = 4.0
    
    # Cross-validation parameters
    DEFAULT_N_SPLITS: int = 5

# Global configuration instances
SAMPLING_CONFIG = SamplingConfig()
TILING_CONFIG = TilingConfig()
POINT_SELECTOR_CONFIG = PointSelectorConfig()

def get_default_config() -> Dict[str, Any]:
    """Get all default configuration parameters."""
    return {
        'sampling': SAMPLING_CONFIG,
        'tiling': TILING_CONFIG,
        'point_selector': POINT_SELECTOR_CONFIG
    } 