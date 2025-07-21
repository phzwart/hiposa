"""
Utility functions for HiPoSa.

This module contains common utility functions used throughout the HiPoSa library.
"""

import logging
import warnings
from typing import Optional, Union, List, Tuple, Callable
import numpy as np

# Configure logging
logger = logging.getLogger(__name__)

def setup_logging(level: int = logging.INFO) -> None:
    """Setup logging configuration for HiPoSa.
    
    Args:
        level: Logging level to use.
    """
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def validate_domain(domain: List[Tuple[float, float]]) -> None:
    """Validate domain boundaries.
    
    Args:
        domain: List of (min, max) tuples for each dimension.
        
    Raises:
        ValueError: If domain is invalid.
    """
    if not domain:
        raise ValueError("Domain cannot be empty")
    
    for i, (min_val, max_val) in enumerate(domain):
        if min_val >= max_val:
            raise ValueError(f"Domain dimension {i}: min ({min_val}) must be less than max ({max_val})")
        if not np.isfinite(min_val) or not np.isfinite(max_val):
            raise ValueError(f"Domain dimension {i}: bounds must be finite")

def validate_points(points: np.ndarray, dimensions: int) -> None:
    """Validate point array.
    
    Args:
        points: Array of points to validate.
        dimensions: Expected number of dimensions.
        
    Raises:
        ValueError: If points are invalid.
    """
    if points.size == 0:
        return
    
    if points.ndim != 2:
        raise ValueError(f"Points must be 2D array, got {points.ndim}D")
    
    if points.shape[1] != dimensions:
        raise ValueError(f"Points must have {dimensions} dimensions, got {points.shape[1]}")
    
    # Check for NaN values
    if np.any(np.isnan(points)):
        raise ValueError("Points cannot contain NaN values")

def check_minimum_distance(points: np.ndarray, r: float, wrap: bool = False, 
                         domain: Optional[List[Tuple[float, float]]] = None) -> bool:
    """Check if all points maintain minimum distance r.
    
    Args:
        points: Array of points to check.
        r: Minimum distance requirement.
        wrap: Whether to use wrap-around distance calculation.
        domain: Domain boundaries for wrap-around calculation.
        
    Returns:
        True if all points maintain minimum distance, False otherwise.
    """
    if len(points) < 2:
        return True
    
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            if wrap and domain is not None:
                # Wrap-around distance calculation
                diff = np.abs(points[i] - points[j])
                domain_size = np.array([max_val - min_val for min_val, max_val in domain])
                wrapped_diff = np.minimum(diff, domain_size - diff)
                dist = np.sqrt(np.sum(wrapped_diff ** 2))
            else:
                dist = np.linalg.norm(points[i] - points[j])
            
            if dist < r and not np.isclose(dist, r):
                logger.warning(f"Points {i} and {j} violate minimum distance: {dist:.6f} < {r}")
                return False
    
    return True

def calculate_density(points: np.ndarray, domain: List[Tuple[float, float]]) -> float:
    """Calculate point density in the domain.
    
    Args:
        points: Array of points.
        domain: Domain boundaries.
        
    Returns:
        Point density (points per unit volume).
    """
    if len(points) == 0:
        return 0.0
    
    # Calculate domain volume
    volume = 1.0
    for min_val, max_val in domain:
        if max_val <= min_val:
            return 0.0  # Zero volume domain
        volume *= (max_val - min_val)
    
    if volume == 0:
        return 0.0
    
    return len(points) / volume

def estimate_optimal_spacing(domain: List[Tuple[float, float]], 
                           target_points: int) -> float:
    """Estimate optimal spacing for target number of points.
    
    Args:
        domain: Domain boundaries.
        target_points: Desired number of points.
        
    Returns:
        Estimated optimal spacing.
    """
    # Calculate domain volume
    volume = 1.0
    for min_val, max_val in domain:
        volume *= (max_val - min_val)
    
    # For Poisson disk sampling, optimal density is approximately 1/(r^d * sqrt(d))
    # where d is the number of dimensions
    dimensions = len(domain)
    optimal_density = target_points / volume
    
    # Solve for r: optimal_density = 1/(r^d * sqrt(d))
    r = (1.0 / (optimal_density * np.sqrt(dimensions))) ** (1.0 / dimensions)
    
    return r

def create_symmetry_operators(rotation_angles: Optional[List[float]] = None,
                             translation_vectors: Optional[List[np.ndarray]] = None,
                             custom_operators: Optional[List[callable]] = None) -> List[callable]:
    """Create common symmetry operators.
    
    Args:
        rotation_angles: List of rotation angles in radians.
        translation_vectors: List of translation vectors.
        custom_operators: List of custom symmetry operators.
        
    Returns:
        List of symmetry operators.
    """
    operators = []
    
    # Add rotation operators
    if rotation_angles is not None:
        for angle in rotation_angles:
            def make_rotator(theta: float) -> Callable[[np.ndarray], np.ndarray]:
                def rotator(point: np.ndarray) -> np.ndarray:
                    x, y = point[:2]
                    return np.array([x * np.cos(theta) - y * np.sin(theta),
                                   x * np.sin(theta) + y * np.cos(theta)] + list(point[2:]))
                return rotator
            operators.append(make_rotator(angle))
    
    # Add translation operators
    if translation_vectors is not None:
        for vector in translation_vectors:
            def make_translator(trans_vec: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
                def translator(point: np.ndarray) -> np.ndarray:
                    return point + trans_vec
                return translator
            operators.append(make_translator(vector))
    
    # Add custom operators
    if custom_operators is not None:
        operators.extend(custom_operators)
    
    return operators 