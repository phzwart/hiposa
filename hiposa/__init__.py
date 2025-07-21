"""HiPoSa: Hierarchical Poisson Sampling

A Python library for generating hierarchical Poisson disk sampling patterns
with support for symmetry operations, tiling, and multi-dimensional spaces.

Main Features:
- Poisson disk sampling with minimum distance constraints
- Hierarchical sampling with multiple spacing levels
- Symmetry operations (rotational, translational, etc.)
- Periodic boundary conditions for tiling
- Multi-dimensional support (2D, 3D, 4D, etc.)
- Point selection based on interpolated data

Example:
    >>> import numpy as np
    >>> from hiposa import PoissonDiskSamplerWithExisting
    >>> 
    >>> # Create a 2D sampler
    >>> domain = [(0, 10), (0, 10)]
    >>> sampler = PoissonDiskSamplerWithExisting(domain=domain, r=0.5)
    >>> points, labels = sampler.sample()
    >>> print(f"Generated {len(points)} points")
"""

__version__ = "0.1.0"
__author__ = "Petrus H. Zwart"
__email__ = "phzwart@lbl.gov"

from .poisson_disc_sampling import PoissonDiskSamplerWithExisting
from .poisson_tiler import PoissonTiler
from .point_selector import PointSelector

__all__ = [
    'PoissonDiskSamplerWithExisting',
    'PoissonTiler', 
    'PointSelector'
]
