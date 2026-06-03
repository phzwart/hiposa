import logging
import numpy as np
from .poisson_disc_sampling import PoissonDiskSamplerWithExisting
from multiprocessing import Pool
from functools import partial
import itertools
from typing import List, Tuple, Optional, Union, Any

logger = logging.getLogger(__name__)

class PoissonTiler:
    """
    Creates hierarchical Poisson disc sampling patterns that can be tiled across a large area.
    Supports arbitrary dimensions, defaulting to 2D.
    """
    def __init__(self, tile_size: float, spacings: List[float], dimensions: int = 2, min_tile_factor: float = 2.0) -> None:
        """
        Initialize the tiler with tile size and spacing levels.
        
        Args:
            tile_size: Size of the square tile.
            spacings: List of inter-point distances, from largest to smallest.
            dimensions: Number of dimensions for the tiling (default: 2).
            min_tile_factor: Minimum factor for tile size relative to largest spacing.
        """
        self.spacings = sorted(spacings, reverse=True)  # Ensure largest spacing first
        
        # Ensure tile size is large enough compared to largest spacing
        # For periodic tiling to work, tile size should be at least 4x the largest spacing
        min_tile_size = min_tile_factor * self.spacings[0]
        self.tile_size = max(tile_size, min_tile_size)
        
        self.dimensions = dimensions
        self.tile_domain = [(0, self.tile_size)] * dimensions
        self.tile_points = None
        self.tile_labels = None
        
        logger.info("Initializing PoissonTiler:")
        logger.info("Requested tile size: %s", tile_size)
        logger.info("Actual tile size: %s (minimum required: %s)", self.tile_size, min_tile_size)
        logger.info("Spacings: %s", spacings)
        logger.info("Dimensions: %s", dimensions)
        logger.info("Tile domain: %s", self.tile_domain)
        
        # Generate the base tile
        self._generate_base_tile()

    def _generate_base_tile(self) -> None:
        """Generate hierarchical sampling within a single periodic tile."""
        logger.info("Generating base tile...")
        points = None
        labels = None
        
        # Generate points for each spacing level
        for level, spacing in enumerate(self.spacings):
            logger.info("Generating level %s with spacing %s", level, spacing)
            sampler = PoissonDiskSamplerWithExisting(
                domain=self.tile_domain,
                r=spacing,
                existing_points=points,
                existing_labels=labels,
                wrap=True  # Enable periodic boundary conditions
            )
            
            # Get ONLY NEW points for this level using return_new_only=True
            new_points, new_labels = sampler.sample(new_label=int(level), return_new_only=True)
            
            if points is None:
                points = new_points
                labels = new_labels
            else:
                points = np.vstack((points, new_points))
                labels = np.concatenate((labels, new_labels))
            
            logger.info("Level %s: Generated %s points", level, len(new_points))
        
        self.tile_points = points
        self.tile_labels = labels.astype(np.int32)  # Ensure all labels are integers
        logger.info("Base tile complete with %s total points", len(points))

    def _process_tile(self, args: Tuple) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process a single tile - used for parallel processing.
        
        Args:
            args: Tuple containing tile indices and bounds.
            
        Returns:
            Tuple of (points, labels) for the processed tile.
        """
        indices = args[:self.dimensions]
        bounds = args[self.dimensions:]
        
        # Calculate offset based on tile indices
        offset = np.array([bounds[2*dim] + idx * self.tile_size for dim, idx in enumerate(indices)])
        
        # Apply offset to tile points and mask those within bounds
        tile_points = self.tile_points + offset
        mask = np.ones(len(tile_points), dtype=bool)
        for dim in range(self.dimensions):
            min_val, max_val = bounds[2*dim:2*dim + 2]
            mask &= (tile_points[:, dim] >= min_val) & (tile_points[:, dim] < max_val)
        
        return tile_points[mask], self.tile_labels[mask]

    def get_points_in_region(self, region: List[Tuple[float, float]], n_processes: Optional[int] = None, add_corners: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get all points and their levels within a specified region using parallel processing.
        
        Args:
            region: List of tuples defining the region bounds for each dimension.
            n_processes: Number of processes to use for parallel processing.
            add_corners: Whether to add corner points to the result (default: True).
            
        Returns:
            Tuple of (points, labels) arrays for the specified region.
        """
        # Extract bounds and calculate tiles needed
        bounds = [b for dim in region for b in dim]
        n_tiles = [int(np.ceil((bounds[i+1] - bounds[i]) / self.tile_size)) 
                  for i in range(0, len(bounds), 2)]
        
        # Generate tile indices and prepare arguments
        tile_indices = list(itertools.product(*[range(n) for n in n_tiles]))
        tile_args = [tuple(indices) + tuple(bounds) for indices in tile_indices]
        
        # Use single process if only one tile or explicitly requested
        if len(tile_indices) == 1 or n_processes == 1:
            results = [self._process_tile(args) for args in tile_args]
        else:
            with Pool(processes=n_processes) as pool:
                results = pool.map(self._process_tile, tile_args)
        
        # Combine results
        all_points = np.concatenate([points for points, _ in results if len(points) > 0], axis=0) \
                     if results else np.array([])
        all_labels = np.concatenate([labels for _, labels in results if len(labels) > 0], axis=0) \
                     if results else np.array([])
        
        if add_corners:
            # Add corner points
            corner_indices = list(itertools.product(*[[0, 1]] * self.dimensions))
            corner_points = []
            for indices in corner_indices:
                point = []
                for dim in range(self.dimensions):
                    point.append(bounds[2*dim] if indices[dim] == 0 else bounds[2*dim + 1])
                corner_points.append(point)
            
            corner_points = np.array(corner_points)
            corner_labels = np.zeros(len(corner_points), dtype=np.int32)  # Level 0 for corners
            
            # Combine with existing points
            if len(all_points) > 0:
                all_points = np.vstack((all_points, corner_points))
                all_labels = np.concatenate((all_labels, corner_labels))
            else:
                all_points = corner_points
                all_labels = corner_labels
        
        return all_points, all_labels
