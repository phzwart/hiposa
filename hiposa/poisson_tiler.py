import numpy as np
from .poisson_disc_sampling import PoissonDiskSamplerWithExisting
from multiprocessing import Pool
from functools import partial
import itertools

class PoissonTiler:
    """
    Creates hierarchical Poisson disc sampling patterns that can be tiled across a large area.
    Supports arbitrary dimensions, defaulting to 2D.
    """
    def __init__(self, tile_size, spacings, dimensions=2):
        """
        Initialize the tiler with tile size and spacing levels.
        
        Args:
            tile_size (float): Size of the square tile
            spacings (list): List of inter-point distances, from largest to smallest
            dimensions (int): Number of dimensions for the tiling (default: 2)
        """
        self.spacings = sorted(spacings, reverse=True)  # Ensure largest spacing first
        
        # Ensure tile size is large enough compared to largest spacing
        # For periodic tiling to work, tile size should be at least 4x the largest spacing
        min_tile_size = 4.0 * self.spacings[0]
        self.tile_size = max(tile_size, min_tile_size)
        
        self.dimensions = dimensions
        self.tile_domain = [(0, self.tile_size)] * dimensions
        self.tile_points = None
        self.tile_labels = None
        
        print(f"\nInitializing PoissonTiler:")
        print(f"Requested tile size: {tile_size}")
        print(f"Actual tile size: {self.tile_size} (minimum required: {min_tile_size})")
        print(f"Spacings: {spacings}")
        print(f"Dimensions: {dimensions}")
        print(f"Tile domain: {self.tile_domain}")
        
        # Generate the base tile
        self._generate_base_tile()

    def _generate_base_tile(self):
        """Generate hierarchical sampling within a single periodic tile."""
        print("\nGenerating base tile...")
        points = None
        labels = None
        
        # Generate points for each spacing level
        for level, spacing in enumerate(self.spacings):
            print(f"\nGenerating level {level} with spacing {spacing}")
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
            
            print(f"Level {level}: Generated {len(new_points)} points")
        
        self.tile_points = points
        self.tile_labels = labels.astype(np.int32)  # Ensure all labels are integers
        print(f"\nBase tile complete with {len(points)} total points")

    def _process_tile(self, args):
        """Process a single tile - used for parallel processing."""
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

    def get_points_in_region(self, region, n_processes=None):
        """Get all points and their levels within a specified region using parallel processing."""
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
        
        return all_points, all_labels
