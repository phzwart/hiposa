#!/usr/bin/env python3
"""
Basic usage example for HiPoSa.

This script demonstrates the basic functionality of the HiPoSa library
including Poisson disk sampling, tiling, and point selection.
"""

import numpy as np
import matplotlib.pyplot as plt
from hiposa import PoissonDiskSamplerWithExisting, PoissonTiler, PointSelector
from hiposa.utils import setup_logging, validate_domain, check_minimum_distance

def main():
    """Run basic HiPoSa examples."""
    
    # Setup logging
    setup_logging()
    
    print("=== HiPoSa Basic Usage Example ===\n")
    
    # Example 1: Basic Poisson Disk Sampling
    print("1. Basic Poisson Disk Sampling")
    print("-" * 40)
    
    domain = [(0, 10), (0, 10)]
    r = 0.5
    
    # Validate domain
    validate_domain(domain)
    
    # Create sampler and generate points
    sampler = PoissonDiskSamplerWithExisting(domain=domain, r=r)
    points, labels = sampler.sample()
    
    print(f"Generated {len(points)} points")
    print(f"Point density: {len(points) / 100:.2f} points per unit area")
    
    # Verify minimum distance constraint
    if check_minimum_distance(points, r):
        print("✓ All points maintain minimum distance constraint")
    else:
        print("✗ Some points violate minimum distance constraint")
    
    # Plot results
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.scatter(points[:, 0], points[:, 1], c=labels, s=20, alpha=0.7)
    plt.title("Basic Poisson Disk Sampling")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True, alpha=0.3)
    
    # Example 2: Hierarchical Tiling
    print("\n2. Hierarchical Tiling")
    print("-" * 40)
    
    tile_size = 10.0
    spacings = [2.0, 1.0, 0.5]
    
    tiler = PoissonTiler(tile_size=tile_size, spacings=spacings)
    region = ((0, 30), (0, 20))
    tiled_points, tiled_levels = tiler.get_points_in_region(region)
    
    print(f"Generated {len(tiled_points)} points across {len(spacings)} levels")
    print(f"Level distribution: {np.bincount(tiled_levels)}")
    
    plt.subplot(1, 3, 2)
    colors = plt.cm.Set1(tiled_levels / max(tiled_levels))
    plt.scatter(tiled_points[:, 0], tiled_points[:, 1], c=colors, s=20, alpha=0.7)
    plt.title("Hierarchical Tiling")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True, alpha=0.3)
    
    # Example 3: Symmetry Operations
    print("\n3. Symmetry Operations")
    print("-" * 40)
    
    # Define a 90-degree rotation operator
    def rotate_90_degrees(point):
        x, y = point
        return np.array([-y, x])
    
    # Create sampler with symmetry
    symmetric_sampler = PoissonDiskSamplerWithExisting(
        domain=domain,
        r=r,
        symmetry_operators=[rotate_90_degrees]
    )
    
    symmetric_points, symmetric_labels = symmetric_sampler.sample()
    print(f"Generated {len(symmetric_points)} points with symmetry")
    
    plt.subplot(1, 3, 3)
    plt.scatter(symmetric_points[:, 0], symmetric_points[:, 1], 
                c=symmetric_labels, s=20, alpha=0.7)
    plt.title("Symmetric Sampling")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("basic_usage_example.png", dpi=150, bbox_inches='tight')
    print("\n✓ Plot saved as 'basic_usage_example.png'")
    
    # Example 4: Point Selection (if we have a function to evaluate)
    print("\n4. Point Selection")
    print("-" * 40)
    
    # Define a simple function to evaluate
    def test_function(point):
        x, y = point
        return np.sin(x) * np.cos(y) + 0.1 * np.random.randn()
    
    # Create some test data
    n_points = 50
    xy = np.random.rand(n_points, 2) * 10
    levels = np.random.randint(0, 3, n_points)
    scales = [1.0, 0.5, 0.25]
    
    # Create grid for interpolation
    grid_x = np.linspace(0, 10, 50)
    grid_y = np.linspace(0, 10, 50)
    grid_x, grid_y = np.meshgrid(grid_x, grid_y)
    
    try:
        selector = PointSelector(
            xy=xy,
            levels=levels,
            scales=scales,
            f_function=test_function,
            grid_x=grid_x,
            grid_y=grid_y
        )
        
        selected_points = selector.run(max_level=2)
        print(f"Selected {len(selected_points)} points based on function evaluation")
        
    except Exception as e:
        print(f"Point selection example failed: {e}")
        print("(This might be due to missing dependencies)")
    
    print("\n=== Example completed successfully! ===")

if __name__ == "__main__":
    main() 