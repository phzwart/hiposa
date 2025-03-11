#!/usr/bin/env python

"""Tests for `hiposa` package."""

import pytest
import numpy as np
from hiposa.poisson_disc_sampling import (PoissonDiskSamplerWithExisting)
from hiposa.poisson_tiler import PoissonTiler
from scipy.spatial import KDTree


def test_basic_poisson_sampling():
    """Test basic Poisson disc sampling."""
    domain = [(0, 10), (0, 10)]
    r = 0.5
    
    sampler = PoissonDiskSamplerWithExisting(domain=domain, r=r)
    points, labels = sampler.sample()
    
    # Check that we got some points
    assert len(points) > 0
    # Check that points are within domain
    assert np.all(points[:, 0] >= 0) and np.all(points[:, 0] < 10)
    assert np.all(points[:, 1] >= 0) and np.all(points[:, 1] < 10)
    
    # Check minimum distance constraint
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            dist = np.linalg.norm(points[i] - points[j])
            assert dist >= r or np.isclose(dist, r)


def test_poisson_tiler():
    """Test Poisson tiler with hierarchical sampling."""
    tile_size = 15.0
    spacings = [1.0, 0.5]
    
    tiler = PoissonTiler(tile_size=tile_size, spacings=spacings)
    
    # Test tiling a small region (2x2 tiles)
    region = ((0, 50), (0, 30))
    points, levels = tiler.get_points_in_region(region)
    
    # Check that we got points
    assert len(points) > 0
    # Check that points are within region
    assert np.all(points[:, 0] >= 0) and np.all(points[:, 0] < 50)
    assert np.all(points[:, 1] >= 0) and np.all(points[:, 1] < 30)
    
    # Check that we have points from different levels
    assert len(np.unique(levels)) == len(spacings)


def test_kdtree_validation():
    """Test point validation using KDTree."""
    domain = [(0, 10), (0, 10)]
    r = 0.5
    
    # Create sampler with existing points
    existing_points = np.array([[1.0, 1.0], [2.0, 2.0]])
    existing_labels = np.array([0, 1])
    sampler = PoissonDiskSamplerWithExisting(domain=domain, r=r, 
                                            existing_points=existing_points,
                                            existing_labels=existing_labels)
    
    # Add points in a controlled way
    additional_points = np.array([
        [3.0, 3.0],
        [4.0, 4.0],
        [1.0, 4.0],
        [4.0, 1.0]
    ])
    
    for point in additional_points:
        if sampler.is_valid_point(point):
            sampler.samples.append(point)
            sampler.idx_to_point[len(sampler.samples)-1] = point
            sampler.labels = np.append(sampler.labels, "new")
    
    # Convert samples to numpy array and rebuild KDTree
    sampler.samples = np.array(sampler.samples)
    sampler.kdtree = KDTree(sampler.samples)
    
    # Test various points
    test_cases = [
        (np.array([1.1, 1.1]), False, "Point too close to [1.0, 1.0]"),
        (np.array([5.0, 5.0]), True, "Point far from all others"),
        (np.array([2.1, 2.1]), False, "Point too close to [2.0, 2.0]"),
        (np.array([7.0, 7.0]), True, "Point in empty region"),
        (np.array([-1.0, 5.0]), False, "Point outside domain"),
        (np.array([10.1, 5.0]), False, "Point outside domain")
    ]
    
    for point, expected_valid, description in test_cases:
        result = sampler.is_valid_point(point)
        assert result == expected_valid, \
            f"Failed: {description} - expected {expected_valid} but got {result}"
        print(f"Passed: {description}")


def test_wrap_around_distance():
    """Test wrap-around distance calculation."""
    domain = [(0, 10), (0, 10)]
    r = 0.5
    
    sampler = PoissonDiskSamplerWithExisting(domain=domain, r=r, wrap=True)
    
    # Test point near edge
    point = np.array([9.0, 5.0])
    points = np.array([[0.1, 5.0]])  # Point near opposite edge
    
    # This should use wrap-around distance calculation
    assert sampler.is_valid_point(point)


def test_automatic_label_assignment():
    """Test automatic label assignment when new_label is None."""
    domain = [(0, 10), (0, 10)]
    r = 0.5
    
    # Test with no existing points
    sampler = PoissonDiskSamplerWithExisting(domain=domain, r=r)
    points, labels = sampler.sample(new_label=None)
    assert np.all(labels == 0)  # Should start at 0 with no existing labels
    
    # Test with existing points and labels
    existing_points = np.array([[1.0, 1.0], [2.0, 2.0]])
    existing_labels = np.array([0, 1])
    sampler = PoissonDiskSamplerWithExisting(domain=domain, r=r,
                                            existing_points=existing_points,
                                            existing_labels=existing_labels)
    
    # Sample without specifying new_label
    points, labels = sampler.sample(new_label=None)
    
    # Check that new points got label 2 (max existing label + 1)
    new_points_mask = ~np.isin(points, existing_points).all(axis=1)
    assert np.all(labels[new_points_mask] == 2)





def test_return_new_only():
    """Test the return_new_only parameter in sample method."""
    domain = [(0, 10), (0, 10)]
    r = 0.5
    
    # Create sampler with existing points
    existing_points = np.array([[1.0, 1.0], [2.0, 2.0]])
    existing_labels = np.array([0, 1])
    sampler = PoissonDiskSamplerWithExisting(domain=domain, r=r,
                                            existing_points=existing_points,
                                            existing_labels=existing_labels)
    
    # Sample with return_new_only=True
    new_points, new_labels = sampler.sample(return_new_only=True)
    
    # Check that we only got new points
    assert len(new_points) > 0
    assert not np.any(np.isin(new_points, existing_points).all(axis=1))
    assert len(new_labels) == len(new_points)


def test_4d_poisson_sampling():
    """Test Poisson disc sampling in 4D space."""
    print("\nStarting 4D Poisson sampling test...")
    domain = [(0, 10), (0, 10), (0, 10), (0, 10)]  # 4D hypercube
    r = 2.0
    
    print("Creating sampler...")
    sampler = PoissonDiskSamplerWithExisting(domain=domain, r=r)
    
    print("Sampling points...")
    points, labels = sampler.sample()
    
    print(f"Generated {len(points)} points")
    
    # Check that we got some points
    assert len(points) > 0
    
    print("Checking domain bounds...")
    # Check that points are within domain for all dimensions
    for dim in range(4):
        assert np.all(points[:, dim] >= 0) and np.all(points[:, dim] < 10)
    
    print("Checking minimum distance constraints...")
    # Check minimum distance constraint
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            dist = np.linalg.norm(points[i] - points[j])
            assert dist >= r or np.isclose(dist, r)
    
    print("Testing with existing points...")
    # Test with existing points
    existing_points = np.array([[1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0]])
    existing_labels = np.array([0, 1])
    sampler = PoissonDiskSamplerWithExisting(domain=domain, r=r,
                                            existing_points=existing_points,
                                            existing_labels=existing_labels)
    
    print("Testing point validation...")
    # Test point too close to existing point
    point = np.array([1.1, 1.1, 1.1, 1.1])  # Close to first existing point
    assert not sampler.is_valid_point(point)
    
    # Test point far enough from existing points
    point = np.array([5.0, 5.0, 5.0, 5.0])  # Far from existing points
    assert sampler.is_valid_point(point)
    
    print("Testing wrap-around functionality...")
    # Test wrap-around in 4D
    sampler = PoissonDiskSamplerWithExisting(domain=domain, r=r, wrap=True)
    point = np.array([9.0, 5.0, 5.0, 5.0])
    points = np.array([[0.1, 5.0, 5.0, 5.0]])  # Point near opposite edge
    assert sampler.is_valid_point(point)
    
    print("4D Poisson sampling test completed successfully!")


def test_4d_poisson_tiling():
    """Test Poisson tiling in 4D space."""
    print("\nStarting 4D Poisson tiling test...")
    spacing = 1  # Single spacing value
    tile_size = 3.0 
    
    print("Creating tiler...")
    tiler = PoissonTiler(tile_size=tile_size, spacings=[spacing, spacing/np.sqrt(3)], dimensions=4)
    
    print("\nTesting single tile first...")
    # Test a single tile
    single_tile_region = ((0, tile_size), (0, tile_size), (0, tile_size), (0, tile_size))
    points, levels = tiler.get_points_in_region(single_tile_region)
    
    print(f"Generated {len(points)} points in single tile")
    
    # Check that we got points
    assert len(points) > 0
    
    print("Checking domain bounds for single tile...")
    # Check that points are within region for all dimensions
    for dim in range(4):
        assert np.all(points[:, dim] >= 0) and np.all(points[:, dim] < tile_size)
    
    print("Checking minimum distance constraints for single tile...")
    # Check minimum distance constraint
    min_distance = spacing/np.sqrt(3)
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            dist = np.linalg.norm(points[i] - points[j])
            if dist < min_distance:
                print(f"\nDistance constraint violation in single tile:")
                print(f"Point 1: {points[i]}")
                print(f"Point 2: {points[j]}")
                print(f"Distance: {dist:.6f}")
                print(f"Required minimum: {min_distance}")
                assert False, f"Found points with distance {dist:.6f} less than minimum allowed distance {min_distance}"
    
    print("\nSingle tile test passed! Now testing multiple tiles...")
    
    print("\nTesting tiling in a small 4D region...")
    # Test tiling a small region (2x2x2x2 tiles)
    region = ((0, 6), (0, 3), (0, 3), (0, 6))  # Smaller region, 2 tiles per dimension
    points, levels = tiler.get_points_in_region(region)
    
    print(f"Generated {len(points)} points")
    
    # Check that we got points
    assert len(points) > 0
    
    print("Checking domain bounds...")
    # Check that points are within region for all dimensions
    for dim in range(4):
        assert np.all(points[:, dim] >= 0) and np.all(points[:, dim] < 8)
    
    print("Checking minimum distance constraints...")
    # Check minimum distance constraint
    min_distance = spacing/np.sqrt(3)
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            dist = np.linalg.norm(points[i] - points[j])
            if dist < min_distance:
                print(f"\nDistance constraint violation:")
                print(f"Point 1: {points[i]}")
                print(f"Point 2: {points[j]}")
                print(f"Distance: {dist:.6f}")
                print(f"Required minimum: {min_distance}")
                assert False, f"Found points with distance {dist:.6f} less than minimum allowed distance {min_distance}"
    
    print("4D Poisson tiling test completed successfully!")


def test_poisson_tiler_parallel():
    """Test Poisson tiler with parallel processing."""
    tile_size = 4.0
    spacings = [1.0, 0.5]
    dimensions = 3
    
    tiler = PoissonTiler(tile_size=tile_size, spacings=spacings, dimensions=dimensions)
    
    # Test cases designed to hit specific code paths:
    
    # 1. Region smaller than tile size (tests boundary masking)
    region_partial = ((0, 2), (0, 2), (0, 2))
    points_partial, labels_partial = tiler.get_points_in_region(region_partial, n_processes=2)
    
    # 2. Region exactly matching tile size (tests offset calculation)
    region_exact = ((0, tile_size), (0, tile_size), (0, tile_size))
    points_exact, labels_exact = tiler.get_points_in_region(region_exact, n_processes=2)
    
    # 3. Region with partial tiles at edges (tests masking and offset)
    region_edge = ((1, 7), (1, 7), (1, 7))  # Non-zero start, partial tiles
    points_edge, labels_edge = tiler.get_points_in_region(region_edge, n_processes=2)
    
    # 4. Large region with many tiles (tests parallel processing)
    region_large = ((0, 12), (0, 12), (0, 12))
    points_large, labels_large = tiler.get_points_in_region(region_large, n_processes=None)
    
    # Verify all test cases
    test_cases = [
        (points_partial, region_partial, labels_partial, "partial"),
        (points_exact, region_exact, labels_exact, "exact"),
        (points_edge, region_edge, labels_edge, "edge"),
        (points_large, region_large, labels_large, "large")
    ]
    
    for points, region, labels, case_name in test_cases:
        print(f"\nVerifying {case_name} case:")
        print(f"Number of points: {len(points)}")
        
        # Basic assertions
        assert len(points) > 0, f"No points generated for {case_name} case"
        assert len(points) == len(labels), f"Points and labels mismatch in {case_name} case"
        
        # Verify points are within bounds
        for dim in range(dimensions):
            min_val, max_val = region[dim]
            assert np.all(points[:, dim] >= min_val), \
                f"Points below minimum in {case_name} case, dimension {dim}"
            assert np.all(points[:, dim] < max_val), \
                f"Points above maximum in {case_name} case, dimension {dim}"
            print(f"Dimension {dim} bounds verified: {min_val} to {max_val}")
        
        # Verify minimum distance between points
        min_distance = spacings[1]  # Smallest spacing
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                dist = np.linalg.norm(points[i] - points[j])
                assert dist >= min_distance, \
                    f"Distance constraint violated in {case_name} case: {dist} < {min_distance}"
        
        # Verify we got points from all spacing levels
        unique_labels = np.unique(labels)
        assert len(unique_labels) == len(spacings), \
            f"Missing spacing levels in {case_name} case"


def test_rotational_symmetry_with_offset():
    """Test symmetry operator that rotates points and adds an offset."""
    # Domain from -1 to 1 in both dimensions
    domain = [(-1, 1), (-1, 1)]
    r = 0.2  # Small enough to get a good number of points
    
    def rotate_and_offset(point):
        """Transforms (x,y) -> (-y, -x + 1/2) and maps back to [-1, 1] domain"""
        x, y = point
        # Apply transformation
        new_x = y
        new_y = -x  #+ 0.5
        
        # Map back to [-1, 1] domain using periodic boundaries
        new_x = ((new_x + 1) % 2) - 1
        new_y = ((new_y + 1) % 2) - 1
        
        return np.array([new_x, new_y])
    
    # Create sampler with periodic boundaries and symmetry
    sampler = PoissonDiskSamplerWithExisting(
        domain=domain,
        r=r,
        symmetry_operators=[rotate_and_offset],
        wrap=True
    )
    
    # Test the orbit of a single point
    test_point = np.array([0.5, 0.3])
    orbit = sampler.apply_symmetry(test_point)
    for point in orbit:
        print(f"Point: {point}")

    # For this specific operator, we expect 4 points in the orbit
    # as applying the operator 4 times brings us back to start
    assert len(orbit) == 4, f"Expected 4 points in orbit, got {len(orbit)}"
    
    # Verify that applying the operator to the last point gets us back to the first
    final_transform = rotate_and_offset(orbit[-1])
    diff = np.abs(final_transform - orbit[0])
    diff = np.minimum(diff, 2 - diff)  # Account for periodic boundaries
    assert np.all(diff < 1e-10), "Orbit does not close properly"
    
    # Sample points
    points, labels = sampler.sample()
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 8))
    plt.scatter(points[:, 0], points[:, 1], c=labels, cmap='viridis', marker='o')
    plt.title('Generated Points with Rotational Symmetry')
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.axhline(0, color='grey', lw=0.5, ls='--')
    plt.axvline(0, color='grey', lw=0.5, ls='--')
    plt.grid()
    plt.show()
    
    # Basic checks
    assert len(points) > 0
    print(f"Generated {len(points)} points")
    
    # Check domain bounds
    assert np.all(points >= -1) and np.all(points <= 1), "Points outside domain"
    
    # Group points into their orbits and verify each orbit is complete
    remaining_points = points.copy()
    while len(remaining_points) > 0:
        point = remaining_points[0]
        orbit = sampler.apply_symmetry(point)
        
        # Check that all points in the orbit exist in our sample
        for orbit_point in orbit:
            found = False
            for i, sample_point in enumerate(remaining_points):
                diff = np.abs(orbit_point - sample_point)
                diff = np.minimum(diff, 2 - diff)  # Account for periodic boundaries
                if np.all(diff < 1e-10):
                    found = True
                    remaining_points = np.delete(remaining_points, i, axis=0)
                    break
            assert found, f"Missing point {orbit_point} from orbit of {point}"
    
    # Check minimum distance constraint including across periodic boundaries
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            # Calculate direct distance
            diff = points[i] - points[j]
            # Account for periodic boundaries
            diff = np.minimum(np.abs(diff), 2 - np.abs(diff))
            dist = np.sqrt(np.sum(diff ** 2))
            assert dist >= r, f"Points too close: {points[i]} and {points[j]} with distance {dist}"

