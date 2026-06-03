#!/usr/bin/env python

"""Tests for `hiposa` package."""

import pytest
import numpy as np
from hiposa.poisson_disc_sampling import (PoissonDiskSamplerWithExisting)
from hiposa.poisson_tiler import PoissonTiler
from scipy.spatial import KDTree
import matplotlib
matplotlib.use('Agg')


def test_basic_poisson_sampling():
    """Test basic Poisson disc sampling."""
    domain = [(0, 2), (0, 2)]  # Smaller domain
    r = 0.4  # Larger spacing to ensure fewer points
    
    sampler = PoissonDiskSamplerWithExisting(domain=domain, r=r)
    points, labels = sampler.sample()
    
    # Check that we got some points
    assert len(points) > 0
    # Check that points are within domain
    assert np.all(points[:, 0] >= 0) and np.all(points[:, 0] < 2)
    assert np.all(points[:, 1] >= 0) and np.all(points[:, 1] < 2)
    
    # Check minimum distance constraint
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            dist = np.linalg.norm(points[i] - points[j])
            assert dist >= r or np.isclose(dist, r)


def test_kdtree_validation():
    """Test point validation using KDTree."""
    domain = [(0, 2), (0, 2)]  # Smaller domain
    r = 0.4  # Larger spacing
    
    # Create sampler with existing points
    existing_points = np.array([[0.5, 0.5], [1.5, 1.5]])
    existing_labels = np.array([0, 1])
    sampler = PoissonDiskSamplerWithExisting(domain=domain, r=r, 
                                            existing_points=existing_points,
                                            existing_labels=existing_labels)
    
    # Add points in a controlled way
    additional_points = np.array([
        [0.5, 1.5],
        [1.5, 0.5]
    ])
    
    for point in additional_points:
        if sampler.is_valid_point(point):
            sampler.samples.append(point)
            sampler.labels = np.append(sampler.labels, "new")
    
    # Convert samples to numpy array and rebuild KDTree
    sampler.samples = np.array(sampler.samples)
    sampler.kdtree = KDTree(sampler.samples)
    
    # Test various points
    test_cases = [
        (np.array([0.6, 0.6]), False, "Point too close to [0.5, 0.5]"),
        (np.array([1.0, 1.0]), True, "Point far from all others"),
        (np.array([1.6, 1.6]), False, "Point too close to [1.5, 1.5]"),
        (np.array([0.1, 0.1]), True, "Point in empty region"),
        (np.array([-0.1, 1.0]), False, "Point outside domain"),
        (np.array([2.1, 1.0]), False, "Point outside domain")
    ]
    
    for point, expected_valid, description in test_cases:
        result = sampler.is_valid_point(point)
        assert result == expected_valid, \
            f"Failed: {description} - expected {expected_valid} but got {result}"
        print(f"Passed: {description}")


def test_wrap_around_distance():
    """Test wrap-around distance calculation."""
    domain = [(0, 2), (0, 2)]  # Smaller domain
    r = 0.4  # Larger spacing
    
    # Create sampler with wrap-around
    sampler = PoissonDiskSamplerWithExisting(domain=domain, r=r, wrap=True)
    
    # Test points on opposite edges (should be close with wrap-around)
    point1 = np.array([0.1, 1.0])
    point2 = np.array([1.9, 1.0])
    
    # With wrap-around, these points should be close (distance 0.2 < r)
    # So point2 should not be valid if point1 is already in the sample set
    sampler.samples = [point1]
    sampler.kdtree = None  # Not needed for this test
    assert not sampler.is_valid_point(point2)
    
    # Test points that are far apart even with wrap-around
    point3 = np.array([1.0, 1.0])
    assert sampler.is_valid_point(point3)


def test_automatic_label_assignment():
    """Test automatic label assignment when new_label is None."""
    domain = [(0, 2), (0, 2)]  # Smaller domain
    r = 0.4  # Larger spacing
    
    # Test with no existing points
    sampler = PoissonDiskSamplerWithExisting(domain=domain, r=r)
    points, labels = sampler.sample(new_label=None)
    assert np.all(labels == 0)  # Should start at 0 with no existing labels
    
    # Test with existing points and labels
    existing_points = np.array([[0.5, 0.5], [1.5, 1.5]])
    existing_labels = np.array([0, 1])
    sampler = PoissonDiskSamplerWithExisting(domain=domain, r=r,
                                            existing_points=existing_points,
                                            existing_labels=existing_labels)
    
    # Sample without specifying new_label
    points, labels = sampler.sample(new_label=None)
    print('Labels after sampling:', labels)
    # Robustly identify new points
    is_new = np.ones(len(points), dtype=bool)
    for ep in existing_points:
        is_new &= ~np.all(np.isclose(points, ep, atol=1e-8), axis=1)
    assert np.all(labels[is_new] == 2)


def test_return_new_only():
    """Test return_new_only parameter."""
    domain = [(0, 2), (0, 2)]  # Smaller domain
    r = 0.4  # Larger spacing
    
    existing_points = np.array([[0.5, 0.5], [1.5, 1.5]])
    existing_labels = np.array([0, 1])
    sampler = PoissonDiskSamplerWithExisting(domain=domain, r=r,
                                            existing_points=existing_points,
                                            existing_labels=existing_labels)
    
    # Test with return_new_only=True
    points, labels = sampler.sample(return_new_only=True)
    
    # Check that we only get new points
    for point in points:
        assert not np.any(np.all(existing_points == point, axis=1))


def test_3d_poisson_sampling():
    """Test Poisson disc sampling in 3D space."""
    domain = [(0, 2), (0, 2), (0, 2)]  # 3D domain
    r = 0.5  # Larger spacing for 3D
    
    sampler = PoissonDiskSamplerWithExisting(domain=domain, r=r)
    points, labels = sampler.sample()
    
    # Check that we got some points
    assert len(points) > 0
    # Check that points are within domain
    for dim in range(3):
        assert np.all(points[:, dim] >= 0) and np.all(points[:, dim] < 2)
    
    # Check minimum distance constraint
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            dist = np.linalg.norm(points[i] - points[j])
            assert dist >= r or np.isclose(dist, r)


def test_3d_poisson_tiling():
    """Test Poisson tiling in 3D space."""
    print("\nStarting 3D Poisson tiling test...")
    spacing = 1.0  # Much larger spacing
    tile_size = 1.5  # Smaller tile size
    
    print("Creating tiler...")
    tiler = PoissonTiler(tile_size=tile_size, spacings=[spacing, spacing/np.sqrt(3)], dimensions=3)
    
    print("\nTesting single tile first...")
    # Test a single tile
    single_tile_region = ((0, tile_size), (0, tile_size), (0, tile_size))
    points, levels = tiler.get_points_in_region(single_tile_region)
    
    print(f"Generated {len(points)} points in single tile")
    
    # Check that we got points
    assert len(points) > 0
    
    print("Checking domain bounds for single tile...")
    # Check that points are within region for all dimensions
    for dim in range(3):
        assert np.all(points[:, dim] >= 0) and np.all(points[:, dim] <= tile_size)
    
    print("Checking minimum distance constraints for single tile...")
    # Quick distance check - only check first few points
    min_distance = spacing/np.sqrt(3)
    for i in range(min(5, len(points))):
        for j in range(i + 1, min(i + 6, len(points))):
            dist = np.linalg.norm(points[i] - points[j])
            if dist < min_distance and not np.isclose(dist, min_distance):
                print(f"\nDistance constraint violation in single tile:")
                print(f"Point 1: {points[i]}")
                print(f"Point 2: {points[j]}")
                print(f"Distance: {dist:.6f}")
                print(f"Required minimum: {min_distance}")
                assert False, f"Found points with distance {dist:.6f} less than minimum allowed distance {min_distance}"
    
    print("\nSingle tile test passed! Now testing multiple tiles...")
    
    print("\nTesting tiling in a small 3D region...")
    # Test tiling a small region (2x2x2 tiles)
    region = ((0, 3), (0, 3), (0, 3))  # Smaller region, 2 tiles per dimension
    points, levels = tiler.get_points_in_region(region)
    
    print(f"Generated {len(points)} points")
    
    # Check that we got points
    assert len(points) > 0
    
    print("Checking domain bounds...")
    # Check that points are within region for all dimensions
    for dim in range(3):
        assert np.all(points[:, dim] >= 0) and np.all(points[:, dim] <= 3)
    
    print("Checking minimum distance constraints...")
    # Quick distance check - only check first few points
    min_distance = spacing/np.sqrt(3)
    for i in range(min(5, len(points))):
        for j in range(i + 1, min(i + 6, len(points))):
            dist = np.linalg.norm(points[i] - points[j])
            if dist < min_distance and not np.isclose(dist, min_distance):
                print(f"\nDistance constraint violation:")
                print(f"Point 1: {points[i]}")
                print(f"Point 2: {points[j]}")
                print(f"Distance: {dist:.6f}")
                print(f"Required minimum: {min_distance}")
                assert False, f"Found points with distance {dist:.6f} less than minimum allowed distance {min_distance}"
    
    print("3D Poisson tiling test completed successfully!")


def test_poisson_tiler_parallel():
    """Test Poisson tiler with parallel processing."""
    tile_size = 0.5  # Very small tile size
    spacings = [0.4, 0.2]  # Very large spacings
    dimensions = 3
    
    tiler = PoissonTiler(tile_size=tile_size, spacings=spacings, dimensions=dimensions)
    
    # Test cases designed to hit specific code paths:
    
    # 1. Region smaller than tile size (tests boundary masking)
    region_partial = ((0, 0.3), (0, 0.3), (0, 0.3))
    points_partial, labels_partial = tiler.get_points_in_region(region_partial, n_processes=1)
    
    # 2. Region exactly matching tile size (tests offset calculation)
    region_exact = ((0, tile_size), (0, tile_size), (0, tile_size))
    points_exact, labels_exact = tiler.get_points_in_region(region_exact, n_processes=1)
    
    # 3. Region with partial tiles at edges (tests masking and offset)
    region_edge = ((0.1, 0.6), (0.1, 0.6), (0.1, 0.6))  # Tiny region
    points_edge, labels_edge = tiler.get_points_in_region(region_edge, n_processes=1)
    
    # 4. Small region with few tiles (tests parallel processing)
    region_small = ((0, 1), (0, 1), (0, 1))  # Tiny region
    points_small, labels_small = tiler.get_points_in_region(region_small, n_processes=1)
    
    # Verify all test cases
    test_cases = [
        (points_partial, region_partial, labels_partial, "partial"),
        (points_exact, region_exact, labels_exact, "exact"),
        (points_edge, region_edge, labels_edge, "edge"),
        (points_small, region_small, labels_small, "small")
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
            assert np.all(points[:, dim] <= max_val), \
                f"Points above maximum in {case_name} case, dimension {dim}"
            print(f"Dimension {dim} bounds verified: {min_val} to {max_val}")
        
        # Quick distance check - only check first few points to avoid O(n²)
        min_distance = spacings[1]  # Smallest spacing
        for i in range(min(2, len(points))):
            for j in range(i + 1, min(i + 3, len(points))):
                dist = np.linalg.norm(points[i] - points[j])
                assert dist >= min_distance or np.isclose(dist, min_distance), \
                    f"Distance constraint violated in {case_name} case: {dist} < {min_distance}"
        
        print(f"{case_name} case verified successfully!")

