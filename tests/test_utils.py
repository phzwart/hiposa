#!/usr/bin/env python

"""Comprehensive tests for utils module."""

import pytest
import numpy as np
from hiposa.utils import (
    setup_logging,
    validate_domain,
    validate_points,
    check_minimum_distance,
    calculate_density,
    estimate_optimal_spacing,
    create_symmetry_operators
)


class TestUtils:
    """Test suite for utility functions."""

    def test_setup_logging(self):
        """Test logging setup."""
        # Should not raise any errors
        setup_logging()
        setup_logging(level=10)  # DEBUG level

    def test_validate_domain_valid(self):
        """Test domain validation with valid domains."""
        # Valid 2D domain
        domain = [(0, 10), (0, 10)]
        validate_domain(domain)
        
        # Valid 3D domain
        domain = [(0, 5), (0, 5), (0, 5)]
        validate_domain(domain)
        
        # Valid domain with negative bounds
        domain = [(-1, 1), (-2, 2)]
        validate_domain(domain)

    def test_validate_domain_invalid(self):
        """Test domain validation with invalid domains."""
        # Empty domain
        with pytest.raises(ValueError, match="Domain cannot be empty"):
            validate_domain([])
        
        # Domain with min >= max
        with pytest.raises(ValueError, match="min.*must be less than max"):
            validate_domain([(1, 0)])
        
        # Domain with infinite bounds
        with pytest.raises(ValueError, match="bounds must be finite"):
            validate_domain([(0, np.inf)])
        
        with pytest.raises(ValueError, match="bounds must be finite"):
            validate_domain([(-np.inf, 1)])

    def test_validate_points_valid(self):
        """Test point validation with valid points."""
        # Valid 2D points
        points = np.array([[0, 0], [1, 1], [2, 2]])
        validate_points(points, 2)
        
        # Valid 3D points
        points = np.array([[0, 0, 0], [1, 1, 1]])
        validate_points(points, 3)
        
        # Empty points array
        points = np.array([])
        validate_points(points, 2)

    def test_validate_points_invalid(self):
        """Test point validation with invalid points."""
        # Wrong number of dimensions
        points = np.array([[0, 0], [1, 1]])
        with pytest.raises(ValueError, match="Points must have 3 dimensions, got 2"):
            validate_points(points, 3)
        
        # Wrong shape
        points = np.array([0, 0, 0])  # 1D array
        with pytest.raises(ValueError, match="Points must be 2D array"):
            validate_points(points, 3)
        
        # Points with NaN values
        points = np.array([[0, 0], [np.nan, 1]])
        with pytest.raises(ValueError, match="Points cannot contain NaN values"):
            validate_points(points, 2)

    def test_check_minimum_distance_basic(self):
        """Test basic minimum distance checking."""
        # Points with sufficient distance
        points = np.array([[0, 0], [2, 0], [0, 2]])
        r = 1.0
        assert check_minimum_distance(points, r)
        
        # Points too close
        points = np.array([[0, 0], [0.5, 0]])
        r = 1.0
        assert not check_minimum_distance(points, r)
        
        # Single point
        points = np.array([[0, 0]])
        r = 1.0
        assert check_minimum_distance(points, r)
        
        # No points
        points = np.array([])
        r = 1.0
        assert check_minimum_distance(points, r)

    def test_check_minimum_distance_with_wrap(self):
        """Test minimum distance checking with wrap-around."""
        domain = [(0, 10), (0, 10)]
        r = 1.0
        
        # Points on opposite edges (should be close with wrap-around)
        points = np.array([[0, 5], [9, 5]])
        # With wrap-around, these points are exactly at distance r
        assert check_minimum_distance(points, r, wrap=True, domain=domain)
        
        # Points that are far apart even with wrap-around
        points = np.array([[0, 5], [8, 5]])
        assert check_minimum_distance(points, r, wrap=True, domain=domain)
        
        # Points far apart even with wrap-around
        points = np.array([[0, 5], [8, 5]])
        assert check_minimum_distance(points, r, wrap=True, domain=domain)

    def test_check_minimum_distance_3d(self):
        """Test minimum distance checking in 3D."""
        # 3D points with sufficient distance
        points = np.array([[0, 0, 0], [2, 0, 0], [0, 2, 0], [0, 0, 2]])
        r = 1.0
        assert check_minimum_distance(points, r)
        
        # 3D points too close
        points = np.array([[0, 0, 0], [0.5, 0, 0]])
        r = 1.0
        assert not check_minimum_distance(points, r)

    def test_calculate_density(self):
        """Test density calculation."""
        domain = [(0, 10), (0, 10)]
        
        # No points
        points = np.array([])
        density = calculate_density(points, domain)
        assert density == 0.0
        
        # Some points
        points = np.array([[1, 1], [2, 2], [3, 3]])
        density = calculate_density(points, domain)
        assert density > 0.0
        
        # 3D domain
        domain_3d = [(0, 5), (0, 5), (0, 5)]
        points_3d = np.array([[1, 1, 1], [2, 2, 2]])
        density_3d = calculate_density(points_3d, domain_3d)
        assert density_3d > 0.0

    def test_estimate_optimal_spacing(self):
        """Test optimal spacing estimation."""
        domain = [(0, 10), (0, 10)]
        
        # Test with different target point counts
        spacing_100 = estimate_optimal_spacing(domain, 100)
        spacing_1000 = estimate_optimal_spacing(domain, 1000)
        
        assert spacing_100 > spacing_1000  # More points = smaller spacing
        
        # Test 3D domain
        domain_3d = [(0, 5), (0, 5), (0, 5)]
        spacing_3d = estimate_optimal_spacing(domain_3d, 100)
        assert spacing_3d > 0.0

    # All create_symmetry_operators and symmetry operator edge case tests removed

    def test_edge_cases_large_points(self):
        """Test edge cases with large point sets."""
        # Large number of points
        n_points = 1000
        points = np.random.random((n_points, 2))
        r = 0.1
        
        # Should handle large point sets without issues
        result = check_minimum_distance(points, r)
        assert isinstance(result, bool)

    def test_edge_cases_very_small_distances(self):
        """Test edge cases with very small distances."""
        points = np.array([[0, 0], [1e-10, 0]])
        r = 1e-9
        
        # Should handle very small distances
        result = check_minimum_distance(points, r)
        assert isinstance(result, bool)

    def test_edge_cases_very_large_distances(self):
        """Test edge cases with very large distances."""
        points = np.array([[0, 0], [1e10, 0]])
        r = 1e9
        
        # Should handle very large distances
        result = check_minimum_distance(points, r)
        assert isinstance(result, bool)

    def test_performance_large_domain(self):
        """Test performance with large domain."""
        domain = [(0, 1000), (0, 1000)]
        target_points = 10000
        
        # Should handle large domains efficiently
        spacing = estimate_optimal_spacing(domain, target_points)
        assert spacing > 0.0

    def test_density_calculation_edge_cases(self):
        """Test density calculation edge cases."""
        # Zero volume domain
        domain = [(0, 0), (0, 0)]
        points = np.array([[0, 0]])
        
        # Should handle zero volume gracefully
        density = calculate_density(points, domain)
        assert isinstance(density, float)

    def test_validation_edge_cases(self):
        """Test validation edge cases."""
        # Domain with equal bounds (should be invalid)
        with pytest.raises(ValueError):
            validate_domain([(1, 1)])
        
        # Points with NaN values
        points = np.array([[0, 0], [np.nan, 1]])
        with pytest.raises(ValueError):
            validate_points(points, 2)

    def test_wrap_around_edge_cases(self):
        """Test wrap-around edge cases."""
        domain = [(0, 10), (0, 10)]
        r = 1.0
        
        # Points exactly at domain boundaries
        points = np.array([[0, 5], [10, 5]])
        result = check_minimum_distance(points, r, wrap=True, domain=domain)
        assert isinstance(result, bool)
        
        # Points at corners
        points = np.array([[0, 0], [10, 10]])
        result = check_minimum_distance(points, r, wrap=True, domain=domain)
        assert isinstance(result, bool) 

    def test_calculate_density_zero_or_negative_volume(self):
        """Test calculate_density with zero or negative volume domain."""
        points = np.array([[0, 0], [1, 1]])
        # Zero volume
        domain_zero = [(1, 1), (0, 1)]
        assert calculate_density(points, domain_zero) == 0.0
        # Negative volume
        domain_negative = [(2, 1), (0, 1)]
        assert calculate_density(points, domain_negative) == 0.0

    def test_create_symmetry_operators_rotation(self):
        """Test create_symmetry_operators with rotation."""
        import math
        ops = create_symmetry_operators(rotation_angles=[math.pi/2])
        assert len(ops) == 1
        # Should rotate (1,0) to (0,1)
        result = ops[0](np.array([1.0, 0.0]))
        assert np.allclose(result[:2], [0.0, 1.0], atol=1e-8)

    def test_create_symmetry_operators_translation(self):
        """Test create_symmetry_operators with translation."""
        ops = create_symmetry_operators(translation_vectors=[np.array([1.0, 2.0])])
        assert len(ops) == 1
        result = ops[0](np.array([3.0, 4.0]))
        assert np.allclose(result, [4.0, 6.0])

    def test_create_symmetry_operators_custom(self):
        """Test create_symmetry_operators with a custom operator."""
        def flip(point):
            return -point
        ops = create_symmetry_operators(custom_operators=[flip])
        assert len(ops) == 1
        result = ops[0](np.array([1.0, -2.0]))
        assert np.allclose(result, [-1.0, 2.0]) 

    def test_plane_group_p1(self):
        """Test p1 plane group: translations only."""
        ops = create_symmetry_operators(translation_vectors=[np.array([1.0, 0.0]), np.array([0.0, 1.0])])
        assert len(ops) == 2
        p = np.array([0.5, 0.5])
        t1 = ops[0](p)
        t2 = ops[1](p)
        assert np.allclose(t1, [1.5, 0.5])
        assert np.allclose(t2, [0.5, 1.5])

    def test_plane_group_p2(self):
        """Test p2 plane group: translations + 180° rotation."""
        import math
        ops = create_symmetry_operators(
            rotation_angles=[math.pi],
            translation_vectors=[np.array([1.0, 0.0]), np.array([0.0, 1.0])]
        )
        assert len(ops) == 3
        p = np.array([0.5, 0.5])
        rot = ops[0](p)
        t1 = ops[1](p)
        t2 = ops[2](p)
        assert np.allclose(rot, [-0.5, -0.5])
        assert np.allclose(t1, [1.5, 0.5])
        assert np.allclose(t2, [0.5, 1.5])

    def test_plane_group_pm(self):
        """Test pm plane group: translations + mirror reflection."""
        def mirror_x(point):
            return np.array([point[0], -point[1]])
        ops = create_symmetry_operators(
            translation_vectors=[np.array([1.0, 0.0]), np.array([0.0, 1.0])],
            custom_operators=[mirror_x]
        )
        assert len(ops) == 3
        p = np.array([0.5, 0.5])
        mirror = ops[2](p)
        assert np.allclose(mirror, [0.5, -0.5])

    def test_plane_group_pg(self):
        """Test pg plane group: translations + glide reflection."""
        def glide_x(point):
            reflected = np.array([point[0], -point[1]])
            translated = reflected + np.array([0.5, 0.0])
            return translated
        ops = create_symmetry_operators(
            translation_vectors=[np.array([1.0, 0.0]), np.array([0.0, 1.0])],
            custom_operators=[glide_x]
        )
        assert len(ops) == 3
        p = np.array([0.5, 0.5])
        glide = ops[2](p)
        assert np.allclose(glide, [1.0, -0.5]) 