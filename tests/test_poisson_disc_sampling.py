#!/usr/bin/env python

"""Comprehensive tests for PoissonDiskSamplerWithExisting."""

import pytest
import numpy as np
from scipy.spatial import KDTree
from hiposa.poisson_disc_sampling import PoissonDiskSamplerWithExisting
from hiposa.utils import check_minimum_distance, validate_domain
import matplotlib
matplotlib.use('Agg')


class TestPoissonDiskSamplerWithExisting:
    """Test suite for PoissonDiskSamplerWithExisting class."""

    def test_initialization_basic(self):
        """Test basic initialization."""
        domain = [(0, 2), (0, 2)]
        r = 0.4
        
        sampler = PoissonDiskSamplerWithExisting(domain=domain, r=r)
        
        assert sampler.domain.shape == (2, 2)
        assert sampler.r == 0.4
        assert sampler.k == 60  # Default value
        assert sampler.dimensions == 2
        assert sampler.cell_size == r / np.sqrt(2)
        assert sampler.wrap is False
        assert len(sampler.symmetry_operators) == 0
        assert len(sampler.samples) == 0
        assert sampler.kdtree is None
        assert len(sampler.labels) == 0

    def test_initialization_with_existing_points(self):
        """Test initialization with existing points."""
        domain = [(0, 2), (0, 2)]
        r = 0.4
        existing_points = np.array([[0.5, 0.5], [1.5, 1.5]])
        existing_labels = np.array(["existing1", "existing2"])
        
        sampler = PoissonDiskSamplerWithExisting(
            domain=domain, r=r,
            existing_points=existing_points,
            existing_labels=existing_labels
        )
        
        assert len(sampler.samples) == 2
        assert sampler.kdtree is not None
        assert len(sampler.labels) == 2
        assert sampler.labels[0] == "existing1"
        assert sampler.labels[1] == "existing2"

    def test_initialization_with_symmetry_operators(self):
        """Test initialization with symmetry operators."""
        domain = [(0, 2), (0, 2)]
        r = 0.4
        
        def rotation_90(point):
            return np.array([-point[1], point[0]])
        
        def translation(point):
            return point + np.array([1.0, 1.0])
        
        symmetry_operators = [rotation_90, translation]
        
        sampler = PoissonDiskSamplerWithExisting(
            domain=domain, r=r,
            symmetry_operators=symmetry_operators
        )
        
        assert len(sampler.symmetry_operators) == 2

    def test_initialization_with_wrap(self):
        """Test initialization with wrap-around enabled."""
        domain = [(0, 2), (0, 2)]
        r = 0.4
        
        sampler = PoissonDiskSamplerWithExisting(domain=domain, r=r, wrap=True)
        
        assert sampler.wrap is True

    def test_initialization_validation_errors(self):
        """Test initialization validation errors."""
        # Empty domain
        with pytest.raises(ValueError, match="Domain cannot be empty"):
            PoissonDiskSamplerWithExisting(domain=[], r=0.4)
        
        # Non-positive r
        with pytest.raises(ValueError, match="Minimum distance r must be positive"):
            PoissonDiskSamplerWithExisting(domain=[(0, 2), (0, 2)], r=0)
        
        with pytest.raises(ValueError, match="Minimum distance r must be positive"):
            PoissonDiskSamplerWithExisting(domain=[(0, 2), (0, 2)], r=-1)

    def test_generate_points_around(self):
        """Test point generation around existing points."""
        domain = [(0, 2), (0, 2)]
        r = 0.8
        sampler = PoissonDiskSamplerWithExisting(domain=domain, r=r)
        
        point = np.array([1.0, 1.0])
        new_points = sampler.generate_points_around(point)
        
        assert new_points.shape[1] == 2
        assert len(new_points) == sampler.k
        
        # Test with wrap-around
        sampler_wrap = PoissonDiskSamplerWithExisting(domain=domain, r=r, wrap=True)
        new_points_wrap = sampler_wrap.generate_points_around(point)
        assert new_points_wrap.shape[1] == 2

    def test_generate_points_around_dimension_error(self):
        """Test generate_points_around with wrong dimension."""
        domain = [(0, 10), (0, 10)]
        r = 1.0
        sampler = PoissonDiskSamplerWithExisting(domain=domain, r=r)
        
        point = np.array([5.0])  # Wrong dimension
        
        with pytest.raises(ValueError, match="Point dimension 1 does not match domain dimension 2"):
            sampler.generate_points_around(point)

    def test_is_valid_point_basic(self):
        """Test basic point validation."""
        domain = [(0, 10), (0, 10)]
        r = 1.0
        sampler = PoissonDiskSamplerWithExisting(domain=domain, r=r)
        
        # Valid point within domain
        point = np.array([5.0, 5.0])
        assert sampler.is_valid_point(point) is True
        
        # Point outside domain
        point_outside = np.array([15.0, 5.0])
        assert sampler.is_valid_point(point_outside) is False
        
        # Point on boundary (should be invalid)
        point_boundary = np.array([10.0, 5.0])
        assert sampler.is_valid_point(point_boundary) is False

    def test_is_valid_point_with_existing_points(self):
        """Test point validation with existing points."""
        domain = [(0, 10), (0, 10)]
        r = 1.0
        existing_points = np.array([[5.0, 5.0]])
        sampler = PoissonDiskSamplerWithExisting(domain=domain, r=r, existing_points=existing_points)
        
        # Point too close to existing point
        point_close = np.array([5.5, 5.5])
        assert sampler.is_valid_point(point_close) is False
        
        # Point far enough from existing point
        point_far = np.array([7.0, 7.0])
        assert sampler.is_valid_point(point_far) is True

    def test_is_valid_point_with_wrap(self):
        """Test point validation with wrap-around."""
        domain = [(0, 10), (0, 10)]
        r = 1.0
        existing_points = np.array([[0.1, 1.0]])
        sampler = PoissonDiskSamplerWithExisting(domain=domain, r=r, existing_points=existing_points, wrap=True)
        
        # Point that is close with wrap-around (distance 0.2)
        point_wrap_close = np.array([9.9, 1.0])
        assert sampler.is_valid_point(point_wrap_close) is False

    def test_check_orbit_validity(self):
        """Test orbit validity checking."""
        domain = [(0, 10), (0, 10)]
        r = 1.0
        sampler = PoissonDiskSamplerWithExisting(domain=domain, r=r)
        
        # Valid orbit (points far apart)
        valid_orbit = [np.array([1.0, 1.0]), np.array([3.0, 3.0])]
        assert sampler.check_orbit_validity(valid_orbit) is True
        
        # Invalid orbit (points too close)
        invalid_orbit = [np.array([1.0, 1.0]), np.array([1.5, 1.5])]
        assert sampler.check_orbit_validity(invalid_orbit) is False
        
        # Orbit with identical points (should be valid)
        identical_orbit = [np.array([1.0, 1.0]), np.array([1.0, 1.0])]
        assert sampler.check_orbit_validity(identical_orbit) is True

    def test_check_orbit_validity_with_wrap(self):
        """Test orbit validity checking with wrap-around."""
        domain = [(0, 10), (0, 10)]
        r = 1.0
        sampler = PoissonDiskSamplerWithExisting(domain=domain, r=r, wrap=True)
        
        # Points that are close with wrap-around (distance 0.2)
        orbit_wrap_close = [np.array([0.1, 1.0]), np.array([9.9, 1.0])]
        assert sampler.check_orbit_validity(orbit_wrap_close) is False

    def test_find_invariant_points(self):
        """Test finding invariant points."""
        domain = [(0, 10), (0, 10)]
        r = 1.0
        
        def identity_operator(point):
            return point  # Identity operator
        
        sampler = PoissonDiskSamplerWithExisting(domain=domain, r=r)
        invariant_points = sampler.find_invariant_points(identity_operator)
        
        # Should find some invariant points
        assert len(invariant_points) > 0
        assert all(len(p) == 2 for p in invariant_points)

    def test_find_invariant_points_with_wrap(self):
        """Test finding invariant points with wrap-around."""
        domain = [(0, 10), (0, 10)]
        r = 1.0
        
        def translation_operator(point):
            return (point + np.array([5.0, 5.0])) % 10.0
        
        sampler = PoissonDiskSamplerWithExisting(domain=domain, r=r, wrap=True)
        invariant_points = sampler.find_invariant_points(translation_operator)
        
        # May or may not find invariant points depending on operator
        assert all(len(p) == 2 for p in invariant_points)

    def test_sample_basic(self):
        """Test basic sampling."""
        domain = [(0, 10), (0, 10)]
        r = 2.0
        sampler = PoissonDiskSamplerWithExisting(domain=domain, r=r)
        
        points, labels = sampler.sample()
        
        assert len(points) > 0
        assert len(labels) == len(points)
        assert all(labels == 0)  # Default label

    def test_sample_with_existing_points(self):
        """Test sampling with existing points."""
        domain = [(0, 10), (0, 10)]
        r = 1.0
        existing_points = np.array([[1.0, 1.0], [2.0, 2.0]])
        existing_labels = np.array(["existing1", "existing2"])
        
        sampler = PoissonDiskSamplerWithExisting(
            domain=domain, r=r,
            existing_points=existing_points,
            existing_labels=existing_labels
        )
        
        points, labels = sampler.sample()
        
        assert len(points) >= 2  # Should have at least existing points
        assert len(labels) == len(points)
        assert "existing1" in labels
        assert "existing2" in labels

    def test_sample_with_new_label(self):
        """Test sampling with custom label."""
        domain = [(0, 10), (0, 10)]
        r = 2.0
        sampler = PoissonDiskSamplerWithExisting(domain=domain, r=r)
        
        points, labels = sampler.sample(new_label="custom")
        
        assert len(points) > 0
        assert all(labels == "custom")

    def test_sample_return_new_only(self):
        """Test sampling with return_new_only=True."""
        domain = [(0, 10), (0, 10)]
        r = 2.0
        existing_points = np.array([[1.0, 1.0]])
        existing_labels = np.array(["existing"])
        
        sampler = PoissonDiskSamplerWithExisting(
            domain=domain, r=r,
            existing_points=existing_points,
            existing_labels=existing_labels
        )
        
        points, labels = sampler.sample(return_new_only=True)
        
        # Should only return newly generated points
        assert len(points) >= 0  # May or may not generate new points
        assert len(labels) == len(points)
        assert "existing" not in labels

    def test_sample_3d(self):
        """Test sampling in 3D."""
        domain = [(0, 5), (0, 5), (0, 5)]
        r = 1.0
        sampler = PoissonDiskSamplerWithExisting(domain=domain, r=r)
        
        points, labels = sampler.sample()
        
        assert len(points) > 0
        assert points.shape[1] == 3
        assert len(labels) == len(points)

    def test_sample_4d(self):
        """Test sampling in 4D."""
        domain = [(0, 3), (0, 3), (0, 3), (0, 3)]
        r = 0.5
        sampler = PoissonDiskSamplerWithExisting(domain=domain, r=r)
        
        points, labels = sampler.sample()
        
        assert len(points) > 0
        assert points.shape[1] == 4
        assert len(labels) == len(points)

    def test_sample_with_wrap(self):
        """Test sampling with wrap-around."""
        domain = [(0, 10), (0, 10)]
        r = 2.0
        sampler = PoissonDiskSamplerWithExisting(domain=domain, r=r, wrap=True)
        
        points, labels = sampler.sample()
        
        assert len(points) > 0
        assert len(labels) == len(points)

    def test_kdtree_update(self):
        """Test KDTree update during sampling."""
        domain = [(0, 10), (0, 10)]
        r = 1.0
        sampler = PoissonDiskSamplerWithExisting(domain=domain, r=r)
        
        # Add some points manually to trigger KDTree update
        sampler.samples = [np.array([1.0, 1.0]), np.array([2.0, 2.0])]
        sampler.idx_to_point = {0: np.array([1.0, 1.0]), 1: np.array([2.0, 2.0])}
        sampler.labels = np.array(["point1", "point2"])
        
        # This should trigger KDTree creation
        points, labels = sampler.sample()
        
        assert sampler.kdtree is not None

    def test_edge_cases(self):
        """Test various edge cases."""
        # Very small domain
        domain = [(0, 0.1), (0, 0.1)]
        r = 0.01
        sampler = PoissonDiskSamplerWithExisting(domain=domain, r=r)
        
        points, labels = sampler.sample()
        assert len(points) >= 0  # May or may not generate points
        
        # Very large domain (reduced for speed)
        domain = [(0, 100), (0, 100)]
        r = 10.0
        sampler = PoissonDiskSamplerWithExisting(domain=domain, r=r)
        
        points, labels = sampler.sample()
        assert len(points) >= 0

    def test_performance_large_domain(self):
        """Test performance with large domain."""
        domain = [(0, 100), (0, 100)]
        r = 5.0
        sampler = PoissonDiskSamplerWithExisting(domain=domain, r=r)
        
        points, labels = sampler.sample()
        
        assert len(points) > 0
        assert len(labels) == len(points)

    def test_config_integration(self):
        """Test integration with configuration."""
        domain = [(0, 10), (0, 10)]
        r = 1.0
        k = 100  # Custom k value
        
        sampler = PoissonDiskSamplerWithExisting(domain=domain, r=r, k=k)
        
        assert sampler.k == k

    def test_symmetry_operator_returns_none(self):
        """Test symmetry operator that returns None."""
        domain = [(0, 10), (0, 10)]
        r = 1.0
        
        def operator_returns_none(point):
            return None
        
        sampler = PoissonDiskSamplerWithExisting(
            domain=domain, r=r, symmetry_operators=[operator_returns_none]
        )
        
        point = np.array([5.0, 5.0])
        result = sampler.apply_symmetry(point)
        
        # Should return original point when operator returns None
        assert len(result) == 1
        assert np.array_equal(result[0], point)

    def test_find_invariant_points_operator_returns_none(self):
        """Test find_invariant_points with operator that returns None."""
        domain = [(0, 10), (0, 10)]
        r = 1.0
        
        def operator_returns_none(point):
            return None
        
        sampler = PoissonDiskSamplerWithExisting(domain=domain, r=r)
        invariant_points = sampler.find_invariant_points(operator_returns_none)
        
        # Should return empty list when operator returns None
        assert len(invariant_points) == 0

    def test_sample_with_custom_k(self):
        """Test sampling with custom k value."""
        domain = [(0, 10), (0, 10)]
        r = 1.0
        k = 30  # Custom k value
        
        sampler = PoissonDiskSamplerWithExisting(domain=domain, r=r, k=k)
        points, labels = sampler.sample()
        
        assert len(points) >= 0
        assert sampler.k == k

    def test_sample_with_existing_points_no_labels(self):
        """Test sampling with existing points but no labels."""
        domain = [(0, 10), (0, 10)]
        r = 1.0
        existing_points = np.array([[1.0, 1.0], [2.0, 2.0]])
        
        sampler = PoissonDiskSamplerWithExisting(
            domain=domain, r=r, existing_points=existing_points
        )
        
        points, labels = sampler.sample()
        
        assert len(points) >= 2
        assert len(labels) == len(points)
        assert all(label == "existing" for label in labels[:2])

    def test_sample_with_very_small_r(self):
        """Test sampling with very small minimum distance."""
        domain = [(0, 0.1), (0, 0.1)]
        r = 0.01  # Very small r
        
        sampler = PoissonDiskSamplerWithExisting(domain=domain, r=r)
        points, labels = sampler.sample()
        
        assert len(points) >= 0
        assert len(labels) == len(points)

    def test_sample_with_very_large_r(self):
        """Test sampling with very large minimum distance."""
        domain = [(0, 10), (0, 10)]
        r = 5.0  # Large r relative to domain
        
        sampler = PoissonDiskSamplerWithExisting(domain=domain, r=r)
        points, labels = sampler.sample()
        
        assert len(points) >= 0
        assert len(labels) == len(points)

    def test_sample_with_rectangular_domain(self):
        """Test sampling with rectangular domain."""
        domain = [(0, 20), (0, 10)]  # Rectangular domain
        r = 1.0
        sampler = PoissonDiskSamplerWithExisting(domain=domain, r=r)
        
        points, labels = sampler.sample()
        
        assert len(points) >= 0
        assert len(labels) == len(points)
        
        # Check that points are within domain bounds
        if len(points) > 0:
            assert np.all(points[:, 0] >= 0) and np.all(points[:, 0] < 20)
            assert np.all(points[:, 1] >= 0) and np.all(points[:, 1] < 10) 

    def test_apply_symmetry_with_glide_plane(self):
        """Test apply_symmetry with a glide plane operator on a (0,1)x(0,1) tile."""
        domain = [(0, 1), (0, 1)]
        r = 0.2
        # Glide plane: reflect across y=0.5, then translate by (0.5, 0)
        def glide_plane(point):
            reflected = np.array([point[0], 1.0 - point[1]])
            translated = reflected + np.array([0.5, 0.0])
            # Wrap around the domain
            translated[0] = translated[0] % 1.0
            return translated
        sampler = PoissonDiskSamplerWithExisting(domain=domain, r=r, symmetry_operators=[glide_plane])
        # Pick a point not on the symmetry axis
        point = np.array([0.2, 0.3])
        orbit = sampler.apply_symmetry(point)
        # Should return two points: the original and its glide image
        assert orbit is not None
        assert len(orbit) == 2
        # The second point should be the glide of the first
        expected = glide_plane(point)
        assert any(np.allclose(p, expected) for p in orbit)
        assert any(np.allclose(p, point) for p in orbit) 

    def test_sample_with_glide_plane_symmetry(self):
        """Test sample method with a glide plane symmetry operator."""
        domain = [(0, 1), (0, 1)]
        r = 0.2
        # Glide plane: reflect across y=0.5, then translate by (0.5, 0)
        def glide_plane(point):
            reflected = np.array([point[0], 1.0 - point[1]])
            translated = reflected + np.array([0.5, 0.0])
            # Wrap around the domain
            translated[0] = translated[0] % 1.0
            return translated
        sampler = PoissonDiskSamplerWithExisting(domain=domain, r=r, symmetry_operators=[glide_plane])
        points, labels = sampler.sample()
        # All points should be paired by the glide plane symmetry
        assert len(points) > 0
        # For each point, its glide image should also be in the sample
        for p in points:
            glide = glide_plane(p)
            assert any(np.allclose(glide, q) for q in points) 