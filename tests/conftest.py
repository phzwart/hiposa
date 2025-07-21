#!/usr/bin/env python

"""Pytest configuration for HiPoSa tests."""

import pytest
import numpy as np
import tempfile
import os


@pytest.fixture
def sample_2d_domain():
    """Fixture for a 2D domain."""
    return [(0, 10), (0, 10)]


@pytest.fixture
def sample_3d_domain():
    """Fixture for a 3D domain."""
    return [(0, 5), (0, 5), (0, 5)]


@pytest.fixture
def sample_4d_domain():
    """Fixture for a 4D domain."""
    return [(0, 3), (0, 3), (0, 3), (0, 3)]


@pytest.fixture
def sample_points_2d():
    """Fixture for 2D sample points."""
    return np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])


@pytest.fixture
def sample_points_3d():
    """Fixture for 3D sample points."""
    return np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]])


@pytest.fixture
def sample_symmetry_operators():
    """Fixture for sample symmetry operators."""
    def rotation_90(point):
        return np.array([-point[1], point[0]])
    
    def translation(point):
        return point + np.array([5.0, 5.0])
    
    return [rotation_90, translation]


@pytest.fixture
def sample_function():
    """Fixture for a sample function."""
    def f_function(point):
        return point[0] + point[1]
    
    return f_function


@pytest.fixture
def sample_grid():
    """Fixture for sample grid data."""
    grid_x = np.linspace(0, 1, 10)
    grid_y = np.linspace(0, 1, 10)
    f_gt = np.random.random((10, 10))
    
    return grid_x, grid_y, f_gt


@pytest.fixture
def temp_dir():
    """Fixture for temporary directory."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield tmp_dir


@pytest.fixture
def sample_point_selector_data():
    """Fixture for PointSelector test data."""
    xy = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
    levels = np.array([0, 0, 1, 1])
    scales = np.array([1.0, 1.0, 0.5, 0.5])
    
    def f_function(point):
        return point[0] + point[1]
    
    grid_x = np.linspace(0, 1, 10)
    grid_y = np.linspace(0, 1, 10)
    f_gt = np.zeros((10, 10))
    
    return {
        'xy': xy,
        'levels': levels,
        'scales': scales,
        'f_function': f_function,
        'grid_x': grid_x,
        'grid_y': grid_y,
        'f_gt': f_gt
    }


@pytest.fixture(scope="session")
def large_test_data():
    """Fixture for large test data (session scope for performance)."""
    np.random.seed(42)
    
    # Large domain
    domain = [(0, 100), (0, 100)]
    
    # Large number of points
    n_points = 1000
    points = np.random.random((n_points, 2)) * 100
    
    # Large grid
    grid_x = np.linspace(0, 100, 50)
    grid_y = np.linspace(0, 100, 50)
    f_gt = np.random.random((50, 50))
    
    return {
        'domain': domain,
        'points': points,
        'grid_x': grid_x,
        'grid_y': grid_y,
        'f_gt': f_gt
    }


def pytest_configure(config):
    """Configure pytest."""
    # Add custom markers
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection."""
    for item in items:
        # Mark integration tests
        if "integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)
        
        # Mark slow tests based on test name
        if any(keyword in item.nodeid.lower() for keyword in 
               ["performance", "large", "memory", "thread"]):
            item.add_marker(pytest.mark.slow)
        
        # Mark unit tests
        if "test_" in item.nodeid and "integration" not in item.nodeid:
            item.add_marker(pytest.mark.unit) 