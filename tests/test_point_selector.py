#!/usr/bin/env python

"""Comprehensive tests for PointSelector."""

import pytest
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CloughTocher2DInterpolator
from hiposa.point_selector import PointSelector
from hiposa.utils import validate_points


class TestPointSelector:
    """Test suite for PointSelector class."""

    def test_initialization_basic(self):
        """Test basic initialization."""
        xy = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
        levels = np.array([0, 0, 1, 1])
        scales = np.array([1.0, 1.0, 0.5, 0.5])
        
        def f_function(point):
            return point[0] + point[1]
        
        grid_x = np.linspace(0, 1, 10)
        grid_y = np.linspace(0, 1, 10)
        f_gt = np.zeros((10, 10))
        
        selector = PointSelector(
            xy=xy, levels=levels, scales=scales,
            f_function=f_function,
            grid_x=grid_x, grid_y=grid_y, f_gt=f_gt
        )
        
        assert selector.xy.shape == (4, 2)
        assert len(selector.levels) == 4
        assert len(selector.scales) == 4
        assert selector.tau == 75.0  # Default value
        assert selector.sign == 1  # Default value
        assert selector.eps == 0.0  # Default value
        assert selector.start_level == 0  # Default value
        assert selector.set_aside == 0.5  # Default value

    def test_initialization_with_custom_parameters(self):
        """Test initialization with custom parameters."""
        xy = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
        levels = np.array([0, 0, 1, 1])
        scales = np.array([1.0, 1.0, 0.5, 0.5])
        
        def f_function(point):
            return point[0] + point[1]
        
        grid_x = np.linspace(0, 1, 10)
        grid_y = np.linspace(0, 1, 10)
        f_gt = np.zeros((10, 10))
        
        selector = PointSelector(
            xy=xy, levels=levels, scales=scales,
            f_function=f_function,
            grid_x=grid_x, grid_y=grid_y, f_gt=f_gt,
            tau=80.0, sign=-1, eps=0.1, start_level=1, set_aside=0.3
        )
        
        assert selector.tau == 80.0
        assert selector.sign == -1
        assert selector.eps == 0.1
        assert selector.start_level == 1
        assert selector.set_aside == 0.3

    def test_initialization_with_config(self):
        """Test initialization with config integration."""
        xy = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
        levels = np.array([0, 0, 1, 1])
        scales = np.array([1.0, 1.0, 0.5, 0.5])
        
        def f_function(point):
            return point[0] + point[1]
        
        grid_x = np.linspace(0, 1, 10)
        grid_y = np.linspace(0, 1, 10)
        f_gt = np.zeros((10, 10))
        
        # Test with None parameters to use config defaults
        selector = PointSelector(
            xy=xy, levels=levels, scales=scales,
            f_function=f_function,
            grid_x=grid_x, grid_y=grid_y, f_gt=f_gt,
            tau=None, sign=None, eps=None, start_level=None, set_aside=None
        )
        
        # Should use config defaults
        assert selector.tau is not None
        assert selector.sign is not None
        assert selector.eps is not None
        assert selector.start_level is not None
        assert selector.set_aside is not None

    def test_initialization_parameter_clamping(self):
        """Test parameter clamping during initialization."""
        xy = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
        levels = np.array([0, 0, 1, 1])
        scales = np.array([1.0, 1.0, 0.5, 0.5])
        
        def f_function(point):
            return point[0] + point[1]
        
        grid_x = np.linspace(0, 1, 10)
        grid_y = np.linspace(0, 1, 10)
        f_gt = np.zeros((10, 10))
        
        # Test border_bias clamping
        selector = PointSelector(
            xy=xy, levels=levels, scales=scales,
            f_function=f_function,
            grid_x=grid_x, grid_y=grid_y, f_gt=f_gt,
            border_bias=2.0  # Should be clamped to 1.0
        )
        
        assert selector.border_bias == 1.0
        
        # Test radius_scale clamping
        selector2 = PointSelector(
            xy=xy, levels=levels, scales=scales,
            f_function=f_function,
            grid_x=grid_x, grid_y=grid_y, f_gt=f_gt,
            radius_scale=0.5  # Should be clamped to 1.0
        )
        
        assert selector2.radius_scale == 1.0

    def test_interpolate_sparse_data(self):
        """Test sparse data interpolation."""
        # Use at least three non-collinear points for 2D interpolation
        xy = np.array([[0, 0], [1, 0], [0, 1]])
        levels = np.array([0, 0, 1])
        scales = np.array([1.0, 1.0, 0.5])

        def f_function(point):
            return point[0] + point[1]

        grid_x = np.linspace(0, 1, 10)
        grid_y = np.linspace(0, 1, 10)
        f_gt = np.zeros((10, 10))

        selector = PointSelector(
            xy=xy, levels=levels, scales=scales,
            f_function=f_function,
            grid_x=grid_x, grid_y=grid_y, f_gt=f_gt
        )

        # Test interpolation
        these_xy = np.array([[0, 0], [1, 0], [0, 1]])
        these_values = np.array([0.0, 1.0, 1.0])

        result = selector.interpolate_sparse_data(these_xy, these_values)

        assert result.shape == (10, 10)
        # Allow some NaN, but require at least some finite values
        assert np.any(np.isfinite(result))

    def test_interpolate_sparse_data_with_nan(self):
        """Test interpolation with NaN values."""
        xy = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
        levels = np.array([0, 0, 1, 1])
        scales = np.array([1.0, 1.0, 0.5, 0.5])
        
        def f_function(point):
            return point[0] + point[1]
        
        grid_x = np.linspace(0, 1, 10)
        grid_y = np.linspace(0, 1, 10)
        f_gt = np.zeros((10, 10))
        
        selector = PointSelector(
            xy=xy, levels=levels, scales=scales,
            f_function=f_function,
            grid_x=grid_x, grid_y=grid_y, f_gt=f_gt
        )
        
        # Test with NaN values and only one point
        these_xy = np.array([[0.5, 0.5]])
        these_values = np.array([np.nan])
        
        result = selector.interpolate_sparse_data(these_xy, these_values, fill_value=0.0)
        
        assert result.shape == (10, 10)
        # Should be filled with fill_value (0.0) if not enough points
        assert np.all(result == 0.0)

    def test_get_distance_transform(self):
        """Test distance transform calculation."""
        # Create a simple heatmap
        heatmap = np.zeros((10, 10))
        heatmap[4:6, 4:6] = 1.0  # Create a 2x2 region
        
        threshold = 0.5
        distance_transform = PointSelector.get_distance_transform(heatmap, threshold)
        
        assert distance_transform.shape == (10, 10)
        assert np.all(distance_transform >= 0)  # Distances should be non-negative

    def test_get_quantiles(self):
        """Test quantile calculation."""
        data = np.random.random(100)
        tau = 75.0
        n_splits = 5
        
        quantiles = PointSelector.get_quantiles(data, tau, n_splits)
        
        assert len(quantiles) > 0
        assert all(isinstance(q, float) for q in quantiles)

    def test_get_quantiles_small_data(self):
        """Test quantile calculation with small dataset."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        tau = 80.0
        n_splits = 3
        
        quantiles = PointSelector.get_quantiles(data, tau, n_splits)
        
        assert len(quantiles) > 0

    def test_evaluate_points(self):
        """Test point evaluation."""
        xy = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
        levels = np.array([0, 0, 1, 1])
        scales = np.array([1.0, 1.0, 0.5, 0.5])
        
        def f_function(point):
            return point[0] + point[1]
        
        grid_x = np.linspace(0, 1, 10)
        grid_y = np.linspace(0, 1, 10)
        f_gt = np.zeros((10, 10))
        
        selector = PointSelector(
            xy=xy, levels=levels, scales=scales,
            f_function=f_function,
            grid_x=grid_x, grid_y=grid_y, f_gt=f_gt
        )
        
        points = np.array([[0.5, 0.5], [0.25, 0.75]])
        values = selector.evaluate_points(points)
        
        assert len(values) == 2
        assert values[0] == 1.0  # 0.5 + 0.5
        assert values[1] == 1.0  # 0.25 + 0.75

    def test_compute_threshold(self):
        """Test threshold computation."""
        xy = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
        levels = np.array([0, 0, 1, 1])
        scales = np.array([1.0, 1.0, 0.5, 0.5])
        
        def f_function(point):
            return point[0] + point[1]
        
        grid_x = np.linspace(0, 1, 10)
        grid_y = np.linspace(0, 1, 10)
        f_gt = np.zeros((10, 10))
        
        selector = PointSelector(
            xy=xy, levels=levels, scales=scales,
            f_function=f_function,
            grid_x=grid_x, grid_y=grid_y, f_gt=f_gt
        )
        
        work_xy = np.array([[0.5, 0.5], [0.25, 0.75]])
        work_f_values = np.array([1.0, 1.0])
        cal_f_values = np.array([0.5, 1.5])
        
        threshold, grid_values, mean_work, mean_cal, std_cal = selector.compute_threshold(
            work_xy, work_f_values, cal_f_values
        )
        
        assert isinstance(threshold, float)
        assert grid_values.shape == (10, 10)
        assert isinstance(mean_work, float)
        assert isinstance(mean_cal, float)
        assert isinstance(std_cal, float)

    def test_select_points_at_level(self):
        """Test point selection at specific level."""
        xy = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
        levels = np.array([0, 0, 1, 1])
        scales = np.array([1.0, 1.0, 0.5, 0.5])
        
        def f_function(point):
            return point[0] + point[1]
        
        grid_x = np.linspace(0, 1, 10)
        grid_y = np.linspace(0, 1, 10)
        f_gt = np.zeros((10, 10))
        
        selector = PointSelector(
            xy=xy, levels=levels, scales=scales,
            f_function=f_function,
            grid_x=grid_x, grid_y=grid_y, f_gt=f_gt
        )
        
        level = 1
        threshold = 0.5
        grid_values = np.random.random((10, 10))
        
        selected_points = selector.select_points_at_level(level, threshold, grid_values)
        
        assert isinstance(selected_points, np.ndarray)
        assert selected_points.shape[1] == 2  # Should be 2D points

    def test_select_points_at_level_with_border(self):
        """Test point selection with border bias."""
        xy = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
        levels = np.array([0, 0, 1, 1])
        scales = np.array([1.0, 1.0, 0.5, 0.5])
        
        def f_function(point):
            return point[0] + point[1]
        
        grid_x = np.linspace(0, 1, 10)
        grid_y = np.linspace(0, 1, 10)
        f_gt = np.zeros((10, 10))
        
        selector = PointSelector(
            xy=xy, levels=levels, scales=scales,
            f_function=f_function,
            grid_x=grid_x, grid_y=grid_y, f_gt=f_gt,
            border_bias=0.8  # High border bias
        )
        
        level = 1
        threshold = 0.5
        grid_values = np.random.random((10, 10))
        
        selected_points = selector.select_points_at_level(level, threshold, grid_values)
        
        assert isinstance(selected_points, np.ndarray)

    def test_plot_results(self):
        """Test plotting functionality."""
        xy = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
        levels = np.array([0, 0, 1, 1])
        scales = np.array([1.0, 1.0, 0.5, 0.5])
        
        def f_function(point):
            return point[0] + point[1]
        
        grid_x = np.linspace(0, 1, 10)
        grid_y = np.linspace(0, 1, 10)
        f_gt = np.zeros((10, 10))
        
        selector = PointSelector(
            xy=xy, levels=levels, scales=scales,
            f_function=f_function,
            grid_x=grid_x, grid_y=grid_y, f_gt=f_gt
        )
        
        these_xy = np.array([[0.5, 0.5]])
        new_ones = np.array([[0.25, 0.75]])
        threshold = 0.5
        surface = np.random.random((10, 10))
        mask = np.random.random((10, 10)) > 0.5
        
        # Test plotting (should not raise exception)
        try:
            selector.plot_results(these_xy, new_ones, threshold, surface, mask)
            plt.close()  # Close the plot to free memory
        except Exception as e:
            pytest.fail(f"Plotting failed: {e}")

    def test_plot_results_with_title(self):
        """Test plotting with custom title."""
        xy = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
        levels = np.array([0, 0, 1, 1])
        scales = np.array([1.0, 1.0, 0.5, 0.5])
        
        def f_function(point):
            return point[0] + point[1]
        
        grid_x = np.linspace(0, 1, 10)
        grid_y = np.linspace(0, 1, 10)
        f_gt = np.zeros((10, 10))
        
        selector = PointSelector(
            xy=xy, levels=levels, scales=scales,
            f_function=f_function,
            grid_x=grid_x, grid_y=grid_y, f_gt=f_gt
        )
        
        these_xy = np.array([[0.5, 0.5]])
        new_ones = np.array([[0.25, 0.75]])
        threshold = 0.5
        surface = np.random.random((10, 10))
        mask = np.random.random((10, 10)) > 0.5
        
        # Test plotting with title
        try:
            selector.plot_results(these_xy, new_ones, threshold, surface, mask, 
                               title="Test Plot", level=1)
            plt.close()
        except Exception as e:
            pytest.fail(f"Plotting with title failed: {e}")

    def test_run_basic(self):
        """Test basic run functionality."""
        xy = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
        levels = np.array([0, 0, 1, 1])
        scales = np.array([1.0, 1.0, 0.5, 0.5])
        
        def f_function(point):
            return point[0] + point[1]
        
        grid_x = np.linspace(0, 1, 10)
        grid_y = np.linspace(0, 1, 10)
        f_gt = np.zeros((10, 10))
        
        selector = PointSelector(
            xy=xy, levels=levels, scales=scales,
            f_function=f_function,
            grid_x=grid_x, grid_y=grid_y, f_gt=f_gt
        )
        
        selected_mask = selector.run(max_level=2)
        selected_points = xy[selected_mask]

        assert isinstance(selected_points, np.ndarray)
        if len(selected_points) > 0:
            assert selected_points.shape[1] == 2

    def test_run_with_custom_max_level(self):
        """Test run with custom max_level."""
        xy = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
        levels = np.array([0, 0, 1, 1])
        scales = np.array([1.0, 1.0, 0.5, 0.5])
        
        def f_function(point):
            return point[0] + point[1]
        
        grid_x = np.linspace(0, 1, 10)
        grid_y = np.linspace(0, 1, 10)
        f_gt = np.zeros((10, 10))
        
        selector = PointSelector(
            xy=xy, levels=levels, scales=scales,
            f_function=f_function,
            grid_x=grid_x, grid_y=grid_y, f_gt=f_gt
        )
        
        selected_points = selector.run(max_level=5)
        
        assert isinstance(selected_points, np.ndarray)

    def test_run_with_negative_sign(self):
        """Test run with negative sign (less than threshold)."""
        xy = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
        levels = np.array([0, 0, 1, 1])
        scales = np.array([1.0, 1.0, 0.5, 0.5])
        
        def f_function(point):
            return point[0] + point[1]
        
        grid_x = np.linspace(0, 1, 10)
        grid_y = np.linspace(0, 1, 10)
        f_gt = np.zeros((10, 10))
        
        selector = PointSelector(
            xy=xy, levels=levels, scales=scales,
            f_function=f_function,
            grid_x=grid_x, grid_y=grid_y, f_gt=f_gt,
            sign=-1  # Less than threshold
        )
        
        selected_points = selector.run(max_level=2)
        
        assert isinstance(selected_points, np.ndarray)

    def test_run_with_custom_set_aside(self):
        """Test run with custom set_aside fraction."""
        xy = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
        levels = np.array([0, 0, 1, 1])
        scales = np.array([1.0, 1.0, 0.5, 0.5])
        
        def f_function(point):
            return point[0] + point[1]
        
        grid_x = np.linspace(0, 1, 10)
        grid_y = np.linspace(0, 1, 10)
        f_gt = np.zeros((10, 10))
        
        selector = PointSelector(
            xy=xy, levels=levels, scales=scales,
            f_function=f_function,
            grid_x=grid_x, grid_y=grid_y, f_gt=f_gt,
            set_aside=0.3  # Custom set_aside
        )
        
        selected_points = selector.run(max_level=2)
        
        assert isinstance(selected_points, np.ndarray)

    def test_run_with_border_bias(self):
        """Test run with border bias."""
        xy = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
        levels = np.array([0, 0, 1, 1])
        scales = np.array([1.0, 1.0, 0.5, 0.5])
        
        def f_function(point):
            return point[0] + point[1]
        
        grid_x = np.linspace(0, 1, 10)
        grid_y = np.linspace(0, 1, 10)
        f_gt = np.zeros((10, 10))
        
        selector = PointSelector(
            xy=xy, levels=levels, scales=scales,
            f_function=f_function,
            grid_x=grid_x, grid_y=grid_y, f_gt=f_gt,
            border_bias=0.7  # High border bias
        )
        
        selected_points = selector.run(max_level=2)
        
        assert isinstance(selected_points, np.ndarray)

    def test_run_with_custom_border(self):
        """Test run with custom border size."""
        xy = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
        levels = np.array([0, 0, 1, 1])
        scales = np.array([1.0, 1.0, 0.5, 0.5])
        
        def f_function(point):
            return point[0] + point[1]
        
        grid_x = np.linspace(0, 1, 10)
        grid_y = np.linspace(0, 1, 10)
        f_gt = np.zeros((10, 10))
        
        selector = PointSelector(
            xy=xy, levels=levels, scales=scales,
            f_function=f_function,
            grid_x=grid_x, grid_y=grid_y, f_gt=f_gt,
            border=8  # Custom border size
        )
        
        selected_points = selector.run(max_level=2)
        
        assert isinstance(selected_points, np.ndarray)

    def test_run_with_custom_radius_scale(self):
        """Test run with custom radius scale."""
        xy = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
        levels = np.array([0, 0, 1, 1])
        scales = np.array([1.0, 1.0, 0.5, 0.5])
        
        def f_function(point):
            return point[0] + point[1]
        
        grid_x = np.linspace(0, 1, 10)
        grid_y = np.linspace(0, 1, 10)
        f_gt = np.zeros((10, 10))
        
        selector = PointSelector(
            xy=xy, levels=levels, scales=scales,
            f_function=f_function,
            grid_x=grid_x, grid_y=grid_y, f_gt=f_gt,
            radius_scale=6.0  # Custom radius scale
        )
        
        selected_points = selector.run(max_level=2)
        
        assert isinstance(selected_points, np.ndarray)

    def test_run_with_custom_quantile_parameters(self):
        """Test run with custom quantile parameters."""
        xy = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
        levels = np.array([0, 0, 1, 1])
        scales = np.array([1.0, 1.0, 0.5, 0.5])
        
        def f_function(point):
            return point[0] + point[1]
        
        grid_x = np.linspace(0, 1, 10)
        grid_y = np.linspace(0, 1, 10)
        f_gt = np.zeros((10, 10))
        
        selector = PointSelector(
            xy=xy, levels=levels, scales=scales,
            f_function=f_function,
            grid_x=grid_x, grid_y=grid_y, f_gt=f_gt,
            lower_tau_quantile=0.8,
            n_splits_quantile=8
        )
        
        selected_points = selector.run(max_level=2)
        
        assert isinstance(selected_points, np.ndarray)

    def test_run_with_plotting_bounds(self):
        """Test run with plotting bounds."""
        xy = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
        levels = np.array([0, 0, 1, 1])
        scales = np.array([1.0, 1.0, 0.5, 0.5])
        
        def f_function(point):
            return point[0] + point[1]
        
        grid_x = np.linspace(0, 1, 10)
        grid_y = np.linspace(0, 1, 10)
        f_gt = np.zeros((10, 10))
        
        selector = PointSelector(
            xy=xy, levels=levels, scales=scales,
            f_function=f_function,
            grid_x=grid_x, grid_y=grid_y, f_gt=f_gt,
            lower=0.0, upper=2.0  # Custom plotting bounds
        )
        
        selected_points = selector.run(max_level=2)
        
        assert isinstance(selected_points, np.ndarray)

    def test_edge_cases(self):
        """Test various edge cases."""
        # Single point
        xy = np.array([[0.5, 0.5]])
        levels = np.array([0])
        scales = np.array([1.0])
        
        def f_function(point):
            return point[0] + point[1]
        
        grid_x = np.linspace(0, 1, 5)
        grid_y = np.linspace(0, 1, 5)
        f_gt = np.zeros((5, 5))
        
        selector = PointSelector(
            xy=xy, levels=levels, scales=scales,
            f_function=f_function,
            grid_x=grid_x, grid_y=grid_y, f_gt=f_gt
        )
        
        selected_points = selector.run(max_level=1)
        assert isinstance(selected_points, np.ndarray)

    def test_edge_cases_large_dataset(self):
        """Test with larger dataset."""
        np.random.seed(42)
        n_points = 50
        xy = np.random.random((n_points, 2))
        levels = np.random.randint(0, 3, n_points)
        scales = np.random.random(n_points)
        
        def f_function(point):
            return np.sin(point[0]) + np.cos(point[1])
        
        grid_x = np.linspace(0, 1, 20)
        grid_y = np.linspace(0, 1, 20)
        f_gt = np.random.random((20, 20))
        
        selector = PointSelector(
            xy=xy, levels=levels, scales=scales,
            f_function=f_function,
            grid_x=grid_x, grid_y=grid_y, f_gt=f_gt
        )
        
        selected_points = selector.run(max_level=3)
        assert isinstance(selected_points, np.ndarray)

    def test_edge_cases_all_same_level(self):
        """Test with all points at same level."""
        xy = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
        levels = np.array([1, 1, 1, 1])  # All same level
        scales = np.array([1.0, 1.0, 1.0, 1.0])
        
        def f_function(point):
            return point[0] + point[1]
        
        grid_x = np.linspace(0, 1, 10)
        grid_y = np.linspace(0, 1, 10)
        f_gt = np.zeros((10, 10))
        
        selector = PointSelector(
            xy=xy, levels=levels, scales=scales,
            f_function=f_function,
            grid_x=grid_x, grid_y=grid_y, f_gt=f_gt
        )
        
        selected_points = selector.run(max_level=2)
        assert isinstance(selected_points, np.ndarray)

    def test_edge_cases_constant_function(self):
        """Test with constant function."""
        xy = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
        levels = np.array([0, 0, 1, 1])
        scales = np.array([1.0, 1.0, 0.5, 0.5])
        
        def f_function(point):
            return 1.0  # Constant function
        
        grid_x = np.linspace(0, 1, 10)
        grid_y = np.linspace(0, 1, 10)
        f_gt = np.zeros((10, 10))
        
        selector = PointSelector(
            xy=xy, levels=levels, scales=scales,
            f_function=f_function,
            grid_x=grid_x, grid_y=grid_y, f_gt=f_gt
        )
        
        selected_points = selector.run(max_level=2)
        assert isinstance(selected_points, np.ndarray)

    def test_config_integration(self):
        """Test integration with configuration system."""
        xy = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
        levels = np.array([0, 0, 1, 1])
        scales = np.array([1.0, 1.0, 0.5, 0.5])
        
        def f_function(point):
            return point[0] + point[1]
        
        grid_x = np.linspace(0, 1, 10)
        grid_y = np.linspace(0, 1, 10)
        f_gt = np.zeros((10, 10))
        
        # Test that config integration works
        selector = PointSelector(
            xy=xy, levels=levels, scales=scales,
            f_function=f_function,
            grid_x=grid_x, grid_y=grid_y, f_gt=f_gt
        )
        
        # Should have default values from config
        assert hasattr(selector, 'tau')
        assert hasattr(selector, 'sign')
        assert hasattr(selector, 'eps')
        assert hasattr(selector, 'start_level')
        assert hasattr(selector, 'set_aside')

    def test_error_handling(self):
        """Test error handling."""
        # Test with invalid input shapes
        xy = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
        levels = np.array([0, 0, 1])  # Wrong length
        scales = np.array([1.0, 1.0, 0.5, 0.5])
        
        def f_function(point):
            return point[0] + point[1]
        
        grid_x = np.linspace(0, 1, 10)
        grid_y = np.linspace(0, 1, 10)
        f_gt = np.zeros((10, 10))
        
        with pytest.raises(ValueError):
            PointSelector(
                xy=xy, levels=levels, scales=scales,
                f_function=f_function,
                grid_x=grid_x, grid_y=grid_y, f_gt=f_gt
            )

    def test_performance_large_grid(self):
        """Test performance with large grid."""
        xy = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
        levels = np.array([0, 0, 1, 1])
        scales = np.array([1.0, 1.0, 0.5, 0.5])
        
        def f_function(point):
            return point[0] + point[1]
        
        grid_x = np.linspace(0, 1, 50)  # Large grid
        grid_y = np.linspace(0, 1, 50)
        f_gt = np.zeros((50, 50))
        
        selector = PointSelector(
            xy=xy, levels=levels, scales=scales,
            f_function=f_function,
            grid_x=grid_x, grid_y=grid_y, f_gt=f_gt
        )
        
        selected_points = selector.run(max_level=2)
        assert isinstance(selected_points, np.ndarray)

    def test_memory_usage(self):
        """Test memory usage with large datasets."""
        np.random.seed(42)
        n_points = 100
        xy = np.random.random((n_points, 2))
        levels = np.random.randint(0, 3, n_points)
        scales = np.random.random(n_points)
        
        def f_function(point):
            return np.sin(point[0]) + np.cos(point[1])
        
        grid_x = np.linspace(0, 1, 30)
        grid_y = np.linspace(0, 1, 30)
        f_gt = np.random.random((30, 30))
        
        selector = PointSelector(
            xy=xy, levels=levels, scales=scales,
            f_function=f_function,
            grid_x=grid_x, grid_y=grid_y, f_gt=f_gt
        )
        
        selected_points = selector.run(max_level=3)
        assert isinstance(selected_points, np.ndarray)

    def test_thread_safety(self):
        """Test thread safety (basic test)."""
        xy = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
        levels = np.array([0, 0, 1, 1])
        scales = np.array([1.0, 1.0, 0.5, 0.5])
        
        def f_function(point):
            return point[0] + point[1]
        
        grid_x = np.linspace(0, 1, 10)
        grid_y = np.linspace(0, 1, 10)
        f_gt = np.zeros((10, 10))
        
        selector = PointSelector(
            xy=xy, levels=levels, scales=scales,
            f_function=f_function,
            grid_x=grid_x, grid_y=grid_y, f_gt=f_gt
        )
        
        # Test that multiple runs don't interfere
        result1 = selector.run(max_level=1)
        result2 = selector.run(max_level=1)
        
        assert isinstance(result1, np.ndarray)
        assert isinstance(result2, np.ndarray)

    def test_api_consistency(self):
        """Test API consistency."""
        xy = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
        levels = np.array([0, 0, 1, 1])
        scales = np.array([1.0, 1.0, 0.5, 0.5])
        
        def f_function(point):
            return point[0] + point[1]
        
        grid_x = np.linspace(0, 1, 10)
        grid_y = np.linspace(0, 1, 10)
        f_gt = np.zeros((10, 10))
        
        selector = PointSelector(
            xy=xy, levels=levels, scales=scales,
            f_function=f_function,
            grid_x=grid_x, grid_y=grid_y, f_gt=f_gt
        )
        
        # Test that all expected attributes exist
        expected_attrs = [
            'xy', 'levels', 'scales', 'f', 'grid_x', 'grid_y', 'f_gt',
            'tau', 'sign', 'eps', 'start_level', 'set_aside',
            'lower_tau_quantile', 'n_splits_quantile', 'border',
            'border_bias', 'radius_scale', 'lower', 'upper',
            'factor', 'sel', 'index_array'
        ]
        
        for attr in expected_attrs:
            assert hasattr(selector, attr)

    def test_validation_errors(self):
        """Test validation error handling."""
        # Test with invalid xy shape
        xy = np.array([0, 0])  # Wrong shape
        levels = np.array([0, 0, 1, 1])
        scales = np.array([1.0, 1.0, 0.5, 0.5])
        
        def f_function(point):
            return point[0] + point[1]
        
        grid_x = np.linspace(0, 1, 10)
        grid_y = np.linspace(0, 1, 10)
        f_gt = np.zeros((10, 10))
        
        with pytest.raises((ValueError, IndexError)):
            PointSelector(
                xy=xy, levels=levels, scales=scales,
                f_function=f_function,
                grid_x=grid_x, grid_y=grid_y, f_gt=f_gt
            )

    def test_interpolation_edge_cases(self):
        """Test interpolation with edge cases."""
        xy = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
        levels = np.array([0, 0, 1, 1])
        scales = np.array([1.0, 1.0, 0.5, 0.5])
        
        def f_function(point):
            return point[0] + point[1]
        
        grid_x = np.linspace(0, 1, 10)
        grid_y = np.linspace(0, 1, 10)
        f_gt = np.zeros((10, 10))
        
        selector = PointSelector(
            xy=xy, levels=levels, scales=scales,
            f_function=f_function,
            grid_x=grid_x, grid_y=grid_y, f_gt=f_gt
        )
        
        # Test with single point
        these_xy = np.array([[0.5, 0.5]])
        these_values = np.array([1.0])
        
        result = selector.interpolate_sparse_data(these_xy, these_values)
        assert result.shape == (10, 10)
        
        # Test with empty arrays
        these_xy = np.array([])
        these_values = np.array([])
        
        result = selector.interpolate_sparse_data(these_xy, these_values)
        assert result.shape == (10, 10)

    def test_threshold_computation_edge_cases(self):
        """Test threshold computation with edge cases."""
        xy = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
        levels = np.array([0, 0, 1, 1])
        scales = np.array([1.0, 1.0, 0.5, 0.5])
        
        def f_function(point):
            return point[0] + point[1]
        
        grid_x = np.linspace(0, 1, 10)
        grid_y = np.linspace(0, 1, 10)
        f_gt = np.zeros((10, 10))
        
        selector = PointSelector(
            xy=xy, levels=levels, scales=scales,
            f_function=f_function,
            grid_x=grid_x, grid_y=grid_y, f_gt=f_gt
        )
        
        # Test with identical values
        work_xy = np.array([[0.5, 0.5]])
        work_f_values = np.array([1.0])
        cal_f_values = np.array([1.0])
        
        threshold, grid_values, mean_work, mean_cal, std_cal = selector.compute_threshold(
            work_xy, work_f_values, cal_f_values
        )
        
        assert isinstance(threshold, float)
        assert grid_values.shape == (10, 10)
        
        # Test with single value
        work_xy = np.array([[0.5, 0.5]])
        work_f_values = np.array([1.0])
        cal_f_values = np.array([2.0])
        
        threshold, grid_values, mean_work, mean_cal, std_cal = selector.compute_threshold(
            work_xy, work_f_values, cal_f_values
        )
        
        assert isinstance(threshold, float) 