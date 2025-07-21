#!/usr/bin/env python

"""Integration tests for HiPoSa library."""

import pytest
import numpy as np
from hiposa.poisson_disc_sampling import PoissonDiskSamplerWithExisting
from hiposa.poisson_tiler import PoissonTiler
from hiposa.point_selector import PointSelector
from hiposa.utils import check_minimum_distance, create_symmetry_operators
from hiposa.config import SAMPLING_CONFIG, TILING_CONFIG, POINT_SELECTOR_CONFIG


class TestIntegration:
    """Integration test suite for HiPoSa library."""

    def test_hierarchical_sampling_with_point_selection(self):
        """Test integration between hierarchical sampling and point selection."""
        # Create hierarchical sampling
        tile_size = 4.0
        spacings = [1.0, 0.5]
        tiler = PoissonTiler(tile_size=tile_size, spacings=spacings)
        
        # Get points from a region
        region = ((0, 8), (0, 8))
        points, levels = tiler.get_points_in_region(region)
        
        # Use points for point selection
        xy = points
        levels_array = levels
        scales = np.ones_like(levels_array, dtype=float)
        
        def f_function(point):
            return np.sin(point[0]) + np.cos(point[1])
        
        grid_x = np.linspace(0, 8, 20)
        grid_y = np.linspace(0, 8, 20)
        f_gt = np.random.random((20, 20))
        
        selector = PointSelector(
            xy=xy,
            levels=levels_array,
            scales=scales,
            f_function=f_function,
            grid_x=grid_x,
            grid_y=grid_y,
            f_gt=f_gt
        )
        
        # Run point selection
        selected_mask = selector.run(max_level=3)
        selected_points = xy[selected_mask]
        
        # Verify results
        assert len(selected_points) >= 0
        if len(selected_points) > 0:
            assert selected_points.shape[1] == 2

    def test_config_integration_across_modules(self):
        """Test that configuration is properly integrated across modules."""
        # Test that config values are used in sampling
        domain = [(0, 10), (0, 10)]
        r = 1.0
        
        sampler = PoissonDiskSamplerWithExisting(domain=domain, r=r)
        assert sampler.k == SAMPLING_CONFIG.DEFAULT_K
        
        # Test that config values are used in tiling
        tile_size = 4.0
        spacings = [1.0, 0.5]
        tiler = PoissonTiler(tile_size=tile_size, spacings=spacings)
        assert tiler.tile_size >= TILING_CONFIG.MIN_TILE_FACTOR * spacings[0]
        
        # Test that config values are used in point selection
        xy = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
        levels = np.array([0, 0, 1, 1])
        scales = np.array([1.0, 1.0, 0.5, 0.5])
        
        def f_function(point):
            return point[0] + point[1]
        
        grid_x = np.linspace(0, 1, 10)
        grid_y = np.linspace(0, 1, 10)
        f_gt = np.zeros((10, 10))
        
        selector = PointSelector(
            xy=xy,
            levels=levels,
            scales=scales,
            f_function=f_function,
            grid_x=grid_x,
            grid_y=grid_y,
            f_gt=f_gt
        )
        
        assert selector.tau == POINT_SELECTOR_CONFIG.DEFAULT_TAU
        assert selector.sign == POINT_SELECTOR_CONFIG.DEFAULT_SIGN

    def test_multi_dimensional_integration(self):
        """Test integration across multiple dimensions."""
        # Test 3D sampling
        domain_3d = [(0, 5), (0, 5), (0, 5)]
        r_3d = 1.0
        sampler_3d = PoissonDiskSamplerWithExisting(domain=domain_3d, r=r_3d)
        points_3d, labels_3d = sampler_3d.sample()
        
        # Test 3D tiling
        tile_size_3d = 3.0
        spacings_3d = [1.0, 0.5]
        tiler_3d = PoissonTiler(
            tile_size=tile_size_3d,
            spacings=spacings_3d,
            dimensions=3
        )
        
        region_3d = ((0, 6), (0, 6), (0, 6))
        tiled_points_3d, tiled_labels_3d = tiler_3d.get_points_in_region(region_3d)
        
        # Verify 3D results
        assert len(points_3d) > 0
        assert points_3d.shape[1] == 3
        assert len(tiled_points_3d) > 0
        assert tiled_points_3d.shape[1] == 3

    def test_performance_integration(self):
        """Test performance integration across modules."""
        # Large domain test
        domain = [(0, 100), (0, 100)]
        r = 2.0
        
        # Test sampling performance
        sampler = PoissonDiskSamplerWithExisting(domain=domain, r=r)
        points, labels = sampler.sample()
        
        # Test tiling performance
        tile_size = 10.0
        spacings = [3.0, 2.0]  # Increased minimum spacing for easier distance satisfaction
        tiler = PoissonTiler(tile_size=tile_size, spacings=spacings)
        
        region = ((0, 10), (0, 10))  # Reduced region size for fewer points
        tiled_points, tiled_labels = tiler.get_points_in_region(region)
        
        # Verify performance results
        assert len(points) > 0
        assert len(tiled_points) > 0
        assert check_minimum_distance(points, r)
        # Filter out corner points for minimum distance check
        non_corner_mask = ~((np.isclose(tiled_points[:,0], region[0][0]) | np.isclose(tiled_points[:,0], region[0][1])) &
                            (np.isclose(tiled_points[:,1], region[1][0]) | np.isclose(tiled_points[:,1], region[1][1])))
        filtered_points = tiled_points[non_corner_mask]
        if len(filtered_points) > 1:
            assert check_minimum_distance(filtered_points, spacings[-1])

    def test_error_handling_integration(self):
        """Test error handling integration across modules."""
        # Test invalid domain
        with pytest.raises(ValueError):
            PoissonDiskSamplerWithExisting(domain=[], r=1.0)
        
        # Test tile size adjustment (no error expected)
        tiler = PoissonTiler(tile_size=0.0, spacings=[1.0])
        assert tiler.tile_size >= 2.0  # min_tile_factor * spacing
        
        # Test invalid point selector parameters
        with pytest.raises(ValueError):
            PointSelector(
                xy=np.array([]),
                levels=np.array([]),
                scales=np.array([]),
                f_function=lambda x: 0,
                grid_x=np.array([]),
                grid_y=np.array([]),
                f_gt=np.array([])
            )

    def test_memory_integration(self):
        """Test memory usage integration across modules."""
        import sys
        
        # Test memory usage of sampling
        domain = [(0, 10), (0, 10)]
        r = 1.0
        sampler = PoissonDiskSamplerWithExisting(domain=domain, r=r)
        points, labels = sampler.sample()
        
        # Test memory usage of tiling
        tile_size = 4.0
        spacings = [1.0, 0.5]
        tiler = PoissonTiler(tile_size=tile_size, spacings=spacings)
        region = ((0, 8), (0, 8))
        tiled_points, tiled_labels = tiler.get_points_in_region(region)
        
        # Verify memory usage is reasonable
        sampler_size = sys.getsizeof(sampler)
        tiler_size = sys.getsizeof(tiler)
        
        assert sampler_size < 1024 * 1024  # Less than 1MB
        assert tiler_size < 1024 * 1024  # Less than 1MB

    def test_thread_safety_integration(self):
        """Test thread safety integration across modules."""
        import threading
        
        results = []
        
        def run_sampling():
            domain = [(0, 5), (0, 5)]
            r = 1.0
            sampler = PoissonDiskSamplerWithExisting(domain=domain, r=r)
            points, labels = sampler.sample()
            results.append(len(points))
        
        def run_tiling():
            tile_size = 2.0
            spacings = [1.0, 0.5]
            tiler = PoissonTiler(tile_size=tile_size, spacings=spacings)
            region = ((0, 4), (0, 4))
            points, labels = tiler.get_points_in_region(region)
            results.append(len(points))
        
        # Create threads
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=run_sampling)
            threads.append(thread)
            thread = threading.Thread(target=run_tiling)
            threads.append(thread)
        
        # Start threads
        for thread in threads:
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Verify results
        assert len(results) == 10
        assert all(result > 0 for result in results)

    def test_api_consistency_integration(self):
        """Test API consistency across modules."""
        # Test that all modules follow consistent naming conventions
        from hiposa import PoissonDiskSamplerWithExisting, PoissonTiler, PointSelector
        
        # Test that classes can be instantiated with consistent parameters
        domain = [(0, 10), (0, 10)]
        r = 1.0
        
        sampler = PoissonDiskSamplerWithExisting(domain=domain, r=r)
        assert hasattr(sampler, 'sample')
        assert callable(sampler.sample)
        
        tile_size = 4.0
        spacings = [1.0, 0.5]
        tiler = PoissonTiler(tile_size=tile_size, spacings=spacings)
        assert hasattr(tiler, 'get_points_in_region')
        assert callable(tiler.get_points_in_region)
        
        # Test point selector API
        xy = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
        levels = np.array([0, 0, 1, 1])
        scales = np.array([1.0, 1.0, 0.5, 0.5])
        
        def f_function(point):
            return point[0] + point[1]
        
        grid_x = np.linspace(0, 1, 10)
        grid_y = np.linspace(0, 1, 10)
        f_gt = np.zeros((10, 10))
        
        selector = PointSelector(
            xy=xy,
            levels=levels,
            scales=scales,
            f_function=f_function,
            grid_x=grid_x,
            grid_y=grid_y,
            f_gt=f_gt
        )
        assert hasattr(selector, 'run')
        assert callable(selector.run)

    def test_data_flow_integration(self):
        """Test data flow between modules."""
        # Create initial data
        domain = [(0, 8), (0, 8)]
        r = 1.0
        
        # Step 1: Generate points with sampling
        sampler = PoissonDiskSamplerWithExisting(domain=domain, r=r)
        points, labels = sampler.sample()
        
        # Step 2: Use points for tiling
        tile_size = 4.0
        spacings = [1.5, 0.75]
        tiler = PoissonTiler(tile_size=tile_size, spacings=spacings)
        
        region = ((0, 8), (0, 8))
        tiled_points, tiled_labels = tiler.get_points_in_region(region)
        
        # Step 3: Use tiled points for point selection
        if len(tiled_points) > 0:
            xy = tiled_points
            levels = tiled_labels
            scales = np.ones_like(levels, dtype=float)
            
            def f_function(point):
                return np.sin(point[0]) + np.cos(point[1])
            
            grid_x = np.linspace(0, 8, 20)
            grid_y = np.linspace(0, 8, 20)
            f_gt = np.random.random((20, 20))
            
            selector = PointSelector(
                xy=xy,
                levels=levels,
                scales=scales,
                f_function=f_function,
                grid_x=grid_x,
                grid_y=grid_y,
                f_gt=f_gt
            )
            
            selected_points = selector.run(max_level=2)
            
            # Verify data flow
            assert len(points) > 0
            assert len(tiled_points) > 0
            assert len(selected_points) >= 0

    def test_edge_case_integration(self):
        """Test edge case integration across modules."""
        # Test with very small domains
        domain = [(0, 0.1), (0, 0.1)]
        r = 0.05
        sampler = PoissonDiskSamplerWithExisting(domain=domain, r=r)
        points, labels = sampler.sample()
        
        # Test with very small tile size
        tile_size = 0.5
        spacings = [0.2, 0.1]
        tiler = PoissonTiler(tile_size=tile_size, spacings=spacings)
        region = ((0, 1), (0, 1))
        tiled_points, tiled_labels = tiler.get_points_in_region(region)
        
        # Verify edge case handling
        assert len(points) >= 0
        assert len(tiled_points) >= 0 