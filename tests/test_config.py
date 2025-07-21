#!/usr/bin/env python

"""Comprehensive tests for config module."""

import pytest
import numpy as np
from hiposa.config import (
    SamplingConfig,
    TilingConfig,
    PointSelectorConfig,
    SAMPLING_CONFIG,
    TILING_CONFIG,
    POINT_SELECTOR_CONFIG,
    get_default_config
)


class TestConfig:
    """Test suite for configuration classes."""

    def test_sampling_config_defaults(self):
        """Test SamplingConfig default values."""
        config = SamplingConfig()
        
        assert config.DEFAULT_K == 60
        assert config.DEFAULT_K_SECOND_PASS == 80
        assert config.DEFAULT_EPSILON == 1e-10
        assert config.KDTree_UPDATE_FREQUENCY == 10
        assert config.MINIMIZATION_XATOL == 1e-10
        assert config.MINIMIZATION_FATOL == 1e-10
        assert config.INVARIANT_POINT_ATTEMPTS == 10

    def test_sampling_config_custom(self):
        """Test SamplingConfig with custom values."""
        config = SamplingConfig(
            DEFAULT_K=100,
            DEFAULT_K_SECOND_PASS=120,
            DEFAULT_EPSILON=1e-12,
            KDTree_UPDATE_FREQUENCY=20,
            MINIMIZATION_XATOL=1e-12,
            MINIMIZATION_FATOL=1e-12,
            INVARIANT_POINT_ATTEMPTS=15
        )
        
        assert config.DEFAULT_K == 100
        assert config.DEFAULT_K_SECOND_PASS == 120
        assert config.DEFAULT_EPSILON == 1e-12
        assert config.KDTree_UPDATE_FREQUENCY == 20
        assert config.MINIMIZATION_XATOL == 1e-12
        assert config.MINIMIZATION_FATOL == 1e-12
        assert config.INVARIANT_POINT_ATTEMPTS == 15

    def test_tiling_config_defaults(self):
        """Test TilingConfig default values."""
        config = TilingConfig()
        
        assert config.MIN_TILE_FACTOR == 2.0
        assert config.DEFAULT_N_PROCESSES is None
        assert config.DEFAULT_ADD_CORNERS is True

    def test_tiling_config_custom(self):
        """Test TilingConfig with custom values."""
        config = TilingConfig(
            MIN_TILE_FACTOR=3.0,
            DEFAULT_N_PROCESSES=4,
            DEFAULT_ADD_CORNERS=False
        )
        
        assert config.MIN_TILE_FACTOR == 3.0
        assert config.DEFAULT_N_PROCESSES == 4
        assert config.DEFAULT_ADD_CORNERS is False

    def test_point_selector_config_defaults(self):
        """Test PointSelectorConfig default values."""
        config = PointSelectorConfig()
        
        assert config.DEFAULT_TAU == 75.0
        assert config.DEFAULT_SIGN == 1
        assert config.DEFAULT_EPS == 0.0
        assert config.DEFAULT_START_LEVEL == 0
        assert config.DEFAULT_SET_ASIDE == 0.5
        assert config.DEFAULT_LOWER_TAU_QUANTILE == 1.0
        assert config.DEFAULT_N_SPLITS_QUANTILE == 10
        assert config.DEFAULT_BORDER == 4
        assert config.DEFAULT_BORDER_BIAS == 0.5
        assert config.DEFAULT_RADIUS_SCALE == 4.0
        assert config.DEFAULT_N_SPLITS == 5

    def test_point_selector_config_custom(self):
        """Test PointSelectorConfig with custom values."""
        config = PointSelectorConfig(
            DEFAULT_TAU=80.0,
            DEFAULT_SIGN=-1,
            DEFAULT_EPS=0.1,
            DEFAULT_START_LEVEL=1,
            DEFAULT_SET_ASIDE=0.3,
            DEFAULT_LOWER_TAU_QUANTILE=0.8,
            DEFAULT_N_SPLITS_QUANTILE=5,
            DEFAULT_BORDER=2,
            DEFAULT_BORDER_BIAS=0.7,
            DEFAULT_RADIUS_SCALE=3.0,
            DEFAULT_N_SPLITS=3
        )
        
        assert config.DEFAULT_TAU == 80.0
        assert config.DEFAULT_SIGN == -1
        assert config.DEFAULT_EPS == 0.1
        assert config.DEFAULT_START_LEVEL == 1
        assert config.DEFAULT_SET_ASIDE == 0.3
        assert config.DEFAULT_LOWER_TAU_QUANTILE == 0.8
        assert config.DEFAULT_N_SPLITS_QUANTILE == 5
        assert config.DEFAULT_BORDER == 2
        assert config.DEFAULT_BORDER_BIAS == 0.7
        assert config.DEFAULT_RADIUS_SCALE == 3.0
        assert config.DEFAULT_N_SPLITS == 3

    def test_global_config_instances(self):
        """Test global configuration instances."""
        # Test that global instances exist and have correct types
        assert isinstance(SAMPLING_CONFIG, SamplingConfig)
        assert isinstance(TILING_CONFIG, TilingConfig)
        assert isinstance(POINT_SELECTOR_CONFIG, PointSelectorConfig)
        
        # Test that they have the expected default values
        assert SAMPLING_CONFIG.DEFAULT_K == 60
        assert TILING_CONFIG.MIN_TILE_FACTOR == 2.0
        assert POINT_SELECTOR_CONFIG.DEFAULT_TAU == 75.0

    def test_get_default_config(self):
        """Test get_default_config function."""
        config = get_default_config()
        
        # Check that it returns a dictionary with expected keys
        assert isinstance(config, dict)
        assert 'sampling' in config
        assert 'tiling' in config
        assert 'point_selector' in config
        
        # Check that the values are the correct config instances
        assert config['sampling'] is SAMPLING_CONFIG
        assert config['tiling'] is TILING_CONFIG
        assert config['point_selector'] is POINT_SELECTOR_CONFIG

    def test_config_immutability(self):
        """Test that config instances are not shared between imports."""
        # Create a new instance
        custom_sampling = SamplingConfig(DEFAULT_K=100)
        
        # Global instance should remain unchanged
        assert SAMPLING_CONFIG.DEFAULT_K == 60
        assert custom_sampling.DEFAULT_K == 100

    def test_config_dataclass_features(self):
        """Test that config classes behave as proper dataclasses."""
        # Test equality
        config1 = SamplingConfig()
        config2 = SamplingConfig()
        assert config1 == config2
        
        # Test inequality
        config3 = SamplingConfig(DEFAULT_K=100)
        assert config1 != config3
        
        # Test repr
        repr_str = repr(config1)
        assert "SamplingConfig" in repr_str
        assert "DEFAULT_K=60" in repr_str

    def test_config_validation(self):
        """Test config parameter validation."""
        # Test with valid values
        config = SamplingConfig(DEFAULT_K=50)
        assert config.DEFAULT_K == 50
        
        # Test with zero values (should be valid for some parameters)
        config = SamplingConfig(DEFAULT_EPSILON=0.0)
        assert config.DEFAULT_EPSILON == 0.0
        
        # Test with negative values (should be valid for some parameters)
        config = PointSelectorConfig(DEFAULT_SIGN=-1)
        assert config.DEFAULT_SIGN == -1

    def test_config_edge_cases(self):
        """Test config edge cases."""
        # Test with very large values
        config = SamplingConfig(DEFAULT_K=1000000)
        assert config.DEFAULT_K == 1000000
        
        # Test with very small values
        config = SamplingConfig(DEFAULT_EPSILON=1e-20)
        assert config.DEFAULT_EPSILON == 1e-20
        
        # Test with boolean values
        config = TilingConfig(DEFAULT_ADD_CORNERS=False)
        assert config.DEFAULT_ADD_CORNERS is False

    def test_config_integration_with_modules(self):
        """Test that config can be imported and used by other modules."""
        # This test verifies that the config can be imported
        # and used without causing import errors
        try:
            from hiposa.config import SAMPLING_CONFIG
            assert SAMPLING_CONFIG.DEFAULT_K == 60
        except ImportError:
            pytest.fail("Failed to import SAMPLING_CONFIG")

    def test_config_consistency(self):
        """Test config parameter consistency."""
        # Test that related parameters are consistent
        sampling_config = SamplingConfig()
        tiling_config = TilingConfig()
        point_selector_config = PointSelectorConfig()
        
        # All should be valid configurations
        assert sampling_config.DEFAULT_K > 0
        assert tiling_config.MIN_TILE_FACTOR > 0
        assert 0 <= point_selector_config.DEFAULT_TAU <= 100
        assert point_selector_config.DEFAULT_SIGN in [-1, 1]
        assert 0 <= point_selector_config.DEFAULT_SET_ASIDE <= 1

    def test_config_documentation(self):
        """Test that config classes have proper documentation."""
        # Test that classes have docstrings
        assert SamplingConfig.__doc__ is not None
        assert TilingConfig.__doc__ is not None
        assert PointSelectorConfig.__doc__ is not None
        
        # Test that the docstrings are not empty
        assert len(SamplingConfig.__doc__.strip()) > 0
        assert len(TilingConfig.__doc__.strip()) > 0
        assert len(PointSelectorConfig.__doc__.strip()) > 0

    def test_config_serialization(self):
        """Test config serialization (for potential future use)."""
        config = SamplingConfig()
        
        # Test that we can access all attributes
        attrs = [
            'DEFAULT_K', 'DEFAULT_K_SECOND_PASS', 'DEFAULT_EPSILON',
            'KDTree_UPDATE_FREQUENCY', 'MINIMIZATION_XATOL',
            'MINIMIZATION_FATOL', 'INVARIANT_POINT_ATTEMPTS'
        ]
        
        for attr in attrs:
            assert hasattr(config, attr)
            value = getattr(config, attr)
            assert value is not None

    def test_config_performance(self):
        """Test config access performance."""
        import time
        
        # Test that config access is fast
        config = SamplingConfig()
        
        start_time = time.time()
        for _ in range(1000):
            _ = config.DEFAULT_K
        end_time = time.time()
        
        # Should be very fast (less than 1ms for 1000 accesses)
        assert (end_time - start_time) < 0.001

    def test_config_memory_usage(self):
        """Test config memory usage."""
        import sys
        
        # Test that config instances are lightweight
        config = SamplingConfig()
        size = sys.getsizeof(config)
        
        # Should be reasonably small (less than 1KB)
        assert size < 1024

    def test_config_thread_safety(self):
        """Test config thread safety (basic test)."""
        import threading
        
        config = SamplingConfig()
        results = []
        
        def read_config():
            results.append(config.DEFAULT_K)
        
        # Create multiple threads reading the config
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=read_config)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # All results should be the same
        assert all(result == 60 for result in results)
        assert len(results) == 10 