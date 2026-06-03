#!/usr/bin/env python

"""Tests for the :class:`PoissonTiler` hierarchical tiling component."""

import numpy as np
import pytest

from hiposa.poisson_tiler import PoissonTiler


@pytest.fixture
def tiler_2d():
    """A small 2D tiler reused across tests."""
    return PoissonTiler(tile_size=2.0, spacings=[1.0, 0.5], dimensions=2)


class TestPoissonTilerInit:
    def test_spacings_sorted_descending(self):
        tiler = PoissonTiler(tile_size=5.0, spacings=[0.3, 1.0, 0.6], dimensions=2)
        assert tiler.spacings == [1.0, 0.6, 0.3]

    def test_tile_size_enforced_to_minimum(self):
        # Requested tile size is smaller than min_tile_factor * largest spacing.
        tiler = PoissonTiler(
            tile_size=0.1, spacings=[1.0, 0.5], dimensions=2, min_tile_factor=2.0
        )
        assert tiler.tile_size == pytest.approx(2.0)

    def test_tile_size_respected_when_large_enough(self):
        tiler = PoissonTiler(tile_size=10.0, spacings=[1.0], dimensions=2)
        assert tiler.tile_size == pytest.approx(10.0)

    def test_base_tile_generated(self, tiler_2d):
        assert tiler_2d.tile_points is not None
        assert len(tiler_2d.tile_points) > 0
        assert len(tiler_2d.tile_points) == len(tiler_2d.tile_labels)

    def test_tile_labels_are_integers(self, tiler_2d):
        assert tiler_2d.tile_labels.dtype == np.int32

    def test_dimensions_match_tile_points(self):
        tiler = PoissonTiler(tile_size=2.0, spacings=[1.0], dimensions=3)
        assert tiler.tile_points.shape[1] == 3

    def test_hierarchical_levels_present(self, tiler_2d):
        # Two spacing levels means labels should include 0 and 1.
        unique_levels = set(np.unique(tiler_2d.tile_labels).tolist())
        assert unique_levels == {0, 1}


class TestGetPointsInRegion:
    def test_points_within_region_bounds(self, tiler_2d):
        region = [(0.0, 4.0), (0.0, 4.0)]
        points, labels = tiler_2d.get_points_in_region(region, n_processes=1)
        assert len(points) > 0
        assert len(points) == len(labels)
        for dim, (lo, hi) in enumerate(region):
            assert np.all(points[:, dim] >= lo)
            assert np.all(points[:, dim] <= hi)

    def test_corners_added_by_default(self, tiler_2d):
        region = [(0.0, 4.0), (0.0, 4.0)]
        points, _ = tiler_2d.get_points_in_region(region, n_processes=1)
        corners = [(0.0, 0.0), (0.0, 4.0), (4.0, 0.0), (4.0, 4.0)]
        for corner in corners:
            assert np.any(np.all(np.isclose(points, corner), axis=1)), (
                f"corner {corner} missing"
            )

    def test_corners_excluded_when_disabled(self, tiler_2d):
        region = [(0.0, 4.0), (0.0, 4.0)]
        with_corners, _ = tiler_2d.get_points_in_region(
            region, n_processes=1, add_corners=True
        )
        without_corners, _ = tiler_2d.get_points_in_region(
            region, n_processes=1, add_corners=False
        )
        assert len(with_corners) > len(without_corners)

    def test_single_tile_region(self, tiler_2d):
        region = [(0.0, tiler_2d.tile_size), (0.0, tiler_2d.tile_size)]
        points, labels = tiler_2d.get_points_in_region(region, n_processes=1)
        assert len(points) > 0
        assert len(points) == len(labels)

    def test_deterministic_with_seed(self):
        np.random.seed(7)
        tiler_a = PoissonTiler(tile_size=2.0, spacings=[1.0, 0.5], dimensions=2)
        pts_a = tiler_a.tile_points

        np.random.seed(7)
        tiler_b = PoissonTiler(tile_size=2.0, spacings=[1.0, 0.5], dimensions=2)
        pts_b = tiler_b.tile_points

        assert pts_a.shape == pts_b.shape
        assert np.allclose(pts_a, pts_b)
