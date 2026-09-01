"""Tests for the multi-camera calibration dock helpers."""

import numpy as np
import pytest

from sleap.gui.widgets.calibration_dock import CalibrationWorker


def _row(points: np.ndarray) -> dict:
    """Build an aniposelib-style detection row from an ``(N, 2)`` point array."""
    points = np.asarray(points, dtype="float64")
    return {
        "framenum": 0,
        "corners": points.reshape(-1, 1, 2),
        "ids": np.arange(len(points)).reshape(-1, 1),
    }


def _grid_row(n_x: int, n_y: int) -> dict:
    """A well-spread planar grid of ChArUco-like corners."""
    xs, ys = np.meshgrid(np.arange(n_x), np.arange(n_y))
    pts = np.column_stack([xs.ravel(), ys.ravel()]).astype("float64")
    return _row(pts)


def _collinear_row(n: int) -> dict:
    """``n`` corners along a single board row (the case that breaks OpenCV)."""
    pts = np.column_stack([np.arange(n), np.zeros(n)]).astype("float64")
    return _row(pts)


class TestRowIsDegenerate:
    def test_collinear_row_is_degenerate(self):
        assert CalibrationWorker._row_is_degenerate(_collinear_row(9))

    def test_spread_grid_is_not_degenerate(self):
        assert not CalibrationWorker._row_is_degenerate(_grid_row(11, 8))

    def test_near_collinear_row_is_degenerate(self):
        row = _collinear_row(9)
        # Add sub-pixel noise off the line — still unusable for a homography.
        row["corners"] = row["corners"] + np.random.default_rng(0).normal(
            0, 1e-3, row["corners"].shape
        )
        assert CalibrationWorker._row_is_degenerate(row)

    def test_too_few_points_not_reported_as_collinear(self):
        assert not CalibrationWorker._row_is_degenerate(_collinear_row(3))

    def test_identical_points_are_degenerate(self):
        assert CalibrationWorker._row_is_degenerate(_row(np.ones((9, 2))))


class TestFilterDegenerateRows:
    def test_drops_collinear_keeps_good(self):
        cam_a = [_grid_row(11, 8), _collinear_row(9), _grid_row(6, 5)]
        cam_b = [_grid_row(11, 8)]
        filtered, dropped = CalibrationWorker._filter_degenerate_rows(
            [cam_a, cam_b]
        )
        assert dropped == [1, 0]
        assert len(filtered[0]) == 2
        assert len(filtered[1]) == 1

    def test_no_degenerate_rows_is_noop(self):
        rows = [[_grid_row(11, 8)], [_grid_row(11, 8)]]
        filtered, dropped = CalibrationWorker._filter_degenerate_rows(rows)
        assert dropped == [0, 0]
        assert filtered == rows


class TestValidateRows:
    def test_passes_with_usable_rows(self):
        rows = [[_grid_row(11, 8)], [_grid_row(11, 8)]]
        CalibrationWorker._validate_rows(rows, ["Cam1", "Cam2"], [0, 0])

    def test_all_collinear_dropped_reports_collinearity(self):
        # Cam2 had only collinear detections, all removed by the pre-filter.
        rows = [[_grid_row(11, 8)], []]
        with pytest.raises(ValueError, match="near-collinear"):
            CalibrationWorker._validate_rows(rows, ["Cam1", "Cam2"], [0, 3])

    def test_weak_camera_mentions_non_collinear_corners(self):
        rows = [[_grid_row(11, 8)], [_grid_row(2, 2)]]
        with pytest.raises(ValueError, match="non-collinear ChArUco"):
            CalibrationWorker._validate_rows(rows, ["Cam1", "Cam2"], [0, 0])

    def test_no_detections_at_all(self):
        with pytest.raises(ValueError, match="No usable ChArUco boards"):
            CalibrationWorker._validate_rows([[], []], ["Cam1", "Cam2"], [0, 0])
