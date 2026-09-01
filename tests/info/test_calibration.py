"""Tests for :mod:`sleap.info.calibration`."""

import numpy as np
import pytest

from sleap.info.calibration import (
    GOOD_PX,
    OUTPUT_DIRNAME,
    evaluate_calibration,
    format_report,
    resolve_calibration_paths,
    verdict,
    _default_output_path,
    _summarize,
)


def test_verdict_thresholds():
    assert verdict(0.1) == "GOOD"
    assert verdict(1.5) == "OK"
    assert verdict(5.0) == "CHECK"
    assert verdict(float("nan")) == "NO DATA"


def test_summarize_is_nan_aware():
    err = np.array([[0.0, np.nan], [2.0, 4.0]])
    s = _summarize(err)
    assert s["n"] == 3
    assert s["median"] == 2.0
    assert s["max"] == 4.0


def test_summarize_all_nan():
    s = _summarize(np.full((3, 3), np.nan))
    assert s["n"] == 0
    assert np.isnan(s["rms"])


def test_default_output_path_uses_subfolder(tmp_path):
    meta = tmp_path / "calibration_20260729.metadata.h5"
    out = _default_output_path(str(meta), ".quality.txt")
    assert out == tmp_path / OUTPUT_DIRNAME / "calibration_20260729.quality.txt"
    assert out.parent.is_dir()  # created as a side effect


class TestResolveCalibrationPaths:
    @pytest.fixture
    def calib_pair(self, tmp_path):
        toml_path = tmp_path / "calibration_20260729.toml"
        meta_path = tmp_path / "calibration_20260729.metadata.h5"
        toml_path.write_text("[cam_0]\n")
        meta_path.write_bytes(b"")
        return toml_path, meta_path

    def test_from_toml(self, calib_pair):
        toml_path, meta_path = calib_pair
        assert resolve_calibration_paths(toml_path) == (toml_path, meta_path)

    def test_from_metadata_h5(self, calib_pair):
        toml_path, meta_path = calib_pair
        assert resolve_calibration_paths(meta_path) == (toml_path, meta_path)

    def test_from_shared_prefix(self, calib_pair):
        toml_path, meta_path = calib_pair
        prefix = toml_path.with_name("calibration_20260729")
        assert resolve_calibration_paths(prefix) == (toml_path, meta_path)

    def test_from_folder(self, calib_pair):
        toml_path, meta_path = calib_pair
        assert resolve_calibration_paths(toml_path.parent) == (toml_path, meta_path)

    def test_explicit_metadata_override(self, calib_pair, tmp_path):
        toml_path, _ = calib_pair
        other = tmp_path / "somewhere_else.metadata.h5"
        other.write_bytes(b"")
        assert resolve_calibration_paths(toml_path, other) == (toml_path, other)

    def test_missing_metadata_raises(self, tmp_path):
        (tmp_path / "calibration_x.toml").write_text("[cam_0]\n")
        with pytest.raises(FileNotFoundError, match="metadata"):
            resolve_calibration_paths(tmp_path / "calibration_x.toml")

    def test_multiple_tomls_in_folder_raises(self, tmp_path):
        (tmp_path / "calibration_a.toml").write_text("")
        (tmp_path / "calibration_b.toml").write_text("")
        with pytest.raises(ValueError, match="[Mm]ultiple"):
            resolve_calibration_paths(tmp_path)

    def test_unknown_suffix_raises(self, tmp_path):
        bad = tmp_path / "calibration.txt"
        bad.write_text("")
        with pytest.raises(ValueError, match="interpret"):
            resolve_calibration_paths(bad)


@pytest.fixture
def synthetic_calibration(tmp_path):
    """A tiny, near-perfect 3-camera calibration + metadata file."""
    pytest.importorskip("aniposelib")
    pytest.importorskip("jax")
    import h5py
    from aniposelib.cameras import Camera, CameraGroup

    rng = np.random.default_rng(0)

    # Planar board: 6 x 5 grid of corners, centred on the origin.
    gx, gy = np.meshgrid(np.arange(6), np.arange(5))
    board = np.column_stack(
        [gx.ravel(), gy.ravel(), np.zeros(gx.size)]
    ).astype("float64")
    board -= board.mean(axis=0)
    n_corners = board.shape[0]

    # Rigidly move/rotate the board through a handful of "frames".
    frames_3d = []
    for k in range(12):
        ang = 0.15 * k
        rot = np.array([
            [np.cos(ang), 0, np.sin(ang)],
            [0, 1, 0],
            [-np.sin(ang), 0, np.cos(ang)],
        ])
        trans = np.array([0.3 * np.sin(k), 0.2 * k - 1.0, 10.0 + 0.1 * k])
        frames_3d.append(board @ rot.T + trans)
    pts3d = np.stack(frames_3d, axis=0)  # (n_frames, n_corners, 3)

    K = np.array([[900.0, 0.0, 320.0], [0.0, 900.0, 240.0], [0.0, 0.0, 1.0]])
    cam_specs = [
        (np.zeros(3), np.zeros(3)),
        (np.array([0.0, 0.5, 0.0]), np.array([-2.0, 0.0, 1.0])),
        (np.array([0.0, -0.5, 0.0]), np.array([2.0, 0.0, 1.0])),
    ]
    cams = [
        Camera(matrix=K.copy(), dist=np.zeros(5), size=(640, 480),
               rvec=rvec, tvec=tvec, name=f"cam{i}")
        for i, (rvec, tvec) in enumerate(cam_specs)
    ]
    cgroup = CameraGroup(cams)

    proj = cgroup.project(pts3d.reshape(-1, 3))  # (n_cams, N, 2)
    proj = proj.reshape(len(cams), pts3d.shape[0], n_corners, 2)
    detected = proj + rng.normal(0.0, 0.05, proj.shape)  # sub-pixel noise

    toml_path = tmp_path / "calibration_20260101.toml"
    meta_path = tmp_path / "calibration_20260101.metadata.h5"
    cgroup.dump(str(toml_path))
    with h5py.File(meta_path, "w") as f:
        f.create_dataset("frames", data=np.arange(pts3d.shape[0]))
        f.create_dataset("detected_corners", data=detected)
    return toml_path, meta_path


class TestEvaluateCalibration:
    def test_low_error_reads_as_good(self, synthetic_calibration):
        toml_path, _ = synthetic_calibration
        result = evaluate_calibration(toml_path)

        assert result["n_cameras"] == 3
        assert result["n_corners"] == 30
        assert result["overall"]["rms"] < GOOD_PX
        assert result["verdict"] == "GOOD"
        assert [c["verdict"] for c in result["per_camera"]] == ["GOOD"] * 3
        assert set(result["camera_positions"]) == {"cam0", "cam1", "cam2"}
        assert result["pairwise_distance_px"]["max"] > 0

    def test_resolves_from_metadata_argument(self, synthetic_calibration):
        _, meta_path = synthetic_calibration
        result = evaluate_calibration(meta_path)
        assert result["overall"]["rms"] < GOOD_PX

    def test_camera_count_mismatch_raises(self, synthetic_calibration, tmp_path):
        import h5py

        toml_path, meta_path = synthetic_calibration
        with h5py.File(meta_path, "r") as f:
            detected = f["detected_corners"][:]
        bad_meta = tmp_path / "calibration_20260101.metadata.h5"
        with h5py.File(bad_meta, "w") as f:
            f.create_dataset("detected_corners", data=detected[:2])  # drop a cam
        with pytest.raises(ValueError, match="cameras"):
            evaluate_calibration(toml_path)

    def test_format_report_contains_key_lines(self, synthetic_calibration):
        toml_path, _ = synthetic_calibration
        text = format_report(evaluate_calibration(toml_path))
        assert "CALIBRATION QUALITY REPORT" in text
        assert "cam0" in text
        assert "Verdict thresholds" in text
