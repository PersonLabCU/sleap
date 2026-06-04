import csv
from pathlib import Path

import h5py
import numpy as np
from qtpy.QtWidgets import QCheckBox, QDoubleSpinBox, QLabel, QTableWidget

from sleap.gui.reach_detection import (
    ReachOutcome,
    ReachSegment,
    detect_reaches_absolute,
    detect_reaches_kpn,
    extract_hand_position_3d_with_confidence,
    save_kpn_reach_details_csv,
    suggest_hand_nodes,
)
from sleap.gui.reach_projection import (
    _camera_index_map,
    _match_camera_names,
    _scores_for_reprojection_cameras,
    _triangulate_vectorized,
    load_points3d_h5,
    translate_points3d_h5,
)
from sleap.gui.widgets.docks import ReachesDock


class DummyNode:
    def __init__(self, name):
        self.name = name


class DummySkeleton:
    def __init__(self, names):
        self.nodes = [DummyNode(name) for name in names]


class DummyInstance:
    def __init__(self, skeleton, points, point_scores=None):
        self.skeleton = skeleton
        self._points = np.asarray(points, dtype=np.float64)
        self.point_scores = point_scores

    def numpy(self):
        return self._points


class DummyLabeledFrame:
    def __init__(self, frame_idx, instances):
        self.frame_idx = frame_idx
        self.predicted_instances = instances
        self.user_instances = []


class DummyLabels:
    def __init__(self, frames):
        self.frames = frames

    def find(self, video):
        return self.frames


def test_suggest_hand_nodes_prefers_reach_hand_nodes():
    nodes = [
        "pellet",
        "tip_middle_L",
        "wrist_mid_R",
        "mid_index_R",
        "tip_index_L",
        "wrist_mid_L",
        "tip_ring_R",
        "tip_middle_R",
        "wrist_pinky_L",
        "mid_pinky_L",
        "tip_pinky_L",
        "mid_index_L",
        "wrist_index_L",
        "tip_ring_L",
        "wrist_pinky_R",
        "mid_pinky_R",
        "tip_pinky_R",
        "tip_index_R",
        "wrist_index_R",
        "tail_R",
    ]

    assert suggest_hand_nodes(nodes) == {
        "left": [
            "wrist_mid_L",
            "wrist_pinky_L",
            "mid_pinky_L",
            "tip_pinky_L",
            "tip_index_L",
            "mid_index_L",
            "wrist_index_L",
            "tip_ring_L",
            "tip_middle_L",
        ],
        "right": [
            "wrist_mid_R",
            "wrist_pinky_R",
            "mid_pinky_R",
            "tip_pinky_R",
            "tip_index_R",
            "mid_index_R",
            "wrist_index_R",
            "tip_ring_R",
            "tip_middle_R",
        ],
    }


def test_suggest_hand_nodes_keeps_keyword_fallback():
    assert suggest_hand_nodes(["left_hand", "right_hand", "tail"]) == {
        "left": ["left_hand"],
        "right": ["right_hand"],
    }


def test_extract_hand_position_with_confidence_filters_low_score_nodes():
    skeleton = DummySkeleton(["a", "b", "c"])
    labels = DummyLabels(
        [
            DummyLabeledFrame(
                0,
                [
                    DummyInstance(
                        skeleton,
                        [[0, 0], [10, 0], [100, 0]],
                        point_scores=np.asarray([0.9, 0.8, 0.1]),
                    )
                ],
            ),
            DummyLabeledFrame(
                1,
                [
                    DummyInstance(
                        skeleton,
                        [[1, 1], [2, 2], [3, 3]],
                        point_scores=np.asarray([0.2, 0.1, 0.3]),
                    )
                ],
            ),
        ]
    )

    traj, conf = extract_hand_position_3d_with_confidence(
        labels,
        video=object(),
        node_names=["a", "b", "c"],
        min_confidence=0.5,
    )

    expected_x = ((0 * 0.9) + (10 * 0.8)) / (0.9 + 0.8)
    np.testing.assert_allclose(traj[0], [expected_x, 0, 0])
    assert conf[0] == 0.8
    assert np.all(np.isnan(traj[1]))
    assert conf[1] == 0.2


def test_detect_reaches_ignores_low_confidence_pellet_trace():
    n_frames = 30
    right_hand = np.zeros((n_frames, 3), dtype=np.float64)
    left_hand = np.zeros((n_frames, 3), dtype=np.float64)
    pellet = np.zeros((n_frames, 3), dtype=np.float64)

    reaches = detect_reaches_kpn(
        right_hand=right_hand,
        left_hand=left_hand,
        pellet=pellet,
        right_hand_confidence=np.ones(n_frames),
        left_hand_confidence=np.ones(n_frames),
        pellet_confidence=np.full(n_frames, 0.1),
        confidence=0.5,
    )

    assert reaches == []


def test_detect_reaches_absolute_reports_first_and_subreach_peaks():
    signal = np.asarray(
        [0, 0, 1, 3, 6, 9, 7, 10, 8, 4, 3, 4, 0],
        dtype=np.float64,
    )

    reaches, details = detect_reaches_absolute(
        signal,
        threshold=5.0,
        peak_prominence=1.0,
        start_padding=0,
        min_frame=3,
        max_frame=50,
        frame_rate=10.0,
        return_details=True,
    )

    assert len(reaches) == 1
    assert reaches[0].max_frame == 5
    assert reaches[0].end_frame == 10
    assert details[0]["first_peak_frame"] == 5
    assert details[0]["first_peak_time_s"] == 0.5
    assert details[0]["mini_peak_frames"] == [7]
    assert details[0]["mini_peak_times_s"] == [0.7]


def test_detect_reaches_absolute_classifies_with_pellet_logic():
    signal = np.zeros(30, dtype=np.float64)
    signal[:13] = [0, 0, 1, 3, 6, 9, 7, 10, 8, 4, 3, 4, 0]
    right_hand = np.column_stack(
        [signal, np.zeros_like(signal), np.zeros_like(signal)]
    )
    left_hand = np.full((30, 3), 100.0, dtype=np.float64)
    pellet = np.zeros((30, 3), dtype=np.float64)
    pellet_conf = np.concatenate([np.ones(15), np.zeros(15)])

    reaches, details = detect_reaches_absolute(
        signal,
        right_hand=right_hand,
        left_hand=left_hand,
        pellet=pellet,
        right_hand_confidence=np.ones(30),
        left_hand_confidence=np.ones(30),
        pellet_confidence=pellet_conf,
        threshold=5.0,
        peak_prominence=1.0,
        start_padding=0,
        min_frame=3,
        max_frame=50,
        frame_rate=30.0,
        return_details=True,
    )

    assert len(reaches) == 1
    assert reaches[0].outcome.name == "GRABBED"
    assert details[0]["result"] == "GRABBED"
    assert details[0]["pellet_method"] == "right_hand"


def test_save_kpn_reach_details_csv_includes_confidence_columns(tmp_path):
    out_file = tmp_path / "_tmp_reach_details.csv"
    save_kpn_reach_details_csv(
        [
            {
                "frame": 1,
                "max_frame": 3,
                "end_frame": 5,
                "dur": 4,
                "result": "GRABBED",
                "rh_conf_median": 0.8,
                "pellet_conf_at_loss": 0.2,
            }
        ],
        out_file,
    )

    with out_file.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    assert rows[0]["rh_conf_median"] == "0.8"
    assert rows[0]["pellet_conf_at_loss"] == "0.2"


def test_projection_camera_matching_uses_embedded_camera_tokens():
    assert _match_camera_names(
        [
            "20260521_Person001_session001_Cam001-0000.mp4.predictions",
            "20260521_Person001_session001_Cam003-0000.mp4.predictions",
            "20260521_Person001_session001_Cam005-0000.mp4.predictions",
        ],
        [
            "20260521_Kat_session001_Cam001-0000",
            "20260521_Kat_session001_Cam002-0000",
            "20260521_Kat_session001_Cam003-0000",
            "20260521_Kat_session001_Cam004-0000",
            "20260521_Kat_session001_Cam005-0000",
            "20260521_Kat_session001_Cam006-0000",
        ],
    ) == [
        "20260521_Kat_session001_Cam001-0000",
        "20260521_Kat_session001_Cam003-0000",
        "20260521_Kat_session001_Cam005-0000",
    ]


def test_projection_scores_expand_to_full_reprojection_camera_set():
    scores = np.asarray(
        [
            [[0.2, 0.3], [0.4, 0.5]],
            [[0.6, 0.7], [0.8, 0.9]],
        ],
        dtype=np.float64,
    )
    expanded = _scores_for_reprojection_cameras(
        scores,
        n_reprojection_views=5,
        input_to_reprojection=[1, 4],
    )

    assert expanded.shape == (5, 2, 2)
    np.testing.assert_allclose(expanded[1], scores[0])
    np.testing.assert_allclose(expanded[4], scores[1])
    assert np.all(np.isnan(expanded[0]))
    assert _camera_index_map(["Cam002", "Cam005"], ["Cam001", "Cam002", "Cam005"]) == [1, 2]


def test_vectorized_projection_writes_full_reprojection_camera_set():
    class FakeTriangulationCalibration:
        def triangulate(self, points):
            return np.column_stack(
                [
                    np.nanmean(points[:, :, 0], axis=0),
                    np.nanmean(points[:, :, 1], axis=0),
                    np.zeros(points.shape[1]),
                ]
            )

    class FakeReprojectionCalibration:
        def project(self, points3d):
            return np.stack(
                [points3d[:, :2] + idx for idx in range(6)],
                axis=0,
            )

    points2d = np.full((3, 2, 2, 2), np.nan, dtype=np.float64)
    points2d[:, :, :, :] = np.asarray(
        [
            [[[0, 0], [1, 1]], [[2, 2], [3, 3]]],
            [[[10, 10], [11, 11]], [[12, 12], [13, 13]]],
            [[[20, 20], [21, 21]], [[22, 22], [23, 23]]],
        ],
        dtype=np.float64,
    )
    points3d = np.full((2, 2, 3), np.nan, dtype=np.float64)
    reprojections = np.full((6, 2, 2, 2), np.nan, dtype=np.float64)
    reprojection_error = np.full((6, 2, 2), np.nan, dtype=np.float64)

    assert _triangulate_vectorized(
        FakeTriangulationCalibration(),
        FakeReprojectionCalibration(),
        points2d,
        points3d,
        reprojections,
        reprojection_error,
        input_to_reprojection=[0, 2, 4],
    )

    assert reprojections.shape == (6, 2, 2, 2)
    np.testing.assert_allclose(reprojections[5, 0, 0], [15, 15])
    np.testing.assert_allclose(points3d[0, 0], [10, 10, 0])
    assert np.all(np.isfinite(reprojection_error[[0, 2, 4]]))
    assert np.all(np.isnan(reprojection_error[[1, 3, 5]]))


def test_reaches_dock_matches_reprojection_camera_by_camera_token():
    class State:
        def get(self, key, default=None):
            return type(
                "Video",
                (),
                {"filename": "20260521_Person001_session001_Cam001-0000.mp4"},
            )()

    dock = ReachesDock.__new__(ReachesDock)
    dock.main_window = type("MainWindow", (), {"state": State(), "labels": None})()
    dock._points3d_reprojections = {
        "reprojections": np.zeros((6, 1, 1, 2), dtype=np.float64),
        "camera_names": [
            "20260521_Kat_session001_Cam001-0000",
            "20260521_Kat_session001_Cam002-0000",
            "20260521_Kat_session001_Cam003-0000",
            "20260521_Kat_session001_Cam004-0000",
            "20260521_Kat_session001_Cam005-0000",
            "20260521_Kat_session001_Cam006-0000",
        ],
        "metadata": {
            "prediction_files": [
                "20260521_Person001_session001_Cam006-0000.predictions.slp",
                "20260521_Person001_session001_Cam005-0000.predictions.slp",
                "20260521_Person001_session001_Cam004-0000.predictions.slp",
                "20260521_Person001_session001_Cam003-0000.predictions.slp",
                "20260521_Person001_session001_Cam002-0000.predictions.slp",
                "20260521_Person001_session001_Cam001-0000.predictions.slp",
            ]
        },
    }

    assert dock._points3d_reprojection_camera_index() == 0


def test_reaches_prediction_source_loads_analysis_h5(tmp_path, monkeypatch):
    src = tmp_path / "video.analysis.h5"
    with h5py.File(src, "w") as f:
        f.create_dataset("track_occupancy", data=np.ones((1, 1), dtype=np.uint8))

    calls = {}

    def fake_load_analysis_h5(path, video=None):
        calls["analysis"] = (path, video)
        return "labels"

    monkeypatch.setattr("sleap_io.load_analysis_h5", fake_load_analysis_h5)

    labels, kind = ReachesDock._load_prediction_source_labels(str(src), video="video.mp4")

    assert labels == "labels"
    assert kind == "analysis_h5"
    assert calls["analysis"] == (str(src), "video.mp4")


def test_reaches_prediction_source_loads_nwb(monkeypatch):
    calls = {}

    def fake_load_nwb(path):
        calls["nwb"] = path
        return "labels"

    monkeypatch.setattr("sleap_io.load_nwb", fake_load_nwb)

    labels, kind = ReachesDock._load_prediction_source_labels("predictions.nwb")

    assert labels == "labels"
    assert kind == "nwb"
    assert calls["nwb"] == "predictions.nwb"


def test_reaches_dock_upsert_reach_updates_table_and_sort(qtbot):
    dock = ReachesDock.__new__(ReachesDock)
    dock._reaches = [
        ReachSegment(20, 4, 9, ReachOutcome.GRABBED),
        ReachSegment(5, 2, 6, ReachOutcome.MISSED),
    ]
    dock._reach_details = [
        {"frame": 20, "max_frame": 24, "end_frame": 29, "result": "GRABBED"},
        {"frame": 5, "max_frame": 7, "end_frame": 11, "result": "MISSED"},
    ]
    dock._results_table = QTableWidget(0, 5)
    dock._status_label = QLabel()
    calls = []
    dock.on_reaches_changed = lambda reaches: calls.append(list(reaches))

    dock._sort_reaches_and_details()
    dock._populate_table()

    dock.upsert_reach(0, 12, 15, 19)

    assert [(r.frame, r.max_frame, r.end_frame, r.outcome) for r in dock._reaches] == [
        (12, 15, 19, ReachOutcome.MISSED),
        (20, 24, 29, ReachOutcome.GRABBED),
    ]
    assert dock._results_table.item(0, 0).text() == "13"
    assert dock._results_table.item(0, 1).text() == "16"
    assert dock._reach_details[0]["curation"] == "manual_edit"
    assert dock._reach_details[0]["result"] == "MISSED"
    assert calls[-1] == dock._reaches

    dock.upsert_reach(-1, 2, 4, 7)

    assert [r.frame for r in dock._reaches] == [2, 12, 20]
    assert dock._results_table.item(0, 0).text() == "3"
    assert dock._reach_details[0]["curation"] == "manual_add"


def test_reaches_dock_filter_hand_trajectories_preserves_nan(qtbot):
    dock = ReachesDock.__new__(ReachesDock)
    dock._filter_hand_traces = QCheckBox()
    dock._filter_hand_traces.setChecked(True)
    dock._filter_cutoff = QDoubleSpinBox()
    dock._filter_cutoff.setRange(0.1, 10000.0)
    dock._filter_cutoff.setValue(30.0)
    dock._filter_sampling_label = QLabel()

    t = np.linspace(0, 2 * np.pi, 80)
    signal = np.column_stack(
        [
            np.sin(t) + 0.25 * np.sin(20 * t),
            np.cos(t) + 0.25 * np.cos(20 * t),
            np.sin(0.5 * t),
        ]
    )
    lh = signal.copy()
    rh = signal.copy()
    rh[10, 0] = np.nan

    filtered_lh, filtered_rh, info = dock._filter_hand_trajectories(
        lh, rh, frame_rate=150.0
    )

    assert info["enabled"]
    assert info["cutoff_frequency_hz"] == 30.0
    assert info["sampling_rate_hz"] == 150.0
    assert info["normalization"] == "cutoff_frequency_hz / sampling_rate_hz"
    assert np.isnan(filtered_rh[10, 0])
    assert not np.allclose(filtered_lh[:, 0], lh[:, 0])
    assert dock._filter_sampling_label.text() == "150 Hz"


def test_reaches_dock_builds_parameter_traces():
    dock = ReachesDock.__new__(ReachesDock)
    right_hand = np.array(
        [
            [3.0, 4.0, 0.0],
            [6.0, 8.0, 0.0],
            [9.0, 12.0, 0.0],
        ],
        dtype=np.float64,
    )
    pellet = np.zeros((3, 3), dtype=np.float64)
    rh_conf = np.array([1.0, 0.2, 1.0], dtype=np.float64)
    pellet_conf = np.ones(3, dtype=np.float64)

    traces = dock._make_reach_parameter_traces(
        right_hand=right_hand,
        pellet=pellet,
        right_hand_confidence=rh_conf,
        pellet_confidence=pellet_conf,
        pellet_nodes=["pellet"],
        confidence=0.5,
    )

    assert [trace["name"] for trace in traces] == ["RH pellet dist", "RH x"]
    assert traces[0]["color"] == "#a3e635"
    assert traces[1]["color"] == "#34d399"
    np.testing.assert_allclose(
        traces[0]["values"], [5.0, np.nan, 15.0], equal_nan=True
    )
    np.testing.assert_allclose(
        traces[1]["values"], [3.0, np.nan, 9.0], equal_nan=True
    )


def test_translate_points3d_h5_uses_selected_node_as_frame_origin(tmp_path):
    src = tmp_path / "points3D.h5"
    points = np.array(
        [
            [[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]],
            [[2.0, 4.0, 6.0], [11.0, 22.0, 33.0]],
        ],
        dtype=np.float64,
    )
    with h5py.File(src, "w") as f:
        f.create_dataset("points3D", data=points)
        f.create_dataset(
            "node_names",
            data=np.asarray(["bar_R", "hand"], dtype=object),
            dtype=h5py.string_dtype(encoding="utf-8"),
        )

    metadata = translate_points3d_h5(src, "bar_R", mode="frame")

    out = Path(metadata["points3d_path"])
    assert out.name == "points3D_translated.h5"
    with h5py.File(out, "r") as f:
        translated = f["points3D"][:]
        np.testing.assert_allclose(translated[:, 0], 0.0)
        np.testing.assert_allclose(translated[:, 1], points[:, 1] - points[:, 0])
        assert f.attrs["translation_origin_node"] == "bar_R"


def test_load_points3d_h5_accepts_matlab_tracks_shape(tmp_path):
    src = tmp_path / "points3D.h5"
    tracks = np.zeros((3, 2, 1, 4), dtype=np.float64)
    tracks[:, 0, 0, :] = np.asarray(
        [
            [1.0, 2.0, 3.0, 4.0],
            [10.0, 20.0, 30.0, 40.0],
            [100.0, 200.0, 300.0, 400.0],
        ]
    )
    tracks[:, 1, 0, :] = tracks[:, 0, 0, :] + 1.0
    with h5py.File(src, "w") as f:
        f.create_dataset("tracks", data=tracks)

    loaded = load_points3d_h5(src)

    assert loaded["points3d"].shape == (4, 2, 3)
    np.testing.assert_allclose(loaded["points3d"][0, 0], [1.0, 10.0, 100.0])
    np.testing.assert_allclose(loaded["points3d"][3, 1], [5.0, 41.0, 401.0])
