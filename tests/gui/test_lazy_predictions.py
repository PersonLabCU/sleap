import h5py
import numpy as np
import sleap_io as sio

from sleap.gui.lazy_predictions import ExternalPredictionManager, ExternalPredictionSet
from sleap_io.model.instance import PredictedInstance


def _prediction_set(video_filename: str, x: float = 1.0) -> ExternalPredictionSet:
    video = sio.Video.from_filename(video_filename)
    skeleton = sio.Skeleton()
    skeleton.add_node("node")
    instance = PredictedInstance.from_numpy(
        np.array([[x, x]], dtype=np.float32),
        skeleton=skeleton,
        point_scores=np.array([1.0], dtype=np.float32),
        score=1.0,
    )
    labels = sio.Labels(
        [sio.LabeledFrame(video=video, frame_idx=0, instances=[instance])]
    )
    return ExternalPredictionSet.from_labels(
        labels, source_path=f"{video_filename}.predictions.slp"
    )


def test_external_prediction_set_matches_exact_video_only():
    predictions = _prediction_set("side.mp4")
    front_video = sio.Video.from_filename("front.mp4")
    side_video = sio.Video.from_filename("side.mp4")

    assert predictions.instances_for(front_video, 0) == []
    assert len(predictions.instances_for(side_video, 0)) == 1


def test_external_prediction_set_does_not_fallback_to_all_videos():
    predictions = _prediction_set("project.slp")
    front_video = sio.Video.from_filename("front.mp4")
    side_video = sio.Video.from_filename("side.mp4")

    assert predictions.instances_for(front_video, 0) == []
    assert predictions.instances_for(side_video, 0) == []


def test_external_prediction_manager_can_pin_file_to_one_video():
    predictions = _prediction_set("project.slp")
    front_video = sio.Video.from_filename("front.mp4")
    side_video = sio.Video.from_filename("side.mp4")
    manager = ExternalPredictionManager()

    manager.add(predictions, target_video=side_video)

    assert manager.instances_for(front_video, 0) == []
    assert len(manager.instances_for(side_video, 0)) == 1


def test_external_prediction_set_loads_analysis_h5(tmp_path, monkeypatch):
    source = tmp_path / "side.analysis.h5"
    with h5py.File(source, "w") as file:
        file.create_dataset("track_occupancy", data=np.ones((1, 1)))

    expected = _prediction_set("side.mp4")
    video = sio.Video.from_filename("side.mp4")
    labels = sio.Labels(
        [
            sio.LabeledFrame(
                video=video,
                frame_idx=0,
                instances=expected.blocks[0].instances_for_frame(0),
            )
        ]
    )
    calls = []

    def fake_load_analysis_h5(path):
        calls.append(path)
        return labels

    monkeypatch.setattr(sio, "load_analysis_h5", fake_load_analysis_h5)

    loaded = ExternalPredictionSet.from_file(str(source))

    assert calls == [str(source)]
    assert loaded.source_path == str(source)
    assert loaded.total_instances == 1


def test_external_prediction_set_uses_sparse_frame_storage():
    """Preview memory should scale with predictions, not the video frame count."""
    video = sio.Video.from_filename("long_video.mp4")
    skeleton = sio.Skeleton()
    skeleton.add_node("node")
    instance = PredictedInstance.from_numpy(
        np.array([[5.0, 6.0]], dtype=np.float32),
        skeleton=skeleton,
        point_scores=np.array([0.8], dtype=np.float32),
        score=0.9,
    )
    labels = sio.Labels(
        [sio.LabeledFrame(video=video, frame_idx=100_000, instances=[instance])]
    )

    predictions = ExternalPredictionSet.from_labels(labels)
    block = predictions.blocks[0]

    assert block.frame_count == 100_001
    assert block.points.shape == (1, 1, 2)
    assert block.point_scores.shape == (1, 1)
    assert block.instances_for_frame(99_999) == []
    assert len(block.instances_for_frame(100_000)) == 1
