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
