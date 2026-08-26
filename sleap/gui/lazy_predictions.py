"""Compact external prediction storage for responsive GUI overlays."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import sleap_io as sio
from sleap_io import PredictedInstance, Track, Video


def video_key(video_or_filename) -> str:
    """Return a stable, filename-based key for matching prediction videos."""
    filename = getattr(video_or_filename, "filename", video_or_filename)
    if isinstance(filename, list):
        filename = filename[0] if filename else ""
    filename = str(filename or "")
    return Path(filename).name.casefold()


@dataclass
class VideoPredictionBlock:
    """Compact prediction arrays for one video in an external prediction file."""

    video_name: str
    video_key: str
    source_path: str
    skeleton: object
    video_frame_count: int
    frame_ranges: Dict[int, Tuple[int, int]]
    points: np.ndarray
    point_scores: np.ndarray
    instance_scores: np.ndarray
    track_names: List[Optional[str]]

    @property
    def frame_count(self) -> int:
        return self.video_frame_count

    @property
    def max_instances(self) -> int:
        return max(
            (stop - start for start, stop in self.frame_ranges.values()), default=0
        )

    @property
    def node_count(self) -> int:
        return int(self.points.shape[1])

    @property
    def instance_count(self) -> int:
        return int(self.points.shape[0])

    def instances_for_frame(self, frame_idx: int) -> List[PredictedInstance]:
        """Create temporary prediction instances for one frame."""
        if frame_idx < 0 or frame_idx >= self.frame_count:
            return []

        frame_range = self.frame_ranges.get(int(frame_idx))
        if frame_range is None:
            return []

        instances = []
        for inst_idx in range(*frame_range):
            points = self.points[inst_idx]
            if not np.isfinite(points[..., 0]).any():
                continue

            score = self.instance_scores[inst_idx]
            if not np.isfinite(score):
                finite_scores = self.point_scores[inst_idx]
                finite_scores = finite_scores[np.isfinite(finite_scores)]
                score = float(np.mean(finite_scores)) if len(finite_scores) else 0.0
                if not np.isfinite(score):
                    score = 0.0

            track_name = self.track_names[inst_idx]
            track = Track(name=str(track_name)) if track_name else None
            instances.append(
                PredictedInstance.from_numpy(
                    points.astype(np.float64, copy=False),
                    skeleton=self.skeleton,
                    point_scores=self.point_scores[inst_idx].astype(
                        np.float64, copy=False
                    ),
                    score=float(score),
                    track=track,
                )
            )
        return instances


@dataclass
class ExternalPredictionSet:
    """Compact prediction data loaded from a SLEAP prediction file."""

    source_path: str
    blocks: List[VideoPredictionBlock] = field(default_factory=list)
    target_video_key: Optional[str] = None
    target_video_name: Optional[str] = None

    @classmethod
    def from_file(cls, path: str) -> "ExternalPredictionSet":
        """Load SLP or Analysis HDF5 predictions and compact them into arrays."""
        path = str(path)
        suffix = Path(path).suffix.lower()
        is_analysis_h5 = False
        if suffix in {".h5", ".hdf5"}:
            import h5py

            try:
                with h5py.File(path, "r") as file:
                    is_analysis_h5 = "track_occupancy" in file
            except OSError:
                pass

        if is_analysis_h5:
            labels = sio.load_analysis_h5(path)
        else:
            labels = sio.load_slp(path)
        return cls.from_labels(labels, source_path=path)

    @classmethod
    def from_labels(cls, labels, source_path: str = "") -> "ExternalPredictionSet":
        """Create a compact prediction set from a loaded labels object."""
        by_video: Dict[Video, List] = {}
        for lf in getattr(labels, "labeled_frames", []) or []:
            predictions = [
                inst
                for inst in getattr(lf, "instances", []) or []
                if isinstance(inst, PredictedInstance)
            ]
            if predictions:
                by_video.setdefault(lf.video, []).append((lf.frame_idx, predictions))

        blocks = []
        for video, frame_predictions in by_video.items():
            block = _build_video_block(video, frame_predictions, source_path)
            if block is not None:
                blocks.append(block)

        return cls(source_path=str(source_path), blocks=blocks)

    @property
    def total_instances(self) -> int:
        return sum(block.instance_count for block in self.blocks)

    def assign_to_video(self, video: Video) -> None:
        """Pin this prediction file to a single project video."""
        self.target_video_key = video_key(video)
        filename = getattr(video, "filename", "")
        if isinstance(filename, list):
            filename = filename[0] if filename else ""
        self.target_video_name = Path(str(filename or "video")).name

    def instances_for(self, video: Video, frame_idx: int) -> List[PredictedInstance]:
        """Return temporary prediction instances matching a video/frame."""
        key = video_key(video)
        if self.target_video_key is not None and key != self.target_video_key:
            return []

        matches = [block for block in self.blocks if block.video_key == key]
        if not matches and self.target_video_key == key and len(self.blocks) == 1:
            matches = self.blocks

        instances = []
        for block in matches:
            instances.extend(block.instances_for_frame(frame_idx))
        return instances


class ExternalPredictionManager:
    """Registry of compact external prediction sets used by the video overlay."""

    def __init__(self):
        self._sets: List[ExternalPredictionSet] = []

    def __len__(self) -> int:
        return len(self._sets)

    @property
    def source_paths(self) -> List[str]:
        return [prediction_set.source_path for prediction_set in self._sets]

    def add(
        self, prediction_set: ExternalPredictionSet, target_video: Optional[Video] = None
    ) -> None:
        if target_video is not None:
            prediction_set.assign_to_video(target_video)
        self.remove_source(prediction_set.source_path)
        self._sets.append(prediction_set)

    def clear(self) -> None:
        self._sets.clear()

    def remove_source(self, source_path: str) -> None:
        source_path = str(source_path)
        self._sets = [
            prediction_set
            for prediction_set in self._sets
            if prediction_set.source_path != source_path
        ]

    def instances_for(self, video: Video, frame_idx: int) -> List[PredictedInstance]:
        instances = []
        for prediction_set in self._sets:
            instances.extend(prediction_set.instances_for(video, frame_idx))
        return instances


def get_external_prediction_manager(state) -> ExternalPredictionManager:
    """Return the GUI state's external prediction manager, creating it if needed."""
    manager = state.get("external predictions", default=None)
    if manager is None:
        manager = ExternalPredictionManager()
        state["external predictions"] = manager
    return manager


def _build_video_block(
    video: Video, frame_predictions: Iterable, source_path: str
) -> Optional[VideoPredictionBlock]:
    frame_predictions = list(frame_predictions)
    if not frame_predictions:
        return None

    first_instance = frame_predictions[0][1][0]
    skeleton = first_instance.skeleton
    node_count = len(skeleton.nodes)
    max_frame_idx = max(frame_idx for frame_idx, _ in frame_predictions)
    # The preview only needs prediction indices. Avoid calling len(video) here:
    # that may open a decoder in this background-loading thread while the GUI's
    # video worker is reading the same media.
    frame_count = int(max_frame_idx) + 1

    # Store only predictions that actually exist. The previous representation
    # allocated frame_count * max_instances slots, including empty frames and
    # unused instance slots. Full-length, multi-camera videos could therefore
    # exhaust system memory while merely linking a preview file.
    records = []
    for frame_idx, predictions in sorted(frame_predictions, key=lambda item: item[0]):
        if frame_idx < 0 or frame_idx >= frame_count:
            continue
        for inst in predictions:
            if inst.skeleton is not skeleton:
                continue
            xy = np.asarray(inst.numpy(), dtype=np.float32)
            if not np.isfinite(xy[..., 0]).any():
                continue
            records.append((int(frame_idx), inst))

    if not records:
        return None

    points = np.full((len(records), node_count, 2), np.nan, dtype=np.float32)
    point_scores = np.full((len(records), node_count), np.nan, dtype=np.float32)
    instance_scores = np.full(len(records), np.nan, dtype=np.float32)
    track_names: List[Optional[str]] = [None] * len(records)
    frame_ranges: Dict[int, Tuple[int, int]] = {}

    for inst_idx, (frame_idx, inst) in enumerate(records):
        xy = np.asarray(inst.numpy(), dtype=np.float32)
        points[inst_idx, : min(node_count, len(xy))] = xy[:node_count]
        scores = _point_scores(inst, node_count)
        point_scores[inst_idx, : len(scores)] = scores
        score = getattr(inst, "score", np.nan)
        instance_scores[inst_idx] = score if score is not None else np.nan
        track = getattr(inst, "track", None)
        if track is not None and getattr(track, "name", None):
            track_names[inst_idx] = track.name

        if frame_idx in frame_ranges:
            start, _ = frame_ranges[frame_idx]
            frame_ranges[frame_idx] = (start, inst_idx + 1)
        else:
            frame_ranges[frame_idx] = (inst_idx, inst_idx + 1)

    filename = getattr(video, "filename", "")
    if isinstance(filename, list):
        filename = filename[0] if filename else ""
    video_name = Path(str(filename or "video")).name
    return VideoPredictionBlock(
        video_name=video_name,
        video_key=video_key(video),
        source_path=str(source_path),
        skeleton=skeleton,
        video_frame_count=frame_count,
        frame_ranges=frame_ranges,
        points=points,
        point_scores=point_scores,
        instance_scores=instance_scores,
        track_names=track_names,
    )


def _point_scores(instance: PredictedInstance, node_count: int) -> np.ndarray:
    scores = np.full(node_count, np.nan, dtype=np.float32)
    points = getattr(instance, "points", None)
    if points is not None and "score" in points.dtype.names:
        raw_scores = np.asarray(points["score"], dtype=np.float32)
        scores[: min(node_count, len(raw_scores))] = raw_scores[:node_count]
    else:
        xy = np.asarray(instance.numpy(), dtype=np.float32)
        scores[: min(node_count, len(xy))] = np.isfinite(xy[:node_count, 0]).astype(
            np.float32
        )
    return scores
