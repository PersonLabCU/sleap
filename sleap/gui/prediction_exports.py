"""Helpers for exporting prediction labels to analysis-friendly formats."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional

import sleap_io as sio
from sleap_io import Labels, Video


def export_prediction_formats(
    labels: Labels,
    *,
    source_path: str | Path,
    output_dir: str | Path,
    output_prefix: str | None = None,
    videos: Optional[Iterable[Video]] = None,
    analysis_h5: bool = False,
    nwb: bool = False,
) -> List[str]:
    """Export prediction labels to selected analysis formats.

    Analysis HDF5 is written once per video because the format stores one video
    trajectory array per file. NWB is written once for the whole labels object.
    """
    written: List[str] = []
    if not analysis_h5 and not nwb:
        return written

    source_path = Path(source_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = output_prefix or _default_prefix(source_path)

    if analysis_h5:
        for video in _videos_with_frames(labels, videos):
            video_idx = labels.videos.index(video) if video in labels.videos else 0
            video_stem = _video_stem(video)
            output_path = output_dir / (
                f"{prefix}.{video_idx:03}_{video_stem}.analysis.h5"
            )
            sio.save_analysis_h5(
                labels,
                str(output_path),
                video=video,
                labels_path=str(source_path),
                all_frames=True,
                preset="matlab",
            )
            written.append(str(output_path))

    if nwb:
        output_path = output_dir / f"{prefix}.nwb"
        sio.save_nwb(labels=labels, filename=str(output_path))
        written.append(str(output_path))

    return written


def export_prediction_file_formats(
    prediction_path: str | Path,
    *,
    analysis_h5: bool = False,
    nwb: bool = False,
) -> List[str]:
    """Load a prediction SLP and export selected formats next to it."""
    prediction_path = Path(prediction_path)
    labels = sio.load_slp(str(prediction_path))
    return export_prediction_formats(
        labels,
        source_path=prediction_path,
        output_dir=prediction_path.parent,
        output_prefix=prediction_path.stem,
        analysis_h5=analysis_h5,
        nwb=nwb,
    )


def _videos_with_frames(
    labels: Labels,
    videos: Optional[Iterable[Video]],
) -> List[Video]:
    selected = list(videos) if videos is not None else list(labels.videos)
    with_frames = [video for video in selected if labels.find(video)]
    if with_frames or videos is None:
        return with_frames

    # After GUI inference, merged prediction frames may refer to equivalent
    # video objects rather than the exact selected objects. Avoid silently
    # writing no analysis H5 files in that case.
    selected_keys = {_video_key(video) for video in selected}
    return [
        video
        for video in labels.videos
        if labels.find(video) and _video_key(video) in selected_keys
    ]


def _default_prefix(source_path: Path) -> str:
    if source_path.name:
        return source_path.stem
    return "predictions"


def _video_stem(video: Video) -> str:
    filename = video.filename
    if isinstance(filename, list):
        filename = filename[0] if filename else ""
    return Path(str(filename)).stem if filename else "video"


def _video_key(video: Video) -> str:
    filename = video.filename
    if isinstance(filename, list):
        filename = filename[0] if filename else ""
    return str(filename or "")
