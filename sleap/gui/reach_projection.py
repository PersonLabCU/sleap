"""Utilities for creating and loading fixed-camera 3D reach trajectories."""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


def run_3d_projection_export(
    prediction_files: Sequence[str | Path],
    calibration_path: str | Path,
    output_dir: str | Path,
    *,
    camera_names: Optional[Sequence[str]] = None,
    points3d_filename: str = "points3d.h5",
    reprojections_filename: str = "reprojections.h5",
    h5_compression: Optional[str] = "lzf",
    progress_callback: Optional[Callable[[str, int, int, str], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> Dict[str, Any]:
    """Triangulate SLEAP prediction files and export points3D/reprojections H5s."""
    def report(stage: str, current: int, total: int, detail: str = "") -> None:
        if progress_callback is not None:
            progress_callback(stage, current, total, detail)

    def raise_if_canceled() -> None:
        if cancel_check is not None and cancel_check():
            raise InterruptedError("3D projection was canceled.")

    prediction_files = [Path(p).expanduser() for p in prediction_files]
    if len(prediction_files) < 2:
        raise ValueError("Select at least two predictions files for 3D projection.")
    missing = [str(p) for p in prediction_files if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing predictions file(s): {', '.join(missing)}")

    calibration_path = Path(calibration_path).expanduser()
    if not calibration_path.exists() or calibration_path.suffix.lower() != ".toml":
        raise FileNotFoundError("Select a valid calibration.toml file.")

    try:
        from aniposelib.cameras import CameraGroup as AniposeCameraGroup
    except ImportError as exc:
        raise ImportError(
            "aniposelib is not installed in this environment. "
            "Install the SLEAP anipose extras before running 3D projections."
        ) from exc

    try:
        import h5py
        import sleap_io
    except ImportError as exc:
        raise ImportError(f"Required dependency is missing: {exc}") from exc

    report("Loading calibration", 0, 1, str(calibration_path))
    raise_if_canceled()
    calibration = AniposeCameraGroup.load(str(calibration_path))
    report("Loading calibration", 1, 1, "Calibration loaded.")

    view_data = []
    for pred_idx, pred_file in enumerate(prediction_files, start=1):
        raise_if_canceled()
        report(
            "Loading predictions",
            pred_idx - 1,
            len(prediction_files),
            pred_file.name,
        )
        labels = sleap_io.load_slp(str(pred_file))
        if not labels.videos:
            raise ValueError(f"No videos found in {pred_file}.")
        video = labels.videos[0]
        node_names = list(labels.skeletons[0].node_names)
        points2d, scores = _extract_prediction_points(labels, video, node_names)
        view_data.append(
            {
                "file": pred_file,
                "labels": labels,
                "video": video,
                "node_names": node_names,
                "points2d": points2d,
                "scores": scores,
            }
        )
        report(
            "Loading predictions",
            pred_idx,
            len(prediction_files),
            f"{pred_file.name}: {points2d.shape[0]:,} frames",
        )

    report("Preparing trajectories", 0, 1, "Aligning shared skeleton nodes.")
    raise_if_canceled()
    node_names = _shared_node_names([item["node_names"] for item in view_data])
    if not node_names:
        raise ValueError("Prediction files do not share any skeleton node names.")
    node_indices = [
        [item["node_names"].index(name) for name in node_names] for item in view_data
    ]

    n_views = len(view_data)
    n_frames = max(item["points2d"].shape[0] for item in view_data)
    n_nodes = len(node_names)
    points2d = np.full((n_views, n_frames, n_nodes, 2), np.nan, dtype=np.float64)
    scores = np.full((n_views, n_frames, n_nodes), np.nan, dtype=np.float64)
    source_frame_counts = []
    for view_idx, item in enumerate(view_data):
        src_points = item["points2d"][:, node_indices[view_idx], :]
        src_scores = item["scores"][:, node_indices[view_idx]]
        points2d[view_idx, : src_points.shape[0]] = src_points
        scores[view_idx, : src_scores.shape[0]] = src_scores
        source_frame_counts.append(int(src_points.shape[0]))

    session_calibration, used_camera_names = _subset_calibration(
        calibration,
        n_views,
        camera_names
        or [_video_stem(item["video"]) for item in view_data]
        or [p.stem for p in prediction_files],
    )
    reprojection_calibration = calibration
    reprojection_camera_names = list(calibration.get_names()) or [
        str(i) for i in range(len(calibration.cameras))
    ]
    n_reprojection_views = len(reprojection_camera_names)
    input_to_reprojection = _camera_index_map(used_camera_names, reprojection_camera_names)

    points3d = np.full((n_frames, n_nodes, 3), np.nan, dtype=np.float64)
    reprojections = np.full(
        (n_reprojection_views, n_frames, n_nodes, 2),
        np.nan,
        dtype=np.float64,
    )
    reprojection_error = np.full(
        (n_reprojection_views, n_frames, n_nodes),
        np.nan,
        dtype=np.float64,
    )

    report(
        "Triangulating",
        0,
        1,
        f"{n_frames:,} frames, {n_nodes} nodes, {n_views} cameras",
    )
    raise_if_canceled()
    if not _triangulate_vectorized(
        session_calibration,
        reprojection_calibration,
        points2d,
        points3d,
        reprojections,
        reprojection_error,
        input_to_reprojection,
    ):
        progress_step = max(1, n_frames // 200)
        for frame_idx in range(n_frames):
            if frame_idx % progress_step == 0:
                raise_if_canceled()
                report(
                    "Triangulating",
                    frame_idx,
                    n_frames,
                    f"Frame {frame_idx + 1:,} of {n_frames:,}",
                )
            frame_points = points2d[:, frame_idx, :, :]
            if _count_valid_views(frame_points) < 2:
                continue
            frame_points3d = session_calibration.triangulate(frame_points)
            frame_reproj = reprojection_calibration.project(frame_points3d)
            points3d[frame_idx] = frame_points3d
            reprojections[:, frame_idx] = frame_reproj
            for input_idx, reproj_idx in enumerate(input_to_reprojection):
                if reproj_idx is None:
                    continue
                reprojection_error[reproj_idx, frame_idx] = np.linalg.norm(
                    frame_reproj[reproj_idx] - frame_points[input_idx],
                    axis=1,
                )
    report("Triangulating", n_frames, n_frames, "Triangulation complete.")

    raise_if_canceled()
    report("Writing outputs", 0, 2, str(output_dir))
    output_dir = Path(output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    points3d_path = output_dir / points3d_filename
    reproj_path = output_dir / reprojections_filename

    metadata = {
        "source": "sleap_gui_3d_projections",
        "calibration_path": str(calibration_path),
        "points3d_path": str(points3d_path),
        "reprojections_path": str(reproj_path),
        "prediction_files": [str(p) for p in prediction_files],
        "camera_names": list(used_camera_names),
        "reprojection_camera_names": list(reprojection_camera_names),
        "node_names": list(node_names),
        "n_frames": int(n_frames),
        "n_nodes": int(n_nodes),
        "n_views": int(n_views),
        "n_reprojection_views": int(n_reprojection_views),
        "source_frame_counts": source_frame_counts,
        "coordinate_system": "calibration",
        "units": "mm",
        "h5_compression": h5_compression or "none",
    }

    str_dtype = h5py.string_dtype(encoding="utf-8")
    compression_kwargs = {"compression": h5_compression} if h5_compression else {}
    with h5py.File(points3d_path, "w") as f:
        f.create_dataset("points3D", data=points3d, **compression_kwargs)
        f.create_dataset("node_names", data=np.asarray(node_names, dtype=object), dtype=str_dtype)
        f.create_dataset("frame_indices", data=np.arange(n_frames, dtype=np.int64))
        f.create_dataset(
            "source_files",
            data=np.asarray([str(p) for p in prediction_files], dtype=object),
            dtype=str_dtype,
        )
        f.create_dataset(
            "camera_names",
            data=np.asarray(list(used_camera_names), dtype=object),
            dtype=str_dtype,
        )
        f.attrs["metadata_json"] = json.dumps(metadata)
    report("Writing outputs", 1, 2, points3d_path.name)

    with h5py.File(reproj_path, "w") as f:
        f.create_dataset("reprojections", data=reprojections, **compression_kwargs)
        f.create_dataset("input_points2D", data=points2d, **compression_kwargs)
        f.create_dataset(
            "point_scores",
            data=_scores_for_reprojection_cameras(
                scores,
                n_reprojection_views,
                input_to_reprojection,
            ),
            **compression_kwargs,
        )
        f.create_dataset("reprojection_error", data=reprojection_error, **compression_kwargs)
        f.create_dataset("node_names", data=np.asarray(node_names, dtype=object), dtype=str_dtype)
        f.create_dataset(
            "camera_names",
            data=np.asarray(list(reprojection_camera_names), dtype=object),
            dtype=str_dtype,
        )
        f.create_dataset(
            "source_files",
            data=np.asarray([str(p) for p in prediction_files], dtype=object),
            dtype=str_dtype,
        )
        f.create_dataset("frame_indices", data=np.arange(n_frames, dtype=np.int64))
        f.attrs["metadata_json"] = json.dumps(metadata)
    report("Writing outputs", 2, 2, reproj_path.name)

    report("Complete", 1, 1, "3D projection export complete.")
    return metadata


def load_points3d_h5(path: str | Path) -> Dict[str, Any]:
    """Load a points3D H5 file produced by :func:`run_3d_projection_export`."""
    try:
        import h5py
    except ImportError as exc:
        raise ImportError("h5py is required to load points3D files.") from exc

    path = Path(path).expanduser()
    with h5py.File(path, "r") as f:
        if "points3D" in f:
            points = np.asarray(f["points3D"], dtype=np.float64)
        elif "points3d" in f:
            points = np.asarray(f["points3d"], dtype=np.float64)
        elif "tracks" in f:
            points = np.asarray(f["tracks"], dtype=np.float64)
        else:
            raise ValueError("H5 file does not contain points3D, points3d, or tracks.")

        if "node_names" in f:
            node_names = _decode_strings(np.asarray(f["node_names"]))
        else:
            node_names = [f"node_{i}" for i in range(_infer_points3d_node_count(points))]

        metadata = {}
        if "metadata_json" in f.attrs:
            try:
                metadata = json.loads(f.attrs["metadata_json"])
            except Exception:
                metadata = {}
        if "camera_names" in f and "camera_names" not in metadata:
            metadata["camera_names"] = _decode_strings(np.asarray(f["camera_names"]))
        if "source_files" in f and "prediction_files" not in metadata:
            metadata["prediction_files"] = _decode_strings(np.asarray(f["source_files"]))

    points = _normalize_points3d_shape(points, len(node_names))
    return {
        "path": str(path),
        "points3d": points,
        "node_names": list(node_names),
        "metadata": metadata,
    }


def translate_points3d_h5(
    path: str | Path,
    origin_node: str,
    *,
    output_dir: str | Path | None = None,
    output_filename: str = "points3d_translated.h5",
    mode: str = "frame",
    h5_compression: Optional[str] = None,
) -> Dict[str, Any]:
    """Save a copy of a points3d H5 with coordinates translated to a node origin.

    Args:
        path: Source H5 containing ``points3D``.
        origin_node: Node whose 3D coordinate should become the origin.
        output_dir: Directory for the translated file. Defaults to source folder.
        output_filename: Name of translated H5 file.
        mode: ``"frame"`` subtracts the selected node position independently for
            each frame. ``"median"`` subtracts the median selected-node position.
        h5_compression: Optional HDF5 compression for the translated points dataset.

    Returns:
        Metadata describing the translated output.
    """
    try:
        import h5py
    except ImportError as exc:
        raise ImportError("h5py is required to translate points3D files.") from exc

    source_path = Path(path).expanduser()
    if not source_path.exists():
        raise FileNotFoundError(f"Missing points3D file: {source_path}")

    output_dir = Path(output_dir).expanduser() if output_dir else source_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / output_filename
    if output_path.resolve() == source_path.resolve():
        raise ValueError("Translated output path must be different from the source file.")

    loaded = load_points3d_h5(source_path)
    points = np.asarray(loaded["points3d"], dtype=np.float64)
    node_names = list(loaded["node_names"])
    if origin_node not in node_names:
        raise ValueError(f"Node '{origin_node}' was not found in {source_path.name}.")

    node_idx = node_names.index(origin_node)
    origin_trace = points[:, node_idx, :3]
    mode = str(mode or "frame").casefold()
    if mode == "median":
        finite = np.all(np.isfinite(origin_trace), axis=1)
        if not np.any(finite):
            raise ValueError(f"Node '{origin_node}' has no finite 3D positions.")
        origin_xyz = np.nanmedian(origin_trace[finite], axis=0)
        translated = points - origin_xyz[None, None, :]
        origin_for_metadata: Any = origin_xyz.tolist()
    elif mode == "frame":
        translated = points - origin_trace[:, None, :]
        origin_for_metadata = "per-frame"
    else:
        raise ValueError("mode must be 'frame' or 'median'.")

    str_dtype = h5py.string_dtype(encoding="utf-8")
    compression_kwargs = {"compression": h5_compression} if h5_compression else {}
    with h5py.File(source_path, "r") as src, h5py.File(output_path, "w") as dst:
        wrote_points = False
        for key in src.keys():
            if key in {"points3D", "points3d", "tracks"}:
                if not wrote_points:
                    dst.create_dataset("points3D", data=translated, **compression_kwargs)
                    wrote_points = True
            elif key == "node_names":
                dst.create_dataset(
                    "node_names",
                    data=np.asarray(node_names, dtype=object),
                    dtype=str_dtype,
                )
            else:
                src.copy(key, dst)

        if "node_names" not in dst:
            dst.create_dataset(
                "node_names",
                data=np.asarray(node_names, dtype=object),
                dtype=str_dtype,
            )
        if "points3D" not in dst:
            dst.create_dataset("points3D", data=translated, **compression_kwargs)

        for key, value in src.attrs.items():
            dst.attrs[key] = value

        metadata = dict(loaded.get("metadata", {}))
        metadata.update(
            {
                "source": "sleap_gui_3d_translate",
                "source_points3d_path": str(source_path),
                "points3d_path": str(output_path),
                "translation_origin_node": origin_node,
                "translation_mode": mode,
                "translation_origin_xyz": origin_for_metadata,
                "coordinate_system": f"{metadata.get('coordinate_system', 'calibration')}_translated",
            }
        )
        dst.attrs["metadata_json"] = json.dumps(metadata)
        dst.attrs["translation_origin_node"] = origin_node
        dst.attrs["translation_mode"] = mode

    return {
        "points3d_path": str(output_path),
        "source_points3d_path": str(source_path),
        "origin_node": origin_node,
        "mode": mode,
        "n_frames": int(points.shape[0]),
        "n_nodes": int(points.shape[1]),
    }


def load_reprojections_h5(path: str | Path) -> Dict[str, Any]:
    """Load a reprojections H5 file produced by :func:`run_3d_projection_export`."""
    try:
        import h5py
    except ImportError as exc:
        raise ImportError("h5py is required to load reprojections files.") from exc

    path = Path(path).expanduser()
    with h5py.File(path, "r") as f:
        if "reprojections" not in f:
            raise ValueError("H5 file does not contain reprojections.")
        reprojections = np.asarray(f["reprojections"], dtype=np.float64)
        point_scores = (
            np.asarray(f["point_scores"], dtype=np.float64)
            if "point_scores" in f
            else None
        )
        if "node_names" in f:
            node_names = _decode_strings(np.asarray(f["node_names"]))
        else:
            node_names = [f"node_{i}" for i in range(reprojections.shape[2])]
        if "camera_names" in f:
            camera_names = _decode_strings(np.asarray(f["camera_names"]))
        else:
            camera_names = [f"camera_{i}" for i in range(reprojections.shape[0])]

        metadata = {}
        if "metadata_json" in f.attrs:
            try:
                metadata = json.loads(f.attrs["metadata_json"])
            except Exception:
                metadata = {}
        if "source_files" in f and "prediction_files" not in metadata:
            metadata["prediction_files"] = _decode_strings(np.asarray(f["source_files"]))

    reprojections = _normalize_reprojections_shape(
        reprojections,
        n_cameras=len(camera_names),
        n_nodes=len(node_names),
    )
    if point_scores is not None:
        point_scores = _normalize_point_scores_shape(
            point_scores,
            n_cameras=len(camera_names),
            n_nodes=len(node_names),
        )
    return {
        "path": str(path),
        "reprojections": reprojections,
        "point_scores": point_scores,
        "node_names": list(node_names),
        "camera_names": list(camera_names),
        "metadata": metadata,
    }


def extract_points3d_position(
    points3d: np.ndarray,
    node_names: Sequence[str],
    selected_nodes: Sequence[str],
) -> np.ndarray:
    """Average selected node positions from a points3D array."""
    if not selected_nodes:
        return np.full((0, 3), np.nan, dtype=np.float64)
    index = {name: i for i, name in enumerate(node_names)}
    idxs = [index[name] for name in selected_nodes if name in index]
    if not idxs:
        return np.full((points3d.shape[0], 3), np.nan, dtype=np.float64)
    return np.nanmean(points3d[:, idxs, :3], axis=1)


def extract_reprojection_position(
    reprojections: np.ndarray,
    camera_index: int,
    node_names: Sequence[str],
    selected_nodes: Sequence[str],
) -> np.ndarray:
    """Average selected node positions from a camera reprojection array."""
    if not selected_nodes or camera_index < 0 or camera_index >= reprojections.shape[0]:
        return np.full((0, 2), np.nan, dtype=np.float64)
    index = {name: i for i, name in enumerate(node_names)}
    idxs = [index[name] for name in selected_nodes if name in index]
    if not idxs:
        return np.full((reprojections.shape[1], 2), np.nan, dtype=np.float64)
    camera_points = np.take(reprojections[camera_index], idxs, axis=1)
    return np.nanmean(camera_points[:, :, :2], axis=1)


def extract_reprojection_confidence(
    point_scores: Optional[np.ndarray],
    node_names: Sequence[str],
    selected_nodes: Sequence[str],
) -> Optional[np.ndarray]:
    """Return per-frame median confidence for selected nodes across cameras."""
    if point_scores is None or not selected_nodes:
        return None
    index = {name: i for i, name in enumerate(node_names)}
    idxs = [index[name] for name in selected_nodes if name in index]
    if not idxs:
        return np.full((point_scores.shape[1],), np.nan, dtype=np.float64)
    selected = np.take(point_scores, idxs, axis=2)
    return np.nanmedian(selected, axis=(0, 2))


def _extract_prediction_points(labels, video, node_names: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
    lfs = labels.find(video)
    if not lfs:
        return (
            np.full((0, len(node_names), 2), np.nan, dtype=np.float64),
            np.full((0, len(node_names)), np.nan, dtype=np.float64),
        )

    max_frame = max(lf.frame_idx for lf in lfs)
    points = np.full((max_frame + 1, len(node_names), 2), np.nan, dtype=np.float64)
    scores = np.full((max_frame + 1, len(node_names)), np.nan, dtype=np.float64)

    for lf in lfs:
        inst = _best_instance(lf)
        if inst is None:
            continue
        pts = np.asarray(inst.numpy(), dtype=np.float64)
        n_nodes = min(len(node_names), pts.shape[0])
        n_dims = min(2, pts.shape[1]) if pts.ndim == 2 else 0
        if n_dims == 0:
            continue
        points[lf.frame_idx, :n_nodes, :n_dims] = pts[:n_nodes, :n_dims]
        point_scores = getattr(inst, "point_scores", None)
        if point_scores is not None:
            scores[lf.frame_idx, : min(n_nodes, len(point_scores))] = point_scores[:n_nodes]
        else:
            scores[lf.frame_idx, :n_nodes] = np.isfinite(points[lf.frame_idx, :n_nodes, 0])

    return points, scores


def _best_instance(labeled_frame):
    instances = list(getattr(labeled_frame, "predicted_instances", []) or [])
    if not instances:
        instances = list(getattr(labeled_frame, "instances", []) or [])
    if not instances:
        return None
    return max(instances, key=lambda inst: float(getattr(inst, "score", 0.0) or 0.0))


def _shared_node_names(node_name_lists: Sequence[Sequence[str]]) -> List[str]:
    if not node_name_lists:
        return []
    shared = set(node_name_lists[0])
    for names in node_name_lists[1:]:
        shared &= set(names)
    return [name for name in node_name_lists[0] if name in shared]


def _subset_calibration(calibration, n_views: int, camera_names: Sequence[str]):
    calibration_names = list(calibration.get_names())
    if camera_names and all(name in calibration_names for name in camera_names):
        return calibration.subset_cameras_names(list(camera_names)), list(camera_names)

    matched_names = _match_camera_names(camera_names, calibration_names)
    if matched_names and len(matched_names) == n_views:
        return calibration.subset_cameras_names(matched_names), matched_names

    if len(calibration.cameras) < n_views:
        raise ValueError(
            f"Calibration has fewer cameras than predictions ({len(calibration.cameras)} < {n_views})."
        )
    used_names = calibration_names[:n_views] if calibration_names else [str(i) for i in range(n_views)]
    return calibration.subset_cameras(range(n_views)), used_names


def _match_camera_names(
    requested_names: Sequence[str],
    calibration_names: Sequence[str],
) -> List[str]:
    matched = []
    used = set()
    normalized_calibration = [
        (_normalize_camera_name(name), name) for name in calibration_names
    ]
    token_calibration = [
        (_camera_token(name), name) for name in calibration_names
    ]
    for requested in requested_names or []:
        norm_requested = _normalize_camera_name(requested)
        if not norm_requested:
            return []
        requested_token = _camera_token(requested)
        if requested_token:
            token_candidates = [
                name
                for token, name in token_calibration
                if name not in used and token == requested_token
            ]
            if len(token_candidates) == 1:
                matched.append(token_candidates[0])
                used.add(token_candidates[0])
                continue
            if len(token_candidates) > 1:
                return []
        candidates = [
            name
            for norm_cal, name in normalized_calibration
            if name not in used
            and norm_cal
            and (norm_cal in norm_requested or norm_requested in norm_cal)
        ]
        if len(candidates) != 1:
            return []
        matched.append(candidates[0])
        used.add(candidates[0])
    return matched


def _camera_index_map(
    input_camera_names: Sequence[str],
    reprojection_camera_names: Sequence[str],
) -> List[Optional[int]]:
    reproj_index = {name: idx for idx, name in enumerate(reprojection_camera_names)}
    normalized_reproj = [
        (_normalize_camera_name(name), idx)
        for idx, name in enumerate(reprojection_camera_names)
    ]
    token_reproj = [
        (_camera_token(name), idx)
        for idx, name in enumerate(reprojection_camera_names)
    ]
    mapping: List[Optional[int]] = []
    for name in input_camera_names:
        if name in reproj_index:
            mapping.append(reproj_index[name])
            continue
        token_name = _camera_token(name)
        if token_name:
            token_matches = [
                idx
                for token, idx in token_reproj
                if token == token_name
            ]
            if len(token_matches) == 1:
                mapping.append(token_matches[0])
                continue
            if len(token_matches) > 1:
                mapping.append(None)
                continue
        norm_name = _normalize_camera_name(name)
        matches = [
            idx
            for norm_reproj, idx in normalized_reproj
            if norm_reproj and norm_name and (norm_reproj in norm_name or norm_name in norm_reproj)
        ]
        mapping.append(matches[0] if len(matches) == 1 else None)
    return mapping


def _scores_for_reprojection_cameras(
    scores: np.ndarray,
    n_reprojection_views: int,
    input_to_reprojection: Sequence[Optional[int]],
) -> np.ndarray:
    out = np.full(
        (n_reprojection_views, scores.shape[1], scores.shape[2]),
        np.nan,
        dtype=np.float64,
    )
    for input_idx, reproj_idx in enumerate(input_to_reprojection):
        if reproj_idx is not None:
            out[reproj_idx] = scores[input_idx]
    return out


def _count_valid_views(points: np.ndarray) -> int:
    valid_by_view = np.any(np.all(np.isfinite(points), axis=2), axis=1)
    return int(np.sum(valid_by_view))


def _triangulate_vectorized(
    triangulation_calibration,
    reprojection_calibration,
    points2d: np.ndarray,
    points3d: np.ndarray,
    reprojections: np.ndarray,
    reprojection_error: np.ndarray,
    input_to_reprojection: Sequence[Optional[int]],
) -> bool:
    """Triangulate all valid frame-node samples in one calibration call.

    ``aniposelib`` accepts points with shape ``(n_cameras, n_points, 2)``.
    Flattening frame/node into the point axis avoids one Python call per video
    frame, which is the dominant cost for long sessions. If a calibration
    implementation does not support the vectorized shape, return ``False`` so
    callers can fall back to the slower per-frame loop.
    """
    n_views, n_frames, n_nodes, _ = points2d.shape
    valid = np.all(np.isfinite(points2d), axis=3)
    valid_flat = valid.reshape(n_views, -1)
    triangulatable = np.sum(valid_flat, axis=0) >= 2
    if not np.any(triangulatable):
        return True

    flat_points = points2d.reshape(n_views, -1, 2)
    try:
        flat_points3d = triangulation_calibration.triangulate(
            flat_points[:, triangulatable, :]
        )
        flat_reproj = reprojection_calibration.project(flat_points3d)
        if flat_points3d.shape != (int(np.sum(triangulatable)), 3):
            return False
        if flat_reproj.shape != (
            reprojections.shape[0],
            int(np.sum(triangulatable)),
            2,
        ):
            return False
    except Exception:
        return False

    flat_points3d_all = points3d.reshape(-1, 3)
    flat_points3d_all[triangulatable] = flat_points3d

    flat_reproj_all = reprojections.reshape(reprojections.shape[0], -1, 2)
    flat_reproj_all[:, triangulatable, :] = flat_reproj

    flat_error_all = reprojection_error.reshape(reprojections.shape[0], -1)
    for input_idx, reproj_idx in enumerate(input_to_reprojection):
        if reproj_idx is None:
            continue
        flat_error_all[reproj_idx, triangulatable] = np.linalg.norm(
            flat_reproj[reproj_idx] - flat_points[input_idx, triangulatable, :],
            axis=1,
        )

    return True


def _video_stem(video) -> str:
    filename = getattr(video, "filename", "")
    if isinstance(filename, list):
        filename = filename[0] if filename else ""
    return Path(str(filename)).stem


def _normalize_camera_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").casefold())


def _camera_token(value: str) -> str:
    match = re.search(r"cam\s*0*(\d+)", str(value or ""), flags=re.IGNORECASE)
    if not match:
        return ""
    return f"cam{int(match.group(1)):03d}"


def _decode_strings(values: np.ndarray) -> List[str]:
    out = []
    for value in values:
        if isinstance(value, bytes):
            out.append(value.decode("utf-8"))
        else:
            out.append(str(value))
    return out


def _normalize_points3d_shape(points: np.ndarray, n_nodes: int) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim == 4:
        if points.shape[0] == 3 and points.shape[1] == n_nodes:
            return np.transpose(points[:, :, 0, :], (2, 1, 0))
        if points.shape[1] == n_nodes and points.shape[2] == 3:
            return points[:, :, :3, 0]
        if points.shape[1] == n_nodes and points.shape[-1] == 3:
            return points[:, :, 0, :3]
        if points.shape[1] == 3 and points.shape[2] == n_nodes:
            return np.moveaxis(points[:, :, :, 0], 1, -1)
        raise ValueError(f"Could not interpret points3D shape {points.shape}.")
    if points.ndim != 3:
        raise ValueError(f"Expected points3D to be 3D, got shape {points.shape}.")
    if points.shape[-1] == 3:
        return points
    if points.shape[0] == 3:
        return np.moveaxis(points, 0, -1)
    if points.shape[1] == 3 and points.shape[2] == n_nodes:
        return np.moveaxis(points, 1, -1)
    raise ValueError(f"Could not interpret points3D shape {points.shape}.")


def _infer_points3d_node_count(points: np.ndarray) -> int:
    """Infer node count before normalizing points3D array orientation."""
    points = np.asarray(points)
    if points.ndim == 4:
        if points.shape[0] == 3:
            return int(points.shape[1])
        if points.shape[2] == 3 or points.shape[-1] == 3:
            return int(points.shape[1])
        if points.shape[1] == 3:
            return int(points.shape[2])
    if points.ndim == 3:
        if points.shape[-1] == 3:
            return int(points.shape[1])
        if points.shape[0] == 3:
            return int(points.shape[1])
        if points.shape[1] == 3:
            return int(points.shape[2])
    raise ValueError(f"Could not infer node count from points3D shape {points.shape}.")


def _normalize_reprojections_shape(
    reprojections: np.ndarray,
    *,
    n_cameras: int,
    n_nodes: int,
) -> np.ndarray:
    reprojections = np.asarray(reprojections, dtype=np.float64)
    if reprojections.ndim != 4:
        raise ValueError(
            f"Expected reprojections to be 4D, got shape {reprojections.shape}."
        )
    if reprojections.shape[0] == n_cameras and reprojections.shape[2] == n_nodes:
        return reprojections
    if reprojections.shape[2] == n_cameras and reprojections.shape[1] == n_nodes:
        return np.moveaxis(reprojections, 2, 0)
    raise ValueError(f"Could not interpret reprojections shape {reprojections.shape}.")


def _normalize_point_scores_shape(
    point_scores: np.ndarray,
    *,
    n_cameras: int,
    n_nodes: int,
) -> np.ndarray:
    point_scores = np.asarray(point_scores, dtype=np.float64)
    if point_scores.ndim != 3:
        raise ValueError(
            f"Expected point_scores to be 3D, got shape {point_scores.shape}."
        )
    if point_scores.shape[0] == n_cameras and point_scores.shape[2] == n_nodes:
        return point_scores
    if point_scores.shape[2] == n_cameras and point_scores.shape[1] == n_nodes:
        return np.moveaxis(point_scores, 2, 0)
    raise ValueError(f"Could not interpret point_scores shape {point_scores.shape}.")
