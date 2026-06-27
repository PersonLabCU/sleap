"""Reach-segment detection using the fixed-camera KPN outward-peaks method.

This module mirrors the sleap3d_reach ``kpn_outward`` detector, but uses
trajectories extracted from SLEAP prediction labels instead of
``trajectories.npz``.
"""

from __future__ import annotations

import json
import pickle
import csv
from enum import IntEnum
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

import numpy as np
from scipy.signal import find_peaks


_LEFT_KW = ("left", "l_", "_l", "lh", "lw", "lp", " l ", "foreleft")
_RIGHT_KW = ("right", "r_", "_r", "rh", "rw", "rp", "rhand", " r ", "foreright")
_HAND_KW = (
    "hand",
    "paw",
    "wrist",
    "finger",
    "thumb",
    "palm",
    "forepaw",
    "fore_paw",
    "fore_limb",
    "forelimb",
    "limb",
)
_REACH_HAND_NODE_STEMS = (
    "wrist_mid",
    "wrist_pinky",
    "mid_pinky",
    "tip_pinky",
    "tip_index",
    "mid_index",
    "wrist_index",
    "tip_ring",
    "tip_middle",
)
_PELLET_NODE_NAME = "pellet"

_TIME_TO_PLACE = 0.05
_MAX_DIST_FROM_HOME = 2.0
_TIME_TO_LOST = 0.1
_MIN_INTER_PELLET_INTV = 5.0
_MIN_DIST_FOR_GRAB = 15.0
_BATCH_FRM = 10
_POS_WINDOW_NUM_FRAMES = 100
_SPEED_WINDOW_NUM_FRAMES = 100
_PELLET_LOSS_GRAB_WINDOW_BEFORE = 15
_PELLET_LOSS_GRAB_WINDOW_AFTER = 5
DEFAULT_POINT_CONFIDENCE = 0.5


class ReachOutcome(IntEnum):
    """ReachX result codes used in canonical reach-segment files."""

    UNCLASSIFIED = 0
    GRABBED = 2
    MISSED = 3
    DROPPED = 4
    STALLED = 5

    @property
    def ui_color(self) -> str:
        return {
            ReachOutcome.UNCLASSIFIED: "#808080",
            ReachOutcome.GRABBED: "#27ae60",
            ReachOutcome.MISSED: "#b8860b",
            ReachOutcome.DROPPED: "#e67e22",
            ReachOutcome.STALLED: "#e74c3c",
        }[self]

    @property
    def label(self) -> str:
        return {
            ReachOutcome.UNCLASSIFIED: "Unclassified",
            ReachOutcome.GRABBED: "Grabbed",
            ReachOutcome.MISSED: "Missed",
            ReachOutcome.DROPPED: "Dropped",
            ReachOutcome.STALLED: "Stalled",
        }[self]

    @classmethod
    def selectable(cls) -> List["ReachOutcome"]:
        return [cls.UNCLASSIFIED, cls.GRABBED, cls.MISSED, cls.DROPPED, cls.STALLED]

    @classmethod
    def all_labels(cls) -> List[str]:
        return [outcome.label for outcome in cls.selectable()]


class ReachSegment(NamedTuple):
    """Canonical reach segment: frame, max_delta, dur, result, hand_pos."""

    frame: int
    max_delta: int
    dur: int
    outcome: ReachOutcome
    hand_pos: int = 0

    @property
    def max_frame(self) -> int:
        return self.frame + self.max_delta

    @property
    def end_frame(self) -> int:
        return self.frame + self.dur

    @property
    def is_valid(self) -> bool:
        return self.frame >= 0 and self.dur >= 3 and 0 < self.max_delta < self.dur


def suggest_hand_nodes(node_names: List[str]) -> Dict[str, List[str]]:
    """Return likely left/right hand node names from a skeleton."""
    node_set = set(node_names)
    reach_left = [
        f"{stem}_L" for stem in _REACH_HAND_NODE_STEMS if f"{stem}_L" in node_set
    ]
    reach_right = [
        f"{stem}_R" for stem in _REACH_HAND_NODE_STEMS if f"{stem}_R" in node_set
    ]
    if reach_left or reach_right:
        return {"left": reach_left, "right": reach_right}

    left: List[str] = []
    right: List[str] = []
    for name in node_names:
        ln = name.lower()
        is_hand = any(hk in ln for hk in _HAND_KW)
        is_left = any(lk in ln for lk in _LEFT_KW)
        is_right = any(rk in ln for rk in _RIGHT_KW)
        if is_left and is_hand:
            left.append(name)
        if is_right and is_hand:
            right.append(name)

    if not left:
        left = [n for n in node_names if any(lk in n.lower() for lk in _LEFT_KW)]
    if not right:
        right = [n for n in node_names if any(rk in n.lower() for rk in _RIGHT_KW)]
    return {"left": left, "right": right}


def suggest_pellet_nodes(node_names: List[str]) -> List[str]:
    """Return nodes named exactly ``pellet`` case-insensitively."""
    return [n for n in node_names if n.lower() == _PELLET_NODE_NAME]


def extract_trajectory(
    labels,
    video,
    node_names: List[str],
    axis: str = "x",
    *,
    use_predictions: bool = True,
    use_user: bool = True,
    invert: bool = False,
) -> np.ndarray:
    """Extract a single coordinate trace from labels."""
    axis_idx = {"x": 0, "y": 1, "z": 2}.get(axis.lower(), 0)
    traj = extract_hand_position_nd(
        labels,
        video,
        node_names,
        dims=max(axis_idx + 1, 2),
        use_predictions=use_predictions,
        use_user=use_user,
    )
    if traj.size == 0 or axis_idx >= traj.shape[1]:
        return np.array([], dtype=np.float64)
    out = traj[:, axis_idx]
    return -out if invert else out


def extract_hand_position_2d(
    labels,
    video,
    node_names: List[str],
    *,
    use_predictions: bool = True,
    use_user: bool = True,
) -> np.ndarray:
    """Extract per-frame ``(x, y)`` positions from selected nodes."""
    return extract_hand_position_nd(
        labels,
        video,
        node_names,
        dims=2,
        use_predictions=use_predictions,
        use_user=use_user,
    )


def extract_hand_position_3d(
    labels,
    video,
    node_names: List[str],
    *,
    use_predictions: bool = True,
    use_user: bool = True,
) -> np.ndarray:
    """Extract per-frame ``(x, y, z)`` positions from selected nodes."""
    return extract_hand_position_nd(
        labels,
        video,
        node_names,
        dims=3,
        use_predictions=use_predictions,
        use_user=use_user,
    )


def extract_hand_position_3d_with_confidence(
    labels,
    video,
    node_names: List[str],
    *,
    min_confidence: float = DEFAULT_POINT_CONFIDENCE,
    use_predictions: bool = True,
    use_user: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract score-aware per-frame ``(x, y, z)`` positions and confidences."""
    return extract_hand_position_nd_with_confidence(
        labels,
        video,
        node_names,
        dims=3,
        min_confidence=min_confidence,
        use_predictions=use_predictions,
        use_user=use_user,
    )


def extract_hand_position_nd(
    labels,
    video,
    node_names: List[str],
    *,
    dims: int,
    use_predictions: bool = True,
    use_user: bool = True,
) -> np.ndarray:
    """Extract per-frame node positions, averaged across nodes and instances."""
    if not node_names:
        return np.full((0, dims), np.nan, dtype=np.float64)

    lfs = labels.find(video)
    if not lfs:
        return np.full((0, dims), np.nan, dtype=np.float64)

    node_set = set(node_names)
    max_frame = max(lf.frame_idx for lf in lfs)
    traj = np.full((max_frame + 1, dims), np.nan, dtype=np.float64)

    for lf in lfs:
        instances = []
        if use_predictions:
            instances.extend(lf.predicted_instances)
        if use_user:
            instances.extend(lf.user_instances)

        vals: List[np.ndarray] = []
        for inst in instances:
            pts = np.asarray(inst.numpy(), dtype=np.float64)
            if pts.ndim != 2:
                continue
            for i, node in enumerate(inst.skeleton.nodes):
                if node.name not in node_set or i >= pts.shape[0]:
                    continue
                row = np.full(dims, np.nan, dtype=np.float64)
                n_cols = min(dims, pts.shape[1])
                row[:n_cols] = pts[i, :n_cols]
                if dims >= 3 and n_cols == 2:
                    row[2] = 0.0
                if np.all(np.isfinite(row[: min(dims, n_cols)])):
                    vals.append(row)

        if vals:
            traj[lf.frame_idx] = np.nanmean(np.vstack(vals), axis=0)

    return traj


def extract_hand_position_nd_with_confidence(
    labels,
    video,
    node_names: List[str],
    *,
    dims: int,
    min_confidence: float = DEFAULT_POINT_CONFIDENCE,
    use_predictions: bool = True,
    use_user: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract selected-node positions with confidence-aware aggregation.

    Coordinates are a confidence-weighted mean of selected points whose point
    scores meet ``min_confidence``. The returned confidence is the median score
    across finite selected points for each frame.
    """
    if not node_names:
        return (
            np.full((0, dims), np.nan, dtype=np.float64),
            np.full((0,), np.nan, dtype=np.float64),
        )

    lfs = labels.find(video)
    if not lfs:
        return (
            np.full((0, dims), np.nan, dtype=np.float64),
            np.full((0,), np.nan, dtype=np.float64),
        )

    node_set = set(node_names)
    max_frame = max(lf.frame_idx for lf in lfs)
    traj = np.full((max_frame + 1, dims), np.nan, dtype=np.float64)
    conf = np.full((max_frame + 1,), np.nan, dtype=np.float64)

    for lf in lfs:
        instances = []
        if use_predictions:
            instances.extend(lf.predicted_instances)
        if use_user:
            instances.extend(lf.user_instances)

        vals: List[np.ndarray] = []
        scores: List[float] = []
        for inst in instances:
            pts = np.asarray(inst.numpy(), dtype=np.float64)
            if pts.ndim != 2:
                continue
            point_scores = getattr(inst, "point_scores", None)
            for i, node in enumerate(inst.skeleton.nodes):
                if node.name not in node_set or i >= pts.shape[0]:
                    continue
                row = np.full(dims, np.nan, dtype=np.float64)
                n_cols = min(dims, pts.shape[1])
                row[:n_cols] = pts[i, :n_cols]
                if dims >= 3 and n_cols == 2:
                    row[2] = 0.0
                if not np.all(np.isfinite(row[: min(dims, n_cols)])):
                    continue
                score = _point_score_for_node(point_scores, i)
                vals.append(row)
                scores.append(score)

        if vals:
            arr = np.vstack(vals)
            score_arr = np.asarray(scores, dtype=np.float64)
            finite_scores = score_arr[np.isfinite(score_arr)]
            if finite_scores.size:
                conf[lf.frame_idx] = float(np.nanmedian(finite_scores))
            ok = np.isfinite(score_arr) & (score_arr >= float(min_confidence))
            if np.any(ok):
                weights = score_arr[ok]
                if not np.isfinite(np.sum(weights)) or np.sum(weights) <= 0:
                    traj[lf.frame_idx] = np.nanmean(arr[ok], axis=0)
                else:
                    traj[lf.frame_idx] = np.average(arr[ok], axis=0, weights=weights)

    return traj, conf


def _point_score_for_node(point_scores: Any, node_idx: int) -> float:
    if point_scores is None:
        return 1.0
    try:
        if node_idx < len(point_scores):
            score = float(point_scores[node_idx])
            return score if np.isfinite(score) else np.nan
    except (TypeError, ValueError):
        pass
    return np.nan


def detect_reaches_kpn(
    signal: Optional[np.ndarray] = None,
    *,
    right_hand: Optional[np.ndarray] = None,
    left_hand: Optional[np.ndarray] = None,
    pellet: Optional[np.ndarray] = None,
    right_hand_confidence: Optional[np.ndarray] = None,
    left_hand_confidence: Optional[np.ndarray] = None,
    pellet_confidence: Optional[np.ndarray] = None,
    star: Optional[np.ndarray] = None,
    triangle: Optional[np.ndarray] = None,
    tongue_mid: Optional[np.ndarray] = None,
    events: Optional[List[Dict[str, Any]]] = None,
    frame_rate: float = 30.0,
    min_threshold: float = -10.0,
    max_threshold: float = -7.0,
    peak_prominence: float = 1.0,
    min_outward_travel: float = 4.0,
    start_padding: int = 5,
    min_frame: int = 15,
    max_frame: int = 200,
    confidence: float = DEFAULT_POINT_CONFIDENCE,
    pellet_drop_speed: float = 0.275,
    pellet_drop_dist_z: float = -5.0,
    pellet_dist_to_origin: float = 2.0,
    max_dist_from_home: float = _MAX_DIST_FROM_HOME,
    return_details: bool = False,
) -> List[ReachSegment] | Tuple[List[ReachSegment], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Detect reaches with the fixed-camera KPN outward-peaks method.

    Preferred use passes ``right_hand``, ``left_hand``, and ``pellet`` arrays
    extracted from a SLEAP predictions file. Arrays may be ``(N, 2)`` or
    ``(N, 3)``; 2D arrays get ``z=0``. The legacy scalar ``signal`` path remains
    as a compatibility fallback for callers that have not yet been updated.
    """
    if right_hand is None or left_hand is None or pellet is None:
        reaches = _detect_kpn_periods_only(
            signal,
            min_threshold=min_threshold,
            max_threshold=max_threshold,
            peak_prominence=peak_prominence,
            min_outward_travel=min_outward_travel,
            start_padding=start_padding,
            min_frame=min_frame,
            max_frame=max_frame,
        )
        if return_details:
            return reaches, [], []
        return reaches

    r_hand = _as_xyz(right_hand)
    l_hand = _as_xyz(left_hand)
    pellet_xyz = _as_xyz(pellet)
    n = min(len(r_hand), len(l_hand), len(pellet_xyz))
    if n == 0:
        if return_details:
            return [], [], []
        return []
    r_hand = r_hand[:n]
    l_hand = l_hand[:n]
    pellet_xyz = pellet_xyz[:n]

    pellet_p = _confidence_trace(pellet_confidence, pellet_xyz, n)
    r_hand_p = _confidence_trace(right_hand_confidence, r_hand, n)
    l_hand_p = _confidence_trace(left_hand_confidence, l_hand, n)

    home_source = pellet_xyz[pellet_p >= confidence]
    if home_source.size == 0:
        if return_details:
            return [], [], []
        return []
    pellet_home = np.nanmedian(home_source, axis=0)

    distance_p = _distance_from_home(pellet_xyz, pellet_home)
    distance_rh = _distance_from_home(r_hand, pellet_home)
    distance_lh = _distance_from_home(l_hand, pellet_home)
    distance_p[pellet_p < confidence] = np.nan
    distance_rh[r_hand_p < confidence] = np.nan
    distance_lh[l_hand_p < confidence] = np.nan

    pellet_for_speed = pellet_xyz.copy()
    pellet_for_speed[pellet_p < confidence] = np.nan
    pellet_speed = _filtered_speed(pellet_for_speed, frame_rate)
    coord_pz = pellet_xyz[:, 2] - pellet_home[2]
    coord_pz[pellet_p < confidence] = np.nan
    if star is not None and triangle is not None:
        star_xyz = _fit_length(_as_xyz(star), n)
        triangle_xyz = _fit_length(_as_xyz(triangle), n)
        dist_star_tri = np.sqrt(np.sum((star_xyz[:, :3] - triangle_xyz[:, :3]) ** 2, axis=1))
        dist_star_tri[~(_valid_trace(star_xyz) & _valid_trace(triangle_xyz))] = np.nan
    else:
        dist_star_tri = np.full(n, np.nan, dtype=np.float64)

    if tongue_mid is not None:
        tongue_xyz = _fit_length(_as_xyz(tongue_mid), n)
        dist_midtongue = _distance_from_home(tongue_xyz, pellet_home)
        mid_tongue_p = _valid_trace(tongue_xyz).astype(float)
        dist_midtongue[mid_tongue_p == 0] = np.nan
    else:
        dist_midtongue = np.full(n, 10000.0, dtype=np.float64)
        mid_tongue_p = np.zeros(n, dtype=np.float64)

    pellet_events = _find_pellet_epochs(
        distance_p=distance_p,
        dist_star_tri=dist_star_tri,
        pellet_p=pellet_p,
        distance_rh=distance_rh,
        distance_lh=distance_lh,
        dist_midtongue=dist_midtongue,
        r_hand_p=r_hand_p,
        l_hand_p=l_hand_p,
        mid_tongue_p=mid_tongue_p,
        coord_pz=coord_pz,
        frame_rate=frame_rate,
        confidence=confidence,
        pellet_drop_dist_z=pellet_drop_dist_z,
        max_dist_from_home=max_dist_from_home,
    )
    _annotate_pellet_availability(pellet_events, events or [])

    reaches, details = _find_kpn_outward_reaches_fixed_cam(
        pellet_events=pellet_events,
        distance_rh=distance_rh,
        distance_p=distance_p,
        pellet_p=pellet_p,
        r_hand_p=r_hand_p,
        l_hand_p=l_hand_p,
        pellet_speed=pellet_speed,
        coord_pz=coord_pz,
        min_threshold=min_threshold,
        max_threshold=max_threshold,
        peak_prominence=peak_prominence,
        min_outward_travel=min_outward_travel,
        start_padding=start_padding,
        min_frame=min_frame,
        max_frame=max_frame,
        frame_rate=frame_rate,
        confidence=confidence,
        pellet_drop_speed=pellet_drop_speed,
        pellet_drop_dist_z=pellet_drop_dist_z,
        pellet_dist_to_origin=pellet_dist_to_origin,
    )

    pellet_history = [dict(evt) for evt in pellet_events]
    pellet_history.append(
        {"x": float(pellet_home[0]), "y": float(pellet_home[1]), "z": float(pellet_home[2])}
    )

    if return_details:
        return reaches, details, pellet_history
    return reaches


def detect_reaches_absolute(
    signal: Optional[np.ndarray],
    *,
    right_hand: Optional[np.ndarray] = None,
    left_hand: Optional[np.ndarray] = None,
    pellet: Optional[np.ndarray] = None,
    right_hand_confidence: Optional[np.ndarray] = None,
    left_hand_confidence: Optional[np.ndarray] = None,
    pellet_confidence: Optional[np.ndarray] = None,
    star: Optional[np.ndarray] = None,
    triangle: Optional[np.ndarray] = None,
    tongue_mid: Optional[np.ndarray] = None,
    events: Optional[List[Dict[str, Any]]] = None,
    threshold: float,
    peak_prominence: float = 1.0,
    start_padding: int = 5,
    min_frame: int = 15,
    max_frame: int = 200,
    max_start_value: Optional[float] = 300.0,
    frame_rate: float = 30.0,
    signal_confidence: Optional[np.ndarray] = None,
    confidence: float = DEFAULT_POINT_CONFIDENCE,
    pellet_drop_speed: float = 0.275,
    pellet_drop_dist_z: float = -5.0,
    pellet_dist_to_origin: float = 2.0,
    max_dist_from_home: float = _MAX_DIST_FROM_HOME,
    return_details: bool = False,
    return_pellet_history: bool = False,
) -> List[ReachSegment] | Tuple[List[ReachSegment], List[Dict[str, Any]]] | Tuple[
    List[ReachSegment], List[Dict[str, Any]], List[Dict[str, Any]]
]:
    """Detect reach epochs from an absolute outward-position trace.

    This follows the MATLAB ``find_reaches_full_peaks_kpn`` logic: threshold
    upward/downward crossings seed candidate reaches, the start is estimated
    from pre-crossing velocity/acceleration intersections, the end is the first
    reversal after the threshold-down crossing, and the first peak within the
    full candidate reach is used as the reach maximum. Additional peak locations
    are preserved in the detail records as sub-reaches.
    """
    if signal is None:
        return ([], []) if return_details else []

    x = np.asarray(signal, dtype=np.float64).reshape(-1)
    if x.size == 0:
        return ([], []) if return_details else []

    if signal_confidence is not None:
        conf = np.asarray(signal_confidence, dtype=np.float64).reshape(-1)
        take = min(x.size, conf.size)
        low_conf = np.ones(x.size, dtype=bool)
        low_conf[:take] = ~(np.isfinite(conf[:take]) & (conf[:take] >= confidence))
        x = x.copy()
        x[low_conf] = np.nan
    else:
        conf = np.full(x.size, np.nan, dtype=np.float64)

    x = _interp_nan_1d(x)
    if x.size == 0:
        return ([], []) if return_details else []

    periods = _find_absolute_kpn_periods(
        x,
        threshold=float(threshold),
        peak_prominence=float(peak_prominence),
        padding=int(start_padding),
        max_start_value=max_start_value,
    )
    pellet_context = _build_kpn_pellet_context(
        right_hand=right_hand,
        left_hand=left_hand,
        pellet=pellet,
        right_hand_confidence=right_hand_confidence,
        left_hand_confidence=left_hand_confidence,
        pellet_confidence=pellet_confidence,
        star=star,
        triangle=triangle,
        tongue_mid=tongue_mid,
        events=events,
        frame_rate=frame_rate,
        confidence=confidence,
        pellet_drop_dist_z=pellet_drop_dist_z,
        max_dist_from_home=max_dist_from_home,
    )

    reaches: List[ReachSegment] = []
    details: List[Dict[str, Any]] = []
    for period in periods:
        start = int(period["start"])
        max_fr = int(period["first_peak"])
        end = int(period["end"])
        dur = end - start
        pellet_event = (
            _pellet_event_for_reach(
                pellet_context["pellet_events"],
                start,
                max_fr,
                len(x),
            )
            if pellet_context is not None
            else None
        )
        outcome = ReachOutcome.UNCLASSIFIED
        if pellet_context is not None and pellet_event is not None:
            outcome = _classify_kpn_reach_result(
                pellet_event,
                max_fr,
                end,
                pellet_context["distance_p"],
                pellet_context["pellet_p"],
                pellet_context["pellet_speed"],
                pellet_context["coord_pz"],
                start,
                confidence=confidence,
                pellet_drop_speed=pellet_drop_speed,
                pellet_drop_dist_z=pellet_drop_dist_z,
                pellet_dist_to_origin=pellet_dist_to_origin,
            )
        reach = ReachSegment(start, max_fr - start, dur, outcome)
        if dur < min_frame or dur > max_frame or not reach.is_valid:
            continue
        if _overlaps(reach, reaches):
            continue
        reaches.append(reach)

        peak_frames = [int(p) for p in period["peaks"]]
        mini_peak_frames = [p for p in peak_frames if p > max_fr]
        details.append(
            {
                "frame": int(start),
                "max_frame": int(max_fr),
                "end_frame": int(end),
                "dur": int(dur),
                "result": outcome.name,
                "detection_method": "absolute",
                "is_multi_reach": bool(mini_peak_frames),
                "peak_frames": peak_frames,
                "mini_peak_frames": mini_peak_frames,
                "first_peak_frame": int(max_fr),
                "first_peak_time_s": _frame_time(max_fr, frame_rate),
                "peak_times_s": [_frame_time(p, frame_rate) for p in peak_frames],
                "mini_peak_times_s": [
                    _frame_time(p, frame_rate) for p in mini_peak_frames
                ],
                "peak_values": [float(v) for v in period["peak_values"]],
                "threshold": float(threshold),
                "min_threshold": float(threshold),
                "max_threshold": float("nan"),
                "min_cross_frame": int(period["min_cross"]),
                "max_cross_frame": int(period["min_cross"]),
                "threshold_down_frame": int(period["threshold_down"]),
                "outward_travel": float(period["outward_travel"]),
                "start_value": float(period["start_value"]),
                "max_start_value": (
                    float(max_start_value)
                    if max_start_value is not None
                    else float("nan")
                ),
                "point_confidence_threshold": float(confidence),
                "rh_conf_mean": _confidence_mean(conf, start, end),
                "rh_conf_median": _confidence_median(conf, start, end),
                "rh_conf_min": _confidence_min(conf, start, end),
                "rh_conf_fraction_valid": _confidence_fraction(
                    conf, start, end, confidence
                ),
            }
        )
        if pellet_context is not None:
            pellet_lost_frame = int(pellet_event.get("lost", -1)) if pellet_event else -1
            detail = details[-1]
            detail.update(
                {
                    "pellet_available": bool(
                        pellet_event is not None
                        and _is_pellet_available_for_reach(pellet_event, max_fr)
                    ),
                    "pellet_available_frame": int(
                        pellet_event.get("available", -1) if pellet_event else -1
                    ),
                    "pellet_placed_frame": int(
                        pellet_event.get("placed", -1) if pellet_event else -1
                    ),
                    "pellet_lost_frame": pellet_lost_frame,
                    "lh_conf_mean": _confidence_mean(
                        pellet_context["l_hand_p"], start, end
                    ),
                    "lh_conf_median": _confidence_median(
                        pellet_context["l_hand_p"], start, end
                    ),
                    "lh_conf_min": _confidence_min(
                        pellet_context["l_hand_p"], start, end
                    ),
                    "lh_conf_fraction_valid": _confidence_fraction(
                        pellet_context["l_hand_p"], start, end, confidence
                    ),
                    "pellet_conf_mean": _confidence_mean(
                        pellet_context["pellet_p"], start, end
                    ),
                    "pellet_conf_median": _confidence_median(
                        pellet_context["pellet_p"], start, end
                    ),
                    "pellet_conf_min": _confidence_min(
                        pellet_context["pellet_p"], start, end
                    ),
                    "pellet_conf_fraction_valid": _confidence_fraction(
                        pellet_context["pellet_p"], start, end, confidence
                    ),
                    "pellet_conf_at_loss": _confidence_at(
                        pellet_context["pellet_p"], pellet_lost_frame
                    ),
                    "pellet_method": str(
                        pellet_event.get("method", "") if pellet_event else ""
                    ),
                    "pellet_outcome": str(
                        pellet_event.get("outcome", "") if pellet_event else ""
                    ),
                    "pellet_method_frame": int(
                        pellet_event.get("method_frame", -1) if pellet_event else -1
                    ),
                    "pellet_lost_before_available": bool(
                        pellet_event.get("lost_before_available", False)
                        if pellet_event
                        else False
                    ),
                }
            )

    reaches.sort(key=lambda seg: seg.frame)
    details.sort(key=lambda detail: int(detail["frame"]))
    if return_details:
        if return_pellet_history:
            pellet_history = (
                pellet_context["pellet_history"] if pellet_context is not None else []
            )
            return reaches, details, pellet_history
        return reaches, details
    return reaches


def _build_kpn_pellet_context(
    *,
    right_hand: Optional[np.ndarray],
    left_hand: Optional[np.ndarray],
    pellet: Optional[np.ndarray],
    right_hand_confidence: Optional[np.ndarray],
    left_hand_confidence: Optional[np.ndarray],
    pellet_confidence: Optional[np.ndarray],
    star: Optional[np.ndarray],
    triangle: Optional[np.ndarray],
    tongue_mid: Optional[np.ndarray],
    events: Optional[List[Dict[str, Any]]],
    frame_rate: float,
    confidence: float,
    pellet_drop_dist_z: float,
    max_dist_from_home: float,
) -> Optional[Dict[str, Any]]:
    if right_hand is None or left_hand is None or pellet is None:
        return None

    r_hand = _as_xyz(right_hand)
    l_hand = _as_xyz(left_hand)
    pellet_xyz = _as_xyz(pellet)
    n = min(len(r_hand), len(l_hand), len(pellet_xyz))
    if n == 0:
        return None
    r_hand = r_hand[:n]
    l_hand = l_hand[:n]
    pellet_xyz = pellet_xyz[:n]

    pellet_p = _confidence_trace(pellet_confidence, pellet_xyz, n)
    r_hand_p = _confidence_trace(right_hand_confidence, r_hand, n)
    l_hand_p = _confidence_trace(left_hand_confidence, l_hand, n)

    home_source = pellet_xyz[pellet_p >= confidence]
    if home_source.size == 0:
        return None
    pellet_home = np.nanmedian(home_source, axis=0)

    distance_p = _distance_from_home(pellet_xyz, pellet_home)
    distance_rh = _distance_from_home(r_hand, pellet_home)
    distance_lh = _distance_from_home(l_hand, pellet_home)
    distance_p[pellet_p < confidence] = np.nan
    distance_rh[r_hand_p < confidence] = np.nan
    distance_lh[l_hand_p < confidence] = np.nan

    pellet_for_speed = pellet_xyz.copy()
    pellet_for_speed[pellet_p < confidence] = np.nan
    pellet_speed = _filtered_speed(pellet_for_speed, frame_rate)
    coord_pz = pellet_xyz[:, 2] - pellet_home[2]
    coord_pz[pellet_p < confidence] = np.nan

    if star is not None and triangle is not None:
        star_xyz = _fit_length(_as_xyz(star), n)
        triangle_xyz = _fit_length(_as_xyz(triangle), n)
        dist_star_tri = np.sqrt(
            np.sum((star_xyz[:, :3] - triangle_xyz[:, :3]) ** 2, axis=1)
        )
        dist_star_tri[~(_valid_trace(star_xyz) & _valid_trace(triangle_xyz))] = np.nan
    else:
        dist_star_tri = np.full(n, np.nan, dtype=np.float64)

    if tongue_mid is not None:
        tongue_xyz = _fit_length(_as_xyz(tongue_mid), n)
        dist_midtongue = _distance_from_home(tongue_xyz, pellet_home)
        mid_tongue_p = _valid_trace(tongue_xyz).astype(float)
        dist_midtongue[mid_tongue_p == 0] = np.nan
    else:
        dist_midtongue = np.full(n, 10000.0, dtype=np.float64)
        mid_tongue_p = np.zeros(n, dtype=np.float64)

    pellet_events = _find_pellet_epochs(
        distance_p=distance_p,
        dist_star_tri=dist_star_tri,
        pellet_p=pellet_p,
        distance_rh=distance_rh,
        distance_lh=distance_lh,
        dist_midtongue=dist_midtongue,
        r_hand_p=r_hand_p,
        l_hand_p=l_hand_p,
        mid_tongue_p=mid_tongue_p,
        coord_pz=coord_pz,
        frame_rate=frame_rate,
        confidence=confidence,
        pellet_drop_dist_z=pellet_drop_dist_z,
        max_dist_from_home=max_dist_from_home,
    )
    _annotate_pellet_availability(pellet_events, events or [])

    pellet_history = [dict(evt) for evt in pellet_events]
    pellet_history.append(
        {
            "x": float(pellet_home[0]),
            "y": float(pellet_home[1]),
            "z": float(pellet_home[2]),
        }
    )

    return {
        "pellet_events": pellet_events,
        "pellet_history": pellet_history,
        "distance_p": distance_p,
        "distance_rh": distance_rh,
        "distance_lh": distance_lh,
        "pellet_p": pellet_p,
        "r_hand_p": r_hand_p,
        "l_hand_p": l_hand_p,
        "pellet_speed": pellet_speed,
        "coord_pz": coord_pz,
    }


def _pellet_event_for_reach(
    pellet_events: List[Dict[str, Any]],
    start_frame: int,
    max_frame: int,
    n_frames: int,
) -> Optional[Dict[str, Any]]:
    for idx, pellet_event in enumerate(pellet_events):
        if "placed" not in pellet_event:
            continue
        epoch_start = max(0, int(pellet_event["placed"]) - 20)
        next_start = (
            int(pellet_events[idx + 1]["placed"])
            if idx < len(pellet_events) - 1
            else int(n_frames)
        )
        if epoch_start <= int(max_frame) < next_start:
            return pellet_event
        if epoch_start <= int(start_frame) < next_start:
            return pellet_event
    return None


def _detect_kpn_periods_only(
    signal: Optional[np.ndarray],
    *,
    min_threshold: float,
    max_threshold: float,
    peak_prominence: float,
    min_outward_travel: float,
    start_padding: int,
    min_frame: int,
    max_frame: int,
) -> List[ReachSegment]:
    if signal is None:
        return []
    x = _interp_nan_1d(np.asarray(signal, dtype=np.float64).reshape(-1))
    if x.size == 0:
        return []
    periods = _find_kpn_outward_periods(
        x,
        min_threshold,
        max_threshold,
        peak_prominence,
        min_outward_travel,
        start_padding,
    )
    reaches: List[ReachSegment] = []
    for period in periods:
        start = int(period["start"])
        max_fr = int(period["first_peak"])
        end = int(period["end"])
        dur = end - start
        reach = ReachSegment(start, max_fr - start, dur, ReachOutcome.UNCLASSIFIED)
        if min_frame <= dur <= max_frame and reach.is_valid and not _overlaps(reach, reaches):
            reaches.append(reach)
    return reaches


def _find_absolute_kpn_periods(
    data: np.ndarray,
    *,
    threshold: float,
    peak_prominence: float,
    padding: int,
    max_start_value: Optional[float],
) -> List[Dict[str, Any]]:
    x = np.asarray(data, dtype=np.float64).reshape(-1)
    if x.size < 3:
        return []

    crosses = _absolute_threshold_crosses(x, threshold)
    periods: List[Dict[str, Any]] = []
    for up, down in crosses:
        start_idx = _kpn_start_index_from_pretrace(x[: up + 1])
        start_with_padding = max(0, int(start_idx) - int(padding))
        end_idx = _absolute_end_index_after_down_crossing(x, int(down))
        if end_idx <= start_with_padding:
            continue

        reach = x[start_with_padding : end_idx + 1]
        if reach.size == 0:
            continue
        start_value = float(reach[0])
        if (
            max_start_value is not None
            and np.isfinite(start_value)
            and start_value > float(max_start_value)
        ):
            continue

        peaks, props = find_peaks(reach, prominence=peak_prominence)
        if peaks.size == 0:
            continue

        peak_indices = np.asarray(
            [int(start_with_padding + p) for p in peaks],
            dtype=np.int64,
        )
        peak_idx = int(peak_indices[0])
        outward_travel = float(x[peak_idx]) - float(x[start_with_padding])
        if peak_idx <= start_with_padding:
            continue

        periods.append(
            {
                "start": int(start_with_padding),
                "first_peak": int(peak_idx),
                "end": int(end_idx),
                "peaks": [int(p) for p in peak_indices],
                "peak_values": [float(x[p]) for p in peak_indices],
                "min_cross": int(up),
                "threshold_down": int(down),
                "outward_travel": float(outward_travel),
                "start_value": start_value,
                "prominences": [
                    float(v) for v in props.get("prominences", np.asarray([]))
                ],
            }
        )
    return periods


def _absolute_threshold_crosses(x: np.ndarray, threshold: float) -> List[Tuple[int, int]]:
    above_thresh = x >= float(threshold)
    thresh_diffs = np.concatenate(([0], np.diff(above_thresh.astype(np.int8))))
    cross_up = np.where(thresh_diffs == 1)[0]
    cross_down = np.where(thresh_diffs == -1)[0]
    if cross_up.size == 0 or cross_down.size == 0:
        return []

    if cross_up.size < cross_down.size:
        cross_down = cross_down[1:]
    elif cross_up[0] > cross_down[0]:
        cross_up = cross_up[:-1]
        cross_down = cross_down[1:]
    else:
        n = min(cross_up.size, cross_down.size)
        cross_up = cross_up[:n]
        cross_down = cross_down[:n]

    n = min(cross_up.size, cross_down.size)
    return [
        (int(up), int(down))
        for up, down in zip(cross_up[:n], cross_down[:n])
        if int(down) > int(up)
    ]


def _absolute_end_index_after_down_crossing(x: np.ndarray, down: int) -> int:
    post_x = x[int(down) :]
    diff_post_x = np.diff(post_x)
    rising = np.where(diff_post_x > 0)[0]
    if rising.size == 0:
        return int(x.size - 1)
    return int(down + rising[0])


def _as_xyz(points: np.ndarray) -> np.ndarray:
    arr = np.asarray(points, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    out = np.full((arr.shape[0], 3), np.nan, dtype=np.float64)
    cols = min(arr.shape[1], 3)
    out[:, :cols] = arr[:, :cols]
    if cols == 2 or (cols == 3 and np.isnan(out[:, 2]).all()):
        out[:, 2] = 0.0
    return out


def _fit_length(points: np.ndarray, n: int) -> np.ndarray:
    out = np.full((n, 3), np.nan, dtype=np.float64)
    take = min(n, len(points))
    if take > 0:
        out[:take] = points[:take]
    return out


def _confidence_trace(
    confidence: Optional[np.ndarray],
    points: np.ndarray,
    n: int,
) -> np.ndarray:
    valid = _valid_trace(points[:n])
    if confidence is None:
        return valid.astype(float)
    out = np.zeros(n, dtype=np.float64)
    arr = np.asarray(confidence, dtype=np.float64).reshape(-1)
    take = min(n, arr.size)
    if take > 0:
        out[:take] = arr[:take]
    out[~valid] = 0.0
    out[~np.isfinite(out)] = 0.0
    return out


def _min_valid_distance_in_window(
    distance: np.ndarray,
    present: np.ndarray,
    start: int,
    end: int,
    confidence: float,
) -> Tuple[float, int]:
    idx = np.arange(start, end)
    if idx.size == 0:
        return float("nan"), -1
    vals = np.asarray(distance[start:end], dtype=np.float64)
    ok = np.isfinite(vals) & (np.asarray(present[start:end]) >= confidence)
    if not np.any(ok):
        return float("nan"), -1
    local = int(np.nanargmin(np.where(ok, vals, np.nan)))
    return float(vals[local]), int(start + local)


def _valid_trace(points: np.ndarray) -> np.ndarray:
    return np.all(np.isfinite(points[:, :3]), axis=1)


def _distance_from_home(points: np.ndarray, home: np.ndarray) -> np.ndarray:
    return np.sqrt(np.sum((points[:, :3] - home[:3]) ** 2, axis=1))


def _filtered_speed(points: np.ndarray, frame_rate: float) -> np.ndarray:
    speed = np.full(points.shape[0], np.nan, dtype=np.float64)
    valid = _valid_trace(points)
    if points.shape[0] < 2:
        return np.nan_to_num(speed, nan=0.0)
    diffs = np.linalg.norm(np.diff(points[:, :3], axis=0), axis=1)
    speed[1:] = diffs * (float(frame_rate) / 1000.0)
    speed[~np.concatenate(([False], valid[1:] & valid[:-1]))] = np.nan
    return np.nan_to_num(speed, nan=0.0)


def _interp_nan_1d(x: np.ndarray) -> np.ndarray:
    if x.size == 0 or np.isnan(x).all():
        return np.array([], dtype=np.float64)
    out = x.copy()
    nans = np.isnan(out)
    if nans.any():
        ok = ~nans
        out[nans] = np.interp(np.where(nans)[0], np.where(ok)[0], out[ok])
    return out


def _find_pellet_epochs(
    *,
    distance_p: np.ndarray,
    dist_star_tri: np.ndarray,
    pellet_p: np.ndarray,
    distance_rh: np.ndarray,
    distance_lh: np.ndarray,
    dist_midtongue: np.ndarray,
    r_hand_p: np.ndarray,
    l_hand_p: np.ndarray,
    mid_tongue_p: np.ndarray,
    coord_pz: np.ndarray,
    frame_rate: float,
    confidence: float,
    pellet_drop_dist_z: float,
    max_dist_from_home: float = _MAX_DIST_FROM_HOME,
) -> List[Dict[str, Any]]:
    n_place = max(1, int(np.ceil(_TIME_TO_PLACE * frame_rate)))
    n_lost = max(1, int(np.ceil(_TIME_TO_LOST * frame_rate)))
    n_inter = max(1, int(np.ceil(_MIN_INTER_PELLET_INTV * frame_rate)))

    pellet_events: List[Dict[str, Any]] = []
    state = 0
    count = 0
    start_frame = 0

    for frame, (dist, st, p) in enumerate(zip(distance_p, dist_star_tri, pellet_p)):
        if state == 0:
            cover_open = (not np.isfinite(st)) or st > 12
            placed = (
                np.isfinite(dist)
                and dist <= max_dist_from_home
                and p >= confidence
                and cover_open
            )
            if not placed:
                count = 0
                start_frame = frame
                continue
            count += 1
            if count >= n_place:
                pellet_events.append(
                    {"placed": int(start_frame), "lost": -1, "method": "none", "outcome": "none"}
                )
                state = 1
                count = 0

        elif state == 1:
            lost = (not np.isfinite(dist)) or dist > max_dist_from_home or p < confidence
            if lost:
                count += 1
            else:
                count = 0
                start_frame = frame

            if count >= n_lost:
                event = pellet_events[-1]
                event["lost"] = int(start_frame)
                lost_frame = int(np.clip(start_frame, 0, len(distance_rh) - 1))

                win_start = max(0, lost_frame - _PELLET_LOSS_GRAB_WINDOW_BEFORE)
                win_end = min(len(distance_rh), lost_frame + _PELLET_LOSS_GRAB_WINDOW_AFTER + 1)
                rh_dist, rh_frame = _min_valid_distance_in_window(
                    distance_rh, r_hand_p, win_start, win_end, confidence
                )
                lh_dist, lh_frame = _min_valid_distance_in_window(
                    distance_lh, l_hand_p, win_start, win_end, confidence
                )
                tongue_dist, tongue_frame = _min_valid_distance_in_window(
                    dist_midtongue, mid_tongue_p, win_start, win_end, confidence
                )
                r_grabbed = np.isfinite(rh_dist) and rh_dist < _MIN_DIST_FOR_GRAB
                l_grabbed = np.isfinite(lh_dist) and lh_dist < _MIN_DIST_FOR_GRAB
                tongue_present = np.isfinite(tongue_dist)
                r_closer = np.isfinite(rh_dist) and (not np.isfinite(lh_dist) or rh_dist < lh_dist)
                t_closer_than_r = np.isfinite(tongue_dist) and (
                    not np.isfinite(rh_dist) or tongue_dist < rh_dist
                )
                t_closer_than_l = np.isfinite(tongue_dist) and (
                    not np.isfinite(lh_dist) or tongue_dist < lh_dist
                )

                event["outcome"] = "eaten"
                if tongue_present and t_closer_than_r and t_closer_than_l:
                    event["method"] = "tongue"
                    event["method_frame"] = int(tongue_frame)
                elif r_grabbed and r_closer:
                    event["method"] = "right_hand"
                    event["method_frame"] = int(rh_frame)
                elif l_grabbed and not r_closer:
                    event["method"] = "left_hand"
                    event["method_frame"] = int(lh_frame)
                else:
                    event["method"] = "other"
                    event["outcome"] = "dropped"
                state = 2

        else:
            count += 1
            if count >= n_inter:
                state = 0
                count = 0

    if pellet_events and int(pellet_events[-1].get("lost", -1)) < 0:
        pellet_events.pop()

    for event in pellet_events:
        lost = int(event.get("lost", -1))
        if lost < 0:
            continue
        idx_start = int(event["placed"])
        idx_end = min(len(coord_pz), lost + _POS_WINDOW_NUM_FRAMES)
        if idx_end > idx_start and np.nanmin(coord_pz[idx_start:idx_end]) <= pellet_drop_dist_z:
            event["outcome"] = "dropped"

    return pellet_events


def _annotate_pellet_availability(
    pellet_events: List[Dict[str, Any]], events: List[Dict[str, Any]]
) -> None:
    delivery_frames: List[int] = []
    for event in events or []:
        name = str(event.get("event", "")).lower()
        if not any(token in name for token in ("delivery", "available", "pellet_placed")):
            continue
        try:
            delivery_frames.append(int(event["frame"]))
        except (KeyError, TypeError, ValueError):
            continue

    if not delivery_frames:
        return
    deliveries = np.asarray(sorted(delivery_frames), dtype=np.int64)
    for idx, event in enumerate(pellet_events):
        placed = int(event.get("placed", -1))
        lost = int(event.get("lost", -1))
        if placed < 0:
            event["available"] = -1
            continue
        available = -1
        next_idx = int(np.searchsorted(deliveries, placed, side="left"))
        if next_idx < deliveries.size and (lost < 0 or deliveries[next_idx] <= lost):
            available = int(deliveries[next_idx])
        else:
            prev_idx = next_idx - 1
            prev_lost = int(pellet_events[idx - 1].get("lost", -1)) if idx > 0 else -1
            if prev_idx >= 0 and deliveries[prev_idx] >= max(0, prev_lost):
                available = int(deliveries[prev_idx])
            elif next_idx < deliveries.size:
                next_placed = (
                    int(pellet_events[idx + 1].get("placed", 10**12))
                    if idx < len(pellet_events) - 1
                    else 10**12
                )
                if deliveries[next_idx] < next_placed:
                    available = int(deliveries[next_idx])
                    if lost >= 0 and available > lost:
                        event["lost_before_available"] = True
        event["available"] = available


def _find_kpn_outward_reaches_fixed_cam(
    *,
    pellet_events: List[Dict[str, Any]],
    distance_rh: np.ndarray,
    distance_p: np.ndarray,
    pellet_p: np.ndarray,
    r_hand_p: np.ndarray,
    l_hand_p: np.ndarray,
    pellet_speed: np.ndarray,
    coord_pz: np.ndarray,
    min_threshold: float,
    max_threshold: float,
    peak_prominence: float,
    min_outward_travel: float,
    start_padding: int,
    min_frame: int,
    max_frame: int,
    frame_rate: float,
    confidence: float,
    pellet_drop_speed: float,
    pellet_drop_dist_z: float,
    pellet_dist_to_origin: float,
) -> Tuple[List[ReachSegment], List[Dict[str, Any]]]:
    reaches: List[ReachSegment] = []
    reach_details: List[Dict[str, Any]] = []
    outward = -_interp_nan_1d(np.asarray(distance_rh, dtype=np.float64))
    if outward.size == 0:
        return reaches, reach_details

    for idx, pellet_event in enumerate(pellet_events):
        if "placed" not in pellet_event:
            continue
        epoch_start = max(0, int(pellet_event["placed"]) - 20)
        next_start = int(pellet_events[idx + 1]["placed"]) if idx < len(pellet_events) - 1 else len(outward)
        epoch_end = min(len(outward), max(epoch_start + 1, next_start))
        periods = _find_kpn_outward_periods(
            outward[epoch_start:epoch_end],
            min_threshold,
            max_threshold,
            peak_prominence,
            min_outward_travel,
            start_padding,
        )

        for period in periods:
            start = int(epoch_start + period["start"])
            max_fr = int(epoch_start + period["first_peak"])
            end = int(epoch_start + period["end"])
            dur = end - start
            if end <= max_fr or max_fr <= start or dur < min_frame or dur > max_frame:
                continue

            outcome = _classify_kpn_reach_result(
                pellet_event,
                max_fr,
                end,
                distance_p,
                pellet_p,
                pellet_speed,
                coord_pz,
                start,
                confidence=confidence,
                pellet_drop_speed=pellet_drop_speed,
                pellet_drop_dist_z=pellet_drop_dist_z,
                pellet_dist_to_origin=pellet_dist_to_origin,
            )
            reach = ReachSegment(start, max_fr - start, dur, outcome, 0)
            if reach.is_valid and not _overlaps(reach, reaches):
                reaches.append(reach)
                peak_frames = [int(epoch_start + p) for p in period["peaks"]]
                mini_peak_frames = [p for p in peak_frames if p > max_fr]
                pellet_lost_frame = int(pellet_event.get("lost", -1))
                pellet_lost_conf = _confidence_at(pellet_p, pellet_lost_frame)
                reach_details.append(
                    {
                        "frame": int(start),
                        "max_frame": int(max_fr),
                        "end_frame": int(end),
                        "dur": int(dur),
                        "result": outcome.name,
                        "detection_method": "from_pellet",
                        "is_multi_reach": bool(mini_peak_frames),
                        "first_peak_frame": int(max_fr),
                        "first_peak_time_s": _frame_time(max_fr, frame_rate),
                        "peak_frames": peak_frames,
                        "peak_times_s": [
                            _frame_time(p, frame_rate) for p in peak_frames
                        ],
                        "mini_peak_frames": mini_peak_frames,
                        "mini_peak_times_s": [
                            _frame_time(p, frame_rate) for p in mini_peak_frames
                        ],
                        "peak_values": [float(v) for v in period["peak_values"]],
                        "min_threshold": float(min_threshold),
                        "max_threshold": float(max_threshold),
                        "min_cross_frame": int(epoch_start + period["min_cross"]),
                        "max_cross_frame": int(epoch_start + period["max_cross"]),
                        "outward_travel": float(period["outward_travel"]),
                        "pellet_available": bool(_is_pellet_available_for_reach(pellet_event, max_fr)),
                        "pellet_available_frame": int(pellet_event.get("available", -1)),
                        "pellet_placed_frame": int(pellet_event.get("placed", -1)),
                        "pellet_lost_frame": int(pellet_event.get("lost", -1)),
                        "point_confidence_threshold": float(confidence),
                        "rh_conf_mean": _confidence_mean(r_hand_p, start, end),
                        "rh_conf_median": _confidence_median(r_hand_p, start, end),
                        "rh_conf_min": _confidence_min(r_hand_p, start, end),
                        "rh_conf_fraction_valid": _confidence_fraction(
                            r_hand_p, start, end, confidence
                        ),
                        "lh_conf_mean": _confidence_mean(l_hand_p, start, end),
                        "lh_conf_median": _confidence_median(l_hand_p, start, end),
                        "lh_conf_min": _confidence_min(l_hand_p, start, end),
                        "lh_conf_fraction_valid": _confidence_fraction(
                            l_hand_p, start, end, confidence
                        ),
                        "pellet_conf_mean": _confidence_mean(pellet_p, start, end),
                        "pellet_conf_median": _confidence_median(pellet_p, start, end),
                        "pellet_conf_min": _confidence_min(pellet_p, start, end),
                        "pellet_conf_fraction_valid": _confidence_fraction(
                            pellet_p, start, end, confidence
                        ),
                        "pellet_conf_at_loss": pellet_lost_conf,
                        "pellet_method": str(pellet_event.get("method", "")),
                        "pellet_outcome": str(pellet_event.get("outcome", "")),
                        "pellet_method_frame": int(pellet_event.get("method_frame", -1)),
                        "pellet_lost_before_available": bool(
                            pellet_event.get("lost_before_available", False)
                        ),
                    }
                )

    reaches.sort(key=lambda seg: seg.frame)
    reach_details.sort(key=lambda detail: int(detail["frame"]))
    return reaches, reach_details


def _find_kpn_outward_periods(
    data: np.ndarray,
    min_threshold: float,
    max_threshold: float,
    peak_prominence: float,
    min_outward_travel: float,
    padding: int,
) -> List[Dict[str, Any]]:
    x = np.asarray(data, dtype=np.float64).reshape(-1)
    if x.size < 3:
        return []

    max_threshold = max(max_threshold, min_threshold)
    above_thresh = x >= min_threshold
    thresh_diffs = np.concatenate(([0], np.diff(above_thresh.astype(np.int8))))
    cross_up = np.where(thresh_diffs == 1)[0]
    cross_down = np.where(thresh_diffs == -1)[0]
    if cross_up.size == 0:
        return []

    periods: List[Dict[str, Any]] = []
    last_end = -1
    for up in cross_up:
        if int(up) <= last_end:
            continue
        down_candidates = cross_down[cross_down > up]
        down = int(down_candidates[0]) if down_candidates.size > 0 else x.size - 1
        if down <= up:
            continue

        max_cross_candidates = np.where(x[up : down + 1] >= max_threshold)[0]
        if max_cross_candidates.size == 0:
            last_end = down
            continue
        max_cross = int(up + max_cross_candidates[0])

        start_idx = _kpn_start_index_from_pretrace(x[: up + 1])
        start_with_padding = max(0, start_idx - padding)
        end_idx = _kpn_end_index_from_min_crossing(x, int(up), int(down))
        if end_idx <= start_with_padding:
            last_end = down
            continue

        reach = x[start_with_padding : end_idx + 1]
        peaks, _ = find_peaks(reach, prominence=peak_prominence)
        if peaks.size == 0:
            last_end = down
            continue

        peak_indices = np.asarray([int(start_with_padding + p) for p in peaks], dtype=np.int64)
        peak_indices = peak_indices[(peak_indices >= max_cross) & (x[peak_indices] >= max_threshold)]
        if peak_indices.size == 0:
            last_end = down
            continue

        peak_idx = int(peak_indices[0])
        baseline = min(float(x[start_with_padding]), min_threshold)
        outward_travel = float(x[peak_idx]) - baseline
        if peak_idx > start_with_padding and outward_travel >= min_outward_travel:
            periods.append(
                {
                    "start": int(start_with_padding),
                    "first_peak": peak_idx,
                    "end": int(end_idx),
                    "peaks": [int(p) for p in peak_indices],
                    "peak_values": [float(x[p]) for p in peak_indices],
                    "min_cross": int(up),
                    "max_cross": int(max_cross),
                    "outward_travel": float(outward_travel),
                }
            )
        last_end = down

    return periods


def _kpn_end_index_from_min_crossing(x: np.ndarray, up: int, down: int) -> int:
    return int(down)


def _kpn_start_index_from_pretrace(pre_x: np.ndarray) -> int:
    if pre_x.size < 3:
        return 0
    diff_pre_x = np.gradient(pre_x)
    diff_diff_pre_x = np.gradient(diff_pre_x)
    norm_v = _zscore_or_zero(diff_pre_x)
    norm_a = _zscore_or_zero(diff_diff_pre_x)
    diffs = norm_v - norm_a
    crossings = np.where(diffs[:-1] * diffs[1:] < 0)[0]
    if crossings.size > 0:
        closer = [int(i + (abs(diffs[i + 1]) < abs(diffs[i]))) for i in crossings]
        return closer[-1]
    falling = np.where(diff_pre_x < 0)[0]
    if falling.size > 0:
        return min(int(falling[-1] + 1), pre_x.size - 1)
    return 0


def _zscore_or_zero(v: np.ndarray) -> np.ndarray:
    sd = float(np.std(v))
    if sd == 0 or not np.isfinite(sd):
        return np.zeros_like(v, dtype=np.float64)
    return (v - float(np.mean(v))) / sd


def _confidence_window(confidence: np.ndarray, start: int, end: int) -> np.ndarray:
    lo = max(0, int(start))
    hi = min(len(confidence), max(lo, int(end) + 1))
    vals = np.asarray(confidence[lo:hi], dtype=np.float64)
    return vals[np.isfinite(vals)]


def _confidence_mean(confidence: np.ndarray, start: int, end: int) -> float:
    vals = _confidence_window(confidence, start, end)
    return float(np.nanmean(vals)) if vals.size else float("nan")


def _confidence_median(confidence: np.ndarray, start: int, end: int) -> float:
    vals = _confidence_window(confidence, start, end)
    return float(np.nanmedian(vals)) if vals.size else float("nan")


def _confidence_min(confidence: np.ndarray, start: int, end: int) -> float:
    vals = _confidence_window(confidence, start, end)
    return float(np.nanmin(vals)) if vals.size else float("nan")


def _confidence_fraction(
    confidence: np.ndarray,
    start: int,
    end: int,
    threshold: float,
) -> float:
    vals = _confidence_window(confidence, start, end)
    if vals.size == 0:
        return float("nan")
    return float(np.mean(vals >= threshold))


def _confidence_at(confidence: np.ndarray, frame_idx: int) -> float:
    if 0 <= frame_idx < len(confidence):
        value = float(confidence[frame_idx])
        return value if np.isfinite(value) else float("nan")
    return float("nan")


def _frame_time(frame_idx: int, frame_rate: float) -> float:
    rate = float(frame_rate)
    if rate <= 0 or not np.isfinite(rate):
        return float("nan")
    return float(frame_idx) / rate


def _classify_kpn_reach_result(
    pellet_event: Dict[str, Any],
    max_frame: int,
    end_frame: int,
    distance_p: np.ndarray,
    pellet_p: np.ndarray,
    pellet_speed: np.ndarray,
    coord_pz: np.ndarray,
    start_frame: int,
    *,
    confidence: float,
    pellet_drop_speed: float,
    pellet_drop_dist_z: float,
    pellet_dist_to_origin: float,
) -> ReachOutcome:
    if not _is_pellet_available_for_reach(pellet_event, max_frame):
        return ReachOutcome.MISSED

    lost = int(pellet_event.get("lost", -1))
    available = int(pellet_event.get("available", -1))
    lost_before_available = available >= 0 and lost > 0 and lost < available
    if lost > 0 and start_frame > lost and not lost_before_available:
        return ReachOutcome.DROPPED

    if lost > 0 and not lost_before_available:
        post_loss_slice = slice(lost, min(len(coord_pz), lost + _SPEED_WINDOW_NUM_FRAMES))
        high_conf_z = coord_pz[post_loss_slice][pellet_p[post_loss_slice] > confidence]
        post_loss_drop_z = abs(pellet_drop_dist_z) + 3
        if high_conf_z.size > 0 and np.nanmax(high_conf_z) > post_loss_drop_z:
            return ReachOutcome.DROPPED

    if str(pellet_event.get("method", "")) == "right_hand" and str(pellet_event.get("outcome", "")) == "eaten":
        return ReachOutcome.GRABBED

    if lost > max_frame:
        end_frame = min(end_frame, lost)

    if lost_before_available and max_frame >= available:
        return ReachOutcome.GRABBED

    speed_seg = np.asarray(pellet_speed[max_frame : min(len(pellet_speed), end_frame + 1)])
    if np.sum(speed_seg > pellet_drop_speed) > 1:
        return ReachOutcome.DROPPED

    pos_slice = slice(max_frame, min(len(coord_pz), end_frame + 1))
    if np.any(coord_pz[pos_slice] < pellet_drop_dist_z):
        z_dist_indices = np.where(coord_pz[pos_slice] < pellet_drop_dist_z)
        if np.any(pellet_p[pos_slice][z_dist_indices] > confidence):
            return ReachOutcome.DROPPED

    p_end = min(len(distance_p), end_frame + _BATCH_FRM)
    pellet_still_home = (
        np.nanmean(distance_p[end_frame:p_end]) < pellet_dist_to_origin
        if p_end > end_frame
        else False
    )
    if pellet_still_home:
        return ReachOutcome.MISSED
    if lost > 0 and max_frame <= lost <= end_frame:
        return ReachOutcome.GRABBED
    if str(pellet_event.get("outcome", "")) == "dropped":
        return ReachOutcome.DROPPED
    return ReachOutcome.GRABBED if str(pellet_event.get("outcome", "")) == "eaten" else ReachOutcome.MISSED


def _is_pellet_available_for_reach(pellet_event: Dict[str, Any], max_frame: int) -> bool:
    if "available" not in pellet_event:
        return True
    available_frame = int(pellet_event.get("available", -1))
    return available_frame >= 0 and max_frame >= available_frame


def _overlaps(test_seg: ReachSegment, existing: List[ReachSegment]) -> bool:
    for seg in existing:
        if test_seg.frame <= seg.end_frame and test_seg.end_frame >= seg.frame:
            return True
    return False


def classify_reach_outcome(
    signal: np.ndarray,
    reach: ReachSegment,
    *,
    events: Optional[List[Dict[str, Any]]] = None,
    **kwargs,
) -> ReachOutcome:
    """Compatibility classifier; KPN fixed-cam detection classifies internally."""
    return reach.outcome if reach.outcome != ReachOutcome.UNCLASSIFIED else ReachOutcome.MISSED


def classify_reaches(
    signal: np.ndarray,
    reaches: List[ReachSegment],
    *,
    events: Optional[List[Dict[str, Any]]] = None,
    **kwargs,
) -> List[ReachSegment]:
    """Compatibility wrapper; does not assign STALLED for KPN detections."""
    return reaches


def save_reaches(reaches: List[ReachSegment], path: str | Path) -> None:
    """Save canonical ReachX-compatible reach segments."""
    path = Path(path)
    if not reaches:
        path.unlink(missing_ok=True)
        return
    raw = np.zeros((len(reaches), 5), dtype="i4")
    for idx, reach in enumerate(reaches):
        raw[idx] = [
            int(reach.frame),
            int(reach.max_delta),
            int(reach.dur),
            int(reach.outcome),
            int(reach.hand_pos),
        ]
    np.savetxt(path, raw, fmt="%d", header="frame  max_delta  dur  result  hand_pos")


def load_reaches(path: str | Path) -> List[ReachSegment]:
    """Load reach segments from 4- or 5-column text/CSV files."""
    path = Path(path)
    if not path.exists():
        return []
    reaches: List[ReachSegment] = []
    try:
        raw = np.loadtxt(path, dtype="i4", ndmin=2, delimiter=None)
    except Exception:
        try:
            raw = np.loadtxt(path, dtype="i4", ndmin=2, delimiter=",", comments="#", skiprows=1)
        except Exception:
            return []

    if raw.ndim != 2 or raw.shape[1] < 4:
        return []
    for row in raw:
        try:
            outcome = ReachOutcome(int(row[3]))
        except ValueError:
            outcome = ReachOutcome.UNCLASSIFIED
        hand_pos = int(row[4]) if raw.shape[1] >= 5 else 0
        reach = ReachSegment(int(row[0]), int(row[1]), int(row[2]), outcome, hand_pos)
        if reach.is_valid and not _overlaps(reach, reaches):
            reaches.append(reach)
    reaches.sort(key=lambda r: r.frame)
    return reaches


def save_kpn_reach_details(reach_details: List[Dict[str, Any]], out_file: str | Path) -> None:
    Path(out_file).write_text(json.dumps(reach_details, indent=2), encoding="utf-8")


def save_reach_detection_info(info: Dict[str, Any], out_file: str | Path) -> None:
    """Save provenance/settings used to generate detected reaches."""
    Path(out_file).write_text(json.dumps(info, indent=2), encoding="utf-8")


def save_kpn_reach_details_table(reach_details: List[Dict[str, Any]], out_file: str | Path) -> None:
    columns = [
        "frame",
        "max_frame",
        "end_frame",
        "dur",
        "result",
        "detection_method",
        "is_multi_reach",
        "first_peak_frame",
        "first_peak_time_s",
        "peak_frames",
        "peak_times_s",
        "mini_peak_frames",
        "mini_peak_times_s",
        "peak_values",
        "threshold",
        "min_threshold",
        "max_threshold",
        "min_cross_frame",
        "max_cross_frame",
        "threshold_down_frame",
        "outward_travel",
        "start_value",
        "max_start_value",
        "absolute_axis",
        "absolute_invert",
        "pellet_available",
        "pellet_available_frame",
        "pellet_placed_frame",
        "pellet_lost_frame",
        "point_confidence_threshold",
        "rh_conf_mean",
        "rh_conf_median",
        "rh_conf_min",
        "rh_conf_fraction_valid",
        "lh_conf_mean",
        "lh_conf_median",
        "lh_conf_min",
        "lh_conf_fraction_valid",
        "pellet_conf_mean",
        "pellet_conf_median",
        "pellet_conf_min",
        "pellet_conf_fraction_valid",
        "pellet_conf_at_loss",
        "pellet_method",
        "pellet_outcome",
        "pellet_method_frame",
        "pellet_lost_before_available",
    ]
    lines = ["\t".join(columns)]
    for detail in reach_details:
        lines.append("\t".join(_format_detail_value(detail.get(col, "")) for col in columns))
    Path(out_file).write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_kpn_reach_details_csv(reach_details: List[Dict[str, Any]], out_file: str | Path) -> None:
    columns = [
        "frame",
        "max_frame",
        "end_frame",
        "dur",
        "result",
        "detection_method",
        "is_multi_reach",
        "first_peak_frame",
        "first_peak_time_s",
        "peak_frames",
        "peak_times_s",
        "mini_peak_frames",
        "mini_peak_times_s",
        "peak_values",
        "threshold",
        "outward_travel",
        "start_value",
        "max_start_value",
        "absolute_axis",
        "absolute_invert",
        "pellet_available",
        "pellet_available_frame",
        "pellet_placed_frame",
        "pellet_lost_frame",
        "point_confidence_threshold",
        "rh_conf_mean",
        "rh_conf_median",
        "rh_conf_min",
        "rh_conf_fraction_valid",
        "lh_conf_mean",
        "lh_conf_median",
        "lh_conf_min",
        "lh_conf_fraction_valid",
        "pellet_conf_mean",
        "pellet_conf_median",
        "pellet_conf_min",
        "pellet_conf_fraction_valid",
        "pellet_conf_at_loss",
        "pellet_method",
        "pellet_outcome",
        "pellet_method_frame",
        "pellet_lost_before_available",
    ]
    with Path(out_file).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for detail in reach_details:
            writer.writerow({col: _format_detail_value(detail.get(col, "")) for col in columns})


def save_pellet_history(pellet_history: List[Dict[str, Any]], out_file: str | Path) -> None:
    with open(out_file, "wb") as f:
        pickle.dump(pellet_history, f)


def _format_detail_value(value: Any) -> str:
    if isinstance(value, list):
        return ",".join(_format_detail_value(v) for v in value)
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)
