"""Utilities for loading and displaying session event text files."""

from __future__ import annotations

import re
import colorsys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from sleap_io import Labels, RecordingSession, Video


SESSION_EVENTS_KEY = "events"
SESSION_EVENTS_PATH_KEY = "events_path"

_EVENT_COLORS: Tuple[Tuple[int, int, int], ...] = (
    (0, 114, 178),
    (213, 94, 0),
    (0, 158, 115),
    (204, 121, 167),
    (230, 159, 0),
    (86, 180, 233),
    (240, 228, 66),
    (128, 64, 191),
    (80, 80, 80),
    (190, 85, 85),
    (85, 170, 110),
    (70, 130, 180),
)


def find_session_event_file(session_path: str | Path) -> Optional[Path]:
    """Return the first likely session events file directly in a session folder."""
    session_path = Path(session_path)
    if not session_path.is_dir():
        return None

    candidates = sorted(
        (
            path
            for path in session_path.iterdir()
            if path.is_file() and path.suffix.lower() == ".txt"
        ),
        key=lambda path: path.name.lower(),
    )
    event_files = [
        path
        for path in candidates
        if path.stem.lower() == "events" or path.stem.lower().endswith("_events")
    ]
    return event_files[0] if event_files else None


def load_session_events_file(events_path: str | Path) -> List[Dict[str, object]]:
    """Load a two-column event text file into serializable event dictionaries.

    Accepted rows are either ``event_name frame`` or ``frame event_name``. Blank
    lines, comment lines, and header-like rows are skipped.
    """
    events_path = Path(events_path)
    events: List[Dict[str, object]] = []

    with open(events_path, "r") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            tokens = [token for token in re.split(r"[\s,]+", line) if token]
            if len(tokens) < 2:
                continue

            if tokens[0].lower() in {"event", "event_type", "name"}:
                continue

            first_is_frame = tokens[0].lstrip("+-").isdigit()
            second_is_frame = tokens[1].lstrip("+-").isdigit()
            if first_is_frame and not second_is_frame:
                frame_token, event_name = tokens[0], tokens[1]
            else:
                event_name, frame_token = tokens[0], tokens[1]

            try:
                frame = int(frame_token)
            except ValueError:
                continue
            if frame < 0:
                continue

            events.append({"event": str(event_name), "frame": frame, "line": line_no})

    return sorted(events, key=lambda event: int(event["frame"]))


def set_session_events(
    session: RecordingSession, events_path: str | Path, events: List[Dict[str, object]]
) -> bool:
    """Store loaded events in recording-session metadata.

    Returns True when metadata changed.
    """
    events_path = str(Path(events_path))
    old_path = session.metadata.get(SESSION_EVENTS_PATH_KEY)
    old_events = session.metadata.get(SESSION_EVENTS_KEY)
    session.metadata[SESSION_EVENTS_PATH_KEY] = events_path
    session.metadata[SESSION_EVENTS_KEY] = events
    return old_path != events_path or old_events != events


def maybe_load_session_events(session: RecordingSession, session_path: str | Path) -> bool:
    """Load a session's events file from its folder when one exists."""
    events_path = find_session_event_file(session_path)
    if events_path is None:
        return False
    return set_session_events(session, events_path, load_session_events_file(events_path))


def get_session_for_video(
    labels: Optional[Labels], video: Optional[Video]
) -> Optional[RecordingSession]:
    """Return the recording session containing a video, if any."""
    if labels is None or video is None:
        return None

    for session in getattr(labels, "sessions", []) or []:
        if video in getattr(session, "videos", []):
            return session
    return None


def get_session_events(session: Optional[RecordingSession]) -> List[Dict[str, object]]:
    """Return normalized events from recording-session metadata."""
    if session is None:
        return []

    events = session.metadata.get(SESSION_EVENTS_KEY, [])
    out = []
    for event in events or []:
        try:
            out.append({"event": str(event["event"]), "frame": int(event["frame"])})
        except (KeyError, TypeError, ValueError):
            continue
    return sorted(out, key=lambda event: int(event["frame"]))


def get_video_session_events(
    labels: Optional[Labels], video: Optional[Video]
) -> List[Dict[str, object]]:
    """Return events for the session containing a video."""
    return get_session_events(get_session_for_video(labels, video))


def event_names(events: Iterable[Dict[str, object]]) -> List[str]:
    """Return unique event names in display order."""
    return sorted({str(event["event"]) for event in events})


def event_color_map(events: Iterable[Dict[str, object]]) -> Dict[str, Tuple[int, int, int]]:
    """Return stable, unique-ish colors for each event name."""
    colors = {}
    for i, name in enumerate(event_names(events)):
        if i < len(_EVENT_COLORS):
            colors[name] = _EVENT_COLORS[i]
            continue

        hue = ((i - len(_EVENT_COLORS)) * 0.61803398875) % 1.0
        red, green, blue = colorsys.hsv_to_rgb(hue, 0.65, 0.8)
        colors[name] = (int(red * 255), int(green * 255), int(blue * 255))
    return colors
