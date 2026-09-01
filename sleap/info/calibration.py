"""Report reprojection-error quality metrics for a multi-camera calibration.

Consumes the two files written by the SLEAP Calibration dock (or ``anipose`` /
``sleap-anipose``):

* ``<prefix>.toml`` -- the serialised ``aniposelib.CameraGroup``.
* ``<prefix>.metadata.h5`` -- detected / triangulated ChArUco corner data.

The reprojection error is recomputed here from the *raw* detected corners
(``detected_corners`` in the metadata file) via a plain per-frame
``CameraGroup.triangulate`` + ``project`` round trip.  It deliberately does
**not** reuse the ``triangulated_corners`` / ``reprojected_corners`` datasets
stored in the metadata file: those come from ``aniposelib``'s
``triangulate_optim``, which applies a strong temporal-smoothness prior
(``scale_smooth=10000``) meant for continuous animal trajectories.  Calibration
board detections are sampled sparsely while the board is waved around, so that
prior is inappropriate and badly inflates the stored figures.

Command line::

    sleap-calibration-quality calibration_20260729.toml
    sleap-calibration-quality calibration_20260729.metadata.h5
    sleap-calibration-quality /path/to/calibrations           # folder
    sleap-calibration-quality calibrations/calibration_20260729   # shared prefix
    sleap-calibration-quality calibration_20260729.toml --plot -o

``--plot`` / ``-o`` with no path write ``<prefix>.quality.png`` / ``.txt``
into a ``calibration_quality/`` folder next to the calibration files. Also
runnable as ``python -m sleap.info.calibration``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Union

import numpy as np

#: RMS reprojection error (px) below which a calibration is called "GOOD".
GOOD_PX = 1.0
#: RMS reprojection error (px) below which a calibration is called "OK".
OK_PX = 2.0

_TOML_SUFFIX = ".toml"
_METADATA_SUFFIX = ".metadata.h5"


def resolve_calibration_paths(
    calibration: Union[str, Path],
    metadata: Optional[Union[str, Path]] = None,
) -> tuple[Path, Path]:
    """Resolve the ``.toml`` / ``.metadata.h5`` pair from a single argument.

    Args:
        calibration: One of:

            * the ``<prefix>.toml`` file,
            * the ``<prefix>.metadata.h5`` file,
            * the shared ``<prefix>`` (no suffix),
            * a folder containing exactly one ``calibration*.toml``.
        metadata: Optional explicit path to the ``.metadata.h5`` file,
            overriding the derived location.

    Returns:
        ``(toml_path, metadata_path)`` -- both guaranteed to exist.
    """
    path = Path(calibration).expanduser()

    if path.is_dir():
        tomls = sorted(path.glob("*.toml"))
        tomls = [t for t in tomls if not t.name.endswith(_METADATA_SUFFIX)]
        if not tomls:
            raise FileNotFoundError(f"No calibration '*.toml' found in {path}")
        if len(tomls) > 1:
            names = ", ".join(t.name for t in tomls)
            raise ValueError(
                f"Multiple calibration TOMLs in {path} ({names}). "
                "Pass the specific '.toml' file."
            )
        toml_path = tomls[0]
    else:
        name = path.name
        if name.endswith(_METADATA_SUFFIX):
            toml_path = path.with_name(name[: -len(_METADATA_SUFFIX)] + _TOML_SUFFIX)
        elif path.suffix == _TOML_SUFFIX:
            toml_path = path
        elif path.suffix == "":
            toml_path = path.with_name(name + _TOML_SUFFIX)
        else:
            raise ValueError(
                f"Cannot interpret {str(path)!r}. Pass the calibration '.toml', "
                "the '.metadata.h5', the shared prefix, or the folder."
            )

    prefix = toml_path.name[: -len(_TOML_SUFFIX)]
    metadata_path = (
        Path(metadata).expanduser()
        if metadata is not None
        else toml_path.with_name(prefix + _METADATA_SUFFIX)
    )

    if not toml_path.exists():
        raise FileNotFoundError(f"Calibration TOML not found: {toml_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Calibration metadata not found: {metadata_path}\n"
            "Expected a '<prefix>.metadata.h5' next to the '.toml' "
            "(pass --metadata to point elsewhere)."
        )
    return toml_path, metadata_path


def verdict(rms_px: float) -> str:
    """Return a one-word quality label for an RMS reprojection error."""
    if not np.isfinite(rms_px):
        return "NO DATA"
    if rms_px < GOOD_PX:
        return "GOOD"
    if rms_px < OK_PX:
        return "OK"
    return "CHECK"


def _summarize(err: np.ndarray) -> dict:
    """NaN-aware summary stats (px) for an error array."""
    valid = err[~np.isnan(err)]
    if valid.size == 0:
        return {"n": 0, "mean": float("nan"), "median": float("nan"),
                "rms": float("nan"), "p95": float("nan"), "max": float("nan")}
    return {
        "n": int(valid.size),
        "mean": float(np.mean(valid)),
        "median": float(np.median(valid)),
        "rms": float(np.sqrt(np.mean(np.square(valid)))),
        "p95": float(np.percentile(valid, 95)),
        "max": float(np.max(valid)),
    }


def _load_detected_corners(metadata_path: Path) -> np.ndarray:
    """Load ``detected_corners`` -- ``(n_cams, n_frames, n_corners, 2)``."""
    import h5py

    with h5py.File(metadata_path, "r") as f:
        if "detected_corners" not in f:
            raise KeyError(
                f"{metadata_path} has no 'detected_corners' dataset -- is this a "
                "sleap-anipose calibration metadata file?"
            )
        return f["detected_corners"][:]


def _load_camera_group(toml_path: Path):
    try:
        from aniposelib.cameras import CameraGroup
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ImportError(
            "aniposelib is not installed in this environment. Install the SLEAP "
            "anipose extras (e.g. `uv sync --extra anipose`) to use "
            "sleap-calibration-quality."
        ) from exc
    return CameraGroup.load(str(toml_path))


def _reprojection_error(cgroup, detected: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-point pixel error and triangulated 3D points, with no temporal prior.

    Returns:
        ``(error_px, points_3d)`` shaped ``(n_cams, n_frames, n_corners)`` and
        ``(n_frames, n_corners, 3)``.
    """
    n_cams, n_frames, n_corners, _ = detected.shape
    pts = detected.reshape(n_cams, n_frames * n_corners, 2)
    p3d = cgroup.triangulate(pts, undistort=True)
    reproj = cgroup.project(p3d)
    err = np.linalg.norm(pts - reproj, axis=-1)
    return (
        err.reshape(n_cams, n_frames, n_corners),
        p3d.reshape(n_frames, n_corners, 3),
    )


def _camera_positions(cgroup) -> np.ndarray:
    """``(n_cams, 3)`` camera centres in world coordinates."""
    import cv2

    positions = []
    for cam in cgroup.cameras:
        rot, _ = cv2.Rodrigues(np.asarray(cam.get_rotation(), dtype="float64"))
        positions.append(-rot.T @ np.asarray(cam.get_translation(), dtype="float64"))
    return np.array(positions)


def evaluate_calibration(
    calibration: Union[str, Path],
    metadata: Optional[Union[str, Path]] = None,
) -> dict:
    """Compute reprojection-error quality metrics for a calibration.

    Args:
        calibration: The ``.toml``, the ``.metadata.h5``, the shared prefix, or
            the containing folder (see :func:`resolve_calibration_paths`).
        metadata: Optional explicit ``.metadata.h5`` path.

    Returns:
        A dict with keys ``toml``, ``metadata``, ``n_cameras``, ``n_frames``,
        ``n_corners``, ``overall`` (stats dict), ``per_camera`` (list of stats
        dicts with ``name``/``verdict``), ``verdict``, ``worst_camera``,
        ``camera_positions``, ``pairwise_distance_px`` and
        ``triangulated_bbox``.  A non-serialisable ``_error_px`` array is also
        included for plotting.
    """
    toml_path, metadata_path = resolve_calibration_paths(calibration, metadata)
    cgroup = _load_camera_group(toml_path)
    detected = _load_detected_corners(metadata_path)
    if detected.ndim != 4 or detected.shape[-1] != 2:
        raise ValueError(
            f"Unexpected 'detected_corners' shape {detected.shape}; expected "
            "(n_cams, n_frames, n_corners, 2)."
        )

    names = list(cgroup.get_names())
    if len(names) != detected.shape[0]:
        raise ValueError(
            f"{toml_path.name} has {len(names)} cameras but "
            f"{metadata_path.name} has {detected.shape[0]} views."
        )

    err, p3d = _reprojection_error(cgroup, detected)
    n_cams, n_frames, n_corners = err.shape

    per_camera = []
    for i, name in enumerate(names):
        stats = _summarize(err[i])
        stats["name"] = name
        stats["verdict"] = verdict(stats["rms"])
        per_camera.append(stats)

    overall = _summarize(err)
    rms_values = np.array([c["rms"] for c in per_camera], dtype="float64")
    finite = rms_values[np.isfinite(rms_values)]
    worst_camera = None
    if finite.size > 1 and np.max(finite) > 1.5 * np.median(finite):
        worst_camera = names[int(np.nanargmax(rms_values))]

    positions = _camera_positions(cgroup)
    pairwise = None
    if len(positions) > 1:
        dmat = np.linalg.norm(
            positions[:, None, :] - positions[None, :, :], axis=-1
        )
        iu = np.triu_indices(len(positions), k=1)
        pairwise = {"min": float(dmat[iu].min()), "max": float(dmat[iu].max())}

    finite3d = p3d[np.isfinite(p3d).all(axis=-1)]
    bbox = None
    if len(finite3d):
        bbox = {
            "min": finite3d.min(axis=0).tolist(),
            "max": finite3d.max(axis=0).tolist(),
        }

    return {
        "toml": str(toml_path),
        "metadata": str(metadata_path),
        "n_cameras": n_cams,
        "n_frames": n_frames,
        "n_corners": n_corners,
        "overall": overall,
        "per_camera": per_camera,
        "verdict": verdict(overall["rms"]),
        "worst_camera": worst_camera,
        "camera_positions": {
            name: pos.tolist() for name, pos in zip(names, positions)
        },
        "pairwise_distance_px": pairwise,
        "triangulated_bbox": bbox,
        "_error_px": err,
    }


def format_report(result: dict) -> str:
    """Render :func:`evaluate_calibration` output as a plain-text report."""
    lines = []
    add = lines.append
    bar = "=" * 74

    add(bar)
    add("CALIBRATION QUALITY REPORT  (unsmoothed per-frame reprojection error)")
    add(bar)
    add(f"toml     : {result['toml']}")
    add(f"metadata : {result['metadata']}")
    add(
        f"cameras  : {result['n_cameras']}    "
        f"common frames : {result['n_frames']}    "
        f"board corners : {result['n_corners']}"
    )
    add("")

    o = result["overall"]
    add("Overall reprojection error (all cameras, all common frames):")
    add(
        f"  mean={o['mean']:.3f}px  median={o['median']:.3f}px  "
        f"RMS={o['rms']:.3f}px  p95={o['p95']:.3f}px  max={o['max']:.3f}px"
    )
    add(f"  -> {result['verdict']}")
    add("")

    add(
        f"{'Camera':<30} {'n_pts':>8} {'mean':>7} {'RMS':>7} {'p95':>7} "
        f"{'max':>7}  verdict"
    )
    add("-" * 74)
    for c in result["per_camera"]:
        if c["n"] == 0:
            add(f"{c['name']:<30} {'--':>8} {'--':>7} {'--':>7} {'--':>7} "
                f"{'--':>7}  NO DATA")
            continue
        add(
            f"{c['name']:<30} {c['n']:>8} {c['mean']:>7.3f} {c['rms']:>7.3f} "
            f"{c['p95']:>7.3f} {c['max']:>7.3f}  {c['verdict']}"
        )

    if result["worst_camera"]:
        add("")
        add(
            f"NOTE: {result['worst_camera']} is notably worse than the other "
            "cameras -- check its focus, exposure, and board coverage."
        )

    if result["pairwise_distance_px"]:
        add("")
        add("Estimated camera positions (world units == charuco square_length):")
        for name, pos in result["camera_positions"].items():
            add(f"  {name:<30} [{pos[0]:9.2f} {pos[1]:9.2f} {pos[2]:9.2f}]")
        pw = result["pairwise_distance_px"]
        add(f"  pairwise camera distance: min={pw['min']:.2f}  max={pw['max']:.2f}")

    if result["triangulated_bbox"]:
        bb = result["triangulated_bbox"]
        add("")
        add("Triangulated board-corner cloud bounding box (world units):")
        add(f"  min=[{bb['min'][0]:.2f} {bb['min'][1]:.2f} {bb['min'][2]:.2f}]  "
            f"max=[{bb['max'][0]:.2f} {bb['max'][1]:.2f} {bb['max'][2]:.2f}]")

    add(bar)
    add(
        "Verdict thresholds (RMS px): GOOD < "
        f"{GOOD_PX:g} <= OK < {OK_PX:g} <= CHECK. Tune to your pixel scale."
    )
    add(bar)
    return "\n".join(lines)


def _strip_private(result: dict) -> dict:
    return {k: v for k, v in result.items() if not k.startswith("_")}


def render_plot(result: dict, out_path: Union[str, Path]) -> Path:
    """Save a histogram + per-camera bar chart PNG. Returns the output path."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    err = result["_error_px"]
    names = [c["name"] for c in result["per_camera"]]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].hist(err[~np.isnan(err)], bins=60, color="#4C72B0")
    axes[0].set_xlabel("Reprojection error (px)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Overall error distribution")

    means = [c["mean"] for c in result["per_camera"]]
    axes[1].bar(range(len(names)), means, color="#55A868")
    axes[1].set_xticks(range(len(names)))
    axes[1].set_xticklabels(names, rotation=45, ha="right")
    axes[1].set_ylabel("Mean reprojection error (px)")
    axes[1].set_title("Per-camera mean error")

    fig.tight_layout()
    out_path = Path(out_path)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


#: Sub-folder (next to the calibration files) for auto-named output.
OUTPUT_DIRNAME = "calibration_quality"


def _default_output_path(metadata_path: str, extension: str) -> Path:
    """``<meta_dir>/calibration_quality/<prefix><extension>``.

    ``prefix`` is the calibration prefix (the metadata name minus
    ``.metadata.h5``). The sub-folder is created if needed.
    """
    meta = Path(metadata_path)
    name = meta.name
    if name.endswith(_METADATA_SUFFIX):
        name = name[: -len(_METADATA_SUFFIX)]
    else:
        name = meta.stem
    out_dir = meta.parent / OUTPUT_DIRNAME
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / (name + extension)


def main(argv: Optional[list[str]] = None) -> None:
    """CLI entry point for ``sleap-calibration-quality``."""
    parser = argparse.ArgumentParser(
        prog="sleap-calibration-quality",
        description=(
            "Report reprojection-error quality metrics for a multi-camera "
            "calibration produced by the SLEAP Calibration dock."
        ),
    )
    parser.add_argument(
        "calibration",
        help="Calibration '.toml', '.metadata.h5', shared prefix, or folder.",
    )
    parser.add_argument(
        "--metadata",
        default=None,
        help="Explicit path to the '.metadata.h5' file (overrides the derived "
        "location).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the metrics as JSON instead of a text report.",
    )
    parser.add_argument(
        "--plot",
        nargs="?",
        const="auto",
        default=None,
        metavar="PATH",
        help="Also save a histogram + per-camera bar chart PNG. With no PATH, "
        "writes 'calibration_quality/<prefix>.quality.png' next to the "
        "calibration files.",
    )
    parser.add_argument(
        "-o",
        "--output",
        nargs="?",
        const="auto",
        default=None,
        metavar="PATH",
        help="Also write the report to a file. With no PATH, writes "
        "'calibration_quality/<prefix>.quality.txt' (or '.json') next to the "
        "calibration files.",
    )
    args = parser.parse_args(argv)

    result = evaluate_calibration(args.calibration, args.metadata)

    if args.json:
        rendered = json.dumps(_strip_private(result), indent=2)
        default_ext = ".quality.json"
    else:
        rendered = format_report(result)
        default_ext = ".quality.txt"
    print(rendered)

    if args.output is not None:
        out_path = (
            _default_output_path(result["metadata"], default_ext)
            if args.output == "auto"
            else Path(args.output)
        )
        out_path.write_text(rendered + "\n", encoding="utf-8")
        print(f"\nSaved report to {out_path}")

    if args.plot is not None:
        plot_path = (
            _default_output_path(result["metadata"], ".quality.png")
            if args.plot == "auto"
            else Path(args.plot)
        )
        render_plot(result, plot_path)
        print(f"Saved plot to {plot_path}")


if __name__ == "__main__":  # pragma: no cover
    main()
