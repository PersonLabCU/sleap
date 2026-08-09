"""Dock widget for running model inference across many session folders."""

import json
import logging
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from qtpy import QtCore
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLayout,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from sleap.gui.dialogs.filedialog import FileDialog
from sleap.gui.learning.runners import (
    InferenceProgressDialog,
    InferenceTask,
    VideoItemForInference,
    kill_process,
)
from sleap.gui.widgets.docks import DockWidget
from sleap_io.io.video_reading import MediaVideo
from sleap_io import Video

logger = logging.getLogger(__name__)
MOVIE_EXTENSIONS = {f".{ext.lower()}" for ext in MediaVideo.EXTS}


class BatchAnalysisWorker(QtCore.QThread):
    """Background worker for batch inference and optional 3D projection export."""

    progressUpdate = QtCore.Signal(int, int)
    statusUpdate = QtCore.Signal(str)
    logOutput = QtCore.Signal(str)
    finished = QtCore.Signal(bool, dict, str)

    def __init__(
        self,
        *,
        model_paths: Sequence[str],
        calibration_path: str,
        parent_dir: str,
        sessions: Sequence[Tuple[Path, Sequence[Path]]],
        batch_size: int,
        max_instances: int,
        export_analysis_h5: bool = False,
        export_nwb: bool = False,
        detect_reaches: bool = False,
        reach_source: str = "3d",
        reach_camera: str = "",
        reach_settings: Optional[Dict[str, Any]] = None,
        parent=None,
    ):
        super().__init__(parent)
        self._model_paths = list(model_paths)
        self._calibration_path = calibration_path
        self._parent_dir = parent_dir
        self._sessions = [
            (Path(session_dir), [Path(video_path) for video_path in video_paths])
            for session_dir, video_paths in sessions
        ]
        self._batch_size = batch_size
        self._max_instances = max_instances
        self._export_analysis_h5 = export_analysis_h5
        self._export_nwb = export_nwb
        self._detect_reaches = detect_reaches
        self._reach_source = reach_source
        self._reach_camera = reach_camera
        self._reach_settings = dict(reach_settings or {})
        self._canceled = False
        self._current_process = None

    def cancel(self) -> None:
        self._canceled = True
        if self._current_process is not None:
            kill_process(self._current_process.pid)

    def run(self) -> None:
        summary = {
            "sessions": 0,
            "videos": 0,
            "predictions": 0,
            "exports": 0,
            "projections": 0,
            "reach_sessions": 0,
            "reaches": 0,
            "skipped_sessions": [],
            "skipped_reaches": [],
            "failed_sessions": [],
        }
        try:
            sessions = self._sessions
            if not sessions:
                raise ValueError("No session folders with supported videos were found.")

            total_steps = sum(len(videos) for _, videos in sessions)
            if self._calibration_path:
                total_steps += len(sessions)
            if self._detect_reaches:
                total_steps += len(sessions)
            total_steps = max(total_steps, 1)
            current_step = 0

            self.logOutput.emit(f"Parent directory: {self._parent_dir}")
            for model_path in self._model_paths:
                self.logOutput.emit(f"Model: {model_path}")
            if self._calibration_path:
                self.logOutput.emit(f"Calibration: {self._calibration_path}")
            if self._detect_reaches:
                if self._reach_source == "2d":
                    self.logOutput.emit(
                        f"Reach detection: enabled (2D camera {self._reach_camera})"
                    )
                else:
                    self.logOutput.emit("Reach detection: enabled (3D points)")
            self.logOutput.emit(f"Found {len(sessions)} session folder(s).")

            for session_dir, video_paths in sessions:
                self._raise_if_canceled()
                session_step_start = current_step
                try:
                    summary["sessions"] += 1
                    self.logOutput.emit("")
                    self.logOutput.emit(f"Session: {session_dir}")
                    prediction_paths = []
                    prediction_items = []
                    points3d_path = None

                    for video_path in video_paths:
                        self._raise_if_canceled()
                        current_step += 1
                        self.progressUpdate.emit(current_step - 1, total_steps)
                        self.statusUpdate.emit(
                            f"<b>Running inference</b><br>{video_path.name}"
                        )
                        prediction_path = self._run_video_inference(
                            session_dir, video_path
                        )
                        prediction_paths.append(prediction_path)
                        prediction_items.append((video_path, prediction_path))
                        summary["videos"] += 1
                        summary["predictions"] += 1
                        if self._export_analysis_h5 or self._export_nwb:
                            summary["exports"] += self._export_prediction_formats(
                                prediction_path
                            )
                        self.progressUpdate.emit(current_step, total_steps)

                    if self._calibration_path:
                        self._raise_if_canceled()
                        current_step += 1
                        self.progressUpdate.emit(current_step - 1, total_steps)
                        if len(prediction_paths) < 2:
                            message = (
                                f"{session_dir.name}: fewer than two prediction files; "
                                "skipping 3D projection."
                            )
                            summary["skipped_sessions"].append(message)
                            self.logOutput.emit(message)
                        else:
                            self.statusUpdate.emit(
                                f"<b>Running 3D projection</b><br>{session_dir}"
                            )
                            projection_metadata = self._run_projection(
                                session_dir, prediction_paths
                            )
                            points3d_path = projection_metadata.get("points3d_path")
                            summary["projections"] += 1
                        self.progressUpdate.emit(current_step, total_steps)

                    if self._detect_reaches:
                        self._raise_if_canceled()
                        current_step += 1
                        self.progressUpdate.emit(current_step - 1, total_steps)
                        try:
                            self.statusUpdate.emit(
                                f"<b>Detecting reaches</b><br>{session_dir}"
                            )
                            if self._reach_source == "2d":
                                item = self._prediction_for_reach_camera(
                                    prediction_items
                                )
                                if item is None:
                                    raise ValueError(
                                        "no prediction file matched camera "
                                        f"{self._reach_camera}"
                                    )
                                video_path, prediction_path = item
                                n_reaches = self._run_reach_detection(
                                    session_dir,
                                    prediction_path=prediction_path,
                                    source_video_path=video_path,
                                )
                            else:
                                if not points3d_path:
                                    raise ValueError("no points3D output")
                                n_reaches = self._run_reach_detection(
                                    session_dir,
                                    points3d_path=points3d_path,
                                    source_video_path=(
                                        prediction_items[0][0]
                                        if prediction_items
                                        else None
                                    ),
                                )
                        except InterruptedError:
                            raise
                        except Exception as exc:
                            message = (
                                f"{session_dir.name}: reach detection skipped - {exc}. "
                                "Continuing batch."
                            )
                            summary["skipped_reaches"].append(message)
                            self.logOutput.emit(message)
                            self.statusUpdate.emit(
                                "<b>Reach detection skipped</b><br>"
                                f"{session_dir.name}: {exc}"
                            )
                        else:
                            summary["reach_sessions"] += 1
                            summary["reaches"] += n_reaches
                        self.progressUpdate.emit(current_step, total_steps)
                except InterruptedError:
                    raise
                except Exception as exc:
                    message = (
                        f"{session_dir.name}: session failed - {exc}. "
                        "Continuing with the next session."
                    )
                    summary["failed_sessions"].append(message)
                    self.logOutput.emit(message)
                    self.statusUpdate.emit(
                        f"<b>Session skipped</b><br>{session_dir.name}: {exc}"
                    )
                    expected_session_steps = len(video_paths)
                    if self._calibration_path:
                        expected_session_steps += 1
                    if self._detect_reaches:
                        expected_session_steps += 1
                    current_step = max(
                        current_step,
                        session_step_start + expected_session_steps,
                    )
                    self.progressUpdate.emit(current_step, total_steps)

            self.finished.emit(True, summary, "")
        except InterruptedError as exc:
            self.finished.emit(False, summary, str(exc))
        except Exception as exc:
            logger.exception("Batch analysis failed")
            self.logOutput.emit(f"Error: {exc}")
            self.finished.emit(False, summary, str(exc))

    def _run_video_inference(self, session_dir: Path, video_path: Path) -> str:
        predictions_dir = session_dir / "predictions"
        predictions_dir.mkdir(parents=True, exist_ok=True)
        output_path = predictions_dir / f"{video_path.stem}.predictions.slp"

        video = Video.from_filename(str(video_path))
        item = VideoItemForInference(video=video, frames=None, use_absolute_path=True)
        task = InferenceTask(
            trained_job_paths=self._model_paths,
            inference_params={
                "_batch_size": self._batch_size,
                "_max_instances": self._max_instances,
            },
        )
        cli_args, _ = task.make_predict_cli_call(
            item,
            output_path=str(output_path),
            gui=True,
        )

        self.logOutput.emit(f"Video: {video_path.name}")
        self.logOutput.emit(f"Output: {output_path}")
        self.logOutput.emit(f"Running: {' '.join(cli_args)}")

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        with subprocess.Popen(
            cli_args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        ) as proc:
            self._current_process = proc
            while proc.poll() is None:
                self._raise_if_canceled()
                line = proc.stdout.readline().rstrip()
                self._handle_cli_line(line)
                time.sleep(0.02)
            self._current_process = None

            if proc.returncode != 0:
                raise RuntimeError(
                    f"Inference failed for {video_path.name} "
                    f"with code {proc.returncode}."
                )

        return str(output_path)

    def _export_prediction_formats(self, prediction_path: str) -> int:
        from sleap.gui.prediction_exports import export_prediction_file_formats

        self._raise_if_canceled()
        self.statusUpdate.emit(
            f"<b>Exporting predictions</b><br>{Path(prediction_path).name}"
        )
        written = export_prediction_file_formats(
            prediction_path,
            analysis_h5=self._export_analysis_h5,
            nwb=self._export_nwb,
        )
        for path in written:
            self.logOutput.emit(f"Exported: {path}")
        return len(written)

    def _run_projection(
        self, session_dir: Path, prediction_paths: Sequence[str]
    ) -> Dict[str, Any]:
        from sleap.gui.reach_projection import run_3d_projection_export

        def progress(stage: str, current: int, total: int, detail: str) -> None:
            self._raise_if_canceled()
            detail = detail or ""
            self.statusUpdate.emit(f"<b>{stage}</b><br>{detail}")
            if detail:
                self.logOutput.emit(f"{stage}: {detail}")

        metadata = run_3d_projection_export(
            prediction_files=prediction_paths,
            calibration_path=self._calibration_path,
            output_dir=session_dir,
            points3d_filename="points3d.h5",
            reprojections_filename="reprojections.h5",
            h5_compression=None,
            progress_callback=progress,
            cancel_check=lambda: self._canceled,
        )
        self.logOutput.emit(
            "Wrote: "
            f"{Path(metadata['points3d_path']).name}, "
            f"{Path(metadata['reprojections_path']).name}"
        )
        return metadata

    def _run_reach_detection(
        self,
        session_dir: Path,
        *,
        points3d_path: Optional[str] = None,
        prediction_path: Optional[str] = None,
        source_video_path: Optional[Path] = None,
    ) -> int:
        import numpy as np
        import sleap_io
        from sleap.gui.reach_detection import (
            detect_reaches_absolute,
            detect_reaches_kpn,
            extract_hand_position_3d_with_confidence,
            save_kpn_reach_details,
            save_kpn_reach_details_csv,
            save_kpn_reach_details_table,
            save_pellet_history,
            save_reach_detection_info,
            save_reaches,
            suggest_pellet_nodes,
        )
        from sleap.gui.reach_projection import (
            extract_points3d_position,
            extract_reprojection_confidence,
            load_points3d_h5,
            load_reprojections_h5,
        )

        settings = self._reach_settings
        lh_nodes = list(settings.get("left_hand_nodes", []))
        rh_nodes = list(settings.get("right_hand_nodes", []))
        detection_method = str(settings.get("method", "from_pellet"))
        if not rh_nodes:
            raise ValueError("no right-hand nodes are selected in the Reaches dock")
        if detection_method == "from_pellet" and not lh_nodes:
            raise ValueError("no left-hand nodes are selected in the Reaches dock")

        source = {}
        reprojection_source = None
        source_video = None
        source_video_filename = ""
        coordinate_system = "3d_calibration_mm"
        if points3d_path is not None:
            if source_video_path is not None:
                source_video = Video.from_filename(str(source_video_path))
                source_video_filename = str(source_video_path)
            source = load_points3d_h5(points3d_path)
            node_names = list(source.get("node_names", []))
            points3d = source.get("points3d")
            if points3d is None:
                raise ValueError("points3D file did not contain trajectory data")

            reprojections_path = session_dir / "reprojections.h5"
            point_scores = None
            if reprojections_path.is_file():
                reprojection_source = {"path": str(reprojections_path)}
                try:
                    reprojections = load_reprojections_h5(reprojections_path)
                    point_scores = reprojections.get("point_scores")
                except Exception as exc:
                    self.logOutput.emit(
                        f"Could not load reprojection confidence: {exc}"
                    )

            rh_traj = extract_points3d_position(points3d, node_names, rh_nodes)
            lh_traj = (
                extract_points3d_position(points3d, node_names, lh_nodes)
                if lh_nodes
                else np.full_like(rh_traj, np.nan, dtype=np.float64)
            )
            pellet_nodes = suggest_pellet_nodes(node_names)
            pellet_traj = (
                extract_points3d_position(points3d, node_names, pellet_nodes)
                if pellet_nodes
                else np.full_like(rh_traj, np.nan, dtype=np.float64)
            )
            rh_conf = extract_reprojection_confidence(point_scores, node_names, rh_nodes)
            lh_conf = (
                extract_reprojection_confidence(point_scores, node_names, lh_nodes)
                if lh_nodes
                else None
            )
            pellet_conf = (
                extract_reprojection_confidence(point_scores, node_names, pellet_nodes)
                if pellet_nodes
                else None
            )
            n_source_frames = int(points3d.shape[0])
            prediction_source = {
                "source_type": "points3d_file",
                "path": str(points3d_path),
                "coordinate_system": coordinate_system,
                "metadata": source.get("metadata", {}),
            }
        else:
            if prediction_path is None:
                raise ValueError("no 2D prediction file was provided")
            labels = sleap_io.load_slp(str(prediction_path))
            if not labels.videos:
                raise ValueError(f"No videos found in {prediction_path}.")
            source_video = self._matching_prediction_video(labels, source_video_path)
            source_video_filename = self._video_filename(source_video)
            try:
                node_names = list(labels.skeletons[0].node_names)
            except (AttributeError, IndexError):
                raise ValueError("prediction file does not contain a skeleton")

            point_confidence = float(settings.get("point_confidence_threshold", 0.5))
            pellet_nodes = suggest_pellet_nodes(node_names)
            rh_traj, rh_conf = extract_hand_position_3d_with_confidence(
                labels,
                source_video,
                rh_nodes,
                min_confidence=point_confidence,
            )
            lh_traj = np.full_like(rh_traj, np.nan, dtype=np.float64)
            lh_conf = None
            if lh_nodes:
                lh_traj, lh_conf = extract_hand_position_3d_with_confidence(
                    labels,
                    source_video,
                    lh_nodes,
                    min_confidence=point_confidence,
                )
            pellet_traj = np.full_like(rh_traj, np.nan, dtype=np.float64)
            pellet_conf = None
            if pellet_nodes:
                pellet_traj, pellet_conf = extract_hand_position_3d_with_confidence(
                    labels,
                    source_video,
                    pellet_nodes,
                    min_confidence=point_confidence,
                )
            n_source_frames = int(len(rh_traj))
            coordinate_system = "2d_pixels"
            prediction_source = {
                "source_type": "external_predictions_file",
                "path": str(prediction_path),
                "coordinate_system": coordinate_system,
                "camera": str(self._reach_camera),
                "source_video": str(source_video_path or source_video_filename),
            }

        missing_nodes = sorted(
            set(lh_nodes + rh_nodes).difference(set(node_names))
        )
        if missing_nodes:
            raise ValueError(
                "selected hand node(s) not found in the reach source: "
                + ", ".join(missing_nodes)
            )

        pellet_nodes = suggest_pellet_nodes(node_names)
        if detection_method == "from_pellet" and not pellet_nodes:
            raise ValueError("no PELLET node found in the reach source")

        frame_rate = (
            self._video_frame_rate(source_video)
            if source_video is not None
            else 30.0
        )
        point_confidence = float(settings.get("point_confidence_threshold", 0.5))
        lh_traj, rh_traj, filter_info = self._filter_hand_trajectories(
            lh_traj,
            rh_traj,
            frame_rate,
            enabled=bool(settings.get("filter_hand_traces", False)),
            cutoff=float(settings.get("filter_cutoff_frequency_hz", 30.0)),
        )

        lh_valid = self._valid_trajectory_frames(lh_traj, lh_conf, point_confidence)
        rh_valid = self._valid_trajectory_frames(rh_traj, rh_conf, point_confidence)
        pellet_valid = self._valid_trajectory_frames(
            pellet_traj, pellet_conf, point_confidence
        )
        if rh_valid == 0 or (
            detection_method == "from_pellet"
            and (lh_valid == 0 or pellet_valid == 0)
        ):
            raise ValueError(
                "missing valid R_HAND, L_HAND, or PELLET points for reach detection"
            )

        detection_parameters = {
            "method": detection_method,
            "kpn_outward_threshold": float(
                settings.get("kpn_outward_threshold", -10.0)
            ),
            "kpn_outward_max_threshold": float(
                settings.get("kpn_outward_max_threshold", -7.0)
            ),
            "kpn_peak_prominence": float(settings.get("kpn_peak_prominence", 1.0)),
            "kpn_min_outward_travel": float(
                settings.get("kpn_min_outward_travel", 4.0)
            ),
            "kpn_start_padding": int(settings.get("kpn_start_padding", 5)),
            "min_frame": int(settings.get("min_frame", 15)),
            "max_frame": int(settings.get("max_frame", 200)),
            "frame_rate": float(frame_rate),
            "max_dist_from_home": float(settings.get("max_dist_from_home", 15.0)),
            "point_confidence_threshold": float(point_confidence),
            "hand_confidence_aggregation": "median",
            "filter": filter_info,
        }

        if detection_method == "absolute":
            axis_idx = int(settings.get("absolute_axis", 0))
            axis_name = str(settings.get("absolute_axis_name", "x"))
            absolute_signal = np.asarray(rh_traj[:, axis_idx], dtype=np.float64)
            if bool(settings.get("absolute_invert", False)):
                absolute_signal = -absolute_signal
            detection_parameters.update(
                {
                    "absolute_axis": axis_name,
                    "absolute_invert": bool(settings.get("absolute_invert", False)),
                    "absolute_max_start_value": float(
                        settings.get("absolute_max_start_value", 300.0)
                    ),
                }
            )
            reaches, details, pellet_history = detect_reaches_absolute(
                absolute_signal,
                right_hand=rh_traj,
                left_hand=lh_traj,
                pellet=pellet_traj if pellet_nodes else None,
                threshold=detection_parameters["kpn_outward_threshold"],
                peak_prominence=detection_parameters["kpn_peak_prominence"],
                start_padding=detection_parameters["kpn_start_padding"],
                min_frame=detection_parameters["min_frame"],
                max_frame=detection_parameters["max_frame"],
                max_start_value=detection_parameters["absolute_max_start_value"],
                frame_rate=frame_rate,
                signal_confidence=rh_conf,
                confidence=point_confidence,
                right_hand_confidence=rh_conf,
                left_hand_confidence=lh_conf,
                pellet_confidence=pellet_conf,
                max_dist_from_home=detection_parameters["max_dist_from_home"],
                return_pellet_history=True,
                return_details=True,
            )
            for detail in details:
                detail["absolute_axis"] = axis_name
                detail["absolute_invert"] = bool(settings.get("absolute_invert", False))
        else:
            reaches, details, pellet_history = detect_reaches_kpn(
                right_hand=rh_traj,
                left_hand=lh_traj,
                pellet=pellet_traj,
                frame_rate=frame_rate,
                min_threshold=detection_parameters["kpn_outward_threshold"],
                max_threshold=detection_parameters["kpn_outward_max_threshold"],
                peak_prominence=detection_parameters["kpn_peak_prominence"],
                min_outward_travel=detection_parameters["kpn_min_outward_travel"],
                start_padding=detection_parameters["kpn_start_padding"],
                min_frame=detection_parameters["min_frame"],
                max_frame=detection_parameters["max_frame"],
                max_dist_from_home=detection_parameters["max_dist_from_home"],
                confidence=point_confidence,
                right_hand_confidence=rh_conf,
                left_hand_confidence=lh_conf,
                pellet_confidence=pellet_conf,
                return_details=True,
            )

        detection_info = {
            "prediction_source": prediction_source,
            "reprojection_source": reprojection_source,
            "video": {
                "filename": source_video_filename,
                "frames": n_source_frames,
            },
            "nodes": {
                "left_hand": list(lh_nodes),
                "right_hand": list(rh_nodes),
                "pellet": list(pellet_nodes),
            },
            "valid_frame_counts": {
                "left_hand": int(lh_valid),
                "right_hand": int(rh_valid),
                "pellet": int(pellet_valid),
            },
            "detection_parameters": detection_parameters,
            "session_events": {
                "count": 0,
                "used_for_pellet_availability": False,
            },
            "outputs": {
                "reach_count": int(len(reaches)),
                "pellet_epoch_count": int(
                    len(pellet_history) - 1 if pellet_history else 0
                ),
                "detail_count": int(len(details)),
            },
        }

        save_reaches(reaches, session_dir / "detected_reaches.txt")
        if details:
            save_kpn_reach_details(details, session_dir / "kpn_reach_details.json")
            save_kpn_reach_details_table(
                details, session_dir / "kpn_reach_details.txt"
            )
            save_kpn_reach_details_csv(details, session_dir / "kpn_reach_details.csv")
        if pellet_history:
            save_pellet_history(pellet_history, session_dir / "pelletHistory.pickle")
        save_reach_detection_info(
            detection_info, session_dir / "reach_detection_info.json"
        )
        self.logOutput.emit(
            f"Wrote reaches: {session_dir / 'detected_reaches.txt'} "
            f"({len(reaches)} reach(es))"
        )
        return int(len(reaches))

    @staticmethod
    def _valid_trajectory_frames(traj, confidence, threshold: float) -> int:
        import numpy as np

        if traj is None:
            return 0
        arr = np.asarray(traj, dtype=np.float64)
        if arr.ndim != 2 or arr.size == 0:
            return 0
        ok = np.all(np.isfinite(arr[:, :3]), axis=1)
        if confidence is not None:
            conf = np.asarray(confidence, dtype=np.float64).reshape(-1)
            conf_ok = np.zeros(arr.shape[0], dtype=bool)
            take = min(arr.shape[0], conf.size)
            if take > 0:
                conf_ok[:take] = np.isfinite(conf[:take]) & (conf[:take] >= threshold)
            ok &= conf_ok
        return int(np.sum(ok))

    @staticmethod
    def _filter_hand_trajectories(
        lh_traj,
        rh_traj,
        frame_rate: float,
        *,
        enabled: bool,
        cutoff: float,
    ):
        info = {
            "enabled": bool(enabled),
            "cutoff_frequency_hz": float(cutoff),
            "sampling_rate_hz": float(frame_rate),
            "order": 1,
            "normalization": "cutoff_frequency_hz / sampling_rate_hz",
            "filtered_traces": [],
        }
        if not enabled:
            return lh_traj, rh_traj, info
        if frame_rate <= 0:
            raise ValueError(
                "Cannot filter hand traces without a positive sampling rate."
            )
        if cutoff <= 0 or cutoff >= frame_rate:
            raise ValueError(
                "Butterworth cutoff frequency must be greater than 0 and less "
                "than the sampling rate."
            )
        lh_filtered = BatchAnalysisWorker._butterworth_filter_trace(
            lh_traj, cutoff, frame_rate
        )
        rh_filtered = BatchAnalysisWorker._butterworth_filter_trace(
            rh_traj, cutoff, frame_rate
        )
        info["filtered_traces"] = ["left_hand", "right_hand"]
        return lh_filtered, rh_filtered, info

    @staticmethod
    def _butterworth_filter_trace(trace, cutoff_frequency: float, sampling_rate: float):
        import numpy as np
        from scipy.signal import butter, filtfilt

        arr = np.asarray(trace, dtype=np.float64)
        if arr.ndim != 2 or arr.size == 0:
            return arr.copy()

        filtered = arr.copy()
        normalized_cutoff = float(cutoff_frequency) / float(sampling_rate)
        b, a = butter(1, normalized_cutoff)
        min_samples = 3 * max(len(a), len(b)) + 1

        for dim in range(arr.shape[1]):
            values = arr[:, dim]
            valid = np.isfinite(values)
            if int(np.sum(valid)) < min_samples:
                continue
            idx = np.arange(values.size)
            interpolated = np.interp(idx, idx[valid], values[valid])
            try:
                smoothed = filtfilt(b, a, interpolated)
            except ValueError:
                continue
            smoothed[~valid] = np.nan
            filtered[:, dim] = smoothed
        return filtered

    def _prediction_for_reach_camera(
        self, prediction_items: Sequence[Tuple[Path, str]]
    ) -> Optional[Tuple[Path, str]]:
        for video_path, prediction_path in prediction_items:
            if self._video_matches_reach_camera(video_path):
                return video_path, prediction_path
        return None

    def _video_matches_reach_camera(self, video_path: Path) -> bool:
        selected = str(self._reach_camera or "")
        if not selected:
            return False
        return selected == self._camera_key(video_path)

    @staticmethod
    def _camera_key(video_path: str | Path) -> str:
        stem = Path(video_path).stem
        token = BatchAnalysisWorker._camera_token(stem)
        return token or stem

    @staticmethod
    def _camera_token(value: str) -> str:
        import re

        match = re.search(r"cam\s*0*(\d+)", str(value or ""), flags=re.IGNORECASE)
        if not match:
            return ""
        return f"Cam{int(match.group(1)):03d}"

    @staticmethod
    def _matching_prediction_video(labels, source_video_path: Optional[Path]):
        if source_video_path is None:
            return labels.videos[0]
        source_key = BatchAnalysisWorker._camera_key(source_video_path)
        source_stem = Path(source_video_path).stem
        for video in labels.videos:
            filename = BatchAnalysisWorker._video_filename(video)
            if BatchAnalysisWorker._camera_key(filename) == source_key:
                return video
            if Path(filename).stem == source_stem:
                return video
        return labels.videos[0]

    @staticmethod
    def _video_filename(video) -> str:
        filename = getattr(video, "filename", "")
        if isinstance(filename, list):
            filename = filename[0] if filename else ""
        return str(filename or "")

    @staticmethod
    def _video_frame_rate(video) -> float:
        for obj in (video, getattr(video, "backend", None)):
            if obj is None:
                continue
            for attr in ("frame_rate", "fps"):
                value = getattr(obj, attr, None)
                if value is None:
                    continue
                try:
                    value = value() if callable(value) else value
                    if float(value) > 0:
                        return float(value)
                except (TypeError, ValueError):
                    pass
        return 30.0

    def _handle_cli_line(self, line: str) -> None:
        if not line:
            return
        if line.startswith("{"):
            try:
                data = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                data = None
            if data is not None:
                n_processed = data.get("n_processed")
                n_total = data.get("n_total")
                rate = data.get("rate")
                if n_processed is not None and n_total is not None:
                    msg = f"Predicted: <b>{n_processed:,}/{n_total:,}</b>"
                    if rate is not None:
                        msg += f" &nbsp; FPS: <b>{rate:.1f}</b>"
                    self.statusUpdate.emit(msg)
                return
        self.logOutput.emit(line)

    def _raise_if_canceled(self) -> None:
        if self._canceled:
            raise InterruptedError("Batch analysis canceled.")


class BatchAnalysisDock(DockWidget):
    """Dock for batch inference over nested project session folders."""

    def __init__(
        self,
        main_window: QMainWindow,
        tab_with: Optional[QLayout] = None,
    ):
        super().__init__(
            name="Batch Analysis",
            main_window=main_window,
            model_type=None,
            tab_with=tab_with,
        )

    def create_models(self):
        return None

    def create_tables(self):
        return None

    def lay_everything_out(self) -> None:
        content = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        layout.addWidget(self._build_inputs_group())
        layout.addWidget(self._build_options_group())
        layout.addWidget(self._build_sessions_group())
        layout.addSpacing(4)

        self._run_btn = QPushButton("Run Batch Analysis")
        self._run_btn.setEnabled(False)
        self._run_btn.clicked.connect(self._run_batch_analysis)
        layout.addWidget(self._run_btn)

        self._status_label = QLabel("")
        self._status_label.setWordWrap(True)
        self._status_label.setStyleSheet(
            "color: #9ca3af; font-size: 10px; padding: 2px 0;"
        )
        layout.addWidget(self._status_label)
        layout.addStretch()

        content.setLayout(layout)

        scroll = QScrollArea()
        scroll.setWidget(content)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.wgt_layout.setContentsMargins(0, 0, 0, 0)
        self.wgt_layout.addWidget(scroll)

    def _build_inputs_group(self) -> QGroupBox:
        gb = QGroupBox("Inputs")
        layout = QVBoxLayout()
        layout.setSpacing(4)

        model_row = QHBoxLayout()
        self._model_path_edit = QLineEdit()
        self._model_path_edit.setPlaceholderText("Path to trained model directory...")
        self._model_path_edit.setToolTip(
            "For top-down inference, select the centroid model here."
        )
        self._model_path_edit.textChanged.connect(self._update_run_btn)
        model_btn = QPushButton("Browse")
        model_btn.setFixedWidth(70)
        model_btn.clicked.connect(self._browse_model)
        model_row.addWidget(self._model_path_edit)
        model_row.addWidget(model_btn)
        layout.addLayout(model_row)

        top_down_model_row = QHBoxLayout()
        self._centered_model_path_edit = QLineEdit()
        self._centered_model_path_edit.setPlaceholderText(
            "Optional centered-instance model directory..."
        )
        self._centered_model_path_edit.setToolTip(
            "For top-down inference, select the centered-instance model here."
        )
        self._centered_model_path_edit.textChanged.connect(self._update_run_btn)
        centered_model_btn = QPushButton("Browse")
        centered_model_btn.setFixedWidth(70)
        centered_model_btn.clicked.connect(self._browse_centered_model)
        top_down_model_row.addWidget(self._centered_model_path_edit)
        top_down_model_row.addWidget(centered_model_btn)
        layout.addLayout(top_down_model_row)

        calibration_row = QHBoxLayout()
        self._calibration_path_edit = QLineEdit()
        self._calibration_path_edit.setPlaceholderText("Optional calibration.toml...")
        self._calibration_path_edit.textChanged.connect(self._update_run_btn)
        calibration_btn = QPushButton("Browse")
        calibration_btn.setFixedWidth(70)
        calibration_btn.clicked.connect(self._browse_calibration)
        clear_btn = QPushButton("Clear")
        clear_btn.setFixedWidth(58)
        clear_btn.clicked.connect(lambda: self._calibration_path_edit.clear())
        calibration_row.addWidget(self._calibration_path_edit)
        calibration_row.addWidget(calibration_btn)
        calibration_row.addWidget(clear_btn)
        layout.addLayout(calibration_row)

        parent_row = QHBoxLayout()
        self._parent_path_edit = QLineEdit()
        self._parent_path_edit.setPlaceholderText(
            "Parent directory containing sessions..."
        )
        self._parent_path_edit.textChanged.connect(self._on_parent_changed)
        parent_btn = QPushButton("Browse")
        parent_btn.setFixedWidth(70)
        parent_btn.clicked.connect(self._browse_parent)
        parent_row.addWidget(self._parent_path_edit)
        parent_row.addWidget(parent_btn)
        layout.addLayout(parent_row)

        gb.setLayout(layout)
        return gb

    def _build_options_group(self) -> QGroupBox:
        gb = QGroupBox("Options")
        form = QFormLayout()
        form.setSpacing(4)

        self._batch_spin = QSpinBox()
        self._batch_spin.setRange(1, 128)
        self._batch_spin.setValue(4)
        form.addRow("Batch size:", self._batch_spin)

        self._max_inst_spin = QSpinBox()
        self._max_inst_spin.setRange(1, 200)
        self._max_inst_spin.setValue(10)
        form.addRow("Max instances:", self._max_inst_spin)

        self._export_analysis_h5_check = QCheckBox("Analysis HDF5")
        self._export_analysis_h5_check.setToolTip(
            "After each prediction file is saved, export it to SLEAP Analysis HDF5."
        )
        form.addRow("Also export:", self._export_analysis_h5_check)

        self._export_nwb_check = QCheckBox("NWB")
        self._export_nwb_check.setToolTip(
            "After each prediction file is saved, export it to Neurodata Without Borders."
        )
        form.addRow("", self._export_nwb_check)

        self._detect_reaches_check = QCheckBox("Detect reaches")
        self._detect_reaches_check.setToolTip(
            "After each session is processed, detect reaches with the current "
            "settings from the Reaches dock."
        )
        self._detect_reaches_check.toggled.connect(self._update_reach_controls)
        form.addRow("Also run:", self._detect_reaches_check)

        self._reach_source_combo = QComboBox()
        self._reach_source_combo.addItem("3D points", "3d")
        self._reach_source_combo.addItem("2D camera", "2d")
        self._reach_source_combo.setToolTip(
            "Use 3D projected points or one selected 2D camera prediction."
        )
        self._reach_source_combo.currentIndexChanged.connect(
            self._update_reach_controls
        )
        form.addRow("Reach data:", self._reach_source_combo)

        self._reach_camera_combo = QComboBox()
        self._reach_camera_combo.setToolTip(
            "Camera to use when detecting reaches from 2D predictions."
        )
        form.addRow("2D camera:", self._reach_camera_combo)
        self._update_reach_controls()

        gb.setLayout(form)
        return gb

    def _build_sessions_group(self) -> QGroupBox:
        gb = QGroupBox("Detected Sessions")
        layout = QVBoxLayout()
        layout.setSpacing(4)

        suffix_row = QHBoxLayout()
        suffix_row.addWidget(QLabel("Suffix:"))
        self._video_suffix_edit = QLineEdit()
        self._video_suffix_edit.setPlaceholderText("Video name suffix, e.g. _synced")
        self._video_suffix_edit.setToolTip(
            "Match the end of each video filename before its extension. Matching "
            "is case-insensitive."
        )
        keep_suffix_btn = QPushButton("Keep Matching")
        keep_suffix_btn.setToolTip(
            "Include videos ending with this suffix and exclude all other videos."
        )
        keep_suffix_btn.clicked.connect(
            lambda: self._apply_video_suffix(keep_matches=True)
        )
        exclude_suffix_btn = QPushButton("Exclude Matching")
        exclude_suffix_btn.setToolTip(
            "Exclude videos ending with this suffix without changing other videos."
        )
        exclude_suffix_btn.clicked.connect(
            lambda: self._apply_video_suffix(keep_matches=False)
        )
        self._video_suffix_edit.returnPressed.connect(
            lambda: self._apply_video_suffix(keep_matches=True)
        )
        suffix_row.addWidget(self._video_suffix_edit)
        suffix_row.addWidget(keep_suffix_btn)
        suffix_row.addWidget(exclude_suffix_btn)
        layout.addLayout(suffix_row)

        self._session_list = QTreeWidget()
        self._session_list.setHeaderHidden(True)
        self._session_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self._session_list.setAlternatingRowColors(True)
        self._session_list.itemChanged.connect(self._on_video_selection_changed)
        layout.addWidget(self._session_list)

        selection_row = QHBoxLayout()
        include_all_btn = QPushButton("Include All Views")
        include_all_btn.clicked.connect(lambda: self._set_all_videos_checked(True))
        exclude_selected_btn = QPushButton("Exclude Selected Views")
        exclude_selected_btn.setToolTip(
            "Uncheck the selected video rows. Selecting a session excludes all of "
            "its video views."
        )
        exclude_selected_btn.clicked.connect(self._exclude_selected_videos)
        selection_row.addWidget(include_all_btn)
        selection_row.addWidget(exclude_selected_btn)
        layout.addLayout(selection_row)

        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self._refresh_sessions)
        layout.addWidget(refresh_btn)

        gb.setLayout(layout)
        return gb

    def _browse_model(self) -> None:
        path = FileDialog.openDir(self, caption="Select Trained Model Directory")
        if path:
            self._model_path_edit.setText(path)

    def _browse_centered_model(self) -> None:
        path = FileDialog.openDir(
            self,
            caption="Select Centered-Instance Model Directory",
        )
        if path:
            self._centered_model_path_edit.setText(path)

    def _selected_model_paths(self) -> List[str]:
        model_paths = [self._model_path_edit.text().strip()]
        centered_model_path = self._centered_model_path_edit.text().strip()
        if centered_model_path:
            model_paths.append(centered_model_path)
        return model_paths

    def _browse_calibration(self) -> None:
        path, _ = FileDialog.open(
            self,
            caption="Select calibration.toml",
            filter="TOML files (*.toml);;All Files (*)",
        )
        if path:
            self._calibration_path_edit.setText(path)

    def _browse_parent(self) -> None:
        path = FileDialog.openDir(self, caption="Select Parent Directory")
        if path:
            self._parent_path_edit.setText(path)

    def _on_parent_changed(self, _: str) -> None:
        self._refresh_sessions()
        self._update_run_btn()

    def _refresh_sessions(self) -> None:
        if not hasattr(self, "_session_list"):
            return
        self._session_list.blockSignals(True)
        self._session_list.clear()
        self._session_list.blockSignals(False)
        parent_dir = self._parent_path_edit.text().strip()
        if not parent_dir:
            self._refresh_reach_camera_options([])
            self._set_status("")
            return

        parent = Path(parent_dir)
        if not parent.is_dir():
            self._refresh_reach_camera_options([])
            self._set_status("Parent directory not found.", error=True)
            return

        sessions = self.find_session_dirs(parent)
        self._session_list.blockSignals(True)
        for session_dir, videos in sessions:
            rel = self._relative_display(session_dir, parent)
            session_item = QTreeWidgetItem(
                self._session_list, [f"{rel}  ({len(videos)} video(s))"]
            )
            session_item.setData(0, Qt.UserRole, str(session_dir))
            session_item.setToolTip(0, str(session_dir))
            session_item.setExpanded(True)
            for video_path in videos:
                video_item = QTreeWidgetItem(session_item, [video_path.name])
                video_item.setFlags(video_item.flags() | Qt.ItemIsUserCheckable)
                video_item.setCheckState(0, Qt.Checked)
                video_item.setData(0, Qt.UserRole, str(video_path))
                video_item.setToolTip(0, str(video_path))
        self._session_list.blockSignals(False)

        if sessions:
            self._on_video_selection_changed()
        else:
            self._refresh_reach_camera_options([])
            self._set_status("No session folders with supported videos found.")

    def _selected_sessions(self) -> List[Tuple[Path, List[Path]]]:
        """Return session folders and video views currently checked in the tree."""
        if not hasattr(self, "_session_list"):
            return []

        sessions = []
        for session_idx in range(self._session_list.topLevelItemCount()):
            session_item = self._session_list.topLevelItem(session_idx)
            videos = []
            for video_idx in range(session_item.childCount()):
                video_item = session_item.child(video_idx)
                if video_item.checkState(0) == Qt.Checked:
                    videos.append(Path(video_item.data(0, Qt.UserRole)))
            if videos:
                sessions.append(
                    (Path(session_item.data(0, Qt.UserRole)), videos)
                )
        return sessions

    def _on_video_selection_changed(self, *_) -> None:
        sessions = self._selected_sessions()
        self._refresh_reach_camera_options(sessions)
        selected_count = sum(len(videos) for _, videos in sessions)
        detected_count = sum(
            self._session_list.topLevelItem(idx).childCount()
            for idx in range(self._session_list.topLevelItemCount())
        )
        self._set_status(
            f"Selected {selected_count} of {detected_count} video(s) across "
            f"{len(sessions)} session folder(s)."
        )
        self._update_run_btn()

    def _set_all_videos_checked(self, checked: bool) -> None:
        state = Qt.Checked if checked else Qt.Unchecked
        self._session_list.blockSignals(True)
        for session_idx in range(self._session_list.topLevelItemCount()):
            session_item = self._session_list.topLevelItem(session_idx)
            for video_idx in range(session_item.childCount()):
                session_item.child(video_idx).setCheckState(0, state)
        self._session_list.blockSignals(False)
        self._on_video_selection_changed()

    def _exclude_selected_videos(self) -> None:
        self._session_list.blockSignals(True)
        for item in self._session_list.selectedItems():
            if item.parent() is None:
                for video_idx in range(item.childCount()):
                    item.child(video_idx).setCheckState(0, Qt.Unchecked)
            else:
                item.setCheckState(0, Qt.Unchecked)
        self._session_list.blockSignals(False)
        self._on_video_selection_changed()

    def _apply_video_suffix(self, *, keep_matches: bool) -> None:
        suffix = self._video_suffix_edit.text().strip()
        if not suffix:
            self._set_status("Enter a video name suffix first.", error=True)
            return

        self._session_list.blockSignals(True)
        for session_idx in range(self._session_list.topLevelItemCount()):
            session_item = self._session_list.topLevelItem(session_idx)
            for video_idx in range(session_item.childCount()):
                video_item = session_item.child(video_idx)
                matches = self.video_name_has_suffix(
                    video_item.data(0, Qt.UserRole), suffix
                )
                if keep_matches:
                    video_item.setCheckState(
                        0, Qt.Checked if matches else Qt.Unchecked
                    )
                elif matches:
                    video_item.setCheckState(0, Qt.Unchecked)
        self._session_list.blockSignals(False)
        self._on_video_selection_changed()

    def _refresh_reach_camera_options(
        self, sessions: Sequence[Tuple[Path, List[Path]]]
    ) -> None:
        if not hasattr(self, "_reach_camera_combo"):
            return

        current = self._reach_camera_combo.currentData()
        options: Dict[str, str] = {}
        for _, videos in sessions:
            for video_path in videos:
                key = BatchAnalysisWorker._camera_key(video_path)
                label = key if key.startswith("Cam") else video_path.stem
                options.setdefault(key, label)

        self._reach_camera_combo.blockSignals(True)
        self._reach_camera_combo.clear()
        for key in sorted(options, key=lambda value: value.casefold()):
            self._reach_camera_combo.addItem(options[key], key)
        if current:
            idx = self._reach_camera_combo.findData(current)
            if idx >= 0:
                self._reach_camera_combo.setCurrentIndex(idx)
        self._reach_camera_combo.blockSignals(False)
        self._update_reach_controls()

    def _update_reach_controls(self, *_) -> None:
        if not hasattr(self, "_reach_source_combo"):
            return
        detect_reaches = self._detect_reaches_check.isChecked()
        use_2d = self._reach_source_combo.currentData() == "2d"
        self._reach_source_combo.setEnabled(detect_reaches)
        self._reach_camera_combo.setEnabled(detect_reaches and use_2d)

    def _update_run_btn(self) -> None:
        if not hasattr(self, "_run_btn"):
            return
        has_model = bool(self._model_path_edit.text().strip())
        has_parent = Path(self._parent_path_edit.text().strip()).is_dir()
        has_videos = bool(self._selected_sessions())
        self._run_btn.setEnabled(has_model and has_parent and has_videos)

    def _run_batch_analysis(self) -> None:
        model_paths = self._selected_model_paths()
        calibration_path = self._calibration_path_edit.text().strip()
        parent_dir = self._parent_path_edit.text().strip()

        missing_model_paths = [
            model_path for model_path in model_paths if not Path(model_path).exists()
        ]
        if missing_model_paths:
            QMessageBox.warning(
                self,
                "Model Not Found",
                "Please select valid trained model directories.",
            )
            return
        if calibration_path:
            cal = Path(calibration_path)
            if not cal.is_file() or cal.suffix.lower() != ".toml":
                QMessageBox.warning(
                    self,
                    "Calibration Not Found",
                    "Please select a valid calibration.toml file, or clear the field.",
                )
                return
        sessions = self._selected_sessions()
        if not sessions:
            QMessageBox.warning(
                self,
                "No Videos Selected",
                "Please include at least one detected video before running batch "
                "analysis.",
            )
            return

        detect_reaches = self._detect_reaches_check.isChecked()
        reach_source = str(self._reach_source_combo.currentData() or "3d")
        reach_camera = ""
        reach_settings = None
        if detect_reaches:
            if reach_source == "3d" and not calibration_path:
                QMessageBox.warning(
                    self,
                    "Calibration Required",
                    "3D batch reach detection uses each session's 3D projection, "
                    "so please select calibration.toml first.",
                )
                return
            if reach_source == "2d":
                reach_camera = str(self._reach_camera_combo.currentData() or "")
                if not reach_camera:
                    QMessageBox.warning(
                        self,
                        "Camera Required",
                        "Please choose the camera to use for 2D batch reach detection.",
                    )
                    return
            reaches_dock = getattr(self.main_window, "reaches_dock", None)
            if reaches_dock is None or not hasattr(
                reaches_dock, "batch_detection_settings"
            ):
                QMessageBox.warning(
                    self,
                    "Reaches Dock Not Available",
                    "Could not read reach detection settings from the Reaches dock.",
                )
                return
            reach_settings = reaches_dock.batch_detection_settings()
            if not reach_settings.get("right_hand_nodes"):
                QMessageBox.warning(
                    self,
                    "Reach Nodes Required",
                    "Please check at least one right-hand node in the Reaches dock.",
                )
                return
            if (
                reach_settings.get("method") == "from_pellet"
                and not reach_settings.get("left_hand_nodes")
            ):
                QMessageBox.warning(
                    self,
                    "Reach Nodes Required",
                    "Please check at least one left-hand node in the Reaches dock.",
                )
                return

        dialog = InferenceProgressDialog(self)
        dialog.setWindowTitle("Running Batch Analysis")
        dialog.setLabelText("<b>Starting batch analysis...</b>")
        n_session_steps = len(sessions) if calibration_path else 0
        n_reach_steps = len(sessions) if detect_reaches else 0
        dialog.setMaximum(
            max(
                1,
                sum(len(videos) for _, videos in sessions)
                + n_session_steps
                + n_reach_steps,
            )
        )

        worker = BatchAnalysisWorker(
            model_paths=model_paths,
            calibration_path=calibration_path,
            parent_dir=parent_dir,
            sessions=sessions,
            batch_size=self._batch_spin.value(),
            max_instances=self._max_inst_spin.value(),
            export_analysis_h5=self._export_analysis_h5_check.isChecked(),
            export_nwb=self._export_nwb_check.isChecked(),
            detect_reaches=detect_reaches,
            reach_source=reach_source,
            reach_camera=reach_camera,
            reach_settings=reach_settings,
            parent=self,
        )
        result: Dict = {"success": False, "summary": {}, "error": ""}

        def on_finished(success: bool, summary: dict, error: str) -> None:
            result["success"] = success
            result["summary"] = summary
            result["error"] = error
            dialog._ok_button.setEnabled(True)
            dialog._cancel_button.setEnabled(False)
            if success:
                skipped_reaches = len(summary.get("skipped_reaches", []))
                failed_sessions = len(summary.get("failed_sessions", []))
                skipped_note = (
                    f"<br>{skipped_reaches:,} session(s) skipped for reach detection."
                    if skipped_reaches
                    else ""
                )
                failed_note = (
                    f"<br>{failed_sessions:,} session(s) failed and were skipped."
                    if failed_sessions
                    else ""
                )
                dialog.setLabelText(
                    "<b>Batch analysis complete!</b><br><br>"
                    f"{summary.get('videos', 0):,} video(s), "
                    f"{summary.get('sessions', 0):,} session(s), "
                    f"{summary.get('exports', 0):,} export file(s), "
                    f"{summary.get('projections', 0):,} 3D export(s), "
                    f"{summary.get('reaches', 0):,} reach(es)."
                    f"{skipped_note}"
                    f"{failed_note}"
                )
            else:
                dialog.setLabelText(f"<b>Batch analysis failed.</b><br><br>{error}")

        def on_progress(current: int, total: int) -> None:
            dialog.setMaximum(total)
            dialog.setValue(current)

        worker.progressUpdate.connect(on_progress)
        worker.statusUpdate.connect(dialog.setLabelText)
        worker.logOutput.connect(dialog.appendLog)
        worker.finished.connect(on_finished)
        dialog._cancel_button.clicked.connect(worker.cancel)

        self._run_btn.setEnabled(False)
        self._set_status("Running batch analysis...")
        worker.start()
        dialog.exec_()

        if worker.isRunning():
            worker.cancel()
            worker.wait(5000)

        if result["success"]:
            summary = result["summary"]
            skipped_reaches = len(summary.get("skipped_reaches", []))
            failed_sessions = len(summary.get("failed_sessions", []))
            skipped_note = (
                f", {skipped_reaches:,} reach session(s) skipped"
                if skipped_reaches
                else ""
            )
            failed_note = (
                f", {failed_sessions:,} session(s) failed"
                if failed_sessions
                else ""
            )
            self._set_status(
                f"Done: {summary.get('videos', 0):,} video(s), "
                f"{summary.get('exports', 0):,} export file(s), "
                f"{summary.get('projections', 0):,} 3D export(s), "
                f"{summary.get('reaches', 0):,} reach(es)"
                f"{skipped_note}"
                f"{failed_note}."
            )
        else:
            self._set_status(
                f"Batch analysis failed: {result['error'] or 'canceled'}",
                error=True,
            )
        self._update_run_btn()

    @classmethod
    def find_session_dirs(cls, parent_dir: str | Path) -> List[Tuple[Path, List[Path]]]:
        """Return directories under ``parent_dir`` that directly contain videos."""
        parent = Path(parent_dir).expanduser()
        if not parent.is_dir():
            return []

        sessions = []
        for directory in sorted(
            [parent, *[p for p in parent.rglob("*") if p.is_dir()]],
            key=lambda p: str(p).casefold(),
        ):
            videos = cls.find_video_files(directory)
            if videos:
                sessions.append((directory, videos))
        return sessions

    @staticmethod
    def find_video_files(session_dir: str | Path) -> List[Path]:
        """Find source movie files directly inside a session folder."""
        session = Path(session_dir)
        videos = []
        for path in sorted(session.iterdir(), key=lambda p: p.name.casefold()):
            suffix = path.suffix.lower()
            if not path.is_file() or suffix not in MOVIE_EXTENSIONS:
                continue
            videos.append(path)
        return videos

    @staticmethod
    def video_name_has_suffix(video_path: str | Path, suffix: str) -> bool:
        """Return whether a video's stem ends with a suffix, ignoring case."""
        suffix = str(suffix or "").strip()
        if not suffix:
            return False
        return Path(video_path).stem.casefold().endswith(suffix.casefold())

    @staticmethod
    def _relative_display(path: Path, parent: Path) -> str:
        try:
            return str(path.relative_to(parent))
        except ValueError:
            return str(path)

    def _set_status(self, text: str, *, error: bool = False) -> None:
        color = "#ef4444" if error else "#9ca3af"
        self._status_label.setStyleSheet(
            f"color: {color}; font-size: 10px; padding: 2px 0;"
        )
        self._status_label.setText(text)
