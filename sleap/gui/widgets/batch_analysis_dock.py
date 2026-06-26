"""Dock widget for running model inference across many session folders."""

import json
import logging
import os
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from qtpy import QtCore
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLayout,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
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
        batch_size: int,
        max_instances: int,
        export_analysis_h5: bool = False,
        export_nwb: bool = False,
        parent=None,
    ):
        super().__init__(parent)
        self._model_paths = list(model_paths)
        self._calibration_path = calibration_path
        self._parent_dir = parent_dir
        self._batch_size = batch_size
        self._max_instances = max_instances
        self._export_analysis_h5 = export_analysis_h5
        self._export_nwb = export_nwb
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
            "skipped_sessions": [],
        }
        try:
            sessions = BatchAnalysisDock.find_session_dirs(self._parent_dir)
            if not sessions:
                raise ValueError("No session folders with supported videos were found.")

            total_steps = sum(len(videos) for _, videos in sessions)
            if self._calibration_path:
                total_steps += len(sessions)
            total_steps = max(total_steps, 1)
            current_step = 0

            self.logOutput.emit(f"Parent directory: {self._parent_dir}")
            for model_path in self._model_paths:
                self.logOutput.emit(f"Model: {model_path}")
            if self._calibration_path:
                self.logOutput.emit(f"Calibration: {self._calibration_path}")
            self.logOutput.emit(f"Found {len(sessions)} session folder(s).")

            for session_dir, video_paths in sessions:
                self._raise_if_canceled()
                summary["sessions"] += 1
                self.logOutput.emit("")
                self.logOutput.emit(f"Session: {session_dir}")
                prediction_paths = []

                for video_path in video_paths:
                    self._raise_if_canceled()
                    current_step += 1
                    self.progressUpdate.emit(current_step - 1, total_steps)
                    self.statusUpdate.emit(
                        f"<b>Running inference</b><br>{video_path.name}"
                    )
                    prediction_path = self._run_video_inference(session_dir, video_path)
                    prediction_paths.append(prediction_path)
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
                        self._run_projection(session_dir, prediction_paths)
                        summary["projections"] += 1
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
    ) -> None:
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

        gb.setLayout(form)
        return gb

    def _build_sessions_group(self) -> QGroupBox:
        gb = QGroupBox("Detected Sessions")
        layout = QVBoxLayout()
        layout.setSpacing(4)

        self._session_list = QListWidget()
        self._session_list.setSelectionMode(QAbstractItemView.NoSelection)
        self._session_list.setAlternatingRowColors(True)
        layout.addWidget(self._session_list)

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
        self._session_list.clear()
        parent_dir = self._parent_path_edit.text().strip()
        if not parent_dir:
            self._set_status("")
            return

        parent = Path(parent_dir)
        if not parent.is_dir():
            self._set_status("Parent directory not found.", error=True)
            return

        sessions = self.find_session_dirs(parent)
        for session_dir, videos in sessions:
            rel = self._relative_display(session_dir, parent)
            item = QListWidgetItem(f"{rel}  ({len(videos)} video(s))")
            item.setToolTip("\n".join(str(path) for path in videos))
            self._session_list.addItem(item)

        if sessions:
            n_videos = sum(len(videos) for _, videos in sessions)
            self._set_status(
                f"Found {len(sessions)} session folder(s), {n_videos} video(s)."
            )
        else:
            self._set_status("No session folders with supported videos found.")

    def _update_run_btn(self) -> None:
        if not hasattr(self, "_run_btn"):
            return
        has_model = bool(self._model_path_edit.text().strip())
        has_parent = Path(self._parent_path_edit.text().strip()).is_dir()
        self._run_btn.setEnabled(has_model and has_parent)

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
        sessions = self.find_session_dirs(parent_dir)
        if not sessions:
            QMessageBox.warning(
                self,
                "No Sessions Found",
                "No session folders with supported videos were found under "
                "the parent directory.",
            )
            return

        dialog = InferenceProgressDialog(self)
        dialog.setWindowTitle("Running Batch Analysis")
        dialog.setLabelText("<b>Starting batch analysis...</b>")
        dialog.setMaximum(max(1, sum(len(videos) for _, videos in sessions)))

        worker = BatchAnalysisWorker(
            model_paths=model_paths,
            calibration_path=calibration_path,
            parent_dir=parent_dir,
            batch_size=self._batch_spin.value(),
            max_instances=self._max_inst_spin.value(),
            export_analysis_h5=self._export_analysis_h5_check.isChecked(),
            export_nwb=self._export_nwb_check.isChecked(),
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
                dialog.setLabelText(
                    "<b>Batch analysis complete!</b><br><br>"
                    f"{summary.get('videos', 0):,} video(s), "
                    f"{summary.get('sessions', 0):,} session(s), "
                    f"{summary.get('exports', 0):,} export file(s), "
                    f"{summary.get('projections', 0):,} 3D export(s)."
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
            self._set_status(
                f"Done: {summary.get('videos', 0):,} video(s), "
                f"{summary.get('exports', 0):,} export file(s), "
                f"{summary.get('projections', 0):,} 3D export(s)."
            )
        else:
            self._set_status(
                f"Batch analysis failed: {result['error'] or 'canceled'}",
                error=True,
            )
        self._run_btn.setEnabled(True)

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
