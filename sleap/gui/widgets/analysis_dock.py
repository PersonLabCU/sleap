"""Dock widget for running inference on selected videos and auto-saving predictions."""

import logging
from pathlib import Path
import re
from typing import List, Optional

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
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QLayout,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from sleap.gui.commands import UpdateTopic
from sleap.gui.dialogs.filedialog import FileDialog
from sleap.gui.lazy_predictions import (
    ExternalPredictionSet,
    get_external_prediction_manager,
    video_key,
)
from sleap.gui.widgets.docks import DockWidget
from sleap_io import PredictedInstance, Video

logger = logging.getLogger(__name__)


class ProjectionWorker(QtCore.QThread):
    """Background worker for 3D projection export."""

    progressUpdate = QtCore.Signal(int, int)
    statusUpdate = QtCore.Signal(str)
    logOutput = QtCore.Signal(str)
    finished = QtCore.Signal(bool, dict, str)

    def __init__(
        self,
        prediction_files: List[str],
        calibration_path: str,
        output_dir: str,
        parent=None,
    ):
        super().__init__(parent)
        self._prediction_files = prediction_files
        self._calibration_path = calibration_path
        self._output_dir = output_dir
        self._canceled = False
        self._last_log = ""

    def cancel(self) -> None:
        self._canceled = True

    def run(self) -> None:
        from sleap.gui.reach_projection import run_3d_projection_export

        def progress(stage: str, current: int, total: int, detail: str) -> None:
            total = max(int(total), 1)
            current = max(0, min(int(current), total))
            detail = detail or ""
            self.progressUpdate.emit(current, total)
            self.statusUpdate.emit(f"<b>{stage}</b><br>{detail}")
            log_line = f"{stage}: {detail}" if detail else stage
            if log_line != self._last_log:
                self.logOutput.emit(log_line)
                self._last_log = log_line

        try:
            result = run_3d_projection_export(
                prediction_files=self._prediction_files,
                calibration_path=self._calibration_path,
                output_dir=self._output_dir,
                h5_compression=None,
                progress_callback=progress,
                cancel_check=lambda: self._canceled,
            )
        except InterruptedError as exc:
            self.finished.emit(False, {}, str(exc))
            return
        except Exception as exc:
            self.logOutput.emit(f"Error: {exc}")
            self.finished.emit(False, {}, str(exc))
            return

        self.finished.emit(True, result, "")


class ExternalPredictionLoadWorker(QtCore.QThread):
    """Background worker for compact external prediction preview loading."""

    statusUpdate = QtCore.Signal(str)
    resultReady = QtCore.Signal(bool, object, str)

    def __init__(self, prediction_files: List[str], parent=None):
        super().__init__(parent)
        self._prediction_files = prediction_files

    def run(self) -> None:
        prediction_sets = []
        try:
            for filename in self._prediction_files:
                self.statusUpdate.emit(f"Loading {Path(filename).name}...")
                prediction_sets.append(ExternalPredictionSet.from_file(filename))
        except Exception as exc:
            logger.exception("External prediction preview load failed")
            self.resultReady.emit(False, [], str(exc))
            return

        self.resultReady.emit(True, prediction_sets, "")


class AnalysisDock(DockWidget):
    """Dock for running inference on project videos and auto-saving predictions.

    Provides:
    - Trained model directory selector
    - Per-video checkboxes (refreshes when the project's video list changes)
    - Frame-range options (all / unlabeled / suggested)
    - Batch size and max-instance controls
    - One-click "Run Inference" that merges predictions and saves the project
    """

    def __init__(
        self,
        main_window: QMainWindow,
        tab_with: Optional[QLayout] = None,
    ):
        super().__init__(
            name="Analysis",
            main_window=main_window,
            model_type=None,
            tab_with=tab_with,
        )
        self._prediction_load_worker = None
        self._prediction_load_labels = None
        # Refresh the video list whenever the project or its videos change.
        main_window.state.connect("labels", self._on_project_labels_changed)
        main_window.state.connect("video", lambda _: self._refresh_videos())

    # ── DockWidget interface ─────────────────────────────────────────────── #

    def create_models(self):
        return None

    def create_tables(self):
        return None

    def lay_everything_out(self) -> None:
        content = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        layout.addWidget(self._build_model_group())
        layout.addWidget(self._build_videos_group())
        layout.addWidget(self._build_options_group())
        layout.addSpacing(4)

        self._run_btn = QPushButton("Run Inference")
        self._run_btn.setEnabled(False)
        self._run_btn.setToolTip(
            "Run inference on the selected videos and save predictions automatically."
        )
        self._run_btn.clicked.connect(self._run_inference)
        layout.addWidget(self._run_btn)

        self._status_label = QLabel("")
        self._status_label.setWordWrap(True)
        self._status_label.setStyleSheet(
            "color: #9ca3af; font-size: 10px; padding: 2px 0;"
        )
        layout.addWidget(self._status_label)
        layout.addWidget(self._build_prediction_preview_group())
        layout.addWidget(self._build_convert_predictions_group())
        layout.addWidget(self._build_projection_group())
        layout.addStretch()

        content.setLayout(layout)

        scroll = QScrollArea()
        scroll.setWidget(content)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.wgt_layout.setContentsMargins(0, 0, 0, 0)
        self.wgt_layout.addWidget(scroll)

        self._refresh_videos()
        self._discover_projection_predictions()

    # ── section builders ─────────────────────────────────────────────────── #

    def _build_model_group(self) -> QGroupBox:
        gb = QGroupBox("Model")
        layout = QVBoxLayout()
        layout.setSpacing(4)

        row = QHBoxLayout()
        self._model_path_edit = QLineEdit()
        self._model_path_edit.setPlaceholderText(
            "Path to trained model directory…"
        )
        self._model_path_edit.setToolTip(
            "For top-down inference, select the centroid model here."
        )
        self._model_path_edit.textChanged.connect(self._update_run_btn)
        browse_btn = QPushButton("Browse")
        browse_btn.setFixedWidth(70)
        browse_btn.clicked.connect(self._browse_model)
        row.addWidget(self._model_path_edit)
        row.addWidget(browse_btn)
        layout.addLayout(row)

        top_down_row = QHBoxLayout()
        self._centered_model_path_edit = QLineEdit()
        self._centered_model_path_edit.setPlaceholderText(
            "Optional centered-instance model directory..."
        )
        self._centered_model_path_edit.setToolTip(
            "For top-down inference, select the centered-instance model here."
        )
        self._centered_model_path_edit.textChanged.connect(self._update_run_btn)
        centered_browse_btn = QPushButton("Browse")
        centered_browse_btn.setFixedWidth(70)
        centered_browse_btn.clicked.connect(self._browse_centered_model)
        top_down_row.addWidget(self._centered_model_path_edit)
        top_down_row.addWidget(centered_browse_btn)
        layout.addLayout(top_down_row)

        gb.setLayout(layout)
        return gb

    def _build_videos_group(self) -> QGroupBox:
        gb = QGroupBox("Videos")
        layout = QVBoxLayout()
        layout.setSpacing(4)

        self._video_list = QListWidget()
        self._video_list.setSelectionMode(QAbstractItemView.NoSelection)
        self._video_list.setAlternatingRowColors(True)
        self._video_list.itemChanged.connect(lambda _: self._update_run_btn())
        layout.addWidget(self._video_list)

        sel_row = QHBoxLayout()
        btn_all = QPushButton("Select All")
        btn_all.clicked.connect(lambda: self._set_all_checked(True))
        btn_none = QPushButton("Select None")
        btn_none.clicked.connect(lambda: self._set_all_checked(False))
        sel_row.addWidget(btn_all)
        sel_row.addWidget(btn_none)
        sel_row.addStretch()
        layout.addLayout(sel_row)

        gb.setLayout(layout)
        return gb

    def _build_options_group(self) -> QGroupBox:
        gb = QGroupBox("Options")
        form = QFormLayout()
        form.setSpacing(4)

        self._frames_combo = QComboBox()
        self._frames_combo.addItem("All frames", "all")
        self._frames_combo.addItem("Unlabeled frames only", "unlabeled")
        self._frames_combo.addItem("Suggested frames only", "suggested")
        self._frames_combo.setToolTip(
            "Which frames to run inference on for each selected video."
        )
        form.addRow("Frames:", self._frames_combo)

        self._batch_spin = QSpinBox()
        self._batch_spin.setRange(1, 128)
        self._batch_spin.setValue(4)
        self._batch_spin.setToolTip("Number of frames processed in each batch.")
        form.addRow("Batch size:", self._batch_spin)

        self._max_inst_spin = QSpinBox()
        self._max_inst_spin.setRange(1, 200)
        self._max_inst_spin.setValue(10)
        self._max_inst_spin.setToolTip(
            "Maximum number of instances to detect per frame."
        )
        form.addRow("Max instances:", self._max_inst_spin)

        self._export_analysis_h5_check = QCheckBox("Analysis HDF5")
        self._export_analysis_h5_check.setToolTip(
            "After inference, export predictions to SLEAP Analysis HDF5."
        )
        form.addRow("Also export:", self._export_analysis_h5_check)

        self._export_nwb_check = QCheckBox("NWB")
        self._export_nwb_check.setToolTip(
            "After inference, export predictions to Neurodata Without Borders."
        )
        form.addRow("", self._export_nwb_check)

        gb.setLayout(form)
        return gb

    def _build_prediction_preview_group(self) -> QGroupBox:
        gb = QGroupBox("Prediction Preview")
        layout = QVBoxLayout()
        layout.setSpacing(4)

        self._preview_pred_list = QListWidget()
        self._preview_pred_list.setSelectionMode(QAbstractItemView.NoSelection)
        self._preview_pred_list.setAlternatingRowColors(True)
        self._preview_pred_list.setMaximumHeight(92)
        layout.addWidget(self._preview_pred_list)

        row = QHBoxLayout()
        self._preview_link_btn = QPushButton("Link Files")
        self._preview_link_btn.setToolTip(
            "Preview full-video predictions without importing them into the project."
        )
        self._preview_link_btn.clicked.connect(self._link_external_predictions)
        self._preview_clear_btn = QPushButton("Clear")
        self._preview_clear_btn.setToolTip("Remove linked prediction previews.")
        self._preview_clear_btn.clicked.connect(self._clear_external_predictions)
        self._preview_import_frame_btn = QPushButton("Import Frame")
        self._preview_import_frame_btn.setToolTip(
            "Import linked predictions for the visible frame into the project."
        )
        self._preview_import_frame_btn.clicked.connect(
            self._import_current_preview_frame
        )
        row.addWidget(self._preview_link_btn)
        row.addWidget(self._preview_import_frame_btn)
        row.addWidget(self._preview_clear_btn)
        layout.addLayout(row)

        self._preview_status_label = QLabel("")
        self._preview_status_label.setWordWrap(True)
        self._preview_status_label.setStyleSheet(
            "color: #9ca3af; font-size: 10px; padding: 2px 0;"
        )
        layout.addWidget(self._preview_status_label)

        gb.setLayout(layout)
        return gb

    def _build_convert_predictions_group(self) -> QGroupBox:
        gb = QGroupBox("Convert Predictions")
        layout = QVBoxLayout()
        layout.setSpacing(4)

        self._convert_pred_list = QListWidget()
        self._convert_pred_list.setSelectionMode(QAbstractItemView.NoSelection)
        self._convert_pred_list.setAlternatingRowColors(True)
        self._convert_pred_list.setMaximumHeight(92)
        layout.addWidget(self._convert_pred_list)

        file_row = QHBoxLayout()
        add_btn = QPushButton("Add Files")
        add_btn.setToolTip("Select prediction files to convert.")
        add_btn.clicked.connect(self._add_convert_predictions)
        clear_btn = QPushButton("Clear")
        clear_btn.setToolTip("Remove selected prediction files from this list.")
        clear_btn.clicked.connect(self._clear_convert_predictions)
        file_row.addWidget(add_btn)
        file_row.addWidget(clear_btn)
        file_row.addStretch()
        layout.addLayout(file_row)

        export_row = QHBoxLayout()
        export_row.addWidget(QLabel("Export:"))
        self._convert_analysis_h5_check = QCheckBox("Analysis HDF5")
        self._convert_analysis_h5_check.setChecked(True)
        self._convert_analysis_h5_check.toggled.connect(self._update_convert_btn)
        export_row.addWidget(self._convert_analysis_h5_check)
        self._convert_nwb_check = QCheckBox("NWB")
        self._convert_nwb_check.toggled.connect(self._update_convert_btn)
        export_row.addWidget(self._convert_nwb_check)
        export_row.addStretch()
        layout.addLayout(export_row)

        self._convert_run_btn = QPushButton("Convert")
        self._convert_run_btn.setEnabled(False)
        self._convert_run_btn.setToolTip(
            "Convert selected prediction files without running inference."
        )
        self._convert_run_btn.clicked.connect(self._convert_predictions)
        layout.addWidget(self._convert_run_btn)

        self._convert_status_label = QLabel("")
        self._convert_status_label.setWordWrap(True)
        self._convert_status_label.setStyleSheet(
            "color: #9ca3af; font-size: 10px; padding: 2px 0;"
        )
        layout.addWidget(self._convert_status_label)

        gb.setLayout(layout)
        return gb

    def _build_projection_group(self) -> QGroupBox:
        gb = QGroupBox("3D Projections")
        layout = QVBoxLayout()
        layout.setSpacing(4)

        cal_row = QHBoxLayout()
        self._proj_calibration_edit = QLineEdit()
        self._proj_calibration_edit.setPlaceholderText("calibration.toml")
        self._proj_calibration_edit.textChanged.connect(self._update_projection_btn)
        cal_row.addWidget(self._proj_calibration_edit)
        cal_btn = QPushButton("Browse")
        cal_btn.setFixedWidth(70)
        cal_btn.clicked.connect(self._browse_projection_calibration)
        cal_row.addWidget(cal_btn)
        layout.addLayout(cal_row)

        self._proj_pred_list = QListWidget()
        self._proj_pred_list.setSelectionMode(QAbstractItemView.NoSelection)
        self._proj_pred_list.setAlternatingRowColors(True)
        self._proj_pred_list.itemChanged.connect(lambda _: self._update_projection_btn())
        layout.addWidget(self._proj_pred_list)

        pred_row = QHBoxLayout()
        discover_btn = QPushButton("Discover")
        discover_btn.clicked.connect(self._discover_projection_predictions)
        add_btn = QPushButton("Add Files")
        add_btn.clicked.connect(self._add_projection_predictions)
        clear_btn = QPushButton("Clear")
        clear_btn.clicked.connect(self._clear_projection_predictions)
        pred_row.addWidget(discover_btn)
        pred_row.addWidget(add_btn)
        pred_row.addWidget(clear_btn)
        layout.addLayout(pred_row)

        out_row = QHBoxLayout()
        self._proj_output_edit = QLineEdit()
        self._proj_output_edit.setPlaceholderText("Output folder (session folder)")
        self._proj_output_edit.textChanged.connect(self._update_projection_btn)
        out_row.addWidget(self._proj_output_edit)
        out_btn = QPushButton("Browse")
        out_btn.setFixedWidth(70)
        out_btn.clicked.connect(self._browse_projection_output)
        out_row.addWidget(out_btn)
        layout.addLayout(out_row)

        self._proj_run_btn = QPushButton("Run 3D Projections")
        self._proj_run_btn.setEnabled(False)
        self._proj_run_btn.setToolTip(
            "Triangulate selected prediction files with calibration.toml and save "
            "points3d.h5 plus reprojections.h5."
        )
        self._proj_run_btn.clicked.connect(self._run_3d_projections)
        layout.addWidget(self._proj_run_btn)

        self._proj_status_label = QLabel("")
        self._proj_status_label.setWordWrap(True)
        self._proj_status_label.setStyleSheet(
            "color: #9ca3af; font-size: 10px; padding: 2px 0;"
        )
        layout.addWidget(self._proj_status_label)
        layout.addWidget(self._build_translate_group())

        gb.setLayout(layout)
        return gb

    def _build_translate_group(self) -> QGroupBox:
        gb = QGroupBox("3D Translate")
        layout = QVBoxLayout()
        layout.setSpacing(4)

        points_row = QHBoxLayout()
        self._translate_points_edit = QLineEdit()
        self._translate_points_edit.setPlaceholderText("points3d.h5")
        self._translate_points_edit.textChanged.connect(self._update_translate_btn)
        points_row.addWidget(self._translate_points_edit)
        points_btn = QPushButton("Browse")
        points_btn.setFixedWidth(70)
        points_btn.clicked.connect(self._browse_translate_points3d)
        points_row.addWidget(points_btn)
        layout.addLayout(points_row)

        node_row = QHBoxLayout()
        self._translate_node_combo = QComboBox()
        self._translate_node_combo.setEnabled(False)
        self._translate_node_combo.currentIndexChanged.connect(
            lambda _: self._update_translate_btn()
        )
        node_row.addWidget(QLabel("Origin node:"))
        node_row.addWidget(self._translate_node_combo, 1)
        load_btn = QPushButton("Load Nodes")
        load_btn.clicked.connect(self._load_translate_nodes)
        node_row.addWidget(load_btn)
        layout.addLayout(node_row)

        mode_row = QHBoxLayout()
        self._translate_mode_combo = QComboBox()
        self._translate_mode_combo.addItem("Per-frame node position", "frame")
        self._translate_mode_combo.addItem("Median node position", "median")
        self._translate_mode_combo.setToolTip(
            "Per-frame makes the selected node exactly (0,0,0) in each frame. "
            "Median uses one fixed origin across the file."
        )
        mode_row.addWidget(QLabel("Origin mode:"))
        mode_row.addWidget(self._translate_mode_combo, 1)
        layout.addLayout(mode_row)

        self._translate_run_btn = QPushButton("Save Translated 3D Points")
        self._translate_run_btn.setEnabled(False)
        self._translate_run_btn.setToolTip(
            "Write points3d_translated.h5 with all points translated relative "
            "to the selected node."
        )
        self._translate_run_btn.clicked.connect(self._run_translate_points3d)
        layout.addWidget(self._translate_run_btn)

        self._translate_status_label = QLabel("")
        self._translate_status_label.setWordWrap(True)
        self._translate_status_label.setStyleSheet(
            "color: #9ca3af; font-size: 10px; padding: 2px 0;"
        )
        layout.addWidget(self._translate_status_label)

        gb.setLayout(layout)
        return gb

    # ── video-list helpers ───────────────────────────────────────────────── #

    def _refresh_videos(self) -> None:
        """Repopulate the video checklist from the current project."""
        if not hasattr(self, "_video_list"):
            return

        labels = self.main_window.labels
        # Remember which videos were previously checked by filename so that a
        # project reload doesn't silently deselect everything.
        prev_checked = set()
        for i in range(self._video_list.count()):
            item = self._video_list.item(i)
            if item.checkState() == Qt.Checked:
                prev_checked.add(item.data(Qt.UserRole + 1))  # stored filename

        self._video_list.clear()

        if labels is None:
            self._set_status("No project loaded.")
            self._update_run_btn()
            return

        videos = list(getattr(labels, "videos", []) or [])
        if not videos:
            self._set_status("No videos found in this project.")
            self._update_run_btn()
            return

        frame_count_errors = []
        for video in videos:
            filename = video.filename
            if isinstance(filename, list):
                filename = filename[0]
            filename_str = str(filename) if filename else ""
            display_name = Path(filename_str).name if filename_str else "Unknown"
            try:
                n_frames = len(video)
                frame_text = f"{n_frames:,} frames"
            except Exception as exc:
                frame_text = "frame count unavailable"
                frame_count_errors.append(f"{display_name}: {exc}")
            item = QListWidgetItem(f"{display_name}  ({frame_text})")
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            # Default to checked; preserve previous state if project reloaded
            checked_key = filename_str or display_name
            was_checked = (not prev_checked) or (checked_key in prev_checked)
            item.setCheckState(Qt.Checked if was_checked else Qt.Unchecked)
            item.setData(Qt.UserRole, video)
            item.setData(Qt.UserRole + 1, checked_key)
            item.setToolTip(filename_str)
            self._video_list.addItem(item)

        if frame_count_errors:
            self._set_status(
                "Some video frame counts could not be read; they are still listed "
                "for inference. Check video paths if inference fails.",
                error=False,
            )
            logger.warning(
                "Analysis video frame-count errors:\n%s",
                "\n".join(frame_count_errors),
            )
        else:
            self._set_status("")

        self._update_run_btn()

    def _set_all_checked(self, checked: bool) -> None:
        state = Qt.Checked if checked else Qt.Unchecked
        for i in range(self._video_list.count()):
            self._video_list.item(i).setCheckState(state)

    def _checked_videos(self) -> List[Video]:
        videos = []
        for i in range(self._video_list.count()):
            item = self._video_list.item(i)
            if item.checkState() == Qt.Checked:
                videos.append(item.data(Qt.UserRole))
        return videos

    def _update_run_btn(self) -> None:
        has_model = bool(getattr(self, "_model_path_edit", None) and
                         self._model_path_edit.text().strip())
        has_videos = bool(getattr(self, "_video_list", None) and self._checked_videos())
        self._run_btn.setEnabled(has_model and has_videos)

    def _update_projection_btn(self) -> None:
        if not hasattr(self, "_proj_run_btn"):
            return
        has_calibration = bool(self._proj_calibration_edit.text().strip())
        has_output = bool(self._projection_output_dir())
        self._proj_run_btn.setEnabled(
            has_calibration and has_output and len(self._checked_projection_files()) >= 2
        )
        self._update_translate_default_path()

    def _update_translate_btn(self) -> None:
        if not hasattr(self, "_translate_run_btn"):
            return
        has_points = bool(self._translate_points_edit.text().strip())
        has_node = (
            self._translate_node_combo.isEnabled()
            and self._translate_node_combo.currentData() is not None
        )
        self._translate_run_btn.setEnabled(has_points and has_node)

    # ── browse ───────────────────────────────────────────────────────────── #

    def _update_convert_btn(self) -> None:
        if not hasattr(self, "_convert_run_btn"):
            return
        has_files = bool(self._convert_prediction_files())
        has_format = (
            self._convert_analysis_h5_check.isChecked()
            or self._convert_nwb_check.isChecked()
        )
        self._convert_run_btn.setEnabled(has_files and has_format)

    def _browse_model(self) -> None:
        path = FileDialog.openDir(
            self,
            caption="Select Trained Model Directory",
        )
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

    def _browse_projection_calibration(self) -> None:
        filename, _ = FileDialog.open(
            self,
            caption="Select calibration.toml",
            filter="TOML Files (*.toml);;All Files (*)",
        )
        if filename:
            self._proj_calibration_edit.setText(filename)

    def _browse_projection_output(self) -> None:
        path = FileDialog.openDir(
            self,
            caption="Select 3D Projection Output Folder",
        )
        if path:
            self._proj_output_edit.setText(path)

    def _browse_translate_points3d(self) -> None:
        filename, _ = FileDialog.open(
            self,
            caption="Select points3d.h5",
            filter="HDF5 Files (*.h5);;All Files (*)",
        )
        if filename:
            self._translate_points_edit.setText(filename)
            self._load_translate_nodes()

    def _add_projection_predictions(self) -> None:
        filenames, _ = FileDialog.openMultiple(
            self,
            caption="Select prediction files",
            filter="SLEAP Files (*.slp *.h5);;All Files (*)",
        )
        for filename in filenames or []:
            self._add_projection_prediction_file(Path(filename), checked=True)
        self._update_projection_btn()

    def _clear_projection_predictions(self) -> None:
        self._proj_pred_list.clear()
        self._update_projection_btn()

    def _add_convert_predictions(self) -> None:
        filenames, _ = FileDialog.openMultiple(
            self,
            caption="Select prediction files to convert",
            filter="SLEAP Prediction Files (*.slp *.h5);;All Files (*)",
        )
        for filename in filenames or []:
            self._add_convert_prediction_file(Path(filename))
        self._update_convert_btn()

    def _clear_convert_predictions(self) -> None:
        if not hasattr(self, "_convert_pred_list"):
            return
        self._convert_pred_list.clear()
        self._convert_status("")
        self._update_convert_btn()

    def _link_external_predictions(self) -> None:
        filenames, _ = FileDialog.openMultiple(
            self,
            caption="Select prediction files to preview",
            filter=(
                "Prediction Files (*.slp *.h5 *.hdf5);;"
                "SLEAP Files (*.slp);;"
                "SLEAP Analysis HDF5 (*.h5 *.hdf5);;"
                "All Files (*)"
            ),
        )
        filenames = [str(filename) for filename in filenames or []]
        if not filenames:
            return

        if getattr(self, "_prediction_load_worker", None) is not None:
            worker = self._prediction_load_worker
            if worker.isRunning():
                self._prediction_preview_status(
                    "Prediction preview load is already running."
                )
                return

        self._set_prediction_preview_loading(True)
        self._prediction_preview_status("Loading prediction preview...")
        worker = ExternalPredictionLoadWorker(filenames, parent=self)
        self._prediction_load_worker = worker
        self._prediction_load_labels = self.main_window.labels
        worker.statusUpdate.connect(self._prediction_preview_status)
        worker.resultReady.connect(self._on_external_predictions_loaded)
        worker.finished.connect(self._on_prediction_load_thread_finished)
        worker.finished.connect(worker.deleteLater)
        worker.start()

    def _set_prediction_preview_loading(self, loading: bool) -> None:
        """Keep preview controls consistent while a background load is active."""
        self._preview_link_btn.setEnabled(not loading)
        self._preview_clear_btn.setEnabled(not loading)
        self._preview_import_frame_btn.setEnabled(not loading)

    def _on_prediction_load_thread_finished(self) -> None:
        """Release the completed worker without retaining Qt thread objects."""
        worker = self.sender()
        if self._prediction_load_worker is worker:
            self._prediction_load_worker = None
            self._prediction_load_labels = None
        self._set_prediction_preview_loading(False)

    def _on_external_predictions_loaded(
        self, success: bool, prediction_sets: object, error: str
    ) -> None:
        if not success:
            self._prediction_preview_status(
                f"Prediction preview load failed: {error}", error=True
            )
            return

        if self._prediction_load_labels is not self.main_window.labels:
            self._prediction_preview_status(
                "The project changed while predictions were loading; ignored the "
                "stale preview result."
            )
            return

        manager = get_external_prediction_manager(self.main_window.state)
        matched_count = self._assign_external_prediction_videos(prediction_sets)
        skeleton_loaded = self._load_skeleton_from_predictions_if_needed(
            prediction_sets
        )
        loaded_count = 0
        instance_count = 0
        for prediction_set in prediction_sets:
            manager.add(prediction_set)
            loaded_count += 1
            instance_count += prediction_set.total_instances
            self._add_preview_prediction_file(
                Path(prediction_set.source_path),
                target_name=prediction_set.target_video_name,
            )

        target_text = (
            f" Matched {matched_count}/{loaded_count} file(s) to project videos."
            if loaded_count
            else ""
        )
        self._prediction_preview_status(
            f"Linked {loaded_count} file(s), {instance_count:,} prediction instance(s)."
            f"{target_text}{' Loaded skeleton from predictions.' if skeleton_loaded else ''}"
        )
        self.main_window.plotFrame()

    def _on_project_labels_changed(self, _labels) -> None:
        """Clear previews when a different project is loaded into this window."""
        self._refresh_videos()
        manager = self.main_window.state.get("external predictions", default=None)
        had_previews = manager is not None and len(manager) > 0
        if manager is not None:
            manager.clear()
        if hasattr(self, "_preview_pred_list"):
            had_previews = had_previews or self._preview_pred_list.count() > 0
            self._preview_pred_list.clear()
        if had_previews:
            self._prediction_preview_status(
                "Cleared prediction previews from the previous project."
            )

    def _load_skeleton_from_predictions_if_needed(
        self, prediction_sets: List[ExternalPredictionSet]
    ) -> bool:
        """Load a project skeleton from linked predictions for empty projects."""
        labels = self.main_window.labels
        if labels is None or getattr(labels, "skeletons", None):
            return False

        skeleton = next(
            (
                block.skeleton
                for prediction_set in prediction_sets
                for block in prediction_set.blocks
                if block.skeleton is not None
            ),
            None,
        )
        if skeleton is None:
            return False

        labels.skeletons.append(skeleton)
        self.main_window.state["skeleton"] = skeleton
        self.main_window.commands.signal_update([UpdateTopic.skeleton])
        self.main_window.commands.changestack_push(
            "skeleton_from_prediction_preview"
        )
        return True

    def _clear_external_predictions(self) -> None:
        manager = get_external_prediction_manager(self.main_window.state)
        manager.clear()
        self._preview_pred_list.clear()
        self._prediction_preview_status("Cleared linked prediction previews.")
        self.main_window.plotFrame()

    def _import_current_preview_frame(self) -> None:
        labels = self.main_window.labels
        if labels is None:
            self._prediction_preview_status("No project loaded.", error=True)
            return

        manager = get_external_prediction_manager(self.main_window.state)
        if len(manager) == 0:
            self._prediction_preview_status("No linked prediction previews.")
            return

        player = getattr(self.main_window, "player", None)
        video = (
            player.interaction_video()
            if player is not None and hasattr(player, "interaction_video")
            else self.main_window.state.get("video", default=None)
        )
        frame_idx = (
            player.interaction_frame_idx()
            if player is not None and hasattr(player, "interaction_frame_idx")
            else self.main_window.state.get("frame_idx", default=0)
        )
        if video is None or frame_idx is None:
            self._prediction_preview_status("No visible video frame.", error=True)
            return

        predictions = manager.instances_for(video, int(frame_idx))
        if not predictions:
            self._prediction_preview_status("No linked predictions on this frame.")
            return

        labeled_frame = labels.find(video, int(frame_idx), return_new=True)[0]
        if any(
            isinstance(inst, PredictedInstance)
            for inst in getattr(labeled_frame, "instances", []) or []
        ):
            self._prediction_preview_status(
                "This frame already has project predictions."
            )
            return

        predictions = [
            self.main_window._canonicalize_preview_prediction(prediction)
            if hasattr(self.main_window, "_canonicalize_preview_prediction")
            else prediction
            for prediction in predictions
        ]
        labeled_frame.instances.extend(predictions)
        if labeled_frame not in labels:
            labels.append(labeled_frame)

        self.main_window.commands.signal_update(
            [UpdateTopic.project_instances, UpdateTopic.frame]
        )
        self.main_window.commands.changestack_push(
            "external_prediction_preview_frame"
        )
        self._prediction_preview_status(
            f"Imported {len(predictions)} prediction(s) for frame {int(frame_idx)}."
        )

    def _add_preview_prediction_file(
        self, path: Path, target_name: Optional[str] = None
    ) -> None:
        path = path.expanduser()
        existing = {
            self._preview_pred_list.item(i).data(Qt.UserRole)
            for i in range(self._preview_pred_list.count())
        }
        label = path.name if not target_name else f"{path.name} -> {target_name}"
        tooltip = (
            str(path) if not target_name else f"{path}\nPreview target: {target_name}"
        )
        if str(path) in existing:
            for i in range(self._preview_pred_list.count()):
                item = self._preview_pred_list.item(i)
                if item.data(Qt.UserRole) == str(path):
                    item.setText(label)
                    item.setToolTip(tooltip)
            return
        item = QListWidgetItem(label)
        item.setData(Qt.UserRole, str(path))
        item.setToolTip(tooltip)
        self._preview_pred_list.addItem(item)

    def _add_convert_prediction_file(self, path: Path) -> None:
        path = path.expanduser()
        existing = set(self._convert_prediction_files())
        if str(path) in existing:
            return
        item = QListWidgetItem(path.name)
        item.setData(Qt.UserRole, str(path))
        item.setToolTip(str(path))
        self._convert_pred_list.addItem(item)

    def _convert_prediction_files(self) -> List[str]:
        if not hasattr(self, "_convert_pred_list"):
            return []
        return [
            self._convert_pred_list.item(i).data(Qt.UserRole)
            for i in range(self._convert_pred_list.count())
        ]

    def _convert_predictions(self) -> None:
        from sleap.gui.prediction_exports import export_prediction_file_formats

        prediction_files = self._convert_prediction_files()
        analysis_h5 = self._convert_analysis_h5_check.isChecked()
        nwb = self._convert_nwb_check.isChecked()
        if not prediction_files:
            self._convert_status("Select at least one prediction file.", error=True)
            return
        if not analysis_h5 and not nwb:
            self._convert_status("Select at least one export format.", error=True)
            return

        self._convert_run_btn.setEnabled(False)
        written_paths: List[str] = []
        try:
            for prediction_file in prediction_files:
                self._convert_status(f"Converting {Path(prediction_file).name}...")
                QtCore.QCoreApplication.processEvents()
                written_paths.extend(
                    export_prediction_file_formats(
                        prediction_file,
                        analysis_h5=analysis_h5,
                        nwb=nwb,
                    )
                )
        except Exception as exc:
            logger.exception("Prediction conversion failed")
            self._convert_status(f"Conversion failed: {exc}", error=True)
            self._update_convert_btn()
            return

        if written_paths:
            names = ", ".join(Path(path).name for path in written_paths)
            self._convert_status(f"Converted {len(written_paths):,} file(s): {names}.")
        else:
            self._convert_status(
                "Conversion finished, but no files were written.", error=True
            )
        self._update_convert_btn()

    def _assign_external_prediction_videos(
        self, prediction_sets: List[ExternalPredictionSet]
    ) -> int:
        """Assign single-video prediction files to one project video."""
        labels = self.main_window.labels
        videos = list(getattr(labels, "videos", []) or []) if labels is not None else []
        if not videos:
            return 0

        matched_count = 0
        for prediction_set in prediction_sets:
            video = self._match_external_prediction_video(prediction_set, videos)
            if video is not None:
                prediction_set.assign_to_video(video)
                matched_count += 1

        return matched_count

    def _match_external_prediction_video(
        self, prediction_set: ExternalPredictionSet, videos: List[Video]
    ) -> Optional[Video]:
        """Return the project video that should receive a prediction preview."""
        block_keys = {block.video_key for block in prediction_set.blocks}
        exact_matches = [video for video in videos if video_key(video) in block_keys]
        if len(exact_matches) == 1:
            return exact_matches[0]

        if len(videos) == 1 and len(prediction_set.blocks) == 1:
            return videos[0]

        source_text = self._match_text(Path(prediction_set.source_path).name)
        scores = []
        for video in videos:
            score = self._prediction_video_match_score(
                prediction_set, video, source_text
            )
            if score > 0:
                scores.append((score, video))

        if not scores:
            return None

        scores.sort(key=lambda item: item[0], reverse=True)
        if len(scores) > 1 and scores[0][0] == scores[1][0]:
            return None
        return scores[0][1]

    def _prediction_video_match_score(
        self,
        prediction_set: ExternalPredictionSet,
        video: Video,
        source_text: str,
    ) -> int:
        """Score how confidently a prediction file name matches a project video."""
        best = 0
        block_texts = [
            self._match_text(block.video_name)
            for block in prediction_set.blocks
        ] + [
            self._match_text(block.video_key)
            for block in prediction_set.blocks
        ]

        for token in self._video_match_tokens(video):
            token_text = self._match_text(token)
            if not token_text:
                continue
            if source_text == token_text:
                best = max(best, 10000 + len(token_text))
            elif token_text in source_text:
                best = max(best, 1000 + len(token_text))

            for block_text in block_texts:
                if block_text == token_text:
                    best = max(best, 500 + len(token_text))
                elif token_text in block_text:
                    best = max(best, 100 + len(token_text))

        return best

    def _video_match_tokens(self, video: Video) -> List[str]:
        """Return filename and camera-name tokens that can identify a video."""
        tokens = []
        filename = video.filename
        if isinstance(filename, list):
            filename = filename[0] if filename else ""
        if filename:
            path = Path(str(filename))
            tokens.extend([path.name, path.stem])

        labels = self.main_window.labels
        for session in getattr(labels, "sessions", []) or []:
            videos = list(getattr(session, "videos", []) or [])
            if video not in videos:
                continue
            video_idx = videos.index(video)
            cameras = list(getattr(session, "cameras", []) or [])
            if not cameras and getattr(session, "camera_group", None) is not None:
                cameras = list(getattr(session.camera_group, "cameras", []) or [])
            if video_idx < len(cameras) and getattr(cameras[video_idx], "name", None):
                tokens.append(cameras[video_idx].name)

        return tokens

    @staticmethod
    def _match_text(text: object) -> str:
        """Normalize text for loose filename/camera matching."""
        return re.sub(r"[^a-z0-9]+", "", str(text or "").casefold())

    def _discover_projection_predictions(self) -> None:
        if not hasattr(self, "_proj_pred_list"):
            return
        candidates = []
        session_dir = self._current_session_dir()
        if session_dir is not None:
            for search_root in (session_dir, session_dir.parent):
                pred_dir = search_root / "predictions"
                if pred_dir.is_dir():
                    candidates.extend(pred_dir.glob("*.slp"))
                    candidates.extend(pred_dir.glob("*.h5"))
            cal = self._discover_calibration_path(session_dir)
            if cal.exists() and not self._proj_calibration_edit.text().strip():
                self._proj_calibration_edit.setText(str(cal))
            if not self._proj_output_edit.text().strip():
                self._proj_output_edit.setText(str(session_dir))
            self._update_translate_default_path()

        if not candidates:
            self._projection_status("No prediction files found in predictions/.")
            self._update_projection_btn()
            return

        for candidate in sorted(set(candidates)):
            self._add_projection_prediction_file(candidate, checked=True)
        self._projection_status(f"Found {len(candidates)} prediction file(s).")
        self._update_projection_btn()

    def _add_projection_prediction_file(self, path: Path, *, checked: bool) -> None:
        path = path.expanduser()
        existing = {
            self._proj_pred_list.item(i).data(Qt.UserRole)
            for i in range(self._proj_pred_list.count())
        }
        if str(path) in existing:
            return
        item = QListWidgetItem(path.name)
        item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
        item.setCheckState(Qt.Checked if checked else Qt.Unchecked)
        item.setData(Qt.UserRole, str(path))
        item.setToolTip(str(path))
        self._proj_pred_list.addItem(item)

    def _checked_projection_files(self) -> List[str]:
        if not hasattr(self, "_proj_pred_list"):
            return []
        paths = []
        for i in range(self._proj_pred_list.count()):
            item = self._proj_pred_list.item(i)
            if item.checkState() == Qt.Checked:
                paths.append(item.data(Qt.UserRole))
        return paths

    def _projection_output_dir(self) -> Optional[str]:
        if not hasattr(self, "_proj_output_edit"):
            return None
        explicit = self._proj_output_edit.text().strip()
        if explicit:
            return explicit
        session_dir = self._current_session_dir()
        return str(session_dir) if session_dir is not None else None

    def _current_session_dir(self) -> Optional[Path]:
        labels = self.main_window.labels
        video = self.main_window.state.get("video", default=None)
        if video is None and labels is not None and getattr(labels, "videos", None):
            video = labels.videos[0]
        if video is None:
            filename = self.main_window.state.get("filename", default=None)
            return Path(filename).parent if filename else None
        filename = video.filename
        if isinstance(filename, list):
            filename = filename[0] if filename else ""
        return Path(str(filename)).parent if filename else None

    def _discover_calibration_path(self, session_dir: Path) -> Path:
        for candidate in [session_dir, *session_dir.parents]:
            calibration = candidate / "calibration.toml"
            if calibration.exists():
                return calibration
        return session_dir / "calibration.toml"

    # ── inference ────────────────────────────────────────────────────────── #

    def _update_translate_default_path(self, *, force: bool = False) -> None:
        if not hasattr(self, "_translate_points_edit"):
            return
        if self._translate_points_edit.text().strip() and not force:
            return
        output_dir = self._projection_output_dir()
        if not output_dir:
            return
        candidates = [
            Path(output_dir) / "points3d.h5",
            Path(output_dir) / "points3D.h5",
        ]
        for candidate in candidates:
            if candidate.exists():
                self._translate_points_edit.setText(str(candidate))
                self._load_translate_nodes()
                break

    def _load_translate_nodes(self) -> None:
        if not hasattr(self, "_translate_node_combo"):
            return
        path = self._translate_points_edit.text().strip()
        self._translate_node_combo.clear()
        self._translate_node_combo.setEnabled(False)
        self._update_translate_btn()
        if not path:
            self._translate_status("Select a points3d.h5 file.", error=True)
            return

        try:
            from sleap.gui.reach_projection import load_points3d_h5

            data = load_points3d_h5(path)
        except Exception as exc:
            logger.exception("Could not load points3d nodes")
            self._translate_status(f"Could not load nodes: {exc}", error=True)
            return

        node_names = list(data.get("node_names", []) or [])
        if not node_names:
            self._translate_status("No node names found in points3D file.", error=True)
            return

        for node_name in node_names:
            self._translate_node_combo.addItem(str(node_name), str(node_name))
        self._translate_node_combo.setEnabled(True)
        preferred = self._preferred_translate_node(node_names)
        if preferred in node_names:
            self._translate_node_combo.setCurrentIndex(node_names.index(preferred))
        self._translate_status(
            f"Loaded {len(node_names)} node(s) from {Path(path).name}."
        )
        self._update_translate_btn()

    @staticmethod
    def _preferred_translate_node(node_names: List[str]) -> Optional[str]:
        lowered = {name.lower(): name for name in node_names}
        for key in ("bar_r", "barr", "bar right", "bar-right"):
            if key in lowered:
                return lowered[key]
        return node_names[0] if node_names else None

    def _run_inference(self) -> None:
        from sleap.gui.learning.runners import (
            DatasetItemForInference,
            InferenceTask,
            ItemsForInference,
            VideoItemForInference,
            run_gui_inference,
        )
        from sleap_io import save_file

        model_paths = self._selected_model_paths()
        if not model_paths[0]:
            return

        videos = self._checked_videos()
        if not videos:
            QMessageBox.warning(
                self,
                "No Videos Selected",
                "Please check at least one video before running inference.",
            )
            return

        labels = self.main_window.labels
        labels_filename = self.main_window.state.get("filename")

        if not labels_filename:
            QMessageBox.warning(
                self,
                "Project Not Saved",
                "Please save the project first (File → Save) so that predictions "
                "have a location to be written to.",
            )
            return

        frame_mode = self._frames_combo.currentData()

        # Build inference items without serialising large frame lists into CLI
        # arguments (which overflows Windows's ~32k-char command-line limit).
        # "suggested" uses DatasetItemForInference so the CLI filters server-side.
        # "all" / "unlabeled" use VideoItemForInference with frames=None so no
        # --frames flag is emitted. Results replace prior predictions while
        # preserving user-labeled instances.
        inference_items: List = []
        total_frames = 0

        for video in videos:
            video_idx = labels.videos.index(video)

            if frame_mode == "suggested":
                n = len([sf for sf in labels.suggestions if sf.video is video])
                if n == 0:
                    continue
                total_frames += n
                inference_items.append(
                    DatasetItemForInference(
                        labels_path=labels_filename,
                        frame_filter="suggested",
                        video_idx=video_idx,
                    )
                )
            else:
                # "all" or "unlabeled" — omit --frames to avoid the argument
                # length limit; the full video is processed and _prediction_mode
                # controls whether existing user labels are preserved.
                total_frames += len(video)
                inference_items.append(
                    VideoItemForInference(
                        video=video,
                        frames=None,
                        use_absolute_path=True,
                    )
                )

        if not inference_items:
            QMessageBox.information(
                self,
                "Nothing to Process",
                "No frames match the current 'Frames' setting for the selected "
                "videos (e.g. no suggestions exist).",
            )
            return

        items = ItemsForInference(
            items=inference_items,
            total_frame_count=total_frames,
        )

        task = InferenceTask(
            trained_job_paths=model_paths,
            labels=labels,
            labels_filename=labels_filename,
            inference_params={
                "_batch_size": self._batch_spin.value(),
                "_max_instances": self._max_inst_spin.value(),
                "_predict_target": "all",
                "_clear_all_first": False,
                "_prediction_mode": "replace",
            },
        )

        self._run_btn.setEnabled(False)
        self._set_status("Running inference…")

        try:
            new_frames = run_gui_inference(task, items, gui=True)
        except Exception as e:
            logger.exception("Inference failed")
            self._set_status(f"Error during inference: {e}", error=True)
            self._run_btn.setEnabled(True)
            return

        if new_frames < 0:
            self._set_status("Inference cancelled.")
            self._run_btn.setEnabled(True)
            return

        # Auto-save predictions into the project file.
        try:
            save_file(labels, labels_filename)
            save_name = Path(labels_filename).name
            export_paths = []
            if (
                self._export_analysis_h5_check.isChecked()
                or self._export_nwb_check.isChecked()
            ):
                from sleap.gui.prediction_exports import export_prediction_formats

                export_paths = export_prediction_formats(
                    labels,
                    source_path=labels_filename,
                    output_dir=Path(labels_filename).parent,
                    output_prefix=f"{Path(labels_filename).stem}.predictions",
                    videos=videos,
                    analysis_h5=self._export_analysis_h5_check.isChecked(),
                    nwb=self._export_nwb_check.isChecked(),
                )
            export_text = (
                " Exported: "
                + ", ".join(Path(path).name for path in export_paths)
                + "."
                if export_paths
                else (
                    " Export requested, but no export files were written."
                    if (
                        self._export_analysis_h5_check.isChecked()
                        or self._export_nwb_check.isChecked()
                    )
                    else ""
                )
            )
            self._set_status(
                f"Done - {new_frames:,} frame(s) predicted and saved to "
                f"{save_name}.{export_text}"
            )
            # Notify the rest of the GUI that predictions changed.
            self.main_window.commands.signal_update(
                [UpdateTopic.project_instances, UpdateTopic.frame]
            )
            self.main_window.commands.changestack_push("analysis_dock_inference")
        except Exception as e:
            logger.exception("Auto-save after inference failed")
            self._set_status(
                f"Predicted {new_frames:,} frame(s) but auto-save failed: {e}",
                error=True,
            )

        self._run_btn.setEnabled(True)

    def _run_3d_projections(self) -> None:
        from sleap.gui.learning.runners import InferenceProgressDialog

        prediction_files = self._checked_projection_files()
        calibration_path = self._proj_calibration_edit.text().strip()
        output_dir = self._projection_output_dir()
        if len(prediction_files) < 2 or not calibration_path or not output_dir:
            self._projection_status(
                "Select calibration.toml, an output folder, and at least two prediction files.",
                error=True,
            )
            return

        self._proj_run_btn.setEnabled(False)
        self._projection_status("Running 3D projections...")

        dialog = InferenceProgressDialog(self)
        dialog.setWindowTitle("Running 3D Projections")
        dialog.setLabelText("<b>Running 3D projections...</b>")
        dialog.appendLog(f"Calibration: {calibration_path}")
        dialog.appendLog("Prediction files:")
        for path in prediction_files:
            dialog.appendLog(f"  {path}")
        dialog.appendLog(f"Output folder: {output_dir}")

        worker = ProjectionWorker(
            prediction_files=prediction_files,
            calibration_path=calibration_path,
            output_dir=output_dir,
            parent=self,
        )
        result = {"success": False, "metadata": {}, "error": ""}

        def on_progress(current: int, total: int) -> None:
            dialog.setMaximum(total)
            dialog.setValue(current)

        def on_status(message: str) -> None:
            if message:
                dialog.setLabelText(message)

        def on_log(line: str) -> None:
            if line:
                dialog.appendLog(line)

        def on_finished(success: bool, metadata: dict, error: str) -> None:
            result["success"] = success
            result["metadata"] = metadata
            result["error"] = error
            dialog._ok_button.setEnabled(True)
            dialog._cancel_button.setEnabled(False)
            if success:
                excluded_cameras = metadata.get("excluded_camera_names", [])
                validation_result = (
                    "<br>Excluded camera(s): "
                    + ", ".join(str(name) for name in excluded_cameras)
                    if excluded_cameras
                    else "<br>All input cameras passed validation."
                )
                dialog.setLabelText(
                    "<b>3D projections complete!</b><br><br>"
                    f"Triangulated {metadata.get('n_frames', 0):,} frames, "
                    f"{metadata.get('n_nodes', 0)} nodes, "
                    f"{metadata.get('n_views', 0)} cameras."
                    f"{validation_result}"
                )
            else:
                dialog.setLabelText(f"<b>3D projection failed.</b><br><br>{error}")

        worker.progressUpdate.connect(on_progress)
        worker.statusUpdate.connect(on_status)
        worker.logOutput.connect(on_log)
        worker.finished.connect(on_finished)
        dialog._cancel_button.clicked.connect(worker.cancel)

        worker.start()
        dialog.exec_()
        if worker.isRunning():
            worker.cancel()
            worker.wait(5000)

        if result["success"]:
            metadata = result["metadata"]
            if "points3d_path" in metadata:
                self._translate_points_edit.setText(str(metadata["points3d_path"]))
                self._load_translate_nodes()
            excluded_cameras = metadata.get("excluded_camera_names", [])
            validation_summary = (
                f" Excluded camera(s): {', '.join(excluded_cameras)}."
                if excluded_cameras
                else ""
            )
            self._projection_status(
                f"Wrote {Path(metadata['points3d_path']).name} and "
                f"{Path(metadata['reprojections_path']).name} "
                f"({metadata['n_frames']:,} frames, {metadata['n_nodes']} nodes)."
                f"{validation_summary}"
            )
        else:
            self._projection_status(
                f"3D projection failed: {result['error'] or 'canceled'}",
                error=True,
            )
        self._update_projection_btn()

    def _run_translate_points3d(self) -> None:
        points_path = self._translate_points_edit.text().strip()
        origin_node = self._translate_node_combo.currentData()
        mode = self._translate_mode_combo.currentData()
        if not points_path or origin_node is None:
            self._translate_status("Select points3d.h5 and an origin node.", error=True)
            return

        try:
            from sleap.gui.reach_projection import translate_points3d_h5

            metadata = translate_points3d_h5(
                points_path,
                str(origin_node),
                output_dir=Path(points_path).expanduser().parent,
                output_filename="points3d_translated.h5",
                mode=str(mode or "frame"),
                h5_compression=None,
            )
        except Exception as exc:
            logger.exception("3D points translation failed")
            self._translate_status(f"3D translate failed: {exc}", error=True)
            return

        self._translate_status(
            f"Wrote {Path(metadata['points3d_path']).name} relative to "
            f"{metadata['origin_node']} ({metadata['n_frames']:,} frames)."
        )

    def _set_status(self, text: str, *, error: bool = False) -> None:
        color = "#ef4444" if error else "#9ca3af"
        self._status_label.setStyleSheet(
            f"color: {color}; font-size: 10px; padding: 2px 0;"
        )
        self._status_label.setText(text)

    def _prediction_preview_status(self, text: str, *, error: bool = False) -> None:
        if not hasattr(self, "_preview_status_label"):
            return
        color = "#ef4444" if error else "#9ca3af"
        self._preview_status_label.setStyleSheet(
            f"color: {color}; font-size: 10px; padding: 2px 0;"
        )
        self._preview_status_label.setText(text)

    def _convert_status(self, text: str, *, error: bool = False) -> None:
        if not hasattr(self, "_convert_status_label"):
            return
        color = "#ef4444" if error else "#9ca3af"
        self._convert_status_label.setStyleSheet(
            f"color: {color}; font-size: 10px; padding: 2px 0;"
        )
        self._convert_status_label.setText(text)

    def _projection_status(self, text: str, *, error: bool = False) -> None:
        if not hasattr(self, "_proj_status_label"):
            return
        color = "#ef4444" if error else "#9ca3af"
        self._proj_status_label.setStyleSheet(
            f"color: {color}; font-size: 10px; padding: 2px 0;"
        )
        self._proj_status_label.setText(text)

    def _translate_status(self, text: str, *, error: bool = False) -> None:
        if not hasattr(self, "_translate_status_label"):
            return
        color = "#ef4444" if error else "#9ca3af"
        self._translate_status_label.setStyleSheet(
            f"color: {color}; font-size: 10px; padding: 2px 0;"
        )
        self._translate_status_label.setText(text)
