"""Dockable widget for multi-camera calibration using sleap-anipose."""

import logging
import tomllib
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from qtpy import QtCore
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
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
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from sleap.gui.dialogs.filedialog import FileDialog
from sleap.gui.widgets.docks import DockWidget

logger = logging.getLogger(__name__)

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".mpeg", ".mpg", ".mts", ".m4v"}


class CalibrationWorker(QtCore.QThread):
    """Background thread that runs sleap-anipose calibration.

    Accepts video files directly from a flat session folder — no per-camera
    subdirectory or ``calibration_images/`` structure required.
    """

    logOutput = QtCore.Signal(str)
    # finished(success, calib_toml_path, error_message)
    finished = QtCore.Signal(bool, str, str)

    def __init__(
        self,
        video_paths: List[str],
        camera_names: List[str],
        board_dict: Dict,
        output_dir: str,
        parent=None,
    ):
        super().__init__(parent)
        self._video_paths = video_paths
        self._camera_names = camera_names
        self._board_dict = board_dict
        self._output_dir = output_dir
        self._canceled = False

    def cancel(self) -> None:
        self._canceled = True

    @staticmethod
    def _row_corner_count(row: Dict) -> int:
        ids = row.get("ids", None)
        if ids is not None:
            try:
                return int(ids.size)
            except AttributeError:
                return len(ids)
        corners = row.get("corners", None)
        return 0 if corners is None else len(corners)

    @staticmethod
    def _row_corner_xy(row: Dict) -> Optional[np.ndarray]:
        """Return an ``(N, 2)`` array of a row's detected corner coordinates."""
        corners = row.get("corners", None)
        if corners is None:
            return None
        pts = np.asarray(corners, dtype="float64").reshape(-1, 2)
        pts = pts[~np.isnan(pts).any(axis=1)]
        return pts if len(pts) else None

    @classmethod
    def _row_is_degenerate(
        cls,
        row: Dict,
        *,
        min_points: int = 4,
        min_spread_ratio: float = 0.02,
    ) -> bool:
        """Return True if a row's corners are (near-)collinear.

        ``cv2.initCameraMatrix2D`` estimates a board→image homography for every
        frame it is handed and asserts the result is a 3×3 matrix.  A frame
        whose detected ChArUco corners all fall on a single board row or column
        produces an empty homography and aborts the whole calibration with a
        raw OpenCV assertion (``matH0.size() == Size(3, 3)``).  Such frames hold
        no usable calibration information, so they are dropped up front.

        Rows with fewer than ``min_points`` corners are left for the existing
        corner-count checks to handle and are not reported as collinear here.
        """
        pts = cls._row_corner_xy(row)
        if pts is None or len(pts) < min_points:
            return False
        centered = pts - pts.mean(axis=0)
        singular_values = np.linalg.svd(centered, compute_uv=False)
        if singular_values[0] <= 0:
            return True
        return bool(singular_values[1] / singular_values[0] < min_spread_ratio)

    @classmethod
    def _filter_degenerate_rows(cls, all_rows: List[List[Dict]]) -> tuple:
        """Drop (near-)collinear board detections that would break calibration.

        Returns ``(filtered_rows, dropped_per_camera)`` where ``filtered_rows``
        mirrors ``all_rows`` with degenerate rows removed and
        ``dropped_per_camera`` is the per-camera count of removed rows.
        """
        filtered: List[List[Dict]] = []
        dropped_per_camera: List[int] = []
        for rows in all_rows:
            kept = [row for row in rows if not cls._row_is_degenerate(row)]
            filtered.append(kept)
            dropped_per_camera.append(len(rows) - len(kept))
        return filtered, dropped_per_camera

    @classmethod
    def _validate_rows(
        cls,
        all_rows: List[List[Dict]],
        camera_names: List[str],
        dropped_per_camera: Optional[List[int]] = None,
    ) -> None:
        """Raise a readable error if board detection found no usable samples."""
        if dropped_per_camera is None:
            dropped_per_camera = [0] * len(camera_names)

        def _collinear_hint(name: str) -> str:
            dropped = dropped_per_camera[camera_names.index(name)]
            if not dropped:
                return ""
            return (
                f" ({dropped} near-collinear detection(s) were ignored — show "
                "the board tilted and fully within the frame)"
            )

        counts = [len(rows) for rows in all_rows]
        if sum(counts) == 0:
            extra = (
                " Every detection was too close to collinear to use."
                if sum(dropped_per_camera) > 0
                else ""
            )
            raise ValueError(
                "No usable ChArUco boards were detected in any calibration "
                "video." + extra + " Check that the videos open correctly, the "
                "board parameters match your printed board, and the board is "
                "visible — tilted and fully in frame — in the selected videos."
            )

        missing = [
            name for name, rows in zip(camera_names, all_rows)
            if len(rows) == 0
        ]
        if missing:
            raise ValueError(
                "No usable ChArUco boards were detected for camera(s): "
                + ", ".join(name + _collinear_hint(name) for name in missing)
                + ". Each selected camera needs at least one usable board "
                "detection."
            )

        weak = [
            name
            for name, rows in zip(camera_names, all_rows)
            if not any(cls._row_corner_count(row) >= 9 for row in rows)
        ]
        if weak:
            raise ValueError(
                "Detected boards did not have enough non-collinear ChArUco "
                "corners for intrinsic calibration in camera(s): "
                + ", ".join(name + _collinear_hint(name) for name in weak)
                + ". Use clearer frames with the board tilted and fully "
                "visible, or verify board_x, board_y, marker_bits, and "
                "dict_size."
            )

    def run(self) -> None:
        try:
            from aniposelib.boards import CharucoBoard
            from aniposelib.cameras import CameraGroup

            from sleap_anipose.calibration import get_metadata

            n = len(self._video_paths)
            self.logOutput.emit(f"Setting up calibration with {n} camera(s)…")
            for name, path in zip(self._camera_names, self._video_paths):
                self.logOutput.emit(f"  {name}  →  {Path(path).name}")

            if self._canceled:
                raise InterruptedError("Calibration canceled.")

            cgroup = CameraGroup.from_names(self._camera_names)

            bd = self._board_dict
            board = CharucoBoard(
                bd["board_x"],
                bd["board_y"],
                bd["square_length"],
                bd["marker_length"],
                bd["marker_bits"],
                bd["dict_size"],
            )

            # One list per camera — each list holds the single video path for
            # that camera.  aniposelib accepts multiple videos per camera but
            # we always have exactly one here.
            calib_videos = [[p] for p in self._video_paths]

            self.logOutput.emit(
                "\nDetecting calibration board in videos…"
                "  (this may take several minutes)"
            )
            # Capture aniposelib's "error: <float>" stdout prints and reformat
            # them as readable reprojection-error lines in the log dialog.
            import io
            import sys

            _stdout_buf = io.StringIO()
            _old_stdout = sys.stdout
            sys.stdout = _stdout_buf
            corners = []
            calibration_error = None
            try:
                try:
                    corners = cgroup.get_rows_videos(
                        calib_videos, board, verbose=True
                    )
                    corners, dropped_per_cam = self._filter_degenerate_rows(
                        corners
                    )
                    n_dropped = sum(dropped_per_cam)
                    if n_dropped:
                        self.logOutput.emit(
                            f"Ignored {n_dropped} near-collinear board "
                            "detection(s) unusable for calibration."
                        )
                    self._validate_rows(
                        corners, self._camera_names, dropped_per_cam
                    )
                    cgroup.set_camera_sizes_videos(calib_videos)
                    cgroup.calibrate_rows(corners, board)
                except Exception as exc:
                    calibration_error = exc
            finally:
                sys.stdout = _old_stdout
            for raw_line in _stdout_buf.getvalue().splitlines():
                stripped = raw_line.strip()
                if not stripped:
                    continue
                # aniposelib prints "error: <float>" for reprojection error
                if stripped.lower().startswith("error:"):
                    try:
                        val = float(stripped.split(":", 1)[1].strip())
                        self.logOutput.emit(
                            f"Reprojection error: {val:.4f} px"
                        )
                    except ValueError:
                        self.logOutput.emit(stripped)
                else:
                    self.logOutput.emit(stripped)

            for name, rows in zip(self._camera_names, corners):
                self.logOutput.emit(f"{name}: {len(rows)} board detections")

            if calibration_error is not None:
                raise calibration_error

            if self._canceled:
                raise InterruptedError("Calibration canceled.")

            self.logOutput.emit("Computing reprojection metadata…")
            output_dir = Path(self._output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            metadata_path = str(output_dir / "calibration.metadata.h5")
            get_metadata(corners, cgroup, metadata_path)

            calib_path = str(output_dir / "calibration.toml")
            cgroup.dump(calib_path)

            self.logOutput.emit("\nSaved:")
            self.logOutput.emit(f"  {calib_path}")
            self.logOutput.emit(f"  {metadata_path}")
            self.finished.emit(True, calib_path, "")

        except InterruptedError as exc:
            self.finished.emit(False, "", str(exc))
        except Exception as exc:
            import traceback

            self.logOutput.emit(f"\nError: {exc}")
            self.logOutput.emit(traceback.format_exc())
            self.finished.emit(False, "", str(exc))


class _CalibrationLogDialog(QDialog):
    """Modal progress/log dialog shown while calibration runs."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Running Calibration…")
        self.setMinimumSize(580, 380)

        layout = QVBoxLayout()
        layout.setSpacing(6)

        self._status_label = QLabel("<b>Running calibration…</b>")
        self._status_label.setWordWrap(True)
        layout.addWidget(self._status_label)

        self._log = QPlainTextEdit()
        self._log.setReadOnly(True)
        font = self._log.font()
        font.setFamily("Courier")
        font.setPointSize(9)
        self._log.setFont(font)
        layout.addWidget(self._log)

        self._buttons = QDialogButtonBox()
        self._ok_btn = self._buttons.addButton(QDialogButtonBox.Ok)
        self._cancel_btn = self._buttons.addButton(QDialogButtonBox.Cancel)
        self._ok_btn.setEnabled(False)
        self._buttons.accepted.connect(self.accept)
        self._buttons.rejected.connect(self.reject)
        layout.addWidget(self._buttons)

        self.setLayout(layout)

    def append_log(self, text: str) -> None:
        self._log.appendPlainText(text)

    def mark_done(self, success: bool) -> None:
        self._ok_btn.setEnabled(True)
        self._cancel_btn.setEnabled(False)
        if success:
            self._status_label.setText("<b>Calibration complete!</b>")
        else:
            self._status_label.setText("<b>Calibration failed.</b>  See log above.")


class CalibrationDock(DockWidget):
    """Dockable tab for multi-camera calibration via sleap-anipose.

    The user selects a session folder that contains:
    - Two or more calibration video files (any common container format).
    - Optionally a ``charuco.toml`` describing the calibration board (the
      board fields can also be filled in manually).

    Outputs written to the session folder (or a chosen output directory):
    - ``calibration.toml``  — serialised ``aniposelib.CameraGroup``.
    - ``calibration.metadata.h5`` — detected / triangulated corner data.
    """

    def __init__(
        self,
        main_window: QMainWindow,
        tab_with: Optional[QLayout] = None,
    ):
        super().__init__(
            name="Calibration",
            main_window=main_window,
            model_type=None,
            tab_with=tab_with,
        )

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

        layout.addWidget(self._build_session_group())
        layout.addWidget(self._build_board_group())
        layout.addWidget(self._build_cameras_group())
        layout.addWidget(self._build_output_group())
        layout.addSpacing(4)

        self._run_btn = QPushButton("Run Calibration")
        self._run_btn.setEnabled(False)
        self._run_btn.setToolTip(
            "Run multi-camera calibration and save calibration.toml and "
            "calibration.metadata.h5 to the output folder."
        )
        self._run_btn.clicked.connect(self._run_calibration)
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

    # ── section builders ─────────────────────────────────────────────────── #

    def _build_session_group(self) -> QGroupBox:
        gb = QGroupBox("Session Folder")
        layout = QVBoxLayout()
        layout.setSpacing(4)

        row = QHBoxLayout()
        self._session_edit = QLineEdit()
        self._session_edit.setPlaceholderText(
            "Folder containing calibration videos and charuco.toml…"
        )
        self._session_edit.textChanged.connect(self._on_session_changed)
        browse_btn = QPushButton("Browse")
        browse_btn.setFixedWidth(70)
        browse_btn.clicked.connect(self._browse_session)
        row.addWidget(self._session_edit)
        row.addWidget(browse_btn)
        layout.addLayout(row)

        self._session_status = QLabel("")
        self._session_status.setWordWrap(True)
        self._session_status.setStyleSheet("color: #9ca3af; font-size: 10px;")
        layout.addWidget(self._session_status)

        gb.setLayout(layout)
        return gb

    def _build_board_group(self) -> QGroupBox:
        gb = QGroupBox("ChArUco Board")
        outer = QVBoxLayout()
        outer.setSpacing(4)

        # Optional path to charuco.toml — auto-detected, can be overridden.
        charuco_row = QHBoxLayout()
        self._charuco_edit = QLineEdit()
        self._charuco_edit.setPlaceholderText(
            "charuco.toml  (auto-detected or browse to select)"
        )
        self._charuco_edit.setReadOnly(True)
        charuco_btn = QPushButton("Browse")
        charuco_btn.setFixedWidth(70)
        charuco_btn.clicked.connect(self._browse_charuco)
        charuco_row.addWidget(self._charuco_edit)
        charuco_row.addWidget(charuco_btn)
        outer.addLayout(charuco_row)

        # Editable board parameters (pre-filled from charuco.toml when found).
        form = QFormLayout()
        form.setSpacing(4)

        self._board_x_spin = QSpinBox()
        self._board_x_spin.setRange(2, 50)
        self._board_x_spin.setValue(5)
        self._board_x_spin.setToolTip("Squares along board width (board_x).")
        form.addRow("Board X:", self._board_x_spin)

        self._board_y_spin = QSpinBox()
        self._board_y_spin.setRange(2, 50)
        self._board_y_spin.setValue(4)
        self._board_y_spin.setToolTip("Squares along board height (board_y).")
        form.addRow("Board Y:", self._board_y_spin)

        self._square_len_spin = QDoubleSpinBox()
        self._square_len_spin.setRange(0.001, 9999.0)
        self._square_len_spin.setDecimals(4)
        self._square_len_spin.setValue(27.5)
        self._square_len_spin.setToolTip("Checkerboard square edge length (any unit).")
        form.addRow("Square length:", self._square_len_spin)

        self._marker_len_spin = QDoubleSpinBox()
        self._marker_len_spin.setRange(0.001, 9999.0)
        self._marker_len_spin.setDecimals(4)
        self._marker_len_spin.setValue(20.0)
        self._marker_len_spin.setToolTip(
            "ArUco marker edge length (same unit as square length)."
        )
        form.addRow("Marker length:", self._marker_len_spin)

        self._marker_bits_spin = QSpinBox()
        self._marker_bits_spin.setRange(4, 7)
        self._marker_bits_spin.setValue(4)
        self._marker_bits_spin.setToolTip("Bits per ArUco marker (4, 5, 6, or 7).")
        form.addRow("Marker bits:", self._marker_bits_spin)

        self._dict_size_spin = QSpinBox()
        self._dict_size_spin.setRange(50, 1000)
        self._dict_size_spin.setValue(50)
        self._dict_size_spin.setToolTip(
            "ArUco dictionary size (50, 100, 250, or 1000)."
        )
        form.addRow("Dict size:", self._dict_size_spin)

        outer.addLayout(form)
        gb.setLayout(outer)
        return gb

    def _build_cameras_group(self) -> QGroupBox:
        gb = QGroupBox("Cameras (detected videos)")
        layout = QVBoxLayout()
        layout.setSpacing(4)

        self._cam_list = QListWidget()
        self._cam_list.setAlternatingRowColors(True)
        self._cam_list.setToolTip(
            "Videos found in the session folder.  Each checked video is used as "
            "one camera view during calibration.  Camera names are derived from "
            "the video filename stem."
        )
        layout.addWidget(self._cam_list)

        btn_row = QHBoxLayout()
        reload_btn = QPushButton("Reload")
        reload_btn.setToolTip("Re-scan the session folder for video files.")
        reload_btn.clicked.connect(self._reload_cameras)
        btn_row.addWidget(reload_btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        gb.setLayout(layout)
        return gb

    def _build_output_group(self) -> QGroupBox:
        gb = QGroupBox("Output")
        layout = QVBoxLayout()
        layout.setSpacing(4)

        lbl = QLabel("Output folder (calibration.toml and .metadata.h5 saved here):")
        lbl.setWordWrap(True)
        layout.addWidget(lbl)

        row = QHBoxLayout()
        self._output_edit = QLineEdit()
        self._output_edit.setPlaceholderText("Defaults to session folder…")
        self._output_edit.textChanged.connect(self._update_run_btn)
        out_btn = QPushButton("Browse")
        out_btn.setFixedWidth(70)
        out_btn.clicked.connect(self._browse_output)
        row.addWidget(self._output_edit)
        row.addWidget(out_btn)
        layout.addLayout(row)

        gb.setLayout(layout)
        return gb

    # ── folder / file browsing ────────────────────────────────────────────── #

    def _browse_session(self) -> None:
        folder = FileDialog.openDir(self, caption="Select calibration session folder")
        if folder:
            self._session_edit.setText(folder)

    def _browse_charuco(self) -> None:
        path, _ = FileDialog.open(
            self,
            caption="Select charuco.toml",
            filter="TOML files (*.toml);;All Files (*)",
        )
        if path:
            self._charuco_edit.setReadOnly(False)
            self._charuco_edit.setText(path)
            self._charuco_edit.setReadOnly(True)
            self._load_charuco_toml(path)

    def _browse_output(self) -> None:
        folder = FileDialog.openDir(self, caption="Select output folder")
        if folder:
            self._output_edit.setText(folder)

    # ── auto-detection logic ─────────────────────────────────────────────── #

    def _on_session_changed(self, folder: str) -> None:
        folder = folder.strip()
        if not folder:
            self._session_status.setText("")
            self._clear_cameras()
            self._update_run_btn()
            return

        p = Path(folder)
        if not p.is_dir():
            self._session_status.setText("Folder not found.")
            self._clear_cameras()
            self._update_run_btn()
            return

        # Auto-detect charuco.toml
        charuco_path = p / "charuco.toml"
        if charuco_path.exists():
            self._charuco_edit.setReadOnly(False)
            self._charuco_edit.setText(str(charuco_path))
            self._charuco_edit.setReadOnly(True)
            self._load_charuco_toml(str(charuco_path))

        # Default output to session folder
        if not self._output_edit.text().strip():
            self._output_edit.setText(folder)

        self._reload_cameras()

    def _reload_cameras(self) -> None:
        """Rescan the session folder and repopulate the camera list."""
        folder = self._session_edit.text().strip()
        self._clear_cameras()
        if not folder:
            return

        p = Path(folder)
        if not p.is_dir():
            return

        videos = sorted(
            f for f in p.iterdir()
            if f.is_file() and f.suffix.lower() in VIDEO_EXTENSIONS
        )

        if not videos:
            self._session_status.setText("No video files found in folder.")
            self._update_run_btn()
            return

        for video in videos:
            item = QListWidgetItem(f"{video.stem}  —  {video.name}")
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked)
            item.setData(Qt.UserRole, str(video))
            item.setToolTip(str(video))
            self._cam_list.addItem(item)

        n = len(videos)
        status_parts = [f"Found {n} video file{'s' if n != 1 else ''}."]
        if self._charuco_edit.text().strip():
            status_parts.append("charuco.toml loaded.")
        self._session_status.setText("  ".join(status_parts))
        self._update_run_btn()

    def _clear_cameras(self) -> None:
        self._cam_list.clear()

    def _load_charuco_toml(self, path: str) -> None:
        """Read charuco.toml and populate the board parameter fields."""
        try:
            with open(path, "rb") as f:
                data = tomllib.load(f)
            if "board_x" in data:
                self._board_x_spin.setValue(int(data["board_x"]))
            if "board_y" in data:
                self._board_y_spin.setValue(int(data["board_y"]))
            if "square_length" in data:
                self._square_len_spin.setValue(float(data["square_length"]))
            if "marker_length" in data:
                self._marker_len_spin.setValue(float(data["marker_length"]))
            if "marker_bits" in data:
                self._marker_bits_spin.setValue(int(data["marker_bits"]))
            if "dict_size" in data:
                self._dict_size_spin.setValue(int(data["dict_size"]))
        except Exception as exc:
            logger.warning("Could not load charuco.toml: %s", exc)

    # ── run-button state ─────────────────────────────────────────────────── #

    def _update_run_btn(self) -> None:
        if not hasattr(self, "_run_btn"):
            return
        enabled = (
            bool(self._session_edit.text().strip())
            and self._checked_camera_count() >= 2
        )
        self._run_btn.setEnabled(enabled)

    def _checked_camera_count(self) -> int:
        if not hasattr(self, "_cam_list"):
            return 0
        return sum(
            1
            for i in range(self._cam_list.count())
            if self._cam_list.item(i).checkState() == Qt.Checked
        )

    def _checked_cameras(self) -> List[tuple]:
        """Return list of (camera_name, video_path) for checked cameras."""
        result = []
        for i in range(self._cam_list.count()):
            item = self._cam_list.item(i)
            if item.checkState() == Qt.Checked:
                path = item.data(Qt.UserRole)
                name = Path(path).stem
                result.append((name, path))
        return result

    # ── calibration ──────────────────────────────────────────────────────── #

    def _run_calibration(self) -> None:
        cameras = self._checked_cameras()
        if len(cameras) < 2:
            QMessageBox.warning(
                self,
                "Not Enough Cameras",
                "Please check at least two camera videos before running calibration.",
            )
            return

        output_dir = (
            self._output_edit.text().strip()
            or self._session_edit.text().strip()
        )
        if not output_dir:
            QMessageBox.warning(
                self, "No Output Folder", "Please set an output folder."
            )
            return

        # Check for duplicate camera names (can happen when two videos share a stem).
        names = [name for name, _ in cameras]
        if len(names) != len(set(names)):
            QMessageBox.warning(
                self,
                "Duplicate Camera Names",
                "Two or more selected videos have the same filename stem, which "
                "would produce duplicate camera names.  Rename the video files so "
                "each has a unique stem.",
            )
            return

        board_dict = {
            "board_x": self._board_x_spin.value(),
            "board_y": self._board_y_spin.value(),
            "square_length": self._square_len_spin.value(),
            "marker_length": self._marker_len_spin.value(),
            "marker_bits": self._marker_bits_spin.value(),
            "dict_size": self._dict_size_spin.value(),
        }

        cam_names = [n for n, _ in cameras]
        video_paths = [p for _, p in cameras]

        dialog = _CalibrationLogDialog(self)

        worker = CalibrationWorker(
            video_paths=video_paths,
            camera_names=cam_names,
            board_dict=board_dict,
            output_dir=output_dir,
            parent=self,
        )
        result: Dict = {"success": False, "calib_path": "", "error": ""}

        def on_log(line: str) -> None:
            dialog.append_log(line)

        def on_finished(success: bool, calib_path: str, error: str) -> None:
            result["success"] = success
            result["calib_path"] = calib_path
            result["error"] = error
            dialog.mark_done(success)

        worker.logOutput.connect(on_log)
        worker.finished.connect(on_finished)
        dialog._cancel_btn.clicked.connect(worker.cancel)

        self._run_btn.setEnabled(False)
        self._set_status("Running calibration…")

        worker.start()
        dialog.exec_()

        if worker.isRunning():
            worker.cancel()
            worker.wait(5000)

        if result["success"]:
            calib_path = result["calib_path"]
            self._set_status(f"Saved {Path(calib_path).name} to {output_dir}")
        else:
            err = result["error"] or "canceled"
            self._set_status(f"Calibration failed: {err}", error=True)

        self._update_run_btn()

    # ── helpers ──────────────────────────────────────────────────────────── #

    def _set_status(self, text: str, *, error: bool = False) -> None:
        color = "#ef4444" if error else "#9ca3af"
        self._status_label.setStyleSheet(
            f"color: {color}; font-size: 10px; padding: 2px 0;"
        )
        self._status_label.setText(text)
