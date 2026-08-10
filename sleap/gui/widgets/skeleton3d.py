"""Widget for viewing 3D skeleton reconstructions from points3D HDF5 files."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
from qtpy import QtCore, QtGui, QtWidgets

import matplotlib
import os

if os.environ.get("MPLBACKEND") != "Agg":
    try:
        matplotlib.use("QtAgg")
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as Canvas
        from matplotlib.backends.backend_qtagg import (
            NavigationToolbar2QT as NavigationToolbar,
        )
    except ImportError:
        matplotlib.use("Agg")
        from matplotlib.backends.backend_agg import FigureCanvasAgg as Canvas
        NavigationToolbar = None
else:
    from matplotlib.backends.backend_agg import FigureCanvasAgg as Canvas
    NavigationToolbar = None

from matplotlib.figure import Figure

from sleap.gui.dialogs.filedialog import FileDialog
from sleap.gui.reach_projection import load_points3d_h5


DEFAULT_3D_AXIS_LIMITS = (
    np.asarray([-20.0, -20.0, -180.0], dtype=np.float64),
    np.asarray([20.0, 20.0, -100.0], dtype=np.float64),
)
DEFAULT_3D_VIEW = {"elev": 70.0, "azim": 30.0, "roll": 120.0}


class CameraTransformGizmo(QtWidgets.QWidget):
    """Draggable XYZ camera translation and rotation overlay."""

    cameraDelta = QtCore.Signal(str, str, float)
    resetRequested = QtCore.Signal()

    _COLORS = {
        "x": QtGui.QColor("#ff3b30"),
        "y": QtGui.QColor("#20c95a"),
        "z": QtGui.QColor("#6f78ff"),
    }
    _AXIS_VECTORS = {
        "x": QtCore.QPointF(46.0, 24.0),
        "y": QtCore.QPointF(-42.0, 26.0),
        "z": QtCore.QPointF(0.0, -52.0),
    }
    _RING_ANGLES = {"x": 62.0, "y": -62.0, "z": 0.0}

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self.setFixedSize(150, 150)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        self.setMouseTracking(True)
        self.setToolTip(
            "Drag an arrow to pan along X/Y/Z. Drag a ring to rotate the 3D "
            "camera. Double-click to reset the view."
        )
        self._active_handle: Optional[Tuple[str, str]] = None
        self._last_pos = QtCore.QPointF()

    def paintEvent(self, event) -> None:
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        center = self._center()

        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(QtGui.QColor(15, 23, 42, 155))
        painter.drawEllipse(self.rect().adjusted(4, 4, -4, -4))
        painter.setBrush(QtCore.Qt.NoBrush)
        painter.setPen(QtGui.QPen(QtGui.QColor(203, 213, 225, 150), 2.0))
        painter.drawEllipse(self.rect().adjusted(7, 7, -7, -7))

        for axis, path in self._ring_paths().items():
            color = self._handle_color("rotate", axis)
            painter.setPen(QtGui.QPen(color, 4.0))
            painter.drawPath(path)

        for axis, vector in self._AXIS_VECTORS.items():
            color = self._handle_color("move", axis)
            endpoint = center + vector
            painter.setPen(
                QtGui.QPen(
                    color, 4.0, QtCore.Qt.SolidLine, QtCore.Qt.RoundCap
                )
            )
            painter.drawLine(center, endpoint)
            self._draw_arrowhead(painter, endpoint, vector, color)
            self._draw_axis_label(painter, endpoint, vector, color, axis.upper())

        painter.setPen(QtGui.QPen(QtGui.QColor(226, 232, 240), 1.0))
        painter.setBrush(QtGui.QColor(30, 41, 59))
        painter.drawEllipse(center, 5.0, 5.0)

    def mousePressEvent(self, event) -> None:
        if event.button() != QtCore.Qt.LeftButton:
            event.ignore()
            return
        handle = self._handle_at(QtCore.QPointF(event.pos()))
        if handle is None:
            event.ignore()
            return
        self._active_handle = handle
        self._last_pos = QtCore.QPointF(event.pos())
        self.setCursor(QtCore.Qt.ClosedHandCursor)
        self.update()
        event.accept()

    def mouseMoveEvent(self, event) -> None:
        pos = QtCore.QPointF(event.pos())
        if self._active_handle is None:
            self.setCursor(
                QtCore.Qt.OpenHandCursor
                if self._handle_at(pos) is not None
                else QtCore.Qt.ArrowCursor
            )
            return

        operation, axis = self._active_handle
        delta = pos - self._last_pos
        self._last_pos = pos
        if operation == "move":
            vector = self._AXIS_VECTORS[axis]
            length = max(1.0, np.hypot(vector.x(), vector.y()))
            amount = (delta.x() * vector.x() + delta.y() * vector.y()) / length
        else:
            amount = delta.x() - delta.y()
        if amount:
            self.cameraDelta.emit(operation, axis, float(amount))
        event.accept()

    def mouseReleaseEvent(self, event) -> None:
        if self._active_handle is None:
            event.ignore()
            return
        self._active_handle = None
        self.setCursor(QtCore.Qt.OpenHandCursor)
        self.update()
        event.accept()

    def mouseDoubleClickEvent(self, event) -> None:
        if event.button() == QtCore.Qt.LeftButton:
            self.resetRequested.emit()
            event.accept()
        else:
            event.ignore()

    def wheelEvent(self, event) -> None:
        event.ignore()

    def _center(self) -> QtCore.QPointF:
        return QtCore.QPointF(self.width() / 2.0, self.height() / 2.0)

    def _ring_paths(self) -> dict:
        paths = {}
        center = self._center()
        base = QtGui.QPainterPath()
        base.addEllipse(QtCore.QRectF(-48.0, -18.0, 96.0, 36.0))
        for axis, angle in self._RING_ANGLES.items():
            transform = QtGui.QTransform()
            transform.translate(center.x(), center.y())
            transform.rotate(angle)
            paths[axis] = transform.map(base)
        return paths

    def _handle_at(self, pos: QtCore.QPointF) -> Optional[Tuple[str, str]]:
        center = self._center()
        for axis, vector in self._AXIS_VECTORS.items():
            if _distance_to_segment(pos, center, center + vector) <= 9.0:
                return "move", axis

        stroker = QtGui.QPainterPathStroker()
        stroker.setWidth(12.0)
        for axis, path in self._ring_paths().items():
            if stroker.createStroke(path).contains(pos):
                return "rotate", axis
        return None

    def _handle_color(self, operation: str, axis: str) -> QtGui.QColor:
        color = QtGui.QColor(self._COLORS[axis])
        if self._active_handle == (operation, axis):
            return color.lighter(155)
        return color

    @staticmethod
    def _draw_arrowhead(
        painter: QtGui.QPainter,
        endpoint: QtCore.QPointF,
        vector: QtCore.QPointF,
        color: QtGui.QColor,
    ) -> None:
        length = max(1.0, np.hypot(vector.x(), vector.y()))
        unit = QtCore.QPointF(vector.x() / length, vector.y() / length)
        perpendicular = QtCore.QPointF(-unit.y(), unit.x())
        base = endpoint - unit * 12.0
        polygon = QtGui.QPolygonF(
            [
                endpoint,
                base + perpendicular * 6.0,
                base - perpendicular * 6.0,
            ]
        )
        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(color)
        painter.drawPolygon(polygon)

    @staticmethod
    def _draw_axis_label(
        painter: QtGui.QPainter,
        endpoint: QtCore.QPointF,
        vector: QtCore.QPointF,
        color: QtGui.QColor,
        label: str,
    ) -> None:
        length = max(1.0, np.hypot(vector.x(), vector.y()))
        unit = QtCore.QPointF(vector.x() / length, vector.y() / length)
        label_center = endpoint + unit * 10.0
        font = painter.font()
        font.setBold(True)
        font.setPixelSize(11)
        painter.setFont(font)
        painter.setPen(color.lighter(135))
        painter.drawText(
            QtCore.QRectF(label_center.x() - 8.0, label_center.y() - 8.0, 16.0, 16.0),
            QtCore.Qt.AlignCenter,
            label,
        )


class Skeleton3DCanvas(Canvas):
    """Matplotlib canvas for plotting one frame of a 3D skeleton."""

    def __init__(self, width: int = 7, height: int = 5, dpi: int = 100):
        self.fig = Figure(figsize=(width, height), dpi=dpi, constrained_layout=True)
        self.axes = None
        self.toolbar = None
        self.camera_gizmo = None
        super().__init__(self.fig)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding
        )
        self.setMinimumSize(520, 380)
        self.mpl_connect("scroll_event", self._on_scroll)
        self.updateGeometry()

    def set_camera_gizmo(self, gizmo: CameraTransformGizmo) -> None:
        """Attach and position a camera gizmo over the canvas."""
        self.camera_gizmo = gizmo
        self._position_camera_gizmo()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._position_camera_gizmo()

    def _position_camera_gizmo(self) -> None:
        if self.camera_gizmo is None:
            return
        margin = 12
        self.camera_gizmo.move(
            margin,
            max(margin, self.height() - self.camera_gizmo.height() - margin),
        )
        self.camera_gizmo.raise_()

    def draw_frame(
        self,
        points: np.ndarray,
        edges: Sequence[Tuple[int, int]],
        frame_idx: int,
        *,
        view_mode: str,
        axis_limits: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        preserve_view: bool = False,
    ) -> None:
        """Draw one frame as either a 3D skeleton or a 2D projection."""
        saved_view = self._current_axis_view(view_mode) if preserve_view else None
        self.fig.clear()
        is_3d = view_mode == "3D"
        if is_3d:
            self.axes = self.fig.add_subplot(111, projection="3d")
        else:
            self.axes = self.fig.add_subplot(111)

        frame_points = points[frame_idx]
        if is_3d:
            self._draw_3d(frame_points, edges, axis_limits)
        else:
            self._draw_2d(frame_points, edges, view_mode, axis_limits)

        if saved_view is not None:
            self._apply_axis_view(saved_view)
        self.axes.set_title(f"Frame {frame_idx + 1}")
        self.axes.grid(True, alpha=0.25)
        self.draw()

    def draw_empty(self, message: str = "Load a points3d.h5 file") -> None:
        """Draw placeholder text before data is loaded."""
        self.fig.clear()
        self.axes = self.fig.add_subplot(111)
        self.axes.text(
            0.5,
            0.5,
            message,
            ha="center",
            va="center",
            transform=self.axes.transAxes,
            color="0.35",
        )
        self.axes.set_axis_off()
        self.draw()

    def _draw_3d(
        self,
        frame_points: np.ndarray,
        edges: Sequence[Tuple[int, int]],
        axis_limits: Optional[Tuple[np.ndarray, np.ndarray]],
    ) -> None:
        for src, dst in edges:
            pts = frame_points[[src, dst], :3]
            if np.all(np.isfinite(pts)):
                self.axes.plot(
                    pts[:, 0],
                    pts[:, 1],
                    pts[:, 2],
                    color="#0097c2",
                    linewidth=3,
                    alpha=0.95,
                )

        finite = np.all(np.isfinite(frame_points[:, :3]), axis=1)
        if np.any(finite):
            self.axes.scatter(
                frame_points[finite, 0],
                frame_points[finite, 1],
                frame_points[finite, 2],
                color="black",
                s=24,
                depthshade=False,
            )

        if axis_limits is not None:
            lo, hi = axis_limits
            self.axes.set_xlim(lo[0], hi[0])
            self.axes.set_ylim(lo[1], hi[1])
            self.axes.set_zlim(lo[2], hi[2])
            try:
                self.axes.set_box_aspect(hi - lo)
            except Exception:
                pass

        self.axes.set_xlabel("x")
        self.axes.set_ylabel("y")
        self.axes.set_zlabel("z")
        _set_3d_view(self.axes, **DEFAULT_3D_VIEW)

    def _draw_2d(
        self,
        frame_points: np.ndarray,
        edges: Sequence[Tuple[int, int]],
        view_mode: str,
        axis_limits: Optional[Tuple[np.ndarray, np.ndarray]],
    ) -> None:
        axis_pairs = {"XY": (0, 1), "XZ": (0, 2), "YZ": (1, 2)}
        x_dim, y_dim = axis_pairs.get(view_mode, (0, 1))

        for src, dst in edges:
            pts = frame_points[[src, dst]][:, [x_dim, y_dim]]
            if np.all(np.isfinite(pts)):
                self.axes.plot(
                    pts[:, 0],
                    pts[:, 1],
                    color="#0097c2",
                    linewidth=3,
                    alpha=0.95,
                )

        finite = np.all(np.isfinite(frame_points[:, [x_dim, y_dim]]), axis=1)
        if np.any(finite):
            self.axes.scatter(
                frame_points[finite, x_dim],
                frame_points[finite, y_dim],
                color="black",
                s=24,
            )

        if axis_limits is not None:
            lo, hi = axis_limits
            self.axes.set_xlim(lo[x_dim], hi[x_dim])
            self.axes.set_ylim(lo[y_dim], hi[y_dim])
        self.axes.set_xlabel("xyz"[x_dim])
        self.axes.set_ylabel("xyz"[y_dim])
        self.axes.set_aspect("equal", adjustable="box")

    def _current_axis_view(self, view_mode: str) -> Optional[dict]:
        """Capture the current axes limits before redrawing a new frame."""
        if self.axes is None:
            return None
        is_3d = view_mode == "3D" and hasattr(self.axes, "get_zlim")
        try:
            view = {
                "view_mode": view_mode,
                "xlim": self.axes.get_xlim(),
                "ylim": self.axes.get_ylim(),
            }
            if is_3d:
                view["zlim"] = self.axes.get_zlim()
                view["elev"] = self.axes.elev
                view["azim"] = self.axes.azim
                view["roll"] = getattr(self.axes, "roll", 0)
            return view
        except Exception:
            return None

    def _apply_axis_view(self, view: dict) -> None:
        """Restore axes limits after redrawing the next frame."""
        self.axes.set_xlim(view["xlim"])
        self.axes.set_ylim(view["ylim"])
        if view.get("view_mode") == "3D" and "zlim" in view:
            self.axes.set_zlim(view["zlim"])
            _set_3d_view(
                self.axes,
                elev=view.get("elev", DEFAULT_3D_VIEW["elev"]),
                azim=view.get("azim", DEFAULT_3D_VIEW["azim"]),
                roll=view.get("roll", DEFAULT_3D_VIEW["roll"]),
            )

    def _on_scroll(self, event) -> None:
        """Zoom with the scroll wheel while the toolbar zoom tool is active."""
        if self.axes is None or not self._toolbar_zoom_is_active():
            return

        scale = 0.85 if event.step > 0 else 1.15
        if hasattr(self.axes, "get_zlim"):
            self._scale_limits_3d(scale)
        else:
            self._scale_limits_2d(scale, event.xdata, event.ydata)
        self.draw()

    def _toolbar_zoom_is_active(self) -> bool:
        """Return True when the Matplotlib zoom tool is selected."""
        mode = getattr(self.toolbar, "mode", "")
        return "zoom" in str(mode).casefold()

    def _scale_limits_2d(
        self, scale: float, x_center: Optional[float], y_center: Optional[float]
    ) -> None:
        xlim = self.axes.get_xlim()
        ylim = self.axes.get_ylim()
        x_center = np.mean(xlim) if x_center is None else x_center
        y_center = np.mean(ylim) if y_center is None else y_center
        self.axes.set_xlim(_scaled_limits(xlim, x_center, scale))
        self.axes.set_ylim(_scaled_limits(ylim, y_center, scale))

    def _scale_limits_3d(self, scale: float) -> None:
        xlim = self.axes.get_xlim()
        ylim = self.axes.get_ylim()
        zlim = self.axes.get_zlim()
        self.axes.set_xlim(_scaled_limits(xlim, np.mean(xlim), scale))
        self.axes.set_ylim(_scaled_limits(ylim, np.mean(ylim), scale))
        self.axes.set_zlim(_scaled_limits(zlim, np.mean(zlim), scale))


class Skeleton3DWidget(QtWidgets.QWidget):
    """Interactive controls for loading and plotting points3D skeleton data."""

    def __init__(self, main_window=None, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self.main_window = main_window
        self._path: Optional[Path] = None
        self._points: Optional[np.ndarray] = None
        self._plot_points: Optional[np.ndarray] = None
        self._node_names: List[str] = []
        self._edges: List[Tuple[int, int]] = []
        self._edge_source = ""
        self._axis_limits = None
        self._saved_view = None
        self._syncing_slider = False
        self._setup_ui()
        self.canvas.draw_empty()

    def _setup_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        path_row = QtWidgets.QHBoxLayout()
        self.path_edit = QtWidgets.QLineEdit()
        self.path_edit.setReadOnly(True)
        path_row.addWidget(self.path_edit, stretch=1)
        browse_button = QtWidgets.QPushButton("Browse")
        browse_button.clicked.connect(self.load_from_dialog)
        path_row.addWidget(browse_button)
        layout.addLayout(path_row)

        controls = QtWidgets.QHBoxLayout()
        controls.addWidget(QtWidgets.QLabel("View"))
        self.view_combo = QtWidgets.QComboBox()
        self.view_combo.addItems(["3D", "XY", "XZ", "YZ"])
        self.view_combo.currentIndexChanged.connect(self.update_plot)
        controls.addWidget(self.view_combo)

        controls.addWidget(QtWidgets.QLabel("Origin"))
        self.origin_combo = QtWidgets.QComboBox()
        self.origin_combo.currentIndexChanged.connect(self._update_transform)
        controls.addWidget(self.origin_combo, stretch=1)

        controls.addWidget(QtWidgets.QLabel("Align X to"))
        self.align_combo = QtWidgets.QComboBox()
        self.align_combo.currentIndexChanged.connect(self._update_transform)
        controls.addWidget(self.align_combo, stretch=1)

        self.scale_check = QtWidgets.QCheckBox("Scale")
        self.scale_check.setChecked(True)
        self.scale_check.stateChanged.connect(self._update_transform)
        controls.addWidget(self.scale_check)
        layout.addLayout(controls)

        self.canvas = Skeleton3DCanvas()
        self.camera_gizmo = CameraTransformGizmo(self.canvas)
        self.camera_gizmo.cameraDelta.connect(self._apply_camera_gizmo_delta)
        self.camera_gizmo.resetRequested.connect(self._reset_view)
        self.camera_gizmo.hide()
        self.canvas.set_camera_gizmo(self.camera_gizmo)
        if NavigationToolbar is not None:
            self.toolbar = NavigationToolbar(self.canvas, self)
            self.canvas.toolbar = self.toolbar
            layout.addWidget(self.toolbar)
        else:
            self.toolbar = None

        view_row = QtWidgets.QHBoxLayout()
        set_view_button = QtWidgets.QPushButton("Set View")
        set_view_button.clicked.connect(self._set_view)
        view_row.addWidget(set_view_button)
        reset_view_button = QtWidgets.QPushButton("Reset View")
        reset_view_button.clicked.connect(self._reset_view)
        view_row.addWidget(reset_view_button)
        gizmo_help = QtWidgets.QLabel(
            "3D gizmo: drag arrows to pan; drag rings to rotate"
        )
        gizmo_help.setStyleSheet("color: #666;")
        view_row.addWidget(gizmo_help)
        view_row.addStretch()
        layout.addLayout(view_row)

        layout.addWidget(self.canvas, stretch=1)

        slider_row = QtWidgets.QHBoxLayout()
        self.frame_label = QtWidgets.QLabel("Frame 0 / 0")
        slider_row.addWidget(self.frame_label)
        self.frame_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.frame_slider.setRange(0, 0)
        self.frame_slider.valueChanged.connect(self._on_slider_changed)
        slider_row.addWidget(self.frame_slider, stretch=1)
        layout.addLayout(slider_row)

        self.status_label = QtWidgets.QLabel("")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)
        self.transform_note_label = QtWidgets.QLabel(
            "Note: plotted y and z values are negated from the raw points3d.h5 import."
        )
        self.transform_note_label.setWordWrap(True)
        self.transform_note_label.setStyleSheet("color: #666;")
        layout.addWidget(self.transform_note_label)

    def load_from_dialog(self) -> None:
        """Open a file picker and load a points3D HDF5 file."""
        filename, _ = FileDialog.open(
            self,
            caption="Select points3D HDF5 file",
            dir=str(self._path.parent) if self._path is not None else None,
            filter="HDF5 Files (*.h5 *.hdf5);;All Files (*)",
        )
        if filename:
            try:
                self.load_file(filename)
            except Exception as exc:
                self.status_label.setText(f"Could not load file: {exc}")
                QtWidgets.QMessageBox.warning(
                    self,
                    "Skeleton3D",
                    f"Could not load points3D file:\n{exc}",
                )

    def load_file(self, filename: str | Path) -> None:
        """Load points3D data from an HDF5 file."""
        loaded = load_points3d_h5(filename)
        points = np.asarray(loaded["points3d"], dtype=np.float64)
        if points.ndim != 3 or points.shape[-1] != 3:
            raise ValueError(f"Expected points with shape frames x nodes x 3, got {points.shape}.")

        self._path = Path(filename)
        h5_node_names = list(loaded["node_names"])
        (
            self._node_names,
            node_indices,
            missing_nodes,
            mapping_note,
        ) = self._project_node_mapping(h5_node_names)
        self._points = points[:, node_indices, :]
        self._edges, self._edge_source = self._load_edges(self._node_names)
        self.path_edit.setText(str(self._path))
        self._populate_node_controls()

        current_frame = 0
        if self.main_window is not None:
            current_frame = int(self.main_window.state.get("frame_idx", default=0) or 0)
        self._syncing_slider = True
        self.frame_slider.setRange(0, max(0, points.shape[0] - 1))
        self.frame_slider.setValue(min(current_frame, points.shape[0] - 1))
        self._syncing_slider = False
        self._update_transform()

        edge_note = (
            f"{len(self._edges)} edge(s) from {self._edge_source}"
            if self._edges
            else "no matching edges"
        )
        missing_note = (
            f" Missing project node(s) in file: {', '.join(missing_nodes)}."
            if missing_nodes
            else ""
        )
        self.status_label.setText(
            f"Loaded {self._path.name}: {points.shape[0]} frame(s), "
            f"{len(self._node_names)}/{points.shape[1]} project node(s), "
            f"{edge_note}.{missing_note}{mapping_note}"
        )

    def set_frame(self, frame_idx: int) -> None:
        """Update the plot to a frame index from the main GUI state."""
        if self._points is None:
            return
        frame_idx = max(0, min(int(frame_idx or 0), self._points.shape[0] - 1))
        if self.frame_slider.value() == frame_idx:
            return
        self._syncing_slider = True
        self.frame_slider.setValue(frame_idx)
        self._syncing_slider = False
        self.update_plot(preserve_view=True)

    def _populate_node_controls(self) -> None:
        self.origin_combo.blockSignals(True)
        self.align_combo.blockSignals(True)
        self.origin_combo.clear()
        self.align_combo.clear()
        self.origin_combo.addItem("None", None)
        self.align_combo.addItem("None", None)
        for idx, name in enumerate(self._node_names):
            self.origin_combo.addItem(name, idx)
            self.align_combo.addItem(name, idx)

        origin_idx = self._find_node_index(("bar_R", "bar_R".lower(), "right_bar"))
        align_idx = self._find_node_index(("bar_L", "bar_L".lower(), "left_bar"))
        if origin_idx is not None:
            self.origin_combo.setCurrentIndex(origin_idx + 1)
        if align_idx is not None:
            self.align_combo.setCurrentIndex(align_idx + 1)
        self.origin_combo.blockSignals(False)
        self.align_combo.blockSignals(False)

    def _find_node_index(self, names: Sequence[str]) -> Optional[int]:
        lowered = {name.casefold(): idx for idx, name in enumerate(self._node_names)}
        for name in names:
            if name.casefold() in lowered:
                return lowered[name.casefold()]
        return None

    def _project_node_mapping(
        self, h5_node_names: List[str]
    ) -> Tuple[List[str], List[int], List[str], str]:
        """Return project node names and matching H5 indices to plot."""
        skeleton = self._project_skeleton()
        project_names = list(getattr(skeleton, "node_names", []) or [])
        if not project_names:
            return h5_node_names, list(range(len(h5_node_names))), [], ""

        is_generic = all(
            name == f"node_{idx}" for idx, name in enumerate(h5_node_names)
        )
        if is_generic:
            n_nodes = min(len(project_names), len(h5_node_names))
            return (
                project_names[:n_nodes],
                list(range(n_nodes)),
                project_names[n_nodes:],
                "",
            )

        h5_name_to_idx = {name.casefold(): idx for idx, name in enumerate(h5_node_names)}
        node_names = []
        node_indices = []
        missing_nodes = []
        for name in project_names:
            if name.casefold() in h5_name_to_idx:
                node_names.append(name)
                node_indices.append(h5_name_to_idx[name.casefold()])
            else:
                missing_nodes.append(name)

        if node_indices:
            return node_names, node_indices, missing_nodes, ""

        n_nodes = min(len(project_names), len(h5_node_names))
        return (
            project_names[:n_nodes],
            list(range(n_nodes)),
            project_names[n_nodes:],
            " H5 node names did not match the project, so nodes were mapped by order.",
        )

    def _project_skeleton(self):
        if self.main_window is None:
            return None
        return self.main_window.state.get("skeleton", default=None)

    def _project_skeleton_edges(
        self, node_names: Sequence[str]
    ) -> List[Tuple[int, int]]:
        skeleton = self._project_skeleton()
        project_names = list(getattr(skeleton, "node_names", []) or [])
        edge_inds = list(getattr(skeleton, "edge_inds", []) or [])
        plot_node_index = {name: idx for idx, name in enumerate(node_names)}
        edges = []
        for src, dst in edge_inds:
            src = int(src)
            dst = int(dst)
            if (
                src < 0
                or dst < 0
                or src >= len(project_names)
                or dst >= len(project_names)
            ):
                continue
            src_name = project_names[src]
            dst_name = project_names[dst]
            if src_name in plot_node_index and dst_name in plot_node_index:
                edges.append((plot_node_index[src_name], plot_node_index[dst_name]))
        return edges

    def _load_edges(
        self, node_names: Sequence[str]
    ) -> Tuple[List[Tuple[int, int]], str]:
        """Return only project skeleton edges that match the plotted nodes."""
        return self._project_skeleton_edges(node_names), "project skeleton"

    def _update_transform(self, *args) -> None:
        if self._points is None:
            return
        self._plot_points = self._transform_points()
        self._plot_points[:, :, 1] *= -1
        self._plot_points[:, :, 2] *= -1
        self._axis_limits = self._compute_axis_limits(self._plot_points)
        self.update_plot()

    def _transform_points(self) -> np.ndarray:
        points = np.asarray(self._points, dtype=np.float64)
        transformed = points.copy()
        origin_idx = self.origin_combo.currentData()
        align_idx = self.align_combo.currentData()

        if origin_idx is None:
            return transformed

        origin_trace = points[:, int(origin_idx), :3]
        finite_origin = np.all(np.isfinite(origin_trace), axis=1)
        if not np.any(finite_origin):
            return transformed
        origin = np.nanmean(origin_trace[finite_origin], axis=0)
        transformed = transformed - origin[None, None, :]

        if align_idx is None or align_idx == origin_idx:
            return transformed

        align_trace = points[:, int(align_idx), :3]
        finite_align = np.all(np.isfinite(align_trace), axis=1)
        finite = finite_origin & finite_align
        if not np.any(finite):
            return transformed

        align = np.nanmean(align_trace[finite], axis=0)
        direction = align - origin
        line_length = float(np.linalg.norm(direction))
        if line_length <= 0:
            return transformed

        rotation = _rotation_to_x_axis(direction)
        transformed = np.einsum("ij,tnj->tni", rotation, transformed)
        transformed = np.einsum("ij,tnj->tni", _rotation_x_180(), transformed)
        if self.scale_check.isChecked():
            transformed = transformed / line_length
        return transformed

    def _compute_axis_limits(self, points: np.ndarray):
        return DEFAULT_3D_AXIS_LIMITS

    def _on_slider_changed(self, frame_idx: int) -> None:
        if self._syncing_slider:
            return
        self.update_plot(preserve_view=True)
        if self.main_window is not None:
            if self.main_window.state.get("frame_idx", default=None) != frame_idx:
                self.main_window.state["frame_idx"] = int(frame_idx)

    def _set_view(self) -> None:
        """Save the current zoom and 3D camera view."""
        self._saved_view = self.canvas._current_axis_view(self.view_combo.currentText())
        if self._saved_view is not None:
            self.status_label.setText("Skeleton3D view saved.")

    def _reset_view(self) -> None:
        """Restore the saved view, or the default 3D view if none was saved."""
        view = self._saved_view or _default_axis_view(self.view_combo.currentText())
        if view is None:
            return

        saved_mode = view.get("view_mode")
        if saved_mode and saved_mode != self.view_combo.currentText():
            idx = self.view_combo.findText(saved_mode)
            if idx >= 0:
                self.view_combo.blockSignals(True)
                self.view_combo.setCurrentIndex(idx)
                self.view_combo.blockSignals(False)

        self.update_plot(preserve_view=False)
        if self.canvas.axes is not None:
            self.canvas._apply_axis_view(view)
            self.canvas.draw()

    def _apply_camera_gizmo_delta(
        self, operation: str, axis: str, amount: float
    ) -> None:
        """Apply one incremental gizmo drag to the current 3D camera view."""
        axes = self.canvas.axes
        if (
            axes is None
            or self.view_combo.currentText() != "3D"
            or not hasattr(axes, "get_zlim")
        ):
            return

        if operation == "move":
            getters = {
                "x": axes.get_xlim,
                "y": axes.get_ylim,
                "z": axes.get_zlim,
            }
            setters = {
                "x": axes.set_xlim,
                "y": axes.set_ylim,
                "z": axes.set_zlim,
            }
            limits = getters[axis]()
            shift = float(amount) * abs(float(limits[1]) - float(limits[0])) * 0.006
            setters[axis](float(limits[0]) + shift, float(limits[1]) + shift)
        elif operation == "rotate":
            elev = float(getattr(axes, "elev", DEFAULT_3D_VIEW["elev"]))
            azim = float(getattr(axes, "azim", DEFAULT_3D_VIEW["azim"]))
            roll = float(getattr(axes, "roll", DEFAULT_3D_VIEW["roll"]))
            angle = float(amount) * 0.6
            if axis == "x":
                elev += angle
            elif axis == "y":
                roll += angle
            elif axis == "z":
                azim += angle
            _set_3d_view(axes, elev=elev, azim=azim, roll=roll)
        else:
            return

        self.canvas.draw_idle()

    def update_plot(self, *args, preserve_view: bool = False) -> None:
        if self._plot_points is None:
            self.camera_gizmo.hide()
            self.canvas.draw_empty()
            self.frame_label.setText("Frame 0 / 0")
            return
        frame_idx = int(self.frame_slider.value())
        total = self._plot_points.shape[0]
        self.frame_label.setText(f"Frame {frame_idx + 1} / {total}")
        self.camera_gizmo.setVisible(self.view_combo.currentText() == "3D")
        self.canvas.draw_frame(
            self._plot_points,
            self._edges,
            frame_idx,
            view_mode=self.view_combo.currentText(),
            axis_limits=self._axis_limits,
            preserve_view=preserve_view,
        )


class Skeleton3DDialog(QtWidgets.QDialog):
    """Non-modal popup window for viewing 3D skeleton reconstructions."""

    def __init__(self, main_window=None, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("Skeleton3D")
        self.setModal(False)
        self.setMinimumSize(760, 620)
        self.resize(900, 700)

        layout = QtWidgets.QVBoxLayout(self)
        self.widget = Skeleton3DWidget(main_window=main_window, parent=self)
        layout.addWidget(self.widget)

        button_box = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close)
        button_box.rejected.connect(self.close)
        layout.addWidget(button_box)

    def load_file(self, filename: str | Path) -> None:
        self.widget.load_file(filename)

    def set_frame(self, frame_idx: int) -> None:
        self.widget.set_frame(frame_idx)


def _distance_to_segment(
    point: QtCore.QPointF,
    start: QtCore.QPointF,
    end: QtCore.QPointF,
) -> float:
    """Return the 2D distance between a point and a finite line segment."""
    segment_x = end.x() - start.x()
    segment_y = end.y() - start.y()
    length_squared = segment_x**2 + segment_y**2
    if length_squared <= 0:
        return float(np.hypot(point.x() - start.x(), point.y() - start.y()))
    projection = (
        (point.x() - start.x()) * segment_x
        + (point.y() - start.y()) * segment_y
    ) / length_squared
    projection = float(np.clip(projection, 0.0, 1.0))
    nearest_x = start.x() + projection * segment_x
    nearest_y = start.y() + projection * segment_y
    return float(np.hypot(point.x() - nearest_x, point.y() - nearest_y))


def _rotation_to_x_axis(direction: np.ndarray) -> np.ndarray:
    """Return a rotation matrix that aligns a vector to the positive x-axis."""
    direction = np.asarray(direction, dtype=np.float64)
    norm = np.linalg.norm(direction)
    if norm <= 0:
        return np.eye(3)
    source = direction / norm
    target = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
    cross = np.cross(source, target)
    dot = float(np.clip(np.dot(source, target), -1.0, 1.0))
    cross_norm = np.linalg.norm(cross)

    if cross_norm < 1e-12:
        if dot > 0:
            return np.eye(3)
        return np.asarray(
            [
                [-1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    axis = cross / cross_norm
    angle = np.arccos(dot)
    kx, ky, kz = axis
    k = np.asarray(
        [
            [0.0, -kz, ky],
            [kz, 0.0, -kx],
            [-ky, kx, 0.0],
        ],
        dtype=np.float64,
    )
    return np.eye(3) + np.sin(angle) * k + (1.0 - np.cos(angle)) * (k @ k)


def _rotation_x_180() -> np.ndarray:
    """Return a 180-degree rotation matrix around the x-axis."""
    theta = np.pi
    return np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(theta), -np.sin(theta)],
            [0.0, np.sin(theta), np.cos(theta)],
        ],
        dtype=np.float64,
    )


def _set_3d_view(axes, *, elev: float, azim: float, roll: float) -> None:
    """Set the 3D camera view with roll when Matplotlib supports it."""
    try:
        axes.view_init(elev=elev, azim=azim, roll=roll)
    except TypeError:
        axes.view_init(elev=elev, azim=azim)


def _scaled_limits(limits: Sequence[float], center: float, scale: float) -> Tuple[float, float]:
    """Scale axis limits around a center point."""
    lower, upper = float(limits[0]), float(limits[1])
    return (
        center + (lower - center) * scale,
        center + (upper - center) * scale,
    )


def _default_axis_view(view_mode: str) -> Optional[dict]:
    """Return the default zoom/camera view for the selected display mode."""
    lo, hi = DEFAULT_3D_AXIS_LIMITS
    view = {
        "view_mode": view_mode,
        "xlim": (float(lo[0]), float(hi[0])),
        "ylim": (float(lo[1]), float(hi[1])),
    }
    if view_mode == "3D":
        view.update(
            {
                "zlim": (float(lo[2]), float(hi[2])),
                "elev": DEFAULT_3D_VIEW["elev"],
                "azim": DEFAULT_3D_VIEW["azim"],
                "roll": DEFAULT_3D_VIEW["roll"],
            }
        )
        return view

    axis_pairs = {"XY": (0, 1), "XZ": (0, 2), "YZ": (1, 2)}
    x_dim, y_dim = axis_pairs.get(view_mode, (0, 1))
    view["xlim"] = (float(lo[x_dim]), float(hi[x_dim]))
    view["ylim"] = (float(lo[y_dim]), float(hi[y_dim]))
    return view
