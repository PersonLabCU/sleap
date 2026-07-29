"""Module for creating dock widgets for the `MainWindow`."""

import base64
from io import BytesIO
import json
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Type, Union

from qtpy import QtGui
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QAbstractItemView,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDockWidget,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QLayout,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSpinBox,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from PIL import Image

from sleap.gui.dataviews import (
    GenericTableModel,
    GenericTableView,
    InstancesTableView,
    LabeledFrameTableModel,
    SkeletonEdgesTableModel,
    SkeletonNodeModel,
    SkeletonNodesTableModel,
    SuggestionsTableModel,
    VideosTableModel,
)
from sleap.gui.dialogs.filedialog import FileDialog
from sleap.gui.dialogs.formbuilder import YamlFormWidget
from sleap.gui.commands import TriangulateSessionLabels, UpdateTopic
from sleap.gui.session_events import (
    SESSION_EVENTS_PATH_KEY,
    event_color_map,
    event_names,
    get_session_events,
    get_session_for_video,
    load_session_events_file,
    set_session_events,
)
from sleap.gui.widgets.views import CollapsibleWidget

# from sleap.skeleton import Skeleton, SkeletonDecoder
from sleap_io.io.skeleton import SkeletonDecoder
from sleap.util import find_files_by_suffix, get_package_file



class DockWidget(QDockWidget):
    """'Abstract' class for a dockable widget attached to the `MainWindow`."""

    def __init__(
        self,
        name: str,
        main_window: Optional[QMainWindow] = None,
        model_type: Optional[
            Union[Type[GenericTableModel], List[Type[GenericTableModel]]]
        ] = None,
        widgets: Optional[Iterable[QWidget]] = None,
        tab_with: Optional[QLayout] = None,
    ):
        # Create the dock and add it to the main window.
        super().__init__(name)
        self.name = name
        self.main_window = main_window
        self.setup_dock(widgets, tab_with)

        # Create the model and table for the dock.
        self.model_type = model_type
        if self.model_type is None:
            self.model = None
            self.table = None
        else:
            self.model = self.create_models()
            self.table = self.create_tables()

        # Lay out the dock widget, adding/creating other widgets if needed.
        self.lay_everything_out()

    @property
    def wgt_layout(self) -> QVBoxLayout:
        return self.widget().layout()

    def setup_dock(self, widgets, tab_with):
        """Create a dock widget.

        Args:
            widgets: The widgets to add to the dock.
            tab_with: The `QLayout` to tabify the `DockWidget` with.
        """

        self.setObjectName(self.name + "Dock")
        self.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)

        dock_widget = QWidget()
        dock_widget.setObjectName(self.name + "Widget")
        layout = QVBoxLayout()

        widgets = widgets or []
        for widget in widgets:
            layout.addWidget(widget)

        dock_widget.setLayout(layout)
        self.setWidget(dock_widget)

        self.add_to_window(self.main_window, tab_with)

    def add_to_window(self, main_window: QMainWindow, tab_with: QVBoxLayout):
        """Add the dock to the `MainWindow`.

        Args:
            tab_with: The `QLayout` to tabify the `DockWidget` with.
        """
        self.main_window = main_window
        self.main_window.addDockWidget(Qt.RightDockWidgetArea, self)
        self.main_window.viewMenu.addAction(self.toggleViewAction())

        if tab_with is not None:
            self.main_window.tabifyDockWidget(tab_with, self)

    def add_button(self, to: QLayout, label: str, action: Callable, key=None):
        key = key or label.lower()
        btn = QPushButton(label)
        btn.clicked.connect(action)
        to.addWidget(btn)
        self.main_window._buttons[key] = btn
        return btn

    def create_models(self) -> GenericTableModel:
        """Create the model for the table in the dock (if any).

        Implement this in the subclass.
        Ex:
            self.model = self.model_type(items=[], context=self.main_window.commands)

        Returns:
            The model.
        """
        raise NotImplementedError

    def create_tables(self) -> GenericTableView:
        """Add a table to the dock.

        Implement this in the subclass.
        Ex:
            self.table = GenericTableView(
                state=self.main_window.state,
                model=self.model or self.create_models(),
            )
            self.wgt_layout.addWidget(self.table)

        Returns:
            The table widget.
        """
        raise NotImplementedError

    def lay_everything_out(self) -> None:
        """Lay out the dock widget, adding/creating other widgets if needed.

        Implement this in the subclass. No example as this is extremely custom.
        """
        raise NotImplementedError


class VideosDock(DockWidget):
    """Dock widget for displaying video information."""

    def __init__(
        self,
        main_window: Optional[QMainWindow] = None,
    ):
        super().__init__(
            name="Videos", main_window=main_window, model_type=VideosTableModel
        )

    def create_models(self) -> VideosTableModel:
        self.model = self.model_type(
            items=self.main_window.labels.videos, context=self.main_window.commands
        )
        return self.model

    def create_tables(self) -> GenericTableView:
        if self.model is None:
            self.create_models()

        main_window = self.main_window
        self.table = GenericTableView(
            state=main_window.state,
            row_name="video",
            is_activatable=True,
            model=self.model,
            ellipsis_left=True,
            multiple_selection=True,
        )

        return self.table

    def create_video_edit_and_nav_buttons(self) -> QWidget:
        """Create the buttons for editing and navigating videos in table."""
        main_window = self.main_window

        hb = QHBoxLayout()
        self.add_button(hb, "Toggle Grayscale", main_window.commands.toggleGrayscale)
        self.add_button(hb, "Show Video", self.table.activateSelected)
        self.add_button(hb, "Add Videos", main_window.commands.addVideo)
        self.add_button(hb, "Replace Videos", main_window.commands.replaceVideo)
        self.add_button(hb, "Add Sessions", main_window.commands.addSession)
        self.add_button(hb, "Remove Video", main_window.commands.removeVideo)
        hbw = QWidget()
        hbw.setLayout(hb)
        return hbw

    def lay_everything_out(self):
        """Lay out the dock widget, adding/creating other widgets if needed."""
        self.wgt_layout.addWidget(self.table)

        video_edit_and_nav_buttons = self.create_video_edit_and_nav_buttons()
        self.wgt_layout.addWidget(video_edit_and_nav_buttons)


class SkeletonDock(DockWidget):
    """Dock widget for displaying skeleton information."""

    def __init__(
        self,
        main_window: Optional[QMainWindow] = None,
        tab_with: Optional[QLayout] = None,
    ):
        self.nodes_model_type = SkeletonNodesTableModel
        self.edges_model_type = SkeletonEdgesTableModel
        super().__init__(
            name="Skeleton",
            main_window=main_window,
            model_type=[self.nodes_model_type, self.edges_model_type],
            tab_with=tab_with,
        )

    def create_models(self) -> GenericTableModel:
        main_window = self.main_window
        self.nodes_model = self.nodes_model_type(
            items=main_window.state["skeleton"], context=main_window.commands
        )
        self.edges_model = self.edges_model_type(
            items=main_window.state["skeleton"], context=main_window.commands
        )
        return [self.nodes_model, self.edges_model]

    def create_tables(self) -> GenericTableView:
        if self.model is None:
            self.create_models()

        main_window = self.main_window
        self.nodes_table = GenericTableView(
            state=main_window.state,
            row_name="node",
            model=self.nodes_model,
        )

        self.edges_table = GenericTableView(
            state=main_window.state,
            row_name="edge",
            model=self.edges_model,
        )

        return [self.nodes_table, self.edges_table]

    def create_project_skeleton_groupbox(self) -> QGroupBox:
        """Create the groupbox for the project skeleton."""
        main_window = self.main_window
        gb = QGroupBox("Project Skeleton")
        vgb = QVBoxLayout()

        nodes_widget = QWidget()
        vb = QVBoxLayout()
        graph_tabs = QTabWidget()

        vb.addWidget(self.nodes_table)
        hb = QHBoxLayout()
        self.add_button(hb, "New Node", main_window.commands.newNode)
        self.add_button(hb, "Delete Node", main_window.commands.deleteNode)
        move_up_button = self.add_button(
            hb, "\u2191", lambda: main_window.commands.moveNode(-1), key="move up"
        )
        move_up_button.setToolTip("Move selected node up")
        move_down_button = self.add_button(
            hb, "\u2193", lambda: main_window.commands.moveNode(1), key="move down"
        )
        move_down_button.setToolTip("Move selected node down")

        hbw = QWidget()
        hbw.setLayout(hb)
        vb.addWidget(hbw)
        nodes_widget.setLayout(vb)
        graph_tabs.addTab(nodes_widget, "Nodes")

        def _update_edge_src():
            self.skeletonEdgesDst.model().skeleton = main_window.state["skeleton"]

        edges_widget = QWidget()

        vb = QVBoxLayout()
        vb.addWidget(self.edges_table)

        hb = QHBoxLayout()
        self.skeletonEdgesSrc = QComboBox()
        self.skeletonEdgesSrc.setEditable(False)
        self.skeletonEdgesSrc.currentIndexChanged.connect(_update_edge_src)
        self.skeletonEdgesSrc.setModel(SkeletonNodeModel(main_window.state["skeleton"]))
        hb.addWidget(self.skeletonEdgesSrc)
        hb.addWidget(QLabel("to"))
        self.skeletonEdgesDst = QComboBox()
        self.skeletonEdgesDst.setEditable(False)
        hb.addWidget(self.skeletonEdgesDst)
        self.skeletonEdgesDst.setModel(
            SkeletonNodeModel(
                main_window.state["skeleton"],
                lambda: self.skeletonEdgesSrc.currentText(),
            )
        )

        def new_edge():
            src_node = self.skeletonEdgesSrc.currentText()
            dst_node = self.skeletonEdgesDst.currentText()
            main_window.commands.newEdge(src_node, dst_node)

        self.add_button(hb, "Add Edge", new_edge)
        self.add_button(hb, "Delete Edge", main_window.commands.deleteEdge)
        hbw = QWidget()
        hbw.setLayout(hb)
        vb.addWidget(hbw)
        edges_widget.setLayout(vb)
        graph_tabs.addTab(edges_widget, "Edges")
        vgb.addWidget(graph_tabs)

        hb = QHBoxLayout()
        self.add_button(hb, "Load From File...", main_window.commands.openSkeleton)
        self.add_button(hb, "Save As...", main_window.commands.saveSkeleton)

        hbw = QWidget()
        hbw.setLayout(hb)
        vgb.addWidget(hbw)

        # Add graph tabs to "Project Skeleton" group box
        gb.setLayout(vgb)
        return gb

    def create_templates_groupbox(self) -> QGroupBox:
        """Create the groupbox for the skeleton templates."""
        main_window = self.main_window

        gb = CollapsibleWidget("Templates")
        vb = QVBoxLayout()
        hb = QHBoxLayout()

        skeletons_folder = get_package_file("skeletons")
        skeletons_json_files = find_files_by_suffix(
            skeletons_folder, suffix=".json", depth=1
        )
        skeletons_names = [json.name.split(".")[0] for json in skeletons_json_files]
        self.skeleton_templates = QComboBox()
        self.skeleton_templates.addItems(skeletons_names)
        self.skeleton_templates.setEditable(False)
        hb.addWidget(self.skeleton_templates)
        self.add_button(hb, "Load", main_window.commands.openSkeletonTemplate)
        hbw = QWidget()
        hbw.setLayout(hb)
        vb.addWidget(hbw)

        hb = QHBoxLayout()
        self.skeleton_preview_image = QLabel("Preview Skeleton")
        hb.addWidget(self.skeleton_preview_image)
        hb.setAlignment(self.skeleton_preview_image, Qt.AlignLeft)

        self.skeleton_description = QLabel(
            f"<strong>Description:</strong> {main_window.state['skeleton_description']}"
        )
        self.skeleton_description.setWordWrap(True)
        hb.addWidget(self.skeleton_description)
        hb.setAlignment(self.skeleton_description, Qt.AlignLeft)

        hbw = QWidget()
        hbw.setLayout(hb)
        vb.addWidget(hbw)

        def updatePreviewImage(preview_image_bytes: bytes):
            # Decode the preview image
            preview_image = decode_preview_image(preview_image_bytes)

            # Create a QImage from the Image
            preview_image = QtGui.QImage(
                preview_image.tobytes(),
                preview_image.size[0],
                preview_image.size[1],
                QtGui.QImage.Format_RGBA8888,  # Format for RGBA images (see Image.mode)
            )

            preview_image = QtGui.QPixmap.fromImage(preview_image)

            self.skeleton_preview_image.setPixmap(preview_image)

        def decode_preview_image(img_b64: bytes, return_bytes: bool = False):
            """Decode a skeleton preview img byte string repr to a `PIL.Image`

            Args:
                img_b64: a byte string representation of a skeleton preview image
                return_bytes: whether to return the decoded image as bytes

            Returns:
                Either a PIL.Image of the skeleton preview image or the decoded image
                as bytes
                (if `return_bytes` is True).
            """
            bytes = base64.b64decode(img_b64)
            if return_bytes:
                return bytes

            buffer = BytesIO(bytes)
            img = Image.open(buffer)
            return img

        def update_skeleton_preview(idx: int):
            with open(skeletons_json_files[idx], "r") as f:
                skeleton_data = json.load(f)
            skel = SkeletonDecoder().decode(data=skeleton_data["nx_graph"])
            description = f"{skel.name}"
            main_window.state["skeleton_description"] = (
                f"<strong>Skeleton name:</strong> {description}<br><br>"
                f"<strong>Nodes({len(skel.node_names)}):</strong>"
                f"{', '.join(skel.node_names)}"
            )
            self.skeleton_description.setText(main_window.state["skeleton_description"])
            updatePreviewImage(
                decode_preview_image(
                    skeleton_data["preview_image"]["py/b64"], return_bytes=True
                )
            )

        self.skeleton_templates.currentIndexChanged.connect(update_skeleton_preview)
        update_skeleton_preview(idx=0)

        gb.set_content_layout(vb)
        return gb

    def lay_everything_out(self):
        """Lay out the dock widget, adding/creating other widgets if needed."""
        templates_gb = self.create_templates_groupbox()
        self.wgt_layout.addWidget(templates_gb)

        project_skeleton_groupbox = self.create_project_skeleton_groupbox()
        self.wgt_layout.addWidget(project_skeleton_groupbox)


class SuggestionsDock(DockWidget):
    """Dock widget for displaying suggestions."""

    def __init__(self, main_window: QMainWindow, tab_with: Optional[QLayout] = None):
        super().__init__(
            name="Labeling Suggestions",
            main_window=main_window,
            model_type=SuggestionsTableModel,
            tab_with=tab_with,
        )

    def create_models(self) -> SuggestionsTableModel:
        self.model = self.model_type(
            items=self.main_window.labels.suggestions, context=self.main_window.commands
        )
        return self.model

    def create_tables(self) -> GenericTableView:
        self.table = GenericTableView(
            state=self.main_window.state,
            is_sortable=True,
            model=self.model,
        )

        # Connect some actions to the table
        def goto_suggestion(*args):
            selected_frame = self.table.getSelectedRowItem()
            self.main_window.commands.gotoVideoAndFrame(
                selected_frame.video, selected_frame.frame_idx
            )

        self.table.doubleClicked.connect(goto_suggestion)
        self.main_window.state.connect("suggestion_idx", self.table.selectRow)

        return self.table

    def lay_everything_out(self) -> None:
        self.wgt_layout.addWidget(self.table)

        table_edit_buttons = self.create_table_edit_buttons()
        self.wgt_layout.addWidget(table_edit_buttons)

        table_nav_buttons = self.create_table_nav_buttons()
        self.wgt_layout.addWidget(table_nav_buttons)

        self.suggestions_form_widget = self.create_suggestions_form()
        self.wgt_layout.addWidget(self.suggestions_form_widget)

    def create_table_nav_buttons(self) -> QWidget:
        main_window = self.main_window
        hb = QHBoxLayout()

        self.add_button(
            hb,
            "Previous",
            main_window.process_events_then(main_window.commands.prevSuggestedFrame),
            "goto previous suggestion",
        )

        self.suggested_count_label = QLabel()
        hb.addWidget(self.suggested_count_label)

        self.add_button(
            hb,
            "Next",
            main_window.process_events_then(main_window.commands.nextSuggestedFrame),
            "goto next suggestion",
        )

        hbw = QWidget()
        hbw.setLayout(hb)
        return hbw

    def create_suggestions_form(self) -> QWidget:
        main_window = self.main_window
        suggestions_form_widget = YamlFormWidget.from_name(
            "suggestions",
            title="Generate Suggestions",
        )
        suggestions_form_widget.mainAction.connect(
            main_window.process_events_then(main_window.commands.generateSuggestions)
        )
        return suggestions_form_widget

    def create_table_edit_buttons(self) -> QWidget:
        main_window = self.main_window
        hb = QHBoxLayout()

        self.add_button(
            hb,
            "Add current frame",
            main_window.process_events_then(
                main_window.commands.addCurrentFrameAsSuggestion
            ),
            "add current frame as suggestion",
        )

        self.add_button(
            hb,
            "Remove",
            main_window.process_events_then(main_window.commands.removeSuggestion),
            "remove suggestion",
        )

        self.add_button(
            hb,
            "Clear all",
            main_window.process_events_then(main_window.commands.clearSuggestions),
            "clear suggestions",
        )

        hbw = QWidget()
        hbw.setLayout(hb)
        return hbw


class InstancesDock(DockWidget):
    """Dock widget for displaying instances."""

    def __init__(self, main_window: QMainWindow, tab_with: Optional[QLayout] = None):
        super().__init__(
            name="Instances",
            main_window=main_window,
            model_type=LabeledFrameTableModel,
            tab_with=tab_with,
        )

    def create_models(self) -> LabeledFrameTableModel:
        self.model = self.model_type(
            items=self.main_window.state["labeled_frame"],
            context=self.main_window.commands,
        )
        return self.model

    def create_tables(self) -> GenericTableView:
        # InstancesTableView adds shift/ctrl multi-select so a second instance
        # can be picked as the merge donor (see Merge Instance).
        self.table = InstancesTableView(
            state=self.main_window.state,
            row_name="instance",
            name_prefix="",
            model=self.model,
        )

        # Apply initial visibility for the optional "mean node score" column,
        # and keep it in sync with the View-menu toggle.
        self._apply_mean_node_score_visibility()
        self.main_window.state.connect(
            "show mean node score",
            lambda _: self._apply_mean_node_score_visibility(),
        )

        # Keep the per-instance checkbox columns ("visibility"/"view only")
        # narrow so they don't crowd out the informational columns.
        self._size_checkbox_columns()

        return self.table

    def _size_checkbox_columns(self) -> None:
        """Resize the visibility/view-only checkbox columns to their contents."""
        header = self.table.horizontalHeader()
        for key in (
            LabeledFrameTableModel.VISIBILITY_KEY,
            LabeledFrameTableModel.VIEW_ONLY_KEY,
        ):
            try:
                col_idx = self.model.properties.index(key)
            except ValueError:
                continue
            header.setSectionResizeMode(col_idx, QHeaderView.ResizeToContents)

    def _apply_mean_node_score_visibility(self) -> None:
        """Hide or show the 'mean node score' column based on the View toggle."""
        if not hasattr(self, "table") or self.table is None:
            return
        try:
            col_idx = self.model.properties.index("mean node score")
        except ValueError:
            return
        show = bool(self.main_window.state.get("show mean node score", default=False))
        self.table.setColumnHidden(col_idx, not show)

    def lay_everything_out(self) -> None:
        self.wgt_layout.addWidget(self.table)

        table_edit_buttons = self.create_table_edit_buttons()
        self.wgt_layout.addWidget(table_edit_buttons)

    def create_table_edit_buttons(self) -> QWidget:
        main_window = self.main_window

        hb = QHBoxLayout()
        self.add_button(
            hb, "New Instance", lambda x: main_window.commands.newInstance(offset=10)
        )
        self.add_button(
            hb, "Delete Instance", main_window.commands.deleteSelectedInstance
        )
        self.add_button(
            hb, "Merge Instance", lambda *_: main_window.commands.mergeInstance()
        )

        hbw = QWidget()
        hbw.setLayout(hb)
        return hbw


class SessionsDock(DockWidget):
    """Dock widget for session-level 3D triangulation workflows."""

    def __init__(self, main_window: QMainWindow, tab_with: Optional[QLayout] = None):
        self.calibration_path_edit = None
        self.events_path_edit = None
        self.events_filter_combo = None
        self.events_table = None
        self.events_status_label = None
        super().__init__(name="Sessions", main_window=main_window, tab_with=tab_with)
        self.main_window.state.connect("video", lambda _: self.refresh())
        self.main_window.state.connect("frame_idx", lambda _: self._select_current_event())
        self.refresh()

    def create_models(self) -> None:
        return None

    def create_tables(self) -> None:
        return None

    def lay_everything_out(self) -> None:
        self.wgt_layout.addWidget(self._create_events_groupbox())
        self.wgt_layout.addWidget(self._create_triangulation_groupbox())
        self.wgt_layout.addStretch()

    def _current_session(self):
        return get_session_for_video(
            self.main_window.labels, self.main_window.state["video"]
        )

    def _create_events_groupbox(self) -> QGroupBox:
        gb = QGroupBox("Session Events")
        layout = QVBoxLayout()

        self.events_path_edit = self._add_path_row(
            layout,
            "Events",
            "Select a two-column session events text file",
            self._select_events_file,
        )
        self.events_path_edit.setReadOnly(True)

        filter_row = QHBoxLayout()
        filter_row.addWidget(QLabel("Show"))
        self.events_filter_combo = QComboBox()
        self.events_filter_combo.currentIndexChanged.connect(self._populate_events_table)
        filter_row.addWidget(self.events_filter_combo)
        filter_widget = QWidget()
        filter_widget.setLayout(filter_row)
        layout.addWidget(filter_widget)

        self.events_table = QTableWidget(0, 2)
        self.events_table.setHorizontalHeaderLabels(["Event", "Frame"])
        self.events_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeToContents
        )
        self.events_table.horizontalHeader().setStretchLastSection(True)
        self.events_table.verticalHeader().setVisible(False)
        self.events_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.events_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.events_table.cellClicked.connect(self._jump_to_event_row)
        layout.addWidget(self.events_table)

        self.events_status_label = QLabel("")
        layout.addWidget(self.events_status_label)

        gb.setLayout(layout)
        return gb

    def refresh(self) -> None:
        """Refresh the session event section from the current video's session."""
        if self.events_path_edit is None:
            return

        session = self._current_session()
        events = get_session_events(session)
        path = "" if session is None else session.metadata.get(SESSION_EVENTS_PATH_KEY, "")
        self.events_path_edit.setText(path)

        selected = self.events_filter_combo.currentText()
        self.events_filter_combo.blockSignals(True)
        self.events_filter_combo.clear()
        self.events_filter_combo.addItem("All events")
        for name in event_names(events):
            self.events_filter_combo.addItem(name)
        if selected:
            idx = self.events_filter_combo.findText(selected)
            if idx >= 0:
                self.events_filter_combo.setCurrentIndex(idx)
        self.events_filter_combo.blockSignals(False)

        self._populate_events_table()

    def _select_events_file(self) -> None:
        session = self._current_session()
        if session is None:
            self._show_error("Select a video that belongs to a recording session first.")
            return

        filename, _ = FileDialog.open(
            self,
            caption="Select session events file",
            dir=self.events_path_edit.text() or session.metadata.get("session_path"),
            filter="Text Files (*.txt);;All Files (*)",
        )
        if not filename:
            return

        try:
            events = load_session_events_file(filename)
        except Exception as exc:
            self._show_error(f"Could not load session events:\n{exc}")
            return

        changed = set_session_events(session, filename, events)
        if changed:
            self.main_window.labels.update()
            self.main_window.commands.changestack_push("load session events")
        self.main_window.on_data_update([UpdateTopic.video, UpdateTopic.frame])

    def _filtered_events(self) -> List[dict]:
        session = self._current_session()
        events = get_session_events(session)
        selected = self.events_filter_combo.currentText()
        if selected and selected != "All events":
            events = [event for event in events if event["event"] == selected]
        return events

    def _populate_events_table(self, *args) -> None:
        if self.events_table is None:
            return

        events = self._filtered_events()
        colors = event_color_map(get_session_events(self._current_session()))
        self.events_table.setRowCount(len(events))
        for row, event in enumerate(events):
            event_item = QTableWidgetItem(str(event["event"]))
            frame = int(event["frame"])
            frame_item = QTableWidgetItem(str(frame + 1))
            frame_item.setData(Qt.UserRole, frame)

            rgb = colors.get(event["event"], (80, 80, 80))
            color = QtGui.QColor(*rgb)
            event_item.setBackground(color)
            luminance = (0.299 * rgb[0]) + (0.587 * rgb[1]) + (0.114 * rgb[2])
            event_item.setForeground(
                QtGui.QColor("black" if luminance > 160 else "white")
            )
            self.events_table.setItem(row, 0, event_item)
            self.events_table.setItem(row, 1, frame_item)

        total = len(get_session_events(self._current_session()))
        if total:
            self.events_status_label.setText(
                f"{len(events)} shown, {total} event(s) loaded"
            )
        else:
            self.events_status_label.setText("No session events loaded")
        self._select_current_event()

    def _jump_to_event_row(self, row: int, column: int) -> None:
        frame_item = self.events_table.item(row, 1)
        if frame_item is None:
            return

        frame = frame_item.data(Qt.UserRole)
        video = self.main_window.state["video"]
        if video is not None:
            frame = min(int(frame), len(video) - 1)
        self.main_window.state["frame_idx"] = int(frame)

    def _select_current_event(self) -> None:
        if self.events_table is None:
            return

        current = self.main_window.state["frame_idx"]
        if current is None:
            return

        best_row = -1
        best_frame = None
        for row in range(self.events_table.rowCount()):
            item = self.events_table.item(row, 1)
            if item is None:
                continue
            frame = int(item.data(Qt.UserRole))
            if frame <= current and (best_frame is None or frame > best_frame):
                best_row = row
                best_frame = frame

        if best_row >= 0:
            self.events_table.selectRow(best_row)
            self.events_table.scrollToItem(self.events_table.item(best_row, 0))

    def _create_triangulation_groupbox(self) -> QGroupBox:
        gb = QGroupBox("Session Triangulation")
        layout = QVBoxLayout()

        self.calibration_path_edit = self._add_path_row(
            layout,
            "Calibration",
            "Select calibration.toml from anipose or sleap-anipose calibration",
            self._select_calibration_file,
        )

        run_row = QHBoxLayout()
        self.add_button(
            run_row,
            "Triangulate",
            self._run_triangulation,
            key="triangulate",
        )
        run_widget = QWidget()
        run_widget.setLayout(run_row)
        layout.addWidget(run_widget)

        gb.setLayout(layout)
        return gb

    def _add_path_row(
        self, layout: QVBoxLayout, label: str, tooltip: str, browse: Callable
    ) -> QLineEdit:
        row = QHBoxLayout()
        row.addWidget(QLabel(label))
        path_edit = QLineEdit()
        path_edit.setToolTip(tooltip)
        row.addWidget(path_edit)
        browse_button = QPushButton("Browse")
        browse_button.clicked.connect(browse)
        row.addWidget(browse_button)
        row_widget = QWidget()
        row_widget.setLayout(row)
        layout.addWidget(row_widget)
        return path_edit

    def _select_calibration_file(self) -> None:
        filename, _ = FileDialog.open(
            self,
            caption="Select calibration.toml",
            dir=self.calibration_path_edit.text() or None,
            filter="TOML Files (*.toml);;All Files (*)",
        )
        if filename:
            self.calibration_path_edit.setText(filename)

    def _validate_calibration_path(self) -> Path:
        calibration = Path(self.calibration_path_edit.text()).expanduser()

        if not calibration.exists() or not calibration.is_file():
            raise FileNotFoundError("Select a valid calibration.toml file.")
        if calibration.suffix.lower() != ".toml":
            raise ValueError("Calibration file must be a .toml file.")

        return calibration

    def _run_triangulation(self) -> None:
        try:
            calibration = self._validate_calibration_path()
        except Exception as exc:
            self._show_error(str(exc))
            return

        self.main_window.statusBar().showMessage(
            "Triangulating session labels into missing views..."
        )
        params = {"calibration_path": str(calibration)}
        try:
            TriangulateSessionLabels.do_action(self.main_window.commands, params)
        except Exception as exc:
            self._show_error(f"Triangulation failed:\n{exc}")
            return

        result = params["result"]
        if result["predicted_instances"] > 0:
            self.main_window.commands.changestack_push("triangulate session labels")
            self.main_window.on_data_update(TriangulateSessionLabels.topics)
        else:
            self.main_window.on_data_update([UpdateTopic.frame])

        skipped = result["skipped_sessions"]
        skipped_msg = "\n\nSkipped sessions:\n" + "\n".join(skipped) if skipped else ""
        msg = (
            f"Predicted {result['predicted_instances']} instance(s) "
            f"on {result['eligible_frames']} frame(s) "
            f"across {result['sessions']} session(s)."
            f"{skipped_msg}"
        )
        self.main_window.statusBar().showMessage("Triangulation complete")
        QMessageBox.information(self, "Triangulation Complete", msg)

    def _show_error(self, message: str) -> None:
        self.main_window.statusBar().showMessage(message)
        QMessageBox.warning(self, "Triangulation", message)


class Skeleton3DDock(DockWidget):
    """Dock widget with a compact launcher for the Skeleton3D popup."""

    def __init__(self, main_window: QMainWindow, tab_with: Optional[QLayout] = None):
        super().__init__(name="Skeleton3D", main_window=main_window, tab_with=tab_with)
        self.main_window.state.connect("frame_idx", self._sync_dialog_frame)

    def create_models(self) -> None:
        return None

    def create_tables(self) -> None:
        return None

    def lay_everything_out(self) -> None:
        gb = QGroupBox("Skeleton3D")
        layout = QVBoxLayout()
        self.add_button(
            layout,
            "Open Skeleton3D...",
            self.open_skeleton3d,
            key="open skeleton3d",
        )
        gb.setLayout(layout)
        self.wgt_layout.addWidget(gb)
        self.wgt_layout.addStretch()

    def open_skeleton3d(self) -> None:
        """Open the standalone Skeleton3D reconstruction window."""
        from sleap.gui.widgets.skeleton3d import Skeleton3DDialog

        dialog = self.main_window._child_windows.get("skeleton3d")
        if dialog is None:
            dialog = Skeleton3DDialog(main_window=self.main_window, parent=self.main_window)
            self.main_window._child_windows["skeleton3d"] = dialog
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    def _sync_dialog_frame(self, frame_idx: int) -> None:
        dialog = self.main_window._child_windows.get("skeleton3d")
        if dialog is not None and dialog.isVisible():
            dialog.set_frame(frame_idx)


class ReachesDock(DockWidget):
    """Dock widget for reach-segment detection and curation (KPN Outward Peaks)."""

    def __init__(self, main_window: QMainWindow, tab_with=None):
        self._reaches: List = []          # List[ReachSegment]
        self._reach_details: List = []
        self._reach_detection_info: dict = {}
        self._pellet_history: List = []
        self._pred_labels_obj = None      # loaded external predictions labels
        self._points3d_source = None      # loaded points3D H5 trajectory source
        self._points3d_reprojections = None  # loaded reprojections H5 overlay source
        self._lh_traj2d = None            # (n_frames, 2) float64 or None
        self._rh_traj2d = None            # (n_frames, 2) float64 or None
        self._pellet_traj2d = None        # (n_frames, 2) float64 or None
        self._pred_source_video_stem: Optional[str] = None
        self.on_reaches_changed: Optional[Callable] = None
        self.on_reach_traces_changed: Optional[Callable] = None
        super().__init__(
            name="Reaches",
            main_window=main_window,
            model_type=None,
            tab_with=tab_with,
        )
        main_window.state.connect("video", lambda _: self._refresh_nodes())
        main_window.state.connect("frame_idx", lambda _: self._update_frame_markers())

    def create_models(self) -> None:
        return None

    def create_tables(self) -> None:
        return None

    def lay_everything_out(self) -> None:
        content = QWidget()
        content_layout = QVBoxLayout()
        content_layout.setContentsMargins(4, 4, 4, 4)
        content_layout.addWidget(self._create_nodes_groupbox())
        content_layout.addWidget(self._create_predictions_groupbox())
        content_layout.addWidget(self._create_filter_groupbox())
        content_layout.addWidget(self._create_params_groupbox())
        content_layout.addWidget(self._create_results_groupbox())
        content_layout.addStretch()
        content.setLayout(content_layout)

        scroll = QScrollArea()
        scroll.setWidget(content)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.wgt_layout.setContentsMargins(0, 0, 0, 0)
        self.wgt_layout.addWidget(scroll)

    # ── node selection (two-column checkbox) ─────────────────────────────── #

    def _create_nodes_groupbox(self) -> QGroupBox:
        gb = QGroupBox("Hand Nodes")
        layout = QVBoxLayout()

        # Two-column header labels
        col_headers = QHBoxLayout()
        lh_header = QLabel("Left Hand")
        lh_header.setAlignment(Qt.AlignCenter)
        lh_header.setStyleSheet("font-weight: bold; color: #38bdf8;")
        rh_header = QLabel("Right Hand")
        rh_header.setAlignment(Qt.AlignCenter)
        rh_header.setStyleSheet("font-weight: bold; color: #34d399;")
        col_headers.addWidget(lh_header)
        col_headers.addWidget(rh_header)
        layout.addLayout(col_headers)

        # Two side-by-side node lists
        lists_row = QHBoxLayout()
        self._lh_list = QListWidget()
        self._lh_list.setFixedHeight(110)
        self._rh_list = QListWidget()
        self._rh_list.setFixedHeight(110)
        lists_row.addWidget(self._lh_list)
        lists_row.addWidget(self._rh_list)
        layout.addLayout(lists_row)

        # Auto-suggest buttons
        btn_row = QHBoxLayout()
        btn_auto_lh = QPushButton("Auto LH")
        btn_auto_lh.setToolTip("Auto-check left-hand keyword matches")
        btn_auto_lh.clicked.connect(self._auto_check_left)
        btn_auto_rh = QPushButton("Auto RH")
        btn_auto_rh.setToolTip("Auto-check right-hand keyword matches")
        btn_auto_rh.clicked.connect(self._auto_check_right)
        btn_clear = QPushButton("Clear All")
        btn_clear.clicked.connect(self._check_none)
        btn_row.addWidget(btn_auto_lh)
        btn_row.addWidget(btn_auto_rh)
        btn_row.addWidget(btn_clear)
        layout.addLayout(btn_row)

        self._diag_label = QLabel("")
        self._diag_label.setWordWrap(True)
        self._diag_label.setStyleSheet(
            "color: #777; font-size: 10px; padding: 2px 0;"
        )
        layout.addWidget(self._diag_label)

        self._show_markers_check = QCheckBox("Show markers on video")
        self._show_markers_check.setChecked(True)
        self._show_markers_check.toggled.connect(self._update_frame_markers)
        layout.addWidget(self._show_markers_check)

        gb.setLayout(layout)
        self._refresh_nodes()
        return gb

    def _refresh_nodes(self) -> None:
        """Repopulate LH/RH node lists from the skeleton, update diagnostics."""
        if not hasattr(self, "_lh_list"):
            return

        nodes = self._skeleton_node_names()
        if not nodes:
            self._diag_label.setText("No skeleton loaded.")
            self._lh_list.clear()
            self._rh_list.clear()
            return

        # Preserve current checked state before clearing
        was_lh = set()
        for i in range(self._lh_list.count()):
            item = self._lh_list.item(i)
            if item.checkState() == Qt.Checked:
                was_lh.add(item.text())
        was_rh = set()
        for i in range(self._rh_list.count()):
            item = self._rh_list.item(i)
            if item.checkState() == Qt.Checked:
                was_rh.add(item.text())

        from sleap.gui.reach_detection import suggest_hand_nodes
        suggestions = suggest_hand_nodes(nodes)
        auto_lh = set(suggestions.get("left", []))
        auto_rh = set(suggestions.get("right", []))

        # If this is a fresh load (nothing was checked before), use auto-suggestions
        first_load_lh = len(was_lh) == 0
        first_load_rh = len(was_rh) == 0

        def _populate(lst: QListWidget, was_checked: set, auto: set, first_load: bool):
            lst.clear()
            for name in nodes:
                item = QListWidgetItem(name)
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                checked = (name in auto) if first_load else (name in was_checked)
                item.setCheckState(Qt.Checked if checked else Qt.Unchecked)
                lst.addItem(item)

        _populate(self._lh_list, was_lh, auto_lh, first_load_lh)
        _populate(self._rh_list, was_rh, auto_rh, first_load_rh)

        self._update_diag(nodes)

    def _skeleton_node_names(self) -> List[str]:
        """Return node names from points3D, predictions file skeleton, or main labels."""
        if self._points3d_source is not None:
            return list(self._points3d_source.get("node_names", []))
        if self._pred_labels_obj is not None:
            try:
                return self._pred_labels_obj.skeletons[0].node_names
            except (IndexError, AttributeError):
                pass
        try:
            return self.main_window.labels.skeletons[0].node_names
        except (IndexError, AttributeError):
            return []

    def _update_diag(self, nodes: List[str]) -> None:
        """Update the diagnostic label with skeleton/data info."""
        try:
            # Frame counts: from predictions file if loaded, else main labels
            if self._points3d_source is not None:
                points3d = self._points3d_source.get("points3d")
                n_frames = int(points3d.shape[0]) if points3d is not None else 0
                src_str = f"points3D: {n_frames} frames"
            elif self._pred_labels_obj is not None:
                idx = self._pred_cam_combo.currentIndex()
                if idx >= 0 and idx < len(self._pred_labels_obj.videos):
                    pred_vid = self._pred_labels_obj.videos[idx]
                    n_pred = sum(
                        1 for lf in self._pred_labels_obj
                        if lf.video == pred_vid and lf.predicted_instances
                    )
                    src_str = f"Predictions: {n_pred} frames"
                else:
                    src_str = "Predictions: (select camera)"
            else:
                labels = self.main_window.labels
                video = self.main_window.state.get("video", default=None)
                n_labeled = n_predicted = 0
                if video is not None:
                    for lf in labels.find(video):
                        if lf.user_instances:
                            n_labeled += 1
                        if lf.predicted_instances:
                            n_predicted += 1
                src_str = f"Labeled: {n_labeled} | Predicted: {n_predicted} frames"

            lh_sel = len(self._selected_lh_nodes())
            rh_sel = len(self._selected_rh_nodes())
            self._diag_label.setText(
                f"Skeleton: {len(nodes)} nodes  |  "
                f"LH: {lh_sel} checked  RH: {rh_sel} checked\n"
                f"{src_str}"
            )
        except Exception:
            pass

    def _selected_lh_nodes(self) -> List[str]:
        result = []
        for i in range(self._lh_list.count()):
            item = self._lh_list.item(i)
            if item.checkState() == Qt.Checked:
                result.append(item.text())
        return result

    def _selected_rh_nodes(self) -> List[str]:
        result = []
        for i in range(self._rh_list.count()):
            item = self._rh_list.item(i)
            if item.checkState() == Qt.Checked:
                result.append(item.text())
        return result

    def _set_checked_nodes_in_list(self, lst: QListWidget, names: set) -> None:
        for i in range(lst.count()):
            item = lst.item(i)
            item.setCheckState(Qt.Checked if item.text() in names else Qt.Unchecked)

    def _auto_check_left(self) -> None:
        from sleap.gui.reach_detection import suggest_hand_nodes
        nodes = self._skeleton_node_names()
        if not nodes:
            return
        suggested = set(suggest_hand_nodes(nodes).get("left", []))
        self._set_checked_nodes_in_list(self._lh_list, suggested)
        self._update_diag(nodes)

    def _auto_check_right(self) -> None:
        from sleap.gui.reach_detection import suggest_hand_nodes
        nodes = self._skeleton_node_names()
        if not nodes:
            return
        suggested = set(suggest_hand_nodes(nodes).get("right", []))
        self._set_checked_nodes_in_list(self._rh_list, suggested)
        self._update_diag(nodes)

    def _check_none(self) -> None:
        self._set_checked_nodes_in_list(self._lh_list, set())
        self._set_checked_nodes_in_list(self._rh_list, set())
        nodes = self._skeleton_node_names()
        if nodes:
            self._update_diag(nodes)

    # ── predictions source ────────────────────────────────────────────────── #

    def _create_predictions_groupbox(self) -> QGroupBox:
        gb = QGroupBox("Predictions Source")
        layout = QVBoxLayout()

        # File path row
        path_row = QHBoxLayout()
        self._pred_path_edit = QLineEdit()
        self._pred_path_edit.setReadOnly(True)
        self._pred_path_edit.setPlaceholderText("(using project labels)")
        path_row.addWidget(self._pred_path_edit)
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._browse_predictions)
        path_row.addWidget(browse_btn)
        disc_btn = QPushButton("Discover")
        disc_btn.setToolTip(
            "Search for *.slp files in the session's predictions/ folder"
        )
        disc_btn.clicked.connect(self._discover_predictions)
        path_row.addWidget(disc_btn)
        path_widget = QWidget()
        path_widget.setLayout(path_row)
        layout.addWidget(path_widget)

        # Camera / video selection within the predictions file
        cam_row = QHBoxLayout()
        cam_row.addWidget(QLabel("Camera:"))
        self._pred_cam_combo = QComboBox()
        self._pred_cam_combo.setEnabled(False)
        self._pred_cam_combo.currentIndexChanged.connect(self._on_pred_cam_changed)
        cam_row.addWidget(self._pred_cam_combo)
        clear_btn = QPushButton("Clear")
        clear_btn.clicked.connect(self._clear_predictions)
        cam_row.addWidget(clear_btn)
        cam_widget = QWidget()
        cam_widget.setLayout(cam_row)
        layout.addWidget(cam_widget)

        points_row = QHBoxLayout()
        points_row.addWidget(QLabel("points3D:"))
        self._points3d_path_edit = QLineEdit()
        self._points3d_path_edit.setReadOnly(True)
        self._points3d_path_edit.setPlaceholderText("(optional 3D source)")
        points_row.addWidget(self._points3d_path_edit)
        points_btn = QPushButton("Browse...")
        points_btn.clicked.connect(self._browse_points3d)
        points_row.addWidget(points_btn)
        points_clear_btn = QPushButton("Clear")
        points_clear_btn.clicked.connect(self._clear_points3d)
        points_row.addWidget(points_clear_btn)
        points_widget = QWidget()
        points_widget.setLayout(points_row)
        layout.addWidget(points_widget)

        self._pred_status_label = QLabel("Source: project labels")
        self._pred_status_label.setWordWrap(True)
        self._pred_status_label.setStyleSheet(
            "color: #777; font-size: 10px;"
        )
        layout.addWidget(self._pred_status_label)

        gb.setLayout(layout)
        return gb

    def _browse_predictions(self) -> None:
        filename, _ = FileDialog.open(
            self,
            caption="Select predictions file",
            filter=(
                "Prediction Sources (*.slp *.h5 *.hdf5 *.nwb);;"
                "SLEAP Files (*.slp);;"
                "SLEAP Analysis HDF5 (*.h5 *.hdf5);;"
                "NWB Files (*.nwb);;"
                "All Files (*)"
            ),
        )
        if filename:
            self._load_predictions_file(filename)

    def _browse_points3d(self) -> None:
        filename, _ = FileDialog.open(
            self,
            caption="Select points3D file",
            filter="HDF5 Files (*.h5 *.hdf5);;All Files (*)",
        )
        if filename:
            self._load_points3d_file(filename)

    def _load_points3d_file(self, path: str) -> None:
        from sleap.gui.reach_projection import load_points3d_h5

        try:
            source = load_points3d_h5(path)
        except Exception as exc:
            self._pred_status_label.setStyleSheet("color: #ef4444; font-size: 10px;")
            self._pred_status_label.setText(f"Error loading points3D: {exc}")
            return

        self._points3d_source = source
        self._points3d_reprojections = self._load_associated_reprojections(source)
        self._points3d_path_edit.setText(path)
        self._pred_status_label.setStyleSheet("color: #2ecc71; font-size: 10px;")
        reproj_msg = (
            f", reprojections: {len(self._points3d_reprojections['camera_names'])} cameras"
            if self._points3d_reprojections is not None
            else ", no reprojections found for markers"
        )
        self._pred_status_label.setText(
            f"Source: points3D - {source['points3d'].shape[0]} frames, "
            f"{len(source['node_names'])} nodes{reproj_msg}"
        )
        self._refresh_nodes()

    def _clear_points3d(self) -> None:
        self._points3d_source = None
        self._points3d_reprojections = None
        self._pred_source_video_stem = None
        if hasattr(self, "_points3d_path_edit"):
            self._points3d_path_edit.clear()
        self._pred_status_label.setStyleSheet("color: #777; font-size: 10px;")
        self._pred_status_label.setText("Source: project labels")
        if self._pred_labels_obj is not None:
            self._on_pred_cam_changed(self._pred_cam_combo.currentIndex())
        self._refresh_nodes()

    def _discover_predictions(self) -> None:
        """Find *.slp files in <session_dir>/predictions/ and load best match."""
        video = self.main_window.state.get("video", default=None)
        if video is None:
            self._pred_status_label.setText("No video loaded.")
            return

        from pathlib import Path
        video_dir = Path(self._video_filename(video)).parent
        candidates: List[Path] = []
        for search_root in (video_dir, video_dir.parent):
            pred_dir = search_root / "predictions"
            if pred_dir.is_dir():
                candidates.extend(pred_dir.glob("*.slp"))
                candidates.extend(pred_dir.glob("*.h5"))
                candidates.extend(pred_dir.glob("*.hdf5"))
                candidates.extend(pred_dir.glob("*.nwb"))

        if not candidates:
            self._pred_status_label.setText(
                "No predictions files found in predictions/ subfolder."
            )
            return

        if len(candidates) == 1:
            self._load_predictions_file(str(candidates[0]))
            return

        # Multiple files: prefer one matching the video stem
        video_stem = Path(self._video_filename(video)).stem.lower()
        best = next(
            (c for c in candidates if video_stem in c.stem.lower()), candidates[0]
        )
        self._load_predictions_file(str(best))

    def _load_predictions_file(self, path: str) -> None:
        """Load prediction-source labels and populate the camera combo."""
        from pathlib import Path
        try:
            current_video = self.main_window.state.get("video", default=None)
            loaded, source_kind = self._load_prediction_source_labels(
                path,
                video=current_video,
            )
        except Exception as exc:
            self._pred_status_label.setText(f"Error loading file: {exc}")
            return

        self._pred_labels_obj = loaded
        self._points3d_source = None
        self._points3d_reprojections = None
        self._pred_path_edit.setText(path)
        if hasattr(self, "_points3d_path_edit"):
            self._points3d_path_edit.clear()

        # Populate camera combo
        self._pred_cam_combo.blockSignals(True)
        self._pred_cam_combo.clear()
        for v in loaded.videos:
            self._pred_cam_combo.addItem(Path(self._video_filename(v)).name)
        self._pred_cam_combo.setEnabled(True)
        self._pred_cam_combo.blockSignals(False)

        # Auto-select camera whose name best matches the current video
        if current_video is not None:
            cur_stem = Path(self._video_filename(current_video)).stem.lower()
            for i, v in enumerate(loaded.videos):
                if cur_stem in Path(self._video_filename(v)).stem.lower():
                    self._pred_cam_combo.setCurrentIndex(i)
                    break
        else:
            self._pred_cam_combo.setCurrentIndex(0)

        self._validate_session_match(path, source_kind=source_kind)
        self._refresh_nodes()

    @staticmethod
    def _load_prediction_source_labels(path: str, video=None):
        import h5py
        import sleap_io

        suffix = Path(path).suffix.lower()
        if suffix == ".nwb":
            return sleap_io.load_nwb(path), "nwb"

        if suffix in {".h5", ".hdf5"}:
            try:
                with h5py.File(path, "r") as f:
                    if "track_occupancy" in f:
                        return sleap_io.load_analysis_h5(path, video=video), "analysis_h5"
            except OSError:
                pass

        return sleap_io.load_slp(path), "slp"

    @staticmethod
    def _video_filename(video) -> str:
        filename = getattr(video, "filename", "")
        if isinstance(filename, list):
            filename = filename[0] if filename else ""
        return str(filename or "")

    def _validate_session_match(self, pred_path: str, *, source_kind: str = "slp") -> None:
        """Check that the predictions file belongs to the current session."""
        from pathlib import Path
        video = self.main_window.state.get("video", default=None)
        if video is None:
            return

        session_dir = Path(self._video_filename(video)).parent
        pred_file = Path(pred_path)

        # Check: pred file is within the session directory tree
        try:
            pred_file.relative_to(session_dir)
            match = True
        except ValueError:
            # Looser check: session dir name appears in the predictions path
            match = session_dir.name in str(pred_file)

        idx = self._pred_cam_combo.currentIndex()
        n_pred = self._pred_frame_count(idx)
        cam_name = (
            self._pred_cam_combo.currentText() if idx >= 0 else "—"
        )
        check = "✓ session match" if match else "⚠ session mismatch"
        self._pred_status_label.setStyleSheet(
            "color: #2ecc71; font-size: 10px;"
            if match
            else "color: #e67e22; font-size: 10px;"
        )
        self._pred_status_label.setText(
            f"Camera: {cam_name} — {n_pred} predicted frames  {check}"
        )

    def _pred_frame_count(self, cam_idx: int) -> int:
        if (
            self._pred_labels_obj is None
            or cam_idx < 0
            or cam_idx >= len(self._pred_labels_obj.videos)
        ):
            return 0
        pred_vid = self._pred_labels_obj.videos[cam_idx]
        return sum(
            1
            for lf in self._pred_labels_obj
            if lf.video == pred_vid and lf.predicted_instances
        )

    def _on_pred_cam_changed(self, idx: int = -1) -> None:
        if self._pred_labels_obj is None:
            return
        n_pred = self._pred_frame_count(
            self._pred_cam_combo.currentIndex()
        )
        cam_name = self._pred_cam_combo.currentText()
        self._pred_status_label.setText(
            f"Camera: {cam_name} — {n_pred} predicted frames"
        )
        self._refresh_nodes()

    def _clear_predictions(self) -> None:
        self._pred_labels_obj = None
        self._points3d_source = None
        self._points3d_reprojections = None
        self._pred_source_video_stem = None
        self._pred_path_edit.clear()
        if hasattr(self, "_points3d_path_edit"):
            self._points3d_path_edit.clear()
        self._pred_cam_combo.clear()
        self._pred_cam_combo.setEnabled(False)
        self._pred_status_label.setStyleSheet("color: #777; font-size: 10px;")
        self._pred_status_label.setText("Source: project labels")
        self._notify_reach_traces_changed([])
        self._refresh_nodes()

    # ── detection parameters ─────────────────────────────────────────────── #

    def _create_filter_groupbox(self) -> QGroupBox:
        gb = QGroupBox("Filter Parameters")
        layout = QVBoxLayout()

        form = QWidget()
        form_layout = QFormLayout()

        self._filter_hand_traces = QCheckBox("Filter LH/RH traces")
        self._filter_hand_traces.setChecked(False)
        form_layout.addRow("", self._filter_hand_traces)

        self._filter_cutoff = QDoubleSpinBox()
        self._filter_cutoff.setRange(0.1, 10000.0)
        self._filter_cutoff.setValue(30.0)
        self._filter_cutoff.setSingleStep(1.0)
        self._filter_cutoff.setSuffix(" Hz")
        self._filter_cutoff.setToolTip(
            "First-order Butterworth cutoff. Sampling rate is read from the video."
        )
        self._filter_cutoff.setEnabled(False)
        self._filter_hand_traces.toggled.connect(self._filter_cutoff.setEnabled)
        form_layout.addRow("Cutoff frequency:", self._filter_cutoff)

        self._filter_sampling_label = QLabel("Auto")
        form_layout.addRow("Sampling rate:", self._filter_sampling_label)

        form.setLayout(form_layout)
        layout.addWidget(form)
        gb.setLayout(layout)
        return gb

    def _create_params_groupbox(self) -> QGroupBox:
        gb = QGroupBox("Detection Parameters")
        layout = QVBoxLayout()

        form = QWidget()
        form_layout = QFormLayout()

        method_row = QWidget()
        method_layout = QHBoxLayout()
        method_layout.setContentsMargins(0, 0, 0, 0)
        self._method_from_pellet_radio = QRadioButton("From pellet")
        self._method_absolute_radio = QRadioButton("Absolute")
        self._method_from_pellet_radio.setChecked(True)
        self._method_group = QButtonGroup(self)
        self._method_group.addButton(self._method_from_pellet_radio)
        self._method_group.addButton(self._method_absolute_radio)
        self._method_from_pellet_radio.toggled.connect(
            self._update_detection_method_controls
        )
        self._method_absolute_radio.toggled.connect(
            self._update_detection_method_controls
        )
        method_layout.addWidget(self._method_from_pellet_radio)
        method_layout.addWidget(self._method_absolute_radio)
        method_layout.addStretch()
        method_row.setLayout(method_layout)
        form_layout.addRow("Detect reaches:", method_row)

        self._min_thresh = QDoubleSpinBox()
        self._min_thresh.setRange(-10000, 10000)
        self._min_thresh.setValue(-10.0)
        self._min_thresh.setSingleStep(1.0)
        form_layout.addRow("KPN outward threshold:", self._min_thresh)

        self._max_thresh = QDoubleSpinBox()
        self._max_thresh.setRange(-10000, 10000)
        self._max_thresh.setValue(-7.0)
        self._max_thresh.setSingleStep(1.0)
        form_layout.addRow("KPN max threshold:", self._max_thresh)

        self._prominence = QDoubleSpinBox()
        self._prominence.setRange(0, 1000)
        self._prominence.setValue(1.0)
        self._prominence.setSingleStep(0.5)
        form_layout.addRow("Peak prominence:", self._prominence)

        self._min_travel = QDoubleSpinBox()
        self._min_travel.setRange(0, 1000)
        self._min_travel.setValue(4.0)
        self._min_travel.setSingleStep(1.0)
        form_layout.addRow("Min outward travel:", self._min_travel)

        self._min_frames = QSpinBox()
        self._min_frames.setRange(1, 10000)
        self._min_frames.setValue(15)
        form_layout.addRow("Min duration (frames):", self._min_frames)

        self._max_frames = QSpinBox()
        self._max_frames.setRange(1, 10000)
        self._max_frames.setValue(200)
        form_layout.addRow("Max duration (frames):", self._max_frames)

        self._start_padding = QSpinBox()
        self._start_padding.setRange(0, 100)
        self._start_padding.setValue(5)
        form_layout.addRow("Start padding (frames):", self._start_padding)

        self._absolute_axis_combo = QComboBox()
        self._absolute_axis_combo.addItem("X", 0)
        self._absolute_axis_combo.addItem("Y", 1)
        self._absolute_axis_combo.addItem("Z", 2)
        self._absolute_axis_combo.setToolTip(
            "Coordinate used as the absolute outward-position signal."
        )
        form_layout.addRow("Absolute axis:", self._absolute_axis_combo)

        self._absolute_invert = QCheckBox("Invert coordinate")
        self._absolute_invert.setToolTip(
            "Use the negative of the selected coordinate before thresholding."
        )
        form_layout.addRow("", self._absolute_invert)

        self._absolute_max_start = QDoubleSpinBox()
        self._absolute_max_start.setRange(-100000, 100000)
        self._absolute_max_start.setValue(300.0)
        self._absolute_max_start.setSingleStep(10.0)
        self._absolute_max_start.setToolTip(
            "Reject absolute reaches whose padded start value is above this value."
        )
        form_layout.addRow("Absolute max start:", self._absolute_max_start)

        self._max_dist_home = QDoubleSpinBox()
        self._max_dist_home.setRange(0.1, 10000)
        self._max_dist_home.setValue(15.0)
        self._max_dist_home.setSingleStep(1.0)
        self._max_dist_home.setToolTip(
            "Maximum distance the pellet can be from its home position to be "
            "considered 'placed'. Use ~2 for 2D pixel coordinates; use 10–25 mm "
            "for 3D calibration space (typical reconstruction noise)."
        )
        form_layout.addRow("Max pellet dist. from home:", self._max_dist_home)

        self._point_confidence = QDoubleSpinBox()
        self._point_confidence.setRange(0.0, 1.0)
        self._point_confidence.setDecimals(2)
        self._point_confidence.setValue(0.5)
        self._point_confidence.setSingleStep(0.05)
        self._point_confidence.setToolTip(
            "Minimum SLEAP point confidence used for paw and pellet trajectories."
        )
        form_layout.addRow("Min point confidence:", self._point_confidence)

        form.setLayout(form_layout)
        layout.addWidget(form)
        self._update_detection_method_controls()

        find_btn = QPushButton("Find Reaches")
        find_btn.clicked.connect(self._run_detection)
        layout.addWidget(find_btn)

        gb.setLayout(layout)
        return gb

    # ── results table ────────────────────────────────────────────────────── #

    def _detection_method(self) -> str:
        if (
            hasattr(self, "_method_absolute_radio")
            and self._method_absolute_radio.isChecked()
        ):
            return "absolute"
        return "from_pellet"

    def batch_detection_settings(self) -> Dict[str, Any]:
        """Return the current reach-detection settings for batch analysis."""
        return {
            "left_hand_nodes": self._selected_lh_nodes(),
            "right_hand_nodes": self._selected_rh_nodes(),
            "method": self._detection_method(),
            "kpn_outward_threshold": float(self._min_thresh.value()),
            "kpn_outward_max_threshold": float(self._max_thresh.value()),
            "kpn_peak_prominence": float(self._prominence.value()),
            "kpn_min_outward_travel": float(self._min_travel.value()),
            "kpn_start_padding": int(self._start_padding.value()),
            "min_frame": int(self._min_frames.value()),
            "max_frame": int(self._max_frames.value()),
            "max_dist_from_home": float(self._max_dist_home.value()),
            "point_confidence_threshold": float(self._point_confidence.value()),
            "absolute_axis": int(self._absolute_axis_combo.currentData()),
            "absolute_axis_name": str(self._absolute_axis_combo.currentText()).lower(),
            "absolute_invert": bool(self._absolute_invert.isChecked()),
            "absolute_max_start_value": float(self._absolute_max_start.value()),
            "filter_hand_traces": bool(self._filter_hand_traces.isChecked()),
            "filter_cutoff_frequency_hz": float(self._filter_cutoff.value()),
        }

    def _update_detection_method_controls(self) -> None:
        if not hasattr(self, "_max_thresh"):
            return
        use_absolute = self._detection_method() == "absolute"
        for widget in (self._max_thresh, self._min_travel, self._max_dist_home):
            widget.setEnabled(not use_absolute)
        for widget in (
            self._absolute_axis_combo,
            self._absolute_invert,
            self._absolute_max_start,
        ):
            widget.setEnabled(use_absolute)

    def _create_results_groupbox(self) -> QGroupBox:
        gb = QGroupBox("Detected Reaches")
        layout = QVBoxLayout()

        self._results_table = QTableWidget(0, 5)
        self._results_table.setHorizontalHeaderLabels(
            ["Start", "Max", "End", "Duration", "Outcome"]
        )
        self._results_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeToContents
        )
        self._results_table.horizontalHeader().setStretchLastSection(True)
        self._results_table.verticalHeader().setVisible(False)
        self._results_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._results_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._results_table.cellClicked.connect(self._jump_to_reach)
        layout.addWidget(self._results_table)

        outcome_row = QHBoxLayout()
        outcome_row.addWidget(QLabel("Outcome:"))
        from sleap.gui.reach_detection import ReachOutcome
        self._outcome_combo = QComboBox()
        for outcome in ReachOutcome.selectable():
            self._outcome_combo.addItem(outcome.label, int(outcome))
        outcome_row.addWidget(self._outcome_combo)
        assign_btn = QPushButton("Assign")
        assign_btn.clicked.connect(self._assign_outcome)
        outcome_row.addWidget(assign_btn)
        outcome_widget = QWidget()
        outcome_widget.setLayout(outcome_row)
        layout.addWidget(outcome_widget)

        edit_row = QHBoxLayout()
        del_btn = QPushButton("Delete")
        del_btn.clicked.connect(self._delete_selected)
        edit_row.addWidget(del_btn)
        del_all_btn = QPushButton("Delete All")
        del_all_btn.clicked.connect(self._delete_all)
        edit_row.addWidget(del_all_btn)
        edit_widget = QWidget()
        edit_widget.setLayout(edit_row)
        layout.addWidget(edit_widget)

        save_btn = QPushButton("Save Reaches...")
        save_btn.clicked.connect(self._save_reaches)
        layout.addWidget(save_btn)

        self._status_label = QLabel("")
        self._status_label.setWordWrap(True)
        layout.addWidget(self._status_label)

        gb.setLayout(layout)
        return gb

    def _populate_table(self) -> None:
        self._results_table.setRowCount(len(self._reaches))
        for row, r in enumerate(self._reaches):
            self._results_table.setItem(row, 0, QTableWidgetItem(str(r.frame + 1)))
            self._results_table.setItem(row, 1, QTableWidgetItem(str(r.max_frame + 1)))
            self._results_table.setItem(row, 2, QTableWidgetItem(str(r.end_frame + 1)))
            self._results_table.setItem(row, 3, QTableWidgetItem(str(r.dur)))
            outcome_item = QTableWidgetItem(r.outcome.label)
            color = QtGui.QColor(r.outcome.ui_color)
            outcome_item.setBackground(color)
            r_lum = 0.299 * color.red()
            g_lum = 0.587 * color.green()
            b_lum = 0.114 * color.blue()
            luminance = r_lum + g_lum + b_lum
            outcome_item.setForeground(
                QtGui.QColor("black" if luminance > 160 else "white")
            )
            self._results_table.setItem(row, 4, outcome_item)

    def upsert_reach(self, row: int, start: int, max_frame: int, end: int) -> None:
        """Add or edit a curated reach from timeline-selected frame controls."""
        from sleap.gui.reach_detection import ReachOutcome, ReachSegment

        start = int(start)
        max_frame = int(max_frame)
        end = int(end)
        reach = ReachSegment(
            frame=start,
            max_delta=max_frame - start,
            dur=end - start,
            outcome=ReachOutcome.UNCLASSIFIED,
        )
        if not reach.is_valid:
            self._status_label.setText(
                "Reach was not updated: start must be before max and end."
            )
            return

        row = int(row)
        old_detail = None
        if 0 <= row < len(self._reaches):
            old_reach = self._reaches[row]
            reach = reach._replace(
                outcome=old_reach.outcome,
                hand_pos=getattr(old_reach, "hand_pos", 0),
            )
            if row < len(self._reach_details):
                old_detail = dict(self._reach_details[row])
            self._reaches[row] = reach
            action = "Updated"
        else:
            self._reaches.append(reach)
            action = "Added"

        new_detail = self._detail_for_curated_reach(
            reach,
            old_detail=old_detail,
            action="manual_edit" if action == "Updated" else "manual_add",
        )
        if 0 <= row < len(self._reach_details):
            self._reach_details[row] = new_detail
        elif self._reach_details:
            self._reach_details.append(new_detail)

        self._sort_reaches_and_details()
        self._populate_table()
        self._notify_reaches_changed()

        selected_row = next(
            (
                idx
                for idx, candidate in enumerate(self._reaches)
                if candidate.frame == reach.frame
                and candidate.max_frame == reach.max_frame
                and candidate.end_frame == reach.end_frame
            ),
            None,
        )
        if selected_row is not None:
            self._results_table.selectRow(selected_row)
        self._status_label.setText(
            f"{action} reach: F{reach.frame + 1}-F{reach.end_frame + 1}."
        )

    def _detail_for_curated_reach(
        self,
        reach,
        old_detail: Optional[dict] = None,
        action: str = "manual_edit",
    ) -> dict:
        detail = dict(old_detail or {})
        detail.update(
            {
                "frame": int(reach.frame),
                "max_frame": int(reach.max_frame),
                "end_frame": int(reach.end_frame),
                "dur": int(reach.dur),
                "result": reach.outcome.name,
                "detection_method": detail.get("detection_method", action),
                "curation": action,
                "first_peak_frame": int(reach.max_frame),
                "is_multi_reach": bool(detail.get("is_multi_reach", False)),
            }
        )
        return detail

    def _sort_reaches_and_details(self) -> None:
        if self._reach_details and len(self._reach_details) == len(self._reaches):
            paired = sorted(
                zip(self._reaches, self._reach_details),
                key=lambda pair: pair[0].frame,
            )
            self._reaches = [reach for reach, _ in paired]
            self._reach_details = [detail for _, detail in paired]
        else:
            self._reaches.sort(key=lambda reach: reach.frame)

    # ── table actions ─────────────────────────────────────────────────────── #

    def _jump_to_reach(self, row: int, column: int) -> None:
        if row < len(self._reaches):
            self.main_window.state["frame_idx"] = self._reaches[row].frame

    def _assign_outcome(self) -> None:
        from sleap.gui.reach_detection import ReachOutcome
        rows = set(idx.row() for idx in self._results_table.selectedIndexes())
        if not rows:
            return
        outcome = ReachOutcome(int(self._outcome_combo.currentData()))
        self._reaches = [
            r._replace(outcome=outcome) if i in rows else r
            for i, r in enumerate(self._reaches)
        ]
        for row in rows:
            if 0 <= row < len(self._reach_details):
                self._reach_details[row]["result"] = outcome.name
        self._populate_table()
        self._notify_reaches_changed()

    def _delete_selected(self) -> None:
        rows = sorted(
            set(idx.row() for idx in self._results_table.selectedIndexes()),
            reverse=True,
        )
        for row in rows:
            if 0 <= row < len(self._reaches):
                self._reaches.pop(row)
            if 0 <= row < len(self._reach_details):
                self._reach_details.pop(row)
        self._populate_table()
        self._notify_reaches_changed()
        self._status_label.setText(f"{len(self._reaches)} reach(es) remaining.")

    def delete_reach(self, row: int) -> None:
        """Delete one reach selected from the timeline."""
        row = int(row)
        if not 0 <= row < len(self._reaches):
            return
        self._reaches.pop(row)
        if row < len(self._reach_details):
            self._reach_details.pop(row)
        self._populate_table()
        self._notify_reaches_changed()
        self._status_label.setText(
            f"Deleted reach. {len(self._reaches)} reach(es) remaining."
        )

    def _delete_all(self) -> None:
        self._reaches = []
        self._reach_details = []
        self._populate_table()
        self._notify_reaches_changed()
        self._status_label.setText("All reaches cleared.")

    def _save_reaches(self) -> None:
        from sleap.gui.reach_detection import (
            save_kpn_reach_details,
            save_kpn_reach_details_csv,
            save_kpn_reach_details_table,
            save_pellet_history,
            save_reach_detection_info,
            save_reaches,
        )
        filename, _ = FileDialog.save(
            self,
            caption="Save reaches",
            dir="detected_reaches.txt",
            filter="Text Files (*.txt);;All Files (*)",
        )
        if not filename:
            return
        save_reaches(self._reaches, filename)
        out_dir = Path(filename).parent
        if self._reach_details:
            save_kpn_reach_details(
                self._reach_details, out_dir / "kpn_reach_details.json"
            )
            save_kpn_reach_details_table(
                self._reach_details, out_dir / "kpn_reach_details.txt"
            )
            save_kpn_reach_details_csv(
                self._reach_details, out_dir / "kpn_reach_details.csv"
            )
        if self._pellet_history:
            save_pellet_history(self._pellet_history, out_dir / "pelletHistory.pickle")
        if self._reach_detection_info:
            save_reach_detection_info(
                self._reach_detection_info, out_dir / "reach_detection_info.json"
            )
        self._status_label.setText(f"Saved {len(self._reaches)} reach(es).")

    # ── detection ─────────────────────────────────────────────────────────── #

    def _filter_hand_trajectories(self, lh_traj, rh_traj, frame_rate: float):
        """Apply optional first-order Butterworth filtering to LH/RH traces."""
        import numpy as np

        enabled = (
            hasattr(self, "_filter_hand_traces")
            and self._filter_hand_traces.isChecked()
        )
        cutoff = (
            float(self._filter_cutoff.value())
            if hasattr(self, "_filter_cutoff")
            else 30.0
        )
        info = {
            "enabled": bool(enabled),
            "cutoff_frequency_hz": float(cutoff),
            "sampling_rate_hz": float(frame_rate),
            "order": 1,
            "normalization": "cutoff_frequency_hz / sampling_rate_hz",
            "filtered_traces": [],
        }

        if hasattr(self, "_filter_sampling_label"):
            self._filter_sampling_label.setText(f"{float(frame_rate):.3g} Hz")

        if not enabled:
            return lh_traj, rh_traj, info
        if not np.isfinite(frame_rate) or frame_rate <= 0:
            raise ValueError("Cannot filter hand traces without a positive sampling rate.")
        if cutoff <= 0 or cutoff >= frame_rate:
            raise ValueError(
                "Butterworth cutoff frequency must be greater than 0 and less "
                "than the sampling rate."
            )

        lh_filtered = self._butterworth_filter_trace(lh_traj, cutoff, frame_rate)
        rh_filtered = self._butterworth_filter_trace(rh_traj, cutoff, frame_rate)
        info["filtered_traces"] = ["left_hand", "right_hand"]
        return lh_filtered, rh_filtered, info

    @staticmethod
    def _butterworth_filter_trace(trace, cutoff_frequency: float, sampling_rate: float):
        """Filter each coordinate axis while preserving the original NaN mask."""
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

    def _run_detection(self) -> None:
        import numpy as np
        from sleap.gui.reach_detection import (
            detect_reaches_absolute,
            extract_hand_position_3d_with_confidence,
            detect_reaches_kpn,
            suggest_pellet_nodes,
        )
        from sleap.gui.reach_projection import extract_points3d_position
        from sleap.gui.reach_projection import extract_reprojection_confidence
        from sleap.gui.reach_projection import extract_reprojection_position
        from qtpy.QtWidgets import QApplication

        self._notify_reach_traces_changed([])
        lh_nodes = self._selected_lh_nodes()
        rh_nodes = self._selected_rh_nodes()
        detection_method = self._detection_method()
        if not rh_nodes:
            self._status_label.setText("Check R_HAND nodes.")
            return
        if detection_method == "from_pellet" and not lh_nodes:
            self._status_label.setText("Check R_HAND and L_HAND nodes.")
            return

        src_video = self.main_window.state.get("video", default=None)
        src_labels = self.main_window.labels
        points3d_source = self._points3d_source

        # Determine source: points3D file, external predictions file, or project labels
        if points3d_source is not None:
            self._pred_source_video_stem = None
            prediction_source = {
                "source_type": "points3d_file",
                "path": points3d_source.get("path", self._points3d_path_edit.text()),
                "coordinate_system": "3d_calibration_mm",
                "metadata": points3d_source.get("metadata", {}),
            }
        elif (self._pred_labels_obj is not None
                and self._pred_cam_combo.currentIndex() >= 0):
            src_labels = self._pred_labels_obj
            cam_idx = self._pred_cam_combo.currentIndex()
            src_video = self._pred_labels_obj.videos[cam_idx]
            self._pred_source_video_stem = ReachesDock._video_stem(src_video)
            prediction_source = {
                "source_type": "external_predictions_file",
                "path": self._pred_path_edit.text(),
                "camera_index": int(cam_idx),
                "camera": self._pred_cam_combo.currentText(),
            }
        else:
            src_labels = self.main_window.labels
            src_video = self.main_window.state.get("video", default=None)
            self._pred_source_video_stem = None
            prediction_source = {
                "source_type": "project_labels",
                "path": str(self.main_window.state.get("filename", default="") or ""),
                "camera_index": None,
                "camera": None,
            }

        if points3d_source is None and (src_video is None or src_labels is None):
            self._status_label.setText("No video/labels source available.")
            return

        self._status_label.setText("Extracting trajectories...")
        QApplication.processEvents()
        point_confidence = float(self._point_confidence.value())

        if points3d_source is not None:
            node_names = list(points3d_source.get("node_names", []))
            points3d = points3d_source.get("points3d")
            pellet_nodes = suggest_pellet_nodes(node_names)
            if detection_method == "from_pellet" and not pellet_nodes:
                self._status_label.setText("No PELLET node found in the points3D file.")
                return
            rh_traj = extract_points3d_position(points3d, node_names, rh_nodes)
            lh_traj = (
                extract_points3d_position(points3d, node_names, lh_nodes)
                if lh_nodes
                else np.full_like(rh_traj, np.nan, dtype=np.float64)
            )
            pellet_traj = (
                extract_points3d_position(points3d, node_names, pellet_nodes)
                if pellet_nodes
                else np.full_like(rh_traj, np.nan, dtype=np.float64)
            )
            point_scores = (
                self._points3d_reprojections.get("point_scores")
                if self._points3d_reprojections is not None
                else None
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
        else:
            # Extract fixed-camera trajectories from the predictions source.
            rh_traj, rh_conf = extract_hand_position_3d_with_confidence(
                src_labels,
                src_video,
                rh_nodes,
                min_confidence=point_confidence,
            )
            if lh_nodes:
                lh_traj, lh_conf = extract_hand_position_3d_with_confidence(
                    src_labels,
                    src_video,
                    lh_nodes,
                    min_confidence=point_confidence,
                )
            else:
                lh_traj = np.full_like(rh_traj, np.nan, dtype=np.float64)
                lh_conf = None
            pellet_nodes = suggest_pellet_nodes(self._skeleton_node_names())
            if detection_method == "from_pellet" and not pellet_nodes:
                self._status_label.setText("No PELLET node found in the predictions skeleton.")
                return
            if pellet_nodes:
                pellet_traj, pellet_conf = extract_hand_position_3d_with_confidence(
                    src_labels,
                    src_video,
                    pellet_nodes,
                    min_confidence=point_confidence,
                )
            else:
                pellet_traj = np.full_like(rh_traj, np.nan, dtype=np.float64)
                pellet_conf = None

        frame_rate = self._video_frame_rate(src_video) if src_video is not None else 30.0
        try:
            lh_traj, rh_traj, filter_info = self._filter_hand_trajectories(
                lh_traj, rh_traj, frame_rate
            )
        except ValueError as exc:
            self._status_label.setText(str(exc))
            return

        self._notify_reach_traces_changed(
            self._make_reach_parameter_traces(
                right_hand=rh_traj,
                pellet=pellet_traj,
                right_hand_confidence=rh_conf,
                pellet_confidence=pellet_conf,
                pellet_nodes=pellet_nodes,
                confidence=point_confidence,
            )
        )

        dims_to_check = 3 if points3d_source is not None else 2
        lh_valid = (
            int(np.sum(
                np.all(np.isfinite(lh_traj[:, :dims_to_check]), axis=1)
                & self._confidence_ok(lh_conf, len(lh_traj), point_confidence)
            ))
            if lh_traj.size else 0
        )
        rh_valid = (
            int(np.sum(
                np.all(np.isfinite(rh_traj[:, :dims_to_check]), axis=1)
                & self._confidence_ok(rh_conf, len(rh_traj), point_confidence)
            ))
            if rh_traj.size else 0
        )
        pellet_valid = (
            int(np.sum(
                np.all(np.isfinite(pellet_traj[:, :dims_to_check]), axis=1)
                & self._confidence_ok(pellet_conf, len(pellet_traj), point_confidence)
            ))
            if pellet_traj.size else 0
        )

        missing_required_data = rh_valid == 0 or (
            detection_method == "from_pellet" and (lh_valid == 0 or pellet_valid == 0)
        )
        if missing_required_data:
            required = "R_HAND, L_HAND, or PELLET" if detection_method == "from_pellet" else "R_HAND"
            self._status_label.setText(
                f"Missing {required} data. "
                "Check that the predictions file/camera matches the skeleton."
            )
            return

        reprojection_source_info = None
        reprojection_note = ""

        # Store fixed-camera 2D trajectories for marker overlay. When detection
        # uses points3D, draw from reprojections.h5 rather than calibration space.
        if points3d_source is None:
            self._lh_traj2d = lh_traj[:, :2]
            self._rh_traj2d = rh_traj[:, :2]
            self._pellet_traj2d = pellet_traj[:, :2]
        else:
            camera_idx = self._points3d_reprojection_camera_index()
            if self._points3d_reprojections is not None and camera_idx is not None:
                reproj = self._points3d_reprojections
                reproj_nodes = list(reproj.get("node_names", []))
                reproj_data = reproj.get("reprojections")
                self._lh_traj2d = extract_reprojection_position(
                    reproj_data, camera_idx, reproj_nodes, lh_nodes
                )
                self._rh_traj2d = extract_reprojection_position(
                    reproj_data, camera_idx, reproj_nodes, rh_nodes
                )
                self._pellet_traj2d = extract_reprojection_position(
                    reproj_data, camera_idx, reproj_nodes, pellet_nodes
                )
                camera_names = list(reproj.get("camera_names", []))
                camera_name = (
                    camera_names[camera_idx]
                    if 0 <= camera_idx < len(camera_names)
                    else str(camera_idx)
                )
                reprojection_source_info = {
                    "path": reproj.get("path", ""),
                    "camera_index": int(camera_idx),
                    "camera": camera_name,
                }
                reprojection_note = f" | markers: {camera_name} reprojection"
            else:
                self._lh_traj2d = None
                self._rh_traj2d = None
                self._pellet_traj2d = None
                reprojection_note = " | markers unavailable: no matching reprojection"

        method_label = "absolute KPN peaks" if detection_method == "absolute" else "KPN outward peaks"
        self._status_label.setText(
            f"Trajectories: RH {rh_valid}, LH {lh_valid}, pellet {pellet_valid} "
            f"frames - detecting {method_label}..."
        )
        QApplication.processEvents()

        # Fetch session events for pellet-based outcome classification
        from sleap.gui.session_events import get_video_session_events
        events = (
            get_video_session_events(self.main_window.labels, src_video)
            if src_video is not None
            else []
        )
        if not events:
            # Also try matching via the current GUI video when using pred labels
            cur_video = self.main_window.state.get("video", default=None)
            if cur_video is not None and cur_video is not src_video:
                events = get_video_session_events(
                    self.main_window.labels, cur_video
                )
        has_events = bool(events)

        detection_parameters = {
            "method": detection_method,
            "kpn_outward_threshold": float(self._min_thresh.value()),
            "kpn_outward_max_threshold": float(self._max_thresh.value()),
            "kpn_peak_prominence": float(self._prominence.value()),
            "kpn_min_outward_travel": float(self._min_travel.value()),
            "kpn_start_padding": int(self._start_padding.value()),
            "min_frame": int(self._min_frames.value()),
            "max_frame": int(self._max_frames.value()),
            "frame_rate": float(frame_rate),
            "max_dist_from_home": float(self._max_dist_home.value()),
            "point_confidence_threshold": float(point_confidence),
            "hand_confidence_aggregation": "median",
            "filter": filter_info,
        }
        if detection_method == "absolute":
            axis_idx = int(self._absolute_axis_combo.currentData())
            axis_name = str(self._absolute_axis_combo.currentText()).lower()
            absolute_signal = np.asarray(rh_traj[:, axis_idx], dtype=np.float64)
            if self._absolute_invert.isChecked():
                absolute_signal = -absolute_signal
            detection_parameters.update(
                {
                    "absolute_axis": axis_name,
                    "absolute_invert": bool(self._absolute_invert.isChecked()),
                    "absolute_max_start_value": float(self._absolute_max_start.value()),
                }
            )
            reaches, details, pellet_history = detect_reaches_absolute(
                absolute_signal,
                right_hand=rh_traj,
                left_hand=lh_traj,
                pellet=pellet_traj if pellet_nodes else None,
                events=events or None,
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
                detail["absolute_invert"] = bool(self._absolute_invert.isChecked())
        else:
            reaches, details, pellet_history = detect_reaches_kpn(
                right_hand=rh_traj,
                left_hand=lh_traj,
                pellet=pellet_traj,
                events=events or None,
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
        self._reaches = reaches
        self._reach_details = details
        self._pellet_history = pellet_history
        self._reach_detection_info = {
            "prediction_source": prediction_source,
            "reprojection_source": reprojection_source_info,
            "video": {
                "filename": str(getattr(src_video, "filename", "") or ""),
                "frames": int(len(src_video)) if src_video is not None else None,
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
                "count": int(len(events)),
                "used_for_pellet_availability": bool(
                    events and len(pellet_history) > 1
                ),
            },
            "outputs": {
                "reach_count": int(len(reaches)),
                "pellet_epoch_count": int(len(pellet_history) - 1 if pellet_history else 0),
                "detail_count": int(len(details)),
            },
        }
        self._populate_table()
        self._notify_reaches_changed()
        self._update_frame_markers()
        events_note = (
            f" | {len(events)} events"
            if has_events and pellet_history
            else " | no events file"
            if pellet_history
            else ""
        )
        if detection_method == "absolute":
            classified_count = sum(
                1
                for reach in self._reaches
                if reach.outcome.name != "UNCLASSIFIED"
            )
            self._status_label.setText(
                f"{len(self._reaches)} reach(es) detected with absolute KPN peaks - "
                f"{classified_count} classified, "
                f"{len(pellet_history) - 1 if pellet_history else 0} pellet epoch(s)"
                f"{events_note}{reprojection_note}"
            )
        else:
            self._status_label.setText(
                f"{len(self._reaches)} reach(es) detected with KPN outward peaks - "
                f"{len(pellet_history) - 1 if pellet_history else 0} pellet epoch(s)"
                f"{events_note}{reprojection_note}"
            )

    # ── internal helpers ──────────────────────────────────────────────────── #

    def _load_associated_reprojections(self, points3d_source):
        """Load the reprojections.h5 file associated with a points3D source."""
        from sleap.gui.reach_projection import load_reprojections_h5

        metadata = points3d_source.get("metadata", {}) or {}
        candidates = []
        if metadata.get("reprojections_path"):
            candidates.append(Path(metadata["reprojections_path"]))
        points_path = Path(points3d_source.get("path", ""))
        if points_path.name:
            candidates.append(points_path.with_name("reprojections.h5"))

        for candidate in candidates:
            try:
                if candidate.exists():
                    return load_reprojections_h5(candidate)
            except Exception:
                continue
        return None

    def _points3d_reprojection_camera_index(self) -> Optional[int]:
        """Return the reprojection camera index matching the current video."""
        reproj = self._points3d_reprojections
        if reproj is None:
            return None

        n_cameras = int(reproj["reprojections"].shape[0])
        current_video = self.main_window.state.get("video", default=None)
        current_stem = self._video_stem(current_video)
        current_token = self._camera_token(current_stem)

        camera_names = list(reproj.get("camera_names", []))
        if current_token:
            token_matches = [
                idx
                for idx, camera_name in enumerate(camera_names)
                if self._camera_token(camera_name) == current_token
            ]
            if len(token_matches) == 1:
                return token_matches[0]

        if current_stem:
            norm_stem = self._normalize_camera_name(current_stem)
            for idx, camera_name in enumerate(camera_names):
                norm_camera = self._normalize_camera_name(camera_name)
                if norm_camera and (
                    norm_camera in norm_stem or norm_stem in norm_camera
                ):
                    return idx

        source_files = list((reproj.get("metadata", {}) or {}).get("prediction_files", []))
        if current_stem:
            norm_stem = self._normalize_camera_name(current_stem)
            for source_file in source_files[:n_cameras]:
                source_token = self._camera_token(Path(source_file).stem)
                if source_token:
                    token_matches = [
                        idx
                        for idx, camera_name in enumerate(camera_names)
                        if self._camera_token(camera_name) == source_token
                    ]
                    if source_token == current_token and len(token_matches) == 1:
                        return token_matches[0]
                    continue
                norm_source = self._normalize_camera_name(Path(source_file).stem)
                if norm_source and (
                    norm_stem in norm_source or norm_source in norm_stem
                ):
                    for idx, camera_name in enumerate(camera_names):
                        norm_camera = self._normalize_camera_name(camera_name)
                        if norm_camera and (
                            norm_camera in norm_source or norm_source in norm_camera
                        ):
                            return idx

        session = get_session_for_video(self.main_window.labels, current_video)
        videos = list(getattr(session, "videos", []) or [])
        for idx, video in enumerate(videos[:n_cameras]):
            if video is current_video:
                return idx

        if n_cameras == 1:
            return 0
        return None

    @staticmethod
    def _video_stem(video) -> str:
        if video is None:
            return ""
        filename = getattr(video, "filename", "")
        if isinstance(filename, list):
            filename = filename[0] if filename else ""
        return Path(str(filename)).stem if filename else ""

    @staticmethod
    def _normalize_camera_name(value: str) -> str:
        return "".join(ch for ch in str(value).lower() if ch.isalnum())

    @staticmethod
    def _camera_token(value: str) -> str:
        import re

        match = re.search(r"cam\s*0*(\d+)", str(value or ""), flags=re.IGNORECASE)
        if not match:
            return ""
        return f"cam{int(match.group(1)):03d}"

    def _video_frame_rate(self, video) -> float:
        """Best-effort frame-rate lookup for time-based KPN pellet epochs."""
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

    @staticmethod
    def _confidence_ok(confidence, n_frames: int, threshold: float):
        import numpy as np

        if confidence is None:
            return np.ones(n_frames, dtype=bool)
        conf = np.asarray(confidence, dtype=np.float64).reshape(-1)
        out = np.zeros(n_frames, dtype=bool)
        take = min(n_frames, conf.size)
        if take > 0:
            out[:take] = np.isfinite(conf[:take]) & (conf[:take] >= threshold)
        return out

    def _make_reach_parameter_traces(
        self,
        *,
        right_hand,
        pellet,
        right_hand_confidence,
        pellet_confidence,
        pellet_nodes: List[str],
        confidence: float,
    ) -> List[dict]:
        """Build timeline-aligned traces used for reach-parameter tuning."""
        import numpy as np

        traces: List[dict] = []
        rh = np.asarray(right_hand, dtype=np.float64)
        if rh.ndim != 2 or rh.size == 0:
            return traces

        n = rh.shape[0]
        rh_ok = self._confidence_ok(right_hand_confidence, n, confidence)
        rh_x = np.full(n, np.nan, dtype=np.float64)
        if rh.shape[1] >= 1:
            rh_x[:] = rh[:, 0]
            rh_x[~rh_ok] = np.nan
            traces.append(
                {
                    "name": "RH x",
                    "values": rh_x,
                    "color": "#34d399",
                }
            )

        pellet_arr = np.asarray(pellet, dtype=np.float64)
        if not pellet_nodes or pellet_arr.ndim != 2 or pellet_arr.size == 0:
            return traces

        n = min(n, pellet_arr.shape[0])
        if n <= 0:
            return traces
        rh_xyz = np.full((n, 3), np.nan, dtype=np.float64)
        pellet_xyz = np.full((n, 3), np.nan, dtype=np.float64)
        rh_cols = min(3, rh.shape[1])
        pellet_cols = min(3, pellet_arr.shape[1])
        rh_xyz[:, :rh_cols] = rh[:n, :rh_cols]
        pellet_xyz[:, :pellet_cols] = pellet_arr[:n, :pellet_cols]
        if rh_cols == 2:
            rh_xyz[:, 2] = 0.0
        if pellet_cols == 2:
            pellet_xyz[:, 2] = 0.0

        pellet_ok = self._confidence_ok(pellet_confidence, n, confidence)
        pellet_ok &= np.all(np.isfinite(pellet_xyz), axis=1)
        if not np.any(pellet_ok):
            return traces

        pellet_home = np.nanmedian(pellet_xyz[pellet_ok], axis=0)
        distance = np.sqrt(np.sum((rh_xyz - pellet_home) ** 2, axis=1))
        distance[~rh_ok[:n]] = np.nan
        distance[~np.all(np.isfinite(rh_xyz), axis=1)] = np.nan
        traces.insert(
            0,
            {
                "name": "RH pellet dist",
                "values": distance,
                "color": "#a3e635",
            },
        )
        return traces

    def _notify_reaches_changed(self) -> None:
        if callable(self.on_reaches_changed):
            self.on_reaches_changed(self._reaches)

    def _notify_reach_traces_changed(self, traces: Optional[List[dict]]) -> None:
        if callable(self.on_reach_traces_changed):
            self.on_reach_traces_changed(traces or [])

    def _update_frame_markers(self) -> None:
        """Push the current-frame LH/RH/pellet (x,y) positions into the view."""
        import numpy as np
        try:
            player = self.main_window.player
            view = player.view
            secondary_view = player.secondary_view
        except AttributeError:
            return

        def _clear_all():
            view.set_hand_markers(None, None, None)
            secondary_view.set_hand_markers(None, None, None)

        show = (
            hasattr(self, "_show_markers_check")
            and self._show_markers_check.isChecked()
        )
        if not show:
            _clear_all()
            return

        frame_idx = self.main_window.state.get("frame_idx", default=None)
        if frame_idx is None:
            _clear_all()
            return

        def _xy(traj2d):
            if traj2d is None or traj2d.size == 0:
                return None
            if frame_idx >= len(traj2d):
                return None
            xy = traj2d[frame_idx]
            if np.any(np.isnan(xy)):
                return None
            return xy

        lh = _xy(self._lh_traj2d)
        rh = _xy(self._rh_traj2d)
        pellet = _xy(self._pellet_traj2d)

        pred_stem = self._pred_source_video_stem
        if pred_stem:
            primary_stem = self._video_stem(getattr(player, "video", None))
            secondary_stem = self._video_stem(getattr(player, "secondary_video", None))
            if pred_stem == primary_stem:
                view.set_hand_markers(lh, rh, pellet)
                secondary_view.set_hand_markers(None, None, None)
            elif pred_stem == secondary_stem:
                secondary_view.set_hand_markers(lh, rh, pellet)
                view.set_hand_markers(None, None, None)
            else:
                _clear_all()
        else:
            view.set_hand_markers(lh, rh, pellet)

    @property
    def reaches(self) -> List:
        return self._reaches
