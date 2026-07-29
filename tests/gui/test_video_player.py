from types import SimpleNamespace

import sleap_io as sio

from sleap.gui.state import GuiState
from sleap.gui.reach_detection import ReachOutcome, ReachSegment
from sleap.gui.widgets.video import (
    GraphicsView,
    QtVideoPlayer,
    QtTextWithBackground,
    VisibleBoundingBox,
    QtInstance,
)
from sleap.sleap_io_adaptors.lf_labels_utils import labels_add_video

from qtpy import QtCore, QtWidgets
from qtpy.QtGui import QColor, QMouseEvent, QWheelEvent
import numpy as np


class FakeGraphicsMouseEvent:
    def __init__(self, pos, modifiers=QtCore.Qt.NoModifier):
        self._pos = pos
        self._modifiers = modifiers
        self.accepted = False

    def button(self):
        return QtCore.Qt.LeftButton

    def buttons(self):
        return QtCore.Qt.LeftButton

    def modifiers(self):
        return self._modifiers

    def pos(self):
        return self._pos

    def accept(self):
        self.accepted = True


def _qt_instance_node_positions(qt_instance):
    return {
        name: (node.scenePos().x(), node.scenePos().y())
        for name, node in qt_instance.nodes.items()
    }


def test_context_menu_add_instance_actions_ignore_checked_arg():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    assert app is not None
    calls = []

    class Context:
        labels = None

        def newInstance(self, **kwargs):
            calls.append(kwargs)

    vp = QtVideoPlayer(context=Context())
    try:
        scene_pos = QtCore.QPointF(12, 34)
        vp.create_contextual_menu(scene_pos, target_frame_idx=7)

        for action_name in (
            "Default",
            "Average",
            "Force Directed",
            "Copy Prior Frame",
            "Random",
        ):
            vp._menu_actions[action_name].trigger()

        assert [call["init_method"] for call in calls] == [
            "best",
            "template",
            "force_directed",
            "prior_frame",
            "random",
        ]
        assert calls[0]["location"] == scene_pos
        assert calls[1]["location"] == scene_pos
        assert calls[2]["location"] == scene_pos
        assert "location" not in calls[3]
        assert calls[4]["location"] == scene_pos
        assert all(call["target_frame_idx"] == 7 for call in calls)
    finally:
        vp.cleanup()


def test_gui_video(qtbot):
    vp = QtVideoPlayer()
    vp.show()
    qtbot.addWidget(vp)

    assert vp.close()


def test_timeline_header_event_controls_navigate(qtbot):
    state = GuiState()
    state["frame_idx"] = 0
    vp = QtVideoPlayer(state=state)
    qtbot.addWidget(vp)

    vp.zoomed_timeline.set_total_frames(50)
    vp.zoomed_timeline.set_events(
        [
            {"event": "delivery", "frame": 3, "color": (255, 0, 0)},
            {"event": "reach", "frame": 5, "color": (0, 255, 0)},
            {"event": "reach", "frame": 20, "color": (0, 255, 0)},
        ]
    )

    assert vp._tl_event_combo.isEnabled()
    assert vp._tl_event_combo.findText("delivery") >= 0
    assert vp._tl_event_combo.findText("reach") >= 0

    vp._tl_event_combo.setCurrentText("reach")
    vp._jump_timeline_event(1)
    assert state["frame_idx"] == 5

    vp._jump_timeline_event(1)
    assert state["frame_idx"] == 20

    vp._jump_timeline_event(-1)
    assert state["frame_idx"] == 5

    vp.zoomed_timeline.set_events([])
    assert not vp._tl_event_combo.isEnabled()
    assert not vp._tl_prev_event_btn.isEnabled()
    assert not vp._tl_next_event_btn.isEnabled()

    vp.cleanup()
    assert vp.close()


def test_reach_trace_plot_matches_timeline_span_and_frame():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    assert app is not None
    state = GuiState()
    state["frame_idx"] = 0
    vp = QtVideoPlayer(state=state)
    try:
        vp.reach_trace_plot.set_total_frames(20)
        vp.reach_trace_plot.set_traces(
            [
                {
                    "name": "RH x",
                    "values": np.arange(20, dtype=np.float64),
                    "color": "#34d399",
                }
            ]
        )

        vp._set_timeline_span(50)
        assert vp.reach_trace_plot._span == 50
        assert vp.zoomed_timeline._span == 50

        state["frame_idx"] = 7
        assert vp.reach_trace_plot._curr_frame == 7
        assert vp.zoomed_timeline._curr_frame == 7
        assert "RH x: 7.00" in vp.reach_trace_plot._value_label.text()
        label_items = [
            item
            for item in vp.reach_trace_plot._trace_items
            if isinstance(item, QtWidgets.QGraphicsSimpleTextItem)
        ]
        assert label_items[-1].brush().color().name() == "#34d399"
    finally:
        vp.cleanup()
        assert vp.close()


def test_timeline_reach_edit_uses_three_clicks_without_frame_jumps(qtbot):
    state = GuiState()
    state["frame_idx"] = 50
    vp = QtVideoPlayer(state=state)
    qtbot.addWidget(vp)
    timeline = vp.zoomed_timeline
    timeline.resize(400, timeline.height())
    timeline.set_total_frames(150)
    timeline.set_span(50)

    timeline._start_reach_edit()
    assert timeline._reach_edit_stage == "start"

    start_event = FakeGraphicsMouseEvent(
        QtCore.QPointF(timeline._frame_to_x(60), 10)
    )
    timeline._handle_reach_edit_click(start_event)
    assert timeline._reach_edit_stage == "max"
    assert timeline._reach_edit_frames == {"start": 60}
    assert state["frame_idx"] == 60

    max_event = FakeGraphicsMouseEvent(
        QtCore.QPointF(timeline._frame_to_x(70), 10)
    )
    timeline._handle_reach_edit_click(max_event)
    assert timeline._reach_edit_stage == "end"
    assert timeline._reach_edit_frames == {"start": 60, "max": 70}
    assert state["frame_idx"] == 70

    with qtbot.waitSignal(timeline.reachEditRequested) as emitted:
        end_event = FakeGraphicsMouseEvent(
            QtCore.QPointF(timeline._frame_to_x(80), 10)
        )
        timeline._handle_reach_edit_click(end_event)

    assert emitted.args == [-1, 60, 70, 80]
    assert not timeline._reach_edit_active
    assert state["frame_idx"] == 80

    vp.cleanup()
    assert vp.close()


def test_timeline_reach_edit_scrubs_and_edits_current_reach(qtbot):
    state = GuiState()
    state["frame_idx"] = 50
    vp = QtVideoPlayer(state=state)
    qtbot.addWidget(vp)
    timeline = vp.zoomed_timeline
    timeline.resize(400, timeline.height())
    timeline.set_total_frames(150)
    timeline.set_span(50)
    timeline.set_reaches(
        [ReachSegment(40, 20, 40, ReachOutcome.UNCLASSIFIED)]
    )

    timeline._start_reach_edit()
    hover_x = timeline._frame_to_x(55)
    move_event = QMouseEvent(
        QtCore.QEvent.MouseMove,
        QtCore.QPointF(hover_x, 10),
        QtCore.Qt.NoButton,
        QtCore.Qt.NoButton,
        QtCore.Qt.NoModifier,
    )
    timeline.mouseMoveEvent(move_event)

    assert state["frame_idx"] == 55
    assert timeline._x_to_frame(hover_x) == 55

    start_event = FakeGraphicsMouseEvent(QtCore.QPointF(hover_x, 10))
    timeline._handle_reach_edit_click(start_event)
    assert timeline._reach_edit_index == 0
    assert timeline._reach_edit_stage == "max"
    assert timeline._reach_edit_frames == {"start": 55, "max": 60, "end": 80}
    assert state["frame_idx"] == 55

    vp.cleanup()
    assert vp.close()


def test_timeline_reach_delete_mode_emits_clicked_reach(qtbot):
    state = GuiState()
    state["frame_idx"] = 50
    vp = QtVideoPlayer(state=state)
    qtbot.addWidget(vp)
    timeline = vp.zoomed_timeline
    timeline.resize(400, timeline.height())
    timeline.set_total_frames(150)
    timeline.set_span(50)
    timeline.set_reaches(
        [ReachSegment(40, 20, 40, ReachOutcome.UNCLASSIFIED)]
    )
    deleted_rows = []
    timeline.reachDeleteRequested.connect(deleted_rows.append)

    timeline._start_reach_edit()
    qtbot.keyClick(timeline, QtCore.Qt.Key.Key_D)
    delete_event = FakeGraphicsMouseEvent(
        QtCore.QPointF(timeline._frame_to_x(65), 50)
    )
    timeline.mousePressEvent(delete_event)

    assert deleted_rows == [0]
    assert timeline._reach_edit_tool == "delete"

    vp.cleanup()
    assert vp.close()


def test_timeline_reach_drag_mode_moves_existing_handle(qtbot):
    state = GuiState()
    state["frame_idx"] = 50
    vp = QtVideoPlayer(state=state)
    qtbot.addWidget(vp)
    timeline = vp.zoomed_timeline
    timeline.resize(400, timeline.height())
    timeline.set_total_frames(150)
    timeline.set_span(50)
    timeline.set_reaches(
        [ReachSegment(40, 20, 40, ReachOutcome.UNCLASSIFIED)]
    )
    edits = []
    timeline.reachEditRequested.connect(lambda *args: edits.append(list(args)))

    timeline._start_reach_edit()
    qtbot.keyClick(timeline, QtCore.Qt.Key.Key_E)
    for handle, frame in (("start", 40), ("max", 60), ("end", 80)):
        assert timeline._reach_handle_at_position(
            timeline._frame_to_x(frame), 50
        ) == (0, handle)

    press_event = FakeGraphicsMouseEvent(
        QtCore.QPointF(timeline._frame_to_x(60), 50)
    )
    timeline.mousePressEvent(press_event)
    assert timeline._reach_drag_handle == "max"

    move_event = FakeGraphicsMouseEvent(
        QtCore.QPointF(timeline._frame_to_x(70), 50)
    )
    timeline.mouseMoveEvent(move_event)
    timeline.mouseReleaseEvent(move_event)

    assert edits == [[0, 40, 70, 80]]
    assert timeline._reach_edit_active
    assert timeline._reach_edit_tool == "drag"
    assert timeline._reach_drag_handle == ""

    vp.cleanup()
    assert vp.close()


def test_inner_bounding_box_drag_requires_shift(qtbot, centered_pair_labels):
    vp = QtVideoPlayer(centered_pair_labels.video)
    qtbot.addWidget(vp)

    labeled_frame = centered_pair_labels.labeled_frames[0]
    views = (vp.view, vp.secondary_view)

    try:
        for view in views:
            vp.addInstance(
                instance=labeled_frame.instances[0],
                frame=labeled_frame,
                view=view,
            )
            qt_instance = view.all_instances[-1]
            box = qt_instance.box
            start = box.rect().center()
            end = start + QtCore.QPointF(15, 7)
            before = _qt_instance_node_positions(qt_instance)

            box.mousePressEvent(FakeGraphicsMouseEvent(start))
            box.mouseMoveEvent(FakeGraphicsMouseEvent(end))

            assert not box.moving
            assert _qt_instance_node_positions(qt_instance) == before
    finally:
        vp.cleanup()


def test_shift_inner_bounding_box_drag_moves_instance(qtbot, centered_pair_labels):
    vp = QtVideoPlayer(centered_pair_labels.video)
    qtbot.addWidget(vp)

    labeled_frame = centered_pair_labels.labeled_frames[0]
    views = (vp.view, vp.secondary_view)

    try:
        for view in views:
            vp.addInstance(
                instance=labeled_frame.instances[0],
                frame=labeled_frame,
                view=view,
            )
            qt_instance = view.all_instances[-1]
            box = qt_instance.box
            start = box.rect().center()
            delta = QtCore.QPointF(15, 7)
            before = _qt_instance_node_positions(qt_instance)

            box.mousePressEvent(
                FakeGraphicsMouseEvent(start, modifiers=QtCore.Qt.ShiftModifier)
            )
            box.mouseMoveEvent(
                FakeGraphicsMouseEvent(
                    start + delta, modifiers=QtCore.Qt.ShiftModifier
                )
            )

            assert box.moving
            for name, (x, y) in _qt_instance_node_positions(qt_instance).items():
                assert (x, y) == (
                    before[name][0] + delta.x(),
                    before[name][1] + delta.y(),
                )
    finally:
        vp.cleanup()


def test_video_player_leave_event(qtbot):
    """QtVideoPlayer leave events should not assume GraphicsView attributes."""
    vp = QtVideoPlayer()
    qtbot.addWidget(vp)

    event = QtCore.QEvent(QtCore.QEvent.Type.Leave)
    vp.leaveEvent(event)


def test_gui_video_instances(qtbot, small_robot_mp4_vid, centered_pair_labels):
    vp = QtVideoPlayer(small_robot_mp4_vid)
    qtbot.addWidget(vp)

    test_frame_idx = 63
    labeled_frames = centered_pair_labels.labeled_frames

    def plot_instances(vp, idx):
        for instance in labeled_frames[test_frame_idx].instances:
            vp.addInstance(instance=instance)

    vp.changedPlot.connect(plot_instances)
    vp.view.updatedViewer.emit()

    vp.show()
    vp.plot()

    # Check that all instances are included in viewer
    assert len(vp.instances) == len(labeled_frames[test_frame_idx].instances)

    # All instances should be selectable
    assert vp.selectable_instances == vp.instances

    vp.zoomToFit()

    # Check that we zoomed correctly
    assert vp.view.zoomFactor > 1

    vp.instances[0].updatePoints(complete=True)

    # Check that node is marked as complete
    nodes = [item for item in vp.instances[0].childItems() if hasattr(item, "point")]
    assert all((node.point[3] for node in nodes))  # point[3] = complete

    # Check that selection via keyboard works
    assert vp.view.getSelectionIndex() is None
    qtbot.keyClick(vp, QtCore.Qt.Key_1)
    assert vp.view.getSelectionIndex() == 0
    qtbot.keyClick(vp, QtCore.Qt.Key_2)
    assert vp.view.getSelectionIndex() == 1

    # Check that updatedSelection signal is emitted
    with qtbot.waitSignal(vp.view.updatedSelection, timeout=10):
        qtbot.keyClick(vp, QtCore.Qt.Key_1)

    # Check that selection by Instance works
    for inst in labeled_frames[test_frame_idx].instances:
        vp.view.selectInstance(inst)
        assert vp.view.getSelectionInstance() == inst

    # Check that sequence selection works
    with qtbot.waitCallback() as cb:
        vp.view.selectInstance(None)
        vp.onSequenceSelect(2, cb)
        qtbot.keyClick(vp, QtCore.Qt.Key_2)
        qtbot.keyClick(vp, QtCore.Qt.Key_1)

    inst_1 = vp.selectable_instances[1].instance
    inst_0 = vp.selectable_instances[0].instance
    assert cb.args[0] == [inst_1, inst_0]

    assert vp.close()


def test_session_views_cycle_and_clamp(
    qtbot,
    centered_pair_labels,
    small_robot_mp4_vid,
    small_robot_3_frame_vid,
):
    """Test linked side-by-side session views and view cycling."""
    labels = centered_pair_labels
    primary_video = labels.video
    labels_add_video(labels, small_robot_mp4_vid)
    labels_add_video(labels, small_robot_3_frame_vid)

    cameras = [
        sio.Camera(name="front"),
        sio.Camera(name="side"),
        sio.Camera(name="top"),
    ]
    session = sio.RecordingSession(camera_group=sio.CameraGroup(cameras=cameras))
    for video, camera in zip(
        [primary_video, small_robot_mp4_vid, small_robot_3_frame_vid], cameras
    ):
        session.add_video(video, camera)
    labels.sessions.append(session)

    state = GuiState()
    state["frame_idx"] = 10
    vp = QtVideoPlayer(state=state, context=SimpleNamespace(labels=labels))
    qtbot.addWidget(vp)
    state["video"] = primary_video

    assert not vp.secondary_view_widget.isHidden()
    assert vp.primary_title.text() == "front"
    assert vp.secondary_title.text() == "side"
    assert vp._clamped_frame_idx(small_robot_3_frame_vid) == 2

    vp.set_hovered_session_view(vp.secondary_view)
    assert vp.cycle_hovered_session_view()
    assert vp.secondary_video is small_robot_3_frame_vid
    assert vp.secondary_title.text() == "top"

    vp.set_hovered_session_view(vp.view)
    assert vp.cycle_hovered_session_view()
    assert state["video"] is primary_video
    assert vp.video is primary_video
    assert vp.primary_title.text() == "front"
    assert vp.secondary_video is small_robot_mp4_vid
    assert vp.secondary_title.text() == "side"

    state["video"] = small_robot_3_frame_vid
    assert state["video"] is primary_video
    assert vp.video is primary_video
    assert vp.secondary_video is small_robot_3_frame_vid
    assert vp.secondary_title.text() == "top"


def test_getInstancesBoundingRect():
    rect = GraphicsView.getInstancesBoundingRect([])
    assert rect.isNull()


def test_QtTextWithBackground(qtbot):
    scene = QtWidgets.QGraphicsScene()
    view = QtWidgets.QGraphicsView()
    view.setScene(scene)

    txt = QtTextWithBackground()

    txt.setDefaultTextColor(QColor("yellow"))
    bg_color = txt.getBackgroundColor()
    assert bg_color.lightness() == 0

    txt.setDefaultTextColor(QColor("black"))
    bg_color = txt.getBackgroundColor()
    assert bg_color.lightness() == 255

    scene.addItem(txt)
    qtbot.addWidget(view)


def test_VisibleBoundingBox(qtbot, centered_pair_labels):
    vp = QtVideoPlayer(centered_pair_labels.video)

    test_idx = 27
    for instance in centered_pair_labels.labeled_frames[test_idx].instances:
        vp.addInstance(instance)

    inst = vp.instances[0]

    # Check if type of bounding box is correct
    assert type(inst.box) == VisibleBoundingBox

    # Scale the bounding box
    start_top_left = inst.box.rect().topLeft()
    start_bottom_right = inst.box.rect().bottomRight()
    initial_width = inst.box.rect().width()
    initial_height = inst.box.rect().height()

    dx = 5
    dy = 10

    end_top_left = QtCore.QPointF(start_top_left.x() - dx, start_top_left.y() - dy)
    end_bottom_right = QtCore.QPointF(
        start_bottom_right.x() + dx, start_bottom_right.y() + dy
    )

    inst.box.setRect(QtCore.QRectF(end_top_left, end_bottom_right))

    # Check if bounding box scaled appropriately
    assert inst.box.rect().width() - initial_width == 2 * dx
    assert inst.box.rect().height() - initial_height == 2 * dy


def test_prediction_box_double_click_emits_instance_signal(centered_pair_labels):
    """Double-clicking a prediction box should convert the prediction."""
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    assert app is not None
    vp = QtVideoPlayer(centered_pair_labels.video)
    try:
        source_inst = centered_pair_labels.labeled_frames[0].instances[0]
        pred_inst = sio.PredictedInstance.from_numpy(
            source_inst.numpy(),
            skeleton=source_inst.skeleton,
            score=0.9,
        )
        vp.addInstance(pred_inst)
        qt_inst = vp.view.predicted_instances[0]

        box_rect = qt_inst.box.rect()
        scene_pos = qt_inst.box.mapToScene(
            QtCore.QPointF(box_rect.left() + 5, box_rect.center().y())
        )
        assert not qt_inst.boundingRect().contains(qt_inst.mapFromScene(scene_pos))
        assert not vp.view._is_frame_background_double_click(scene_pos)

        received = []
        vp.view.instanceDoubleClicked.connect(
            lambda inst, event: received.append((inst, event.modifiers()))
        )

        viewport_pos = vp.view.mapFromScene(scene_pos)
        event = QMouseEvent(
            QtCore.QEvent.Type.MouseButtonDblClick,
            QtCore.QPointF(viewport_pos),
            QtCore.QPointF(viewport_pos),
            QtCore.QPointF(viewport_pos),
            QtCore.Qt.MouseButton.LeftButton,
            QtCore.Qt.MouseButton.LeftButton,
            QtCore.Qt.KeyboardModifier.ShiftModifier,
        )
        assert vp.view._instance_item_at(vp.view.mapToScene(event.pos())) is qt_inst
        vp.view.mouseDoubleClickEvent(event)

        assert received == [(pred_inst, QtCore.Qt.KeyboardModifier.ShiftModifier)]
        assert event.isAccepted()
    finally:
        vp.cleanup()


def test_external_prediction_preview_skips_frames_with_project_predictions(
    centered_pair_labels,
):
    """External previews should not draw over project predictions, hidden or visible."""
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    assert app is not None
    vp = QtVideoPlayer(
        centered_pair_labels.video,
        context=SimpleNamespace(labels=centered_pair_labels),
    )
    try:
        lf = centered_pair_labels.find(
            centered_pair_labels.video, 0, return_new=True
        )[0]
        pred_inst = sio.PredictedInstance.from_numpy(
            centered_pair_labels[0].instances[0].numpy(),
            skeleton=centered_pair_labels.skeleton,
            score=0.9,
        )
        user_inst = sio.Instance.from_numpy(
            pred_inst.numpy(),
            skeleton=centered_pair_labels.skeleton,
        )
        user_inst.from_predicted = pred_inst
        lf.instances.extend([pred_inst, user_inst])

        class Manager:
            def instances_for(self, video, frame_idx):
                return [
                    sio.PredictedInstance.from_numpy(
                        pred_inst.numpy(),
                        skeleton=centered_pair_labels.skeleton,
                        score=0.8,
                    )
                ]

        vp.state["external predictions"] = Manager()
        vp.add_external_prediction_preview(
            centered_pair_labels.video,
            0,
            vp.view,
            lf,
        )

        assert not any(inst.external_preview for inst in vp.view.all_instances)
    finally:
        vp.cleanup()


def test_wheelEvent(qtbot):
    """Test the wheelEvent method of the GraphicsView class."""
    graphics_view = GraphicsView()

    # Create a QWheelEvent
    position = QtCore.QPointF(100, 100)  # The position of the wheel event
    global_position = QtCore.QPointF(100, 100)  # The global position of the wheel event
    pixel_delta = QtCore.QPoint(0, 120)  # The distance in pixels the wheel is rotated
    angle_delta = QtCore.QPoint(0, 120)  # The distance in degrees the wheel is rotated
    buttons = QtCore.Qt.MouseButton.NoButton  # The mouse buttons
    modifiers = QtCore.Qt.KeyboardModifier.NoModifier  # The keyboard modifiers
    phase = QtCore.Qt.ScrollPhase.ScrollUpdate  # The scroll phase
    inverted = False  # The inverted flag
    source = QtCore.Qt.MouseEventSource.MouseEventNotSynthesized  # The source

    event = QWheelEvent(
        position,
        global_position,
        pixel_delta,
        angle_delta,
        buttons,
        modifiers,
        phase,
        inverted,
        source,
    )

    # Call the wheelEvent method
    print(
        "Testing GraphicsView.wheelEvent which will result in exit code 127 "
        "originating from a segmentation fault if it fails."
    )
    graphics_view.wheelEvent(event)


def test_nan_coordinates_bounding_rect(qtbot, centered_pair_labels):
    """Test that NaN coordinates don't create NaN bounding rects.

    Regression test for issue #2427 where NaN coordinates in predicted instances
    caused the GUI to freeze on Linux systems with Qt 6.10+.
    """
    from sleap_io.model.instance import PredictedInstance

    vp = QtVideoPlayer(centered_pair_labels.video)

    # Get a labeled frame with instances
    test_frame = centered_pair_labels.labeled_frames[0]
    original_instance = test_frame.instances[0]

    # Test 1: Instance with some NaN coordinates (failed keypoint detection)
    points_with_nan = original_instance.numpy().copy()
    points_with_nan[0] = [np.nan, np.nan]  # First keypoint has NaN
    points_with_nan[1] = [np.nan, np.nan]  # Second keypoint has NaN

    predicted_instance = PredictedInstance(
        points=points_with_nan, skeleton=original_instance.skeleton, score=0.5
    )

    # Create QtInstance directly to test bounding rect calculation
    qt_instance = QtInstance(instance=predicted_instance, player=vp)

    # Verify bounding rect doesn't have NaN values (issue #2427)
    bounding_rect = qt_instance.getPointsBoundingRect()
    # Should return either a valid rect or null rect, but never NaN rect
    assert not np.isnan(bounding_rect.x()), "Bounding rect x is NaN"
    assert not np.isnan(bounding_rect.y()), "Bounding rect y is NaN"
    assert not np.isnan(bounding_rect.width()), "Bounding rect width is NaN"
    assert not np.isnan(bounding_rect.height()), "Bounding rect height is NaN"

    # Test 2: Instance with all NaN coordinates (complete detection failure)
    n_nodes = len(original_instance.skeleton.nodes)
    all_nan_points = np.full((n_nodes, 2), np.nan)
    all_nan_instance = PredictedInstance(
        points=all_nan_points, skeleton=original_instance.skeleton, score=0.1
    )

    qt_instance_all_nan = QtInstance(instance=all_nan_instance, player=vp)
    bounding_rect_all_nan = qt_instance_all_nan.getPointsBoundingRect()

    # Should return a null rect (which Qt handles gracefully)
    assert bounding_rect_all_nan.isNull() or bounding_rect_all_nan.isEmpty(), (
        "All-NaN instance should have null bounding rect"
    )

    # Verify no NaN values in the rect
    assert not np.isnan(bounding_rect_all_nan.x()), "All-NaN rect x is NaN"
    assert not np.isnan(bounding_rect_all_nan.y()), "All-NaN rect y is NaN"


def test_navigate_highlight(qtbot, small_robot_mp4_vid, centered_pair_labels):
    """Test the navigation highlight feature for Size Distribution click-to-navigate."""
    vp = QtVideoPlayer(small_robot_mp4_vid)
    qtbot.addWidget(vp)

    test_frame_idx = 63
    labeled_frames = centered_pair_labels.labeled_frames
    frame_instances = labeled_frames[test_frame_idx].instances

    def plot_instances(vp, idx):
        for instance in frame_instances:
            vp.addInstance(instance=instance)

    vp.changedPlot.connect(plot_instances)
    vp.view.updatedViewer.emit()
    vp.show()
    vp.plot()

    # Ensure we have instances
    assert len(vp.instances) >= 2

    # Initially, no instances should have navigation highlight
    for inst in vp.instances:
        assert not inst.navigate_highlight

    # Highlight the second instance via player method (using Instance object)
    vp.highlightNavigatedInstance(frame_instances[1])

    # Check that only the second instance is highlighted
    assert not vp.instances[0].navigate_highlight
    assert vp.instances[1].navigate_highlight
    assert vp.instances[1].navigate_box.opacity() == 0.5

    # Highlight a different instance (first)
    vp.highlightNavigatedInstance(frame_instances[0])

    # Check that now only the first instance is highlighted
    assert vp.instances[0].navigate_highlight
    assert not vp.instances[1].navigate_highlight

    # Clear all highlights via player method (pass None)
    vp.highlightNavigatedInstance(None)

    # All instances should now have no highlight
    for inst in vp.instances:
        assert not inst.navigate_highlight
        assert inst.navigate_box.opacity() == 0

    # Test via GraphicsView directly
    vp.view.highlightNavigatedInstance(frame_instances[0])
    assert vp.instances[0].navigate_highlight

    vp.view.clearNavigateHighlight()
    assert not vp.instances[0].navigate_highlight

    assert vp.close()


def test_negative_frame_overlay(qtbot, small_robot_mp4_vid):
    """The negative-frame overlay draws a border + caption only on negatives."""
    from sleap_io import Labels, LabeledFrame, Skeleton

    from sleap.gui.overlays.negative_frame import NegativeFrameOverlay

    labels = Labels(videos=[small_robot_mp4_vid], skeletons=[Skeleton(["A"])])
    labels.append(
        LabeledFrame(video=small_robot_mp4_vid, frame_idx=1, is_negative=True)
    )

    vp = QtVideoPlayer(small_robot_mp4_vid)
    qtbot.addWidget(vp)
    overlay = NegativeFrameOverlay(labels=labels, player=vp)

    # Negative frame: border + caption are added.
    overlay.add_to_scene(small_robot_mp4_vid, 1)
    assert len(overlay.items) == 2
    captions = [
        item for item in overlay.items if isinstance(item, QtTextWithBackground)
    ]
    assert len(captions) == 1
    assert captions[0].toPlainText() == "Negative Frame"

    # Non-negative frame: nothing is drawn.
    overlay.redraw(small_robot_mp4_vid, 0)
    assert overlay.items == []
