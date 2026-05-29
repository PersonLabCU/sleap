"""Module for testing dock widgets for the `MainWindow`."""

import shutil
from pathlib import Path

import numpy as np
from qtpy.QtWidgets import QMessageBox
import sleap_io as sio

from sleap import Labels, Video
from sleap.gui.app import MainWindow
from sleap.gui.commands import (
    AddInstance,
    AddSession,
    OpenSkeleton,
    TriangulateSessionLabels,
    UpdateTopic,
)
from sleap.gui.widgets.docks import (
    InstancesDock,
    SessionsDock,
    SkeletonDock,
    SuggestionsDock,
    VideosDock,
)
from sleap.sleap_io_adaptors.lf_labels_utils import labels_add_video


def test_videos_dock(
    qtbot,
    centered_pair_predictions: Labels,
    small_robot_mp4_vid: Video,
    centered_pair_vid: Video,
    small_robot_3_frame_vid: Video,
):
    """Test the `DockWidget` class."""

    # Add some extra videos to the labels
    labels = centered_pair_predictions
    labels_add_video(labels, small_robot_3_frame_vid)
    labels_add_video(labels, centered_pair_vid)
    labels_add_video(labels, small_robot_mp4_vid)
    assert len(labels.videos) == 4

    # Create the dock
    main_window = MainWindow()

    # Use commands to set the labels instead of setting it directly
    # To make sure other dependent instances like color_manager are also set
    main_window.commands.loadLabelsObject(labels)

    video_state = labels.videos[-1]
    main_window.state["video"] = video_state
    dock = VideosDock(main_window)

    # Test that the dock was created correctly
    assert dock.name == "Videos"
    assert dock.main_window is main_window
    assert dock.wgt_layout is dock.widget().layout()
    assert "add sessions" in dock.main_window._buttons

    # Test that videos can be removed

    # No videos selected, won't remove anything
    dock.main_window._buttons["remove video"].click()
    assert len(labels.videos) == 4

    # Select the last video, should remove that one and update state

    dock.main_window.videos_dock.table.selectRowItem(small_robot_mp4_vid)
    dock.main_window._buttons["remove video"].click()
    assert len(labels.videos) == 3
    assert video_state not in labels.videos
    assert main_window.state["video"] == labels.videos[-1]

    # Select the last two videos, should remove those two and update state
    idxs = [1, 2]
    videos_to_be_removed = [labels.videos[i] for i in idxs]
    main_window.state["selected_batch_video"] = idxs
    dock.main_window._buttons["remove video"].click()
    assert len(labels.videos) == 1
    assert (
        videos_to_be_removed[0] not in labels.videos
        and videos_to_be_removed[1] not in labels.videos
    )
    assert main_window.state["video"] == labels.videos[-1]


def test_add_session_command(qtbot, small_robot_mp4_vid: Video):
    """Test adding a session imports videos with session metadata."""
    main_window = MainWindow()
    video_path = small_robot_mp4_vid.filename
    session_path = str(Path(video_path).parent)

    AddSession.do_action(
        main_window.commands,
        {
            "session_path": session_path,
            "import_list": [{"params": {"filename": video_path}}],
        },
    )
    main_window.on_data_update([UpdateTopic.video])

    assert len(main_window.labels.videos) == 1
    video = main_window.labels.videos[0]
    assert len(main_window.labels.sessions) == 1
    session = main_window.labels.sessions[0]
    assert session.metadata["session_name"] == Path(session_path).name
    assert session.metadata["session_path"] == session_path
    assert session.videos == [video]
    assert session.cameras[0].name == Path(video_path).stem
    assert main_window.videos_dock.table.model().items[0]["session"] == Path(
        session_path
    ).name


def test_add_session_loads_events_file(tmp_path, qtbot, small_robot_mp4_vid: Video):
    """Test adding a session automatically loads a matching events text file."""
    main_window = MainWindow()
    video_path = tmp_path / Path(small_robot_mp4_vid.filename).name
    shutil.copy(small_robot_mp4_vid.filename, video_path)
    events_path = tmp_path / "20260514_Kat_session001_events.txt"
    events_path.write_text("delivery 4\nreach 8\n")

    AddSession.do_action(
        main_window.commands,
        {
            "session_path": str(tmp_path),
            "import_list": [{"params": {"filename": str(video_path)}}],
        },
    )
    main_window.on_data_update([UpdateTopic.video])

    session = main_window.labels.sessions[0]
    assert session.metadata["events_path"] == str(events_path)
    assert session.metadata["events"] == [
        {"event": "delivery", "frame": 4, "line": 1},
        {"event": "reach", "frame": 8, "line": 2},
    ]
    assert main_window.sessions_dock.events_table.rowCount() == 2
    # Events are displayed in the zoomed timeline, not the global seekbar
    assert len(main_window.player.zoomed_timeline._events) == 2


def test_session_metadata_round_trip(tmp_path, qtbot, small_robot_mp4_vid: Video):
    """Test imported recording sessions are saved with the labels file."""
    main_window = MainWindow()
    video_path = small_robot_mp4_vid.filename
    session_path = str(Path(video_path).parent)
    output_path = tmp_path / "labels.slp"

    AddSession.do_action(
        main_window.commands,
        {
            "session_path": session_path,
            "import_list": [{"params": {"filename": video_path}}],
        },
    )
    sio.save_file(main_window.labels, output_path)

    labels = sio.load_file(output_path)
    assert len(labels.sessions) == 1
    assert labels.sessions[0].metadata["session_path"] == session_path
    assert labels.sessions[0].videos[0].filename == video_path


def test_find_session_video_files(tmp_path):
    """Test session discovery only returns supported files in a stable order."""
    (tmp_path / "cam_b.mp4").write_text("")
    (tmp_path / "cam_a.avi").write_text("")
    (tmp_path / "calibration.toml").write_text("")
    (tmp_path / "notes.txt").write_text("")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "cam_c.mp4").write_text("")

    files = AddSession.find_video_files(tmp_path)

    assert [Path(file).name for file in files] == ["cam_a.avi", "cam_b.mp4"]


def test_skeleton_dock(qtbot):
    """Test the `DockWidget` class."""
    main_window = MainWindow()
    dock = SkeletonDock(main_window)

    assert dock.name == "Skeleton"
    assert dock.main_window is main_window
    assert dock.wgt_layout is dock.widget().layout()

    # This method should get called when we click the load button, but let's just call
    # the non-gui parts directly
    fn = Path(
        OpenSkeleton.get_template_skeleton_filename(context=dock.main_window.commands)
    )
    assert fn.name == f"{dock.skeleton_templates.currentText()}.json"


def test_suggestions_dock(qtbot):
    """Test the `DockWidget` class."""
    main_window = MainWindow()
    dock = SuggestionsDock(main_window)

    assert dock.name == "Labeling Suggestions"
    assert dock.main_window is main_window
    assert dock.wgt_layout is dock.widget().layout()


def test_sessions_dock(qtbot):
    """Test the sessions dock is created for label triangulation workflows."""
    main_window = MainWindow()
    dock = SessionsDock(main_window)

    assert dock.name == "Sessions"
    assert dock.main_window is main_window
    assert dock.wgt_layout is dock.widget().layout()
    assert dock.calibration_path_edit is not None
    assert dock.main_window._buttons["triangulate"] is not None


def test_sessions_dock_triangulates(qtbot, tmp_path, monkeypatch):
    """Test sessions dock calls project label triangulation."""
    main_window = MainWindow()
    dock = SessionsDock(main_window)

    calibration_path = tmp_path / "calibration.toml"
    calibration_path.write_text("[cam_0]\n")

    calls = {}

    def fake_do_action(context, params):
        calls["context"] = context
        calls["calibration_path"] = params["calibration_path"]
        params["result"] = {
            "sessions": 1,
            "eligible_frames": 2,
            "predicted_instances": 3,
            "skipped_sessions": [],
        }

    monkeypatch.setattr(TriangulateSessionLabels, "do_action", fake_do_action)
    monkeypatch.setattr(QMessageBox, "information", lambda *args, **kwargs: None)

    dock.calibration_path_edit.setText(str(calibration_path))

    dock._run_triangulation()

    assert calls["context"] is main_window.commands
    assert calls["calibration_path"] == str(calibration_path)
    assert main_window.state["has_changes"]


def test_triangulate_session_labels_predicts_missing_views(
    qtbot,
    centered_pair_labels: Labels,
    small_robot_mp4_vid: Video,
    small_robot_3_frame_vid: Video,
):
    """Test triangulation projects labels into empty views in a session."""
    labels = centered_pair_labels
    skeleton = labels.skeletons[0]
    source_video = labels.video
    labels_add_video(labels, small_robot_mp4_vid)
    labels_add_video(labels, small_robot_3_frame_vid)

    cameras = [
        sio.Camera(name="front"),
        sio.Camera(name="side"),
        sio.Camera(name="top"),
    ]
    session = sio.RecordingSession(camera_group=sio.CameraGroup(cameras=cameras))
    for video, camera in zip(
        [source_video, small_robot_mp4_vid, small_robot_3_frame_vid], cameras
    ):
        session.add_video(video, camera)
    labels.sessions.append(session)

    frame_idx = 0
    points_a = np.full((len(skeleton.nodes), 2), np.nan)
    points_b = np.full((len(skeleton.nodes), 2), np.nan)
    points_a[0] = [10, 20]
    points_b[0] = [20, 30]
    labels.find(source_video, frame_idx)[0].instances = [
        sio.Instance.from_numpy(points_a, skeleton=skeleton)
    ]
    side_frame = labels.find(small_robot_mp4_vid, frame_idx, return_new=True)[0]
    side_frame.instances.append(sio.Instance.from_numpy(points_b, skeleton=skeleton))
    labels.append(side_frame)

    class FakeCalibration:
        cameras = [object(), object(), object()]

        def get_names(self):
            return ["front", "side", "top"]

        def subset_cameras_names(self, names):
            self.names = names
            return self

        def triangulate(self, points):
            valid = np.isfinite(points)
            counts = valid.sum(axis=0)
            xy = np.divide(
                np.nan_to_num(points).sum(axis=0),
                counts,
                out=np.full(points.shape[1:], np.nan),
                where=counts > 0,
            )
            return np.concatenate([xy, np.zeros((xy.shape[0], 1))], axis=1)

        def project(self, points3d):
            return np.stack(
                [points3d[:, :2] + view_idx for view_idx in range(len(self.cameras))]
            )

    result = TriangulateSessionLabels.triangulate_missing_views(
        labels, FakeCalibration()
    )

    target_frame = labels.find(small_robot_3_frame_vid, frame_idx)[0]
    assert result["predicted_instances"] == 1
    assert len(target_frame.predicted_instances) == 1
    np.testing.assert_allclose(
        target_frame.predicted_instances[0].numpy()[0], [17, 27]
    )


def test_add_instance_targets_session_companion_video(
    qtbot,
    centered_pair_labels: Labels,
    small_robot_mp4_vid: Video,
):
    """Test adding an instance can target the right-side session video."""
    labels = centered_pair_labels
    source_video = labels.video
    labels_add_video(labels, small_robot_mp4_vid)
    session = sio.RecordingSession(
        camera_group=sio.CameraGroup(
            cameras=[sio.Camera(name="front"), sio.Camera(name="side")]
        )
    )
    session.add_video(source_video, session.camera_group.cameras[0])
    session.add_video(small_robot_mp4_vid, session.camera_group.cameras[1])
    labels.sessions.append(session)

    main_window = MainWindow(labels=labels)
    main_window.state["video"] = source_video
    main_window.state["frame_idx"] = 0
    source_count = len(labels.find(source_video, 0)[0].instances)

    AddInstance.do_action(
        main_window.commands,
        {
            "target_video": small_robot_mp4_vid,
            "init_method": "random",
            "mark_complete": False,
        },
    )

    target_frame = labels.find(small_robot_mp4_vid, 0)[0]
    assert len(target_frame.user_instances) == 1
    assert len(labels.find(source_video, 0)[0].instances) == source_count


def test_instances_dock(qtbot, centered_pair_predictions: Labels):
    """Test the `DockWidget` class."""
    main_window = MainWindow(labels=centered_pair_predictions)
    context = main_window.commands
    lf = context.state["labeled_frame"]
    dock = InstancesDock(main_window)

    assert dock.name == "Instances"
    assert dock.main_window is main_window
    assert dock.wgt_layout is dock.widget().layout()

    # Test new instance button

    offset = 10

    # Find instance that we will copy from
    (
        copy_instance,
        from_predicted,
        from_prev_frame,
    ) = AddInstance.find_instance_to_copy_from(
        context, copy_instance=None, init_method="best"
    )
    n_instance = len(lf.instances)
    dock.main_window._buttons["new instance"].click()

    # Check that new instance was added with offset
    assert len(lf.instances) == n_instance + 1
    new_inst = lf.instances[-1]
    diff = np.nan_to_num(new_inst.numpy() - copy_instance.numpy(), nan=offset)
    assert np.all(diff == offset)
