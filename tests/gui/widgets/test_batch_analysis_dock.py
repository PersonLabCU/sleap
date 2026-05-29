from sleap.gui.widgets.batch_analysis_dock import BatchAnalysisDock


def test_find_video_files_only_includes_movies(tmp_path):
    session = tmp_path / "session001"
    session.mkdir()
    video = session / "camera1.mp4"
    video.touch()
    (session / "frames.jpg").touch()
    (session / "tracks.h5").touch()
    (session / "camera1.predictions.slp").touch()
    (session / "notes.txt").touch()

    assert BatchAnalysisDock.find_video_files(session) == [video]


def test_find_session_dirs_nested_layout(tmp_path):
    parent = tmp_path / "20260521"
    rig = parent / "JK002"
    session_a = rig / "Person001" / "Animal001" / "session001"
    session_b = rig / "Person001" / "session002"
    non_session = rig / "Person001" / "Animal001"
    session_a.mkdir(parents=True)
    session_b.mkdir(parents=True)
    non_session.mkdir(parents=True, exist_ok=True)
    (session_a / "camA.mp4").touch()
    (session_b / "camB.avi").touch()
    (non_session / "image_only.jpg").touch()
    (non_session / "analysis_only.h5").touch()

    sessions = BatchAnalysisDock.find_session_dirs(parent)

    assert sessions == [
        (session_a, [session_a / "camA.mp4"]),
        (session_b, [session_b / "camB.avi"]),
    ]
