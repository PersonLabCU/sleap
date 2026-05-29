from sleap.gui.prediction_exports import _videos_with_frames


class DummyVideo:
    def __init__(self, filename):
        self.filename = filename


class DummyLabels:
    def __init__(self, videos, frames_by_video):
        self.videos = videos
        self.frames_by_video = frames_by_video

    def find(self, video):
        return self.frames_by_video.get(id(video), [])


def test_videos_with_frames_falls_back_to_matching_label_video():
    selected_video = DummyVideo("session/cam1.mp4")
    merged_video = DummyVideo("session/cam1.mp4")
    labels = DummyLabels(
        [merged_video],
        frames_by_video={id(merged_video): [object()]},
    )

    assert _videos_with_frames(labels, [selected_video]) == [merged_video]
