import warnings

import sleap  # noqa: F401
from sleap_io import LabeledFrame, Labels, Video


def test_duplicate_labeled_frame_lookup_uses_last_without_warning():
    video = Video(filename="test.mp4")
    first = LabeledFrame(video=video, frame_idx=10)
    last = LabeledFrame(video=video, frame_idx=10)
    labels = Labels(labeled_frames=[first, last])
    labels._frame_index = None

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        found = labels.find(video, 10)

    assert found == [last]
    assert not any("Duplicate LabeledFrame" in str(w.message) for w in caught)


def test_frame_index_rebuild_tolerates_reentrant_invalidation():
    class InvalidatingFrames(list):
        def __init__(self, frames, labels):
            super().__init__(frames)
            self.labels = labels

        def __iter__(self):
            self.labels._frame_index = None
            yield from super().__iter__()

    video = Video(filename="test.mp4")
    frame = LabeledFrame(video=video, frame_idx=10)
    labels = Labels(labeled_frames=[frame])
    labels._frame_index = None
    labels.labeled_frames = InvalidatingFrames([frame], labels)

    assert labels.find(video, 10) == [frame]
