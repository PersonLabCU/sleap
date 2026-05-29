import logging
import sys


# Setup logging to stdout
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

# Import submodules we want available at top-level
from sleap.version import __version__, versions
from sleap_io import (
    Labels,
    LabeledFrame,
    Skeleton,
    Node,
    Instance,
    PredictedInstance,
    Video,
    SuggestionFrame,
)


def _install_sleap_io_compat_patches() -> None:
    """Install small compatibility shims for sleap-io behavior used by SLEAP."""

    if getattr(Labels, "_sleap_duplicate_frame_warning_patch", False):
        return

    def get_frame_without_duplicate_warning(
        self: Labels, video: Video, frame_idx: int
    ) -> LabeledFrame | None:
        """Return the last matching frame without warning on duplicate frames."""

        self._check_not_lazy("get_frame")
        n = len(self.labeled_frames)
        frame_index = self._frame_index
        if frame_index is None or self._frame_index_len != n:
            labeled_frames = tuple(self.labeled_frames)
            frame_index = {}
            for lf in labeled_frames:
                frame_index[(id(lf.video), lf.frame_idx)] = lf
            self._frame_index = frame_index
            self._frame_index_len = len(labeled_frames)
        return frame_index.get((id(video), frame_idx))

    Labels.get_frame = get_frame_without_duplicate_warning
    Labels._sleap_duplicate_frame_warning_patch = True


_install_sleap_io_compat_patches()
