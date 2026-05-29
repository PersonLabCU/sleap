"""Zoomed local timeline widget showing the frame neighborhood around the playhead.

Displays frame-state markers (labeled/predicted/suggested) and session-event
markers as colored triangles, mirroring the style of the global VideoSlider
seekbar but focused on the immediate area around the current frame.

Design mirrors the sleap3d_reach TimeLineWidget, adapted to the existing
QGraphicsView/QGraphicsScene infrastructure (no extra dependencies required).
"""

from bisect import bisect_left, bisect_right
from typing import Dict, Iterable, List, Optional

from qtpy import QtWidgets
from qtpy.QtCore import Qt, QLineF, QPointF, QRectF
from qtpy.QtGui import (
    QBrush,
    QColor,
    QFont,
    QPainterPath,
    QPen,
    QPolygonF,
)

from sleap.gui.state import GuiState
from sleap.gui.widgets.slider import SliderMark

# ─────────────────────────────── layout constants ────────────────────────── #
_H = 76  # fixed widget height (px, not counting the 2px border)

_MARK_TOP = 2       # top of frame-state mark lines
_MARK_BOTTOM = 20   # bottom of frame-state mark lines

_TICK_MAJOR_TOP = 20   # top of major tick marks
_TICK_MINOR_TOP = 24   # top of minor tick marks
_AXIS_Y = 28           # y of axis horizontal line

# current-frame indicator: filled ▽ just above the axis
_CF_BASE_Y = _TICK_MAJOR_TOP
_CF_TIP_Y = _AXIS_Y

# event markers: filled △ just below the axis
_EV_TIP_Y = _AXIS_Y + 1
_EV_BASE_Y = _AXIS_Y + 13

# reach bars: horizontal colored rect below events
_REACH_Y = 44       # top of reach bar row
_REACH_H = 12       # height of each reach bar rect
_REACH_EXTRA = 3    # vertical reachMax line extends ±this beyond bar edges

_LABEL_Y = 60  # frame-number label baseline

# ─────────────────────────────── color constants ─────────────────────────── #
# Must stay in sync with the colors defined in slider.py SliderMark.color
_MARK_COLORS: Dict[str, QColor] = {
    "simple":      QColor(52, 211, 153),   # emerald green  – user labeled
    "simple_thin": QColor(251, 191, 36),   # amber          – predicted, no track
    "filled":      QColor(99, 102, 241),   # indigo         – suggested + user
    "open":        QColor(148, 163, 184),  # slate          – suggested, empty
    "predicted":   QColor(56, 189, 248),   # sky blue       – suggested + predicted
}
_MARK_WIDTHS: Dict[str, int] = {
    "simple":      6,
    "simple_thin": 3,
    "filled":      6,
    "open":        6,
    "predicted":   6,
}

_BG_COLOR = QColor(22, 24, 32)
_AXIS_COLOR = QColor(85, 90, 108)
_TICK_COLOR = QColor(70, 75, 92)
_LABEL_COLOR = QColor(105, 110, 132)
_CF_COLOR = QColor(255, 255, 255)
_CURSOR_COLOR = QColor(200, 205, 220, 170)


class ZoomedTimelineWidget(QtWidgets.QGraphicsView):
    """Zoomed local timeline showing ±span frames around the current frame.

    Frame-state markers (labeled / predicted / suggested) are drawn as short
    coloured vertical bars above the axis, mirroring the colours used by the
    global VideoSlider seekbar.  Session-event markers are drawn as coloured
    upward-pointing triangles below the axis.  A white downward triangle marks
    the current frame at the centre of the view.

    Clicking inside the widget navigates to that frame.
    """

    TIME_SPANS = [50, 100, 200, 500, 1000, 5000]
    DEFAULT_SPAN = 200

    def __init__(self, state: GuiState):
        super().__init__()
        self._state = state
        self._curr_frame: int = 0
        self._total_frames: int = 1
        self._span: int = self.DEFAULT_SPAN

        # marks stored sorted per type for O(log N + K) visible-range lookup
        self._sorted_marks_by_type: Dict[str, List[float]] = {
            t: [] for t in _MARK_COLORS
        }
        self._events: List[dict] = []
        self._reaches: List = []  # List[ReachSegment], typed lazily

        # ── scene setup ──────────────────────────────────────────────────── #
        self._scene = QtWidgets.QGraphicsScene()
        self.setScene(self._scene)
        self.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setFixedHeight(_H + 4)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed
        )
        self.setMouseTracking(True)
        from qtpy.QtGui import QPainter
        self.setRenderHint(QPainter.Antialiasing, False)
        self._scene.setBackgroundBrush(QBrush(_BG_COLOR))
        self._scene.setSceneRect(QRectF(0, 0, 400, _H))

        # ── persistent path items: one per mark type ─────────────────────── #
        self._mark_path_items: Dict[str, QtWidgets.QGraphicsPathItem] = {}
        for mtype, color in _MARK_COLORS.items():
            pen = QPen(color, _MARK_WIDTHS[mtype])
            pen.setCosmetic(True)
            item = self._scene.addPath(QPainterPath(), pen)
            item.setZValue(2)
            self._mark_path_items[mtype] = item

        # ── axis horizontal line ─────────────────────────────────────────── #
        _axis_pen = QPen(_AXIS_COLOR, 1)
        _axis_pen.setCosmetic(True)
        self._axis_item = self._scene.addLine(
            QLineF(0, _AXIS_Y, 400, _AXIS_Y), _axis_pen
        )
        self._axis_item.setZValue(1)

        # ── current-frame indicator: filled ▽ above axis ─────────────────── #
        _cf_pen = QPen(_CF_COLOR, 1.5)
        _cf_pen.setCosmetic(True)
        self._cf_item = self._scene.addPolygon(
            _down_triangle(0.0), _cf_pen, QBrush(_CF_COLOR)
        )
        self._cf_item.setZValue(15)

        # ── vertical cursor line ─────────────────────────────────────────── #
        _cursor_pen = QPen(_CURSOR_COLOR, 1)
        _cursor_pen.setCosmetic(True)
        _cursor_pen.setStyle(Qt.DashLine)
        self._cursor_line = self._scene.addLine(
            QLineF(0, 0, 0, _H), _cursor_pen
        )
        self._cursor_line.setZValue(25)
        self._cursor_line.hide()

        # ── cursor label ─────────────────────────────────────────────────── #
        _lbl_font = QFont()
        _lbl_font.setPixelSize(9)
        self._cursor_lbl = self._scene.addSimpleText("", _lbl_font)
        self._cursor_lbl.setBrush(QBrush(QColor(200, 205, 220)))
        self._cursor_lbl.setZValue(26)
        self._cursor_lbl.hide()

        # ── mutable items rebuilt on each update ─────────────────────────── #
        self._tick_items: List = []
        self._tick_label_items: List = []
        self._event_items: List = []
        self._reach_items: List = []

        # ── connect to state ─────────────────────────────────────────────── #
        state.connect("frame_idx", self._on_frame_changed)

        self._full_redraw()

    # ── public API ───────────────────────────────────────────────────────── #

    def set_total_frames(self, n: int) -> None:
        """Notify widget of the total number of frames in the current video."""
        self._total_frames = max(1, n)

    def set_marks(self, marks: Iterable[SliderMark]) -> None:
        """Update the frame-state marks shown in the zoomed timeline.

        Only non-tick marks (labeled / predicted / suggested) are rendered.
        Pre-sorts marks per type for fast visible-range queries on frame change.
        """
        from collections import defaultdict

        buckets: Dict[str, List[float]] = defaultdict(list)
        for m in marks:
            if m.type in _MARK_COLORS:
                buckets[m.type].append(float(m.val))

        self._sorted_marks_by_type = {
            t: sorted(buckets.get(t, [])) for t in _MARK_COLORS
        }
        self._update_mark_paths()

    def set_events(self, events: Optional[Iterable[dict]]) -> None:
        """Update session-event markers shown in the zoomed timeline."""
        self._events = list(events or [])
        self._update_event_items()

    def set_reaches(self, reaches: Iterable) -> None:
        """Update reach-segment bars shown in the zoomed timeline.

        Accepts an iterable of :class:`~sleap.gui.reach_detection.ReachSegment`
        objects.  Pass an empty list to clear all reach bars.
        """
        self._reaches = list(reaches or [])
        self._update_reach_bars()

    def set_span(self, span: int) -> None:
        """Change the visible half-window (±span frames around current frame)."""
        self._span = span
        self._full_redraw()

    # ── coordinate helpers ───────────────────────────────────────────────── #

    def _frame_to_x(self, frame: float) -> float:
        """Map a frame index to a pixel x-coordinate in the scene."""
        w = max(self.width(), 1)
        return (frame - (self._curr_frame - self._span)) / (2 * self._span) * w

    def _x_to_frame(self, x: float) -> int:
        """Map a pixel x-coordinate in the scene to a frame index."""
        w = max(self.width(), 1)
        return int(round((x / w) * (2 * self._span) + self._curr_frame - self._span))

    # ── drawing ───────────────────────────────────────────────────────────── #

    def _full_redraw(self) -> None:
        """Rebuild everything that depends on the current frame / span / size."""
        self._update_axis()
        self._update_curr_frame_indicator()
        self._update_mark_paths()
        self._update_ticks()
        self._update_event_items()
        self._update_reach_bars()

    def _update_axis(self) -> None:
        w = max(self.width(), 1)
        self._axis_item.setLine(QLineF(0, _AXIS_Y, w, _AXIS_Y))
        self._scene.setSceneRect(QRectF(0, 0, w, _H))

    def _update_curr_frame_indicator(self) -> None:
        cx = self._frame_to_x(self._curr_frame)
        self._cf_item.setPolygon(_down_triangle(cx))

    def _update_mark_paths(self) -> None:
        """Rebuild per-type QPainterPaths for marks visible in the current window."""
        win_start = self._curr_frame - self._span
        win_end = self._curr_frame + self._span

        for mtype, vals in self._sorted_marks_by_type.items():
            path = QPainterPath()
            if vals:
                lo = bisect_left(vals, win_start)
                hi = bisect_right(vals, win_end)
                for fv in vals[lo:hi]:
                    x = self._frame_to_x(fv)
                    path.moveTo(x, _MARK_TOP)
                    path.lineTo(x, _MARK_BOTTOM)
            self._mark_path_items[mtype].setPath(path)

    def _update_ticks(self) -> None:
        """Rebuild tick lines and frame-number labels for the visible window."""
        scene = self._scene
        for item in self._tick_items:
            scene.removeItem(item)
        for item in self._tick_label_items:
            scene.removeItem(item)
        self._tick_items = []
        self._tick_label_items = []

        _tick_pen = QPen(_TICK_COLOR, 1)
        _tick_pen.setCosmetic(True)

        font = QFont()
        font.setPixelSize(9)

        s = self._span
        # Aim for 8-16 major ticks across the visible range
        _spacings = [1, 2, 5, 10, 25, 50, 100, 200, 500, 1000, 2000, 5000]
        spacing = _spacings[-1]
        for sp in _spacings:
            if (2 * s) / sp <= 16:
                spacing = sp
                break

        start_f = int((self._curr_frame - s) // spacing) * spacing
        end_f = int(self._curr_frame + s + spacing)

        prev_label_right = -999.0
        for f in range(start_f, end_f, spacing):
            if f < 0 or f > self._total_frames:
                continue
            x = self._frame_to_x(f)
            is_major = spacing == 1 or (f % (spacing * 5) == 0)
            y_top = _TICK_MAJOR_TOP if is_major else _TICK_MINOR_TOP

            tick = scene.addLine(QLineF(x, y_top, x, _AXIS_Y), _tick_pen)
            tick.setZValue(1)
            self._tick_items.append(tick)

            if is_major:
                lbl = scene.addSimpleText(str(f + 1), font)
                lbl.setBrush(QBrush(_LABEL_COLOR))
                lbl_rect = lbl.boundingRect()
                lbl_x = x - lbl_rect.width() / 2
                if lbl_x > prev_label_right + 3:
                    lbl.setPos(lbl_x, _LABEL_Y)
                    lbl.setZValue(1)
                    self._tick_label_items.append(lbl)
                    prev_label_right = lbl_x + lbl_rect.width()
                else:
                    scene.removeItem(lbl)

    def _update_reach_bars(self) -> None:
        """Rebuild reach-segment bars for the visible window.

        Each reach is drawn as:
        - A filled colored rect spanning [frame, end_frame] at height _REACH_Y.
        - A vertical line at max_frame extending ±_REACH_EXTRA beyond the bar.
        """
        scene = self._scene
        for item in self._reach_items:
            scene.removeItem(item)
        self._reach_items = []

        win_start = self._curr_frame - self._span
        win_end = self._curr_frame + self._span

        for reach in self._reaches:
            if reach.end_frame < win_start or reach.frame > win_end:
                continue

            color = QColor(reach.outcome.ui_color)
            bar_pen = QPen(color.lighter(140), 1.2)
            bar_pen.setCosmetic(True)
            bar_brush = QBrush(color)

            x0 = self._frame_to_x(reach.frame)
            x1 = self._frame_to_x(reach.end_frame)
            bar_w = max(x1 - x0, 2.0)
            rect = scene.addRect(
                QRectF(x0, _REACH_Y, bar_w, _REACH_H),
                bar_pen,
                bar_brush,
            )
            rect.setZValue(8)
            rect.setToolTip(
                f"{reach.outcome.label} — F{reach.frame + 1}–F{reach.end_frame + 1}"
            )
            self._reach_items.append(rect)

            # Vertical line at reachMax
            mx = self._frame_to_x(reach.max_frame)
            vline_pen = QPen(QColor(255, 255, 255, 210), 1.5)
            vline_pen.setCosmetic(True)
            vline = scene.addLine(
                QLineF(
                    mx, _REACH_Y - _REACH_EXTRA,
                    mx, _REACH_Y + _REACH_H + _REACH_EXTRA,
                ),
                vline_pen,
            )
            vline.setZValue(9)
            self._reach_items.append(vline)

    def _update_event_items(self) -> None:
        """Rebuild event-marker triangles for the visible window."""
        scene = self._scene
        for item in self._event_items:
            scene.removeItem(item)
        self._event_items = []

        win_start = self._curr_frame - self._span
        win_end = self._curr_frame + self._span

        for evt in self._events:
            frame = int(evt["frame"])
            if not (win_start <= frame <= win_end):
                continue
            cx = self._frame_to_x(frame)
            color = QColor(*evt["color"])
            pen = QPen(color.darker(130), 1)
            pen.setCosmetic(True)
            tri = scene.addPolygon(_up_triangle(cx), pen, QBrush(color))
            tri.setToolTip(f"{evt.get('event', '')} — frame {frame + 1}")
            tri.setZValue(5)
            self._event_items.append(tri)

    # ── Qt event overrides ───────────────────────────────────────────────── #

    def _on_frame_changed(self, frame_idx) -> None:
        self._curr_frame = frame_idx or 0
        self._full_redraw()

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            frame = self._x_to_frame(event.pos().x())
            frame = max(0, min(frame, self._total_frames - 1))
            self._state["frame_idx"] = frame
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        x = float(event.pos().x())
        frame = self._x_to_frame(x)
        frame = max(0, min(frame, self._total_frames - 1))

        self._cursor_line.setLine(QLineF(x, 0, x, _H))
        self._cursor_line.show()

        label_text = f"F {frame + 1}"
        self._cursor_lbl.setText(label_text)
        lbl_w = self._cursor_lbl.boundingRect().width()
        lbl_x = min(x + 4, max(self.width() - lbl_w - 2, 0))
        self._cursor_lbl.setPos(lbl_x, 2)
        self._cursor_lbl.show()

        super().mouseMoveEvent(event)

    def leaveEvent(self, event) -> None:
        self._cursor_line.hide()
        self._cursor_lbl.hide()
        super().leaveEvent(event)

    def resizeEvent(self, event=None) -> None:
        if event:
            super().resizeEvent(event)
        self._full_redraw()


# ─────────────────────────────── module helpers ───────────────────────────── #

def _down_triangle(cx: float, half_w: int = 5) -> QPolygonF:
    """Downward-pointing ▽ centred at cx, spanning _CF_BASE_Y → _CF_TIP_Y."""
    return QPolygonF([
        QPointF(cx - half_w, _CF_BASE_Y),
        QPointF(cx + half_w, _CF_BASE_Y),
        QPointF(cx,          _CF_TIP_Y),
    ])


def _up_triangle(cx: float, half_w: int = 6) -> QPolygonF:
    """Upward-pointing △ centred at cx, spanning _EV_BASE_Y → _EV_TIP_Y."""
    return QPolygonF([
        QPointF(cx - half_w, _EV_BASE_Y),
        QPointF(cx + half_w, _EV_BASE_Y),
        QPointF(cx,          _EV_TIP_Y),
    ])
