"""Timeline scrubber — custom QWidget with paintEvent.

Replaces rope/GUIElements.py:Timeline. Renders a trough, a playhead line,
optional parameter-marker ticks (in light-goldenrod, matching the legacy
markers_canvas), and a numeric frame entry on the right edge.

Mouse semantics mirror the Tk version:
    press   -> pause playback, jump playhead, emit scrub_started + frame_requested
    drag    -> update playhead, emit frame_requested (coalesced upstream)
    release -> emit scrub_ended (re-enable swapper)
    wheel   -> +/- 1 frame
"""

from __future__ import annotations

from PySide6.QtCore import QRect, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QHBoxLayout, QLineEdit, QSizePolicy, QWidget


_TROUGH_COLOR = QColor("#43474D")
_PLAYHEAD_COLOR = QColor("#FFFFFF")
_MARKER_COLOR = QColor("light goldenrod")
_BG_COLOR = QColor("#212126")

_SLIDER_PAD_LEFT = 20
_SLIDER_PAD_RIGHT = 20
_ENTRY_WIDTH = 50


class Timeline(QWidget):
    frame_requested = Signal(int)   # user-driven scrub (request a new frame)
    scrub_started = Signal()        # mouse pressed on trough
    scrub_ended = Signal()          # mouse released
    position_changed = Signal(int)  # any position change, including programmatic

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._max = 100
        self._position = 0
        self._markers: list[int] = []
        self._dragging = False

        self.setMinimumHeight(28)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setAutoFillBackground(False)

        # Entry pinned to the right edge of the widget.
        self._entry = QLineEdit(self)
        self._entry.setFixedWidth(_ENTRY_WIDTH)
        self._entry.setAlignment(Qt.AlignRight)
        self._entry.editingFinished.connect(self._on_entry_committed)
        self._entry.setText("0")

    # --- geometry helpers -----------------------------------------------------

    def _trough_rect(self) -> QRect:
        h = self.height()
        slider_left = _SLIDER_PAD_LEFT
        slider_right = self.width() - _SLIDER_PAD_RIGHT - _ENTRY_WIDTH - 8
        return QRect(slider_left, h // 2 - 1, max(0, slider_right - slider_left), 2)

    def _slider_left(self) -> int:
        return _SLIDER_PAD_LEFT

    def _slider_right(self) -> int:
        return self.width() - _SLIDER_PAD_RIGHT - _ENTRY_WIDTH - 8

    def _coord_to_pos(self, x: int) -> int:
        left, right = self._slider_left(), self._slider_right()
        if right <= left:
            return 0
        x = max(left, min(right, x))
        return int(round((x - left) * self._max / (right - left)))

    def _pos_to_coord(self, pos: int) -> int:
        left, right = self._slider_left(), self._slider_right()
        if self._max <= 0:
            return left
        return int(left + pos * (right - left) / self._max)

    # --- events ---------------------------------------------------------------

    def resizeEvent(self, event):
        self._entry.move(self.width() - _ENTRY_WIDTH - 4, (self.height() - self._entry.sizeHint().height()) // 2)
        super().resizeEvent(event)

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), _BG_COLOR)

        # Trough
        trough = self._trough_rect()
        painter.fillRect(trough, _TROUGH_COLOR)

        # Markers (vertical ticks above + below trough centerline)
        if self._markers and self._max > 0:
            pen = QPen(_MARKER_COLOR)
            pen.setWidth(1)
            painter.setPen(pen)
            centerline_y = self.height() // 2
            for frame in self._markers:
                x = self._pos_to_coord(int(frame))
                painter.drawLine(x, centerline_y - 7, x, centerline_y + 7)

        # Playhead
        pen = QPen(_PLAYHEAD_COLOR)
        pen.setWidth(2)
        painter.setPen(pen)
        x = self._pos_to_coord(self._position)
        painter.drawLine(x, self.height() // 2 - 9, x, self.height() // 2 + 9)

        painter.end()

    def mousePressEvent(self, event):
        if event.button() != Qt.LeftButton:
            return
        self._dragging = True
        new_pos = self._coord_to_pos(event.position().toPoint().x())
        self.scrub_started.emit()
        self._set_position_internal(new_pos, emit_request=True)

    def mouseMoveEvent(self, event):
        if not self._dragging:
            return
        new_pos = self._coord_to_pos(event.position().toPoint().x())
        if new_pos != self._position:
            self._set_position_internal(new_pos, emit_request=True)

    def mouseReleaseEvent(self, event):
        if event.button() != Qt.LeftButton:
            return
        if self._dragging:
            self._dragging = False
            self.scrub_ended.emit()

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        step = 1 if delta > 0 else -1
        new_pos = max(0, min(self._max, self._position + step))
        if new_pos != self._position:
            self._set_position_internal(new_pos, emit_request=True)

    def _on_entry_committed(self):
        try:
            pos = int(float(self._entry.text()))
        except ValueError:
            self._entry.setText(str(self._position))
            return
        pos = max(0, min(self._max, pos))
        self._set_position_internal(pos, emit_request=True)

    # --- internal -------------------------------------------------------------

    def _set_position_internal(self, pos: int, *, emit_request: bool) -> None:
        pos = max(0, min(self._max, int(pos)))
        if pos == self._position:
            return
        self._position = pos
        blocked = self._entry.blockSignals(True)
        try:
            self._entry.setText(str(pos))
        finally:
            self._entry.blockSignals(blocked)
        self.update()
        self.position_changed.emit(pos)
        if emit_request:
            self.frame_requested.emit(pos)

    # --- public API (mirroring Tk Timeline) -----------------------------------

    def set(self, value: int) -> None:
        self._set_position_internal(int(value), emit_request=False)

    def get(self) -> int:
        return int(self._position)

    def set_length(self, value: int) -> None:
        self._max = max(0, int(value))
        if self._position > self._max:
            self._set_position_internal(self._max, emit_request=False)
        self.update()

    def get_length(self) -> int:
        return int(self._max)

    def set_markers(self, frames: list[int]) -> None:
        self._markers = [int(f) for f in frames]
        self.update()

    def add_info_frame(self, _info) -> None:
        # Timeline doesn't carry hover-info text in the Tk version, but we
        # accept the call to keep the common widget API uniform.
        pass
