"""Toggle switch — pill-shaped, custom-painted, with sliding knob.

Replaces rope/GUIElements.py:Switch2 and the prior icon-PNG version.
The toggle is drawn directly so we get clean color transitions and a
short slide animation without depending on bitmap assets that don't
scale or recolor with the rest of the UI. Emits `value_changed(name,
bool)`.
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import (
    Property,
    QEasingCurve,
    QPropertyAnimation,
    QRectF,
    Qt,
    Signal,
)
from PySide6.QtGui import QBrush, QColor, QPainter, QPen
from PySide6.QtWidgets import QAbstractButton, QHBoxLayout, QLabel, QWidget

from rope.qt.parameters import SwitchParam


# Colors picked to match the rest of the app's selection / assigned
# language — gold accent when on, neutral dark when off. Both share
# the same rail darkness so the toggle reads as a single component
# whose accent shifts state-to-state.
_OFF_BG = QColor("#1F1F26")
_OFF_BORDER = QColor("#404040")
_OFF_HOVER_BORDER = QColor("#606070")
_OFF_KNOB = QColor("#9090A0")

_ON_BG = QColor("#3D3520")
_ON_BORDER = QColor("#E5B854")
_ON_HOVER_BORDER = QColor("#FFD400")
_ON_KNOB = QColor("#FFD400")

_DIS_BG = QColor("#1A1A1E")
_DIS_BORDER = QColor("#303034")
_DIS_KNOB = QColor("#3A3A40")


class _ToggleSwitch(QAbstractButton):
    """Custom-painted pill toggle. Owned by Switch; not used directly.

    The knob position is exposed as a Qt property so QPropertyAnimation
    can drive it across the rail on user toggles. Programmatic state
    changes (apply_values / load_default) bypass the animation via
    snap_to_state() — those are bulk operations where animating
    every widget would feel laggy.
    """

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setCheckable(True)
        self.setCursor(Qt.PointingHandCursor)
        # Slightly wider than tall for the classic pill aspect.
        self.setFixedSize(39, 20)
        self._knob_pos: float = 0.0  # 0.0 = left (off), 1.0 = right (on)

        self._anim = QPropertyAnimation(self, b"knobPos", self)
        self._anim.setDuration(120)
        self._anim.setEasingCurve(QEasingCurve.OutCubic)

        # toggled fires on user clicks. Switch.set blocks signals during
        # programmatic state changes, so this connection only animates
        # interactive toggles — exactly the right cadence.
        self.toggled.connect(self._animate_to_state)

    # ---- Qt property bridge for QPropertyAnimation ---------------------------

    def _get_knob_pos(self) -> float:
        return self._knob_pos

    def _set_knob_pos(self, v: float) -> None:
        self._knob_pos = float(v)
        self.update()

    knobPos = Property(float, _get_knob_pos, _set_knob_pos)

    # ---- State transitions ---------------------------------------------------

    def _animate_to_state(self, checked: bool) -> None:
        self._anim.stop()
        self._anim.setStartValue(self._knob_pos)
        self._anim.setEndValue(1.0 if checked else 0.0)
        self._anim.start()

    def snap_to_state(self) -> None:
        """Snap the knob to match isChecked() with no animation. Used
        for programmatic state changes where the visual should be
        settled immediately (e.g. apply_values during JSON load)."""
        self._anim.stop()
        self._knob_pos = 1.0 if self.isChecked() else 0.0
        self.update()

    # ---- Painting ------------------------------------------------------------

    def paintEvent(self, _event) -> None:
        is_on = self.isChecked()
        is_enabled = self.isEnabled()
        is_hover = self.underMouse() and is_enabled

        if not is_enabled:
            bg = _DIS_BG
            border = _DIS_BORDER
            knob = _DIS_KNOB
        elif is_on:
            bg = _ON_BG
            border = _ON_HOVER_BORDER if is_hover else _ON_BORDER
            knob = _ON_KNOB
        else:
            bg = _OFF_BG
            border = _OFF_HOVER_BORDER if is_hover else _OFF_BORDER
            knob = _OFF_KNOB

        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        radius = h / 2.0

        # Background pill. Adjust the rect by half a pixel so the 1-px
        # border stroke sits on whole pixels and reads as one crisp line.
        rail = QRectF(0.5, 0.5, w - 1.0, h - 1.0)
        p.setPen(QPen(border, 1.0))
        p.setBrush(QBrush(bg))
        p.drawRoundedRect(rail, radius - 0.5, radius - 0.5)

        # Knob — a circle that slides between left and right rail padding.
        # Padding inside the rail so the knob doesn't touch the border.
        pad = 2.0
        knob_diam = h - 2.0 * pad
        knob_y = pad
        travel = w - knob_diam - 2.0 * pad
        knob_x = pad + travel * self._knob_pos
        p.setPen(Qt.NoPen)
        p.setBrush(QBrush(knob))
        p.drawEllipse(QRectF(knob_x, knob_y, knob_diam, knob_diam))

    # ---- Repaint on hover so the border highlight is live --------------------

    def enterEvent(self, event) -> None:
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:
        self.update()
        super().leaveEvent(event)


class Switch(QWidget):
    """Public widget — a labeled pill toggle. API mirrors the old
    icon-based Switch so the rest of the app doesn't need to know
    the inner widget changed."""

    value_changed = Signal(str, object)

    def __init__(self, param: SwitchParam, parent: QWidget | None = None):
        super().__init__(parent)
        self._param = param
        self._info: Optional[QLabel] = None

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        self._toggle = _ToggleSwitch()
        self._toggle.toggled.connect(self._on_toggled)
        layout.addWidget(self._toggle, alignment=Qt.AlignVCenter)

        self._label = QLabel(param.label)
        layout.addWidget(self._label, stretch=1)

        self.set(param.default, request_frame=False)

    def _on_toggled(self, checked: bool) -> None:
        self.value_changed.emit(self._param.name, bool(checked))

    def get(self) -> bool:
        return self._toggle.isChecked()

    def set(self, value: bool, request_frame: bool = True) -> None:
        # request_frame=False is used for JSON load / load_default —
        # bulk operations where the toggled signal must not fire (so
        # consumers don't get a flood of fake user-interaction events)
        # and where the visual should arrive settled, not animated.
        prev_signals_blocked = self._toggle.blockSignals(not request_frame)
        try:
            self._toggle.setChecked(bool(value))
        finally:
            self._toggle.blockSignals(prev_signals_blocked)
        if not request_frame:
            self._toggle.snap_to_state()

    def load_default(self) -> None:
        self.set(self._param.default, request_frame=False)

    def add_info_frame(self, info: QLabel) -> None:
        self._info = info

    def enterEvent(self, event) -> None:
        if self._info is not None:
            self._info.setText(self._param.info_text)
        super().enterEvent(event)
