"""ParameterSlider — composite (QLabel + QSlider + QLineEdit).

Replaces rope/GUIElements.py:Slider2 and :Slider3. QSlider operates on
ints, so fractional sliders (e.g. ColorGammaSlider with inc=0.01) are
internally scaled by 1/inc — the slider's int position represents (value -
min) / inc, the visible value is `min + position * inc`. The QLineEdit
shows and accepts the float value.

Emits `value_changed(name, float)`. Wheel adjusts by one inc step.
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QDoubleValidator
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QSlider,
    QWidget,
)

from rope.qt.parameters import SliderParam


def _is_integral(p: SliderParam) -> bool:
    return float(p.inc).is_integer() and float(p.min).is_integer() and float(p.max).is_integer()


class ParameterSlider(QWidget):
    value_changed = Signal(str, object)

    def __init__(
        self,
        param: SliderParam,
        *,
        label_percent: float = 0.38,
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self._param = param
        self._info: Optional[QLabel] = None
        self._integral = _is_integral(param)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self._label = QLabel(param.label + " ")
        self._label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        layout.addWidget(self._label, stretch=int(label_percent * 100))

        # Slider operates on int positions = round((value - min) / inc).
        self._steps = max(1, int(round((param.max - param.min) / param.inc)))
        self._slider = QSlider(Qt.Horizontal)
        self._slider.setRange(0, self._steps)
        self._slider.setSingleStep(1)
        self._slider.setPageStep(max(1, self._steps // 10))
        self._slider.valueChanged.connect(self._on_slider_changed)
        layout.addWidget(self._slider, stretch=int((1 - label_percent) * 100))

        self._entry = QLineEdit()
        # 0.6x the previous 60 px — a narrower value box next to the slider.
        self._entry.setFixedWidth(36)
        self._entry.setAlignment(Qt.AlignRight)
        validator = QDoubleValidator(float(param.min), float(param.max), 4, self._entry)
        validator.setNotation(QDoubleValidator.StandardNotation)
        self._entry.setValidator(validator)
        self._entry.editingFinished.connect(self._on_entry_committed)
        layout.addWidget(self._entry, stretch=0)

        self.set(param.default, request_frame=False)

    # --- conversions ----------------------------------------------------------

    def _pos_to_value(self, pos: int):
        v = float(self._param.min) + pos * float(self._param.inc)
        # Clamp to range (rounding error guard)
        v = max(self._param.min, min(self._param.max, v))
        # Integer-step sliders return an actual int so downstream code
        # that uses these as slice indices, range() bounds, or kernel
        # sizes doesn't have to cast. Float sliders return float.
        return int(round(v)) if self._integral else v

    def _value_to_pos(self, value: float) -> int:
        return int(round((float(value) - float(self._param.min)) / float(self._param.inc)))

    def _format(self, value: float) -> str:
        return str(int(value)) if self._integral else f"{value:g}"

    # --- handlers -------------------------------------------------------------

    def _on_slider_changed(self, pos: int) -> None:
        value = self._pos_to_value(pos)
        # Update entry without recursion
        blocked = self._entry.blockSignals(True)
        try:
            self._entry.setText(self._format(value))
        finally:
            self._entry.blockSignals(blocked)
        self.value_changed.emit(self._param.name, value)

    def _on_entry_committed(self) -> None:
        try:
            value = float(self._entry.text())
        except ValueError:
            self._entry.setText(self._format(self.get()))
            return
        value = max(self._param.min, min(self._param.max, value))
        pos = self._value_to_pos(value)
        # Slider's valueChanged will fire and emit value_changed for us.
        blocked = self._slider.blockSignals(False)  # ensure unblocked
        self._slider.blockSignals(blocked)
        self._slider.setValue(pos)
        # If setValue didn't change anything (already at pos), still emit
        # so external listeners pick up the user's commit.
        if self._slider.value() == pos:
            self.value_changed.emit(self._param.name, self._pos_to_value(pos))

    # --- public API -----------------------------------------------------------

    def get(self) -> float:
        return self._pos_to_value(self._slider.value())

    def set(self, value: float, request_frame: bool = True) -> None:
        try:
            value = float(value)
        except (TypeError, ValueError):
            value = float(self._param.default)
        value = max(self._param.min, min(self._param.max, value))
        pos = self._value_to_pos(value)
        slider_blocked = self._slider.blockSignals(not request_frame)
        entry_blocked = self._entry.blockSignals(True)
        try:
            self._slider.setValue(pos)
            self._entry.setText(self._format(self._pos_to_value(pos)))
        finally:
            self._slider.blockSignals(slider_blocked)
            self._entry.blockSignals(entry_blocked)
        if request_frame:
            self.value_changed.emit(self._param.name, self._pos_to_value(pos))

    def load_default(self) -> None:
        self.set(self._param.default, request_frame=False)

    def add_info_frame(self, info: QLabel) -> None:
        self._info = info

    def enterEvent(self, event) -> None:
        if self._info is not None:
            self._info.setText(self._param.info_text)
        super().enterEvent(event)

    def get_data_type(self) -> str:
        return self._param.scope
