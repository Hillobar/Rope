"""Exclusive radio-button group replacing rope/GUIElements.py:TextSelection.

Renders a QLabel + a row of checkable QPushButtons (QButtonGroup with
setExclusive=True). Selected button gets QSS property state="on".
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QWidget,
)

from rope.qt.parameters import SelectParam


class TextSelection(QWidget):
    value_changed = Signal(str, object)  # (name, mode str)

    def __init__(
        self,
        param: SelectParam,
        *,
        label_percent: float = 0.38,
        justify: str = "fill",
        parent: QWidget | None = None,
    ):
        """justify:
            'fill' (default) — label takes `label_percent` of the width
                (right-aligned) and the buttons stretch to fill the rest.
            'left' — label + buttons pack against the left edge at their
                natural widths, with a trailing stretch pushing them left.
        """
        super().__init__(parent)
        self._param = param
        self._info: Optional[QLabel] = None
        left = justify == "left"

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self._label = QLabel(param.label + " ")
        self._label.setAlignment(
            (Qt.AlignLeft if left else Qt.AlignRight) | Qt.AlignVCenter
        )
        layout.addWidget(self._label, stretch=0 if left else int(label_percent * 100))

        self._group = QButtonGroup(self)
        self._group.setExclusive(True)
        self._buttons: dict[str, QPushButton] = {}
        modes = param.modes
        for mode in modes:
            btn = QPushButton(mode)
            btn.setCheckable(True)
            btn.setProperty("state", "off")
            btn.toggled.connect(lambda checked, m=mode: self._on_toggled(m, checked))
            self._group.addButton(btn)
            self._buttons[mode] = btn
            layout.addWidget(btn, stretch=0 if left else 1)

        if left:
            layout.addStretch(1)

        self.set(param.default, request_frame=False)

    def _on_toggled(self, mode: str, checked: bool) -> None:
        if not checked:
            return  # only react to the newly-on button
        for m, b in self._buttons.items():
            b.setProperty("state", "on" if m == mode else "off")
            b.style().unpolish(b); b.style().polish(b)
        self.value_changed.emit(self._param.name, mode)

    def get(self) -> str:
        for mode, btn in self._buttons.items():
            if btn.isChecked():
                return mode
        return self._param.default

    def set(self, value: str, request_frame: bool = True) -> None:
        if value not in self._buttons:
            value = self._param.default
        for mode, btn in self._buttons.items():
            blocked = btn.blockSignals(not request_frame)
            try:
                btn.setChecked(mode == value)
                btn.setProperty("state", "on" if mode == value else "off")
                btn.style().unpolish(btn); btn.style().polish(btn)
            finally:
                btn.blockSignals(blocked)
        if request_frame:
            self.value_changed.emit(self._param.name, value)

    def load_default(self) -> None:
        self.set(self._param.default, request_frame=False)

    def add_info_frame(self, info: QLabel) -> None:
        self._info = info

    def enterEvent(self, event) -> None:
        if self._info is not None:
            self._info.setText(self._param.info_text)
        super().enterEvent(event)
