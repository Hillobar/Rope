"""Free-form text input — label + QLineEdit composite.

Replaces rope/GUIElements.py:Text_Entry. Emits `value_changed(name, str)`
when the user presses Enter. Public API mirrors the Tk version:

    .get() / .set(value, request_frame=True) / .load_default() / .add_info_frame(info_label)
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QHBoxLayout, QLabel, QLineEdit, QWidget

from rope.qt.parameters import EntryParam


class TextEntry(QWidget):
    value_changed = Signal(str, object)  # (name, value)

    def __init__(self, param: EntryParam, *, label_percent: float = 0.4, parent: QWidget | None = None):
        super().__init__(parent)
        self._param = param
        self._info: Optional[QLabel] = None

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self._label = QLabel(param.label + " ")
        self._label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        layout.addWidget(self._label, stretch=int(label_percent * 100))

        self._entry = QLineEdit()
        self._entry.setText(str(param.default))
        self._entry.editingFinished.connect(self._on_committed)
        layout.addWidget(self._entry, stretch=int((1 - label_percent) * 100))

    def _on_committed(self) -> None:
        self.value_changed.emit(self._param.name, self._entry.text())

    def get(self) -> str:
        return self._entry.text()

    def set(self, value: str, request_frame: bool = True) -> None:
        self._entry.setText("" if value is None else str(value))
        if request_frame:
            self.value_changed.emit(self._param.name, self._entry.text())

    def load_default(self) -> None:
        self.set(self._param.default, request_frame=False)

    def add_info_frame(self, info: QLabel) -> None:
        self._info = info

    def enterEvent(self, event) -> None:
        if self._info is not None:
            self._info.setText(self._param.info_text)
        super().enterEvent(event)
