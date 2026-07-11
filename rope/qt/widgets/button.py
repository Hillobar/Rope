"""State-aware icon/text button replacing rope/GUIElements.py:Button.

State machine encoded as a QSS dynamic property `state`:
    'off'   — default, not toggled
    'on'    — toggled / active (e.g. Play in playback)
    'hover' — mouse over (rendered by QSS :hover automatically)
    'error' — red-tinted, set via error_button() for missing-folder cases

Public API mirrors the Tk version: enable_button, disable_button,
toggle_button, temp_disable_button, temp_enable_button, error_button,
set(value, request_frame), get(), load_default(), add_info_frame(info).
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import QLabel, QPushButton, QWidget

from rope.qt.parameters import ButtonParam


def _load_icon(path: str | None) -> QIcon:
    if not path:
        return QIcon()
    p = Path(path)
    if not p.is_file():
        return QIcon()
    return QIcon(QPixmap(str(p)))


class IconButton(QPushButton):
    """The Tk Button supported callable `callback(argument)` semantics.
    Here we expose a `clicked_with_arg(name, argument)` signal in addition
    to QPushButton's native clicked, so Phase C can wire either way.
    """

    clicked_with_arg = Signal(str, object)

    def __init__(
        self,
        param: ButtonParam,
        *,
        callback: Optional[Callable[..., None]] = None,
        argument: object = None,
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self._param = param
        self._argument = argument
        self._callback = callback
        self._info: Optional[QLabel] = None
        self._state: str = "off"
        self._enabled_state: bool = True

        self.setMinimumSize(param.width, param.height)
        self._icon_off = _load_icon(param.icon_off)
        self._icon_on = _load_icon(param.icon_on)
        self._icon_hover = _load_icon(param.icon_hover)
        self.setIconSize(QSize(param.height - 2, param.height - 2))

        display = param.display
        if display in ("icon", "both"):
            self.setIcon(self._icon_off)
        if display in ("text", "both"):
            self.setText(param.text)
        else:
            self.setText("")

        self._apply_state("off")
        self.clicked.connect(self._on_clicked)

    def _apply_state(self, state: str) -> None:
        self._state = state
        self.setProperty("state", state)
        # Force re-polish so QSS attribute selector takes effect
        self.style().unpolish(self)
        self.style().polish(self)
        if self._param.display in ("icon", "both"):
            if state == "on" and not self._icon_on.isNull():
                self.setIcon(self._icon_on)
            else:
                self.setIcon(self._icon_off)

    def _on_clicked(self) -> None:
        if self._callback is not None:
            try:
                self._callback(self._argument) if self._argument is not None else self._callback()
            except TypeError:
                self._callback()
        self.clicked_with_arg.emit(self._param.name, self._argument)

    # --- Public Tk-style API ---------------------------------------------------

    def get(self) -> bool:
        return self._state == "on"

    def set(self, value: bool, request_frame: bool = True) -> None:
        self._apply_state("on" if value else "off")
        if request_frame and self._callback is not None:
            self._on_clicked()

    def toggle_button(self) -> None:
        self._apply_state("off" if self._state == "on" else "on")

    def enable_button(self) -> None:
        self._enabled_state = True
        self.setEnabled(True)

    def disable_button(self) -> None:
        self._enabled_state = False
        self.setEnabled(False)

    def temp_disable_button(self) -> None:
        self.setEnabled(False)

    def temp_enable_button(self) -> None:
        self.setEnabled(self._enabled_state)

    def error_button(self) -> None:
        self._apply_state("error")

    def load_default(self) -> None:
        self.set(self._param.default, request_frame=False)

    def add_info_frame(self, info: QLabel) -> None:
        self._info = info

    def enterEvent(self, event) -> None:
        if self._info is not None:
            self._info.setText(self._param.info_text)
        if self._state == "off" and not self._icon_hover.isNull():
            self.setIcon(self._icon_hover)
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:
        if self._state == "off" and not self._icon_off.isNull():
            self.setIcon(self._icon_off)
        super().leaveEvent(event)

    def get_data_type(self) -> str:
        # Tk widgets returned 'control' or 'parameter' here; ButtonParam's
        # scope encodes the same thing.
        return self._param.scope
