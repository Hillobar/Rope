"""Thin QLabel wrapper preserving the Tk GE.Text API (.configure(text=...))."""

from __future__ import annotations

from PySide6.QtWidgets import QLabel, QWidget


class Text(QLabel):
    def __init__(self, parent: QWidget | None = None, text: str = "", *, tier: int = 2):
        super().__init__(text, parent)
        self.setProperty("panelTier", str(tier))

    def configure(self, *, text: str | None = None) -> None:
        if text is not None:
            self.setText(text)
