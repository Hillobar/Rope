"""VRAM usage bar — QProgressBar with QSS chunk-color flip at >90%.

Replaces rope/GUIElements.py:VRAM_Indicator.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QProgressBar, QWidget


class VRAMIndicator(QProgressBar):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setRange(0, 100)
        self.setValue(0)
        self.setFormat("VRAM: --")
        self.setAlignment(Qt.AlignCenter)
        self.setProperty("overLimit", "false")

    def set(self, used_gb: float, total_gb: float) -> None:
        if total_gb <= 0:
            pct = 0.0
        else:
            pct = max(0.0, min(100.0, (float(used_gb) / float(total_gb)) * 100.0))
        self.setValue(int(round(pct)))
        self.setFormat(f"VRAM {used_gb:.1f}/{total_gb:.1f} GB ({pct:.0f}%)")
        over = pct >= 90.0
        self.setProperty("overLimit", "true" if over else "false")
        self.style().unpolish(self); self.style().polish(self)
