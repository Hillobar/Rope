"""Horizontal thumbnail strip for detected (or merged) faces.

A row of clickable face thumbnails inside a horizontal QScrollArea.
Used for:
- Found Faces gallery (faces detected by FindFaces in the current frame)
- Merged Faces gallery (saved merged embeddings)

Each entry is a 64x64 thumbnail with an optional "assigned" border tint.
Public API:
    clear()                  — remove all thumbnails
    add(rgb_thumb, assigned) — append one, returns its index
    set_assigned(i, on)      — toggle a thumbnail's assigned highlight
    face_clicked = Signal(int)  — emitted when the user clicks a thumbnail
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)


THUMB_SIZE = 64


def _rgb_to_pixmap(rgb: np.ndarray, *, size: int = THUMB_SIZE) -> QPixmap:
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        return QPixmap()
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    if not rgb.flags["C_CONTIGUOUS"]:
        rgb = np.ascontiguousarray(rgb)
    h, w = rgb.shape[:2]
    qimg = QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888).copy()
    return QPixmap.fromImage(qimg).scaled(
        size, size, Qt.KeepAspectRatio, Qt.SmoothTransformation,
    )


class _FaceTile(QLabel):
    clicked = Signal(int)

    def __init__(self, index: int, parent: QWidget | None = None):
        super().__init__(parent)
        self._index = index
        self._assigned = False
        self._selected = False
        self.setFixedSize(THUMB_SIZE + 8, THUMB_SIZE + 8)
        self.setAlignment(Qt.AlignCenter)
        self.setCursor(Qt.PointingHandCursor)
        self._apply_border()

    def _apply_border(self) -> None:
        # Selected (current target slot) wins the outer border. Assigned
        # state is shown via background tint so both states are readable
        # when a single tile is both assigned and selected.
        if self._selected:
            bg = "#3A3530" if self._assigned else "#2E2E36"
            self.setStyleSheet(
                f"background-color: {bg}; border: 3px solid #FFD400; padding: 1px;"
            )
        elif self._assigned:
            self.setStyleSheet(
                "background-color: #28282E; border: 2px solid #d10303; padding: 2px;"
            )
        else:
            self.setStyleSheet(
                "background-color: #28282E; border: 2px solid #404040; padding: 2px;"
            )

    def set_pixmap(self, pix: QPixmap) -> None:
        self.setPixmap(pix)

    def set_assigned(self, on: bool) -> None:
        self._assigned = bool(on)
        self._apply_border()

    def set_selected(self, on: bool) -> None:
        self._selected = bool(on)
        self._apply_border()

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self.clicked.emit(self._index)


class FaceGallery(QFrame):
    face_clicked = Signal(int)

    def __init__(self, *, title: str = "", parent: QWidget | None = None):
        super().__init__(parent)
        # Match source-faces / target-videos / embeddings panes so
        # Found Faces shares the same #28282E surface tone.
        self.setProperty("panelTier", "3")
        self.setFixedHeight(THUMB_SIZE + 32)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(4, 2, 4, 2)
        outer.setSpacing(2)

        if title:
            label = QLabel(title)
            label.setStyleSheet("color: #A0A0A0; font-size: 8pt;")
            outer.addWidget(label)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self._scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._scroll.setFrameShape(QFrame.NoFrame)
        outer.addWidget(self._scroll)

        self._row = QWidget()
        self._row_layout = QHBoxLayout(self._row)
        self._row_layout.setContentsMargins(2, 2, 2, 2)
        self._row_layout.setSpacing(6)
        self._row_layout.addStretch()
        self._scroll.setWidget(self._row)

        self._tiles: list[_FaceTile] = []
        self._selected_index: int = -1

    def clear(self) -> None:
        for tile in self._tiles:
            self._row_layout.removeWidget(tile)
            tile.deleteLater()
        self._tiles.clear()
        self._selected_index = -1

    def add(self, rgb_thumb: np.ndarray, *, assigned: bool = False) -> int:
        idx = len(self._tiles)
        tile = _FaceTile(idx)
        tile.set_pixmap(_rgb_to_pixmap(rgb_thumb))
        tile.set_assigned(assigned)
        tile.clicked.connect(self.face_clicked)
        # Insert before the trailing stretch
        self._row_layout.insertWidget(self._row_layout.count() - 1, tile)
        self._tiles.append(tile)
        return idx

    def set_assigned(self, index: int, on: bool) -> None:
        if 0 <= index < len(self._tiles):
            self._tiles[index].set_assigned(on)

    def set_selected(self, index: int) -> None:
        """Make exactly one tile the current target slot. Pass -1 to
        deselect everything."""
        if index == self._selected_index:
            return
        if 0 <= self._selected_index < len(self._tiles):
            self._tiles[self._selected_index].set_selected(False)
        self._selected_index = index if 0 <= index < len(self._tiles) else -1
        if self._selected_index >= 0:
            self._tiles[self._selected_index].set_selected(True)

    def selected_index(self) -> int:
        return self._selected_index

    def count(self) -> int:
        return len(self._tiles)
