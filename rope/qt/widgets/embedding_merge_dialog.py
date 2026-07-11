"""EmbeddingMerge dialog — combine selected source-face embeddings.

Per-row checkboxes choose which loaded source faces participate; the
mode dropdown selects the combine strategy from rope/EmbeddingMerge.py
(Mean / Median / Sph / Geo / Qual). A name field sets the label for the
saved merged embedding. OK runs the combine and emits the result; the
caller (main_window) is responsible for persisting it to
settings.merged_embeddings_file.

The dialog itself is backend-free: it accepts a list of
`(name, embedding_or_None, thumbnail_path_or_None)` tuples at
construction and returns a `MergeRequest(name, mode, embeddings, indices)`
on accept. If embeddings are None (Phase E placeholder — the actual
face-detection / embedding-load flow is wired in the post-port follow-up)
the dialog still lets the user pick mode + name, and returns
indices-only so the caller can resolve embeddings later.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QVBoxLayout,
    QWidget,
)

from rope.EmbeddingMerge import combine

MERGE_MODES = ["Mean", "Median", "Sph", "Geo", "Qual"]


@dataclass(frozen=True)
class FaceEntry:
    name: str
    embedding: Optional[np.ndarray] = None
    thumbnail_path: Optional[str] = None


@dataclass
class MergeRequest:
    name: str
    mode: str
    selected_indices: list[int]
    merged_embedding: Optional[np.ndarray]  # None if embeddings weren't provided


class EmbeddingMergeDialog(QDialog):
    def __init__(self, faces: Sequence[FaceEntry], parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("Merge Source-Face Embeddings")
        self.resize(480, 520)
        self.setModal(True)

        self._faces = list(faces)
        self._result: Optional[MergeRequest] = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        layout.addWidget(QLabel("Select faces to combine:"))
        self._list = QListWidget()
        self._list.setSpacing(2)
        self._checkboxes: list[QCheckBox] = []
        for face in self._faces:
            item = QListWidgetItem(self._list)
            row = QWidget()
            row_lay = QHBoxLayout(row)
            row_lay.setContentsMargins(4, 2, 4, 2)
            cb = QCheckBox(face.name or "(unnamed)")
            cb.setChecked(False)
            self._checkboxes.append(cb)
            row_lay.addWidget(cb, stretch=1)
            if face.thumbnail_path:
                thumb = QLabel()
                pix = QPixmap(face.thumbnail_path).scaled(
                    32, 32, Qt.KeepAspectRatio, Qt.SmoothTransformation,
                )
                if not pix.isNull():
                    thumb.setPixmap(pix)
                row_lay.addWidget(thumb)
            item.setSizeHint(row.sizeHint())
            self._list.addItem(item)
            self._list.setItemWidget(item, row)
        layout.addWidget(self._list, stretch=1)

        # Mode + name row
        opts = QHBoxLayout()
        opts.setSpacing(8)
        opts.addWidget(QLabel("Mode:"))
        self._mode = QComboBox()
        for m in MERGE_MODES:
            self._mode.addItem(m)
        opts.addWidget(self._mode)
        opts.addSpacing(12)
        opts.addWidget(QLabel("Name:"))
        self._name = QLineEdit()
        self._name.setPlaceholderText("merged_face_01")
        opts.addWidget(self._name, stretch=1)
        layout.addLayout(opts)

        # Status / error line
        self._status = QLabel("")
        self._status.setStyleSheet("color: #B0B0B0; font-size: 9pt;")
        layout.addWidget(self._status)

        # OK/Cancel
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _on_accept(self):
        selected = [i for i, cb in enumerate(self._checkboxes) if cb.isChecked()]
        if not selected:
            self._status.setText("Pick at least one face.")
            return
        name = self._name.text().strip()
        if not name:
            self._status.setText("Give the merged embedding a name.")
            return
        mode = self._mode.currentText()

        # If embeddings were supplied, compute the merge now. If any are
        # None we skip computation (caller resolves later via indices).
        embs = [self._faces[i].embedding for i in selected]
        merged: Optional[np.ndarray] = None
        if all(e is not None for e in embs) and embs:
            try:
                merged = combine([np.asarray(e, dtype=np.float32) for e in embs], mode)
            except Exception as exc:
                self._status.setText(f"merge failed: {exc}")
                return

        self._result = MergeRequest(
            name=name,
            mode=mode,
            selected_indices=selected,
            merged_embedding=merged,
        )
        self.accept()

    def result_payload(self) -> Optional[MergeRequest]:
        return self._result
