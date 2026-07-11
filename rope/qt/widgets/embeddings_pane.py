"""Embeddings pane — grid of saved merged embeddings.

Reads / writes the classic Rope merged_embeddings.txt format. Each entry
is a `Name: <name>` line followed by 512 float lines. Clicking a tile
selects the embedding as the active source for the next Found Face
assignment. Right-click a tile for Rename / Delete. Save Current...
captures the current source-face selection (combined via the parameter
pane's MergeTextSel mode) and appends a new entry.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
from PySide6.QtCore import Qt, QItemSelectionModel, QSize, Signal
from PySide6.QtGui import QBrush, QColor, QPen
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QFrame,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListView,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QMessageBox,
    QPushButton,
    QStyledItemDelegate,
    QVBoxLayout,
    QWidget,
)


class _ReorderableListWidget(QListWidget):
    """QListWidget with live drag-to-reorder.

    Rather than going through Qt's QDrag machinery (which only updates the
    model on drop), this watches mouse press/move/release directly and
    reshuffles the QListWidgetItems in real time as the cursor crosses
    other tiles. The grid reflows immediately because ResizeMode is
    Adjust, so the user sees tiles slide aside under the cursor as they
    drag, then a final emit on release.

    Click-to-activate (itemClicked) and right-click context menu still
    work because we only intercept drags — short-distance presses fall
    through to the base implementation.
    """

    items_reordered = Signal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._drag_source_row: int = -1
        self._drag_press_pos = None
        self._is_dragging: bool = False

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            idx = self.indexAt(event.position().toPoint())
            if idx.isValid():
                self._drag_source_row = idx.row()
                self._drag_press_pos = event.position().toPoint()
                self._is_dragging = False
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if (
            self._drag_source_row >= 0
            and self._drag_press_pos is not None
            and (event.buttons() & Qt.LeftButton)
        ):
            delta = event.position().toPoint() - self._drag_press_pos
            if (
                not self._is_dragging
                and delta.manhattanLength() >= QApplication.startDragDistance()
            ):
                self._is_dragging = True
                self.setCursor(Qt.ClosedHandCursor)
            if self._is_dragging:
                target = self.indexAt(event.position().toPoint())
                target_row = target.row() if target.isValid() else self.count() - 1
                if target_row >= 0 and target_row != self._drag_source_row:
                    taken = self.takeItem(self._drag_source_row)
                    if taken is not None:
                        self.insertItem(target_row, taken)
                        self.setCurrentItem(taken)
                        self._drag_source_row = target_row
                event.accept()
                return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        was_dragging = self._is_dragging
        self._is_dragging = False
        self._drag_source_row = -1
        self._drag_press_pos = None
        self.unsetCursor()
        if was_dragging:
            self.items_reordered.emit()
            event.accept()
            return
        super().mouseReleaseEvent(event)


def parse_merged_embeddings(path: str) -> list[tuple[str, np.ndarray]]:
    """Parse a merged_embeddings.txt file into (name, vec) pairs.

    Tolerant of stray blank lines, BOMs, and trailing whitespace; skips
    non-numeric value lines silently so a corrupted entry doesn't kill
    the rest of the list.
    """
    entries: list[tuple[str, np.ndarray]] = []
    if not path or not os.path.exists(path):
        return entries
    try:
        with open(path, 'r', encoding='utf-8') as f:
            text = f.read()
    except OSError:
        return entries
    current_name: Optional[str] = None
    current_vals: list[float] = []
    for line in text.splitlines():
        s = line.strip().lstrip('﻿')
        if not s:
            continue
        if s.startswith('Name:'):
            if current_name is not None and current_vals:
                entries.append((current_name, np.asarray(current_vals, dtype=np.float32)))
            current_name = s[5:].strip()
            current_vals = []
        else:
            try:
                current_vals.append(float(s))
            except ValueError:
                pass
    if current_name is not None and current_vals:
        entries.append((current_name, np.asarray(current_vals, dtype=np.float32)))
    return entries


def write_merged_embeddings(path: str, entries: list[tuple[str, np.ndarray]]) -> None:
    """Rewrite the entire file. Matches classic Rope's on-disk format
    so a swap between this app and older Rope builds round-trips."""
    with open(path, 'w', encoding='utf-8') as f:
        for name, vec in entries:
            f.write(f"Name: {name}\n")
            for v in np.asarray(vec, dtype=np.float32).ravel():
                f.write(f"{float(v)}\n")


# Warm gold tint marking embeddings that are the assigned source for
# the currently-selected Found Face. Mirrors left_pane._ASSIGNED_TINT.
_ASSIGNED_TINT = QColor("#4A3D1A")
_ASSIGNED_FG = QColor("#E5B854")
_DEFAULT_FG = QColor("#D0D0D0")

# Custom data role marking an item as an assigned source of the
# currently-selected Found Face. Read by _AssignedBorderDelegate.
_ASSIGNED_ROLE = Qt.UserRole + 100


class _AssignedBorderDelegate(QStyledItemDelegate):
    """Paints a gold border over items flagged via _ASSIGNED_ROLE.

    Mirrors the source-faces panel's delegate so the user sees the
    same "assigned" affordance on either side. The border is drawn
    on top of Qt's selection painting so both states stay readable
    simultaneously."""

    BORDER_COLOR = QColor("#FFD400")
    BORDER_WIDTH = 2

    def paint(self, painter, option, index):
        super().paint(painter, option, index)
        if not bool(index.data(_ASSIGNED_ROLE)):
            return
        painter.save()
        try:
            pen = QPen(self.BORDER_COLOR, self.BORDER_WIDTH)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            inset = self.BORDER_WIDTH // 2 + 1
            painter.drawRect(option.rect.adjusted(inset, inset, -inset, -inset))
        finally:
            painter.restore()


class EmbeddingsPane(QFrame):
    """Grid of saved embeddings with Save/Refresh chrome.

    Signals:
        selection_activated(entries: list[(name, embedding)])
            User clicked a tile (plain / ctrl / shift) — caller treats
            the current selection as the active source for the next
            Found Face assignment. Carries every selected entry so
            multi-select can be merged downstream.
        save_current_requested()
            User clicked Save Current — caller resolves the active
            source embedding, prompts for a name (via save_with_name),
            and calls add_entry().
    """

    selection_activated = Signal(list)
    save_current_requested = Signal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setProperty("panelTier", "3")
        self._path: Optional[str] = None
        self._entries: list[tuple[str, np.ndarray]] = []
        # Names of embeddings that are the assigned source for the
        # currently-selected Found Face. Re-applied on every grid
        # rebuild so the tint survives reload / reorder / save.
        self._assigned_names: set[str] = set()

        outer = QVBoxLayout(self)
        outer.setContentsMargins(4, 4, 4, 4)
        outer.setSpacing(4)

        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(6)
        self._title = QLabel("Embeddings")
        self._title.setStyleSheet("font-weight: bold;")
        header.addWidget(self._title)
        self._count_label = QLabel("(0)")
        self._count_label.setStyleSheet("color: #B0B0B0;")
        header.addWidget(self._count_label)
        header.addStretch()
        self._save_btn = QPushButton("Save Current...")
        self._save_btn.clicked.connect(self.save_current_requested)
        self._refresh_btn = QPushButton("Refresh")
        self._refresh_btn.clicked.connect(self.reload_from_disk)
        header.addWidget(self._save_btn)
        header.addWidget(self._refresh_btn)
        outer.addLayout(header)

        self._list = _ReorderableListWidget()
        self._list.setViewMode(QListView.IconMode)
        self._list.setResizeMode(QListView.Adjust)
        self._list.setMovement(QListView.Static)
        self._list.setSpacing(3)
        self._list.setWordWrap(False)
        self._list.setUniformItemSizes(True)
        self._list.setFlow(QListView.LeftToRight)
        # ExtendedSelection enables ctrl-click (toggle membership) and
        # shift-click (range) just like the source-faces panel. The
        # host merges all selected embeddings via the MergeTextSel
        # mode when applying to a Found Face.
        self._list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        # Qt's QDrag machinery is disabled — _ReorderableListWidget drives
        # the reorder directly from mouse events so the grid reflows under
        # the cursor in real time instead of waiting for a drop.
        self._list.setDragEnabled(False)
        self._list.setAcceptDrops(False)
        self._list.setDragDropMode(QAbstractItemView.NoDragDrop)
        self._list.items_reordered.connect(self._on_items_reordered)
        self._list.setItemDelegate(_AssignedBorderDelegate(self._list))
        self._tile_size = QSize(96, 22)
        self._list.setContextMenuPolicy(Qt.CustomContextMenu)
        self._list.customContextMenuRequested.connect(self._on_context_menu)
        # Drive emission off itemClicked rather than itemSelectionChanged
        # so re-clicking an already-selected single item still re-applies
        # to the current Found Face. itemClicked fires after Qt has
        # resolved the modifier (ctrl/shift) into the new selection, so
        # selectedItems() reflects the final post-click state.
        self._list.itemClicked.connect(self._on_item_clicked)
        outer.addWidget(self._list, stretch=1)

        self.setMinimumHeight(120)

    # --- Public API -----------------------------------------------------------

    def set_source_path(self, path: Optional[str]) -> None:
        self._path = path
        self.reload_from_disk()

    def source_path(self) -> Optional[str]:
        return self._path

    def reload_from_disk(self) -> None:
        self._entries = parse_merged_embeddings(self._path or "")
        self._rebuild_grid()

    def add_entry(self, name: str, embedding: np.ndarray) -> None:
        """Append (or replace by name) and persist."""
        embedding = np.asarray(embedding, dtype=np.float32).ravel()
        kept = [(n, v) for (n, v) in self._entries if n != name]
        kept.append((name, embedding))
        self._entries = kept
        self._persist_and_rebuild()

    def delete_entry(self, name: str) -> None:
        self._entries = [(n, v) for (n, v) in self._entries if n != name]
        self._persist_and_rebuild()

    def rename_entry(self, old: str, new: str) -> None:
        if old == new or not new:
            return
        self._entries = [((new if n == old else n), v) for (n, v) in self._entries]
        self._persist_and_rebuild()

    def entry_names(self) -> list[str]:
        return [n for (n, _v) in self._entries]

    def cycle(self, direction: int) -> None:
        """Step selection by `direction` (positive=next, negative=prev),
        wrapping at the ends, and emit selection_activated so the same
        flow as a click runs. Replaces the current selection (wheel
        cycling is single-step semantically); use ctrl/shift+click on
        the list itself to build a multi-selection. No-op if there are
        no entries."""
        n = len(self._entries)
        if n == 0:
            return
        step = 1 if direction >= 0 else -1
        cur = self._list.currentRow()
        next_idx = (0 if step > 0 else n - 1) if cur < 0 else (cur + step) % n
        item = self._list.item(next_idx)
        if item is None:
            return
        # ClearAndSelect mirrors a plain (no-modifier) click — replaces
        # whatever multi-selection was there with just this row.
        self._list.setCurrentItem(item, QItemSelectionModel.ClearAndSelect)
        self._emit_current_selection()

    # --- Internals ------------------------------------------------------------

    def _persist(self) -> None:
        if not self._path:
            return
        try:
            write_merged_embeddings(self._path, self._entries)
        except OSError as e:
            QMessageBox.warning(self, "Embeddings", f"Failed to write {self._path}: {e}")

    def _persist_and_rebuild(self) -> None:
        self._persist()
        self._rebuild_grid()

    def _on_items_reordered(self) -> None:
        """Sync _entries with the list's new visual order after drag-drop,
        then persist. No grid rebuild — Qt already moved the QListWidgetItems
        into place; rebuilding would flash."""
        by_name = {n: v for (n, v) in self._entries}
        new_entries: list[tuple[str, np.ndarray]] = []
        for i in range(self._list.count()):
            item = self._list.item(i)
            if item is None:
                continue
            name = item.data(Qt.UserRole)
            vec = by_name.get(name)
            if vec is not None:
                new_entries.append((name, vec))
        # Preserve any entries that somehow fell out of the visible list —
        # shouldn't happen for InternalMove, but guards against a half-applied
        # reorder. Append in their original order so nothing is lost on disk.
        seen = {n for n, _ in new_entries}
        for n, v in self._entries:
            if n not in seen:
                new_entries.append((n, v))
        self._entries = new_entries
        self._persist()

    def _rebuild_grid(self) -> None:
        self._list.clear()
        for name, _vec in self._entries:
            item = QListWidgetItem(name)
            item.setData(Qt.UserRole, name)
            item.setSizeHint(self._tile_size)
            item.setTextAlignment(Qt.AlignCenter)
            self._list.addItem(item)
        self._count_label.setText(f"({len(self._entries)})")
        # Re-apply the assigned-source tint after a grid rebuild
        # (reload / reorder / add / delete). Names that no longer
        # exist fall off silently.
        if self._assigned_names:
            self._paint_assigned_items()

    def set_assigned_names(self, names) -> None:
        """Mark embeddings that are the assigned source for the
        currently-selected Found Face. Painted as a warm gold tint.
        Pass an empty iterable to clear."""
        self._assigned_names = set(names)
        self._paint_assigned_items()

    def _paint_assigned_items(self) -> None:
        assigned_brush = QBrush(_ASSIGNED_TINT)
        assigned_fg = QBrush(_ASSIGNED_FG)
        default_fg = QBrush(_DEFAULT_FG)
        clear_bg = QBrush()
        for i in range(self._list.count()):
            item = self._list.item(i)
            if item is None:
                continue
            name = item.data(Qt.UserRole)
            is_assigned = name in self._assigned_names
            if is_assigned:
                item.setBackground(assigned_brush)
                item.setForeground(assigned_fg)
            else:
                item.setBackground(clear_bg)
                item.setForeground(default_fg)
            # Role drives _AssignedBorderDelegate's gold-border overlay,
            # which paints on top of Qt's selection background so the
            # assigned signal stays visible even when the item is
            # also user-selected.
            item.setData(_ASSIGNED_ROLE, is_assigned)

    def _find_embedding(self, name: str) -> Optional[np.ndarray]:
        for n, v in self._entries:
            if n == name:
                return v
        return None

    def _on_item_clicked(self, _item: QListWidgetItem) -> None:
        # itemClicked fires *after* Qt has resolved the ctrl/shift
        # modifier into the new selection, so selectedItems() is the
        # final post-click selection — read from there rather than
        # the lone clicked item so multi-select works correctly.
        self._emit_current_selection()

    def _emit_current_selection(self) -> None:
        by_name = {n: v for (n, v) in self._entries}
        entries: list[tuple[str, np.ndarray]] = []
        for it in self._list.selectedItems():
            name = it.data(Qt.UserRole)
            vec = by_name.get(name)
            if vec is not None:
                entries.append((name, vec))
        if entries:
            self.selection_activated.emit(entries)

    def _on_context_menu(self, pos) -> None:
        item = self._list.itemAt(pos)
        if item is None:
            return
        name = item.data(Qt.UserRole)
        menu = QMenu(self)
        act_rename = menu.addAction("Rename...")
        act_delete = menu.addAction("Delete")
        action = menu.exec(self._list.viewport().mapToGlobal(pos))
        if action == act_rename:
            new, ok = QInputDialog.getText(
                self, "Rename Embedding", "New name:", text=name,
            )
            if ok and new and new != name:
                self.rename_entry(name, new)
        elif action == act_delete:
            if QMessageBox.question(
                self, "Delete Embedding",
                f"Delete '{name}'?",
            ) == QMessageBox.Yes:
                self.delete_entry(name)
