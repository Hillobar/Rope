"""Left pane — folder buttons + QListWidget thumbnail lists.

Replaces ~200 lines of manual canvas grid layout in rope/GUI.py
(populate_target_videos / populate_source_faces). Qt's QListWidget in
icon-view mode handles thumbnail rendering, wrapping, scrollbars, and
selection natively.

Two stacked panels:
    Top:    Target media (videos + images)  — folder button + list
    Bottom: Source faces                     — folder button + list

Public signals (forwarded to bus by main_window):
    pick_videos_folder()          → user clicked LoadTVideos button
    pick_faces_folder()           → user clicked LoadSFaces button
    target_media_activated(path)  → user clicked a target media thumbnail
    source_face_activated(path)   → user clicked a source face thumbnail

For Phase C the lists are populated with file paths only (no thumbnail
decoding yet); thumbnails arrive in Phase E via MediaCache.
"""

from __future__ import annotations

import mimetypes
from pathlib import Path
from typing import Optional

import numpy as np
from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QIcon, QImage, QPen, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListView,
    QListWidget,
    QListWidgetItem,
    QSizePolicy,
    QStyledItemDelegate,
    QVBoxLayout,
    QWidget,
)

from rope.qt.parameters import PARAMETER_BY_NAME
from rope.qt.thumbnails import THUMB_HEIGHT, ThumbnailLoader
from rope.qt.widgets.button import IconButton
from rope.qt.widgets.text import Text


_ICON_SIZE = 96     # logical icon box
_GRID_SIZE = QSize(108, 122)  # cell on the icon-mode grid


def _rgb_to_icon(rgb: np.ndarray) -> QIcon:
    """Convert an HxWx3 uint8 RGB numpy thumbnail into a QIcon.

    Pads the result to a square ICON_SIZE x ICON_SIZE pixmap centered on
    a transparent background so portrait crops (e.g. 72x96 face images)
    don't end up with a narrow visual that under-fills the grid cell
    and pulls the selection rect to a sliver of the top.
    """
    if rgb is None or rgb.ndim != 3 or rgb.shape[2] != 3:
        return QIcon()
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    if not rgb.flags["C_CONTIGUOUS"]:
        rgb = np.ascontiguousarray(rgb)
    h, w = rgb.shape[:2]
    qimg = QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888).copy()
    src = QPixmap.fromImage(qimg)
    canvas = QPixmap(_ICON_SIZE, _ICON_SIZE)
    canvas.fill(Qt.transparent)
    from PySide6.QtGui import QPainter
    painter = QPainter(canvas)
    x = (_ICON_SIZE - src.width()) // 2
    y = (_ICON_SIZE - src.height()) // 2
    painter.drawPixmap(x, y, src)
    painter.end()
    return QIcon(canvas)


_VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v"}
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _is_media(path: Path) -> bool:
    ext = path.suffix.lower()
    if ext in _VIDEO_EXTS or ext in _IMAGE_EXTS:
        return True
    mime, _ = mimetypes.guess_type(str(path))
    return bool(mime and (mime.startswith("video/") or mime.startswith("image/")))


def _is_face_image(path: Path) -> bool:
    return path.suffix.lower() in _IMAGE_EXTS


def _tier(tier: int) -> QFrame:
    f = QFrame()
    f.setProperty("panelTier", str(tier))
    return f


_ITEM_QSS = (
    # Qt's default delegate paints :selected as a tinted overlay across
    # the full item rect (icon + text), so the highlight covers the whole
    # tile rather than just the label strip — which is what setBackground()
    # on a QListWidgetItem suffers from in icon mode.
    "QListWidget { background-color: #28282E; border: 0; }"
    "QListWidget::item { color: #D0D0D0; padding: 4px; }"
    "QListWidget::item:hover { background-color: #2A2A40; }"
    "QListWidget::item:selected { background-color: #3A3A50; color: #E5B854; }"
    "QListWidget::item:selected:!active { background-color: #3A3A50; color: #E5B854; }"
)


# Warm gold tint applied to items that are assigned-source for the
# currently-selected Found Face. Distinct from the cooler #3A3A50
# selection tint so both states are simultaneously readable when a
# user-selected item is also an assigned source.
_ASSIGNED_TINT = QColor("#4A3D1A")
_ASSIGNED_FG = QColor("#E5B854")
_DEFAULT_FG = QColor("#D0D0D0")

# Custom data role marking an item as an assigned source of the
# currently-selected Found Face. Read by _AssignedBorderDelegate.
_ASSIGNED_ROLE = Qt.UserRole + 100


class _AssignedBorderDelegate(QStyledItemDelegate):
    """Paints a gold border over items flagged via _ASSIGNED_ROLE.

    Stacked on top of Qt's default painting (selection background,
    icon, text) so the border survives Qt's selection styling — the
    user can see selection and assigned state simultaneously instead
    of selection overriding the gold background tint."""

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
            # Inset by half-pen so the stroke isn't clipped at the cell edge.
            inset = self.BORDER_WIDTH // 2 + 1
            painter.drawRect(option.rect.adjusted(inset, inset, -inset, -inset))
        finally:
            painter.restore()


def _make_icon_list() -> QListWidget:
    lw = QListWidget()
    lw.setViewMode(QListView.IconMode)
    lw.setResizeMode(QListView.Adjust)
    lw.setMovement(QListView.Static)
    lw.setIconSize(QSize(_ICON_SIZE, _ICON_SIZE))
    lw.setGridSize(_GRID_SIZE)
    lw.setWordWrap(True)
    lw.setSpacing(4)
    lw.setUniformItemSizes(True)
    lw.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    lw.setStyleSheet(_ITEM_QSS)
    # Owned by lw, no need to keep a separate reference.
    lw.setItemDelegate(_AssignedBorderDelegate(lw))
    return lw


def _placeholder_icon() -> QIcon:
    """Transparent ICON_SIZE x ICON_SIZE pixmap, used until the real
    thumbnail arrives from the background loader. Ensures every item
    has a full-size icon from the moment it's added — without it, items
    with no icon render as text-only and Qt's hit-test shrinks to the
    text rect, leaving the bulk of the cell unclickable.
    """
    pix = QPixmap(_ICON_SIZE, _ICON_SIZE)
    pix.fill(Qt.transparent)
    return QIcon(pix)


_PLACEHOLDER_ICON: Optional[QIcon] = None


def _get_placeholder_icon() -> QIcon:
    global _PLACEHOLDER_ICON
    if _PLACEHOLDER_ICON is None:
        _PLACEHOLDER_ICON = _placeholder_icon()
    return _PLACEHOLDER_ICON


class _ThumbListMixin:
    """Shared thumbnail-loading + selection-highlight plumbing for the
    target-media and source-face panels. The two panels diverge only in
    which file extensions count as valid and which thumbnail kind they
    request from ThumbnailLoader, so the rest lives here.
    """

    THUMB_KIND: str = "media"  # subclasses override to "face"

    def _init_thumbs(self) -> None:
        # path -> QListWidgetItem so the ready-thumbnail signal can find
        # its row in O(1) without scanning the list.
        self._items_by_path: dict[str, QListWidgetItem] = {}
        self._loader = ThumbnailLoader(parent=self)
        self._loader.ready.connect(self._on_thumb_ready)
        self._selected_paths: set[str] = set()
        # Paths whose source-face contributed to the currently-selected
        # Found Face's embedding. Stored separately from _selected_paths
        # so the user's pending selection isn't clobbered when they click
        # a Found Face. Re-applied after folder repopulate so the tint
        # survives a re-sort or refresh.
        self._assigned_paths: set[str] = set()

    def _file_predicate(self, entry: Path) -> bool:
        raise NotImplementedError

    def populate_from_folder(self, path: str) -> None:
        # Drop any in-flight thumbnails for the previous folder.
        self._loader.bump_generation()
        self.list.clear()
        self._items_by_path.clear()
        if not path or not Path(path).is_dir():
            return
        try:
            entries = sorted(Path(path).iterdir(), key=lambda p: p.name.lower())
        except OSError:
            return
        for entry in entries:
            if not entry.is_file() or not self._file_predicate(entry):
                continue
            item = QListWidgetItem(entry.name)
            full = str(entry)
            item.setData(Qt.UserRole, full)
            item.setToolTip(full)
            item.setTextAlignment(Qt.AlignHCenter | Qt.AlignTop)
            # Pin the item's hit/render rect to the full grid cell so a
            # click anywhere in the tile selects it — without this Qt
            # sizes the row to its content (text-only before the thumb
            # arrives, or a sub-96px-wide portrait once it does).
            item.setSizeHint(_GRID_SIZE)
            item.setIcon(_get_placeholder_icon())
            self.list.addItem(item)
            self._items_by_path[full] = item
            self._loader.request(full, self.THUMB_KIND)
        # Re-apply selection state for any paths still present.
        for p in list(self._selected_paths):
            if p in self._items_by_path:
                self._apply_selected_style(self._items_by_path[p], True)
            else:
                self._selected_paths.discard(p)
        # Re-apply the assigned-source tint after a folder repopulate.
        # Paths no longer in the new folder fall off silently.
        if self._assigned_paths:
            self._paint_assigned_items()

    def _on_thumb_ready(self, path: str, thumb, _gen: int) -> None:
        item = self._items_by_path.get(path)
        if item is None:
            return
        icon = _rgb_to_icon(thumb)
        if not icon.isNull():
            item.setIcon(icon)

    def set_selected_paths(self, paths) -> None:
        """Drive the list's *native* selection state from an external
        path set. Used when main_window programmatically updates the
        selection (e.g. to reflect a cached embedding's source path).
        Qt's selection delegate paints :selected across the entire item
        — icon + text — without us needing a custom delegate.
        """
        target = set(paths)
        block = self.list.blockSignals(True)
        try:
            for path, item in self._items_by_path.items():
                item.setSelected(path in target)
        finally:
            self.list.blockSignals(block)
        self._selected_paths = target

    def set_assigned_paths(self, paths) -> None:
        """Mark items whose source-face is part of the currently-selected
        Found Face's assigned embedding. Painted as a warm gold tint via
        setBackground; Qt's selection background still wins for selected
        items (per QSS specificity), so user-selected-and-assigned items
        render as selected and lose the tint until deselected. Pass an
        empty iterable to clear the tint."""
        self._assigned_paths = set(paths)
        self._paint_assigned_items()

    def _paint_assigned_items(self) -> None:
        assigned_brush = QBrush(_ASSIGNED_TINT)
        assigned_fg = QBrush(_ASSIGNED_FG)
        default_fg = QBrush(_DEFAULT_FG)
        # QListWidgetItem with no background set uses an invalid (empty)
        # QBrush — passing one back via setBackground returns the cell
        # to the QSS-controlled default. Same idea for foreground.
        clear_bg = QBrush()
        for path, item in self._items_by_path.items():
            is_assigned = path in self._assigned_paths
            if is_assigned:
                item.setBackground(assigned_brush)
                item.setForeground(assigned_fg)
            else:
                item.setBackground(clear_bg)
                item.setForeground(default_fg)
            # Role drives the delegate's gold-border overlay, which
            # paints on top of Qt's selection background — keeping
            # the assigned signal readable even when the item is
            # also user-selected.
            item.setData(_ASSIGNED_ROLE, is_assigned)

    def _shutdown_loader(self) -> None:
        self._loader.shutdown()


class TargetMediaPanel(QFrame, _ThumbListMixin):
    """Top-left: target media folder + thumbnail list."""

    pick_folder = Signal()
    activated = Signal(str)
    THUMB_KIND = "media"

    def __init__(self, folder_path: Optional[str] = None, parent: QWidget | None = None):
        super().__init__(parent)
        self.setProperty("panelTier", "3")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        self.folder_button = IconButton(
            PARAMETER_BY_NAME["LoadTVideos"], callback=lambda *_: self.pick_folder.emit()
        )
        # Left-justify the icon+label instead of Qt's default centering.
        self.folder_button.setStyleSheet("text-align: left; padding-left: 8px;")
        layout.addWidget(self.folder_button)

        self.folder_text = Text(text="", tier=3)
        self.folder_text.setStyleSheet("font-size: 8pt; color: #A0A0A0;")
        layout.addWidget(self.folder_text)

        self.list = _make_icon_list()
        self.list.itemClicked.connect(self._on_item_clicked)
        layout.addWidget(self.list, stretch=1)

        self._init_thumbs()

        if folder_path:
            self.set_folder(folder_path)

    def _file_predicate(self, entry: Path) -> bool:
        return _is_media(entry)

    def set_folder(self, path: str) -> None:
        self.folder_text.configure(text=path or "")
        self.populate_from_folder(path)

    def _on_item_clicked(self, item: QListWidgetItem) -> None:
        path = item.data(Qt.UserRole)
        if path:
            self.activated.emit(path)


class SourceFacesPanel(QFrame, _ThumbListMixin):
    """Bottom-left: source faces folder + thumbnail list.

    Uses Qt's native selection model (ExtendedSelection) so plain click
    replaces the selection, Ctrl-click toggles, Shift-click ranges —
    matching the multi-select behavior the previous manual ctrl-detect
    code was emulating. Selection state is broadcast via
    `selection_changed(list[str])`.
    """

    pick_folder = Signal()
    selection_changed = Signal(list)
    THUMB_KIND = "face"

    def __init__(self, folder_path: Optional[str] = None, parent: QWidget | None = None):
        super().__init__(parent)
        self.setProperty("panelTier", "3")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        self.folder_button = IconButton(
            PARAMETER_BY_NAME["LoadSFaces"], callback=lambda *_: self.pick_folder.emit()
        )
        # Left-justify the icon+label instead of Qt's default centering.
        self.folder_button.setStyleSheet("text-align: left; padding-left: 8px;")
        layout.addWidget(self.folder_button)

        self.folder_text = Text(text="", tier=3)
        self.folder_text.setStyleSheet("font-size: 8pt; color: #A0A0A0;")
        layout.addWidget(self.folder_text)

        self.list = _make_icon_list()
        self.list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.list.itemSelectionChanged.connect(self._on_selection_changed)
        layout.addWidget(self.list, stretch=1)

        self._init_thumbs()

        if folder_path:
            self.set_folder(folder_path)

    def _file_predicate(self, entry: Path) -> bool:
        return _is_face_image(entry)

    def set_folder(self, path: str) -> None:
        self.folder_text.configure(text=path or "")
        self.populate_from_folder(path)

    def _on_selection_changed(self) -> None:
        paths = [item.data(Qt.UserRole) for item in self.list.selectedItems()
                 if item.data(Qt.UserRole)]
        self._selected_paths = set(paths)
        self.selection_changed.emit(paths)
