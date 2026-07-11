"""Frameless click-through capture viewfinder.

Qt replacement for rope/modules/CaptureViewfinder.py (Tk Toplevel with a
magenta chromakey). The Qt version is structurally cleaner:

- A frameless top-level QWidget with WA_TranslucentBackground. The whole
  window is genuinely transparent; we only paint the colored border in
  paintEvent. No chromakey color needed.

- Click-through is opt-in via a "Lock" toggle. When locked we set the
  WindowTransparentForInput flag, which makes mouse clicks fall through
  the entire window to whatever's underneath. While unlocked the user can
  drag and resize the viewfinder normally.

- get_capture_bbox() returns the screen-coord rect of the transparent
  inner area (the captured region), in the mss-style dict the existing
  capture backends (dxcam/bettercam/mss) consume unchanged.

The Tk version's WindowCapture worker (capture thread + swap thread) is
backend-agnostic and Tk-free below the viewfinder level — Phase E reuses
WindowCapture verbatim, only swapping the viewfinder factory.
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import QPoint, QRect, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QWidget


DEFAULT_BORDER_PX = 4
BORDER_COLOR = QColor("#0066FF")


class CaptureViewfinder(QWidget):
    """Frameless, transparent-center widget defining a screen-capture region.

    Public signals:
        closed     — user closed the window
        bbox_changed(left, top, w, h) — geometry changed (drag/resize)
    """

    closed = Signal()
    bbox_changed = Signal(int, int, int, int)

    def __init__(
        self,
        *,
        border_px: int = DEFAULT_BORDER_PX,
        always_on_top: bool = True,
        parent: QWidget | None = None,
    ):
        flags = Qt.Tool | Qt.FramelessWindowHint
        if always_on_top:
            flags |= Qt.WindowStaysOnTopHint
        super().__init__(parent, flags)

        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WA_ShowWithoutActivating, False)

        self._border_px = max(1, int(border_px))
        self._locked = False

        # Drag/resize state
        self._drag_origin: Optional[QPoint] = None
        self._resize_corner: Optional[str] = None  # "br" only for now
        self._drag_initial_geom: Optional[QRect] = None

        self.setMinimumSize(80, 80)
        self.setGeometry(200, 200, 640, 480)
        self.setMouseTracking(True)

    # ----- Public API --------------------------------------------------------

    def get_capture_bbox(self) -> dict | None:
        """mss-style dict for the inner (transparent) rect in screen coords."""
        if not self.isVisible():
            return None
        b = self._border_px
        geom = self.geometry()
        return {
            "left": int(geom.x() + b),
            "top": int(geom.y() + b),
            "width": max(1, int(geom.width() - 2 * b)),
            "height": max(1, int(geom.height() - 2 * b)),
        }

    def set_border_px(self, n: int) -> None:
        self._border_px = max(1, int(n))
        self.update()

    def set_always_on_top(self, on: bool) -> None:
        flags = self.windowFlags()
        if on:
            self.setWindowFlag(Qt.WindowStaysOnTopHint, True)
        else:
            self.setWindowFlag(Qt.WindowStaysOnTopHint, False)
        # Re-show is required for the flag change to take effect on Windows.
        if self.isVisible():
            self.show()

    def set_locked(self, locked: bool) -> None:
        """When locked: window becomes click-through and refuses drag/resize.

        On Windows this maps to the WindowTransparentForInput flag, which
        causes mouse events to pass straight through to whatever app sits
        underneath — exactly what's needed when capturing live UI.
        """
        self._locked = bool(locked)
        self.setWindowFlag(Qt.WindowTransparentForInput, self._locked)
        if self.isVisible():
            self.show()
        self.update()

    # ----- Painting ----------------------------------------------------------

    def paintEvent(self, _e):
        p = QPainter(self)
        rect = self.rect()
        pen = QPen(BORDER_COLOR if not self._locked else QColor("#FF6600"))
        pen.setWidth(self._border_px)
        p.setPen(pen)
        # Inset by half the pen so the stroke sits exactly on the widget edge
        b = self._border_px
        p.drawRect(rect.adjusted(b // 2, b // 2, -(b - b // 2), -(b - b // 2)))
        # Optional small drag/resize hint when not locked
        if not self._locked:
            corner = 14
            p.fillRect(
                rect.right() - corner, rect.bottom() - corner, corner, corner,
                BORDER_COLOR,
            )
        p.end()

    # ----- Drag / resize -----------------------------------------------------

    def _hit_resize_corner(self, pos: QPoint) -> bool:
        # Bottom-right 14px square = resize handle
        corner = 14
        return (
            pos.x() >= self.width() - corner
            and pos.y() >= self.height() - corner
        )

    def mousePressEvent(self, e):
        if self._locked or e.button() != Qt.LeftButton:
            return
        if self._hit_resize_corner(e.position().toPoint()):
            self._resize_corner = "br"
        else:
            self._resize_corner = None
        self._drag_origin = e.globalPosition().toPoint()
        self._drag_initial_geom = self.geometry()

    def mouseMoveEvent(self, e):
        if self._locked or self._drag_origin is None:
            return
        gpos = e.globalPosition().toPoint()
        dx = gpos.x() - self._drag_origin.x()
        dy = gpos.y() - self._drag_origin.y()
        g0 = self._drag_initial_geom
        if self._resize_corner == "br":
            new_w = max(self.minimumWidth(), g0.width() + dx)
            new_h = max(self.minimumHeight(), g0.height() + dy)
            self.setGeometry(g0.x(), g0.y(), new_w, new_h)
        else:
            self.setGeometry(g0.x() + dx, g0.y() + dy, g0.width(), g0.height())
        g = self.geometry()
        self.bbox_changed.emit(g.x(), g.y(), g.width(), g.height())

    def mouseReleaseEvent(self, _e):
        self._drag_origin = None
        self._resize_corner = None
        self._drag_initial_geom = None

    # ----- Close handling ----------------------------------------------------

    def closeEvent(self, event):
        self.closed.emit()
        super().closeEvent(event)
