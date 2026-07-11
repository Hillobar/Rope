"""Background thumbnail decoder for the target-media and source-face lists.

Decoding happens off the UI thread (cv2.VideoCapture seek + read at 1080p
can hit 30-50 ms; cv2.imread on a high-res face image isn't free either)
and the results are delivered back to the main thread via Qt signals.

The disk cache from rope/MediaCache.py is consulted first — repeat runs
load thumbnails effectively for free since the .npz files are tiny and
the cv2 decode is skipped entirely.

A generation counter lets callers cancel a previous folder's pending
requests when the user picks a new folder; the worker still finishes
its in-flight decode (cancelling mid-decode would require platform-
specific hacks) but stale results are filtered before signalling.
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from typing import Literal, Optional

import cv2
import numpy as np
from PySide6.QtCore import QObject, Signal

from rope import MediaCache


THUMB_HEIGHT = 96  # px — matches QListWidget.iconSize


Kind = Literal["face", "media"]


def _decode_image_thumb(path: str) -> Optional[np.ndarray]:
    cached = MediaCache.load_face(path)
    if cached is not None:
        return cached[0]  # (thumb, embedding)
    bgr = cv2.imread(path)
    if bgr is None:
        return None
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return _fit(rgb, THUMB_HEIGHT)


def _decode_video_thumb(path: str) -> Optional[np.ndarray]:
    cached = MediaCache.load_media(path)
    if cached is not None:
        return cached
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        cap.release()
        return None
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        # Seek to ~10% in to avoid black-intro frames common in stock
        # clips. Fall back to frame 0 if the codec doesn't support seek.
        target = max(0, int(total * 0.1)) if total > 1 else 0
        cap.set(cv2.CAP_PROP_POS_FRAMES, target)
        ok, frame = cap.read()
        if not ok:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = cap.read()
        if not ok:
            return None
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        thumb = _fit(rgb, THUMB_HEIGHT)
        try:
            MediaCache.store_media(path, thumb)
        except Exception:
            pass
        return thumb
    finally:
        cap.release()


def _fit(rgb: np.ndarray, target_h: int) -> np.ndarray:
    h, w = rgb.shape[:2]
    if h == 0 or w == 0:
        return rgb
    new_h = target_h
    new_w = max(1, int(round(w * (target_h / h))))
    if (new_w, new_h) == (w, h):
        return rgb
    interp = cv2.INTER_AREA if new_h < h else cv2.INTER_LINEAR
    return cv2.resize(rgb, (new_w, new_h), interpolation=interp)


class ThumbnailLoader(QObject):
    """One loader per panel. Owns a small thread pool and emits `ready`
    with the decoded thumbnail. Stale-generation results are silently
    dropped so a fresh `set_generation(...)` on a folder switch wipes
    any in-flight requests' visible effects.
    """

    ready = Signal(str, object, int)  # (path, np.ndarray RGB, generation)

    def __init__(self, *, max_workers: int = 3, parent=None):
        super().__init__(parent)
        self._pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="thumb")
        self._generation: int = 0

    def bump_generation(self) -> int:
        self._generation += 1
        return self._generation

    def request(self, path: str, kind: Kind) -> None:
        gen = self._generation
        fut = self._pool.submit(self._decode, path, kind)
        fut.add_done_callback(lambda f, p=path, g=gen: self._emit(f, p, g))

    @staticmethod
    def _decode(path: str, kind: Kind) -> Optional[np.ndarray]:
        try:
            if kind == "face":
                return _decode_image_thumb(path)
            return _decode_video_thumb(path)
        except Exception:
            return None

    def _emit(self, fut: Future, path: str, gen: int) -> None:
        # Runs on the pool's worker thread. emit() with AutoConnection
        # marshals onto the main thread because `ready` was created on
        # the main thread (this QObject was constructed there).
        if gen != self._generation:
            return
        try:
            thumb = fut.result()
        except Exception:
            thumb = None
        if thumb is None:
            return
        self.ready.emit(path, thumb, gen)

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)
