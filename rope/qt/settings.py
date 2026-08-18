"""Persistent app settings — reads/writes data.json at the project root.

Same on-disk schema as the legacy Tk GUI (rope/GUI.py:112-118):

    {
        "source videos":          str | None,
        "source faces":           str | None,
        "saved videos":           str | None,
        "merged_embeddings_file": str | None,
        "dock_win_geom":          [width, height, x, y],
        "splitter_main_sizes":    [left, center, right],   # new in Qt port
        "splitter_left_sizes":    [videos, faces],         # new in Qt port
    }

New keys are additive — the Tk version ignores them, and the Qt version
tolerates their absence.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DATA_JSON = Path("data.json")


@dataclass
class Settings:
    source_videos: str | None = None
    source_faces: str | None = None
    saved_videos: str | None = None
    merged_embeddings_file: str | None = None
    # Models folder location. None = use the default './models' next to
    # the repo root (the legacy hardcoded path). Selectable from the
    # Settings tab so users can put weights on a separate drive without
    # symlinks.
    models_folder: str | None = None
    dock_win_geom: list[int] = field(default_factory=lambda: [1600, 950, 100, 80])
    splitter_main_sizes: list[int] = field(default_factory=lambda: [340, 880, 380])
    splitter_left_sizes: list[int] = field(default_factory=lambda: [500, 400])
    # Vertical split inside the center pane: [top half (video + chrome +
    # found-faces), embeddings]. Drag handle lives between Found Faces
    # and Embeddings.
    splitter_center_sizes: list[int] = field(default_factory=lambda: [700, 180])
    # Collapsed state for each parameters-pane section, keyed by title.
    # Missing entries default to expanded (False).
    params_collapsed: dict[str, bool] = field(default_factory=dict)
    # Per-model backend preference. Keys are Models attribute names
    # (e.g. "swapper_model", "retinaface_model"); values are "trt" or
    # "onnx". A missing entry means "auto" — prefer TRT if the engine
    # file is on disk, fall back to ONNX. Set via the Settings tab's
    # Backend toggle and applied at the next lazy model load.
    model_backends: dict[str, str] = field(default_factory=dict)
    # File dialog mode: on Linux, defaults to True (Qt built-in dialogs) to
    # prevent desktop-portal D-Bus hangs. On Windows/macOS, defaults to False (native dialogs).
    dont_use_native_dialogs: bool = field(
        default_factory=lambda: sys.platform.startswith("linux")
    )

    @classmethod
    def load(cls, path: Path | str = DATA_JSON) -> "Settings":
        p = Path(path)
        if not p.is_file():
            return cls()
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return cls()
        if not isinstance(raw, dict):
            return cls()
        pc_raw = raw.get("params_collapsed", {})
        params_collapsed = {
            str(k): bool(v) for k, v in pc_raw.items()
        } if isinstance(pc_raw, dict) else {}
        mb_raw = raw.get("model_backends", {})
        model_backends = {
            str(k): str(v) for k, v in mb_raw.items()
            if v in ("trt", "onnx")
        } if isinstance(mb_raw, dict) else {}
        raw_native = raw.get("dont_use_native_dialogs")
        dont_use_native_dialogs = (
            bool(raw_native)
            if raw_native is not None
            else sys.platform.startswith("linux")
        )
        return cls(
            source_videos=raw.get("source videos"),
            source_faces=raw.get("source faces"),
            saved_videos=raw.get("saved videos"),
            merged_embeddings_file=raw.get("merged_embeddings_file"),
            models_folder=raw.get("models_folder"),
            dock_win_geom=list(raw.get("dock_win_geom", [1600, 950, 100, 80])),
            splitter_main_sizes=list(raw.get("splitter_main_sizes", [340, 880, 380])),
            splitter_left_sizes=list(raw.get("splitter_left_sizes", [500, 400])),
            splitter_center_sizes=list(raw.get("splitter_center_sizes", [700, 180])),
            params_collapsed=params_collapsed,
            model_backends=model_backends,
            dont_use_native_dialogs=dont_use_native_dialogs,
        )

    def save(self, path: Path | str = DATA_JSON) -> None:
        out: dict[str, Any] = {
            "source videos": self.source_videos,
            "source faces": self.source_faces,
            "saved videos": self.saved_videos,
            "merged_embeddings_file": self.merged_embeddings_file,
            "models_folder": self.models_folder,
            "dock_win_geom": list(self.dock_win_geom),
            "splitter_main_sizes": list(self.splitter_main_sizes),
            "splitter_left_sizes": list(self.splitter_left_sizes),
            "splitter_center_sizes": list(self.splitter_center_sizes),
            "params_collapsed": dict(self.params_collapsed),
            "model_backends": dict(self.model_backends),
            "dont_use_native_dialogs": self.dont_use_native_dialogs,
        }
        Path(path).write_text(json.dumps(out, indent=2), encoding="utf-8")


def shorten_path(path: str | None, max_len: int = 28) -> str:
    """Mirror of rope/GUI.py:create_path_string — trim long paths for display."""
    if not path:
        return ""
    if len(path) <= max_len:
        return path
    import os
    last_folder = os.path.basename(os.path.normpath(path))
    if len(last_folder) > max_len:
        return path[:3] + "..." + path[-max_len + 6:]
    return path[: max_len - len(last_folder)] + ".../" + path[-len(last_folder):]
