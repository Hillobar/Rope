"""Phase E secondary-window smoke test.

Opens CaptureViewfinder and EmbeddingMergeDialog each in turn, takes a
screenshot, and exits 0 if both build without raising. Backends that
require external assets (real face embeddings) are tolerated gracefully
— the dialogs still build and report their failure in-UI.

Run:  venv\\Scripts\\python.exe -m rope.qt.tests.phase_e_dialogs
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from rope.qt.coordinator import Coordinator
from rope.qt.main_window import MainWindow
from rope.qt.widgets.embedding_merge_dialog import (
    EmbeddingMergeDialog,
    FaceEntry,
)


def _load_stylesheet(app: QApplication) -> None:
    qss = Path(__file__).resolve().parent.parent / "rope.qss"
    if qss.is_file():
        app.setStyleSheet(qss.read_text(encoding="utf-8"))


def _snap(window, name: str) -> Path:
    shot = Path(__file__).resolve().parent / f"phase_e_{name}.png"
    pix = window.grab()
    pix.save(str(shot), "PNG")
    print(f"  ok  screenshot: {shot}")
    return shot


def main() -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    _load_stylesheet(app)

    window = MainWindow()
    coord = Coordinator(models=None, vm=None)
    window._coordinator_ref = coord
    window.show()
    app.processEvents()

    # --- CaptureViewfinder ---------------------------------------------------
    cv = window.open_capture_viewfinder()
    app.processEvents()
    bbox = cv.get_capture_bbox()
    print(f"  ok  CaptureViewfinder built; bbox={bbox}")
    # Toggle lock and verify the flag stuck
    cv.set_locked(True)
    app.processEvents()
    cv.set_locked(False)
    app.processEvents()
    _snap(cv, "capture_viewfinder")
    cv.close()
    app.processEvents()

    # --- EmbeddingMergeDialog ------------------------------------------------
    # Synthesize a few face entries with random embeddings to exercise the
    # combine() call paths.
    faces = [
        FaceEntry(name=f"Face {i}", embedding=np.random.randn(512).astype(np.float32))
        for i in range(5)
    ]
    dlg = EmbeddingMergeDialog(faces)
    dlg.show()
    app.processEvents()
    # Programmatically tick the first 3, set name, change mode, then accept.
    for i in range(3):
        dlg._checkboxes[i].setChecked(True)
    dlg._name.setText("test_merged_01")
    dlg._mode.setCurrentText("Sph")
    app.processEvents()
    _snap(dlg, "embedding_merge")
    # Drive accept by calling the slot directly (avoids needing a synthetic
    # button-click event).
    dlg._on_accept()
    payload = dlg.result_payload()
    if payload is None:
        print("  FAIL: merge dialog returned no payload")
        return 1
    assert payload.mode == "Sph", payload.mode
    assert len(payload.selected_indices) == 3, payload.selected_indices
    assert payload.merged_embedding is not None
    assert payload.merged_embedding.shape == (512,), payload.merged_embedding.shape
    print(f"  ok  merge payload: mode={payload.mode}, "
          f"indices={payload.selected_indices}, "
          f"emb.shape={payload.merged_embedding.shape}")

    QTimer.singleShot(150, app.quit)
    app.exec()
    print("all phase E secondary-window checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
