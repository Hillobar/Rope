"""Phase A smoke test.

Spins up the Qt skeleton WITHOUT loading Models/VideoManager, runs the event
loop briefly, takes a screenshot of the main window, then exits 0. Catches
import-time and construction-time regressions before more expensive checks.

Run:  venv\Scripts\python.exe -m rope.qt.tests.smoke
"""

import sys
from pathlib import Path

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from rope.qt.bus import bus
from rope.qt.coordinator import Coordinator
from rope.qt.main_window import MainWindow


def _load_stylesheet(app: QApplication) -> None:
    qss_path = Path(__file__).resolve().parent.parent / "rope.qss"
    if qss_path.is_file():
        app.setStyleSheet(qss_path.read_text(encoding="utf-8"))


def main() -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    _load_stylesheet(app)

    window = MainWindow()
    coordinator = Coordinator(models=None, vm=None)
    window._coordinator = coordinator  # keep ref

    # Probe: emit a few bus signals to prove the wiring exists. We don't
    # need slots wired to actually do anything — just that emit() succeeds.
    bus.play_video.emit("play")
    bus.parameters_changed.emit({})
    bus.markers_changed.emit([])
    bus.frame_ready.emit(object(), False)
    bus.slider_length_changed.emit(123)
    bus.stop_play.emit()

    window.show()

    shot_path = Path(__file__).resolve().parent / "phase_a_skeleton.png"

    def _capture_and_quit():
        pixmap = window.grab()
        pixmap.save(str(shot_path), "PNG")
        print(f"[smoke] window size: {window.size().width()}x{window.size().height()}")
        preview = window.preview
        if hasattr(preview, "texture_size"):
            print(f"[smoke] preview texture: {preview.texture_size()}")
        else:
            print(f"[smoke] preview: {type(preview).__name__}")
        print(f"[smoke] splitter sizes: {window.main_splitter.sizes()}")
        print(f"[smoke] screenshot: {shot_path}")
        app.quit()

    QTimer.singleShot(400, _capture_and_quit)
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
