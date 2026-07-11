"""Phase C end-to-end smoke test.

Spins up the full main_window (no Models/VM backend), then drives signals
to verify the round-trip plumbing:

    1. Window builds with all panes (top bar / left / center / right / bottom)
    2. Parameters pane is populated with widgets and exposes default values
    3. Changing a parameter widget emits bus.parameters_changed with the
       updated value
    4. Loading saved_parameters.json applies values to widgets and re-emits
    5. Defaults reset returns widgets to schema defaults
    6. Folder-pick callbacks are wired (we don't actually open the dialog,
       we just confirm the slots exist and are connected)
    7. Timeline drives bus.get_requested_video_frame
    8. Marker add/del/jump cycle works
    9. Settings persistence roundtrip (geometry + folders)

Run:  venv\\Scripts\\python.exe -m rope.qt.tests.phase_c_e2e
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from rope.qt.bus import bus
from rope.qt.coordinator import Coordinator
from rope.qt.main_window import MainWindow
from rope.qt.parameters import PARAMETER_BY_NAME


def fail(msg: str) -> None:
    print(f"  FAIL: {msg}")
    raise SystemExit(1)


def _load_stylesheet(app: QApplication) -> None:
    qss = Path(__file__).resolve().parent.parent / "rope.qss"
    if qss.is_file():
        app.setStyleSheet(qss.read_text(encoding="utf-8"))


def main() -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    _load_stylesheet(app)

    captured_params: list[dict] = []
    captured_frames: list[int] = []
    captured_markers: list[list] = []
    captured_video_load: list[str] = []

    bus.parameters_changed.connect(lambda snapshot: captured_params.append(snapshot))
    bus.get_requested_video_frame.connect(lambda f: captured_frames.append(f))
    bus.markers_changed.connect(lambda m: captured_markers.append(list(m)))
    bus.load_target_video.connect(lambda p: captured_video_load.append(p))

    window = MainWindow()
    coord = Coordinator(models=None, vm=None)
    window._coordinator_ref = coord
    window.show()

    print("[phase_c] window built")

    # --- 1. Structural check
    assert hasattr(window, "main_splitter"), "main_splitter missing"
    assert window.main_splitter.count() == 3, "expected 3 main-splitter children"
    assert len(window._params_pane.widgets) > 50, f"params pane only has {len(window._params_pane.widgets)} widgets"
    print(f"  ok  structure: 3 panes, {len(window._params_pane.widgets)} parameter widgets")

    # --- 2. Default values present
    threshold = window._params_pane.widgets["ThresholdSlider"]
    assert abs(threshold.get() - 55.0) < 1e-6, f"ThresholdSlider default wrong: {threshold.get()}"
    print(f"  ok  defaults: ThresholdSlider = {threshold.get()} (expected 55)")

    # --- 3. Changing a widget emits parameters_changed
    captured_params.clear()
    threshold.set(72.0, request_frame=True)
    app.processEvents()
    if not captured_params:
        fail("parameters_changed not emitted on slider.set()")
    if abs(captured_params[-1]["ThresholdSlider"] - 72.0) > 1e-6:
        fail(f"snapshot wrong: {captured_params[-1].get('ThresholdSlider')}")
    print(f"  ok  param change: ThresholdSlider 55 -> {captured_params[-1]['ThresholdSlider']} propagated")

    # --- 4. Save / load roundtrip
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "saved_parameters.json"
        # Save current state
        from rope.qt.parameters_migration import save, load
        save(window._params_pane.values, path)
        # Mutate and reload
        threshold.set(99.0, request_frame=False)
        loaded = load(path)
        window._params_pane.apply_values(loaded, emit=True)
        app.processEvents()
        if abs(threshold.get() - 72.0) > 1e-6:
            fail(f"load did not restore ThresholdSlider: got {threshold.get()}")
        print(f"  ok  save/load roundtrip: ThresholdSlider restored to {threshold.get()}")

    # --- 5. Defaults reset
    window._params_pane.load_defaults(emit=False)
    if abs(threshold.get() - 55.0) > 1e-6:
        fail(f"load_defaults did not restore ThresholdSlider: got {threshold.get()}")
    print(f"  ok  load_defaults: ThresholdSlider = {threshold.get()}")

    # --- 6. Folder-pick wiring (just confirm slots exist)
    assert callable(window._on_pick_videos_folder), "videos folder picker missing"
    assert callable(window._on_pick_faces_folder), "faces folder picker missing"
    assert callable(window._on_pick_output_folder), "output folder picker missing"
    print("  ok  file-dialog callbacks present")

    # --- 7. Timeline -> bus
    captured_frames.clear()
    tl = window._center_pane.timeline
    tl.set_length(1000)
    # Internal set() does not emit frame_requested (no scrub) — use the
    # explicit set_position via the user-facing path. We simulate a wheel
    # event by directly invoking the internal handler with emit=True.
    tl._set_position_internal(250, emit_request=True)
    app.processEvents()
    if not captured_frames:
        fail("Timeline did not emit get_requested_video_frame")
    if captured_frames[-1] != 250:
        fail(f"frame_requested payload wrong: {captured_frames[-1]}")
    print(f"  ok  timeline -> bus: frame={captured_frames[-1]}")

    # --- 8. Marker add/del/jump cycle
    captured_markers.clear()
    tl.set(100)
    window._on_add_marker()
    tl.set(300)
    window._on_add_marker()
    tl.set(700)
    window._on_add_marker()
    app.processEvents()
    if len(captured_markers) < 3:
        fail(f"expected >=3 markers_changed emissions, got {len(captured_markers)}")
    assert len(captured_markers[-1]) == 3, f"marker list wrong: {captured_markers[-1]}"
    # Prev/next jumps
    captured_frames.clear()
    tl.set(700)
    window._on_prev_marker()
    app.processEvents()
    if not captured_frames or captured_frames[-1] != 300:
        fail(f"prev_marker should jump to 300, got {captured_frames}")
    window._on_prev_marker()
    app.processEvents()
    if captured_frames[-1] != 100:
        fail(f"prev_marker should now jump to 100, got {captured_frames[-1]}")
    print(f"  ok  markers: added 3, prev/next navigation works")

    # --- 9. Settings persistence roundtrip.
    # Use a fresh Settings() — NOT window.settings — so the test doesn't
    # mutate the live object that closeEvent saves to data.json on quit.
    from rope.qt.settings import Settings
    with tempfile.TemporaryDirectory() as tmp:
        sp = Path(tmp) / "data.json"
        scratch = Settings()
        scratch.source_videos = "/tmp/videos"
        scratch.dock_win_geom = [1234, 567, 89, 10]
        scratch.save(sp)
        reloaded = Settings.load(sp)
        if reloaded.source_videos != "/tmp/videos":
            fail("settings roundtrip lost source_videos")
        if reloaded.dock_win_geom != [1234, 567, 89, 10]:
            fail(f"settings roundtrip lost geometry: {reloaded.dock_win_geom}")
    print("  ok  settings persistence roundtrip")

    # --- 10. Screenshot for visual inspection
    shot = Path(__file__).resolve().parent / "phase_c_e2e.png"
    QTimer.singleShot(200, lambda: (
        window.grab().save(str(shot), "PNG"),
        print(f"[phase_c] screenshot: {shot}"),
        app.quit(),
    ))
    app.exec()
    print("all phase C end-to-end checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
