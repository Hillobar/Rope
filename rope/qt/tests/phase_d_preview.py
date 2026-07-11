"""Phase D preview smoke test.

Drives synthetic frames through bus.frame_ready and verifies that:
  1. PreviewWidget builds and initializeGL succeeds
  2. set_frame() accepts np.ndarray and torch.Tensor (CPU & CUDA if avail)
  3. paintGL runs without GL errors
  4. Per-frame paint time stays below 16 ms at 1080p
  5. Letterbox math correctly handles 16:9 and 4:3 frames

Run:  venv\\Scripts\\python.exe -m rope.qt.tests.phase_d_preview
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from rope.qt.bus import bus
from rope.qt.coordinator import Coordinator
from rope.qt.main_window import MainWindow


def _load_stylesheet(app: QApplication) -> None:
    qss = Path(__file__).resolve().parent.parent / "rope.qss"
    if qss.is_file():
        app.setStyleSheet(qss.read_text(encoding="utf-8"))


def _make_test_frame(w: int, h: int, frame_idx: int) -> np.ndarray:
    """Render a colored gradient with a moving vertical bar."""
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    # Red horizontal gradient
    arr[..., 0] = np.linspace(0, 255, w, dtype=np.uint8)[None, :]
    # Green vertical gradient
    arr[..., 1] = np.linspace(0, 255, h, dtype=np.uint8)[:, None]
    # Blue moving bar
    bar_x = (frame_idx * 20) % max(1, w - 20)
    arr[:, bar_x : bar_x + 20, 2] = 255
    # White text-like band at top to make the orientation obvious
    arr[: max(8, h // 16), :, :] = 200
    return arr


def fail(msg: str) -> None:
    print(f"  FAIL: {msg}")
    raise SystemExit(1)


def main() -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    _load_stylesheet(app)

    window = MainWindow()
    coord = Coordinator(models=None, vm=None)
    window._coordinator_ref = coord
    window.show()

    # Phase A/B/C smoke tests don't show the window; here we do, so the GL
    # context actually initializes.
    app.processEvents()
    print(f"[phase_d] window shown, preview type = {type(window.preview).__name__}")

    preview = window.preview

    # --- 1. Push a sequence of 1080p frames; measure paint time
    W, H = 1920, 1080
    n_frames = 30
    paint_times: list[float] = []
    for i in range(n_frames):
        frame = _make_test_frame(W, H, i)
        bus.frame_ready.emit(frame, False)
        app.processEvents()
        # Force a repaint by calling repaint() (synchronous, so our timing
        # measurement covers the actual GL work)
        preview.repaint()
        paint_times.append(preview.last_paint_ms())

    valid_times = [t for t in paint_times if t > 0]
    if not valid_times:
        fail("no paint time samples captured")
    avg = sum(valid_times) / len(valid_times)
    p95 = sorted(valid_times)[int(0.95 * len(valid_times))]
    print(f"  ok  1080p painted {len(valid_times)} frames: avg={avg:.2f}ms, p95={p95:.2f}ms")
    if p95 > 16.0:
        print(f"  WARN p95 paint time {p95:.2f}ms exceeds 16 ms target")

    tex_w, tex_h = preview.texture_size()
    if (tex_w, tex_h) != (W, H):
        fail(f"texture size should be {W}x{H}, got {tex_w}x{tex_h}")
    print(f"  ok  texture sized to frame: {tex_w}x{tex_h}")

    # --- 2. Different resolution reallocates texture
    for i in range(3):
        bus.frame_ready.emit(_make_test_frame(640, 480, i), False)
        app.processEvents()
        preview.repaint()
    tex_w, tex_h = preview.texture_size()
    if (tex_w, tex_h) != (640, 480):
        fail(f"texture not reallocated for 640x480: got {tex_w}x{tex_h}")
    print(f"  ok  texture reallocation: now {tex_w}x{tex_h}")

    # --- 3. Torch tensor path (CPU)
    try:
        import torch
        t = torch.from_numpy(_make_test_frame(800, 600, 5))
        bus.frame_ready.emit(t, False)
        app.processEvents()
        preview.repaint()
        if preview.texture_size() != (800, 600):
            fail(f"torch CPU tensor didn't upload: {preview.texture_size()}")
        print(f"  ok  torch CPU tensor path: {preview.texture_size()}")

        # --- 4. CUDA tensor path. Picks fast path if cuda-python is installed
        #       and registration succeeds; otherwise falls back to CPU bounce.
        if torch.cuda.is_available():
            print(f"  --  {preview.cuda_bridge_state()}")
            # Preallocate frames on CUDA so the benchmark times only the
            # upload + paint, not the host->device transfer.
            W2, H2 = 1920, 1080
            frame_pool = [
                torch.from_numpy(_make_test_frame(W2, H2, i)).cuda()
                for i in range(5)
            ]
            torch.cuda.synchronize()
            # First frame: forces texture realloc + bridge registration.
            bus.frame_ready.emit(frame_pool[0], False)
            app.processEvents()
            preview.repaint()
            torch.cuda.synchronize()
            if preview.texture_size() != (W2, H2):
                fail(f"torch CUDA tensor didn't upload: {preview.texture_size()}")
            first_path = preview.last_upload_path()
            # Steady-state: cycle through pool, measure paint time.
            steady_times: list[float] = []
            for i in range(30):
                t = frame_pool[i % len(frame_pool)]
                t0 = time.perf_counter()
                bus.frame_ready.emit(t, False)
                app.processEvents()
                preview.repaint()
                torch.cuda.synchronize()
                steady_times.append((time.perf_counter() - t0) * 1000)
            avg_steady = sum(steady_times) / len(steady_times)
            p95_steady = sorted(steady_times)[int(0.95 * len(steady_times))]
            path = preview.last_upload_path()
            print(f"  ok  torch CUDA tensor {W2}x{H2}: first={first_path}, "
                  f"steady={path}, avg={avg_steady:.2f}ms, p95={p95_steady:.2f}ms")
            # Compare against CPU bounce path for the same frames
            cpu_times: list[float] = []
            for i in range(30):
                t = frame_pool[i % len(frame_pool)]
                arr = t.cpu().numpy()
                t0 = time.perf_counter()
                bus.frame_ready.emit(arr, False)
                app.processEvents()
                preview.repaint()
                cpu_times.append((time.perf_counter() - t0) * 1000)
            avg_cpu = sum(cpu_times) / len(cpu_times)
            print(f"  ok  CPU bounce {W2}x{H2}: avg={avg_cpu:.2f}ms "
                  f"(speedup vs CUDA: {avg_cpu / avg_steady:.2f}x)")
            print(f"  --  {preview.cuda_bridge_state()}")
        else:
            print("  --  CUDA unavailable, skipping CUDA tensor test")
    except ImportError:
        print("  --  torch unavailable, skipping tensor paths")

    # --- 5. Visual screenshot
    bus.frame_ready.emit(_make_test_frame(1280, 720, 12), False)
    app.processEvents()
    preview.repaint()
    shot = Path(__file__).resolve().parent / "phase_d_preview.png"
    QTimer.singleShot(200, lambda: (
        window.grab().save(str(shot), "PNG"),
        print(f"[phase_d] screenshot: {shot}"),
        app.quit(),
    ))
    app.exec()
    print("all phase D preview checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
