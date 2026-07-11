"""Live screen-capture worker for the Qt port.

Pooled pipeline (ported and extended from the original two-thread
rope/modules/CaptureViewfinder.py):

    capture_thread:  poll viewfinder.get_capture_bbox() -> dxcam/mss
                     grab -> tag with a monotonic seq -> stash as the
                     single freshest "pending" frame -> pump
    _pump:           if a pool slot is free, submit the pending frame's
                     swap to the VideoManager worker pool (vm._executor)
    _swap_task:      runs on a pool thread — vm.swap_video() (if SwapFaces
                     is on) -> publish through a monotonic ordering gate
                     -> vm._publish_frame(rgb, False, False)

Swaps run on the SAME ThreadPoolExecutor the video pacer uses, NOT on a
capture-owned thread pool. This is the whole point of the pooled design:
in Per-Thread model mode, ORT/TRT sessions live in `threading.local`, so
a separate capture thread pool would build its own second set of sessions
(a multi-second "reload" plus double the VRAM) the first time you enter
Capture. Sharing the pacer's threads means each physical worker keeps one
session bound to one CUDA stream, reused by whichever producer — decoder
pacer or capture grabber — is active (they are mutually exclusive by
preview mode). Switching Video<->Capture never reloads models.

Two invariants keep the pipeline honest:

  * Bounded in-flight — the grab thread holds only the single newest
    un-submitted frame (`_pending`); an older un-submitted frame is
    overwritten (freshest-wins). At most `_max_inflight` (== the pool's
    real thread count) swaps are submitted at once, so no backlog builds
    inside the executor and latency stays bounded to ~pool-size frames.
  * Ordering gate — pool threads finish out of order (variable swap
    time), so each publish is gated on a monotonic `_last_published_seq`.
    A task that finishes an older frame after a newer one has already
    been shown simply drops it, instead of flashing the preview backwards.

The pool size follows the VideoManager pool (driven by `ThreadsSlider`).
Capture reuses an existing pool as-is; it only builds one (at the current
Threads value) when none exists yet — e.g. the app opened straight into
Capture with no prior video playback.

Input side: any object with `get_capture_bbox() -> dict|None` (the Qt
CaptureViewfinder qualifies). Output side: calls vm._publish_frame —
the same push-based sink the pacer/scrub use — which marshals through
the Coordinator-installed callback to bus.frame_ready on the GUI thread.
"""

from __future__ import annotations

import threading
import time
from typing import Optional

import numpy as np


class _MssBackend:
    def __init__(self):
        import mss
        self.sct = mss.mss()

    def grab(self, bbox):
        raw = np.asarray(self.sct.grab(bbox))
        # BGRA -> RGB via reversed slice (faster than cv2.cvtColor).
        return np.ascontiguousarray(raw[..., 2::-1])

    def close(self):
        try:
            self.sct.close()
        except Exception:
            pass


class _DxcamBackend:
    """DXGI capture via bettercam (preferred) or dxcam (fallback)."""

    def __init__(self):
        try:
            import bettercam as _dxcam
            self._impl_name = "bettercam"
        except ImportError:
            import dxcam as _dxcam
            self._impl_name = "dxcam"
            print("[WindowCapture] WARNING: using legacy dxcam -- its "
                  "native color-conversion kernel has known GIL bugs. "
                  "Recommended: pip uninstall dxcam && pip install bettercam")
        cam = _dxcam.create(output_color="RGB")
        if cam is None:
            raise RuntimeError(f"{self._impl_name}.create() returned None")
        self.cam = cam
        self.rect = self._probe_rect(cam)
        try:
            cam.grab()  # warm-up to skip the first-grab stall
        except Exception:
            pass

    @staticmethod
    def _probe_rect(cam):
        for getter in (
            lambda c: (int(c.output.position[0]), int(c.output.position[1]),
                       int(c.output.position[0]) + int(c.output.resolution[0]),
                       int(c.output.position[1]) + int(c.output.resolution[1])),
            lambda c: (int(c.region[0]), int(c.region[1]),
                       int(c.region[2]), int(c.region[3])),
        ):
            try:
                return getter(cam)
            except Exception:
                continue
        return None

    def can_grab(self, bbox):
        if self.rect is None:
            return False
        bx1 = bbox["left"]; by1 = bbox["top"]
        bx2 = bx1 + bbox["width"]; by2 = by1 + bbox["height"]
        ox1, oy1, ox2, oy2 = self.rect
        return bx1 >= ox1 and by1 >= oy1 and bx2 <= ox2 and by2 <= oy2

    def grab(self, bbox):
        bx1 = bbox["left"]; by1 = bbox["top"]
        bx2 = bx1 + bbox["width"]; by2 = by1 + bbox["height"]
        return self.cam.grab(region=(bx1, by1, bx2, by2))

    def close(self):
        try:
            self.cam.release()
        except Exception:
            pass


class WindowCapture:
    """Capture pipeline: one grab thread feeding the shared VideoManager
    worker pool. Caller supplies a viewfinder (anything with
    `get_capture_bbox()`), a VideoManager (for swap + frame sink), and a
    parameters getter."""

    def __init__(self, viewfinder, vm, get_parameters):
        """
        Args:
            viewfinder:    object with .get_capture_bbox() returning the
                           mss-style dict
            vm:            VideoManager instance (uses vm.swap_video,
                           vm.control, vm._publish_frame, vm._executor)
            get_parameters: callable returning the live parameter dict
                            (CaptureFPSSlider, ThreadsSlider, ...) — read
                            each capture tick so slider changes take
                            effect without a restart. Swaps run on the
                            shared VideoManager pool, so its size follows
                            ThreadsSlider via the pacer, not this worker.
        """
        self.viewfinder = viewfinder
        self.vm = vm
        self.get_parameters = get_parameters
        self.running = False
        self.measured_fps = 0.0
        self.last_frame_size = (0, 0)

        # Single-slot "freshest pending" hand-off from the grab thread to
        # the VideoManager pool, plus an in-flight counter — both under
        # _sched_lock. The grab thread stores only the newest un-submitted
        # frame (older un-submitted frames are dropped, freshest-wins), and
        # at most _max_inflight swaps are submitted at once so no backlog
        # builds inside the executor. _max_inflight tracks the pool's real
        # thread count (see _sync_pool).
        self._sched_lock = threading.Lock()
        self._pending: Optional[tuple] = None   # (seq, rgb) | None
        self._inflight = 0
        self._max_inflight = 1
        self._seq = 0

        # Monotonic ordering gate: a task only publishes a frame whose
        # seq beats every frame already shown. Also carries the fps
        # counter so it's incremented exactly once per displayed frame.
        self._publish_lock = threading.Lock()
        self._last_published_seq = -1
        self._fps_count = 0
        self._fps_t0 = 0.0

        # Pause/resume gate. The grab thread stays alive while paused (it
        # parks on _resume_event), and the shared VideoManager pool — with
        # its warm per-thread sessions — is never torn down here, so a mode
        # switch back to Capture resumes instantly and never reloads models.
        # When inactive: no frame is grabbed, submitted, or published.
        self._active = False
        self._resume_event = threading.Event()

        self._capture_thread: Optional[threading.Thread] = None
        self._last_err_log_t: float = 0.0

    def _worker_count(self) -> int:
        """Desired pool size, from the Threads slider. Only used to size a
        pool we have to create from scratch; an existing pool is reused as
        its current size (see _sync_pool)."""
        try:
            params = self.get_parameters() or {}
            n = int(params.get("ThreadsSlider", 1))
        except Exception:
            n = 1
        return max(1, n)

    def _sync_pool(self) -> None:
        """Guarantee a VideoManager worker pool exists and align our
        in-flight cap to its real thread count. Reuses an existing pool
        (with its warm per-thread sessions) as-is rather than rebuilding
        it — only builds one, at the current Threads value, when none
        exists yet (app opened straight into Capture, no prior playback)."""
        vm = self.vm
        n = self._worker_count()
        if vm is not None and getattr(vm, "_executor", None) is None:
            ensure = getattr(vm, "_ensure_executor", None)
            if callable(ensure):
                try:
                    ensure(n)
                except Exception as exc:
                    print(f"[WindowCapture] _ensure_executor failed: {exc}")
        size = getattr(vm, "_executor_size", None) if vm is not None else None
        self._max_inflight = int(size) if size else n

    def start(self) -> None:
        if self.running:
            return
        self.running = True
        self._active = True
        self._resume_event.set()
        self._seq = 0
        self._last_published_seq = -1
        self._fps_count = 0
        self._fps_t0 = time.monotonic()
        with self._sched_lock:
            self._pending = None
            self._inflight = 0

        self._sync_pool()

        self._capture_thread = threading.Thread(
            target=self._capture_worker, name="capture-grab", daemon=True,
        )
        self._capture_thread.start()

    def stop(self) -> None:
        self.running = False
        self._active = False
        self._resume_event.set()  # wake the grab thread if it's parked
        with self._sched_lock:
            self._pending = None
        # Join only the grab thread. Swap work runs on the VideoManager
        # pool, which we do NOT own — leave it alive for video playback.
        # Any in-flight swap finishes on a pool thread and is dropped at
        # the ordering gate (which checks _active).
        t = self._capture_thread
        if t is not None:
            t.join(timeout=1.0)
        self._capture_thread = None

    def pause(self) -> None:
        """Stop grabbing/submitting/publishing but keep the grab thread —
        and the shared pool's warm sessions — alive. Clears the pending
        frame and flips inactive under _publish_lock so an in-flight swap
        that finishes after the pause is dropped at the ordering gate, not
        flashed onto the preview that has already switched to another mode."""
        with self._publish_lock:
            self._active = False
        self._resume_event.clear()
        with self._sched_lock:
            self._pending = None

    def resume(self) -> None:
        """Re-activate a paused worker without rebuilding the grab thread.
        If the worker was never started (or was fully stopped), start() it.
        Re-syncs to the pool in case playback changed it while paused."""
        if not self.running:
            self.start()
            return
        self._sync_pool()
        with self._publish_lock:
            self._active = True
        self._resume_event.set()

    def is_running(self) -> bool:
        return self.running

    def is_active(self) -> bool:
        return self.running and self._active

    # ---- worker loops -------------------------------------------------------

    def _make_backends(self):
        dx = None
        try:
            dx = _DxcamBackend()
            print(f"[WindowCapture] using {dx._impl_name} (DXGI).")
        except ImportError:
            pass
        except Exception as exc:
            print(f"[WindowCapture] dxcam/bettercam init failed ({exc}); "
                  "falling back to mss.")

        mss_backend = None
        try:
            mss_backend = _MssBackend()
        except ImportError:
            print("[WindowCapture] mss not installed; capture disabled.")
        except Exception as exc:
            print(f"[WindowCapture] mss init failed: {exc}")
        return dx, mss_backend

    def _capture_worker(self) -> None:
        dx, mss_backend = self._make_backends()
        if dx is None and mss_backend is None:
            self.running = False
            return

        try:
            while self.running:
                if not self._active:
                    # Paused (non-Capture mode): park until resumed instead
                    # of spinning. Cheap idle; threads + sessions stay warm.
                    self._resume_event.wait(timeout=0.2)
                    continue
                loop_t0 = time.monotonic()
                params = self.get_parameters() or {}
                target_fps = max(1.0, float(params.get("CaptureFPSSlider", 30)))
                target_interval = 1.0 / target_fps

                if self.viewfinder is None:
                    time.sleep(0.1)
                    continue
                bbox = self.viewfinder.get_capture_bbox()
                if bbox is None or bbox["width"] < 4 or bbox["height"] < 4:
                    time.sleep(0.05)
                    continue

                rgb = None
                try:
                    if dx is not None and dx.can_grab(bbox):
                        rgb = dx.grab(bbox)
                        if rgb is None:
                            elapsed = time.monotonic() - loop_t0
                            time.sleep(max(0.001, min(target_interval - elapsed, 0.005)))
                            continue
                    elif mss_backend is not None:
                        rgb = mss_backend.grab(bbox)
                    else:
                        time.sleep(0.05)
                        continue
                except Exception:
                    time.sleep(0.1)
                    continue

                self.last_frame_size = (rgb.shape[1], rgb.shape[0])

                # Stash as the single freshest pending frame (overwriting
                # any un-submitted older one) and try to submit it. The
                # pump enforces the in-flight cap, so a swap pool that
                # can't keep up shows fresh frames, not a growing backlog.
                with self._sched_lock:
                    self._seq += 1
                    self._pending = (self._seq, rgb)
                self._pump()

                elapsed = time.monotonic() - loop_t0
                remaining = target_interval - elapsed
                if remaining > 0:
                    time.sleep(remaining)
        finally:
            if dx is not None:
                dx.close()
            if mss_backend is not None:
                mss_backend.close()

    def _pump(self) -> None:
        """Submit the freshest pending frame to the VideoManager pool if a
        slot is free. Called from the grab thread (new frame arrived) and
        from a task's done-callback (a pool slot just freed)."""
        vm = self.vm
        if vm is None:
            return
        with self._sched_lock:
            if (not self._active or self._pending is None
                    or self._inflight >= self._max_inflight):
                return
            seq, rgb = self._pending
            self._pending = None
            self._inflight += 1
        executor = getattr(vm, "_executor", None)
        if executor is None:
            with self._sched_lock:
                self._inflight -= 1
            return
        try:
            fut = executor.submit(self._swap_task, seq, rgb)
        except RuntimeError:
            # Executor was shut down between the read and the submit —
            # release the reserved slot and drop this frame.
            with self._sched_lock:
                self._inflight -= 1
            return
        fut.add_done_callback(self._on_task_done)

    def _on_task_done(self, fut) -> None:
        # Runs on the pool thread that finished the task. Free the slot
        # and pump the freshest frame we're holding, if any.
        with self._sched_lock:
            self._inflight -= 1
        self._pump()

    def _swap_task(self, seq, rgb) -> None:
        # Runs on a VideoManager pool thread, so it reuses that thread's
        # warm per-thread ORT/TRT sessions and CUDA stream — the same ones
        # video playback uses. No session is built on a capture-only thread,
        # so switching Video<->Capture never reloads models.
        if rgb is None:
            return

        # vm.control may be [] or a dict depending on init timing.
        ctrl = getattr(self.vm, "control", None)
        swap_on = isinstance(ctrl, dict) and ctrl.get("SwapFacesButton")
        if swap_on:
            try:
                rgb = self.vm.swap_video(rgb, 0, False)
            except Exception as exc:
                now = time.monotonic()
                if now - self._last_err_log_t > 60.0:
                    print(f"[WindowCapture] swap_video failed: "
                          f"{type(exc).__name__}: {exc}")
                    self._last_err_log_t = now

        # Ordering gate: publish only if this frame is newer than the last
        # one shown. Pool threads finish out of order under load; this
        # keeps the preview moving strictly forward and drops the
        # stragglers. fps is counted here — once per displayed frame.
        publish = False
        with self._publish_lock:
            if self._active and seq > self._last_published_seq:
                self._last_published_seq = seq
                publish = True
                self._fps_count += 1
                now = time.monotonic()
                if now - self._fps_t0 >= 1.0:
                    self.measured_fps = self._fps_count / (now - self._fps_t0)
                    self._fps_count = 0
                    self._fps_t0 = now

        if publish and self.vm is not None:
            # Push through the canonical frame sink — same path the
            # pacer/scrub callbacks use. The coordinator's installed
            # callback emits bus.frame_ready, marshaling to the GUI
            # thread via Qt's AutoConnection. frame_no is meaningless
            # for live capture (no timeline) and requested=False.
            self.vm._publish_frame(rgb, False, False)
