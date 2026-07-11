from PySide6.QtCore import QObject, QTimer, Qt, Slot

from rope.qt.bus import bus
from rope.qt.parameters import default_values, seed_control_dict


class Coordinator(QObject):
    """Replaces rope/Coordinator.py.

    - Connects Bus signals (emitted by widgets in later phases) to
      VideoManager methods.
    - Drives vm.process() via a QTimer(interval=0) — Qt's idle pump,
      equivalent to the old gui.after(1, coordinator) at ~1 kHz.
    - Coalesces rapid scrub frame requests.

    VideoManager emits bus.slider_length_changed / bus.stop_play directly
    from its load and stop paths, so the coordinator no longer needs an
    action-queue drain.
    """

    def __init__(self, models, vm, parent: QObject | None = None):
        super().__init__(parent)
        self.models = models
        self.vm = vm

        # VideoManager.__init__ initializes self.control as [] and self.parameters
        # similarly. The Tk GUI used to backfill these via 'control' / 'parameters'
        # actions right after launch — without that, the first lookup like
        # `self.control['SwapFacesButton']` in get_requested_video_frame raises
        # TypeError: list indices must be integers. Seed defensively here so the
        # backend is usable from the very first frame request, before MainWindow's
        # signal emissions get to run.
        if vm is not None:
            if not isinstance(getattr(vm, "control", None), dict):
                vm.control = seed_control_dict()
            if not isinstance(getattr(vm, "parameters", None), dict):
                vm.parameters = default_values(scope="parameter")

        self._connect_bus_to_vm()

        # Install the push-based frame sink. VideoManager's pacer thread
        # calls this whenever a frame is ready to present (play, scrub,
        # load preview). bus.frame_ready uses auto-connection so the
        # cross-thread emit from the pacer queues to the preview slot
        # on the GUI thread; the same-thread scrub callbacks go through
        # a DirectConnection (synchronous, no roundtrip).
        if self.vm is not None:
            self.vm.set_frame_callback(self._on_vm_frame)

        self._process_timer = QTimer(self)
        self._process_timer.setInterval(0)
        self._process_timer.timeout.connect(self._tick)
        self._process_timer.start()

        # VRAM updates are event-driven: Models.__setattr__ flips
        # models.vram_dirty whenever a model attribute is assigned
        # (load or unload). _tick polls that flag — no periodic timer
        # needed, no QLabel repaints between actual VRAM changes.

        # Throttle for queue-depth diagnostics — emit at most ~10 Hz so
        # the HUD has a steady sample rate without spamming the bus on
        # every 1 kHz tick.
        self._last_qdepth_emit: float = 0.0

    def _connect_bus_to_vm(self):
        if self.vm is None:
            return
        vm = self.vm
        # AutoConnection (the default) picks DirectConnection when the
        # emitter is on the same thread as the slot and QueuedConnection
        # otherwise. UI emitters live on the main thread (widget
        # callbacks); push-side frame delivery runs on vm._pacer_thread
        # and emits bus.frame_ready from there — Qt queues those across
        # threads automatically, so the preview slot still runs on the
        # GUI thread without explicit Queued plumbing.
        bus.load_target_video.connect(vm.load_target_video)
        bus.load_target_image.connect(vm.load_target_image)
        bus.play_video.connect(vm.play_video)
        bus.get_requested_video_frame.connect(self._on_scrub_with_markers)
        bus.get_requested_video_frame_without_markers.connect(self._on_scrub_no_markers)
        bus.target_faces.connect(vm.assign_found_faces)
        bus.parameters_changed.connect(self._on_parameters)
        bus.markers_changed.connect(self._on_markers)
        bus.control_changed.connect(self._on_control)
        bus.ui_vars_changed.connect(self._on_ui_vars)
        bus.saved_video_path.connect(self._on_saved_video_path)
        bus.vid_qual.connect(self._on_vid_qual)
        bus.set_stop.connect(self._on_set_stop)
        bus.perf_test.connect(self._on_perf_test)

    @Slot(int)
    def _on_scrub_with_markers(self, frame: int):
        self._handle_scrub(frame, with_markers=True)

    @Slot(int)
    def _on_scrub_no_markers(self, frame: int):
        self._handle_scrub(frame, with_markers=False)

    def _handle_scrub(self, frame: int, *, with_markers: bool) -> None:
        if self.vm is None:
            return
        # Stop playback up front so the pacer thread isn't running the
        # swap pipeline concurrently with the scrub worker. play_video
        # mutates VM state and is safer when called from the main
        # thread; submit_scrub afterward does the actual work async.
        if getattr(self.vm, 'play', False):
            try:
                self.vm.play_video("stop")
            except Exception:
                pass
        # Every event submits — the VM-side bounded scrub queue
        # (maxlen=5) handles overflow by dropping the oldest pending
        # request. No GUI-side coalescing.
        self.vm.submit_scrub(frame, marker=with_markers)

    @Slot(dict)
    def _on_parameters(self, params: dict):
        if self.vm is None:
            return
        self.vm.parameters = params

    @Slot(list)
    def _on_markers(self, markers: list):
        if self.vm is not None:
            self.vm.markers = markers

    @Slot(dict)
    def _on_control(self, control: dict):
        if self.vm is not None:
            self.vm.control = control

    @Slot(dict)
    def _on_ui_vars(self, ui_data: dict):
        if self.vm is not None:
            self.vm.ui_data = ui_data

    @Slot(str)
    def _on_saved_video_path(self, path: str):
        if self.vm is not None:
            self.vm.saved_video_path = path

    @Slot(int)
    def _on_vid_qual(self, quality: int):
        if self.vm is not None:
            self.vm.vid_qual = int(quality)

    @Slot(int)
    def _on_set_stop(self, value: int):
        if self.vm is not None:
            self.vm.stop_marker = value

    @Slot(bool)
    def _on_perf_test(self, enabled: bool):
        if self.vm is not None:
            self.vm.perf_test = enabled

    @Slot()
    def _tick(self):
        vm = self.vm
        if vm is None:
            return

        # Frame delivery is push, not pull — vm._pacer_thread calls
        # self._on_vm_frame which emits bus.frame_ready directly.
        # Control-plane signals (slider_length_changed, stop_play) are
        # emitted by VideoManager directly via the bus, so the tick only
        # handles queue-depth + VRAM telemetry.

        # Queue-depth telemetry kept for HUD compatibility; the legacy
        # frame_q is always 0 now (frames flow via callback), but
        # get_frame_length/get_requested_frame_length still report the
        # underlying list lengths so anyone watching them in the wild
        # gets a coherent reading.
        import time as _time
        now = _time.perf_counter()
        if now - self._last_qdepth_emit >= 0.1:
            self._last_qdepth_emit = now
            bus.queue_depths.emit(vm.get_frame_length(), vm.get_requested_frame_length())

        self._maybe_emit_vram()

    def _on_vm_frame(self, image, frame_no, requested):
        """VideoManager pacer/scrub callback. Cross-thread emits go
        through Qt.QueuedConnection automatically; same-thread emits
        run synchronously. Either way the preview slot ends up running
        on the GUI thread."""
        bus.frame_ready.emit(image, bool(requested))
        if frame_no is not None and frame_no is not False:
            try:
                bus.playback_frame_changed.emit(int(frame_no))
            except (TypeError, ValueError):
                pass

    def _maybe_emit_vram(self):
        """Emit vram_updated if Models flagged a VRAM change since the
        last emit. Called from _tick — cheap bool read on the common
        path (no model load), one cudaMemGetInfo + emit when dirty."""
        models = self.models
        if models is None or not getattr(models, 'vram_dirty', False):
            return
        # Clear before query: if a worker thread sets the flag again
        # between clear and emit, the value we report still reflects
        # current state (cudaMemGetInfo sees all allocations), and the
        # re-set will trigger another emit on the next tick.
        models.vram_dirty = False
        try:
            used, total = models.get_gpu_memory()
        except Exception:
            return
        bus.vram_updated.emit(float(used), float(total))
