from PySide6.QtCore import QObject, Signal


class Bus(QObject):
    # GUI -> VideoManager
    load_target_video = Signal(str)
    load_target_image = Signal(str)
    get_requested_image = Signal()
    play_video = Signal(str)
    get_requested_video_frame = Signal(int)
    get_requested_video_frame_without_markers = Signal(int)
    target_faces = Signal(object)
    parameters_changed = Signal(dict)
    markers_changed = Signal(list)
    control_changed = Signal(dict)
    ui_vars_changed = Signal(dict)
    saved_video_path = Signal(str)
    vid_qual = Signal(int)
    set_stop = Signal(int)
    perf_test = Signal(bool)
    auto_swap = Signal()

    # VideoManager -> GUI
    frame_ready = Signal(object, bool)
    # Carries the frame number that just arrived from VM. Used by the
    # timeline widget to follow playback (move the playhead, update the
    # entry field). Emitted alongside frame_ready from the coordinator
    # tick so the timeline tracks both live playback and one-shot scrub
    # responses.
    playback_frame_changed = Signal(int)
    stop_play = Signal()
    slider_length_changed = Signal(int)
    vram_updated = Signal(float, float)  # (used_gb, total_gb)
    # Queue depth snapshot from the coordinator tick (frame_q, r_frame_q
    # lengths). Wired to the preview's perf HUD so a glance at the
    # overlay tells you whether the producer is starving the consumer
    # (queues stay 0) or the consumer is starving the producer (queues
    # grow). Throttled inside the coordinator — emitted ~10 Hz, not every
    # 1 kHz tick.
    queue_depths = Signal(int, int)
    # Fired once a preload_models() run finishes building sessions (both
    # Shared and Per-Thread paths), so the GUI can flip the "Preload
    # Models" button to its loaded state. Carries nothing — the GUI
    # re-checks Models.pipeline_sessions_loaded() to decide success, so a
    # partial/failed build reverts the button instead of falsely claiming
    # success. Emitted from a background thread; AutoConnection queues it
    # onto the GUI thread.
    models_preloaded = Signal()


bus = Bus()
