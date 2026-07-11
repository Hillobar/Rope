# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Rope-Pearl is a Windows desktop face-swap application built on PyTorch + ONNX Runtime + TensorRT. The GUI is PySide6 (Qt); the legacy Tk GUI has been fully removed (`grep tkinter rope/` is clean). The LivePortrait portrait-animation feature (and its vendored `LP/` module) has been fully removed.

There is no `git` repository here (`git status` will fail) and no formal test framework — "tests" are runnable phase-screenshot scripts under [rope/qt/tests/](rope/qt/tests/). Git-aware workflows (`/review`, `/ultrareview`, branch/diff/blame operations) won't work until the directory is `git init`-ed.

## Running

The app expects a Windows-style venv at `venv\Scripts\`. All commands assume it exists.

| Action | Command |
|---|---|
| Launch app | `Rope.bat` (activates venv, runs `python Rope.py`) |
| Launch with line profiler | `Rope - lprof.bat` (uses `kernprof -l -v`; produces `*.lprof` files) |
| UI smoke test (no models loaded, ~400ms, screenshots window) | `venv\Scripts\python.exe -m rope.qt.tests.smoke` |
| Parameters schema test | `venv\Scripts\python.exe -m rope.qt.tests.test_parameters` |
| Other phase tests | `venv\Scripts\python.exe -m rope.qt.tests.phase_c_e2e` etc. |

The entry point is one line: [Rope.py](Rope.py) → `rope.qt.app.run()` at [rope/qt/app.py:26](rope/qt/app.py#L26). `run(skip_backend=True)` instantiates the window/Bus/Coordinator without loading any ONNX models — that's what the smoke test relies on.

## Architecture

### Three long-lived objects

[rope/qt/app.py](rope/qt/app.py) constructs three top-level objects and connects them:

1. **`Models`** ([rope/Models.py](rope/Models.py)) — owns every ONNX/PyTorch model. Lazy-loaded; assignment to model attributes flips `models.vram_dirty` so the coordinator can push VRAM updates without instrumenting each load site. `MODEL_INVENTORY` lists every file the app may load from the models folder, with a `required-for-basic-swap` flag used by the Settings tab.
2. **`VideoManager`** ([rope/VideoManager.py](rope/VideoManager.py)) — owns playback, the swap pipeline, scrub workers, and a pacer thread. Pushes finished frames to a callback installed by the coordinator (`vm.set_frame_callback`). Emits `bus.slider_length_changed` and `bus.stop_play` directly from its load and stop paths (it imports `rope.qt.bus`).
3. **`Coordinator`** ([rope/qt/coordinator.py](rope/qt/coordinator.py)) — Qt-side bridge. Connects Bus signals to `VideoManager` methods, drives `vm.process()` via a `QTimer(interval=0)` (~1 kHz idle pump), coalesces rapid scrub requests, polls `models.vram_dirty`, and emits queue-depth telemetry. No action-queue drain — VM signals the Bus itself.

### The Bus

[rope/qt/bus.py](rope/qt/bus.py) is a single `QObject` with all cross-component signals. Two directions:

- **GUI → VideoManager** — load/play/scrub/parameter/marker/control changes.
- **VideoManager → GUI** — `frame_ready`, `playback_frame_changed`, `stop_play`, `slider_length_changed`, `vram_updated`, `queue_depths`.

Connections rely on Qt `AutoConnection`: same-thread emits run directly, cross-thread emits (e.g. the VM pacer thread emitting `frame_ready`) queue onto the GUI thread automatically. There is no explicit `QueuedConnection` plumbing.

### Parameters system

The legacy Tk GUI kept all parameter metadata in a flat `DEFAULT_DATA` dict keyed with stringly-typed suffixes (`Name+"Amount"`, `Name+"State"`, `Name+"Mode"`, ...).

- [rope/qt/_default_data.py](rope/qt/_default_data.py) — the original dict, minus orphan keys (`Clearmem*`, `PerfTest*`, `ImgVid*`) that the Qt UI handles outside this schema.
- [rope/qt/parameters.py](rope/qt/parameters.py) — typed dataclasses (`SliderParam`, `SwitchParam`, `SelectParam`, `EntryParam`, `ButtonParam`) that reflect into `DEFAULT_DATA` via properties. Three scopes: `"control"`, `"parameter"`, `"merge"`.
- [rope/qt/parameters_migration.py](rope/qt/parameters_migration.py) — load/save for `saved_parameters.json`. On-disk format is unchanged from the Tk GUI: flat `{widget_name: value}`. Unknown keys are silently dropped.

Widgets ask params for `.default`, `.min`, `.max`, `.modes`, `.info_text` — they never parse suffixes themselves.

### Settings persistence

[rope/qt/settings.py](rope/qt/settings.py) reads/writes [data.json](data.json) at the project root. Schema is a superset of the Tk version — new Qt-specific keys (`splitter_main_sizes`, `splitter_left_sizes`) are additive, and absent keys are tolerated. When `models_folder` is persisted, [rope/qt/app.py:55-57](rope/qt/app.py#L55-L57) applies it before any model is loaded.

### Media pipeline

- [rope/MediaPlayer.py](rope/MediaPlayer.py) — queued decoder. PyAV demux + audio thread (sounddevice DAC clock is the master sync) + video thread that prefers GPU NVDEC (PyNvVideoCodec first — only Windows NVDEC wheel; torchcodec second — Linux only). Silent fall-back to PyAV CPU decode. Both shapes (CUDA `torch.Tensor` and `np.ndarray` HxWx3) are accepted downstream.
- [rope/MediaCache.py](rope/MediaCache.py) — disk-backed thumbnail cache at `cache/`. Keyed by SHA1 of absolute path; staleness via mtime. Source-face entries also cache the 512-d embedding so subsequent loads skip ONNX detect+recognize.
- [rope/TensorRTEngine.py](rope/TensorRTEngine.py) — now just the models-folder path constants + `set_models_folder()` rebinding (called by `Models.set_models_folder`) and `_patch_inswapper_for_dynamic_batch()`, reused by [rope/scripts/probe_inswapper_fp16.py](rope/scripts/probe_inswapper_fp16.py). The four main pipeline models (inswapper_128, inswapper_512, retinaface, arcface) all route through ORT's `TensorrtExecutionProvider` — see "ORT-TRT-EP uniform model path" below — so there's no hand-built engine path left in the runtime. (The old `TRTRunner` / `LivePortraitTRT` classes and the `grid_sample_3d` plugin path were removed with the LivePortrait feature.)

### ORT-TRT-EP uniform model path

The four pipeline models all build via the same factory pattern (`_create_<name>_session` in [Models.py](rope/Models.py)):

| Model | Factory | Profile | Notes |
|---|---|---|---|
| inswapper_128 | `_create_swapper_session` | dynamic batch (symbolic) | Needs `trt_layer_norm_fp32_fallback=True` because the ONNX uses primitive-op-decomposed normalization (`ReduceMean → Sub → Sqrt → Div`); ORT-TRT-EP exposes that flag, raw TRT API doesn't. Hand-built FP16 engine produced visibly broken output, so this is on ORT-TRT-EP for correctness, not speed. |
| inswapper_512 | `_create_swapper_512_session` | static (1, 3, 512, 512) | FP16. Batch=1; `run_swapper_512` is called once per face. **Backend retained but currently UI-unreachable** — the `512-Native` Swapper-Resolution mode and the `inswapper_512_level2.onnx` inventory row were removed, so nothing selects this path at runtime; the code is kept so re-adding the mode is a one-liner. |
| inswapper_256_phase1 | `_create_swapper_256_session` | dynamic input pinned to (1, 3, 256, 256) | **FP32 engine** (`trt_fp16_enable=False`). Native single-pass 256 swapper (`run_swapper_256` / `_native_256_pass`, once per face). **Backend retained but currently UI-unreachable** — the `256-Native` Swapper-Resolution mode and the `inswapper_256_phase1.onnx` inventory row were removed (mirroring inswapper_512), so nothing selects this path at runtime; the code is kept so re-adding the mode is a one-liner. **FP16 is not viable for this native-256 family** — it normalizes with decomposed `ReduceMean→Sub→Sqrt→Div` variance groups, and FP16 variance computes with catastrophic cancellation → a blurry/noisy face (the sibling `phase2` variant additionally had `InstanceNormalization` ops that defeated even `trt_layer_norm_fp32_fallback`, confirming FP32 is the safe path). The whole engine is built FP32 (the model ships FP32 weights, so it's native precision — TRT still beats CUDA-EP via fusion, at ~2× the engine VRAM of an FP16 build, notable in Per-Thread mode). I/O tensor names are `target256` (input) / `source` (input) / `p2_refined` (output) — all (1,3,256,256) except `source` (1,512). The ONNX declares a **dynamic** spatial input, so the TRT profile pins min=opt=max at 256 (only `target256` is dynamic; `source` is static). Consumes the same emap-projected latent as 128/512 — the model's `buff2fs` emap is byte-identical to inswapper_128's and, like 128/512, is an extracted-by-caller matrix (not applied inside the graph; `source` feeds the style `Gemm`s directly), so `calc_swapper_latent`'s default `latent_mode='emap'` is correct. |
| retinaface (det_10g) | `_create_retinaface_session` | dynamic input 320×320 → 640×640 | FP16. The `DetectInputSizeTextSel` parameter feeds the resize so the user can trade detection accuracy for speed at runtime. Configured via `trt_profile_{min,opt,max}_shapes`. |
| arcface (w600k_r50) | `_create_recognition_session` | static (1, 3, 112, 112) | FP16. Single call per face per frame. |

Common provider options: `trt_fp16_enable=True`, `trt_engine_cache_enable=True` (cache dir at `models/ort_trt_cache/`, must be an absolute path — ORT's TRT-EP concatenates relative paths with itself), `trt_builder_optimization_level=5`, `trt_dump_ep_context_model=True`. ORT manages engine build internally; first call per session is slow (~30-60s build), subsequent runs reload from the cache. The `_<model>_uses_trt` flag is set from `session.get_providers()[0] == 'TensorrtExecutionProvider'` after construction — ORT silently falls back to CUDA-EP if TRT-EP fails to initialize, and we want the flag to reflect reality.

**CUDA-EP (ONNX backend) tuning — shared across all four factories.** Two helpers on `Models` keep the CUDA-EP path (both the explicit-`onnx` branch and the TRT branch's CUDA fallback) tuned identically everywhere; previously each factory inlined its own inconsistent options and inswapper used a crippling `cudnn_conv_algo_search='DEFAULT'`:
- `_make_session_options()` — the fix for CUDA-EP being far slower than it should be in Per-Thread mode. ORT sizes each session's intra-op pool to the core count by default, so N workers × 3 models × core_count threads oversubscribe the CPU (CUDA-EP dispatches node-by-node and leans on that pool; TRT-EP fuses to one engine node and barely uses it). The helper sets `intra_op_num_threads=1` in Per-Thread mode (concurrency comes from the N worker threads) and a small bounded pool `max(2, min(8, cores//2))` in Shared mode (few sessions, N concurrent `Run()`s), plus `inter_op_num_threads=1` + explicit `ORT_SEQUENTIAL`.
- `_cuda_ep_provider_options(cudnn_algo=...)` — returns `arena_extend_strategy='kSameAsRequested'` + `cudnn_conv_algo_search` + the conditional `user_compute_stream` (same stream contract as the TRT branch). Algo policy: **`EXHAUSTIVE`** for the fixed-input-shape models (swapper 128², arcface 112², inswapper_512 512²) — best steady-state, one-time benchmark; **`HEURISTIC`** for retinaface, whose dynamic 320→640 profile would otherwise re-benchmark and stall on every `DetectInputSize` change (per session).

The four TRT provider lists now pass `('CUDAExecutionProvider', cuda_options)` instead of a bare `'CUDAExecutionProvider'` string, so a silent TRT→CUDA fallback keeps `user_compute_stream` (a bare entry dropped it, running ORT on the default stream while torch used the worker stream — a latent stream-ordering bug). `run_GFPGAN` and other non-factory sessions are out of this path and untouched.

### Per-thread sessions + user_compute_stream

[Models._get_model_session](rope/Models.py) is a generic dispatcher serving every ORT-TRT-EP model:

- **Shared mode** (`ModelSessionsTextSel='Shared'`, default): one session per model lives on `self.<model>_model`, handed to every worker. Lowest VRAM, but workers serialize inside ORT on each session's internal state.
- **Per-Thread mode** (`ModelSessionsTextSel='Per-Thread'`): each worker gets its own session per model via `threading.local`. The on-disk TRT engine cache is shared so engine build is paid once across all sessions; subsequent sessions deserialize the existing engine. Cost: ~150-300 MB extra VRAM per worker per model.

Per-Thread mode pairs with [VideoManager._get_worker_stream](rope/VideoManager.py): each VideoManager worker thread owns a `torch.cuda.Stream`, the per-frame `swap_video` body runs inside `with torch.cuda.stream(worker_stream)`, and `_create_*_session` reads `torch.cuda.current_stream()` at session-construction time and passes it to ORT as `user_compute_stream`. ORT enqueues its kernels onto the worker's stream → torch and ORT share the same queue → ordering is implicit → the `syncvec.cpu()` torch-stream drain is skipped (gated by `_should_drain_syncvec()`). Without per-thread streams, the `syncvec_drain` events line up across all workers on the global default stream and become the dominant bottleneck even after Per-Thread parallelizes the inswapper call (this was the lesson of the 2026-05-21 nsys trace).

A single `set_model_session_mode(mode)` setter switches all four models at once — wired from `ModelSessionsTextSel` via [main_window._on_params_changed](rope/qt/main_window.py). `clear_per_thread_sessions(name=None)` is called by `VideoManager._ensure_executor` whenever the thread pool rebuilds so dying workers' sessions release.

`Models.is_model_loaded(attr_name)` returns True if the shared attribute holds a session OR `_all_sessions[tls_name]` is non-empty — Per-Thread mode never writes the shared attribute, so the strong-ref list is the authoritative liveness signal there. The `attr_name → tls_name` mapping is `_ATTR_TO_TLS_NAME` on the class (e.g. `'swapper_model' → 'swapper'`).

### Worker pool + dispatch

[VideoManager._ensure_executor](rope/VideoManager.py) creates a `ThreadPoolExecutor(max_workers=ThreadsSlider)` and immediately submits `n_workers` pre-warm tasks that each block on a `threading.Barrier(n_workers, timeout=10s)`. Without the barrier, ThreadPoolExecutor lazy-spawns threads on demand and reuses any idle worker — so if a worker completed a task between submits, the pool would settle at fewer than `ThreadsSlider` threads. An nsys trace showed this directly: `ThreadsSlider=5` spawned only 4 `swap_N` threads. The barrier forces all N parties to coexist before any can return, guaranteeing the pool's thread count matches `ThreadsSlider`. The barrier wait happens inside the workers; `_ensure_executor` remains non-blocking. `n_workers == 1` skips the prewarm (a 1-party barrier releases instantly).

[VideoManager.process](rope/VideoManager.py) runs ~1kHz on the dedicated `vm-pacer` thread and **drains every currently-clear slot in a single tick** — no early `break` after a successful submit. Previously dispatched one frame per tick, which left workers idle 5-7ms per frame because workers finish in bursts. The decoder is the only rate limit: `get_next_frame(timeout=0)` returning None still breaks out (no point iterating further when the queue is dry).

Slots transition `'finished' → 'clear'` only when their frame has been presented to the GUI sink (audio-clock pacing in normal play, wall-clock fallback, or immediate in benchmark mode). So dispatch is gated by both decoder readiness AND presentation pacing — workers won't run ahead of display.

### NVTX trace marks (nsys)

`from rope._nvtx import nvtx_range` then `with nvtx_range("name"): ...`. Marks degrade to a near-zero-cost no-op when no profiler is attached. Two tiers nest naturally so nsys's hierarchical view shows the full tree:

**Outer (frame-level)** — `swap_video[f=N]` (whole per-frame compute, one per worker per frame), `swap_core` (per-target-face swap pipeline, called once per matched detection from `_swap_video_inner`).

**Inner (sub-stage)**
- `Models.detect_retinaface[s=N]` → `rf_preprocess` / `rf_ort_run` / `rf_postprocess`
- `Models.run_recognize` → `rec_preprocess` / `rec_ort_run`
- `Models.run_swapper` / `run_swapper_batched` → `ort_run_swapper` / `ort_run_swapper_batched[n=N]` / `syncvec_drain` (conditional on `_should_drain_syncvec()`)
- `VideoManager.swap_core` → `sc_input_warp` / `sc_latent` / `sc_polypass[k=K]` / `sc_hf_refine` / `sc_resize_to_pipeline` / `sc_color_match` / `sc_mask_compose` / `paste_back`
- `VideoManager._polyphase_pass_v1` → `poly_assemble[d=D]` / `poly_swap[d=D]` / `poly_interleave[d=D]`
- `VideoManager.apply_*` → `apply_occlusion` / `apply_dfl_xseg` / `apply_face_parser` / `apply_restorer`

The 2026-05-23 trace from these inner marks flagged two hotspots: `rf_postprocess` runs ~3× `rf_ort_run` (many small kernel launches + a final `.cpu().numpy()` device sync that serializes workers through the GPU queue), and `sc_mask_compose` runs ~10ms even with most gating switches off (dominated by two `_get_gaussian_blur` calls and the `v2.Resize` chain). The 2026-05-21 nsys trace earlier identified `syncvec_drain` as the dominant cross-worker stall — that's the lesson behind the per-thread streams + `user_compute_stream` machinery above.

### Inswapper ONNX I/O variants (auto-detected at load)

The `inswapper_128.fp16.onnx` file ships in multiple variants in the wild. `_create_swapper_session` detects two things at session-build time and adapts:

- **I/O dtype** (`_swapper_io_dtype`): `np.float16` if the model declares FP16 inputs, else `np.float32`. `run_swapper` / `run_swapper_batched` cast at the io_binding boundary — input via `.half()`, output via a temp FP16 buffer that `output.copy_(buf)` writes back into the caller's FP32 buffer.
- **Static vs dynamic batch** (`_swapper_batch_unsupported`): if `inputs[0].shape[0]` is an `int == 1`, the ONNX has a hardcoded batch=1 export and batched calls bypass the model entirely (per-call fallback in `run_swapper_batched`).

Two more model-specific contracts that bite if you ignore them:

- **Source binding is always `(1, 512)`** even when `target` is `(N, 3, 128, 128)`. The current export broadcasts the source identity across the target batch internally. `run_swapper_batched` accepts either `(1, 512)` or `(N, 512)` for `embedding_batch` and only ever reads row 0 — callers should pass `(1, 512)` to avoid wasted memory and a redundant device-to-device copy. `_polyphase_pass_v1` passes `latent` (which is `(1, 512)`) directly; the older code allocated an `emb_batch` scratch buffer and `copy_(latent.expand(n_phases, 512))` per frame, which the model never read.
- **emap lookup**: [Models.calc_swapper_latent](rope/Models.py) finds the 512×512 emap matrix by initializer name (`'buff2fs'`), with a fallback to "find the (512, 512) initializer." Previous code used `graph.initializer[-1]`, but the FP16 ONNX has a 32×32 `_ln_scale_32x32` *after* the emap. Look up by name — never by position.

### Per-frame face matching

[VideoManager._swap_video_inner](rope/VideoManager.py) matches each detected face in the current frame against `self.found_faces` (the user-assigned target faces). Semantics: **best-match per detected face**. For each detection, iterate all `found_faces`, pick the one with highest `findCosineDistance` score above `ThresholdSlider`, and call `swap_core` once with that slot's `AssignedEmbedding`. Skips `found_faces` without `SourceFaceAssignments` (target-only entries) or `Embedding=None`. A `try/except` around the distance calc means a malformed embedding on one slot doesn't abort the rest of the frame — important because the previous loop propagated the exception up and aborted ALL of the frame's pending swaps when only the first slot was bad.

This also fixes the prior "every match swaps" behavior: if a detected face cleared the threshold against multiple `found_faces`, the old loop called `swap_core` for each, with each pass overwriting the prior swap on the same kps using a different source — best-match collapses that to a single swap with the highest-sim source.

### Preview path

The preview ([rope/qt/widgets/preview.py](rope/qt/widgets/preview.py)) is a `QOpenGLWidget`. There is a zero-bounce torch-CUDA → GL texture fast path via `cudaGraphicsGLRegisterImage` ([rope/qt/cuda_gl_interop.py](rope/qt/cuda_gl_interop.py)). If `PyOpenGL` or `cuda-python` is missing, or registration fails, the path transparently falls back to a CPU bounce. Both packages are listed as required in `requirements.txt` to lock in the documented performance result, but the code does not depend on them being present.

### Qt UI layout

[rope/qt/main_window.py](rope/qt/main_window.py) composes three panes from [rope/qt/panes/](rope/qt/panes/):

- `left_pane.py` — source faces panel + target media panel.
- `center_pane.py` — preview, timeline, media buttons.
- `parameters_pane.py` — right-side parameter widgets, generated from the `PARAMETERS` list.

Widgets live in [rope/qt/widgets/](rope/qt/widgets/). The stylesheet is [rope/qt/rope.qss](rope/qt/rope.qss); panel tiering uses the `panelTier` dynamic property.

### Live screen capture

[rope/qt/widgets/capture_viewfinder.py](rope/qt/widgets/capture_viewfinder.py) is a frameless transparent window defining a screen region. [rope/qt/window_capture.py](rope/qt/window_capture.py) is a one-grab-thread pipeline feeding the **shared VideoManager worker pool**: the capture thread polls `viewfinder.get_capture_bbox()` and grabs via dxcam/bettercam (DXGI fast path) or mss (fallback), tags each frame with a monotonic seq, stashes it as the single freshest `_pending` frame, and calls `_pump()`; the swap itself (`_swap_task` → `vm.swap_video()` → `vm._publish_frame()`, the canonical sink shared with the pacer and scrub workers) is submitted to **`vm._executor`** — the same `ThreadPoolExecutor` the pacer dispatches video frames onto. `MainWindow.open_capture_viewfinder()` wires both up.

**Why the pool is shared (not a capture-owned thread set):** in Per-Thread model mode, ORT/TRT sessions live in `threading.local`, keyed by thread. A separate capture thread pool would build its *own* second set of sessions the first time you enter Capture — a multi-second deserialize "reload" (even though the engine cache is warm) plus roughly double the VRAM. Running swaps on the pacer's threads means each physical worker keeps one session bound to one CUDA stream (via `user_compute_stream`), reused by whichever producer — decoder pacer or capture grabber — is active. The two are mutually exclusive by preview mode (playback stops in Capture; the pacer's `process()` only submits when `self.play`), so they never contend for the pool. Sharing session + stream + thread is also what keeps `_should_drain_syncvec()` correct — borrowing another thread's session would break that stream-ordering invariant.

`_sync_pool()` (called from `start()`/`resume()`) reuses an existing pool as-is and only builds one — at the current `ThreadsSlider` — when none exists yet (app opened straight into Capture, no prior playback); it sets `_max_inflight` from the pool's real `_executor_size`. Two invariants keep the pipeline correct: the grab thread holds only the newest un-submitted frame (`_pending`, freshest-wins) and at most `_max_inflight` (== pool thread count) swaps are in flight, so no backlog builds inside the executor and latency stays bounded to ~pool-size frames; and a monotonic **publish gate** (`_last_published_seq` under `_publish_lock`) drops any frame a task finishes out of order after a newer one has already been shown — so the preview never flashes backward. The capture target fps comes from the `CaptureFPSSlider` control param (surfaced in the Settings tab's Capture section, read each grab tick via the params-pane value mirror).

**Mode-switch lifecycle** — `PreviewModeTextSel` (`Video` / `Testing` / `Capture`) is GUI-only; VideoManager never reads it. [main_window._on_preview_mode_changed](rope/qt/main_window.py) routes it: entering Capture calls `open_capture_viewfinder()` (creates the worker + viewfinder on first entry, else `worker.resume()`); leaving Capture calls `_exit_capture_mode()` → `worker.pause()` + `viewfinder.hide()` + `_refresh_current_frame()`. Because swaps run on the shared pool, switching Video↔Capture reloads no models in either direction — the warm per-thread sessions serve both. `pause()`/`resume()` keep the grab thread alive across switches and never touch the pool. `pause()` flips `_active=False` under `_publish_lock` and clears `_pending`, so an in-flight swap that finishes on a pool thread after the switch is dropped at the ordering gate instead of flashing a stale capture frame over the video preview; the parked grab thread waits on `_resume_event` (no busy-spin). Only the explicit viewfinder close (the X → `_on_capture_viewfinder_closed`) `stop()`s the grab thread; it never shuts down the pool (VideoManager owns that), and mode switches never stop anything.

### Settings tab — Models inventory + Backend toggle

The Settings tab's Required Models table ([rope/qt/panes/parameters_pane.py](rope/qt/panes/parameters_pane.py) — `_build_models_inventory`) renders a row per `MODEL_INVENTORY` entry with four columns:

| Column | Source |
|---|---|
| File | `MODEL_INVENTORY[i][0]`; trailing `*` when required, tooltip shows the `MODEL_INVENTORY[i][1]` role description |
| ONNX | filesystem check on `{models_folder}/{filename}` |
| Backend | toggle button: `TRT` / `ONNX`. Enabled for the ORT-TRT-EP-capable rows whenever their ONNX is present |
| Loaded | `Models.is_model_loaded(attr)` → `"✓ TRT"` / `"✓ ONNX"` / `—`. Honors Per-Thread mode by checking `_all_sessions[name]` in addition to the shared attribute (which Per-Thread leaves as `[]`). |

The Backend toggle calls `Models.set_backend_preference(attr, "trt"\|"onnx")` and persists the result in `Settings.model_backends` (a dict on `data.json`). `_create_*_session` consults `_backend_pref` on next load — `'onnx'` forces ORT's `CUDAExecutionProvider`, `'trt'` (or absent) requests `TensorrtExecutionProvider` and ORT falls back to CUDA-EP if TRT-EP init fails. The toggle renders for whichever ORT-TRT-EP-capable rows are in `MODEL_INVENTORY` — currently `swapper_model` (inswapper_128), `retinaface_model`, and `recognition_model`. (`swapper_512_model` and `swapper_256_model` are still backend-toggle-capable in `_build_models_inventory`, but their inventory rows were removed with the `512-Native` / `256-Native` modes, so no rows render for them.) Setting the preference auto-unloads the model so the next inference call reloads on the new backend.

### Probing FP16 sensitivity

[rope/scripts/probe_inswapper_fp16.py](rope/scripts/probe_inswapper_fp16.py) uses [polygraphy](https://docs.nvidia.com/deeplearning/tensorrt/polygraphy/docs/index.html) `debug precision` to bisect which inswapper layers need FP32 vs FP16. Was used during development to characterize the FP16 problem before settling on the ORT-TRT-EP backend. Two input modes:

- **Random uniform** (default): `--input-shapes`, deterministic via `--seed 42`. Fast but worst-case for FP16 sensitivity since random data exercises numerics evenly.
- **Real face inputs**: `--source-img test/b.jpg --target-img test/a.jpg` writes a polygraphy `--data-loader-script` that loads images, runs ArcFace (`models/w600k_r50.onnx`) → projects through emap → matches what the production swap pipeline feeds.

Other knobs: `--rtol/--atol` (default 0.01; loosen to 0.05 if convergence stalls), `--dir forward/reverse` (input-side vs output-side bisection), `--mode bisect/linear`. Diagnostic tool only; the runtime no longer consumes hand-built engines for any of the four main models.

## Conventions worth knowing

- **Native DLL bootstrap** — [rope/__init__.py](rope/__init__.py) calls `rope._native_dlls.ensure_native_dll_search_path()` at package-import time. This adds pip-installed DLL directories (currently just `tensorrt_libs/`, where `nvinfer_10.dll` lives) to the Windows DLL search path via `os.add_dll_directory()`. Without this, ORT's `TensorrtExecutionProvider` silently falls back to CUDA-EP with `RegisterTensorRTPluginsAsCustomOps Please install TensorRT libraries... in PATH`. Idempotent, no-op on non-Windows. To add another pip-distributed native package later, append its name to `_NATIVE_DLL_PACKAGES` in [rope/_native_dlls.py](rope/_native_dlls.py).
- **ONNX Runtime log spam** — [rope/qt/app.py:11](rope/qt/app.py#L11) sets `ORT_LOG_SEVERITY_LEVEL=3` before any import, with a belt-and-braces `set_default_logger_severity(3)` after import. If you add a top-level ORT import, keep both in place.
- **VideoManager seeding** — [rope/qt/coordinator.py:41-45](rope/qt/coordinator.py#L41-L45) defensively initializes `vm.control` and `vm.parameters` as dicts because `VideoManager.__init__` leaves them as `[]` and the first frame request would otherwise raise `TypeError: list indices must be integers`.
- **Saved-parameter late re-apply** — [rope/qt/app.py](rope/qt/app.py) calls `window._on_params_changed(dict(window._params_pane.values))` once after `window._coordinator = coordinator` is assigned. The initial `_load_saved_parameters()` in `MainWindow.__init__` runs *before* the coordinator is set, so the emit's `_on_params_changed` call sees `_get_models()` return None and skips the Models-side propagation (`set_model_session_mode`, `update_load_expectation`). The late re-call ensures saved values like `ModelSessionsTextSel="Per-Thread"` and `ThreadsSlider` actually take effect at startup. Wrapped in try/except so a failure here doesn't take down the whole launch.
- **Model preload (manual button)** — preload is **not** automatic at startup (so the app opens instantly and doesn't build engines the user may not want). The **"Preload Models"** button lives in the center-pane toggle row, immediately left of Enable Audio ([center_pane._build_toggle_row](rope/qt/panes/center_pane.py), emits `preload_pressed` → [main_window._on_preload_models](rope/qt/main_window.py)); clicking it re-applies the current params (so session mode + `ThreadsSlider` are current — `bus.parameters_changed` is a same-thread direct connection, so `vm.parameters` updates synchronously before the read) then calls `vm.preload_models()`. The button can be clicked again after changing the backend / detector / swapper / thread count to rebuild. **Load-state feedback:** on click the button shows "Preloading…" (disabled); when the background build finishes, VideoManager emits `bus.models_preloaded` (from `_preload_driver`'s `finally` in Per-Thread mode, or a single-submit `add_done_callback` in Shared/single mode) and `main_window._on_models_preloaded` re-checks `Models.pipeline_sessions_loaded(swapper_type, detect_mode)` — turning the button green **"Models Loaded"** when every needed session is live, or reverting to "Preload Models" on a partial/failed build. The button's three visual states are encapsulated in `CenterPane.set_preload_button_state('idle'|'loading'|'loaded')`. `VideoManager.preload_models` reads `ThreadsSlider`, ensures the worker pool, then builds the thread-scaled pipeline sessions **for the current selections** via `Models.preload_pipeline_sessions(swapper_type, detect_mode)` on each worker thread inside its CUDA stream (so sessions bind to the same stream `swap_video` uses). The preload set is selection-driven (`Models._pipeline_preload_set`): the recognizer (arcface `w600k_r50`) is always built; the detector is whichever `DetectTypeTextSel` picks (Retinaface `det_10g` via the per-thread ORT-TRT-EP getter, or the shared-singleton SCRDF `scrfd_2.5g_bnkps` via `_ensure_scrdf_model`, which `run_detect` also uses); the swapper is inswapper_128 for `SwapperTypeTextSel` in {128,256,512} (all UI options — they use inswapper_128 with polyphase tiling). The `_swapper_preload_for` map still routes `256-Native` → `inswapper_256_phase1`, but that mode is UI-unreachable, so in practice only inswapper_128 preloads. **Per-Thread mode uses a two-phase build** (`_preload_driver`): phase 1 builds one worker's set alone to pay the one-time TRT engine build and warm the on-disk cache, phase 2 then builds the remaining workers in parallel (fast deserialize) — this avoids N threads racing to build the same engine on a cold cache. Shared mode / single worker build once. It runs entirely on a background `vm-preload` thread + the worker pool, so the window stays interactive while engines build; it's guarded by `Models.swap_pipeline_files_present(swapper_type, detect_mode)` (skips cleanly when the selected models aren't in the folder) and wrapped in try/except. Only these thread-scaled models preload; feature-gated shared singletons (restorers, mask nets) still load on demand. Because capture and video share the pool ("Live screen capture" above), the preloaded sessions serve both. `preload_pipeline_sessions` also runs one throwaway swap (`run_swapper` on 128² zeros, or `run_swapper_256` on 256² zeros for native-256) after building — this absorbs the CUDA-EP `EXHAUSTIVE` cuDNN conv benchmark (see "CUDA-EP tuning" above) at load instead of stalling the first live swap frame on each worker; on the TRT backend it's a cheap warm run.
- **FFmpeg lookup** — [rope/VideoManager.py](rope/VideoManager.py) `_find_ffmpeg()` does its own PATH probe + common-install-locations + `imageio_ffmpeg` bundled binary, because `subprocess.Popen` on Windows doesn't search PATH the way the shell does.
- **TRT engine rebuild (main models)** — delete `models/ort_trt_cache/` to force ORT-TRT-EP to rebuild engines for the four pipeline models on next run. Required after a GPU or TensorRT version change. No "Build TensorRT" button anymore — first call after the cache is gone takes ~30-60s per session as ORT builds.
- **Inswapper FP16 is fundamentally unsafe (raw TRT)** — building inswapper_128 with `fp16=True` in the raw TRT API produces visibly blocky output. The instance-norm-style normalization is decomposed into primitive ops (`ReduceMean → Sub → Sqrt → Div`), so the standard `trt_layer_norm_fp32_fallback` flag has nothing to target. We sidestep this by routing inswapper through ORT's TRT-EP — which exposes `trt_layer_norm_fp32_fallback=True` as a provider option and handles the precision negotiation internally.
- **Profiling output** — `*.lprof` files are output of the `kernprof` run from `Rope - lprof.bat`. They are not source.
- **Outstanding bugs / backlog** — [rope/backlog](rope/backlog) is a plain-text scratchpad of known issues and TODOs, not a structured tracker.
