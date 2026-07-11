import os
import cv2
import shutil
from PIL import Image
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import time
import numpy as np
from skimage import transform as trans
import subprocess
from math import floor, ceil
import bisect

from rope.qt.bus import bus
from rope._nvtx import nvtx_range


_FFMPEG_PATH_CACHE = None


def _find_ffmpeg():
    """Locate ffmpeg.exe robustly. subprocess.Popen on Windows does NOT
    do its own PATH search the same way the shell does — it inherits the
    parent process's environment and looks for the literal name. If the
    user launched the app from a context that doesn't have ffmpeg on
    PATH (desktop shortcut, IDE-launched debugger, conda env, etc.), the
    bare 'ffmpeg' arg fails with FileNotFoundError [WinError 2].

    Try in order:
      1. shutil.which (matches what the shell would find on PATH).
      2. Common Windows install locations.
      3. imageio_ffmpeg's bundled binary, if that package is installed.
    Returns the absolute path or None. Cached after first success."""
    global _FFMPEG_PATH_CACHE
    if _FFMPEG_PATH_CACHE is not None:
        return _FFMPEG_PATH_CACHE
    p = shutil.which('ffmpeg')
    if p:
        _FFMPEG_PATH_CACHE = p
        return p
    candidates = [
        r'C:\ffmpeg\bin\ffmpeg.exe',
        r'C:\ffmpeg\ffmpeg-6.0-full_build-shared\bin\ffmpeg.exe',
        r'C:\ffmpeg\ffmpeg-7.0-full_build-shared\bin\ffmpeg.exe',
        r'C:\Program Files\ffmpeg\bin\ffmpeg.exe',
    ]
    for c in candidates:
        if os.path.exists(c):
            _FFMPEG_PATH_CACHE = c
            return c
    try:
        import imageio_ffmpeg
        p = imageio_ffmpeg.get_ffmpeg_exe()
        if p and os.path.exists(p):
            _FFMPEG_PATH_CACHE = p
            return p
    except Exception:
        pass
    return None
import onnxruntime
import torchvision
from torchvision.transforms.functional import normalize #update to v2
import torch
from torchvision import transforms
torchvision.disable_beta_transforms_warning()
from torchvision.transforms import v2
torch.set_grad_enabled(False)
onnxruntime.set_default_logger_severity(1)

import inspect #print(inspect.currentframe().f_back.f_code.co_name, 'resize_image')

from rope.MediaPlayer import MediaPlayer

device = 'cuda'


def _rgb_chw_to_lab(rgb: torch.Tensor) -> torch.Tensor:
    """Convert a (3, H, W) RGB tensor (values nominally in [0, 255]) to a
    (3, H, W) float32 CIE 1976 L*a*b* tensor on the same device. sRGB
    primaries, D65 white point. Equivalent to cv2's RGB->LAB up to the
    8-bit scaling cv2 applies for uint8 output (which we don't need —
    we want raw LAB for the mean/std transfer).
    """
    rgb = rgb.to(torch.float32).clamp_(0, 255) / 255.0
    # sRGB -> linear (piecewise gamma)
    rgb = torch.where(rgb > 0.04045, ((rgb + 0.055) / 1.055).pow(2.4), rgb / 12.92)
    R, G, B = rgb[0], rgb[1], rgb[2]
    # linear RGB -> CIE XYZ (sRGB matrix), normalize by D65 white point
    X = (0.4124564 * R + 0.3575761 * G + 0.1804375 * B) / 0.95047
    Y = 0.2126729 * R + 0.7151522 * G + 0.0721750 * B            # Yn = 1
    Z = (0.0193339 * R + 0.1191920 * G + 0.9503041 * B) / 1.08883
    # XYZ -> L*a*b*
    EPS = 216.0 / 24389.0                                         # (6/29)**3
    KAPPA = 24389.0 / 27.0
    def _f(t):
        return torch.where(
            t > EPS,
            t.clamp(min=1e-12).pow(1.0 / 3.0),
            (KAPPA * t + 16) / 116,
        )
    fx, fy, fz = _f(X), _f(Y), _f(Z)
    L = 116 * fy - 16
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)
    return torch.stack([L, a, b], dim=0)


def _lab_to_rgb_chw_uint8(lab: torch.Tensor) -> torch.Tensor:
    """Inverse of _rgb_chw_to_lab. Returns (3, H, W) uint8 RGB clamped
    to [0, 255]. Out-of-gamut LAB is clamped after the linear-RGB step
    so the sRGB encode stays in real-valued territory."""
    L, a, b = lab[0], lab[1], lab[2]
    fy = (L + 16) / 116
    fx = fy + a / 500
    fz = fy - b / 200
    DELTA = 6.0 / 29.0                                            # EPS**(1/3)
    KAPPA = 24389.0 / 27.0
    def _inv_f(ft):
        return torch.where(ft > DELTA, ft.pow(3), (116 * ft - 16) / KAPPA)
    X = _inv_f(fx) * 0.95047
    Y = _inv_f(fy)
    Z = _inv_f(fz) * 1.08883
    # XYZ -> linear RGB (inverse sRGB matrix)
    R = 3.2404542 * X - 1.5371385 * Y - 0.4985314 * Z
    G = -0.9692660 * X + 1.8760108 * Y + 0.0415560 * Z
    Bc = 0.0556434 * X - 0.2040259 * Y + 1.0572252 * Z
    R = R.clamp_(0, 1); G = G.clamp_(0, 1); Bc = Bc.clamp_(0, 1)
    # linear -> sRGB
    def _to_srgb(x):
        return torch.where(
            x > 0.0031308,
            1.055 * x.clamp(min=1e-12).pow(1.0 / 2.4) - 0.055,
            12.92 * x,
        )
    rgb = torch.stack([_to_srgb(R), _to_srgb(G), _to_srgb(Bc)], dim=0)
    return (rgb.clamp_(0, 1) * 255).to(torch.uint8)

class VideoManager():  
    def __init__(self, models ):
        self.models = models
        # Model related
        self.swapper_model = []             # insightface swapper model
        # self.faceapp_model = []             # insight faceapp model
        self.input_names = []               # names of the inswapper.onnx inputs
        self.input_size = []                # size of the inswapper.onnx inputs

        self.output_names = []              # names of the inswapper.onnx outputs    
        self.arcface_dst = np.array( [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float32) #112
        # self.arcface_dst = self.arcface_dst * 128.0/112.0
        self.arcface_scale = 128.0/np.linalg.norm(self.arcface_dst[0]-self.arcface_dst[1])

        self.video_file = []

        self.FFHQ_kps = np.array([[ 192.98138, 239.94708 ], [ 318.90277, 240.1936 ], [ 256.63416, 314.01935 ], [ 201.26117, 371.41043 ], [ 313.08905, 371.15118 ] ])
        self.FFHQ_kps /= 4.0
     
        
        #Video related
        self.capture = []                   # cv2 video (legacy; only used by load_target_image)
        self.player = None                  # MediaPlayer for current video file
        self.is_video_loaded = False        # flag for video loaded state
        self.video_frame_total = None       # length of currently loaded video
        self.play = False                   # flag for the play button toggle
        self.current_frame = 0              # the current frame of the video
        self.create_video = False
        self.output_video = []       
        self.file_name = []       

        
        # Play related
        # self.set_read_threads = []          # Name of threaded function
        self.frame_timer = 0.0      # used to set the framerate during playing
        
        # Queues
        self.frame_q = []                   # queue for frames that are ready for coordinator

        self.r_frame_q = []                 # queue for frames that are requested by the GUI
        self.read_video_frame_q = []
        
        # swapping related
        # self.source_embedding = []          # array with indexed source embeddings

        self.found_faces = []   # array that maps the found faces to source faces    

        self.parameters = []


        self.target_video = []

        self.fps = 1.0
        self.temp_file = []

        self.start_time = []
        self.record = False
        self.output = []
        self.image = []

        self.saved_video_path = []
        self.sp = []
        self.timer = []
        self.fps_average = []
        self.total_thread_time = 0.0
        
        self.start_play_time = []
        self.start_play_frame = []
        
        self.rec_thread = []
        self.markers = []
        self.is_image_loaded = False
        self.stop_marker = -1
        self.perf_test = False

        # Benchmark state. `benchmark_mode` is set by
        # play_video('benchmark') and disables audio/wall-clock pacing
        # in process()'s present path so frames flow as fast as the
        # decode+swap pipeline allows. Per-frame generation times
        # (dispatch -> publish) accumulate into _bench_gen_times_ms;
        # _bench_start_wall captures perf_counter() at start so we can
        # report total elapsed when stop fires.
        # Per-worker CUDA streams — populated lazily on first swap_video
        # call from each thread. Wrapping the per-frame work in
        # `with torch.cuda.stream(worker_stream):` queues all torch ops
        # in that worker on its own stream instead of the global
        # default. Combined with per-thread ORT sessions configured
        # with `user_compute_stream=worker_stream.cuda_stream`, the
        # entire per-frame pipeline (torch ops + ORT calls) runs on
        # one stream per worker — workers no longer serialize on the
        # global default stream's queue, which was the bottleneck the
        # 2026-05-21 nsys traces revealed even after Per-Thread
        # sessions parallelized the inswapper call itself.
        self._worker_streams_tls = threading.local()

        self.benchmark_mode = False
        # When True (set by play_video('benchmark_headless')), every
        # _publish_frame call returns immediately instead of forwarding
        # to the GUI sink. Removes preview upload + Qt paint from the
        # benchmark wall-clock so the report reflects swap throughput
        # alone. Cleared when the benchmark stops (alongside
        # benchmark_mode).
        self.benchmark_headless = False
        self._bench_gen_times_ms = []
        self._bench_start_wall = 0.0

        self.control = []

        # Auto-orientation state (OrientAutoSwitch). `detected_orient_angle` is
        # the cached winning rotation from the 4-way probe (None = re-probe on
        # next frame). `orient_fine_angle` is an EMA of the eye-axis tilt added
        # on top for sub-90 refinement. `orient_miss_streak` triggers a re-probe
        # after several consecutive detection failures (shot change / scene cut).
        self.detected_orient_angle = None
        self.orient_fine_angle = 0.0
        self.orient_miss_streak = 0
        self.orient_miss_threshold = 5

        self.process_q =    {
                            "Thread":                   [],
                            "FrameNumber":              [],
                            "ProcessedFrame":           [],
                            "Status":                   'clear',
                            "ThreadTime":               []
                            }   
        self.process_qs = []
        self.rec_q =    {
                            "Thread":                   [],
                            "FrameNumber":              [],
                            "Status":                   'clear'
                            }
        self.rec_qs = []

        # Per-frame perf caches. _blur_cache is shared across worker threads
        # (transforms.GaussianBlur is stateless once constructed; dict.get +
        # assignment is GIL-atomic so a duplicate-build race is harmless).
        # _scratch holds per-thread reusable tensors to dodge the per-frame
        # cudaMalloc churn — every worker gets its own set of buffers via
        # threading.local so the in-flight swap_core calls don't trample
        # each other.
        self._blur_cache = {}
        self._scratch = threading.local()

        # Cache of swapper conditioning latents, keyed by source-embedding
        # bytes. The latent depends only on s_e (~constant across frames
        # for a given assigned face), so this avoids a CPU matmul +
        # host->device upload per swap call. Entries are ~2 KB each;
        # even a session with hundreds of unique sources is negligible.
        self._latent_cache: dict[bytes, torch.Tensor] = {}

        # Async scrub. submit_scrub() (called on the GUI thread) appends
        # to a bounded deque drained by a single worker thread. The deque
        # has maxlen=5 — when full, appending drops the oldest entry, so
        # a fast drag keeps at most 5 frames pending instead of letting
        # the queue balloon. Every queued frame is rendered.
        self._scrub_queue: deque = deque(maxlen=5)
        self._scrub_cond = threading.Condition()
        self._scrub_stop = threading.Event()
        self._scrub_thread = threading.Thread(
            target=self._scrub_loop, name='scrub', daemon=True,
        )
        self._scrub_thread.start()

        # Reusable worker pool for thread_video_read dispatches. Sized
        # to ThreadsSlider on each play/record so we don't pay
        # OS thread creation (~50-200us on Windows) per frame. Resized
        # lazily if the slider value changes between plays.
        self._executor = None
        self._executor_size = 0

        # Push-based frame delivery. The coordinator installs a callback
        # via set_frame_callback() that emits bus.frame_ready on the GUI
        # thread (auto-marshaled via QueuedConnection). Eliminates the
        # frame_q -> _tick drain -> emit roundtrip; pacing runs on
        # _pacer_thread instead of the GUI's 1 kHz tick.
        self._frame_callback = None
        self._pacer_thread = None
        self._pacer_stop = threading.Event()
        self._start_pacer()

    def set_frame_callback(self, cb):
        """Install the GUI-thread sink. cb signature: (image, frame_no, requested)."""
        self._frame_callback = cb

    def _warp_grid_sample(self, src, sample_matrix, output_size, padding_mode='zeros'):
        """Affine-warp `src` into `output_size` by building a sampling
        grid from a pixel-space 2x3 matrix.

        sample_matrix maps OUTPUT pixel coords (x, y) → INPUT pixel
        coords (x_in, y_in) — i.e. it's the matrix the GPU consults
        when filling each output pixel ('inverse warp'). For us that
        comes straight from `skimage.SimilarityTransform.inverse.params`
        (input affine case) or `tform.params` (paste-back case).

        Why this beats v2.functional.affine + crop: torchvision's
        affine output has the same H×W as input — for a 1080p frame
        warped to a 256×256 face crop, the affine still computes ~6M
        pixels and we discard all but the corner. grid_sample with a
        target-sized grid only computes the pixels we keep (~65K),
        which is the chunk we were leaving on the table per the
        2026-05-21 nsys trace's swap_core breakdown.

        align_corners=False matches torchvision's affine semantics
        so output is bit-similar (modulo bilinear-vs-bilinear rounding
        in the last decimal place) to the path it replaces."""
        src_h = src.shape[-2]
        src_w = src.shape[-1]
        out_h, out_w = int(output_size[0]), int(output_size[1])

        # Output coordinate grid in pixel space.
        y_coords = torch.arange(out_h, dtype=torch.float32, device='cuda')
        x_coords = torch.arange(out_w, dtype=torch.float32, device='cuda')
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')

        # Apply pixel-space matrix: in_xy = M[:, :2] @ (x, y) + M[:, 2]
        if isinstance(sample_matrix, torch.Tensor):
            M = sample_matrix.to('cuda').to(torch.float32)
        else:
            M = torch.from_numpy(
                np.ascontiguousarray(sample_matrix, dtype=np.float32)
            ).to('cuda')
        flat = torch.stack([xx, yy], dim=-1).reshape(-1, 2)
        in_xy = flat @ M[:, :2].T + M[:, 2]
        in_xy = in_xy.reshape(out_h, out_w, 2)

        # Normalize to [-1, 1] using align_corners=False semantics
        # (pixel center 0 → -1 + 1/W, pixel center W-1 → 1 - 1/W).
        in_xy[..., 0] = ((in_xy[..., 0] + 0.5) / src_w) * 2.0 - 1.0
        in_xy[..., 1] = ((in_xy[..., 1] + 0.5) / src_h) * 2.0 - 1.0

        grid = in_xy.unsqueeze(0)  # (1, out_h, out_w, 2)

        # grid_sample needs float input; src may be uint8.
        src_4d = src.unsqueeze(0) if src.ndim == 3 else src
        if src_4d.dtype != torch.float32:
            src_4d = src_4d.float()

        sampled = torch.nn.functional.grid_sample(
            src_4d, grid,
            mode='bilinear',
            padding_mode=padding_mode,
            align_corners=False,
        )
        return sampled.squeeze(0)  # (C, out_h, out_w)

    def _get_worker_stream(self):
        """Return the calling thread's dedicated CUDA stream, creating
        one lazily on first call. Used to wrap the per-frame swap
        pipeline so each worker thread runs its torch ops + ORT calls
        on its own stream instead of the global default — eliminates
        the cross-worker queue serialization that was the bottleneck
        after Per-Thread inswapper sessions parallelized the actual
        ORT call. Streams are stashed in thread-local storage and die
        with the worker (TLS auto-cleans dead-thread slots)."""
        s = getattr(self._worker_streams_tls, 'stream', None)
        if s is None:
            s = torch.cuda.Stream()
            self._worker_streams_tls.stream = s
        return s

    def _publish_frame(self, image, frame_no, requested):
        # Headless benchmark: drop the frame instead of forwarding to
        # the GUI sink. Bench timing already accumulated upstream; the
        # frame counter is what matters for the report. Skipping the
        # callback removes the GL texture upload + Qt repaint from the
        # measured path so the benchmark reflects swap throughput, not
        # display throughput.
        if self.benchmark_mode and self.benchmark_headless:
            return
        cb = self._frame_callback
        if cb is None:
            return
        try:
            cb(image, frame_no, requested)
        except Exception as e:
            print(f'[VideoManager._publish_frame] callback raised: {e}')

    def _start_pacer(self):
        if self._pacer_thread is not None and self._pacer_thread.is_alive():
            return
        self._pacer_stop.clear()
        self._pacer_thread = threading.Thread(
            target=self._pacer_loop, name='vm-pacer', daemon=True,
        )
        self._pacer_thread.start()

    def _pacer_loop(self):
        # Runs process() at ~1kHz on a dedicated thread instead of from
        # the GUI tick. process() handles dispatch + audio-clock pacing;
        # finished frames are pushed via _publish_frame (callback), so
        # no queue drain is needed on the GUI side.
        while not self._pacer_stop.is_set():
            try:
                self.process()
            except Exception as e:
                import traceback
                print(f'[VideoManager._pacer_loop] {e}')
                traceback.print_exc()
            time.sleep(0.001)

    def _ensure_executor(self, n_workers):
        n_workers = max(1, int(n_workers))
        if self._executor is not None and self._executor_size == n_workers:
            return
        if self._executor is not None:
            # Don't wait — outstanding workers will finish on their own
            # and we already have a fresh pool ready for the new size.
            self._executor.shutdown(wait=False)
        self._executor = ThreadPoolExecutor(
            max_workers=n_workers, thread_name_prefix='swap',
        )
        self._executor_size = n_workers
        # Per-Thread model sessions are pinned by Models' strong-ref
        # list so they survive the dying worker's TLS slot. Whenever the
        # pool is rebuilt (ThreadsSlider change), drop those refs so
        # the old workers' sessions get GC'd as they die — otherwise
        # reducing thread count leaves stale sessions consuming VRAM.
        clear = getattr(self.models, 'clear_per_thread_sessions', None)
        if callable(clear):
            clear()
        # New worker pool → new batch of model loads incoming. Tell
        # Models how many to expect so the [N/M] progress tag on
        # session-create messages is accurate.
        update = getattr(self.models, 'update_load_expectation', None)
        if callable(update):
            update(n_workers)
        # Pre-warm: force the pool to spawn n_workers OS threads up front.
        # ThreadPoolExecutor lazy-spawns workers on demand and reuses any
        # idle worker, so if real submits arrive faster than they're
        # consumed but slower than a worker goes idle, the pool can end
        # up under-sized — observed in nsys traces where ThreadsSlider=5
        # spawned only 4 swap_N threads. A barrier of n_workers parties
        # makes each pre-warm task block until all n_workers have
        # arrived, forcing concurrent thread creation. The submit calls
        # are non-blocking; tasks complete async once the barrier opens.
        if n_workers > 1:
            barrier = threading.Barrier(n_workers, timeout=10.0)
            for _ in range(n_workers):
                self._executor.submit(self._prewarm_pool, barrier)

    @staticmethod
    def _prewarm_pool(barrier):
        try:
            barrier.wait()
        except threading.BrokenBarrierError:
            # Timeout means the pool couldn't spawn enough threads
            # (system thread limit, GIL contention during startup, etc.).
            # The pool still has whatever threads did spawn; the dispatch
            # loop will fill them as usual.
            pass

    def preload_models(self, n_workers=None):
        """Warm the swap-pipeline sessions at startup so the first frame —
        and, in Per-Thread mode, each worker's first frame — never pays the
        session build/deserialize cost mid-session.

        Sizing follows `ThreadsSlider`: Per-Thread mode builds one session
        set per worker thread; Shared mode builds one set total. Runs off
        the GUI thread and returns immediately — the actual building happens
        on the worker pool / a background driver so the UI stays responsive
        while engines build.

        Only the thread-scaled pipeline models are preloaded — the
        recognizer (arcface), the selected detector, and the selected
        inswapper resolution. Which detector (Retinaface / SCRDF)
        and which swapper (inswapper_128 for 128/256/512 vs the native-256
        model for "256-Native") come from the current DetectTypeTextSel /
        SwapperTypeTextSel params, so preload builds exactly what a swap
        will use. Feature-gated models (restorers, mask nets) are shared
        singletons loaded on demand, so they don't belong to a
        thread-sized preload."""
        models = self.models
        if models is None:
            return
        # Read the current selections so preload builds exactly the models a
        # swap will use (which detector, which inswapper resolution).
        try:
            swapper_type = str(self.parameters.get('SwapperTypeTextSel', '128'))
        except Exception:
            swapper_type = '128'
        try:
            detect_mode = str(self.parameters.get('DetectTypeTextSel', 'Retinaface'))
        except Exception:
            detect_mode = 'Retinaface'
        if hasattr(models, 'swap_pipeline_files_present'):
            if not models.swap_pipeline_files_present(swapper_type, detect_mode):
                print('[VideoManager.preload_models] pipeline model files not '
                      'present in models folder; skipping preload')
                return
        if n_workers is None:
            try:
                n_workers = int(self.parameters.get('ThreadsSlider', 1))
            except Exception:
                n_workers = 1
        n_workers = max(1, int(n_workers))
        self._ensure_executor(n_workers)

        mode = getattr(models, '_model_session_mode', 'Shared')
        if mode == 'Per-Thread' and n_workers > 1:
            # Drive the two-phase build on a dedicated thread so the .result()
            # waits below don't block the GUI thread.
            threading.Thread(
                target=self._preload_driver, args=(n_workers, swapper_type, detect_mode),
                name='vm-preload', daemon=True,
            ).start()
        else:
            # Shared mode (one shared session per model) or a single worker:
            # a single build is sufficient and avoids the unlocked Shared-mode
            # build race across threads. add_done_callback fires on the
            # executor thread even if the task raised, so the GUI always
            # gets a completion signal (it re-checks what actually loaded).
            fut = self._executor.submit(self._preload_worker, None, swapper_type, detect_mode)
            fut.add_done_callback(lambda _f: bus.models_preloaded.emit())

    def _preload_driver(self, n_workers, swapper_type='128', detect_mode='Retinaface'):
        """Two-phase Per-Thread preload that avoids a cold-cache TRT build
        stampede. Phase 1 builds one worker's session set alone — paying the
        one-time engine build and warming the on-disk cache. Phase 2 then
        builds the remaining workers in parallel, deserializing from the now-
        warm cache instead of N threads racing to build the same engine."""
        try:
            # Phase 1: single build warms the shared on-disk engine cache.
            self._executor.submit(
                self._preload_worker, None, swapper_type, detect_mode).result()
            # Phase 2: populate every worker thread's session set. The
            # barrier forces one task per distinct thread (ThreadPoolExecutor
            # would otherwise let a fast thread grab several), so all N
            # workers end up warm. The Phase-1 thread participates too; its
            # getters just return the sessions it already built.
            barrier = threading.Barrier(n_workers, timeout=180.0)
            futs = [
                self._executor.submit(self._preload_worker, barrier, swapper_type, detect_mode)
                for _ in range(n_workers)
            ]
            for f in futs:
                f.result()
        except Exception as e:
            print(f'[VideoManager._preload_driver] {e}')
        finally:
            # Always signal the GUI the build finished (success or not) so
            # the Preload button leaves its in-progress state; the GUI
            # re-checks pipeline_sessions_loaded to decide what to show.
            bus.models_preloaded.emit()

    def _preload_worker(self, barrier, swapper_type='128', detect_mode='Retinaface'):
        """Build the pipeline sessions on THIS pool thread, inside the
        worker's CUDA stream so ORT binds to the same stream swap_video
        uses. `barrier` (or None) synchronizes concurrent Phase-2 tasks."""
        if barrier is not None:
            try:
                barrier.wait()
            except threading.BrokenBarrierError:
                pass
        try:
            worker_stream = self._get_worker_stream()
            with torch.cuda.stream(worker_stream):
                self.models.preload_pipeline_sessions(swapper_type, detect_mode)
        except Exception as e:
            print(f'[VideoManager._preload_worker] {e}')

    def _get_gaussian_blur(self, kernel_size, sigma):
        """Memoized GaussianBlur module. Same (kernel_size, sigma) -> same
        module object, so the kernel tensor is built once per parameter
        combination instead of every frame."""
        key = (int(kernel_size), float(sigma))
        blur = self._blur_cache.get(key)
        if blur is None:
            blur = transforms.GaussianBlur(key[0], key[1])
            self._blur_cache[key] = blur
        return blur

    def _get_latent_for(self, s_e, *, scale=1.0, extrap_amount=0.0, latent_mode='emap'):
        """Return the swapper conditioning latent for `s_e` (the source
        embedding) as a CUDA tensor. First call computes (CPU matmul +
        host->device upload); subsequent calls with the same key hit
        the cache. Key includes scale + extrap_amount + latent_mode
        because all three change the output.

        latent_mode: 'emap' for inswapper (default), or 'raw' (skips emap
        projection — for a model that consumes the raw normalized
        embedding directly).
        """
        s = np.ascontiguousarray(s_e, dtype=np.float32)
        key = (s.tobytes(), round(float(scale), 4), round(float(extrap_amount), 4), latent_mode)
        cached = self._latent_cache.get(key)
        if cached is not None:
            return cached
        latent_np = self.models.calc_swapper_latent(
            s, scale=scale, extrap_amount=extrap_amount, latent_mode=latent_mode,
        )
        latent = torch.from_numpy(latent_np).float().to('cuda')
        self._latent_cache[key] = latent
        return latent

    def clear_latent_cache(self):
        """Drop all cached swapper latents. Called when the session mean
        embedding changes (which changes Distinctiveness output for every
        cached entry) — without this, old extrapolations against the
        stale mean would keep being returned."""
        self._latent_cache.clear()

    def submit_scrub(self, frame, marker=True):
        """GUI-thread entry point for async scrub. Appends the request
        to the bounded scrub queue (drop-oldest when full) so every
        intermediate frame in a drag is rendered, capped at 5 pending.
        """
        if not self.is_video_loaded and not self.is_image_loaded:
            return
        with self._scrub_cond:
            self._scrub_queue.append((int(frame), bool(marker)))
            self._scrub_cond.notify()

    def _scrub_loop(self):
        """Drain the scrub queue on a dedicated daemon thread. FIFO —
        a fast drag with the queue full will see the oldest pending
        request displaced by `deque.append`, so the worker keeps
        making forward progress without falling further behind."""
        while not self._scrub_stop.is_set():
            with self._scrub_cond:
                while not self._scrub_queue and not self._scrub_stop.is_set():
                    self._scrub_cond.wait()
                if self._scrub_stop.is_set():
                    return
                frame, marker = self._scrub_queue.popleft()
            try:
                self._scrub_worker(frame, marker)
            except Exception:
                import traceback
                traceback.print_exc()

    def _scrub_worker(self, frame, marker):
        """Decode + swap + publish one queued scrub request. Every
        queued request runs to completion — no generation-check
        bail-outs. The queue's `maxlen=5` cap is the only mechanism
        bounding work."""
        try:
            if self.is_video_loaded and self.player is not None:
                self.current_frame = int(frame)
                try:
                    target_image, _pts = self.player.get_frame_at(self.current_frame)
                except Exception as e:
                    print('Scrub decode failed at frame',
                          self.current_frame, '(', e, ')')
                    return
                if not self.control.get('SwapFacesButton'):
                    image = target_image
                else:
                    image = self.swap_video(target_image, self.current_frame, marker)
                self._publish_frame(image, self.current_frame, True)
                return

            if self.is_image_loaded:
                if not self.control.get('SwapFacesButton'):
                    image = self.image
                else:
                    image = self.swap_video(self.image, self.current_frame, False)
                self._publish_frame(image, self.current_frame, True)
        except Exception as e:
            import traceback
            print(f'[_scrub_worker] {e}')
            traceback.print_exc()

    def _get_scratch(self, name, shape, dtype=torch.float32):
        """Return a per-thread reusable cuda tensor with the requested shape
        and dtype. Re-allocates only when the shape or dtype changes (e.g.
        when pipeline_dim changes between frames). Caller is responsible
        for resetting contents (.zero_() / .fill_()) when they need known
        starting values."""
        cache = getattr(self._scratch, 'tensors', None)
        if cache is None:
            cache = {}
            self._scratch.tensors = cache
        t = cache.get(name)
        shape = tuple(shape)
        if t is None or tuple(t.shape) != shape or t.dtype != dtype:
            t = torch.empty(shape, dtype=dtype, device='cuda')
            cache[name] = t
        return t

    def assign_found_faces(self, found_faces):
        self.found_faces = found_faces


    def load_target_video( self, file ):
        # If we already have a player open, release it cleanly.
        if self.player is not None:
            try:
                self.player.close()
            except Exception:
                pass
            self.player = None
        if self.capture:
            try:
                self.capture.release()
            except Exception:
                pass
            self.capture = []

        self.video_file = file
        try:
            self.player = MediaPlayer(file)
        except Exception as e:
            print("Cannot open file: ", file, '(', e, ')')
            self.is_video_loaded = False
            return

        self.fps = self.player.fps
        self.target_video = file
        self.is_video_loaded = True
        self.is_image_loaded = False
        self.video_frame_total = self.player.video_frame_total
        self.play = False
        self.current_frame = 0
        self.frame_timer = time.time()
        self.frame_q = []
        self.r_frame_q = []
        self.found_faces = []
        self.detected_orient_angle = None
        self.orient_fine_angle = 0.0
        self.orient_miss_streak = 0
        bus.slider_length_changed.emit(self.video_frame_total - 1)

        # First-frame preview. Pushed directly to the GUI sink instead
        # of via the legacy r_frame_q drain.
        try:
            image = self.player.get_first_frame()
            self._publish_frame(image, False, True)
        except Exception as e:
            print('First-frame preview failed for', file, '(', e, ')')
    
    def load_target_image(self, file):
        # Switching to image mode: tear down any video player so its
        # decoder thread isn't sitting idle on a no-longer-relevant file.
        if self.player is not None:
            try:
                self.player.close()
            except Exception:
                pass
            self.player = None
        self.is_video_loaded = False
        self.play = False
        self.frame_q = []
        self.r_frame_q = []
        self.found_faces = []
        self.image = cv2.imread(file) # BGR
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB) # RGB
        self._publish_frame(self.image, False, False)

        self.is_image_loaded = True
        self.detected_orient_angle = None
        self.orient_fine_angle = 0.0
        self.orient_miss_streak = 0


    def probe_orientation(self, img, detect_mode, detect_score, input_size=640):
        """Try rotating to [0, 90, 180, 270] and pick the orientation where
        the detector finds the most faces. Ties go to 0 (the lowest-angle
        candidate), so we prefer no rotation when evidence is weak.

        Returns the winning angle in degrees. Caller is responsible for
        rotating the image by that angle before the real detect pass."""
        best_angle = 0
        best_count = -1
        for angle in (0, 90, 180, 270):
            if angle == 0:
                probe_img = img
            else:
                probe_img = v2.functional.rotate(
                    img, angle=angle,
                    interpolation=v2.InterpolationMode.BILINEAR, expand=True,
                )
            kpss = self.models.run_detect(probe_img, detect_mode, max_num=20, score=detect_score, input_size=input_size)
            n = len(kpss)
            if n > best_count:
                best_count = n
                best_angle = angle
        return best_angle


    ## Queues for the Coordinator
    def get_frame(self):
        frame = self.frame_q[0]
        self.frame_q.pop(0)
        return frame
    
    def get_frame_length(self):
        return len(self.frame_q)  
        
    def get_requested_frame_length(self):
        return len(self.r_frame_q)          
    

    def get_requested_video_frame(self, frame, marker=True):
        temp = []
        if self.is_video_loaded and self.player is not None:
            if self.play == True:
                self.play_video("stop")
                self.process_qs = []

            self.current_frame = int(frame)

            try:
                target_image, _pts = self.player.get_frame_at(self.current_frame)
            except Exception as e:
                print('Scrub decode failed at frame', self.current_frame, '(', e, ')')
                return

            if not self.control['SwapFacesButton']:
                # Pass-through: Qt preview accepts CUDA tensors directly.
                image = target_image
            else:
                image = self.swap_video(target_image, self.current_frame, marker)
            self._publish_frame(image, self.current_frame, True)

        elif self.is_image_loaded:
            if not self.control['SwapFacesButton']:
                image = self.image  # RGB
            else:
                image = self.swap_video(self.image, self.current_frame, False)
            self._publish_frame(image, self.current_frame, True)


    def find_lowest_frame(self, queues):
        min_frame=999999999
        index=-1
        
        for idx, thread in enumerate(queues):
            frame = thread['FrameNumber']
            if frame != []:
                if frame < min_frame:
                    min_frame = frame
                    index=idx
        return index, min_frame


    def _print_benchmark_report(self):
        """Print the benchmark summary collected by process()."""
        elapsed = time.perf_counter() - self._bench_start_wall
        n = len(self._bench_gen_times_ms)
        mode = '[benchmark headless]' if self.benchmark_headless else '[benchmark]'
        print(f'{mode} complete:', flush=True)
        if n == 0:
            print(f'  no frames recorded (total elapsed {elapsed:.2f} s)', flush=True)
            return
        arr = np.array(self._bench_gen_times_ms)
        gen_tag = (
            'decode-pull -> count (preview disabled)'
            if self.benchmark_headless
            else 'decode-pull -> display'
        )
        print(f'  frames:         {n}', flush=True)
        print(f'  total time:     {elapsed:.2f} s', flush=True)
        print(f'  mean gen ms:    {arr.mean():.2f} ms/frame  ({gen_tag})', flush=True)
        print(f'  effective fps:  {n / elapsed:.2f}', flush=True)

    def play_video(self, command):
        # print(inspect.currentframe().f_back.f_code.co_name, '->play_video: ')
        # "benchmark" is "play" with audio/wall-clock pacing disabled and
        # per-frame timing accumulation enabled. Fall through to the
        # normal play setup so we reuse process_qs / _executor / decoder
        # spin-up without duplicating that code.
        if command == "benchmark":
            self.benchmark_mode = True
            self.benchmark_headless = False
            self._bench_gen_times_ms = []
            self._bench_start_wall = time.perf_counter()
            command = "play"

        if command == "benchmark_headless":
            self.benchmark_mode = True
            self.benchmark_headless = True
            self._bench_gen_times_ms = []
            self._bench_start_wall = time.perf_counter()
            command = "play"

        if command == "play":
            # Initialization
            self.play = True
            self.fps_average = []
            self.process_qs = []
            self.frame_timer = time.time()

            # Create reusable queue based on number of threads
            self._ensure_executor(self.parameters['ThreadsSlider'])
            for i in range(self.parameters['ThreadsSlider']):
                    new_process_q = self.process_q.copy()
                    self.process_qs.append(new_process_q)

            # Hand the player the playback start position and spin up its
            # decoder thread (+ audio output if requested). Audio sync is now
            # handled inside MediaPlayer via sounddevice — no ffplay
            # subprocess, no stdout-parse-for-sync hack.
            if self.player is not None:
                self.player.seek(self.current_frame)
                self.player.start_playback(audio_enabled=bool(self.control.get('AudioButton')))

        elif command == "stop":
            self.play = False
            bus.stop_play.emit()

            index, min_frame = self.find_lowest_frame(self.process_qs)

            if index != -1:
                self.current_frame = min_frame-1

            if self.player is not None:
                self.player.stop_playback()

            torch.cuda.empty_cache()

            if self.benchmark_mode:
                self._print_benchmark_report()
                self.benchmark_mode = False
                self.benchmark_headless = False

        elif command=='stop_from_gui':
            self.play = False

            # Find the lowest frame in the current render queue and set the current frame to the one before it
            index, min_frame = self.find_lowest_frame(self.process_qs)
            if index != -1:
                self.current_frame = min_frame-1

            if self.player is not None:
                self.player.stop_playback()

            torch.cuda.empty_cache()

            if self.benchmark_mode:
                self._print_benchmark_report()
                self.benchmark_mode = False
                self.benchmark_headless = False

        elif command == "record":
            self.record = True
            self.play = True
            self.total_thread_time = 0.0
            self.process_qs = []

            self._ensure_executor(self.parameters['ThreadsSlider'])
            for i in range(self.parameters['ThreadsSlider']):
                    new_process_q = self.process_q.copy()
                    self.process_qs.append(new_process_q)

           # Initialize
            self.timer = time.time()
            frame_width = self.player.width if self.player is not None else 0
            frame_height = self.player.height if self.player is not None else 0

            self.start_time = float(self.current_frame) / float(self.fps)

            self.file_name = os.path.splitext(os.path.basename(self.target_video))
            base_filename =  self.file_name[0]+"_"+str(time.time())[:10]
            self.output = os.path.join(self.saved_video_path, base_filename)
            self.temp_file = self.output+"_temp"+self.file_name[1]

            if self.parameters['RecordTypeTextSel']=='FFMPEG':
                ffmpeg_exe = _find_ffmpeg()
                if ffmpeg_exe is None:
                    print('[VideoManager] ffmpeg.exe not found on PATH or in '
                          'common install locations. Install ffmpeg (or '
                          '`pip install imageio-ffmpeg` for a bundled binary) '
                          'and restart, or switch RecordType to OPENCV.')
                    self.record = False
                    self.play = False
                    return
                args =  [ffmpeg_exe,
                        '-hide_banner',
                        '-loglevel',    'error',
                        "-an",
                        "-r",           str(self.fps),
                        "-i",           "pipe:",
                        # '-g',           '25',
                        "-vf",          "format=yuvj420p",
                        "-c:v",         "libx264",
                        "-crf",         str(self.parameters['VideoQualSlider']),
                        "-r",           str(self.fps),
                        "-s",           str(frame_width)+"x"+str(frame_height),
                        self.temp_file]

                self.sp = subprocess.Popen(args, stdin=subprocess.PIPE)

            elif self.parameters['RecordTypeTextSel']=='OPENCV':
                size = (frame_width, frame_height)
                self.sp = cv2.VideoWriter(self.temp_file,  cv2.VideoWriter_fourcc(*'mp4v') , self.fps, size)

            # Start the player so the decoder feeds the worker pool. No
            # audio during recording — recording's audio is muxed in from
            # the original file at end-of-record (see process()'s ffmpeg
            # mux block), and we don't want to play it out the speakers.
            if self.player is not None:
                self.player.seek(self.current_frame)
                self.player.start_playback(audio_enabled=False)
      
    # @profile
    def process(self):
        # Add threads to Queue: pull a pre-decoded frame from the player
        # and dispatch a swap worker with it. Drains every currently-clear
        # slot in a single tick — workers finish in bursts (4-5 within the
        # same wall-clock instant when their frames complete close together)
        # and dispatching one-per-tick at 1kHz left each idle worker
        # waiting 5-7ms before its next frame. The decoder is the only
        # rate limiter now: if `get_next_frame(timeout=0)` returns None,
        # break — no point iterating further when the queue is dry.
        if self.play == True and self.is_video_loaded == True and self.player is not None:
            for item in self.process_qs:
                if item['Status'] == 'clear' and self.current_frame < self.video_frame_total:
                    res = self.player.get_next_frame(timeout=0)
                    if res is None:
                        break  # decoder hasn't produced a frame yet — try later
                    frame, frame_number, pts = res
                    # Assign the slot's bookkeeping fields BEFORE starting
                    # the thread — otherwise the worker can race the main
                    # thread past these lines and fail to find its slot
                    # (FrameNumber would still be []) when it tries to
                    # publish the result.
                    item['FrameNumber'] = frame_number
                    item['Pts'] = pts
                    item['Status'] = 'started'
                    item['ThreadTime'] = time.time()
                    # Capture dispatch time on a separate field so the
                    # worker can't overwrite it (thread_video_read
                    # reuses ThreadTime to record its own elapsed).
                    # Only set when benchmarking — production play
                    # doesn't need it.
                    if self.benchmark_mode:
                        item['BenchStart'] = time.perf_counter()
                    item['Thread'] = self._executor.submit(
                        self.thread_video_read, frame, frame_number,
                    )
                    # Keep current_frame as the "next frame to dispatch"
                    # marker for stop()'s find_lowest_frame fallback.
                    self.current_frame = max(self.current_frame, frame_number + 1)

        else:
            self.play = False

        # Always be emptying the queues. Display scheduling now uses the
        # audio clock when available (sample-accurate sync); otherwise it
        # falls back to wall-clock pacing as before.
        if not self.record and self.play:
            audio_pos = self.player.get_audio_position() if self.player is not None else None

            index, min_frame = self.find_lowest_frame(self.process_qs)
            if index != -1 and self.process_qs[index]['Status'] == 'finished':
                ready_pts = self.process_qs[index].get('Pts')
                if self.benchmark_mode:
                    # No sync — present the moment the swap is done so the
                    # pipeline runs as fast as the GPU + decoder allow.
                    present_now = True
                elif audio_pos is not None and ready_pts is not None:
                    # Audio-clock pacing: present the frame once audio has
                    # reached (or passed) its PTS. Don't drop frames here
                    # — the swap is already paid for, and dropping during
                    # preview defeats the QA workflow.
                    present_now = audio_pos >= ready_pts
                else:
                    # Wall-clock fallback (no audio).
                    time_diff = time.time() - self.frame_timer
                    present_now = time_diff >= 1.0 / float(self.fps)

                if present_now:
                    # Generation time: dispatch -> publish. We capture
                    # before publishing so the timing reflects the work
                    # the user is benchmarking, not the publish cost
                    # itself (which is fixed Qt signal overhead).
                    if self.benchmark_mode:
                        bench_start = self.process_qs[index].get('BenchStart')
                        if bench_start is not None:
                            self._bench_gen_times_ms.append(
                                (time.perf_counter() - bench_start) * 1000.0
                            )
                    self._publish_frame(
                        self.process_qs[index]['ProcessedFrame'],
                        self.process_qs[index]['FrameNumber'],
                        False,
                    )

                    # Report fps based on the present cadence (debug only).
                    now = time.time()
                    self.fps_average.append(now - self.frame_timer)
                    if len(self.fps_average) >= floor(self.fps):
                        avg_dt = float(np.average(self.fps_average))
                        if avg_dt > 0:
                            fps = round(1.0 / avg_dt, 2)
                            msg = "%s fps, %s process time" % (
                                fps, round(self.process_qs[index]['ThreadTime'], 4))
                        self.fps_average = []

                    if (self.process_qs[index]['FrameNumber'] >= self.video_frame_total - 1
                            or self.process_qs[index]['FrameNumber'] == self.stop_marker):
                        self.play_video('stop')

                    self.process_qs[index]['Status'] = 'clear'
                    self.process_qs[index]['Thread'] = []
                    self.process_qs[index]['FrameNumber'] = []
                    self.process_qs[index]['ThreadTime'] = []
                    self.process_qs[index]['Pts'] = None
                    self.process_qs[index].pop('BenchStart', None)
                    if audio_pos is None:
                        self.frame_timer += 1.0 / self.fps
                    else:
                        self.frame_timer = now

        elif self.record:

            index, min_frame = self.find_lowest_frame(self.process_qs)

            if index != -1:

                # If the swapper thread has finished generating a frame
                if self.process_qs[index]['Status'] == 'finished':
                    image = self.process_qs[index]['ProcessedFrame']
                    # swap_video now returns a CUDA tensor; record I/O
                    # needs a CPU numpy view. The bounce is unavoidable
                    # for disk write, so do it here once and reuse for
                    # both the file write and the preview frame_q.
                    if isinstance(image, torch.Tensor):
                        image = image.cpu().numpy()

                    if self.parameters['RecordTypeTextSel']=='FFMPEG':
                        pil_image = Image.fromarray(image)
                        pil_image.save(self.sp.stdin, 'BMP')

                    elif self.parameters['RecordTypeTextSel']=='OPENCV':
                        self.sp.write(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

                    self._publish_frame(
                        image, self.process_qs[index]['FrameNumber'], False,
                    )

                    # Close video and process
                    if self.process_qs[index]['FrameNumber'] >= self.video_frame_total-1 or self.process_qs[index]['FrameNumber'] == self.stop_marker or self.play == False:
                        # Capture the last-recorded frame number BEFORE
                        # play_video("stop") clears process_qs slots.
                        last_recorded_frame = self.process_qs[index]['FrameNumber']
                        self.play_video("stop")
                        stop_time = float(last_recorded_frame + 1) / float(self.fps)
                        if stop_time == 0:
                            stop_time = float(self.video_frame_total) / float(self.fps)
                        
                        if self.parameters['RecordTypeTextSel']=='FFMPEG':
                            self.sp.stdin.close()
                            self.sp.wait()
                        elif self.parameters['RecordTypeTextSel']=='OPENCV':    
                            self.sp.release()

                        orig_file = self.target_video
                        final_file = self.output+self.file_name[1]
                        ffmpeg_exe = _find_ffmpeg()
                        if ffmpeg_exe is None:
                            # Recording succeeded but we can't mux audio.
                            # Promote the temp file to the final filename
                            # so the user at least keeps the silent video.
                            print('[VideoManager] ffmpeg.exe not found for '
                                  'audio mux step — saving video without '
                                  'audio as %s' % final_file)
                            try:
                                os.replace(self.temp_file, final_file)
                            except OSError as e:
                                print('[VideoManager] could not rename temp '
                                      'file: %s' % e)
                        else:
                            print("adding audio...")
                            args = [ffmpeg_exe,
                                    '-hide_banner',
                                    '-loglevel',    'error',
                                    "-i", self.temp_file,
                                    "-ss", str(self.start_time), "-to", str(stop_time), "-i",  orig_file,
                                    "-c",  "copy", # may be c:v
                                    "-map", "0:v:0", "-map", "1:a:0?",
                                    "-shortest",
                                    final_file]
                            subprocess.run(args)
                            try:
                                os.remove(self.temp_file)
                            except OSError:
                                pass

                        timef= time.time() - self.timer 
                        self.record = False
                        print('Video saved as:', final_file)
                        msg = "Total time: %s s." % (round(timef,1))
                        print(msg)

                        
                    self.total_thread_time = []
                    self.process_qs[index]['Status'] = 'clear'
                    self.process_qs[index]['FrameNumber'] = []
                    self.process_qs[index]['Thread'] = []
                    self.frame_timer = time.time()
    # @profile
    def thread_video_read(self, target_image, frame_number):
        # The frame is already decoded — process() pulled it from the
        # MediaPlayer's queue. Run the swap (or pass through) and stash the
        # result in this worker's slot for process() to present.
        if not getattr(self, '_dbg_logged_input', False):
            kind = type(target_image).__name__
            extra = ''
            if isinstance(target_image, torch.Tensor):
                extra = f' device={target_image.device} dtype={target_image.dtype} shape={tuple(target_image.shape)}'
            elif hasattr(target_image, 'shape'):
                extra = f' shape={target_image.shape} dtype={getattr(target_image, "dtype", "?")}'
            gpu = getattr(self.player, 'gpu_decode_active', '?') if self.player is not None else '?'
            print(f'[thread_video_read] first frame: type={kind}{extra} gpu_decode_active={gpu}')
            self._dbg_logged_input = True
        if not self.control['SwapFacesButton']:
            # Pass the decoder output through untouched — the Qt preview
            # handles both CUDA HxWx3 uint8 tensors (fast path) and numpy
            # (CPU fallback). Avoid the GPU->CPU bounce that would force
            # the slow path even when torchcodec decoded on GPU.
            output = target_image
        else:
            output = self.swap_video(target_image, frame_number, True)

        for item in self.process_qs:
            if item['FrameNumber'] == frame_number:
                item['ProcessedFrame'] = output
                item['Status'] = 'finished'
                item['ThreadTime'] = time.time() - item['ThreadTime']
                break




    # @profile
    def swap_video(self, target_image, frame_number, use_markers):
        with nvtx_range(f"swap_video[f={frame_number}]"):
            worker_stream = self._get_worker_stream()
            with torch.cuda.stream(worker_stream):
                result = self._swap_video_inner(
                    target_image, frame_number, use_markers,
                )
            # Cross-stream wait, not a host sync. Anyone reading `result`
            # on the global default stream (the GUI thread that uploads
            # to GL, the recorder that copies to CPU, etc.) now sees a
            # dependency on worker_stream's pending writes. The default
            # stream queues a wait token instead of blocking the host,
            # so workers keep producing frames at full speed while the
            # consumer's stream waits for the data it needs.
            torch.cuda.default_stream().wait_stream(worker_stream)
            return result

    def _swap_video_inner(self, target_image, frame_number, use_markers):
        # Grab a local copy of the parameters to prevent threading issues
        parameters = self.parameters.copy()
        control = self.control.copy()

        # Find out if the frame is in a marker zone and copy the parameters if true
        if self.markers and use_markers:
            temp=[]
            for i in range(len(self.markers)):
                temp.append(self.markers[i]['frame'])
            idx = bisect.bisect(temp, frame_number)

            parameters = self.markers[idx-1]['parameters'].copy()

        # Accept either a CPU numpy array (legacy / PyAV-CPU decode path)
        # or a CUDA tensor (torchcodec GPU-resident decode path). Skipping
        # the CPU upload when it's already on GPU is the entire point of
        # the GPU-resident path.
        if isinstance(target_image, torch.Tensor):
            img = target_image
            if img.device.type != 'cuda':
                img = img.to('cuda')
            if img.dtype != torch.uint8:
                img = img.to(torch.uint8)
        else:
            img = torch.from_numpy(target_image.astype('uint8')).to('cuda') #HxWxc
        img = img.permute(2,0,1)#cxHxW
        
        #Scale up frame if it is smaller than 512
        img_x = img.size()[2]
        img_y = img.size()[1]
        
        if img_x<512 and img_y<512:
            # if x is smaller, set x to 512
            if img_x <= img_y:
                tscale = v2.Resize((int(512*img_y/img_x), 512), antialias=False)
            else:
                tscale = v2.Resize((512, int(512*img_x/img_y)), antialias=False)

            img = tscale(img)
            
        elif img_x<512:
            tscale = v2.Resize((int(512*img_y/img_x), 512), antialias=False)
            img = tscale(img)
        
        elif img_y<512:
            tscale = v2.Resize((512, int(512*img_x/img_y)), antialias=False)
            img = tscale(img)    

        # Decide how much to rotate the frame before detection.
        # Auto mode (OrientAutoSwitch) probes 4 orientations on first frame
        # (or after a detection-failure streak), caches the winning angle, and
        # adds an EMA of the eye-axis tilt for sub-90 refinement. Auto takes
        # precedence over the manual OrientSwitch/Slider combo.
        auto_orient = parameters.get('OrientAutoSwitch', False)
        detect_mode = parameters['DetectTypeTextSel']
        detect_score = parameters['DetectScoreSlider'] / 100.0
        detect_input_size = int(parameters.get('DetectInputSizeTextSel', 640))

        applied_angle = 0.0
        if auto_orient:
            if self.detected_orient_angle is None:
                self.detected_orient_angle = self.probe_orientation(img, detect_mode, detect_score, detect_input_size)
                self.orient_miss_streak = 0
                self.orient_fine_angle = 0.0
            applied_angle = float(self.detected_orient_angle) + float(self.orient_fine_angle)
        elif parameters['OrientSwitch']:
            applied_angle = float(parameters['OrientSlider'])

        if applied_angle != 0.0:
            img = v2.functional.rotate(img, angle=applied_angle, interpolation=v2.InterpolationMode.BILINEAR, expand=True)

        # Find all faces in frame and return a list of 5-pt kpss
        kpss = self.models.run_detect(img, detect_mode, max_num=10, score=detect_score, input_size=detect_input_size)

        # Auto-orientation feedback: track detection successes for re-probe and
        # update the fine-angle EMA from the first detected face's eye axis.
        if auto_orient:
            if len(kpss) == 0:
                self.orient_miss_streak += 1
                if self.orient_miss_streak >= self.orient_miss_threshold:
                    self.detected_orient_angle = None
                    self.orient_fine_angle = 0.0
            else:
                self.orient_miss_streak = 0
                try:
                    e_l = kpss[0][0]
                    e_r = kpss[0][1]
                    tilt = float(np.degrees(np.arctan2(e_r[1] - e_l[1], e_r[0] - e_l[0])))
                    # Guard against spurious large tilts (face in odd pose); the
                    # gross-orientation probe should have handled 90-degree cases.
                    if abs(tilt) < 30.0:
                        self.orient_fine_angle = 0.9 * self.orient_fine_angle + 0.1 * tilt
                except (IndexError, TypeError, ValueError):
                    pass
        # Get embeddings for all faces found in the fram
        ret = []
        for face_kps in kpss:
            face_emb, _ = self.models.run_recognize(img, face_kps)
            ret.append([face_kps, face_emb])
        
        if ret:
            # For each detected face in the frame, find the best-matching
            # entry in self.found_faces and (if it has a source assignment)
            # swap it. Detected faces are independent of each other: a slot
            # earlier in self.found_faces that doesn't match this frame
            # must not block swaps for slots that do. Best-match (highest
            # sim above threshold) ensures one detection → one swap even
            # when several found_face embeddings clear the threshold.
            threshold = float(parameters["ThresholdSlider"])
            for fface in ret:
                best_sim = -float('inf')
                best_slot = None
                for found_face in self.found_faces:
                    if not found_face.get("SourceFaceAssignments"):
                        continue  # target-only slot, no source assigned yet
                    emb = found_face.get("Embedding")
                    if emb is None:
                        continue
                    try:
                        sim = self.findCosineDistance(fface[1], emb)
                    except Exception:
                        # Malformed embedding on one slot shouldn't abort
                        # the whole frame — skip and keep matching others.
                        continue
                    if sim >= threshold and sim > best_sim:
                        best_sim = sim
                        best_slot = found_face
                if best_slot is not None:
                    s_e = best_slot["AssignedEmbedding"]
                    with nvtx_range("swap_core"):
                        img = self.swap_core(img, fface[0], s_e, parameters, control, slot=best_slot)

            img = img.permute(1,2,0)
            if not control['MaskViewButton'] and applied_angle != 0.0:
                img = img.permute(2,0,1)
                img = transforms.functional.rotate(img, angle=-applied_angle, expand=True)
                img = img.permute(1,2,0)

        else:
            img = img.permute(1,2,0)
            if applied_angle != 0.0:
                img = img.permute(2,0,1)
                img = v2.functional.rotate(img, angle=-applied_angle, interpolation=v2.InterpolationMode.BILINEAR, expand=True)
                img = img.permute(1,2,0)
        
        if self.perf_test:
            print('------------------------')  
        
        # Unscale small videos
        if img_x <512 or img_y < 512:
            tscale = v2.Resize((img_y, img_x), antialias=False)
            img = img.permute(2,0,1)
            img = tscale(img)
            img = img.permute(1,2,0)
            

        # Keep the result on CUDA as HxWx3 uint8 so the Qt preview's
        # CUDA-GL bridge can upload without a GPU->CPU->GPU bounce.
        # Consumers that need numpy (record file I/O, Save Image) do
        # the copy themselves — moves the unavoidable bounce out of
        # the per-frame display hot path.
        return img.to(torch.uint8).contiguous()

    def findCosineDistance(self, vector1, vector2):
        cos_dist = 1.0 - np.dot(vector1, vector2)/(np.linalg.norm(vector1)*np.linalg.norm(vector2)) # 2..0

        return 100.0-cos_dist*50.0



    # @profile
    def swap_core(self, img, kps, s_e, parameters, control, precomputed_latent=None, slot=None): # img = RGB
        # "512-Native" selects the inswapper_512 model: full 512x512 face
        # crop in a single forward pass, no polyphase decomposition. The
        # numeric modes ('128'/'256'/'512') use the inswapper_128 model
        # with NxN polyphase (dim = swap_size / 128). int() conversion
        # only in the numeric branch since "512-Native" isn't a plain
        # integer; any other/stale value (e.g. a removed mode saved in a
        # marker's parameter snapshot) falls back to the 128 baseline
        # instead of raising on int('256-Native').
        swapper_type = str(parameters['SwapperTypeTextSel'])
        use_native_512 = (swapper_type == '512-Native')
        use_native_256 = (swapper_type == '256-Native')
        if use_native_512:
            swap_size = 512
            dim = 1
        elif use_native_256:
            swap_size = 256
            dim = 1
        elif swapper_type.isdigit():
            swap_size = int(swapper_type)
            dim = int(swap_size/128)#1, 2, 4
        else:
            swap_size = 128
            dim = 1
        # inswapper variants (128 / 256-native / 512-native) all consume the
        # emap-projected latent.
        latent_mode = 'emap'

        # Calculate Pipeline size
        # find largeest arcface pairwise distance
        pdist = np.square(kps - kps[:, None])
        pdist2 = (pdist[:,:,0] + pdist[:,:,1])

        lmk1 = int(pdist2.argmax()/4)
        lmk2 = pdist2.argmax() % 4

        scale = np.linalg.norm(kps[lmk1] - kps[lmk2])/np.linalg.norm(self.arcface_dst[lmk1] - self.arcface_dst[lmk2])
        pipeline_size = int(scale*128)

        # Floor at 128 (the smallest swapper input) and cap at swap_size.
        # The cap is the load-bearing one: when a face fills a big chunk
        # of the source frame, scale*128 can balloon to 800-1500. The
        # swapper itself never sees more than swap_size (128/256/512), so
        # everything beyond swap_size is just bilinear/bicubic stretching
        # of the model's actual detail bandwidth — bigger pipeline buffers
        # cost O(pipeline_size^2) per frame in the input affine, the
        # swap upsample (line ~1083), the mask resize (line ~1193), and
        # the restorer crop (line ~1740) without adding any real fidelity.
        # Capping moves the unavoidable upsample to a single bilinear pass
        # in the inverse-affine paste-back at line ~1221.
        if pipeline_size < 128:
            pipeline_size = 128
        elif pipeline_size > swap_size:
            pipeline_size = swap_size
        pipeline_dim = pipeline_size/128.0

        # transforms
        dst = self.arcface_dst * pipeline_dim
        dst[:,0] += 8.0*pipeline_dim

        # Change the ref points
        if parameters['FaceAdjSwitch']:
            dst[:,0] += parameters['KPSXSlider']
            dst[:,1] += parameters['KPSYSlider']
            dst[:,0] -= (swap_size-1)
            dst[:,0] *= (1+parameters['KPSScaleSlider']/100)
            dst[:,0] += (swap_size+1)
            dst[:,1] -= (swap_size-1)
            dst[:,1] *= (1+parameters['KPSScaleSlider']/100)
            dst[:,1] += (swap_size+1)

        tform = trans.SimilarityTransform.from_estimate(kps, dst)

        # Grab the aligned face crop at pipeline_size in a single
        # grid_sample call instead of warp-the-whole-frame + crop. The
        # old v2.functional.affine path computed every pixel of the
        # input frame (~6M pixels at 1080p) and then v2.functional.crop
        # discarded all but the corner pipeline_size x pipeline_size
        # region. grid_sample with a pipeline_size-sized grid only
        # samples the pixels we keep. tform.inverse.params is the
        # pixel-space matrix mapping output (pipeline coords) → input
        # (img coords), which is exactly what _warp_grid_sample needs.
        with nvtx_range("sc_input_warp"):
            pipeline_input_face = self._warp_grid_sample(
                img,
                np.asarray(tform.inverse.params)[:2, :],
                (pipeline_size, pipeline_size),
            )
            # Cast back to uint8 to match the dtype downstream consumers
            # (apply_occlusion etc.) and the MaskView display branch
            # expect; grid_sample returns float32 because that's what it
            # works in internally.
            pipeline_input_face = pipeline_input_face.to(torch.uint8)

        # Likeness / Distinctiveness tuning — latent shaping over the
        # raw inswapper input. Both default to neutral (1.0 / 0.0).
        latent_scale = float(parameters.get('LikenessSlider', 100)) / 100.0
        extrap_amount = float(parameters.get('EmbExtrapSlider', 0)) / 100.0
        # High Fidelity Cached mode: if this slot already has a measured
        # identity gap, fold it into s_e here so the first (and only)
        # pass uses the corrected embedding. Bootstrap (no cached gap)
        # falls through to the 2-pass block below, which measures the
        # gap from this frame's swap output and persists it.
        hf_switch_on = bool(parameters.get('HighFidelitySwitch'))
        hf_mode = str(parameters.get('HighFidelityModeTextSel', 'Cached'))
        hf_alpha = float(parameters.get('HighFidelityAlphaSlider', 50)) / 100.0
        cached_gap = slot.get('HFCorrectionGap') if slot is not None else None
        # HFRefinePending forces a re-measurement on this swap even when
        # a cached gap exists; the 2-pass block below then blends the
        # new measurement into the cache via running mean. The flag is
        # cleared after the merge so the next frame returns to the
        # single-pass cached path.
        hf_refine_pending = bool(slot.get('HFRefinePending')) if slot is not None else False
        hf_cached_hit = (
            hf_switch_on and hf_mode == 'Cached'
            and cached_gap is not None and hf_alpha > 0.0
            and not hf_refine_pending
        )
        with nvtx_range("sc_latent"):
            s_e_for_latent = s_e
            if hf_cached_hit:
                s_e_arr = np.asarray(s_e, dtype=np.float32).reshape(-1)
                s_e_for_latent = s_e_arr + hf_alpha * cached_gap
            if precomputed_latent is not None:
                latent = precomputed_latent
            else:
                latent = self._get_latent_for(
                    s_e_for_latent, scale=latent_scale, extrap_amount=extrap_amount,
                    latent_mode=latent_mode,
                )
        # Bind without cloning. Both branches below (FaceAdjSwitch affine
        # and the unconditional Resize) return fresh tensors, so the
        # original pipeline_input_face is never mutated in place.
        swap_face_input = pipeline_input_face

        # Optional Scaling # change the thransform matrix
        if parameters['FaceAdjSwitch']:
            swap_face_input = v2.functional.affine(swap_face_input, 0, (0, 0), 1 + parameters['FaceScaleSlider'] / 100, 0, center=(pipeline_dim*128-1, pipeline_dim*128-1), interpolation=v2.InterpolationMode.BILINEAR)

        itex = 1
        if parameters['StrengthSwitch']:
            itex = ceil(parameters['StrengthSlider'] / 100.)

        # output_size = int(128 * dim)  -- unused; swap_face_output is
        # always reassigned by poly_pass(...) below before any read.
        # Use the functional form — constructing v2.Resize() per call
        # allocates a transform object, which adds up across the per-
        # face loop. v2.functional.resize is the underlying call the
        # class wraps. Same arg semantics, same output.
        swap_face_input = v2.functional.resize(
            swap_face_input,
            [swap_size, swap_size],
            interpolation=v2.InterpolationMode.BILINEAR,
            antialias=False,
        )
        swap_face_input = swap_face_input.permute(1, 2, 0)
        swap_face_input = torch.div(swap_face_input, 255.0)

        # Dispatch one polyphase pass per strength iteration. The pass
        # implementation is owned by _polyphase_pass_v1 (production),
        # _native_512_pass (single-shot inswapper_512, selected via
        # "512-Native"), or _native_256_pass (single-shot
        # inswapper_256_phase1, selected via "256-Native").
        if use_native_512:
            poly_pass = self._native_512_pass
        elif use_native_256:
            poly_pass = self._native_256_pass
        else:
            poly_pass = self._polyphase_pass_v1

        for k in range(itex):
            with nvtx_range(f"sc_polypass[k={k}]"):
                swap_face_output = poly_pass(swap_face_input, latent, dim, swap_size, parameters,)
                # Track current input as prev_face by reference — no clone.
                # poly_pass writes into a per-thread cached buffer; nothing
                # downstream mutates swap_face_input in place, so a reference
                # is sufficient. (Old code cloned defensively; redundant.)
                prev_face = swap_face_input
                # Only clone the output to feed it back as the next iter's
                # input; otherwise the next poly_pass call would .zero_() the
                # cached buffer and wipe the data we want to swap from. On the
                # final iter there's no next pass, so skip the clone.
                if k < itex - 1:
                    swap_face_input = swap_face_output.clone()
                # mul/clamp are out-of-place: swap_face_output rebinds to a
                # fresh tensor, leaving the cached buffer untouched for reuse.
                swap_face_output = torch.mul(swap_face_output, 255)
                swap_face_output = torch.clamp(swap_face_output, 0, 255)

        # ====== High Fidelity 2-pass refinement ==========================
        # Measure the swap output's identity via ArcFace, compute a
        # corrected source embedding that pushes back toward s_e, and
        # re-run the swap once.
        #
        # Two modes:
        #   Per-frame: measure + re-run every frame (doubles per-face cost).
        #   Cached:    measure once per slot, persist the gap on the slot,
        #              and skip this block on subsequent frames — the
        #              correction was already folded into the first-pass
        #              latent above. Only the bootstrap frame is 2-pass.
        # Cached hits are gated out here so we don't pay the second swap.
        if hf_switch_on and not hf_cached_hit and hf_alpha > 0.0:
          with nvtx_range("sc_hf_refine"):
            # swap_face_output is HWC 0-255 float; arcface wants CHW uint8.
            chip = swap_face_output.permute(2, 0, 1).to(torch.uint8)
            try:
                output_emb = self.models.embed_face_chip(chip)
            except Exception as e:
                output_emb = None
                print(f'[swap_core] HighFidelity arcface failed: {e}')
            if output_emb is not None:
                # Correct in the same space as the original s_e
                # (no normalization here — calc_swapper_latent
                # handles that). alpha=0 → no correction; alpha=1
                # → full one-step correction.
                s_e_arr = np.asarray(s_e, dtype=np.float32).reshape(-1)
                out_arr = np.asarray(output_emb, dtype=np.float32).reshape(-1)
                gap = s_e_arr - out_arr
                # Cached mode: persist the gap on the slot so subsequent
                # frames take the single-pass cached path above. When a
                # gap already exists (refine path), blend the new sample
                # into the cache via running mean — multiple refine
                # clicks across different frames converge the gap to
                # the source-identity bias's true mean, reducing the
                # per-frame measurement noise that drove this in the
                # first place.
                if hf_mode == 'Cached' and slot is not None:
                    prev = slot.get('HFCorrectionGap')
                    n_prev = int(slot.get('HFCorrectionGapSamples', 0) or 0)
                    if prev is None or n_prev <= 0:
                        merged_gap = gap
                        merged_n = 1
                    else:
                        merged_gap = (n_prev * prev + gap) / (n_prev + 1)
                        merged_gap = merged_gap.astype(np.float32, copy=False)
                        merged_n = n_prev + 1
                    slot['HFCorrectionGap'] = merged_gap
                    slot['HFCorrectionGapSamples'] = merged_n
                    slot['HFRefinePending'] = False
                    gap = merged_gap
                s_e_corrected = s_e_arr + hf_alpha * gap
                latent2_np = self.models.calc_swapper_latent(
                    s_e_corrected,
                    scale=latent_scale,
                    extrap_amount=extrap_amount,
                    latent_mode=latent_mode,
                )
                latent2 = torch.from_numpy(latent2_np).float().to('cuda')
                # Re-run the same iteration pipeline once with the
                # corrected latent. Don't re-loop StrengthSwitch
                # iterations — those compound strength, not identity.
                swap_face_output = poly_pass(
                    swap_face_input, latent2, dim, swap_size, parameters,
                )
                swap_face_output = torch.mul(swap_face_output, 255)
                swap_face_output = torch.clamp(swap_face_output, 0, 255)

        swap = swap_face_output.permute(2, 0, 1)

        # Debug dump of the per-frame swap output. Pipeline tensors are RGB;
        # cv2 expects BGR, so flip the channel order before writing.
        # cv2.imwrite('out.jpg', cv2.cvtColor(swap.permute(1, 2, 0).cpu().numpy(), cv2.COLOR_RGB2BGR))

        if parameters['StrengthSwitch']:
            if itex == 0:
                swap = pipeline_input_face.clone()
            else:
                alpha = np.mod(parameters['StrengthSlider'], 100)*0.01
                if alpha==0:
                    alpha=1

                # Blend the images
                prev_face = torch.mul(prev_face, 255)
                prev_face = torch.clamp(prev_face, 0, 255)
                prev_face = prev_face.permute(2, 0, 1)

                swap = torch.mul(swap, alpha)
                prev_face = torch.mul(prev_face, 1-alpha)
                swap = torch.add(swap, prev_face)
        # Resize in float32 without the antialias low-pass, then quantize.
        # Previously the cast to uint8 + antialias=True softened the swap
        # (float -> rounded ints before interpolation, plus a Gaussian-ish
        # low-pass baked into the downsample path).
        with nvtx_range("sc_resize_to_pipeline"):
            swap = v2.Resize(
                (pipeline_size, pipeline_size),
                interpolation=v2.InterpolationMode.BICUBIC,
                antialias=False,
            )(swap.type(torch.float32))
            swap = swap.clamp(0, 255).type(torch.uint8)

        # LAB color transfer on the GPU: shift swap's per-channel LAB
        # mean/std to match the pre-swap target crop. No device->host
        # bounces, no cv2 — same Reinhard transform as before, but the
        # toggle stops causing the one-time cv2/torch warmup stutter.
        if parameters['ColorMatchSwitch']:
          with nvtx_range("sc_color_match"):
            swap_lab = _rgb_chw_to_lab(swap)
            tgt_lab = _rgb_chw_to_lab(pipeline_input_face)
            s_mean = swap_lab.mean(dim=(1, 2), keepdim=True)
            s_std = swap_lab.std(dim=(1, 2), keepdim=True) + 1e-6
            t_mean = tgt_lab.mean(dim=(1, 2), keepdim=True)
            t_std = tgt_lab.std(dim=(1, 2), keepdim=True)
            swap = _lab_to_rgb_chw_uint8(
                (swap_lab - s_mean) / s_std * t_std + t_mean
            )

        # Apply color corerctions
        if parameters['ColorSwitch']:
            swap = v2.functional.adjust_contrast(swap, parameters['ColorContrastSlider'])
            swap = v2.functional.adjust_gamma(swap, parameters['ColorGammaSlider'], 1.0)
            swap = v2.functional.adjust_saturation(swap, parameters['ColorSaturationSlider'])

            swap = swap.permute(1, 2, 0).type(torch.float32)
            del_color = torch.tensor([parameters['ColorRedSlider'], parameters['ColorGreenSlider'], parameters['ColorBlueSlider']], device=device)
            swap += del_color
            swap = torch.clamp(swap, min=0, max=255)
            swap = swap.permute(2, 0, 1)

        with nvtx_range("sc_mask_compose"):
            # Create border mask. Per-thread reusable (1, 128, 128) buffer; we
            # reset to ones each frame and stamp the four border bands to zero.
            border_mask = self._get_scratch('border_mask', (1, 128, 128)).fill_(1.0)

            # if parameters['BorderState']:
            top = parameters['BorderTopSlider']
            left = parameters['BorderSidesSlider']
            right = 128-parameters['BorderSidesSlider']
            bottom = 128-parameters['BorderBottomSlider']

            border_mask[:, :top, :] = 0
            border_mask[:, bottom:, :] = 0
            border_mask[:, :, :left] = 0
            border_mask[:, :, right:] = 0

            border_mask = self._get_gaussian_blur( parameters['BorderBlurSlider']*2+1, (parameters['BorderBlurSlider']+1)*0.2,)(border_mask)

            # Create image mask. Per-thread reusable (1, 128, 128) buffer reset
            # to ones each frame; downstream multiplies (Occluder/FaceParser/
            # Diff) rebind to fresh tensors so the cached storage is only
            # used as the starting value.
            swap_mask = self._get_scratch('swap_mask', (1, 128, 128)).fill_(1.0)

            # Face Diffing
            if parameters["DiffSwitch"]:
                mask = self.apply_fake_diff(swap, pipeline_input_face, parameters["DiffSlider"])
                mask = self._get_gaussian_blur( parameters['BlendSlider']*2+1, (parameters['BlendSlider']+1)*0.2, )(mask.type(torch.float32))
                swap = swap*mask + pipeline_input_face*(1-mask)

            # Restorer
            if parameters["RestorerSwitch"]:
                swap = self.apply_restorer(swap, parameters)

            # Occluder
            if parameters["OccluderSwitch"]:
                mask = self.apply_occlusion(pipeline_input_face, parameters["OccluderSlider"])
                mask = v2.Resize((128, 128))(mask)
                swap_mask = torch.mul(swap_mask, mask)

            # DeepFaceLab XSeg (user-supplied occlusion model)
            if parameters["DFLXSegSwitch"]:
                mask = self.apply_dfl_xseg(pipeline_input_face, parameters["DFLXSegSizeSlider"])
                mask = v2.Resize((128, 128))(mask)
                if parameters["DFLXSegBlurSlider"] > 0:
                    mask = self._get_gaussian_blur(
                        parameters['DFLXSegBlurSlider']*2+1,
                        (parameters['DFLXSegBlurSlider']+1)*0.2,
                    )(mask)
                swap_mask = torch.mul(swap_mask, mask)


            if parameters["FaceParserSwitch"]:
                mask = self.apply_face_parser(swap, parameters["FaceParserSlider"], parameters['MouthParserSlider'])
                mask = v2.Resize((128, 128))(mask)
                swap_mask = torch.mul(swap_mask, mask)

            # Add blur to swap_mask results
            swap_mask = self._get_gaussian_blur(
                parameters['BlendSlider']*2+1,
                (parameters['BlendSlider']+1)*0.2,
            )(swap_mask)


            # Combine border and swap mask, scale, and apply to swap
            swap_mask = torch.mul(swap_mask, border_mask)
            # swap_mask = t512(swap_mask)
            swap_mask = v2.Resize((pipeline_size, pipeline_size))(swap_mask)
            swap = torch.mul(swap, swap_mask)

        if not control['MaskViewButton']:
          with nvtx_range("paste_back"):
            # Cslculate the area to be mergerd back to the original frame
            tform = trans.SimilarityTransform.from_estimate(kps, dst)
            IM512 = tform.inverse.params[0:2, :]
            corners = np.array([[0,0], [0,pipeline_size-1], [pipeline_size-1, 0], [pipeline_size-1, pipeline_size-1]])

            x = (IM512[0][0]*corners[:,0] + IM512[0][1]*corners[:,1] + IM512[0][2])
            y = (IM512[1][0]*corners[:,0] + IM512[1][1]*corners[:,1] + IM512[1][2])
            
            left = floor(np.min(x))
            if left<0:
                left=0
            top = floor(np.min(y))
            if top<0: 
                top=0
            right = ceil(np.max(x))
            if right>img.shape[2]:
                right=img.shape[2]            
            bottom = ceil(np.max(y))
            if bottom>img.shape[1]:
                bottom=img.shape[1]   

            # Inverse-warp swap + mask back onto the img canvas with
            # one grid_sample call sized to the destination bbox only.
            # Previous path padded swap to full-frame, warped the
            # full-frame, then cropped to bbox — most of the affine
            # output was thrown away. grid_sample samples directly at
            # the bbox-sized output. Stack swap (3 chan) + mask (1 chan)
            # so one warp handles both at the cost of an extra channel.
            # tform.params is the pixel-space matrix mapping img → swap,
            # which is exactly the "for each output, where to sample
            # from input" matrix grid_sample needs. Bake the (left, top)
            # offset into the matrix's translate column so the helper
            # can use plain output pixel coords [0, bbox_w/h).
            stacked = torch.cat((swap, swap_mask), dim=0)
            bbox_h = bottom - top
            bbox_w = right - left
            if bbox_h > 0 and bbox_w > 0:
                T = np.asarray(tform.params)[:2, :].astype(np.float32, copy=True)
                T[:, 2] = T[:, :2] @ np.array([left, top], dtype=np.float32) + T[:, 2]
                sampled = self._warp_grid_sample(
                    stacked, T, (bbox_h, bbox_w),
                )
                swap = sampled[0:3].permute(1, 2, 0)
                swap_mask = sampled[3:4].permute(1, 2, 0)
            else:
                # Degenerate bbox (face entirely off-canvas after
                # clamping). Skip the warp; downstream composite is
                # a no-op anyway because torch.add with empty tensors
                # would error. Set swap/swap_mask to empty so the
                # in-place img assignment below is also no-op.
                swap = torch.empty(
                    (0, 0, 3), dtype=torch.float32, device='cuda',
                )
                swap_mask = torch.empty(
                    (0, 0, 1), dtype=torch.float32, device='cuda',
                )
            swap_mask = torch.sub(1, swap_mask)

            # Apply the mask to the original image areas
            img_crop = img[0:3, top:bottom, left:right]
            img_crop = img_crop.permute(1,2,0)            
            img_crop = torch.mul(swap_mask,img_crop)
            
            #Add the cropped areas and place them back into the original image
            swap = torch.add(swap, img_crop)
            swap = swap.type(torch.uint8)
            swap = swap.permute(2,0,1)
            img[0:3, top:bottom, left:right] = swap  

        else:
            # Invert swap mask
            swap_mask = torch.sub(1, swap_mask)
            
            # Combine preswapped face with swap
            pipeline_input_face = torch.mul(swap_mask, pipeline_input_face)
            pipeline_input_face = torch.add(swap, pipeline_input_face)
            pipeline_input_face = pipeline_input_face.type(torch.uint8)
            pipeline_input_face = pipeline_input_face.permute(1, 2, 0)

            # Uninvert and create image from swap mask
            swap_mask = torch.sub(1, swap_mask) 
            swap_mask = torch.cat((swap_mask,swap_mask,swap_mask),0)
            swap_mask = swap_mask.permute(1, 2, 0)

            # Place them side by side
            img = torch.hstack([pipeline_input_face, swap_mask*255])
            img = img.permute(2,0,1)

        return img

    # ------------------------------------------------------------------
    # Polyphase passes
    #
    # Each function implements one decimate → swap-per-phase → recombine
    # cycle. Input is the aligned face in (swap_size, swap_size, 3) layout,
    # float32 in [0, 1] on cuda. Output is the recombined image in the same
    # layout/range; the caller applies *255 / clamp / strength-blend.
    # ------------------------------------------------------------------

    def _native_512_pass(self, swap_face_input, latent, dim, swap_size, parameters):
        """Direct single-tile swap using the 512x512-native inswapper.
        No polyphase decomposition: feed the model the full
        (512, 512, 3) face crop and take its (512, 512, 3) output as-is.
        dim is always 1 in this path."""
        del dim, parameters  # signature parity with polyphase passes; unused here.

        # Per-thread scratch — same key as v1 because the output shape /
        # purpose are identical, so the two paths can share the buffer.
        swap_face_output = self._get_scratch(
            f'poly_out_{swap_size}', (swap_size, swap_size, 3),
        ).zero_()

        # The inswapper_512 model expects (1, 3, 512, 512). swap_face_input
        # is (H, W, 3) at swap_size; permute + unsqueeze into a contiguous
        # scratch slot. Embedding scratch matches the 1x512 contract of
        # run_swapper_512's `source` binding.
        batch_in = self._get_scratch(
            f'native512_in_{swap_size}', (1, 3, swap_size, swap_size),
        )
        batch_out = self._get_scratch(
            f'native512_out_{swap_size}', (1, 3, swap_size, swap_size),
        )
        emb_in = self._get_scratch('native512_emb', (1, 512))

        batch_in[0].copy_(swap_face_input.permute(2, 0, 1))
        # latent is (1, 512) on cuda from calc_swapper_latent / _get_latent_for.
        emb_in.copy_(latent)

        self.models.run_swapper_512(batch_in, emb_in, batch_out)

        # (1, 3, H, W) -> (H, W, 3) into the output buffer the caller expects.
        swap_face_output[:] = batch_out[0].permute(1, 2, 0)
        return swap_face_output

    def _native_256_pass(self, swap_face_input, latent, dim, swap_size, parameters):
        """Direct single-tile swap using the native inswapper_256_phase1
        model. No polyphase decomposition: feed the full (256, 256, 3) face
        crop and take its (256, 256, 3) output as-is. dim is always 1."""
        del dim, parameters  # signature parity with polyphase passes; unused here.

        # Per-thread scratch — shares the poly_out_{swap_size} key with v1
        # since the output shape / purpose are identical.
        swap_face_output = self._get_scratch(
            f'poly_out_{swap_size}', (swap_size, swap_size, 3),
        ).zero_()

        # inswapper_256_phase1 expects (1, 3, 256, 256); swap_face_input is
        # (H, W, 3) at swap_size. Embedding scratch matches run_swapper_256's
        # 1x512 `source` binding.
        batch_in = self._get_scratch(
            f'native256_in_{swap_size}', (1, 3, swap_size, swap_size),
        )
        batch_out = self._get_scratch(
            f'native256_out_{swap_size}', (1, 3, swap_size, swap_size),
        )
        emb_in = self._get_scratch('native256_emb', (1, 512))

        batch_in[0].copy_(swap_face_input.permute(2, 0, 1))
        # latent is (1, 512) on cuda from calc_swapper_latent / _get_latent_for.
        emb_in.copy_(latent)

        self.models.run_swapper_256(batch_in, emb_in, batch_out)

        # (1, 3, H, W) -> (H, W, 3) into the output buffer the caller expects.
        swap_face_output[:] = batch_out[0].permute(1, 2, 0)
        return swap_face_output

    # @profile
    def _polyphase_pass_v1(self, swap_face_input, latent, dim, swap_size, parameters):
        """Production polyphase pass. For each phase offset (j, i) in dim×dim
        the input is stride-subsampled, run through the swapper at 128×128,
        and interleaved back."""
        # Per-thread reusable scratch keyed by swap_size so multiple
        # pipeline_dim values coexist without realloc. No .zero_() — the
        # vectorized interleave below writes every output element
        # unconditionally, so pre-zeroing is dead defensive work.
        swap_face_output = self._get_scratch(
            f'poly_out_{swap_size}', (swap_size, swap_size, 3),
        )

        # Batched polyphase: stack all dim^2 stride-decimated phases into a
        # single (N, 3, 128, 128) tensor and run one ONNX call instead of N.
        # Saves N-1 Python/ONNX session round-trips per frame; biggest win
        # at dim=4 where N=16. Per-thread scratch buffers keyed by dim.
        n_phases = dim * dim
        batch_in = self._get_scratch(f'batch_in_d{dim}', (n_phases, 3, 128, 128))
        batch_out = self._get_scratch(f'batch_out_d{dim}', (n_phases, 3, 128, 128))

        # Vectorized stride-decimation. Equivalent to:
        #   for j, i in product(range(dim), range(dim)):
        #     batch_in[j*dim+i].copy_(swap_face_input[j::dim, i::dim].permute(2,0,1))
        # but in one strided→contiguous copy instead of dim*dim kernels.
        # The view (128, dim, 128, dim, 3) splits each H/W axis into
        # (inner=128, outer=dim) so view5d[:, j, :, i, :] is exactly the
        # phase tile [j::dim, i::dim]. Permute (1, 3, 4, 0, 2) reorders
        # to (j, i, c, row, col) which lines up with batch_in viewed as
        # (dim, dim, 3, 128, 128). swap_face_input.contiguous() is
        # required because permute+div upstream can leave the strides
        # non-canonical, and .view() refuses non-contiguous input;
        # already-contiguous tensors short-circuit it to self.
        with nvtx_range(f"poly_assemble[d={dim}]"):
            swap_face_input = swap_face_input.contiguous()
            batch_in.view(dim, dim, 3, 128, 128).copy_(
                swap_face_input.view(128, dim, 128, dim, 3).permute(1, 3, 4, 0, 2)
            )

        with nvtx_range(f"poly_swap[d={dim}]"):
            # Source embedding is shared across all polyphase phases — the
            # inswapper source binding is always (1, 512) and the model
            # broadcasts internally. Pass latent as-is instead of expanding
            # to (n_phases, 512); the expand-and-copy was binding-time
            # padding that the model never read.
            self.models.run_swapper_batched(batch_in, latent, batch_out)

        with nvtx_range(f"poly_interleave[d={dim}]"):
            # Vectorized interleave-back: inverse of the assemble permute.
            # batch_out viewed as (dim, dim, 3, 128, 128) → permute
            # (3, 0, 4, 1, 2) lands at (row, j, col, i, c) which is exactly
            # swap_face_output.view(128, dim, 128, dim, 3). One copy instead
            # of dim*dim per-phase writes.
            swap_face_output.view(128, dim, 128, dim, 3).copy_(
                batch_out.view(dim, dim, 3, 128, 128).permute(3, 0, 4, 1, 2)
            )

        return swap_face_output

    # @profile
    def apply_occlusion(self, img, amount):
        with nvtx_range("apply_occlusion"):
            return self._apply_occlusion_inner(img, amount)

    def _apply_occlusion_inner(self, img, amount):
        img = v2.Resize((256, 256), interpolation=v2.InterpolationMode.BICUBIC, antialias=True)(img)
        # img = torch.mul(img, 255.0)
        # img = img.permute(2, 0, 1)

        img = torch.div(img, 255)
        img = torch.unsqueeze(img, 0)

        outpred = torch.ones((256,256), dtype=torch.float32, device=device).contiguous()
        
        self.models.run_occluder(img, outpred)        
                
        outpred = torch.squeeze(outpred)
        outpred = (outpred > 0)
        outpred = torch.unsqueeze(outpred, 0).type(torch.float32)

        if amount >0:                   
            kernel = torch.ones((1,1,3,3), dtype=torch.float32, device=device)

            for i in range(int(amount)):
                outpred = torch.nn.functional.conv2d(outpred, kernel, padding=(1, 1))       
                outpred = torch.clamp(outpred, 0, 1)
            
            outpred = torch.squeeze(outpred)
            
        if amount <0:      
            outpred = torch.neg(outpred)
            outpred = torch.add(outpred, 1)
            kernel = torch.ones((1,1,3,3), dtype=torch.float32, device=device)

            for i in range(int(-amount)):
                outpred = torch.nn.functional.conv2d(outpred, kernel, padding=(1, 1))       
                outpred = torch.clamp(outpred, 0, 1)
            
            outpred = torch.squeeze(outpred)
            outpred = torch.neg(outpred)
            outpred = torch.add(outpred, 1)

        outpred = torch.reshape(outpred, (1, 256, 256))
        return outpred


    def apply_dfl_xseg(self, img, amount):
        with nvtx_range("apply_dfl_xseg"):
            return self._apply_dfl_xseg_inner(img, amount)

    def _apply_dfl_xseg_inner(self, img, amount):
        # Models.run_dfl_xseg handles resize / BGR / layout and returns a
        # (res, res) soft mask in [0, 1] where 1 = face region (keep swap).
        outpred = self.models.run_dfl_xseg(img)
        res = outpred.shape[-1]

        # Threshold the soft XSeg mask to a binary region so the grow/shrink
        # dilation below has clean edges to push around (mirrors Occluder).
        outpred = (outpred > 0.5)
        outpred = torch.unsqueeze(outpred, 0).type(torch.float32)

        if amount > 0:
            kernel = torch.ones((1, 1, 3, 3), dtype=torch.float32, device=device)

            for i in range(int(amount)):
                outpred = torch.nn.functional.conv2d(outpred, kernel, padding=(1, 1))
                outpred = torch.clamp(outpred, 0, 1)

            outpred = torch.squeeze(outpred)

        if amount < 0:
            outpred = torch.neg(outpred)
            outpred = torch.add(outpred, 1)
            kernel = torch.ones((1, 1, 3, 3), dtype=torch.float32, device=device)

            for i in range(int(-amount)):
                outpred = torch.nn.functional.conv2d(outpred, kernel, padding=(1, 1))
                outpred = torch.clamp(outpred, 0, 1)

            outpred = torch.squeeze(outpred)
            outpred = torch.neg(outpred)
            outpred = torch.add(outpred, 1)

        outpred = torch.reshape(outpred, (1, res, res))
        return outpred


    # # @profile
    def apply_face_parser(self, img, FaceAmount, MouthAmount):
        with nvtx_range("apply_face_parser"):
            return self._apply_face_parser_inner(img, FaceAmount, MouthAmount)

    def _apply_face_parser_inner(self, img, FaceAmount, MouthAmount):

        # atts = [1 'skin', 2 'l_brow', 3 'r_brow', 4 'l_eye', 5 'r_eye', 6 'eye_g', 7 'l_ear', 8 'r_ear', 9 'ear_r', 10 'nose', 11 'mouth', 12 'u_lip', 13 'l_lip', 14 'neck', 15 'neck_l', 16 'cloth', 17 'hair', 18 'hat']

        img = v2.Resize((512, 512), interpolation=v2.InterpolationMode.BICUBIC, antialias=True)(img)
        img = torch.div(img, 255)
        img = v2.functional.normalize(img, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        img = torch.reshape(img, (1, 3, 512, 512)).contiguous()
        outpred = torch.empty((1,19,512,512), dtype=torch.float32, device='cuda').contiguous()

        self.models.run_faceparser(img, outpred)

        outpred = torch.squeeze(outpred)
        outpred = torch.argmax(outpred, 0)

        # Mouth Parse
        if MouthAmount <0:
            mouth_idxs = torch.tensor([11], device='cuda')
            iters = int(-MouthAmount)

            mouth_parse = torch.isin(outpred, mouth_idxs)
            mouth_parse = torch.clamp(~mouth_parse, 0, 1).type(torch.float32)
            mouth_parse = torch.reshape(mouth_parse, (1, 1, 512, 512))
            mouth_parse = torch.neg(mouth_parse)
            mouth_parse = torch.add(mouth_parse, 1)

            kernel = torch.ones((1, 1, 3, 3), dtype=torch.float32, device='cuda')

            for i in range(iters):
                mouth_parse = torch.nn.functional.conv2d(mouth_parse, kernel, padding=(1, 1))
                mouth_parse = torch.clamp(mouth_parse, 0, 1)

            mouth_parse = torch.squeeze(mouth_parse)
            mouth_parse = torch.neg(mouth_parse)
            mouth_parse = torch.add(mouth_parse, 1)
            mouth_parse = torch.reshape(mouth_parse, (1, 512, 512))

        elif MouthAmount >0:
            mouth_idxs = torch.tensor([11,12,13], device='cuda')
            iters = int(MouthAmount)

            mouth_parse = torch.isin(outpred, mouth_idxs)
            mouth_parse = torch.clamp(~mouth_parse, 0, 1).type(torch.float32)
            mouth_parse = torch.reshape(mouth_parse, (1,1,512,512))
            mouth_parse = torch.neg(mouth_parse)
            mouth_parse = torch.add(mouth_parse, 1)

            kernel = torch.ones((1,1,3,3), dtype=torch.float32, device='cuda')

            for i in range(iters):
                mouth_parse = torch.nn.functional.conv2d(mouth_parse, kernel, padding=(1, 1))
                mouth_parse = torch.clamp(mouth_parse, 0, 1)

            mouth_parse = torch.squeeze(mouth_parse)
            mouth_parse = torch.neg(mouth_parse)
            mouth_parse = torch.add(mouth_parse, 1)
            mouth_parse = torch.reshape(mouth_parse, (1, 512, 512))

        else:
            mouth_parse = torch.ones((1, 512, 512), dtype=torch.float32, device='cuda')

        # BG Parse
        bg_idxs = torch.tensor([0, 14, 15, 16, 17, 18], device=device)
        bg_parse = torch.isin(outpred, bg_idxs)
        bg_parse = torch.clamp(~bg_parse, 0, 1).type(torch.float32)
        bg_parse = torch.reshape(bg_parse, (1, 512, 512))



        if FaceAmount > 0:
            kernel = torch.ones((1, 1, 3, 3), dtype=torch.float32, device=device)

            for i in range(int(FaceAmount)):
                bg_parse = torch.nn.functional.conv2d(bg_parse, kernel, padding=(1, 1))
                bg_parse = torch.clamp(bg_parse, 0, 1)

            bg_parse = torch.squeeze(bg_parse)

        elif FaceAmount < 0:
            bg_parse = torch.neg(bg_parse)
            bg_parse = torch.add(bg_parse, 1)
            kernel = torch.ones((1, 1, 3, 3), dtype=torch.float32, device=device)

            for i in range(int(-FaceAmount)):
                bg_parse = torch.nn.functional.conv2d(bg_parse, kernel, padding=(1, 1))
                bg_parse = torch.clamp(bg_parse, 0, 1)

            bg_parse = torch.squeeze(bg_parse)
            bg_parse = torch.neg(bg_parse)
            bg_parse = torch.add(bg_parse, 1)
            bg_parse = torch.reshape(bg_parse, (1, 512, 512))
        else:
            bg_parse = torch.ones((1,512,512), dtype=torch.float32, device='cuda')


        # cv2.imwrite('test.jpg', bg_parse.permute(1, 2, 0).cpu().numpy()*255)

        out_parse = torch.mul(bg_parse, mouth_parse)

        return out_parse
        
    

    def apply_restorer(self, swapped_face_upscaled, parameters):
        with nvtx_range("apply_restorer"):
            return self._apply_restorer_inner(swapped_face_upscaled, parameters)

    def _apply_restorer_inner(self, swapped_face_upscaled, parameters):
        face_size = swapped_face_upscaled.shape[1]
        face_dim = face_size / 128.0

        input_face = swapped_face_upscaled.clone()

        # If using a separate detection mode
        transformed = False
        if parameters['RestorerDetTypeTextSel'] == 'Blend' or parameters['RestorerDetTypeTextSel'] == 'Reference':
            transformed = True
            if parameters['RestorerDetTypeTextSel'] == 'Blend':
                dst_init = self.arcface_dst

            elif parameters['RestorerDetTypeTextSel'] == 'Reference':
                try:
                    dst_init = self.models.resnet50(input_face, score=parameters['DetectScoreSlider']/100.0)/4.0
                except:
                    return

            dst = dst_init * face_dim
            dst[:, 0] += 8.0 * face_dim

            f_kps = self.FFHQ_kps * face_dim
            f_kps[:, 0] += 8.0 * face_dim

            tform = trans.SimilarityTransform.from_estimate(dst, f_kps)

            # Transform, scale, and normalize
            input_face = v2.functional.affine(input_face, tform.rotation*57.2958, (tform.translation[0], tform.translation[1]) , tform.scale, 0, center = (0,0) )
            input_face = v2.functional.crop(input_face, 0,0, face_size, face_size)

        input_face = torch.div(input_face, 255)
        input_face = v2.functional.normalize(input_face, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=False)
        input_face = torch.unsqueeze(input_face, 0).contiguous()

        # Run the chosen restorer. We DO NOT resize the model output back
        # to face_size here. The previous code paid O(face_size^2) for a
        # Resize(out_size -> face_size) and then another O(face_size^2)
        # for an in-place inverse affine. The new path collapses those
        # into a single grid_sample below, sampling directly from the
        # model's native output buffer (256 or 512) into a face_size
        # output. For tracking, each branch records `out_size`.
        if parameters['RestorerTypeTextSel'] == 'GPEN256':
            input_face = v2.Resize((256, 256), antialias=False)(input_face)
            outpred = torch.empty((1, 3, 256, 256), dtype=torch.float32, device=device).contiguous()
            self.models.run_GPEN_256(input_face, outpred)
            out_size = 256

        elif parameters['RestorerTypeTextSel'] == 'GFPGAN':
            input_face = v2.Resize((512, 512), antialias=False)(input_face)
            outpred = torch.empty((1, 3, 512, 512), dtype=torch.float32, device=device).contiguous()
            self.models.run_GFPGAN(input_face, outpred)
            out_size = 512

        elif parameters['RestorerTypeTextSel'] == 'CF':
            input_face = v2.Resize((512, 512), antialias=False)(input_face)
            outpred = torch.empty((1, 3, 512, 512), dtype=torch.float32, device=device).contiguous()
            self.models.run_codeformer(input_face, outpred)
            out_size = 512

        elif parameters['RestorerTypeTextSel'] == 'GPEN512':
            input_face = v2.Resize((512, 512), antialias=False)(input_face)
            outpred = torch.empty((1, 3, 512, 512), dtype=torch.float32, device=device).contiguous()
            self.models.run_GPEN_512(input_face, outpred)
            out_size = 512


        # Format back to cxHxW @ 255 — outpred is still at out_size here.
        outpred = torch.squeeze(outpred)
        outpred = torch.clamp(outpred, -1, 1)
        outpred = torch.add(outpred, 1)
        outpred = torch.div(outpred, 2)
        outpred = torch.mul(outpred, 255)

        if transformed:
            # Combined inverse-affine + downsize from out_size to face_size
            # in one grid_sample. The original two-step path was:
            #   Resize(out_size -> face_size) at face_size_aligned
            #   v2.functional.affine(tform.inverse) at face_size_unaligned
            # v2.functional.affine treats its arg as a FORWARD transform
            # and inverts it for backward sampling, so passing tform.inverse
            # there resolves to sampling input at tform.forward(output_pixel).
            # Composing with the uniform (out_size/face_size) scale that
            # the prior Resize introduced gives the backward map our
            # grid_sample helper consumes: face_size_unaligned output ->
            # out_size_aligned input via scale * tform.forward.
            scale_io = out_size / float(face_size)
            T = np.eye(3, dtype=np.float32)
            T[:2, :2] = tform.params[:2, :2] * scale_io
            T[:2, 2] = tform.params[:2, 2] * scale_io
            outpred = self._affine_resize_grid_sample(
                outpred.unsqueeze(0), T,
                in_h=out_size, in_w=out_size,
                out_h=face_size, out_w=face_size,
            ).squeeze(0)
        elif out_size != face_size:
            # No alignment warp requested — still need to land on the
            # face_size pipeline buffer. Single Resize, no affine.
            outpred = v2.Resize((face_size, face_size), antialias=False)(outpred)

        # Blend
        alpha = float(parameters["RestorerSlider"])/100.0
        outpred = torch.add(torch.mul(outpred, alpha), torch.mul(swapped_face_upscaled, 1-alpha))

        return outpred

    def _affine_resize_grid_sample(self, src, tform_pix, in_h, in_w, out_h, out_w):
        """One-shot affine warp + size change via affine_grid + grid_sample.

        `tform_pix` is a 3x3 pixel-space matrix that maps OUTPUT pixel
        coords -> INPUT pixel coords (the standard backward-sampling map
        an affine warper consumes). `src` is (N, C, in_h, in_w); the
        result is (N, C, out_h, out_w). align_corners=True is used to
        match the pixel-center conventions used elsewhere in this file's
        grid math (corners landmark calc at line ~1201 uses inclusive
        endpoints, same convention).

        Used by apply_restorer to fuse [Resize(out_size -> face_size)] and
        [InverseAffine(face_size)] into a single pass at out_h x out_w.
        """
        n, c, _, _ = src.shape
        dev = src.device
        # Convert pixel-space tform to normalized-space theta:
        #   theta_norm = M_in_inv @ T_pix @ M_out
        # where M(W,H) maps norm[-1,1] -> pixel[0, W-1].
        M_out = torch.tensor([
            [(out_w - 1) * 0.5, 0.0, (out_w - 1) * 0.5],
            [0.0, (out_h - 1) * 0.5, (out_h - 1) * 0.5],
            [0.0, 0.0, 1.0],
        ], device=dev, dtype=torch.float32)
        M_in_inv = torch.tensor([
            [2.0 / (in_w - 1), 0.0, -1.0],
            [0.0, 2.0 / (in_h - 1), -1.0],
            [0.0, 0.0, 1.0],
        ], device=dev, dtype=torch.float32)
        T = torch.as_tensor(np.asarray(tform_pix, dtype=np.float32), device=dev)
        theta = (M_in_inv @ T @ M_out)[:2, :].unsqueeze(0).expand(n, -1, -1)
        grid = torch.nn.functional.affine_grid(
            theta, [n, c, out_h, out_w], align_corners=True,
        )
        return torch.nn.functional.grid_sample(
            src, grid, mode='bilinear', padding_mode='zeros', align_corners=True,
        )

    def apply_fake_diff(self, swapped_face, original_face, DiffAmount):
        swapped_face = swapped_face.permute(1,2,0)
        original_face = original_face.permute(1,2,0)

        diff = swapped_face-original_face
        diff = torch.abs(diff)
        
        # Find the diffrence between the swap and original, per channel
        fthresh = DiffAmount*2.55
        
        # Bimodal
        diff[diff<fthresh] = 0
        diff[diff>=fthresh] = 1 
        
        # If any of the channels exceeded the threshhold, them add them to the mask
        diff = torch.sum(diff, dim=2)
        diff = torch.unsqueeze(diff, 2)
        diff[diff>0] = 1
        
        diff = diff.permute(2,0,1)

        return diff    
    

        # cv2.imwrite('ab.jpg', mask.permute(1, 2, 0).cpu().numpy())
