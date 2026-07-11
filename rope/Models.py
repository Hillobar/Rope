
import os
import cv2
import numpy as np
from skimage import transform as trans
import torch
import torchvision
torchvision.disable_beta_transforms_warning()
from torchvision.transforms import v2
from numpy.linalg import norm as l2norm
import onnxruntime
import onnx
from itertools import product as product
import subprocess as sp
import threading

from rope._nvtx import nvtx_range
onnxruntime.set_default_logger_severity(4)


DEFAULT_MODELS_FOLDER = './models'


# Inventory of model files the application can load from the models
# folder. Used by the Settings tab to show which files are present /
# missing so the user can verify a freshly-picked folder is complete.
# Each entry: (filename, human-readable role, required-for-basic-swap?).
# Required = True means a baseline swap won't work without it; optional
# entries are tied to specific features (alternate detector, restorer,
# occlusion masking, etc.).
MODEL_INVENTORY = (
    ('det_10g.onnx',              'RetinaFace detector',                True),
    ('scrfd_2.5g_bnkps.onnx',     'SCRFD detector (alt)',               False),
    ('w600k_r50.onnx',            'ArcFace recognition (r50)',          True),
    ('inswapper_128.fp16.onnx',   'Inswapper 128',                      True),
    ('GFPGANv1.4.onnx',           'GFPGAN v1.4 restorer',               False),
    ('GPEN-BFR-512.onnx',         'GPEN-BFR-512 restorer',              False),
    ('GPEN-BFR-256.onnx',         'GPEN-BFR-256 restorer',              False),
    ('codeformer_fp16.onnx',      'CodeFormer restorer',                False),
    ('occluder.onnx',             'Occluder mask',                      False),
    ('faceparser_resnet34.onnx',  'Face parser (ResNet34)',             False),
    ('dfl_xseg.onnx',             'DeepFaceLab XSeg mask',              False),
    ('res50.onnx',                'ResNet50 detector',                  False),
)


# Per-inventory-row metadata used by the Settings tab to render Loaded /
# TRT columns and the per-model Unload button. Keys are the source ONNX
# filename (same as MODEL_INVENTORY[i][0]).
#
#   models_attr: the Models attribute name set on load; `is_model_loaded`
#                / `unload_model` accept this string. None = the row
#                doesn't correspond to a tracked Models slot, so the row
#                shows no Loaded state or Unload button.
#
#   trt_engine:  filename of the TRT engine built from this ONNX (sits
#                next to the ONNX in the models folder), or None if no
#                TRT path exists. The "Build TensorRT" button covers the
#                identity / detect / recognize set: inswapper_128,
#                inswapper_512, retinaface (det_10g), arcface (w600k_r50).
MODEL_INVENTORY_DETAILS = {
    'det_10g.onnx':              {'models_attr': 'retinaface_model',  'trt_engine': 'det_10g.engine'},
    'scrfd_2.5g_bnkps.onnx':     {'models_attr': 'scrdf_model',       'trt_engine': None},
    'w600k_r50.onnx':            {'models_attr': 'recognition_model', 'trt_engine': 'w600k_r50.engine'},
    'inswapper_128.fp16.onnx':   {'models_attr': 'swapper_model',     'trt_engine': 'inswapper_128.engine'},
    'GFPGANv1.4.onnx':           {'models_attr': 'GFPGAN_model',      'trt_engine': None},
    'GPEN-BFR-512.onnx':         {'models_attr': 'GPEN_512_model',    'trt_engine': None},
    'GPEN-BFR-256.onnx':         {'models_attr': 'GPEN_256_model',    'trt_engine': None},
    'codeformer_fp16.onnx':      {'models_attr': 'codeformer_model',  'trt_engine': None},
    'occluder.onnx':             {'models_attr': 'occluder_model',    'trt_engine': None},
    'faceparser_resnet34.onnx':  {'models_attr': 'faceparser_model',  'trt_engine': None},
    'dfl_xseg.onnx':             {'models_attr': 'dfl_xseg_model',    'trt_engine': None},
    'res50.onnx':                {'models_attr': 'resnet50_model',    'trt_engine': None},
}

class Models():
    # Attribute names whose (re)assignment indicates VRAM has changed.
    # __setattr__ flips vram_dirty whenever any of these is written, so
    # the Qt coordinator can emit vram_updated on the next tick without
    # any call-site instrumentation at the ~15 lazy load points.
    _VRAM_TRACKED_ATTRS = frozenset([
        'retinaface_model', 'scrdf_model', 'resnet50_model',
        'recognition_model', 'swapper_model',
        'swapper_512_model', 'swapper_256_model',
        'GFPGAN_model', 'GPEN_256_model', 'GPEN_512_model', 'codeformer_model',
        'occluder_model', 'faceparser_model', 'dfl_xseg_model',
    ])

    def __setattr__(self, name, value):
        object.__setattr__(self, name, value)
        if name in self._VRAM_TRACKED_ATTRS:
            object.__setattr__(self, 'vram_dirty', True)

    def __init__(self):
        # Models folder — the directory all *.onnx / *.engine / *.pth
        # files are resolved against. Override via set_models_folder()
        # (driven from the Settings tab's "Models Folder" picker).
        self.models_folder = DEFAULT_MODELS_FOLDER

        self.arcface_dst = np.array( [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float32)
        self.providers = [('CUDAExecutionProvider')]
        
        self.retinaface_model = []
        self.scrdf_model = []
        self.resnet50_model, self.anchors  = [], []

        self.recognition_model = []
        self.swapper_model = []
        # inswapper_512: parallel 512x512-in/512x512-out variant. Loaded
        # lazily on first run_swapper_512 call; same source conditioning
        # format as the 128 model, so calc_swapper_latent's `latent`
        # works for both.
        self.swapper_512_model = []
        # inswapper_256_phase1: native 256x256 single-pass variant. Same
        # emap-projected latent as 128/512; loaded lazily on first
        # run_swapper_256 call (only when the "256-Native" mode is chosen).
        self.swapper_256_model = []

        self.emap = []
        self.GFPGAN_model = []
        self.GPEN_256_model = []
        self.GPEN_512_model = []
        self.codeformer_model = []
        
        self.occluder_model = []
        self.faceparser_model = []
        # DeepFaceLab XSeg occlusion mask (user-supplied ONNX). The export
        # variants in the wild differ in tensor layout (NHWC from the TF
        # original vs NCHW from some converters), input/output node names,
        # resolution, and FP16/FP32 I/O. _load_dfl_xseg_model introspects
        # the graph once and stores the contract in these attrs so
        # run_dfl_xseg can bind correctly regardless of the export.
        self.dfl_xseg_model = []
        self._dfl_xseg_in_name = None
        self._dfl_xseg_out_name = None
        self._dfl_xseg_res = 256
        self._dfl_xseg_nhwc = True
        self._dfl_xseg_in_dtype = np.float32
        self._dfl_xseg_out_dtype = np.float32

        self.swapper_model_kps = []
        self.swapper_model_swap = []
        # TRT vs ONNX state. Each "_uses_trt" flag is set by the matching
        # lazy loader (_load_*_model) and consulted by the matching call
        # site. Reset to False by delete_models / unload_model so the next
        # call re-detects whether an engine is present.
        self._retinaface_uses_trt = False
        self._recognition_uses_trt = False
        # Anchor centers used by retinaface post-processing depend only
        # on (height, width, stride), all derived from input_size which
        # changes only when the DetectInputSize slider moves. Cache on
        # self so the meshgrid + reshape + stack work is paid once per
        # (input_size, stride) instead of every frame on every worker.
        # Tensors are kept on GPU; cache miss builds + syncs the device
        # once so other workers' streams see the writes. Dict mutations
        # are GIL-protected — concurrent computes on first call just
        # produce identical results; one write wins.
        self._retinaface_center_cache: dict = {}
        # Per-worker (1, 3, H, W) float32 input buffer for retinaface,
        # keyed by input_size. Pre-allocating skips the torch.zeros call
        # per frame and keeps the (1, 3, H, W) layout end-to-end, so the
        # preprocessing path no longer needs a final permute+contiguous
        # round-trip. Per-worker because workers write concurrently
        # into their own copies.
        self._retinaface_buffers_tls = threading.local()
        # Progress counter for the [N/M] prefix on main-pipeline session
        # load messages. Counter is phase-scoped: model loads arrive in
        # distinct user-driven phases (find = retinaface + arcface, swap
        # = inswapper variant), so the counter resets when the phase
        # changes. M = (models in this phase) × workers (Per-Thread) or
        # × 1 (Shared). Counter overflow is fine — e.g. loading both
        # inswapper_128 and _512 in the same session shows [2/1].
        self._load_lock = threading.Lock()
        self._load_count = 0
        self._load_expected = 0
        self._load_phase = None
        # Worker-count hint from VideoManager / ThreadsSlider. Used to
        # compute M in Per-Thread mode. Defaults to 1 (Shared at start).
        self._workers_hint = 1
        # Per-attribute backend preference. Keys: Models attribute names
        # that have both ONNX and TRT paths ('swapper_model',
        # 'swapper_512_model', 'retinaface_model', 'recognition_model').
        # Values: 'trt' (force TRT), 'onnx' (force ONNX), or absent =
        # auto (current behavior — prefer TRT if engine present).
        # Driven by the Settings-tab Backend toggle and applied at the
        # next lazy load.
        self._backend_pref: dict = {}
        self.syncvec = torch.empty((1,1), dtype=torch.float32, device='cuda:0')

        # Model session mode — controls every ORT-TRT-EP session the
        # main pipeline uses (inswapper, inswapper_512, retinaface,
        # recognition). 'Shared' uses one session per model across all
        # worker threads (low VRAM, but workers serialize on each
        # session's internal state); 'Per-Thread' gives each worker
        # its own session per model for parallel runs at the cost of
        # ~150-300MB extra VRAM per worker per model. The on-disk TRT
        # engine cache is shared across sessions so engine build is
        # paid once.
        self._model_session_mode: str = 'Shared'
        self._sessions_tls = threading.local()
        # name → list of strong refs to every Per-Thread session we
        # hand out. The mode-change / unload path drops these refs so
        # the dying workers' sessions GC properly (TLS alone leaves
        # them pinned because Python's thread-local can't reach into
        # other threads' slots).
        self._sessions_lock = threading.Lock()
        self._all_sessions: dict[str, list] = {}

        # Session-wide mean source-face embedding, set by main_window as
        # source faces get recognized. Used by the Distinctiveness slider
        # (calc_swapper_latent) — None means "no mean yet" and the
        # extrap branch falls through to a no-op.
        self._mean_emb = None

    def _mp(self, name):
        """Resolve a model filename against the configured folder."""
        return os.path.join(self.models_folder, name)

    def set_models_folder(self, path):
        """Point all model lookups at a new folder. Falsy = revert to
        default. Drops every loaded model so the next inference call
        reloads from the new path; also re-points TensorRTEngine's
        inswapper engine constants so TRT picks up the new location."""
        new_folder = path if path else DEFAULT_MODELS_FOLDER
        if new_folder == self.models_folder:
            return
        self.models_folder = new_folder
        try:
            from rope import TensorRTEngine as TRT
            TRT.set_models_folder(new_folder)
        except Exception:
            # TRT is optional — silently skip if it can't be updated.
            pass
        # Force a reload so the next call picks up the new path.
        self.delete_models()

    def get_gpu_memory(self):
        # cudaMemGetInfo via torch — microseconds, no subprocess. The
        # previous nvidia-smi shell-out blocked the GUI event loop for
        # 100-400 ms per call, causing a visible 1 Hz stutter when this
        # was polled by the Qt VRAM indicator.
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        mib = 1024 * 1024
        return (total_bytes - free_bytes) // mib, total_bytes // mib

    def run_detect(self, img, detect_mode='Retinaface', max_num=1, score=0.5, input_size=640):
        """Run face detection. `input_size` is the square model input
        edge (a multiple of 32 — 320 / 416 / 480 / 640). Retinaface
        and SCRDF accept dynamic input shapes so they honor it."""
        kpss = []

        if detect_mode=='Retinaface':
            if not self.retinaface_model:
                self._load_retinaface_model()

            kpss = self.detect_retinaface(img, max_num=max_num, score=score, input_size=input_size)

        elif detect_mode=='SCRDF':
            self._ensure_scrdf_model()

            kpss = self.detect_scrdf(img, max_num=max_num, score=score, input_size=input_size)

        return kpss
        
    # Maps Models attribute name → the short TLS/strong-ref key used by
    # _get_model_session. Per-Thread mode never writes the shared attr,
    # so is_model_loaded has to consult the strong-ref list instead.
    _ATTR_TO_TLS_NAME = {
        'swapper_model':       'swapper',
        'swapper_512_model':   'swapper_512',
        'swapper_256_model':   'swapper_256',
        'retinaface_model':    'retinaface',
        'recognition_model':   'recognition',
    }

    def is_model_loaded(self, attr_name):
        """True if the Models attribute named attr_name currently holds a
        live session/wrapper. Empty list / None / False all read as not
        loaded. In Per-Thread mode the shared attribute is never written
        — sessions live on `_sessions_tls` and in `_all_sessions[name]`
        — so we also check the strong-ref list. Used by the Settings tab
        inventory."""
        val = getattr(self, attr_name, None)
        if val is not None and val is not False:
            if not (isinstance(val, (list, tuple)) and len(val) == 0):
                return True
        # Per-Thread fallback: a session may exist for at least one
        # worker even though the shared attribute is still [].
        tls_name = self._ATTR_TO_TLS_NAME.get(attr_name)
        if tls_name is not None:
            all_sessions = getattr(self, '_all_sessions', None)
            if all_sessions:
                lock = getattr(self, '_sessions_lock', None)
                if lock is not None:
                    with lock:
                        sessions = all_sessions.get(tls_name)
                else:
                    sessions = all_sessions.get(tls_name)
                if sessions:
                    return True
        return False

    def unload_model(self, attr_name):
        """Drop a single loaded model. Mirrors what delete_models() does
        for each attribute (assign back to []/None), letting the user
        free one slot without nuking the rest. Unknown / non-tracked
        attribute names are ignored."""
        if attr_name not in self._VRAM_TRACKED_ATTRS:
            return
        if attr_name == 'swapper_model':
            # Keep the TRT/batch flags in sync with delete_models so the
            # next swap re-detects engine + batch capability.
            self._swapper_uses_trt = False
            self._swapper_batch_unsupported = False
            # Drop any Per-Thread sessions too so a full unload actually
            # frees the VRAM — TLS alone holds strong refs we'd otherwise
            # leak across the unload.
            self._sessions_tls = threading.local()
            self.clear_per_thread_sessions('swapper')
        elif attr_name == 'swapper_512_model':
            self._swapper_512_uses_trt = False
            self._swapper_512_batch_unsupported = False
            self.clear_per_thread_sessions('swapper_512')
        elif attr_name == 'swapper_256_model':
            self._swapper_256_uses_trt = False
            self.clear_per_thread_sessions('swapper_256')
        elif attr_name == 'retinaface_model':
            self._retinaface_uses_trt = False
            self.clear_per_thread_sessions('retinaface')
        elif attr_name == 'recognition_model':
            self._recognition_uses_trt = False
            self.clear_per_thread_sessions('recognition')
        setattr(self, attr_name, [])

    # Attributes whose lazy loader honors _backend_pref. Other models
    # have only an ONNX path so the toggle is irrelevant.
    _BACKEND_PREF_ATTRS = frozenset([
        'swapper_model', 'swapper_512_model', 'swapper_256_model',
        'retinaface_model', 'recognition_model',
    ])

    def set_backend_preference(self, attr_name, backend, *, unload=True):
        """Set the per-model backend preference.

            backend == 'trt':  force TRT path (falls back to ONNX only
                               if engine load fails at runtime).
            backend == 'onnx': force ONNX path even if engine is on disk.
            backend in (None, 'auto'): clear the override — auto-detect
                               at next load (prefer TRT if engine present).

        When unload=True (default) and the model is currently loaded,
        also drop it so the next inference call re-loads with the new
        preference. Unknown attribute names are ignored."""
        if attr_name not in self._BACKEND_PREF_ATTRS:
            return
        if backend in (None, 'auto'):
            self._backend_pref.pop(attr_name, None)
        elif backend in ('trt', 'onnx'):
            self._backend_pref[attr_name] = backend
        else:
            return
        if unload and self.is_model_loaded(attr_name):
            self.unload_model(attr_name)

    def get_backend_preference(self, attr_name):
        """Return 'trt' / 'onnx' / None (auto)."""
        return self._backend_pref.get(attr_name)

    def delete_models(self):

        self.retinaface_model = []
        self.scrdf_model = []
        self.resnet50_model = []
        self.recognition_model = []
        self.swapper_model = []
        self.swapper_512_model = []
        self.swapper_256_model = []
        self.GFPGAN_model = []
        self.GPEN_256_model = []
        self.GPEN_512_model = []
        self.codeformer_model = []
        self.occluder_model = []
        self.faceparser_model = []
        self.dfl_xseg_model = []
        # Drop the TRT/batched-unsupported flags too — the matching
        # lazy loaders will re-detect on next use (so a Build TensorRT
        # done after a Clear VRAM is picked up automatically).
        self._swapper_uses_trt = False
        self._swapper_batch_unsupported = False
        self._swapper_512_uses_trt = False
        self._swapper_512_batch_unsupported = False
        self._swapper_256_uses_trt = False
        self._retinaface_uses_trt = False
        self._recognition_uses_trt = False

    def run_recognize(self, img, kps, dim=1):
        with nvtx_range("run_recognize"):
            if not self.recognition_model:
                self._load_recognition_model()

            embedding, cropped_image = self.recognize(img, kps, dim)
            return embedding, cropped_image
    

    def set_session_mean_embedding(self, mean):
        """Stash a session-wide mean face embedding for the Distinctiveness
        slider's extrapolation. Pass None to clear. Caller should also
        clear the latent cache after changing this (the cache is keyed
        by raw source embedding, not effective embedding)."""
        if mean is None:
            self._mean_emb = None
            return
        m = np.asarray(mean, dtype=np.float32).reshape(-1)
        n = float(np.linalg.norm(m))
        if n > 0:
            m = m / n  # store normalized so extrap_amount has predictable scale
        self._mean_emb = m

    def calc_swapper_latent(self, source_embedding, *, scale=1.0, extrap_amount=0.0, latent_mode='emap'):
        """Compute the swapper conditioning latent. Extra knobs over the
        original Insightface formula:
            extrap_amount: pushes source_embedding away from the session
                mean (set via set_session_mean_embedding) before the
                normalize -> emap -> normalize pipeline. 0 = no change.
            scale: multiplies the final latent. >1 increases identity
                intensity in the swap output.
            latent_mode: 'emap' (default — insightface inswapper:
                normalize -> emap projection -> normalize) or 'raw'
                (just L2-normalize; no emap projection — for a model that
                consumes the raw embedding directly).
        """
        s = np.asarray(source_embedding, dtype=np.float32).reshape(-1)
        mean = getattr(self, '_mean_emb', None)
        if extrap_amount > 0.0 and mean is not None and mean.shape == s.shape:
            # Re-scale mean to match s_e's magnitude before subtraction so
            # the extrapolation is in a consistent space regardless of how
            # the user's embeddings were normalized.
            s_norm = float(np.linalg.norm(s))
            mean_scaled = mean * s_norm
            s = s + float(extrap_amount) * (s - mean_scaled)
        n_e = s / l2norm(s)
        latent = n_e.reshape((1, -1))

        if latent_mode == 'raw':
            # Hyperswap path: model consumes the raw L2-normalized
            # embedding directly. No emap, no second normalize.
            if scale != 1.0:
                latent = latent * float(scale)
            return latent

        # Lazy-load the emap projection matrix on first emap-mode call.
        # Historical code used graph.initializer[-1] but the FP16 ONNX
        # has a 32x32 _ln_scale_32x32 tensor AFTER the emap, breaking
        # that heuristic. Look up by name ('buff2fs' is stable across
        # FP32 and FP16 exports), and fall back to "the (512, 512)
        # initializer" for robustness against future renames.
        if getattr(self, 'emap', None) is None or len(self.emap) == 0:
            graph = onnx.load(self._mp("inswapper_128.fp16.onnx")).graph
            emap_arr = None
            for init in graph.initializer:
                if init.name == 'buff2fs':
                    emap_arr = onnx.numpy_helper.to_array(init)
                    break
            if emap_arr is None:
                for init in graph.initializer:
                    arr = onnx.numpy_helper.to_array(init)
                    if arr.shape == (512, 512):
                        emap_arr = arr
                        break
            if emap_arr is None:
                raise RuntimeError(
                    "emap (512x512) not found in inswapper_128.fp16.onnx"
                )
            # FP16 ONNX stores the emap in float16; promote to float32
            # so the downstream np.dot stays in FP32 for numerical safety.
            self.emap = emap_arr.astype(np.float32, copy=False)

        latent = np.dot(latent, self.emap)
        latent /= np.linalg.norm(latent)
        if scale != 1.0:
            latent = latent * float(scale)
        return latent

    def embed_face_chip(self, img_chw_uint8):
        """ArcFace embedding of an already-aligned face crop. Skips the
        kps-driven similarity warp that `recognize()` does — the input
        must already be a tight, aligned face. Used by High Fidelity to
        measure the swap output's identity without re-detecting kps.

        img_chw_uint8: torch (3, H, W) uint8 RGB CUDA tensor; resized to
            112x112 inside the function.
        Returns: numpy (512,) float32 embedding (NOT L2-normalized,
            matching what `recognize()` returns)."""
        # Resize to the recognizer's expected 112x112 input.
        img = img_chw_uint8.to(torch.float32)
        if img.shape[-1] != 112 or img.shape[-2] != 112:
            img = v2.functional.resize(img, [112, 112], antialias=True)
        # ArcFace expects BGR, (x - 127.5) / 127.5.
        img = img[[2, 1, 0]]
        img = (img - 127.5) / 127.5
        img = img.unsqueeze(0).contiguous()

        recognition_session = self._get_recognition_session()
        output = torch.empty((1, 512), dtype=torch.float32, device='cuda').contiguous()
        io_binding = recognition_session.io_binding()
        io_binding.bind_input(
            name='input.1', device_type='cuda', device_id=0,
            element_type=np.float32, shape=(1, 3, 112, 112),
            buffer_ptr=img.data_ptr(),
        )
        io_binding.bind_output(
            name='683', device_type='cuda', device_id=0,
            element_type=np.float32, shape=(1, 512),
            buffer_ptr=output.data_ptr(),
        )
        if self._should_drain_syncvec():
            self.syncvec.cpu()
        recognition_session.run_with_iobinding(io_binding)
        return output.squeeze().cpu().numpy()
        
    # @profile
    def run_swapper(self, image, embedding, output):
        # Get the session appropriate for the calling thread. In
        # Shared mode this is the single self.swapper_model; in
        # Per-Thread mode each worker gets its own session for true
        # parallel inswapper runs.
        swapper_session = self._get_swapper_session()

        # The active inswapper ONNX may declare FP16 or FP32 I/O (set
        # at load time as _swapper_io_dtype). Callers always pass and
        # expect FP32 torch tensors, so we cast at the boundary when
        # the model wants FP16 — input via .half(), output via a temp
        # FP16 buffer that we copy back into the caller's buffer.
        io_dtype = self._swapper_io_dtype
        if io_dtype == np.float16:
            image_in = image.half() if image.dtype != torch.float16 else image
            embedding_in = embedding.half() if embedding.dtype != torch.float16 else embedding
            output_buf = torch.empty_like(output, dtype=torch.float16).contiguous()
        else:
            image_in = image
            embedding_in = embedding
            output_buf = output

        # Both backends (TRT-EP and CUDA-EP) return an ORT
        # InferenceSession, so a single io_binding path serves both —
        # ORT routes the call to the active provider internally.
        io_binding = swapper_session.io_binding()
        io_binding.bind_input(name='target', device_type='cuda', device_id=0, element_type=io_dtype, shape=(1,3,128,128), buffer_ptr=image_in.data_ptr())
        io_binding.bind_input(name='source', device_type='cuda', device_id=0, element_type=io_dtype, shape=(1,512), buffer_ptr=embedding_in.data_ptr())
        io_binding.bind_output(name='output', device_type='cuda', device_id=0, element_type=io_dtype, shape=(1,3,128,128), buffer_ptr=output_buf.data_ptr())

        # Drain torch's pending writes on the current CUDA stream before
        # ORT reads from the bound input buffers — but ONLY when torch
        # and ORT are on different streams. In Per-Thread mode with a
        # worker stream active, the ORT session was built with
        # `user_compute_stream` pointing at the same stream torch is
        # using, so ORT enqueues its kernels strictly after torch's on
        # that one stream — no race, no sync required. Skipping the
        # syncvec drain here is the structural payoff of the per-thread
        # streams refactor; it removes the cross-worker host stall on
        # the global default stream that prior nsys traces showed as
        # the next bottleneck.
        if self._swapper_should_drain_syncvec():
            with nvtx_range("syncvec_drain"):
                self.syncvec.cpu()
        with nvtx_range("ort_run_swapper"):
            swapper_session.run_with_iobinding(io_binding)

        # If we ran in FP16 mode, lift the result back into the
        # caller's FP32 buffer. copy_ handles the dtype conversion.
        if io_dtype == np.float16:
            output.copy_(output_buf)

    def _create_retinaface_session(self):
        """Build a fresh retinaface ORT-TRT-EP session. Honors the
        backend preference: 'onnx' → CUDA-EP only; default → TRT-EP
        with dynamic 320-640 input profile, CUDA-EP fallback if TRT-EP
        init fails. `user_compute_stream` picks up the calling thread's
        torch stream when one is active (Per-Thread + worker stream)."""
        try:
            onnxruntime.set_default_logger_severity(3)
        except (AttributeError, OSError):
            pass

        pref = self._backend_pref.get('retinaface_model')
        want_trt = pref != 'onnx'
        onnx_path = self._mp('det_10g.onnx')
        sess = None

        if want_trt:
            cache_dir = os.path.abspath(self._mp('ort_trt_cache'))
            try:
                os.makedirs(cache_dir, exist_ok=True)
            except OSError:
                pass
            trt_ep_options = {
                'trt_max_workspace_size': 2 << 30,
                'trt_engine_cache_enable': True,
                'trt_engine_cache_path': cache_dir,
                'trt_timing_cache_enable': True,
                'trt_timing_cache_path': cache_dir,
                'trt_fp16_enable': True,
                'trt_builder_optimization_level': 5,
                # Note: we intentionally do NOT set
                # `trt_dump_ep_context_model=True`. That flag writes an
                # extra EP-context-model ONNX wrapper next to the engine
                # in the cache dir; ORT additionally REQUIRES the cache
                # path to be relative when this flag is on (for security
                # reasons it printed an [E] warning at every session
                # load). We rely on `trt_engine_cache_enable=True` for
                # fast subsequent loads — that path works fine with
                # absolute cache dirs and is what makes Per-Thread mode
                # cheap after the first build.
                # Dynamic input profile: one engine covers every
                # DetectInputSize value (320 / 416 / 480 / 640). opt
                # at 640 — TRT specializes for that point.
                'trt_profile_min_shapes': 'input.1:1x3x320x320',
                'trt_profile_opt_shapes': 'input.1:1x3x640x640',
                'trt_profile_max_shapes': 'input.1:1x3x640x640',
            }
            try:
                cur_stream = torch.cuda.current_stream()
                if cur_stream is not None and cur_stream != torch.cuda.default_stream():
                    trt_ep_options['user_compute_stream'] = str(int(cur_stream.cuda_stream))
            except Exception:
                pass
            try:
                sess_options = self._make_session_options()
                cuda_options = self._cuda_ep_provider_options(cudnn_algo='HEURISTIC')
                sess = onnxruntime.InferenceSession(
                    onnx_path,
                    sess_options=sess_options,
                    providers=[
                        ('TensorrtExecutionProvider', trt_ep_options),
                        ('CUDAExecutionProvider', cuda_options),
                    ],
                )
            except Exception as e:
                print('[Models] retinaface TRT-EP init failed (%s: %s); '
                      'falling back to CUDA EP' % (type(e).__name__, e))
                sess = None

        if sess is None:
            # retinaface's input is a dynamic 320→640 profile, so EXHAUSTIVE
            # would re-benchmark on every DetectInputSize change (per
            # session). HEURISTIC picks a near-optimal algo with no
            # benchmarking → no interactive stalls.
            cuda_options = self._cuda_ep_provider_options(cudnn_algo='HEURISTIC')
            sess_options = self._make_session_options()
            sess = onnxruntime.InferenceSession(
                onnx_path,
                sess_options=sess_options,
                providers=[('CUDAExecutionProvider', cuda_options)],
            )

        active = sess.get_providers()
        self._retinaface_uses_trt = bool(active) and active[0] == 'TensorrtExecutionProvider'
        # Profile bounds are what we configured above; no engine
        # introspection needed since ORT-TRT-EP honors what we asked.
        self._retinaface_trt_input_min = 320
        self._retinaface_trt_input_max = 640
        self._retinaface_engine_size_warned = False
        backend = 'TensorrtExecutionProvider' if self._retinaface_uses_trt else 'CUDAExecutionProvider'
        print(f'{self._load_tag("find")} [Models] retinaface: {backend} (accepts input 320-640)')
        return sess

    def _load_retinaface_model(self):
        """Shared-mode loader: builds the session and stashes it on
        self.retinaface_model. Per-Thread mode goes through
        _get_retinaface_session directly."""
        self.retinaface_model = self._create_retinaface_session()

    def _get_retinaface_session(self):
        return self._get_model_session(
            'retinaface', 'retinaface_model', self._create_retinaface_session,
        )

    def _get_retinaface_input_buffer(self, input_size):
        """Per-worker cached (1, 3, H, W) float32 input buffer for the
        retinaface ORT call. Built lazily on first use per (worker
        thread, input_size). Caller zeros and writes into it; ORT's
        io_binding reads from .data_ptr() directly. Sized once per
        input_size, so when the DetectInputSize slider is stable there's
        zero allocator pressure on detect's hot path."""
        bufs = getattr(self._retinaface_buffers_tls, 'cache', None)
        if bufs is None:
            bufs = {}
            self._retinaface_buffers_tls.cache = bufs
        key = int(input_size)
        buf = bufs.get(key)
        if buf is None:
            buf = torch.zeros(
                (1, 3, key, key), dtype=torch.float32, device='cuda:0',
            )
            bufs[key] = buf
        return buf

    def _get_retinaface_gpu_anchors(self, height, width, stride):
        """Return a (height*width*2, 2) torch tensor of anchor centers
        on CUDA for the given det_10g feature map. The detector has 2
        anchors per cell; the doubled rows are interleaved to match
        the model's output ordering.

        Cached on self._retinaface_center_cache. First call per
        (h, w, stride) builds + torch.cuda.synchronize()s so the writes
        are visible to other workers' streams; subsequent calls return
        the cached tensor with no sync."""
        key = (int(height), int(width), int(stride))
        cached = self._retinaface_center_cache.get(key)
        if cached is not None:
            return cached
        y_coords = torch.arange(height, dtype=torch.float32, device='cuda:0')
        x_coords = torch.arange(width, dtype=torch.float32, device='cuda:0')
        # indexing='ij' → yy[h, w] = h, xx[h, w] = w. Stacking [xx, yy]
        # along the last axis gives anchors[h, w] = (x, y) where x is
        # the column and y the row, matching the numpy original via
        # np.mgrid[:H, :W][::-1].
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
        anchors = torch.stack([xx, yy], dim=-1) * float(stride)  # (H, W, 2)
        anchors = anchors.reshape(-1, 2)                          # (H*W, 2)
        # Duplicate each row for the 2 anchors per cell. Interleaved
        # via expand+reshape (matches np.stack([a]*2, axis=1).reshape).
        anchors = anchors.unsqueeze(1).expand(-1, 2, -1).reshape(-1, 2).contiguous()
        # Force the build kernels to materialize before another worker
        # on a different stream reads the cache entry. Only paid on
        # cache miss (~3 times per input_size at app startup).
        torch.cuda.synchronize()
        self._retinaface_center_cache[key] = anchors
        return anchors

    def _create_recognition_session(self):
        """Build a fresh arcface (w600k_r50) ORT-TRT-EP session.
        Same pattern as retinaface: TRT-EP by default with CUDA-EP
        fallback. Input is a fixed (1, 3, 112, 112) face chip — no
        dynamic profile needed."""
        try:
            onnxruntime.set_default_logger_severity(3)
        except (AttributeError, OSError):
            pass

        pref = self._backend_pref.get('recognition_model')
        want_trt = pref != 'onnx'
        onnx_path = self._mp('w600k_r50.onnx')
        sess = None

        if want_trt:
            cache_dir = os.path.abspath(self._mp('ort_trt_cache'))
            try:
                os.makedirs(cache_dir, exist_ok=True)
            except OSError:
                pass
            trt_ep_options = {
                'trt_max_workspace_size': 1 << 30,
                'trt_engine_cache_enable': True,
                'trt_engine_cache_path': cache_dir,
                'trt_timing_cache_enable': True,
                'trt_timing_cache_path': cache_dir,
                'trt_fp16_enable': True,
                'trt_builder_optimization_level': 5,
                # Note: we intentionally do NOT set
                # `trt_dump_ep_context_model=True`. That flag writes an
                # extra EP-context-model ONNX wrapper next to the engine
                # in the cache dir; ORT additionally REQUIRES the cache
                # path to be relative when this flag is on (for security
                # reasons it printed an [E] warning at every session
                # load). We rely on `trt_engine_cache_enable=True` for
                # fast subsequent loads — that path works fine with
                # absolute cache dirs and is what makes Per-Thread mode
                # cheap after the first build.
            }
            try:
                cur_stream = torch.cuda.current_stream()
                if cur_stream is not None and cur_stream != torch.cuda.default_stream():
                    trt_ep_options['user_compute_stream'] = str(int(cur_stream.cuda_stream))
            except Exception:
                pass
            try:
                sess_options = self._make_session_options()
                cuda_options = self._cuda_ep_provider_options(cudnn_algo='EXHAUSTIVE')
                sess = onnxruntime.InferenceSession(
                    onnx_path,
                    sess_options=sess_options,
                    providers=[
                        ('TensorrtExecutionProvider', trt_ep_options),
                        ('CUDAExecutionProvider', cuda_options),
                    ],
                )
            except Exception as e:
                print('[Models] arcface TRT-EP init failed (%s: %s); '
                      'falling back to CUDA EP' % (type(e).__name__, e))
                sess = None

        if sess is None:
            # arcface has a fixed 112×112 input → EXHAUSTIVE, one-time warmup.
            cuda_options = self._cuda_ep_provider_options(cudnn_algo='EXHAUSTIVE')
            sess_options = self._make_session_options()
            sess = onnxruntime.InferenceSession(
                onnx_path,
                sess_options=sess_options,
                providers=[('CUDAExecutionProvider', cuda_options)],
            )

        active = sess.get_providers()
        self._recognition_uses_trt = bool(active) and active[0] == 'TensorrtExecutionProvider'
        backend = 'TensorrtExecutionProvider' if self._recognition_uses_trt else 'CUDAExecutionProvider'
        print(f'{self._load_tag("find")} [Models] arcface (w600k_r50): {backend}')
        return sess

    def _load_recognition_model(self):
        self.recognition_model = self._create_recognition_session()

    def _get_recognition_session(self):
        return self._get_model_session(
            'recognition', 'recognition_model', self._create_recognition_session,
        )

    def _create_swapper_session(self):
        """Build and return a fresh inswapper ORT InferenceSession,
        picking provider per the user's backend preference. Shared
        and Per-Thread modes both call into this; the caller decides
        where to stash the returned session.

        Provider semantics:

            'onnx' (or no TRT available) → CUDAExecutionProvider
                Plain cuDNN/cuBLAS path. Always works. Used as the
                fallback when TRT init fails.
            'trt' (default if absent)    → TensorrtExecutionProvider
                ORT builds and caches a TensorRT engine internally on
                first run. The crucial option is
                trt_layer_norm_fp32_fallback=True — without it, FP16
                LayerNorm accumulators overflow on the inswapper graph
                and produce visibly garbled output (the same failure
                our hand-built FP16 engine hit). This option is an
                ORT-TRT-EP-specific knob with no raw TRT API equivalent,
                which is why we route through ORT instead of building
                the engine ourselves like we do for retinaface/arcface.

        Inspired by VisoMaster's models_processor.trt_ep_options. ORT
        persists the built engine to {models_folder}/ort_trt_cache so
        subsequent sessions reload it quickly.

        After construction we read session.get_providers() to learn
        which provider actually ended up active — ORT silently falls
        back to CUDA EP if TRT-EP fails to initialize. The
        _swapper_uses_trt flag reflects reality, not the request."""
        # Defensive: clamp the env-level Default logger severity right
        # before we create the session. ORT_LOG_SEVERITY_LEVEL is set in
        # rope/_native_dlls.py before ORT imports, but if ORT was somehow
        # imported earlier (transitive site-packages import, hot reload)
        # the Env logger may have been created at INFO. Re-asserting via
        # the in-process API silences the BFCArena chatter that
        # otherwise floods stdout on first CUDA-EP session.
        try:
            onnxruntime.set_default_logger_severity(3)
        except (AttributeError, OSError):
            pass

        pref = self._backend_pref.get('swapper_model')
        want_trt = pref != 'onnx'

        onnx_path = self._mp("inswapper_128.fp16.onnx")
        sess = None

        if want_trt:
            # ORT-TRT-EP joins the configured cache path with its own
            # working-directory base — passing a relative path produces
            # nonsense like "./models\ort_trt_cache\./models\ort_trt_cache"
            # and the provider falls back silently to CUDA EP. Always
            # pass an absolute, normalized path.
            cache_dir = os.path.abspath(self._mp("ort_trt_cache"))
            try:
                os.makedirs(cache_dir, exist_ok=True)
            except OSError:
                pass
            trt_ep_options = {
                'trt_max_workspace_size': 4 << 30,
                'trt_engine_cache_enable': True,
                'trt_engine_cache_path': cache_dir,
                'trt_timing_cache_enable': True,
                'trt_timing_cache_path': cache_dir,
                'trt_fp16_enable': True,
                # The critical flag: keeps LayerNorm reductions in FP32
                # even under FP16. Without this the FP16 path on
                # inswapper produces visually garbled output.
                'trt_layer_norm_fp32_fallback': True,
                'trt_builder_optimization_level': 5,
                # Note: we intentionally do NOT set
                # `trt_dump_ep_context_model=True`. That flag writes an
                # extra EP-context-model ONNX wrapper next to the engine
                # in the cache dir; ORT additionally REQUIRES the cache
                # path to be relative when this flag is on (for security
                # reasons it printed an [E] warning at every session
                # load). We rely on `trt_engine_cache_enable=True` for
                # fast subsequent loads — that path works fine with
                # absolute cache dirs and is what makes Per-Thread mode
                # cheap after the first build.
            }
            # If the calling thread has its own CUDA stream set as
            # current (the VideoManager worker pattern in Per-Thread
            # mode), pass that to ORT-TRT-EP via user_compute_stream
            # so ORT enqueues its kernels onto the same stream torch
            # is using. Torch and ORT then share ordering — no cross-
            # stream race, syncvec drain becomes unnecessary, workers
            # don't serialize on the global default stream.
            try:
                cur_stream = torch.cuda.current_stream()
                # Skip when the worker is on the default stream
                # (Shared mode or no per-worker streams active): the
                # device→host syncvec keeps cross-stream ordering
                # safe and we don't gain anything from binding ORT
                # to the same stream torch already serializes on.
                if cur_stream is not None and cur_stream != torch.cuda.default_stream():
                    trt_ep_options['user_compute_stream'] = str(int(cur_stream.cuda_stream))
            except Exception:
                pass
            try:
                sess_options = self._make_session_options()
                # The CUDA-EP fallback carries the same tuned options
                # (incl. user_compute_stream) so a silent TRT→CUDA fallback
                # still binds to the worker stream — a bare string entry
                # would drop the stream and run on the default stream.
                cuda_options = self._cuda_ep_provider_options(cudnn_algo='EXHAUSTIVE')
                sess = onnxruntime.InferenceSession(
                    onnx_path,
                    sess_options=sess_options,
                    providers=[
                        ('TensorrtExecutionProvider', trt_ep_options),
                        ('CUDAExecutionProvider', cuda_options),
                    ],
                )
            except Exception as e:
                print('[Models] inswapper TRT-EP init failed (%s: %s); '
                      'falling back to CUDA EP' % (type(e).__name__, e))
                sess = None

        if sess is None:
            # inswapper_128 has a fixed 128×128 input, so EXHAUSTIVE's
            # one-time conv benchmark is absorbed by the startup preload
            # warm-up (see preload_pipeline_sessions).
            cuda_options = self._cuda_ep_provider_options(cudnn_algo='EXHAUSTIVE')
            sess_options = self._make_session_options()
            sess = onnxruntime.InferenceSession(
                onnx_path,
                sess_options=sess_options,
                providers=[('CUDAExecutionProvider', cuda_options)],
            )

        # Was TRT-EP actually selected? get_providers() returns the
        # active list in priority order; if the first entry is TRT,
        # it's hot. Silent fallbacks (TRT-EP requested but ORT picked
        # CUDA anyway) end up here as False. Detect once and cache —
        # all sessions of the same model under the same backend pref
        # land on the same provider, so we don't need to re-detect
        # for every Per-Thread session.
        active = sess.get_providers()
        self._swapper_uses_trt = bool(active) and active[0] == 'TensorrtExecutionProvider'
        # Detect the model's actual I/O dtype. Real FP16-throughout
        # inswapper exports declare FP16 inputs/outputs and require
        # FP16-bound buffers; older "FP16-weights, FP32-IO" exports
        # declare FP32 and accept FP32 buffers directly. We cast at
        # the io_binding boundary based on this. ORT's input.type is a
        # string like "tensor(float16)" or "tensor(float)".
        inp = sess.get_inputs()[0]
        self._swapper_io_dtype = (
            np.float16 if 'float16' in inp.type else np.float32
        )
        # Detect static-batch ONNX exports (shape[0] is an int rather
        # than a string/None for symbolic dims). If so, pre-set the
        # batch-unsupported flag so run_swapper_batched routes to the
        # per-call fallback without paying the one-shot exception cost.
        try:
            shape0 = inp.shape[0]
            self._swapper_batch_unsupported = isinstance(shape0, int) and shape0 == 1
        except Exception:
            self._swapper_batch_unsupported = False
        backend = 'TensorrtExecutionProvider' if self._swapper_uses_trt else 'CUDAExecutionProvider'
        batch_note = ' static-batch=1' if self._swapper_batch_unsupported else ''
        print(f'{self._load_tag("swap")} [Models] inswapper: {backend} (io={self._swapper_io_dtype.__name__}{batch_note})')
        return sess

    # ===== Generic per-thread session dispatcher ==========================
    # Every ORT-TRT-EP model in the main pipeline (inswapper, inswapper_512,
    # retinaface, recognition) routes through these helpers. The mode is
    # process-wide; each model carries its own (name, shared-attr, factory)
    # so the same dispatcher serves all of them.

    def _get_model_session(self, name, shared_attr, create_fn):
        """Return the ORT session for `name` in the current mode.

        Shared mode: lazy-load into self.<shared_attr> and hand the same
        session to every thread. Per-Thread mode: lazy-load into thread-
        local storage so each worker gets its own session. The on-disk
        TRT engine cache is shared, so the second-through-Nth Per-Thread
        session deserializes the existing engine instead of rebuilding.

        create_fn() is the factory that produces a fresh session — built
        on the calling thread so its `user_compute_stream` (if any) picks
        up that thread's current torch stream."""
        if self._model_session_mode == 'Per-Thread':
            sess = getattr(self._sessions_tls, name, None)
            if sess is None:
                sess = create_fn()
                setattr(self._sessions_tls, name, sess)
                with self._sessions_lock:
                    self._all_sessions.setdefault(name, []).append(sess)
            return sess
        # Shared mode: lazy-load once, reuse across all threads.
        sess = getattr(self, shared_attr, None)
        if not sess:
            sess = create_fn()
            setattr(self, shared_attr, sess)
        return sess

    def _should_drain_syncvec(self):
        """True when an ORT call needs an explicit torch-stream drain
        before ORT reads its input buffers. False only when torch's
        current stream is the same stream the ORT session was built
        with — that happens in Per-Thread mode when a worker is running
        inside its `with torch.cuda.stream(worker_stream)` context.
        Same stream = implicit ordering, no race."""
        if self._model_session_mode != 'Per-Thread':
            return True
        try:
            cur = torch.cuda.current_stream()
            return cur is None or cur == torch.cuda.default_stream()
        except Exception:
            return True

    # ===== Shared ORT session configuration ==============================
    # Both the explicit-ONNX branch and the TRT branch's CUDA fallback in
    # every _create_*_session factory route through these two helpers so the
    # CUDA-EP path is tuned identically everywhere (previously each factory
    # inlined its own — and inconsistent — options).

    def _cuda_ep_provider_options(self, cudnn_algo='EXHAUSTIVE'):
        """CUDAExecutionProvider options dict. Returns a fresh dict each
        call (ORT may mutate it, and the stream value is thread-specific).
        Must be called on the worker thread so `torch.cuda.current_stream()`
        picks up that worker's stream — same construction-time contract the
        TRT branch already uses.

        cudnn_algo:
            'EXHAUSTIVE' (default) — benchmark the fastest conv algorithm.
                Best steady-state; a one-time per-shape benchmark on first
                inference. Right for the fixed-input-shape models (swapper,
                arcface, inswapper_512), whose warmup the startup preload
                absorbs.
            'HEURISTIC' — pick a near-optimal algo analytically, no
                benchmarking. Right for retinaface, whose input is a dynamic
                320→640 profile (EXHAUSTIVE would re-benchmark and stall on
                every DetectInputSize change, per session).
        """
        opts = {
            'arena_extend_strategy': 'kSameAsRequested',
            'cudnn_conv_algo_search': cudnn_algo,
        }
        try:
            cur = torch.cuda.current_stream()
            if cur is not None and cur != torch.cuda.default_stream():
                opts['user_compute_stream'] = str(int(cur.cuda_stream))
        except Exception:
            pass
        return opts

    def _make_session_options(self):
        """SessionOptions tuned to avoid CPU thread oversubscription. ORT
        sizes each session's intra-op pool to the CPU core count by default;
        in Per-Thread mode ~3·N sessions are live (N = worker threads), so
        N·3·core_count ORT threads would contend with each other and the N
        torch workers — which hurts CUDA-EP (node-by-node dispatch) far more
        than TRT-EP (one fused engine node). We keep each session lean and
        let the N worker threads supply the concurrency instead.

        Also sets log_severity_level=3 (was previously the only thing set)
        because ORT 1.26's default-severity inheritance via -1 is broken and
        every session otherwise spews ~100 INFO lines at init."""
        so = onnxruntime.SessionOptions()
        so.log_severity_level = 3
        so.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
        so.inter_op_num_threads = 1
        if self._model_session_mode == 'Per-Thread':
            # One intra-op thread per session; concurrency comes from the
            # N worker threads each driving their own session.
            so.intra_op_num_threads = 1
        else:
            # Shared mode: ~3 sessions total, hit by N concurrent Run()
            # calls — give them a small pool so CPU-side work isn't
            # serialized, but stay well under core_count to avoid
            # re-introducing oversubscription.
            cores = os.cpu_count() or 4
            so.intra_op_num_threads = max(2, min(8, cores // 2))
        return so

    # Per-phase model count: how many distinct models load in each
    # user-driven phase. Used to compute M in _load_tag — multiplied by
    # the worker count when in Per-Thread mode.
    _PHASE_PER_WORKER = {
        'find': 2,  # retinaface + arcface
        'swap': 1,  # inswapper_128 or inswapper_512 (one at a time)
    }

    def update_load_expectation(self, threads):
        """Store the worker-count hint and reset phase tracking. Called
        by VideoManager._ensure_executor when the pool rebuilds, and
        from main_window on every parameter change. The next session
        create starts a fresh phase counter."""
        threads = max(1, int(threads))
        with self._load_lock:
            self._workers_hint = threads
            self._load_phase = None
            self._load_count = 0
            self._load_expected = 0

    def _load_tag(self, phase):
        """Return '[N/M]' for the next session-create print. `phase` is
        'find' or 'swap'. When phase changes from the previous call,
        the counter resets and M is recomputed from _PHASE_PER_WORKER
        × workers (Per-Thread) or × 1 (Shared)."""
        per_worker = self._PHASE_PER_WORKER.get(phase, 1)
        with self._load_lock:
            workers = self._workers_hint if self._model_session_mode == 'Per-Thread' else 1
            if self._load_phase != phase:
                self._load_phase = phase
                self._load_count = 0
                self._load_expected = per_worker * workers
            self._load_count += 1
            n = self._load_count
            m = self._load_expected or n
        return f'[{n}/{m}]'

    def clear_per_thread_sessions(self, name=None):
        """Release strong refs to Per-Thread sessions so dying workers'
        sessions can GC. Pass `name` to clear just one model's
        per-thread sessions; pass None to clear all. Each live worker
        keeps its TLS slot intact, so in-flight calls are unaffected —
        Python's threading.local auto-cleans the dead thread's slot
        when the worker exits, and the session GCs once the strong-ref
        list no longer holds it."""
        with self._sessions_lock:
            if name is None:
                self._all_sessions = {}
            else:
                self._all_sessions.pop(name, None)
        self.vram_dirty = True

    def set_model_session_mode(self, mode):
        """Switch between 'Shared' (one ORT session per model across
        all workers) and 'Per-Thread' (one session per model per
        worker). Drops all existing sessions across every model so the
        new mode takes effect on next inference call — active workers
        will re-load lazily."""
        mode = str(mode)
        if mode not in ('Shared', 'Per-Thread'):
            return
        if mode == self._model_session_mode:
            return
        self._model_session_mode = mode
        # Drop everything — Shared attrs and Per-Thread refs both.
        # Per-Thread mode change resets `_sessions_tls` which severs
        # every thread's slot at once (workers will re-create on next
        # access). Shared singletons are cleared individually.
        self._sessions_tls = threading.local()
        with self._sessions_lock:
            self._all_sessions = {}
        for shared_attr in (
            'swapper_model',
            'swapper_512_model',
            'swapper_256_model',
            'retinaface_model',
            'recognition_model',
        ):
            if getattr(self, shared_attr, None):
                setattr(self, shared_attr, None)
        # Per-model "uses TRT" / batch / engine-profile caches are
        # detected fresh on the next session build, so no clear needed
        # — but mark VRAM dirty so the indicator catches the drop.
        self.vram_dirty = True
        print(f'[Models] model session mode -> {mode}')

    # ===== Inswapper session glue =========================================
    # Inswapper is the first model on the generic dispatcher. The helper
    # name `_get_swapper_session` is preserved so existing call sites in
    # run_swapper / run_swapper_batched keep working without churn.

    def _get_swapper_session(self):
        return self._get_model_session(
            'swapper', 'swapper_model', self._create_swapper_session,
        )

    # ===== Startup preload ================================================
    # The three per-thread swap-pipeline sessions, in the order preload
    # builds them: (ONNX filename, session-getter method name). Getters
    # route through _get_model_session, so calling one on a worker thread
    # populates that thread's per-thread slot (Per-Thread mode) or the
    # shared singleton (Shared mode) — exactly what a lazy swap would do.
    # The recognizer (arcface) is always part of a swap regardless of the
    # user's detector / swapper choices, so it's preloaded unconditionally.
    _RECOGNITION_PRELOAD = ('w600k_r50.onnx', '_get_recognition_session')

    # Detector selection (DetectTypeTextSel) → (ONNX filename, loader). Only
    # Retinaface routes through the per-thread ORT-TRT-EP getter (thread-
    # scaled, benefits most from preload); SCRDF is a shared singleton
    # warmed once via _ensure_scrdf_model. Preloading the *selected*
    # detector (and not the others) is what makes the button honor
    # "which detector".
    _DETECTOR_PRELOAD = {
        'Retinaface': ('det_10g.onnx',          '_get_retinaface_session'),
        'SCRDF':      ('scrfd_2.5g_bnkps.onnx', '_ensure_scrdf_model'),
    }

    # Swapper-Resolution selection (SwapperTypeTextSel) → (ONNX filename,
    # getter). 128/256/512 all drive inswapper_128 with polyphase tiling;
    # only "256-Native" uses the dedicated single-pass inswapper_256_phase1.
    _SWAPPER_PRELOAD_128 = ('inswapper_128.fp16.onnx', '_get_swapper_session')
    _SWAPPER_PRELOAD_NATIVE256 = ('inswapper_256_phase1.onnx', '_get_swapper_256_session')

    def _detector_preload_for(self, detect_mode):
        return self._DETECTOR_PRELOAD.get(
            str(detect_mode), self._DETECTOR_PRELOAD['Retinaface'])

    def _swapper_preload_for(self, swapper_type):
        if str(swapper_type) == '256-Native':
            return self._SWAPPER_PRELOAD_NATIVE256
        return self._SWAPPER_PRELOAD_128

    def _pipeline_preload_set(self, swapper_type='128', detect_mode='Retinaface'):
        """(filename, getter) triples to preload for the given detector +
        swapper selection, plus the always-needed recognizer."""
        return [
            self._detector_preload_for(detect_mode),
            self._RECOGNITION_PRELOAD,
            self._swapper_preload_for(swapper_type),
        ]

    def _ensure_scrdf_model(self):
        """Load the shared SCRDF detector session if not already loaded.
        Shared singleton (not per-thread) — mirrors the lazy load in
        run_detect so both paths build it identically."""
        if not self.scrdf_model:
            self.scrdf_model = onnxruntime.InferenceSession(
                self._mp('scrfd_2.5g_bnkps.onnx'), providers=self.providers)
        return self.scrdf_model

    # Detector selection (DetectTypeTextSel) → the Models attribute whose
    # liveness is_model_loaded checks. Retinaface routes through the
    # per-thread session list; SCRDF is a shared singleton.
    _DETECTOR_LOADED_ATTR = {
        'Retinaface': 'retinaface_model',
        'SCRDF':      'scrdf_model',
    }

    def _pipeline_loaded_attrs(self, swapper_type='128', detect_mode='Retinaface'):
        """Models attributes whose sessions must be live for the given
        detector + swapper selection to run a swap (mirrors
        _pipeline_preload_set, but in is_model_loaded's attr vocabulary)."""
        det_attr = self._DETECTOR_LOADED_ATTR.get(str(detect_mode), 'retinaface_model')
        sw_attr = ('swapper_256_model' if str(swapper_type) == '256-Native'
                   else 'swapper_model')
        return [det_attr, 'recognition_model', sw_attr]

    def pipeline_sessions_loaded(self, swapper_type='128', detect_mode='Retinaface'):
        """True when every model the current selection needs has a live
        session. Used by the GUI to flip the Preload button to its loaded
        state (and to verify a preload actually succeeded)."""
        return all(self.is_model_loaded(a)
                   for a in self._pipeline_loaded_attrs(swapper_type, detect_mode))

    def swap_pipeline_files_present(self, swapper_type='128', detect_mode='Retinaface'):
        """True only if every ONNX backing the *selected* preloadable swap-
        pipeline sessions exists in the current models folder. Preload is
        skipped when False so a models-less install doesn't spew load
        errors. Defaults reproduce the legacy retinaface + inswapper_128
        check when called with no arguments."""
        return all(os.path.exists(self._mp(f))
                   for f, _ in self._pipeline_preload_set(swapper_type, detect_mode))

    def preload_pipeline_sessions(self, swapper_type='128', detect_mode='Retinaface'):
        """Build (or return the already-built) detector + recognizer +
        swapper sessions matching the user's current selections, on the
        CALLING thread. Meant to run on a VideoManager worker thread inside
        its `with torch.cuda.stream(...)` block so the per-thread sessions
        bind to that worker's stream, matching how swap_video would build
        them lazily. Each model is guarded so a single missing/failed model
        doesn't abort the rest.

        `swapper_type` (SwapperTypeTextSel) picks inswapper_128 vs the
        native-256 model; `detect_mode` (DetectTypeTextSel) picks which
        detector to warm — so the button only builds what the current
        selections will actually use."""
        for fname, getter in self._pipeline_preload_set(swapper_type, detect_mode):
            if not os.path.exists(self._mp(fname)):
                continue
            try:
                getattr(self, getter)()
            except Exception as e:
                print(f'[Models.preload_pipeline_sessions] {fname}: '
                      f'{type(e).__name__}: {e}')

        # Warm up the selected swapper with one throwaway inference so the
        # CUDA-EP EXHAUSTIVE cuDNN conv benchmark (and workspace/handle
        # allocation) happens here — at load, on this worker's stream —
        # instead of stalling the first live swap frame on each of the N
        # workers. On the TRT backend this is a cheap already-built engine
        # run. Guarded so a warm-up failure never breaks startup. run_swapper
        # handles FP16/FP32 casting internally, so plain FP32 zeros are fine.
        if str(swapper_type) == '256-Native':
            if os.path.exists(self._mp('inswapper_256_phase1.onnx')):
                try:
                    z_img = torch.zeros((1, 3, 256, 256), dtype=torch.float32, device='cuda')
                    z_emb = torch.zeros((1, 512), dtype=torch.float32, device='cuda')
                    z_out = torch.zeros((1, 3, 256, 256), dtype=torch.float32, device='cuda')
                    self.run_swapper_256(z_img, z_emb, z_out)
                except Exception as e:
                    print(f'[Models.preload_pipeline_sessions] swapper_256 warm-up: '
                          f'{type(e).__name__}: {e}')
        elif os.path.exists(self._mp('inswapper_128.fp16.onnx')):
            try:
                z_img = torch.zeros((1, 3, 128, 128), dtype=torch.float32, device='cuda')
                z_emb = torch.zeros((1, 512), dtype=torch.float32, device='cuda')
                z_out = torch.zeros((1, 3, 128, 128), dtype=torch.float32, device='cuda')
                self.run_swapper(z_img, z_emb, z_out)
            except Exception as e:
                print(f'[Models.preload_pipeline_sessions] swapper warm-up: '
                      f'{type(e).__name__}: {e}')

    def _swapper_should_drain_syncvec(self):
        return self._should_drain_syncvec()
    # @profile
    def run_swapper_batched(self, image_batch, embedding_batch, output_batch):
        """Run the swapper on a batch of N target tiles sharing one source.
        Shapes:
            image_batch    (N, 3, 128, 128) float32 cuda — contiguous
            embedding_batch(1, 512) or (N, 512) float32 cuda — only row 0 is
                read; the inswapper source binding is always (1, 512) and
                the model broadcasts internally. (N, 512) is accepted for
                callers that pre-existing code paths still produce, but is
                wasted memory.
            output_batch   (N, 3, 128, 128) float32 cuda — preallocated
        Falls back to per-call run_swapper when the model's first batched
        attempt fails (some inswapper exports lock the batch dim to 1).
        After the first failure the unsupported flag is sticky so subsequent
        calls skip the probe.

        Both backends (ORT TRT-EP and CUDA-EP) accept dynamic batch via
        a single io_binding call — the ONNX itself declares the batch
        dim as symbolic, and ORT routes the call to whichever provider
        is active."""
        # Lazy-init the appropriate session (Shared or Per-Thread)
        # by probing with an N=1 call. After this, run_swapper has
        # populated _swapper_io_dtype + _swapper_batch_unsupported and
        # the calling thread has its session ready.
        n = image_batch.shape[0]
        # Source is shared across all targets, so always bind row 0
        # regardless of whether the caller passed (1, 512) or (N, 512).
        emb_one_caller = embedding_batch[0:1]
        if n == 1:
            self.run_swapper(image_batch, emb_one_caller, output_batch)
            return

        if getattr(self, '_swapper_batch_unsupported', False):
            for k in range(n):
                self.run_swapper(
                    image_batch[k:k+1], emb_one_caller, output_batch[k:k+1],
                )
            return

        swapper_session = self._get_swapper_session()

        io_dtype = self._swapper_io_dtype
        if io_dtype == np.float16:
            image_in = image_batch.half() if image_batch.dtype != torch.float16 else image_batch
            # Only the first row of the source is ever read; cast just
            # that slice to FP16 instead of casting all N rows.
            embedding_in = emb_one_caller.half() if emb_one_caller.dtype != torch.float16 else emb_one_caller
            output_buf = torch.empty_like(output_batch, dtype=torch.float16).contiguous()
        else:
            image_in = image_batch
            embedding_in = emb_one_caller
            output_buf = output_batch

        # The inswapper source binding is fixed (1, 512); the model
        # broadcasts the source identity across the target batch
        # internally. embedding_in is already (1, 512) from emb_one_caller
        # above; .contiguous() is a no-op on contiguous row-0 slices.
        emb_one = embedding_in.contiguous()

        io_binding = swapper_session.io_binding()
        io_binding.bind_input(
            name='target', device_type='cuda', device_id=0,
            element_type=io_dtype, shape=(n, 3, 128, 128),
            buffer_ptr=image_in.data_ptr(),
        )
        io_binding.bind_input(
            name='source', device_type='cuda', device_id=0,
            element_type=io_dtype, shape=(1, 512),
            buffer_ptr=emb_one.data_ptr(),
        )
        io_binding.bind_output(
            name='output', device_type='cuda', device_id=0,
            element_type=io_dtype, shape=(n, 3, 128, 128),
            buffer_ptr=output_buf.data_ptr(),
        )
        # Same torch-stream drain as run_swapper — conditional on
        # whether ORT and torch are on different streams. See
        # _swapper_should_drain_syncvec / run_swapper for rationale.
        if self._swapper_should_drain_syncvec():
            with nvtx_range("syncvec_drain"):
                self.syncvec.cpu()
        try:
            with nvtx_range(f"ort_run_swapper_batched[n={n}]"):
                swapper_session.run_with_iobinding(io_binding)
        except Exception as e:
            print('[Models] swapper batched call rejected (%s: %s); '
                  'falling back to per-call for the rest of the session'
                  % (type(e).__name__, e))
            self._swapper_batch_unsupported = True
            for k in range(n):
                self.run_swapper(
                    image_batch[k:k+1], embedding_batch[k:k+1], output_batch[k:k+1],
                )
            return

        if io_dtype == np.float16:
            output_batch.copy_(output_buf)

    # ===== inswapper_512 (512x512 in / 512x512 out) =========================
    # Parallel API to run_swapper / run_swapper_batched. Same conditioning
    # vector (1,512 float32), same source-embedding -> latent pipeline via
    # calc_swapper_latent. Only the target/output spatial shape differs.

    def run_swapper_512(self, image, embedding, output):
        swapper_512_session = self._get_swapper_512_session()

        io_binding = swapper_512_session.io_binding()
        io_binding.bind_input(
            name='target', device_type='cuda', device_id=0,
            element_type=np.float32, shape=(1, 3, 512, 512),
            buffer_ptr=image.data_ptr(),
        )
        io_binding.bind_input(
            name='source', device_type='cuda', device_id=0,
            element_type=np.float32, shape=(1, 512),
            buffer_ptr=embedding.data_ptr(),
        )
        io_binding.bind_output(
            name='output', device_type='cuda', device_id=0,
            element_type=np.float32, shape=(1, 3, 512, 512),
            buffer_ptr=output.data_ptr(),
        )

        if self._should_drain_syncvec():
            self.syncvec.cpu()
        swapper_512_session.run_with_iobinding(io_binding)

    def _create_swapper_512_session(self):
        """Build a fresh inswapper_512 ORT-TRT-EP session. Static
        (1, 3, 512, 512) input — no dynamic profile needed. Batch is
        always 1 at the engine level; callers invoke run_swapper_512
        once per face."""
        try:
            onnxruntime.set_default_logger_severity(3)
        except (AttributeError, OSError):
            pass

        pref = self._backend_pref.get('swapper_512_model')
        want_trt = pref != 'onnx'
        onnx_path = self._mp('inswapper_512_level2.onnx')
        sess = None

        if want_trt:
            cache_dir = os.path.abspath(self._mp('ort_trt_cache'))
            try:
                os.makedirs(cache_dir, exist_ok=True)
            except OSError:
                pass
            trt_ep_options = {
                'trt_max_workspace_size': 4 << 30,
                'trt_engine_cache_enable': True,
                'trt_engine_cache_path': cache_dir,
                'trt_timing_cache_enable': True,
                'trt_timing_cache_path': cache_dir,
                'trt_fp16_enable': True,
                'trt_builder_optimization_level': 5,
                # Note: we intentionally do NOT set
                # `trt_dump_ep_context_model=True`. That flag writes an
                # extra EP-context-model ONNX wrapper next to the engine
                # in the cache dir; ORT additionally REQUIRES the cache
                # path to be relative when this flag is on (for security
                # reasons it printed an [E] warning at every session
                # load). We rely on `trt_engine_cache_enable=True` for
                # fast subsequent loads — that path works fine with
                # absolute cache dirs and is what makes Per-Thread mode
                # cheap after the first build.
            }
            try:
                cur_stream = torch.cuda.current_stream()
                if cur_stream is not None and cur_stream != torch.cuda.default_stream():
                    trt_ep_options['user_compute_stream'] = str(int(cur_stream.cuda_stream))
            except Exception:
                pass
            try:
                sess_options = self._make_session_options()
                cuda_options = self._cuda_ep_provider_options(cudnn_algo='EXHAUSTIVE')
                sess = onnxruntime.InferenceSession(
                    onnx_path,
                    sess_options=sess_options,
                    providers=[
                        ('TensorrtExecutionProvider', trt_ep_options),
                        ('CUDAExecutionProvider', cuda_options),
                    ],
                )
            except Exception as e:
                print('[Models] inswapper_512 TRT-EP init failed (%s: %s); '
                      'falling back to CUDA EP' % (type(e).__name__, e))
                sess = None

        if sess is None:
            # inswapper_512 has a fixed 512×512 input → EXHAUSTIVE.
            cuda_options = self._cuda_ep_provider_options(cudnn_algo='EXHAUSTIVE')
            sess_options = self._make_session_options()
            sess = onnxruntime.InferenceSession(
                onnx_path,
                sess_options=sess_options,
                providers=[('CUDAExecutionProvider', cuda_options)],
            )

        active = sess.get_providers()
        self._swapper_512_uses_trt = bool(active) and active[0] == 'TensorrtExecutionProvider'
        # Detect static-batch ONNX exports. inswapper_512_level2.onnx
        # typically ships with batch=1 (the level-2 variant is per-call).
        # If batch is symbolic, dynamic batch is supported.
        try:
            inp_shape = sess.get_inputs()[0].shape
            shape0 = inp_shape[0]
            self._swapper_512_batch_unsupported = (
                isinstance(shape0, int) and shape0 == 1
            )
        except Exception:
            self._swapper_512_batch_unsupported = True
        backend = 'TensorrtExecutionProvider' if self._swapper_512_uses_trt else 'CUDAExecutionProvider'
        batch_note = ' static-batch=1' if self._swapper_512_batch_unsupported else ''
        print(f'{self._load_tag("swap")} [Models] inswapper_512: {backend}{batch_note}')
        return sess

    def _get_swapper_512_session(self):
        return self._get_model_session(
            'swapper_512', 'swapper_512_model', self._create_swapper_512_session,
        )

    # ===== inswapper_256_phase1 (256x256 native, emap latent) =============
    # Single-tile native 256 swapper. Same source conditioning as the 128 /
    # 512 inswappers (emap-projected latent fed as `source`). I/O tensor
    # names are `target256` (input, 1×3×256×256), `source` (input, 1×512),
    # and `p2_refined` (output, 1×3×256×256). The ONNX declares a dynamic
    # spatial input, so the TRT-EP branch pins a fixed 256 profile (we only
    # ever feed 256×256).

    def run_swapper_256(self, image, embedding, output):
        swapper_256_session = self._get_swapper_256_session()

        io_binding = swapper_256_session.io_binding()
        io_binding.bind_input(
            name='target256', device_type='cuda', device_id=0,
            element_type=np.float32, shape=(1, 3, 256, 256),
            buffer_ptr=image.data_ptr(),
        )
        io_binding.bind_input(
            name='source', device_type='cuda', device_id=0,
            element_type=np.float32, shape=(1, 512),
            buffer_ptr=embedding.data_ptr(),
        )
        io_binding.bind_output(
            name='p2_refined', device_type='cuda', device_id=0,
            element_type=np.float32, shape=(1, 3, 256, 256),
            buffer_ptr=output.data_ptr(),
        )

        if self._should_drain_syncvec():
            self.syncvec.cpu()
        swapper_256_session.run_with_iobinding(io_binding)

    def _create_swapper_256_session(self):
        """Build a fresh inswapper_256_phase1 ORT-TRT-EP session. The ONNX
        has a dynamic (1, 3, H, W) input; we only ever feed 256×256, so the
        TRT profile is pinned min=opt=max at 256. FP32 I/O; batch is always
        1 (run_swapper_256 is called once per face)."""
        try:
            onnxruntime.set_default_logger_severity(3)
        except (AttributeError, OSError):
            pass

        pref = self._backend_pref.get('swapper_256_model')
        want_trt = pref != 'onnx'
        onnx_path = self._mp('inswapper_256_phase1.onnx')
        sess = None

        if want_trt:
            cache_dir = os.path.abspath(self._mp('ort_trt_cache'))
            try:
                os.makedirs(cache_dir, exist_ok=True)
            except OSError:
                pass
            trt_ep_options = {
                'trt_max_workspace_size': 4 << 30,
                'trt_engine_cache_enable': True,
                'trt_engine_cache_path': cache_dir,
                'trt_timing_cache_enable': True,
                'trt_timing_cache_path': cache_dir,
                # FP16 is unsafe for this native-256 inswapper family. The
                # model normalizes with decomposed ReduceMean→Sub→Sqrt→Div
                # variance groups, and FP16 variance computes with catastrophic
                # cancellation → a blurry/noisy face. Build the whole engine in
                # FP32: the model ships FP32 weights (no FP16 export), so this
                # is its native precision, and TRT still beats CUDA-EP via
                # kernel fusion. Cost: ~2x the engine VRAM of an FP16 build
                # (notable in Per-Thread mode with N per-worker engines). (The
                # sibling phase2 variant additionally had InstanceNormalization
                # ops that defeated trt_layer_norm_fp32_fallback — FP32 is the
                # confirmed-safe path for these native-256 swappers.)
                'trt_fp16_enable': False,
                'trt_builder_optimization_level': 5,
                # Dynamic spatial input pinned to the single size we feed.
                # Only `target256` is dynamic; `source` (1, 512) is static
                # and needs no profile.
                'trt_profile_min_shapes': 'target256:1x3x256x256',
                'trt_profile_opt_shapes': 'target256:1x3x256x256',
                'trt_profile_max_shapes': 'target256:1x3x256x256',
            }
            try:
                cur_stream = torch.cuda.current_stream()
                if cur_stream is not None and cur_stream != torch.cuda.default_stream():
                    trt_ep_options['user_compute_stream'] = str(int(cur_stream.cuda_stream))
            except Exception:
                pass
            try:
                sess_options = self._make_session_options()
                cuda_options = self._cuda_ep_provider_options(cudnn_algo='EXHAUSTIVE')
                sess = onnxruntime.InferenceSession(
                    onnx_path,
                    sess_options=sess_options,
                    providers=[
                        ('TensorrtExecutionProvider', trt_ep_options),
                        ('CUDAExecutionProvider', cuda_options),
                    ],
                )
            except Exception as e:
                print('[Models] inswapper_256 TRT-EP init failed (%s: %s); '
                      'falling back to CUDA EP' % (type(e).__name__, e))
                sess = None

        if sess is None:
            # We only ever feed a fixed 256×256 → EXHAUSTIVE (one-time warmup).
            cuda_options = self._cuda_ep_provider_options(cudnn_algo='EXHAUSTIVE')
            sess_options = self._make_session_options()
            sess = onnxruntime.InferenceSession(
                onnx_path,
                sess_options=sess_options,
                providers=[('CUDAExecutionProvider', cuda_options)],
            )

        active = sess.get_providers()
        self._swapper_256_uses_trt = bool(active) and active[0] == 'TensorrtExecutionProvider'
        backend = 'TensorrtExecutionProvider' if self._swapper_256_uses_trt else 'CUDAExecutionProvider'
        print(f'{self._load_tag("swap")} [Models] inswapper_256_phase1: {backend}')
        return sess

    def _get_swapper_256_session(self):
        return self._get_model_session(
            'swapper_256', 'swapper_256_model', self._create_swapper_256_session,
        )

    def run_GFPGAN(self, image, output):
        if not self.GFPGAN_model:
            cuda_options = {"arena_extend_strategy": "kSameAsRequested", 'cudnn_conv_algo_search': 'DEFAULT'}
            sess_options = onnxruntime.SessionOptions()
            sess_options.log_severity_level = 3
            sess_options.enable_cpu_mem_arena = False

            # Fast load
            self.GFPGAN_model = onnxruntime.InferenceSession( self._mp("GFPGANv1.4.onnx"), sess_options, providers=[("CUDAExecutionProvider", cuda_options), 'CPUExecutionProvider'])

            # Normal load
            # self.GFPGAN_model = onnxruntime.InferenceSession( "./models/GFPGANv1.4.onnx", providers=self.providers)

        io_binding = self.GFPGAN_model.io_binding()
        io_binding.bind_input(name='input', device_type='cuda', device_id=0, element_type=np.float32, shape=(1,3,512,512), buffer_ptr=image.data_ptr())
        io_binding.bind_output(name='output', device_type='cuda', device_id=0, element_type=np.float32, shape=(1,3,512,512), buffer_ptr=output.data_ptr())  
                
        self.syncvec.cpu()          
        self.GFPGAN_model.run_with_iobinding(io_binding)                

    
    def run_GPEN_512(self, image, output):
        if not self.GPEN_512_model:
            self.GPEN_512_model = onnxruntime.InferenceSession( self._mp("GPEN-BFR-512.onnx"), providers=self.providers)
 
        io_binding = self.GPEN_512_model.io_binding() 
        io_binding.bind_input(name='input', device_type='cuda', device_id=0, element_type=np.float32, shape=(1,3,512,512), buffer_ptr=image.data_ptr())
        io_binding.bind_output(name='output', device_type='cuda', device_id=0, element_type=np.float32, shape=(1,3,512,512), buffer_ptr=output.data_ptr())            
        
        self.syncvec.cpu()          
        self.GPEN_512_model.run_with_iobinding(io_binding)   

    def run_GPEN_256(self, image, output):
        if not self.GPEN_256_model:
            self.GPEN_256_model = onnxruntime.InferenceSession( self._mp("GPEN-BFR-256.onnx"), providers=self.providers)

        io_binding = self.GPEN_256_model.io_binding()         
        io_binding.bind_input(name='input', device_type='cuda', device_id=0, element_type=np.float32, shape=(1,3,256,256), buffer_ptr=image.data_ptr())
        io_binding.bind_output(name='output', device_type='cuda', device_id=0, element_type=np.float32, shape=(1,3,256,256), buffer_ptr=output.data_ptr())
        
        self.syncvec.cpu()          
        self.GPEN_256_model.run_with_iobinding(io_binding) 

    def run_codeformer(self, image, output):   
        if not self.codeformer_model:    
            self.codeformer_model = onnxruntime.InferenceSession( self._mp("codeformer_fp16.onnx"), providers=self.providers)

        io_binding = self.codeformer_model.io_binding() 
        io_binding.bind_input(name='x', device_type='cuda', device_id=0, element_type=np.float32, shape=(1,3,512,512), buffer_ptr=image.data_ptr())
        w = np.array([0.9], dtype=np.double)
        io_binding.bind_cpu_input('w', w)
        io_binding.bind_output(name='y', device_type='cuda', device_id=0, element_type=np.float32, shape=(1,3,512,512), buffer_ptr=output.data_ptr())
        
        self.syncvec.cpu()          
        self.codeformer_model.run_with_iobinding(io_binding)        

    def run_occluder(self, image, output):    
        if not self.occluder_model:
            self.occluder_model = onnxruntime.InferenceSession(self._mp("occluder.onnx"), providers=self.providers)

        io_binding = self.occluder_model.io_binding()            
        io_binding.bind_input(name='img', device_type='cuda', device_id=0, element_type=np.float32, shape=(1,3,256,256), buffer_ptr=image.data_ptr())
        io_binding.bind_output(name='output', device_type='cuda', device_id=0, element_type=np.float32, shape=(1,1,256,256), buffer_ptr=output.data_ptr())   

        # torch.cuda.synchronize('cuda')  
        self.syncvec.cpu()         
        self.occluder_model.run_with_iobinding(io_binding)  


    def run_faceparser(self, image, output):    
        if not self.faceparser_model:
            self.faceparser_model = onnxruntime.InferenceSession(self._mp("faceparser_resnet34.onnx"), providers=self.providers)

        io_binding = self.faceparser_model.io_binding()            
        io_binding.bind_input(name='input', device_type='cuda', device_id=0, element_type=np.float32, shape=(1,3,512,512), buffer_ptr=image.data_ptr())
        io_binding.bind_output(name='output', device_type='cuda', device_id=0, element_type=np.float32, shape=(1,19,512,512), buffer_ptr=output.data_ptr())   

        # torch.cuda.synchronize('cuda')
        self.syncvec.cpu()
        self.faceparser_model.run_with_iobinding(io_binding)

    @staticmethod
    def _ort_type_to_np(type_str):
        """Map an ORT NodeArg type string ('tensor(float16)') to a numpy
        dtype. Defaults to float32 for anything unrecognised."""
        return np.float16 if 'float16' in type_str else np.float32

    def _load_dfl_xseg_model(self):
        """Lazy-load the user-supplied DeepFaceLab XSeg ONNX and introspect
        its I/O contract. DFL's native export is NHWC (1, res, res, 3) with
        an NHWC (1, res, res, 1) mask output; some community converters emit
        NCHW. Node names and resolution also vary, so read them off the graph
        rather than hardcoding."""
        self.dfl_xseg_model = onnxruntime.InferenceSession(
            self._mp("dfl_xseg.onnx"), providers=self.providers,
        )

        inp = self.dfl_xseg_model.get_inputs()[0]
        out = self.dfl_xseg_model.get_outputs()[0]
        self._dfl_xseg_in_name = inp.name
        self._dfl_xseg_out_name = out.name
        self._dfl_xseg_in_dtype = self._ort_type_to_np(inp.type)
        self._dfl_xseg_out_dtype = self._ort_type_to_np(out.type)

        # Decide layout + resolution from the input shape. Symbolic dims
        # (e.g. 'batch') come through as strings; the channel/spatial dims
        # are normally concrete. NCHW iff dim 1 == 3, else assume NHWC.
        shape = list(inp.shape)
        def _as_int(v, fallback):
            return v if isinstance(v, int) and v > 0 else fallback
        if len(shape) == 4 and shape[1] == 3:
            self._dfl_xseg_nhwc = False
            self._dfl_xseg_res = _as_int(shape[2], 256)
        else:
            self._dfl_xseg_nhwc = True
            self._dfl_xseg_res = _as_int(shape[1], 256)

        print(
            f"[Models] dfl_xseg.onnx loaded: res={self._dfl_xseg_res} "
            f"layout={'NHWC' if self._dfl_xseg_nhwc else 'NCHW'} "
            f"in='{self._dfl_xseg_in_name}'({inp.type}) "
            f"out='{self._dfl_xseg_out_name}'({out.type})"
        )

    def run_dfl_xseg(self, img):
        """Run the DeepFaceLab XSeg occlusion model on an aligned face.

        `img` is a (3, H, W) RGB float tensor in [0, 255] on cuda (the
        pipeline-aligned input face). Returns a (res, res) float32 mask in
        [0, 1] on cuda where 1 = keep the swap, 0 = let the original show
        through (DFL convention: XSeg paints the face region as foreground).

        Preprocessing matches DFL: BGR channel order, values in [0, 1].

        The model is user-supplied and may be absent; if so we warn once
        and return an all-ones mask (a no-op composite) rather than crashing
        the swap pipeline on every frame."""
        if not self.dfl_xseg_model:
            path = self._mp("dfl_xseg.onnx")
            if not os.path.exists(path):
                if not getattr(self, '_dfl_xseg_missing_warned', False):
                    self._dfl_xseg_missing_warned = True
                    print(
                        f"[Models] DFL XSeg mask enabled but '{path}' is "
                        "missing — skipping (drop your exported XSeg .onnx "
                        "there to use it)."
                    )
                return torch.ones(
                    (self._dfl_xseg_res, self._dfl_xseg_res),
                    dtype=torch.float32, device='cuda',
                )
            self._load_dfl_xseg_model()

        res = self._dfl_xseg_res

        # (3, H, W) RGB [0,255] -> (1, 3, res, res) BGR [0,1]. The aligned
        # face arrives as uint8; cast first since F.interpolate's bilinear
        # path doesn't implement the Byte dtype.
        x = img.unsqueeze(0).to(torch.float32)
        if x.shape[-1] != res or x.shape[-2] != res:
            x = torch.nn.functional.interpolate(
                x, size=(res, res), mode='bilinear', align_corners=False,
            )
        x = x.flip(1)            # RGB -> BGR (DFL trains on cv2/BGR faces)
        x = torch.div(x, 255.0)
        if self._dfl_xseg_nhwc:
            x = x.permute(0, 2, 3, 1)            # -> (1, res, res, 3)
        x = x.contiguous()
        if self._dfl_xseg_in_dtype == np.float16:
            x = x.half()
        else:
            x = x.float()

        if self._dfl_xseg_nhwc:
            in_shape = (1, res, res, 3)
            out_shape = (1, res, res, 1)
        else:
            in_shape = (1, 3, res, res)
            out_shape = (1, 1, res, res)

        out_torch_dtype = (
            torch.float16 if self._dfl_xseg_out_dtype == np.float16
            else torch.float32
        )
        outpred = torch.empty(
            out_shape, dtype=out_torch_dtype, device='cuda',
        ).contiguous()

        io_binding = self.dfl_xseg_model.io_binding()
        io_binding.bind_input(
            name=self._dfl_xseg_in_name, device_type='cuda', device_id=0,
            element_type=self._dfl_xseg_in_dtype, shape=in_shape,
            buffer_ptr=x.data_ptr(),
        )
        io_binding.bind_output(
            name=self._dfl_xseg_out_name, device_type='cuda', device_id=0,
            element_type=self._dfl_xseg_out_dtype, shape=out_shape,
            buffer_ptr=outpred.data_ptr(),
        )

        self.syncvec.cpu()
        self.dfl_xseg_model.run_with_iobinding(io_binding)

        return outpred.float().reshape(res, res)


    def detect_retinaface(self, img, max_num, score, input_size=640):
        with nvtx_range(f"detect_retinaface[s={input_size}]"):
            return self._detect_retinaface_inner(img, max_num, score, input_size)

    def _detect_retinaface_inner(self, img, max_num, score, input_size=640):
        # Resize image to fit within the input_size. Model is fully
        # convolutional so it accepts any multiple-of-32 edge length;
        # smaller input → linearly less compute → faster detect at the
        # cost of recall on small faces.
        #
        # Clamp to the configured TRT-EP profile range (320-640) when
        # TRT is active. The session is configured with that explicit
        # profile in _create_retinaface_session, so requests outside
        # the range would error inside ORT.
        with nvtx_range("rf_preprocess"):
            if self._retinaface_uses_trt and hasattr(self, '_retinaface_trt_input_max'):
                lo = self._retinaface_trt_input_min
                hi = self._retinaface_trt_input_max
                req = int(input_size)
                if req < lo or req > hi:
                    if not self._retinaface_engine_size_warned:
                        print(
                            '[Models] retinaface TRT-EP profile accepts '
                            f'sizes {lo}-{hi}; requested {req}. Clamping to '
                            f'{hi}.'
                        )
                        self._retinaface_engine_size_warned = True
                    req = max(lo, min(hi, req))
                input_size = req
            input_size = (int(input_size), int(input_size))
            # Plain Python float math — the previous torch.div on int args
            # wrapped both in CUDA tensors for a single scalar divide, paid
            # tensor-allocation overhead, and forced a downstream sync via
            # `.numpy()`. None of that is needed; these are scalars.
            im_ratio = float(img.size()[1]) / float(img.size()[2])

            model_ratio = float(input_size[1]) / input_size[0]
            if im_ratio>model_ratio:
                new_height = input_size[1]
                new_width = int(new_height / im_ratio)
            else:
                new_width = input_size[0]
                new_height = int(new_width * im_ratio)
            det_scale = float(new_height) / float(img.size()[1])

            # Resize input to (3, new_h, new_w) uint8 on GPU. This kernel
            # is the dominant cost of preprocessing; the rest of the chain
            # below is cheap arithmetic on the resized tile.
            img_resized = v2.functional.resize(
                img, [new_height, new_width], antialias=True,
            )

            # Pre-allocated per-worker letterbox buffer (1, 3, H, W) float32.
            # Reusing it across calls avoids the torch.zeros allocator hit
            # every frame; keeping the canonical (1, 3, H, W) shape end-to-
            # end avoids the HWC↔CHW permute + .contiguous() round-trip
            # the legacy path needed before the io_binding bind_input call.
            det_img = self._get_retinaface_input_buffer(input_size[0])
            # Zero the previous frame's letterbox padding. zero_() on the
            # cached buffer is one kernel and small; restricting to the
            # padding strips would need more launches than just clearing
            # the whole thing.
            det_img.zero_()
            # BGR swap + uint8→float + normalize + slice-write fused into
            # one chained expression. img_resized[[2, 1, 0]] is the BGR
            # reindex (advanced indexing returns a contiguous copy);
            # .float() materializes a float32 buffer; .sub_()/.div_() are
            # in-place on that fresh buffer (no extra allocations); .copy_()
            # writes the result into det_img's letterbox slot. Total: one
            # gather + one type-cast + two in-place ops + one copy_, vs the
            # legacy six-op chain (zero, slice, BGR, sub, div, contiguous).
            det_img[0, :, :new_height, :new_width].copy_(
                img_resized[[2, 1, 0]].float().sub_(127.5).div_(128.0)
            )

        with nvtx_range("rf_ort_run"):
            # det_10g has 9 outputs — 3 scores, 3 bbox deltas, 3 kps deltas
            # at strides 8/16/32. The downstream loop consumes them by index.
            _RF_OUTPUT_ORDER = (
                '448', '471', '494',  # scores @ stride 8 / 16 / 32
                '451', '474', '497',  # bbox deltas
                '454', '477', '500',  # kps deltas
            )

            # Pre-allocate output buffers on GPU. Shapes are deterministic
            # from input_size + stride (det_10g has 2 anchors per cell).
            # Binding them as buffer_ptr lets ORT write directly into our
            # torch tensors, so the entire post-processing stays on GPU —
            # no copy_outputs_to_cpu, no host round-trip before NMS.
            input_height = det_img.shape[2]
            input_width = det_img.shape[3]
            strides = (8, 16, 32)
            score_bufs = []
            bbox_bufs = []
            kps_bufs = []
            for s in strides:
                n_anchors = (input_height // s) * (input_width // s) * 2
                score_bufs.append(torch.empty((n_anchors, 1), dtype=torch.float32, device='cuda'))
                bbox_bufs.append(torch.empty((n_anchors, 4), dtype=torch.float32, device='cuda'))
                kps_bufs.append(torch.empty((n_anchors, 10), dtype=torch.float32, device='cuda'))

            retinaface_session = self._get_retinaface_session()
            io_binding = retinaface_session.io_binding()
            io_binding.bind_input(
                name='input.1', device_type='cuda', device_id=0,
                element_type=np.float32, shape=det_img.size(),
                buffer_ptr=det_img.data_ptr(),
            )
            for i, name in enumerate(_RF_OUTPUT_ORDER[:3]):
                buf = score_bufs[i]
                io_binding.bind_output(
                    name=name, device_type='cuda', device_id=0,
                    element_type=np.float32, shape=tuple(buf.shape),
                    buffer_ptr=buf.data_ptr(),
                )
            for i, name in enumerate(_RF_OUTPUT_ORDER[3:6]):
                buf = bbox_bufs[i]
                io_binding.bind_output(
                    name=name, device_type='cuda', device_id=0,
                    element_type=np.float32, shape=tuple(buf.shape),
                    buffer_ptr=buf.data_ptr(),
                )
            for i, name in enumerate(_RF_OUTPUT_ORDER[6:9]):
                buf = kps_bufs[i]
                io_binding.bind_output(
                    name=name, device_type='cuda', device_id=0,
                    element_type=np.float32, shape=tuple(buf.shape),
                    buffer_ptr=buf.data_ptr(),
                )

            # Conditional torch-stream drain. Skipped when the session was
            # built with user_compute_stream pointing at the worker's
            # current torch stream (Per-Thread + worker stream): same
            # stream means implicit ordering.
            if self._should_drain_syncvec():
                self.syncvec.cpu()
            retinaface_session.run_with_iobinding(io_binding)

        with nvtx_range("rf_postprocess"):
            # ===== GPU-side post-processing =====
            # Everything from here runs in torch on the worker's CUDA
            # stream — bbox decode, kps decode, score threshold, cross-
            # stride concat, sort, scale-back, NMS. The GIL is released
            # during the CUDA kernels, so other workers don't queue up
            # waiting for it during the bulk of detect_retinaface's tail
            # (which was the dominant source of "User Request" GIL waits
            # in the 2026-05-21 nsys trace).
            scores_list = []
            bboxes_list = []
            kpss_list = []
            for idx, stride in enumerate(strides):
                scores_t = score_bufs[idx].view(-1)              # (N,)
                bbox_preds = bbox_bufs[idx] * stride             # (N, 4)
                kps_preds = kps_bufs[idx] * stride               # (N, 10)
                height = input_height // stride
                width = input_width // stride
                anchor_centers = self._get_retinaface_gpu_anchors(
                    height, width, stride,
                )                                                # (N, 2)

                pos_mask = scores_t >= score

                # Bbox decode: anchor ± delta. Sign pattern [-x1, -y1, +x2,
                # +y2] matches det_10g's FCOS-style face regression head.
                ax = anchor_centers[:, 0]
                ay = anchor_centers[:, 1]
                bboxes = torch.stack(
                    [
                        ax - bbox_preds[:, 0],
                        ay - bbox_preds[:, 1],
                        ax + bbox_preds[:, 2],
                        ay + bbox_preds[:, 3],
                    ],
                    dim=-1,
                )                                                # (N, 4)

                # KPS decode: 5 (x, y) pairs per anchor, all sharing the
                # same anchor offset. The numpy original looped over pairs;
                # vectorized this is `view + broadcast-add`.
                kpss = kps_preds.view(-1, 5, 2) + anchor_centers.unsqueeze(1)

                scores_list.append(scores_t[pos_mask])
                bboxes_list.append(bboxes[pos_mask])
                kpss_list.append(kpss[pos_mask])

            all_scores = torch.cat(scores_list)
            all_bboxes = torch.cat(bboxes_list)
            all_kpss = torch.cat(kpss_list)

            if all_scores.numel() == 0:
                return np.zeros((0, 5, 2), dtype=np.float32)

            # Sort by score descending (matches the np.argsort()[::-1] in
            # the old code), then NMS in IoU space. det_scale division
            # rescales from the letterboxed model space back to original
            # image coordinates.
            order = torch.argsort(all_scores, descending=True)
            all_scores = all_scores[order]
            all_bboxes = all_bboxes[order] / det_scale
            all_kpss = all_kpss[order] / det_scale

            keep = torchvision.ops.nms(all_bboxes, all_scores, iou_threshold=0.4)

            # max_num truncation with area-minus-center-distance weighting,
            # matches the legacy heuristic. Preserves the original's quirk
            # of using det_img.shape[0]/[1] (the batch and channel dims of
            # the (1, 3, H, W) tensor — values 0 and 1) as the "image
            # center"; the center-distance term is small relative to area
            # so the practical difference is negligible.
            if max_num > 0 and keep.shape[0] > max_num:
                kept_bboxes = all_bboxes[keep]
                area = (kept_bboxes[:, 2] - kept_bboxes[:, 0]) * (
                    kept_bboxes[:, 3] - kept_bboxes[:, 1]
                )
                cy = float(det_img.shape[0] // 2)
                cx = float(det_img.shape[1] // 2)
                dx = (kept_bboxes[:, 0] + kept_bboxes[:, 2]) / 2 - cy
                dy = (kept_bboxes[:, 1] + kept_bboxes[:, 3]) / 2 - cx
                offset_dist_squared = dx * dx + dy * dy
                values = area - offset_dist_squared * 2.0
                bindex = torch.argsort(values, descending=True)[:max_num]
                keep = keep[bindex]

            # Single device boundary at the very end. Everything before
            # was queued on the worker's CUDA stream.
            return all_kpss[keep].cpu().numpy()

    def detect_scrdf(self, img, max_num, score, input_size=640):
        # Resize image to fit within the input_size. Model is fully
        # convolutional so it accepts any multiple-of-32 edge length;
        # smaller input → linearly less compute → faster detect at the
        # cost of recall on small faces.
        input_size = (int(input_size), int(input_size))
        # Plain Python float math — the previous torch.div on int args
        # wrapped both in CUDA tensors for a single scalar divide, paid
        # tensor-allocation overhead, and forced a downstream sync via
        # `.numpy()`. None of that is needed; these are scalars.
        im_ratio = float(img.size()[1]) / float(img.size()[2])

        model_ratio = float(input_size[1]) / input_size[0]
        if im_ratio>model_ratio:
            new_height = input_size[1]
            new_width = int(new_height / im_ratio)
        else:
            new_width = input_size[0]
            new_height = int(new_width * im_ratio)
        det_scale = float(new_height) / float(img.size()[1])

        # Functional form avoids constructing a fresh v2.Resize
        # transform object on every call.
        img = v2.functional.resize(
            img, [new_height, new_width], antialias=True,
        )
        img = img.permute(1,2,0)

        det_img = torch.zeros((input_size[1], input_size[0], 3), dtype=torch.float32, device='cuda:0')
        det_img[:new_height,:new_width,  :] = img

        # Switch to BGR and normalize
        det_img = det_img[:, :, [2,1,0]]
        det_img = torch.sub(det_img, 127.5)
        det_img = torch.div(det_img, 128.0)
        det_img = det_img.permute(2, 0, 1) #3,128,128
        
        # Prepare data and find model parameters 
        det_img = torch.unsqueeze(det_img, 0).contiguous()
        input_name = self.scrdf_model.get_inputs()[0].name
        
        outputs = self.scrdf_model.get_outputs()
        output_names = []
        for o in outputs:
            output_names.append(o.name)
        
        io_binding = self.scrdf_model.io_binding() 
        io_binding.bind_input(name=input_name, device_type='cuda', device_id=0, element_type=np.float32,  shape=det_img.size(), buffer_ptr=det_img.data_ptr())
        
        for i in range(len(output_names)):
            io_binding.bind_output(output_names[i], 'cuda') 
        
        # Sync and run model
        syncvec = self.syncvec.cpu()        
        self.scrdf_model.run_with_iobinding(io_binding)
        
        net_outs = io_binding.copy_outputs_to_cpu()

        input_height = det_img.shape[2]
        input_width = det_img.shape[3]
        
        fmc = 3
        center_cache = {}
        scores_list = []
        bboxes_list = []
        kpss_list = []
        for idx, stride in enumerate([8, 16, 32]):
            scores = net_outs[idx]
            bbox_preds = net_outs[idx+fmc]
            bbox_preds = bbox_preds * stride

            kps_preds = net_outs[idx+fmc*2] * stride
            height = input_height // stride
            width = input_width // stride
            K = height * width
            key = (height, width, stride)
            if key in center_cache:
                anchor_centers = center_cache[key]
            else:
                anchor_centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
                anchor_centers = (anchor_centers * stride).reshape( (-1, 2) )
                anchor_centers = np.stack([anchor_centers]*2, axis=1).reshape( (-1,2) )
                if len(center_cache)<100:
                    center_cache[key] = anchor_centers
            
            pos_inds = np.where(scores>=score)[0]

            x1 = anchor_centers[:, 0] - bbox_preds[:, 0]
            y1 = anchor_centers[:, 1] - bbox_preds[:, 1]
            x2 = anchor_centers[:, 0] + bbox_preds[:, 2]
            y2 = anchor_centers[:, 1] + bbox_preds[:, 3]

            bboxes = np.stack([x1, y1, x2, y2], axis=-1)  
            
            pos_scores = scores[pos_inds]
            pos_bboxes = bboxes[pos_inds]
            scores_list.append(pos_scores)
            bboxes_list.append(pos_bboxes)

            preds = []
            for i in range(0, kps_preds.shape[1], 2):
                px = anchor_centers[:, i%2] + kps_preds[:, i]
                py = anchor_centers[:, i%2+1] + kps_preds[:, i+1]

                preds.append(px)
                preds.append(py)
            kpss = np.stack(preds, axis=-1) 
            #kpss = kps_preds
            kpss = kpss.reshape( (kpss.shape[0], -1, 2) )
            pos_kpss = kpss[pos_inds]
            kpss_list.append(pos_kpss)

        scores = np.vstack(scores_list)
        scores_ravel = scores.ravel()
        order = scores_ravel.argsort()[::-1]

        # det_scale is a Python float now (used to be a torch tensor
        # forced through .numpy() here, which was a hidden sync point).
        bboxes = np.vstack(bboxes_list) / det_scale

        kpss = np.vstack(kpss_list) / det_scale
        pre_det = np.hstack((bboxes, scores)).astype(np.float32, copy=False)
        pre_det = pre_det[order, :]
        
        # torchvision.ops.nms (CUDA): see detect_retinaface for the
        # bit-exactness note on the +1 area convention.
        thresh = 0.4
        if pre_det.shape[0] == 0:
            keep = []
        else:
            boxes_t = torch.from_numpy(pre_det[:, :4]).contiguous().cuda()
            scores_t = torch.from_numpy(pre_det[:, 4]).contiguous().cuda()
            keep_t = torchvision.ops.nms(boxes_t, scores_t, iou_threshold=thresh)
            keep = keep_t.cpu().tolist()

        det = pre_det[keep, :]

        kpss = kpss[order,:,:]
        kpss = kpss[keep,:,:]

        if max_num > 0 and det.shape[0] > max_num:
            area = (det[:, 2] - det[:, 0]) * (det[:, 3] -
                                                    det[:, 1])
            det_img_center = det_img.shape[0] // 2, det_img.shape[1] // 2
            offsets = np.vstack([
                (det[:, 0] + det[:, 2]) / 2 - det_img_center[1],
                (det[:, 1] + det[:, 3]) / 2 - det_img_center[0]
            ])
            offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)

            values = area - offset_dist_squared * 2.0  # some extra weight on the centering
            bindex = np.argsort(values)[::-1]  # some extra weight on the centering
            bindex = bindex[0:max_num]

            if kpss is not None:
                kpss = kpss[bindex, :]

        return kpss

    def recognize(self, img, face_kps, dim):
        with nvtx_range("rec_preprocess"):
            # Find transform
            dst = self.arcface_dst.copy() * dim

            tform = trans.SimilarityTransform.from_estimate(face_kps, dst)

            # Transform
            img = v2.functional.affine(img, tform.rotation*57.2958, (tform.translation[0], tform.translation[1]) , tform.scale, 0, center = (0,0) )
            img = v2.functional.crop(img, 0,0, dim*112, dim*112)

            # Switch to BGR and normalize
            img = img.permute(1,2,0) #112,112,3
            cropped_image = img
            img = img[:, :, [2,1,0]]
            img = torch.sub(img, 127.5)
            img = torch.div(img, 127.5) #dim*112, dim*112, 3

        with nvtx_range("rec_ort_run"):
            temp_holder = []
            recognition_session = self._get_recognition_session()
            output = torch.empty((1, 512), dtype=torch.float32, device='cuda').contiguous()
            io_binding = recognition_session.io_binding()
            io_binding.bind_output(
                name='683', device_type='cuda', device_id=0,
                element_type=np.float32, shape=(1, 512),
                buffer_ptr=output.data_ptr(),
            )

            drain = self._should_drain_syncvec()
            for j in range(dim):
                for i in range(dim):
                    input = img[j::dim, i::dim].permute(2, 0, 1)
                    input = torch.unsqueeze(input, 0).contiguous()
                    io_binding.bind_input(
                        name='input.1', device_type='cuda', device_id=0,
                        element_type=np.float32, shape=(1, 3, 112, 112),
                        buffer_ptr=input.data_ptr(),
                    )
                    if drain:
                        self.syncvec.cpu()
                    recognition_session.run_with_iobinding(io_binding)
                    temp_holder.append(torch.squeeze(output).cpu().numpy())

            # Return embedding
            out = np.mean(temp_holder, 0)
            return out, cropped_image
        
    def resnet50(self, image, score=.5):   
        if not self.resnet50_model:
            self.resnet50_model = onnxruntime.InferenceSession(self._mp("res50.onnx"), providers=self.providers)
            
            feature_maps = [[64, 64], [32, 32], [16, 16]]
            min_sizes = [[16, 32], [64, 128], [256, 512]]
            steps = [8, 16, 32]
            image_size = 512
            
            for k, f in enumerate(feature_maps):
                min_size_array = min_sizes[k]
                for i, j in product(range(f[0]), range(f[1])):
                    for min_size in min_size_array:
                        s_kx = min_size / image_size
                        s_ky = min_size / image_size
                        dense_cx = [x * steps[k] / image_size for x in [j + 0.5]]
                        dense_cy = [y * steps[k] / image_size for y in [i + 0.5]]
                        for cy, cx in product(dense_cy, dense_cx):
                            self.anchors += [cx, cy, s_kx, s_ky]   

        # image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        image = image.permute(1,2,0)
        
        # image = image - [104, 117, 123]
        mean = torch.tensor([104, 117, 123], dtype=torch.float32, device='cuda')
        image = torch.sub(image, mean)
        
        # image = image.transpose(2, 0, 1)
        # image = np.float32(image[np.newaxis,:,:,:])
        image = image.permute(2,0,1)
        image = torch.unsqueeze(v2.Resize((512, 512), antialias=False)(image), 0)

        height, width = (512, 512)
        tmp = [width, height, width, height, width, height, width, height, width, height]
        scale1 = torch.tensor(tmp, dtype=torch.float32, device='cuda')
        
        # ort_inputs = {"input": image}        
        conf = torch.empty((1,10752,2), dtype=torch.float32, device='cuda').contiguous()
        landmarks = torch.empty((1,10752,10), dtype=torch.float32, device='cuda').contiguous()

        io_binding = self.resnet50_model.io_binding() 
        io_binding.bind_input(name='input', device_type='cuda', device_id=0, element_type=np.float32, shape=(1,3,512,512), buffer_ptr=image.data_ptr())
        io_binding.bind_output(name='conf', device_type='cuda', device_id=0, element_type=np.float32, shape=(1,10752,2), buffer_ptr=conf.data_ptr())
        io_binding.bind_output(name='landmarks', device_type='cuda', device_id=0, element_type=np.float32, shape=(1,10752,10), buffer_ptr=landmarks.data_ptr())
        
        # _, conf, landmarks = self.resnet_model.run(None, ort_inputs)        
        torch.cuda.synchronize('cuda')
        self.resnet50_model.run_with_iobinding(io_binding)        
        

        # conf = torch.from_numpy(conf)
        # scores = conf.squeeze(0).numpy()[:, 1]
        scores = torch.squeeze(conf)[:, 1]
        
        # landmarks = torch.from_numpy(landmarks)
        # landmarks = landmarks.to('cuda')        

        priors = torch.tensor(self.anchors).view(-1, 4)
        priors = priors.to('cuda')

        # pre = landmarks.squeeze(0) 
        pre = torch.squeeze(landmarks, 0)
        
        tmp = (priors[:, :2] + pre[:, :2] * 0.1 * priors[:, 2:], priors[:, :2] + pre[:, 2:4] * 0.1 * priors[:, 2:], priors[:, :2] + pre[:, 4:6] * 0.1 * priors[:, 2:], priors[:, :2] + pre[:, 6:8] * 0.1 * priors[:, 2:], priors[:, :2] + pre[:, 8:10] * 0.1 * priors[:, 2:])
        landmarks = torch.cat(tmp, dim=1)
        # landmarks = landmarks * scale1
        landmarks = torch.mul(landmarks, scale1)

        landmarks = landmarks.cpu().numpy()  

        # ignore low scores
        inds = torch.where(scores>score)[0]
        inds = inds.cpu().numpy()  
        scores = scores.cpu().numpy()  
        
        landmarks, scores = landmarks[inds], scores[inds]

        # sort
        order = scores.argsort()[::-1]
        landmarks = landmarks[order][0]

        return np.array([[landmarks[i], landmarks[i + 1]] for i in range(0,10,2)])