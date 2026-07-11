"""
Inswapper ONNX patch helper for the FP16-sensitivity probe.

The four main pipeline models (inswapper_128/256, retinaface, arcface) all
route through ORT's TensorrtExecutionProvider now, which builds and caches
its own engines internally under models/ort_trt_cache/ — there is no
hand-built engine path left in the runtime. What remains in this module is:

  * the models-folder path constants + set_models_folder() rebinding that
    Models.set_models_folder() calls, and
  * _patch_inswapper_for_dynamic_batch(), reused by
    rope/scripts/probe_inswapper_fp16.py so its probe engine has the same
    dynamic-batch I/O semantics as the production ORT-TRT-EP engine.

(The LivePortrait TRT path — grid_sample_3d plugin, hand-built LP engines,
TRTRunner, LivePortraitTRT — was removed when the LivePortrait feature was
dropped.)

The `onnx` package is imported lazily inside the function so that
`import rope.TensorRTEngine` is safe even where it isn't installed.
"""

import os


_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Models folder used to derive inswapper paths. Defaults to the repo's
# ./models directory; Models.set_models_folder() rebinds this and the
# downstream INSWAPPER_* constants when the user picks a different
# location in the Settings tab.
_MODELS_FOLDER = os.path.join(_REPO_ROOT, 'models')

INSWAPPER_ONNX = os.path.join(_MODELS_FOLDER, 'inswapper_128.onnx')
INSWAPPER_ENGINE = os.path.join(_MODELS_FOLDER, 'inswapper_128.engine')
INSWAPPER_512_ONNX = os.path.join(_MODELS_FOLDER, 'inswapper_512_level2.onnx')
INSWAPPER_512_ENGINE = os.path.join(_MODELS_FOLDER, 'inswapper_512_level2.engine')
RETINAFACE_ONNX = os.path.join(_MODELS_FOLDER, 'det_10g.onnx')
RETINAFACE_ENGINE = os.path.join(_MODELS_FOLDER, 'det_10g.engine')
ARCFACE_ONNX = os.path.join(_MODELS_FOLDER, 'w600k_r50.onnx')
ARCFACE_ENGINE = os.path.join(_MODELS_FOLDER, 'w600k_r50.engine')


def set_models_folder(path):
    """Re-point the swapper / detector / recognizer ONNX/engine paths at
    a new folder.

    Called from Models.set_models_folder so the TRT module sees the same
    folder as the ORT loaders. Both constants and the internal
    _MODELS_FOLDER are updated. Idempotent."""
    global _MODELS_FOLDER
    global INSWAPPER_ONNX, INSWAPPER_ENGINE
    global INSWAPPER_512_ONNX, INSWAPPER_512_ENGINE
    global RETINAFACE_ONNX, RETINAFACE_ENGINE
    global ARCFACE_ONNX, ARCFACE_ENGINE
    if not path:
        return
    _MODELS_FOLDER = path
    INSWAPPER_ONNX = os.path.join(_MODELS_FOLDER, 'inswapper_128.onnx')
    INSWAPPER_ENGINE = os.path.join(_MODELS_FOLDER, 'inswapper_128.engine')
    INSWAPPER_512_ONNX = os.path.join(_MODELS_FOLDER, 'inswapper_512_level2.onnx')
    INSWAPPER_512_ENGINE = os.path.join(_MODELS_FOLDER, 'inswapper_512_level2.engine')
    RETINAFACE_ONNX = os.path.join(_MODELS_FOLDER, 'det_10g.onnx')
    RETINAFACE_ENGINE = os.path.join(_MODELS_FOLDER, 'det_10g.engine')
    ARCFACE_ONNX = os.path.join(_MODELS_FOLDER, 'w600k_r50.onnx')
    ARCFACE_ENGINE = os.path.join(_MODELS_FOLDER, 'w600k_r50.engine')


def _patch_inswapper_for_dynamic_batch(onnx_path, log=print):
    """Rewrite target/source/output first-dim from literal 1 to dim_param='batch'.

    The inswapper graph body is dynamic-safe (zero Reshape, zero Concat,
    zero axis-0 Slice — verified by graph audit), so making the I/O
    declarations symbolic is the only patch needed. With this patched
    ONNX + an optimization profile spanning the real polyphase batch
    range, one engine handles n ∈ {1, 4, 16}."""
    import onnx
    m = onnx.load(onnx_path)
    patched = []
    for tensors in (m.graph.input, m.graph.output):
        for t in tensors:
            d0 = t.type.tensor_type.shape.dim[0]
            d0.ClearField('dim_value')
            d0.dim_param = 'batch'
            patched.append(t.name)
    log('  patched batch dim -> symbolic on: %s' % ', '.join(patched))
    return m.SerializeToString()
