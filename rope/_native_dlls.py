"""Native-DLL bootstrap for Windows.

Why this exists
---------------
ORT's TensorrtExecutionProvider tries to load nvinfer_10.dll at session-
creation time via the OS loader, which only searches PATH and explicitly
registered directories. The pip-installed `tensorrt_libs` package puts
its DLLs in `venv\\Lib\\site-packages\\tensorrt_libs\\` — not on PATH —
so the load fails and ORT silently falls back to CUDA EP with a warning:

    EP Error ... RegisterTensorRTPluginsAsCustomOps Please install
    TensorRT libraries as mentioned in the GPU requirements page, make
    sure they're in the PATH or LD_LIBRARY_PATH ...

We avoid that by calling `os.add_dll_directory()` on the relevant pip
package directories at module import time. The list is curated rather
than auto-discovered because (a) discovery walks site-packages which is
slow, and (b) we want the registration order deterministic.

Idempotent and safe to call repeatedly. On non-Windows platforms it's a
no-op — Linux/macOS rely on rpath / LD_LIBRARY_PATH which the wheel
packaging sets up correctly.

Importing `rope` (or any submodule) triggers the bootstrap via
`rope/__init__.py`, so the app, the test scripts, and the bisection
probe all benefit without each one having to remember to call it.

Curated package list
--------------------
- `tensorrt_libs` — TRT runtime (nvinfer_*.dll, nvonnxparser_10.dll,
  nvinfer_plugin_10.dll). Required by ORT's TRT-EP.

Other pip-distributed native libs we DON'T add here:
- `torch/lib` — torch loads its own libs from this path internally.
- `onnxruntime/capi` — same; ORT manages its own DLL loading.
- `torchcodec`, `torchvision` — self-contained.
- CUDA Toolkit DLLs (cudart, cublas, etc.) — typically on PATH from
  the system CUDA install. If pip-installed via `nvidia-*` packages
  in the future, add them to the list below.
"""

from __future__ import annotations

import importlib
import os
import sys


_NATIVE_DLL_PACKAGES = (
    'tensorrt_libs',
)


# ORT's default-level logger (used by EPs, BFCArena, graph optimizer)
# defaults to INFO on the onnxruntime-gpu wheels we use, which produces
# ~hundreds of log lines on every fresh session + memory pool
# expansion. Force ERROR before ORT's first import anywhere in the
# process. Plain assignment, not setdefault — we want to override a
# stray parent-env setting. Per-session severity is set on the
# SessionOptions in Models.py as a belt-and-braces measure.
os.environ['ORT_LOG_SEVERITY_LEVEL'] = '3'


_BOOTSTRAPPED = False


def ensure_native_dll_search_path() -> list[str]:
    """Idempotent. Returns the list of directories actually added (for
    diagnostic / logging purposes)."""
    global _BOOTSTRAPPED
    if _BOOTSTRAPPED:
        return []
    _BOOTSTRAPPED = True

    if sys.platform != 'win32':
        return []

    added: list[str] = []
    for pkg_name in _NATIVE_DLL_PACKAGES:
        try:
            mod = importlib.import_module(pkg_name)
        except ImportError:
            continue
        if not getattr(mod, '__file__', None):
            continue
        pkg_dir = os.path.dirname(mod.__file__)
        if not os.path.isdir(pkg_dir):
            continue
        try:
            os.add_dll_directory(pkg_dir)
        except (OSError, FileNotFoundError):
            # add_dll_directory raises FileNotFoundError on Windows
            # for nonexistent paths; we already checked isdir but be
            # defensive in case the dir got removed between checks.
            continue
        added.append(pkg_dir)

    if added:
        # Print once at startup so the user can confirm. Quiet if no
        # packages were found — the user might intentionally not have
        # TRT installed and ORT will route to CUDA EP transparently.
        print('[rope] native DLL search path: added '
              + ', '.join(os.path.basename(p) for p in added))

    return added
