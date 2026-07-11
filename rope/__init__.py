"""rope package init.

Side effect at import time: registers pip-installed native DLL
directories with the Windows DLL search path so ORT's
TensorrtExecutionProvider and similar GPU-backed providers can find
their runtime libraries (e.g. nvinfer_10.dll inside the tensorrt_libs
pip package). See rope/_native_dlls.py for the full story.

Runs once, idempotent, no-op on non-Windows platforms.
"""

from rope._native_dlls import ensure_native_dll_search_path as _ensure

_ensure()
del _ensure
