"""NVTX annotation helper — labels Nsight Systems traces.

Nsight Systems captures every CUDA driver event, but raw kernel
launches don't carry any application-level meaning. NVTX ranges
(annotations) bracket sections of code so the trace shows labeled
bands instead of an undifferentiated kernel timeline.

Usage:

    from rope._nvtx import nvtx_range

    with nvtx_range("swap_core"):
        ...

When no profiler is attached the calls degrade to a near-zero-cost
no-op driver call. If torch.cuda.nvtx itself isn't available (CPU-
only torch build, or unusual install), this module collapses to a
pure-Python no-op so callers don't need to guard their imports.
"""

from __future__ import annotations

from contextlib import contextmanager

try:
    import torch
    _push = torch.cuda.nvtx.range_push
    _pop = torch.cuda.nvtx.range_pop
except (ImportError, AttributeError):
    def _push(_msg: str) -> None:  # type: ignore[misc]
        return None

    def _pop() -> None:  # type: ignore[misc]
        return None


@contextmanager
def nvtx_range(name: str):
    """Push a named NVTX range; pop on exit. Cheap when no profiler
    is attached. Use as `with nvtx_range("section"): ...`."""
    _push(name)
    try:
        yield
    finally:
        _pop()
