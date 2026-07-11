"""CUDA-GL interop bridge for zero-bounce video preview.

Maps a GL_RGBA8 texture as a CUDA graphics resource and copies torch CUDA
tensors straight into it. Skips the GPU→CPU→GPU bounce that
`tensor.cpu().numpy()` + glTexSubImage2D would otherwise do.

Constraints baked in here:
- The GL texture MUST be GL_RGBA8. cudaGraphicsGLRegisterImage rejects
  3-channel formats. We allocate an internal 4-channel torch buffer and
  copy the 3-channel input into its first 3 channels each frame; the
  alpha column is initialized once to 255.
- The GL context must be current when register_texture / upload / unmap
  is called. The caller (PreviewWidget) drives this from paintGL.
- Re-registration is automatic whenever (texture_id, width, height) changes.

Public surface:
    available() -> bool                — cuda-python importable
    CudaGLBridge:
        register_texture(tex_id, w, h)
        unregister()
        upload_from_cuda_tensor(t_chw or t_hwc) -> True if used CUDA path
        last_upload_ms

If anything goes wrong (driver mismatch, registration error, etc.), the
bridge marks itself disabled — subsequent uploads return False and the
caller falls back to CPU bounce permanently.
"""

from __future__ import annotations

import time
from typing import Optional

try:
    from cuda.bindings import runtime as cudart
    _CUDA_PY_AVAILABLE = True
except ImportError:
    cudart = None  # type: ignore
    _CUDA_PY_AVAILABLE = False

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore
    _TORCH_AVAILABLE = False


def available() -> bool:
    """Whether the runtime libs needed for the fast path are importable."""
    return _CUDA_PY_AVAILABLE and _TORCH_AVAILABLE and (
        torch is not None and torch.cuda.is_available()
    )


def _check(status, *, what: str) -> None:
    """Raise on a non-success cudaError_t. cuda-python returns a tuple
    (status, *outputs) for most calls."""
    if isinstance(status, tuple):
        err = status[0]
    else:
        err = status
    if int(err) != 0:
        try:
            name = cudart.cudaGetErrorName(err)[1].decode("utf-8", "replace")
            msg = cudart.cudaGetErrorString(err)[1].decode("utf-8", "replace")
        except Exception:
            name, msg = str(err), ""
        raise RuntimeError(f"{what} failed: {name} ({msg})")


class CudaGLBridge:
    def __init__(self):
        self._resource = None       # cudaGraphicsResource_t
        self._tex_id: int = 0
        self._tex_w: int = 0
        self._tex_h: int = 0
        self._rgba_buffer = None    # torch.Tensor (H, W, 4) uint8 on CUDA
        self._disabled: bool = not available()
        self.last_upload_ms: float = 0.0
        self.fail_reason: str = "" if available() else "cuda-python or torch.cuda unavailable"

    # ----- Registration ------------------------------------------------------

    def register_texture(self, texture_id: int, width: int, height: int) -> bool:
        """Register a GL texture as a CUDA resource. Returns False if the
        bridge is disabled or registration failed (in which case the bridge
        is marked permanently disabled).
        """
        if self._disabled:
            return False
        if (
            self._resource is not None
            and texture_id == self._tex_id
            and width == self._tex_w
            and height == self._tex_h
        ):
            return True

        # Unregister any previous binding
        self.unregister()

        # GL_TEXTURE_2D = 0x0DE1
        GL_TEXTURE_2D = 0x0DE1
        try:
            status, resource = cudart.cudaGraphicsGLRegisterImage(
                texture_id,
                GL_TEXTURE_2D,
                cudart.cudaGraphicsRegisterFlags.cudaGraphicsRegisterFlagsWriteDiscard,
            )
            _check(status, what="cudaGraphicsGLRegisterImage")
        except Exception as exc:
            self._disabled = True
            self.fail_reason = f"registerImage: {exc}"
            return False

        self._resource = resource
        self._tex_id = texture_id
        self._tex_w = width
        self._tex_h = height

        # (Re)allocate the 4-channel staging buffer on CUDA, prefill alpha.
        device = torch.device("cuda")
        self._rgba_buffer = torch.empty((height, width, 4), dtype=torch.uint8, device=device)
        self._rgba_buffer[..., 3] = 255
        return True

    def unregister(self) -> None:
        if self._resource is None:
            return
        try:
            status = cudart.cudaGraphicsUnregisterResource(self._resource)
            _check(status, what="cudaGraphicsUnregisterResource")
        except Exception as exc:
            # Don't propagate — best-effort cleanup. Log via fail_reason
            # for diagnostics but keep the bridge usable.
            self.fail_reason = f"unregister: {exc}"
        finally:
            self._resource = None
            self._tex_id = 0

    # ----- Upload ------------------------------------------------------------

    def upload_from_cuda_tensor(self, tensor) -> bool:
        """Copy a torch CUDA tensor of shape (H, W, 3) uint8 into the
        registered GL texture. Returns True if the CUDA path was used;
        False if the caller should fall back to CPU bounce.
        """
        if self._disabled or self._resource is None:
            return False
        if not _TORCH_AVAILABLE or not isinstance(tensor, torch.Tensor):
            return False
        if not tensor.is_cuda:
            return False
        if tensor.ndim != 3 or tensor.shape[2] != 3:
            return False
        h, w = tensor.shape[0], tensor.shape[1]
        if h != self._tex_h or w != self._tex_w:
            return False
        if tensor.dtype != torch.uint8:
            tensor = tensor.to(torch.uint8)
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()

        t0 = time.perf_counter()
        try:
            # 1. Expand HxWx3 -> HxWx4 on the GPU. Alpha is preinitialized.
            self._rgba_buffer[..., :3] = tensor

            # 2. Map the GL texture as a CUDA resource and get its array.
            status = cudart.cudaGraphicsMapResources(1, self._resource, 0)
            _check(status, what="cudaGraphicsMapResources")
            try:
                status, cuda_array = cudart.cudaGraphicsSubResourceGetMappedArray(
                    self._resource, 0, 0,
                )
                _check(status, what="cudaGraphicsSubResourceGetMappedArray")

                # 3. Device-to-device copy from rgba_buffer to the array.
                #    Row pitch = 4 bytes per pixel * width.
                src_ptr = self._rgba_buffer.data_ptr()
                pitch = w * 4
                status = cudart.cudaMemcpy2DToArray(
                    cuda_array,
                    0, 0,                       # dst offset (x, y)
                    src_ptr,                    # src
                    pitch,                      # src pitch
                    pitch,                      # row width in bytes
                    h,                          # row count
                    cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice,
                )
                _check(status, what="cudaMemcpy2DToArray")
            finally:
                status = cudart.cudaGraphicsUnmapResources(1, self._resource, 0)
                _check(status, what="cudaGraphicsUnmapResources")
        except Exception as exc:
            # Disable permanently on first hard failure — the bridge state
            # might be inconsistent, and CPU bounce is fast enough.
            self.fail_reason = f"upload: {exc}"
            self._disabled = True
            self.unregister()
            return False

        self.last_upload_ms = (time.perf_counter() - t0) * 1000.0
        return True

    # ----- Diagnostics -------------------------------------------------------

    def is_disabled(self) -> bool:
        return self._disabled

    def state(self) -> str:
        if self._disabled:
            return f"disabled ({self.fail_reason})"
        if self._resource is None:
            return "registered=no"
        return f"registered tex={self._tex_id} size={self._tex_w}x{self._tex_h}"
