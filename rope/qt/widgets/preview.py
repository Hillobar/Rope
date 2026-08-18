"""GPU-accelerated video preview replacing the QLabel placeholder.

A single-texture, single-quad QOpenGLWidget. The CPU path uploads frames
via glTexSubImage2D from a host numpy buffer. The CUDA fast path
(activated when cuda-python is importable AND the frame is a CUDA-resident
torch tensor) does a device-to-device copy into a CUDA-mapped GL texture,
skipping the host bounce entirely.

Architecture choices:
- Letterboxing is done in the vertex shader via a vec2 uScale uniform
  (computed in resizeGL / set_frame). The geometry is a static fullscreen
  quad; only the uniform changes. Keeps the render path branchless.
- Texture format is GL_RGB / GL_UNSIGNED_BYTE matching what
  VideoManager.swap_video emits (numpy uint8 HxWx3 RGB, or torch tensor
  same shape).
- If the frame dimensions change (different video loaded), the texture
  is reallocated; otherwise glTexSubImage2D reuses the same texture.
- bus.frame_ready arrives on the main thread (Qt.QueuedConnection
  auto-marshals from worker threads), so all GL calls happen here safely.
"""

from __future__ import annotations

import time
from collections import deque
from typing import Any, Optional

import numpy as np
from PySide6.QtCore import Qt, QTimer, Signal, Slot
from PySide6.QtGui import QColor, QImage, QPainter
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtWidgets import QLabel

try:
    from OpenGL import GL
    _OPENGL_AVAILABLE = True
except ImportError:
    GL = None  # type: ignore
    _OPENGL_AVAILABLE = False

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore
    _TORCH_AVAILABLE = False

from rope.qt.cuda_gl_interop import CudaGLBridge, available as _cuda_gl_available


def _fps_from_timeline(times: deque) -> float:
    """fps = (n - 1) / (last - first) over a sliding window of perf_counter
    timestamps. Returns 0 if the window has <2 samples or the span is
    zero/negative (clock jitter)."""
    if len(times) < 2:
        return 0.0
    dt = times[-1] - times[0]
    if dt <= 0:
        return 0.0
    return (len(times) - 1) / dt


_VERTEX_SHADER = """
#version 330 core
layout (location = 0) in vec2 aPos;
layout (location = 1) in vec2 aUV;
uniform vec2 uScale;
out vec2 vUV;
void main() {
    gl_Position = vec4(aPos * uScale, 0.0, 1.0);
    vUV = aUV;
}
"""

_FRAGMENT_SHADER = """
#version 330 core
in vec2 vUV;
out vec4 fragColor;
uniform sampler2D uTex;
void main() {
    fragColor = vec4(texture(uTex, vUV).rgb, 1.0);
}
"""

# Fullscreen quad. UV is flipped vertically (V = 1-y) because OpenGL's
# texture origin is bottom-left but image data is top-left.
_QUAD_VERTICES = np.array(
    [
        # x,   y,    u,    v
        -1.0, -1.0,  0.0,  1.0,
         1.0, -1.0,  1.0,  1.0,
        -1.0,  1.0,  0.0,  0.0,
         1.0,  1.0,  1.0,  0.0,
    ],
    dtype=np.float32,
)


class PreviewWidget(QOpenGLWidget):
    """One-quad textured preview. Slot `set_frame(frame, requested)` accepts
    np.ndarray (H, W, 3) uint8 RGB or torch.Tensor of the same shape (CPU or
    CUDA).
    """

    # Left-click anywhere on the preview canvas. The center pane wires this
    # to the same handler as the Play button (matches the Tk binding on
    # tk.Label <ButtonRelease-1> -> toggle_play_video).
    clicked = Signal()
    # Mouse wheel over the preview. +1 = next, -1 = previous. Wired by
    # main_window to cycle the Embeddings pane selection so the user can
    # flip through saved embeddings without leaving the video area.
    wheel_scrolled = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._texture_id: int = 0
        self._program: int = 0
        self._vao: int = 0
        self._vbo: int = 0
        self._uniform_scale: int = -1
        self._uniform_tex: int = -1
        self._scale = (1.0, 1.0)

        # Pending frame staged from set_frame(); applied in paintGL so we
        # never touch GL from a non-render thread or before initializeGL.
        # Either _pending_cpu (np.ndarray) or _pending_cuda (torch CUDA
        # tensor) is set, not both — set_frame chooses based on input type.
        self._pending_cpu: Optional[np.ndarray] = None
        self._pending_cuda = None  # torch.Tensor on CUDA, HxWx3 uint8
        self._tex_w: int = 0
        self._tex_h: int = 0

        # Most recent frame held for Save Image. Stored in whatever form
        # set_frame received (np.ndarray, CPU tensor, or CUDA tensor) so
        # the fast path doesn't pay a per-frame GPU->host bounce — the
        # bounce moves into last_frame_rgb() which only runs on Save.
        self._last_frame_source: Optional[Any] = None

        # Perf HUD state. Two timelines: when set_frame was called (input
        # fps from VM via bus.frame_ready) and when paintGL completed
        # (display fps). Divergence between them = bottleneck somewhere —
        # paint stall keeps in-fps high while paint-fps drops; producer
        # stall does the opposite. Toggle visibility via toggle_hud().
        self._set_frame_times: deque = deque(maxlen=120)
        self._paint_times: deque = deque(maxlen=120)
        self._paint_ms_samples: deque = deque(maxlen=120)
        self._hud_label: Optional[QLabel] = None
        self._hud_visible: bool = False
        self._hud_timer = QTimer(self)
        self._hud_timer.setInterval(500)
        self._hud_timer.timeout.connect(self._update_hud)
        self._hud_timer.start()
        # Latest VM queue depths from the coordinator. Wired below in
        # _ensure_hud once Qt has finished the widget construction.
        self._qd_frame: int = 0
        self._qd_rframe: int = 0
        try:
            from rope.qt.bus import bus as _bus
            _bus.queue_depths.connect(self._on_queue_depths)
        except ImportError:
            pass

        # Perf probe
        self._last_paint_ms: float = 0.0
        self._last_path: str = ""  # "cpu" or "cuda" — which upload path ran
        self._last_input_kind: str = "--"  # what set_frame received

        # CUDA-GL interop bridge. Disabled if cuda-python missing or
        # registration fails; falls back to CPU bounce.
        self._cuda_bridge = CudaGLBridge() if _cuda_gl_available() else None

        # Black background between frames
        self.setStyleSheet("background-color: #000000;")

    # --- GL lifecycle ---------------------------------------------------------

    def initializeGL(self):
        if not _OPENGL_AVAILABLE:
            return

        # Compile + link program
        self._program = self._build_program()
        self._uniform_scale = GL.glGetUniformLocation(self._program, "uScale")
        self._uniform_tex = GL.glGetUniformLocation(self._program, "uTex")

        # VAO + VBO for the fullscreen quad
        self._vao = GL.glGenVertexArrays(1)
        self._vbo = GL.glGenBuffers(1)
        GL.glBindVertexArray(self._vao)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, _QUAD_VERTICES.nbytes, _QUAD_VERTICES, GL.GL_STATIC_DRAW)
        # aPos (location 0): vec2 at offset 0, stride 4 floats
        GL.glEnableVertexAttribArray(0)
        GL.glVertexAttribPointer(0, 2, GL.GL_FLOAT, GL.GL_FALSE, 4 * 4, None)
        # aUV (location 1): vec2 at offset 2 floats
        GL.glEnableVertexAttribArray(1)
        GL.glVertexAttribPointer(1, 2, GL.GL_FLOAT, GL.GL_FALSE, 4 * 4, GL.ctypes.c_void_p(2 * 4))
        GL.glBindVertexArray(0)

        # Texture is allocated lazily on first frame via _ensure_texture_size.
        # initializeGL just sets the clear color.
        GL.glClearColor(0.0, 0.0, 0.0, 1.0)

    def _build_program(self) -> int:
        def _compile(src: str, kind: int) -> int:
            shader = GL.glCreateShader(kind)
            GL.glShaderSource(shader, src)
            GL.glCompileShader(shader)
            ok = GL.glGetShaderiv(shader, GL.GL_COMPILE_STATUS)
            if not ok:
                err = GL.glGetShaderInfoLog(shader).decode("utf-8", "replace")
                GL.glDeleteShader(shader)
                raise RuntimeError(f"shader compile failed: {err}")
            return shader

        vs = _compile(_VERTEX_SHADER, GL.GL_VERTEX_SHADER)
        fs = _compile(_FRAGMENT_SHADER, GL.GL_FRAGMENT_SHADER)
        prog = GL.glCreateProgram()
        GL.glAttachShader(prog, vs)
        GL.glAttachShader(prog, fs)
        GL.glLinkProgram(prog)
        ok = GL.glGetProgramiv(prog, GL.GL_LINK_STATUS)
        if not ok:
            err = GL.glGetProgramInfoLog(prog).decode("utf-8", "replace")
            GL.glDeleteProgram(prog)
            raise RuntimeError(f"program link failed: {err}")
        GL.glDeleteShader(vs)
        GL.glDeleteShader(fs)
        return prog

    def resizeGL(self, w: int, h: int):
        if not _OPENGL_AVAILABLE:
            return
        GL.glViewport(0, 0, max(1, w), max(1, h))
        self._recompute_scale()

    def _recompute_scale(self):
        widget_w = max(1, self.width())
        widget_h = max(1, self.height())
        if self._tex_w == 0 or self._tex_h == 0:
            self._scale = (1.0, 1.0)
            return
        widget_aspect = widget_w / widget_h
        image_aspect = self._tex_w / self._tex_h
        if widget_aspect > image_aspect:
            # Widget is wider — letterbox left/right (shrink x)
            self._scale = (image_aspect / widget_aspect, 1.0)
        else:
            # Widget is taller — letterbox top/bottom (shrink y)
            self._scale = (1.0, widget_aspect / image_aspect)

    def paintGL(self):
        if not _OPENGL_AVAILABLE or self._program == 0:
            # QPainter fallback for platforms where OpenGL shader init failed or PyOpenGL is missing
            t0 = time.perf_counter()
            painter = QPainter(self)
            painter.fillRect(self.rect(), QColor(0, 0, 0))
            frame = self._pending_cpu
            if frame is None and self._pending_cuda is not None:
                try:
                    frame = self._pending_cuda.detach().cpu().numpy()
                except Exception:
                    frame = None
            if frame is None and self._last_frame_source is not None:
                if isinstance(self._last_frame_source, np.ndarray):
                    frame = self._last_frame_source
                elif _TORCH_AVAILABLE and isinstance(self._last_frame_source, torch.Tensor):
                    try:
                        frame = self._last_frame_source.detach().cpu().numpy()
                    except Exception:
                        frame = None
            if frame is not None and isinstance(frame, np.ndarray) and frame.ndim == 3:
                h, w, c = frame.shape
                if not frame.flags["C_CONTIGUOUS"]:
                    frame = np.ascontiguousarray(frame)
                qimg = QImage(frame.data, w, h, w * c, QImage.Format.Format_RGB888)
                scaled = qimg.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                x = (self.width() - scaled.width()) // 2
                y = (self.height() - scaled.height()) // 2
                painter.drawImage(x, y, scaled)
                self._tex_w, self._tex_h = w, h
                self._last_path = "qpainter-fallback"
            painter.end()
            self._pending_cpu = None
            self._pending_cuda = None
            self._last_paint_ms = (time.perf_counter() - t0) * 1000.0
            return

        t0 = time.perf_counter()
        # Record paint timeline at the start so a stalled paint still
        # registers an interval boundary (avoids fps spikes after long
        # gaps when the next paint finally completes).

        GL.glClear(GL.GL_COLOR_BUFFER_BIT)

        if self._pending_cuda is not None:
            self._upload_cuda(self._pending_cuda)
            self._pending_cuda = None
        elif self._pending_cpu is not None:
            self._upload_cpu(self._pending_cpu)
            self._pending_cpu = None

        if self._tex_w == 0:
            self._last_paint_ms = (time.perf_counter() - t0) * 1000.0
            return

        GL.glUseProgram(self._program)
        GL.glUniform2f(self._uniform_scale, self._scale[0], self._scale[1])
        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._texture_id)
        GL.glUniform1i(self._uniform_tex, 0)
        GL.glBindVertexArray(self._vao)
        GL.glDrawArrays(GL.GL_TRIANGLE_STRIP, 0, 4)
        GL.glBindVertexArray(0)
        GL.glUseProgram(0)

        self._last_paint_ms = (time.perf_counter() - t0) * 1000.0
        self._paint_times.append(t0)
        self._paint_ms_samples.append(self._last_paint_ms)

    # --- Texture upload -------------------------------------------------------

    def _ensure_texture_size(self, w: int, h: int) -> bool:
        """Reallocate the texture as RGBA8 if dimensions changed. CUDA-GL
        interop requires a 4-channel format, so we always use RGBA8 — the
        CPU path uploads 3-channel data and the driver expands; the CUDA
        path writes 4-channel directly. Returns True if a reallocation
        happened.

        glTexStorage2D would be cleaner here but it creates immutable
        storage, which complicates the realloc-on-resize path. Instead we
        allocate via glTexImage2D with a tiny zero-filled buffer (PyOpenGL
        rejects None for the data arg). At 4 bytes per pixel even a 4K
        texture's allocation buffer is 67 MB — significantly larger than we
        want to allocate transiently. So we use glTexStorage2D for size
        changes (immutable storage) and delete+recreate the texture on
        size mismatch.
        """
        if w == self._tex_w and h == self._tex_h:
            return False
        # Delete and recreate the texture for the new size — glTexStorage2D
        # creates immutable storage that can't be resized.
        if self._texture_id:
            # Bridge holds a CUDA registration referencing the old texture;
            # tear it down before the GL object goes away.
            if self._cuda_bridge is not None:
                self._cuda_bridge.unregister()
            GL.glDeleteTextures([self._texture_id])
        self._texture_id = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._texture_id)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
        GL.glTexStorage2D(GL.GL_TEXTURE_2D, 1, GL.GL_RGBA8, w, h)
        self._tex_w, self._tex_h = w, h
        self._recompute_scale()
        # CUDA bridge has to re-register the new texture.
        if self._cuda_bridge is not None and not self._cuda_bridge.is_disabled():
            self._cuda_bridge.register_texture(int(self._texture_id), w, h)
        return True

    def _upload_cpu(self, arr: np.ndarray) -> None:
        if arr.ndim != 3 or arr.shape[2] != 3:
            return
        h, w = arr.shape[0], arr.shape[1]
        if not arr.flags["C_CONTIGUOUS"]:
            arr = np.ascontiguousarray(arr)
        self._ensure_texture_size(w, h)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._texture_id)
        GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
        # Driver expands GL_RGB src into the RGBA8 texture, filling alpha.
        GL.glTexSubImage2D(
            GL.GL_TEXTURE_2D, 0, 0, 0, w, h,
            GL.GL_RGB, GL.GL_UNSIGNED_BYTE, arr,
        )
        self._last_path = "cpu"

    def _upload_cuda(self, tensor) -> None:
        """Upload an HxWx3 uint8 torch CUDA tensor via cudaGraphicsMapResources.

        Falls back to CPU bounce on first failure (the bridge marks itself
        disabled so subsequent frames go straight to CPU).
        """
        h, w = tensor.shape[0], tensor.shape[1]
        reallocated = self._ensure_texture_size(w, h)
        # Register on first use or after a reallocation.
        if self._cuda_bridge is None:
            self._upload_cpu_from_cuda(tensor); return
        if reallocated or self._cuda_bridge.is_disabled() is False and self._cuda_bridge._resource is None:
            ok = self._cuda_bridge.register_texture(int(self._texture_id), w, h)
            if not ok:
                self._upload_cpu_from_cuda(tensor); return
        if self._cuda_bridge.upload_from_cuda_tensor(tensor):
            self._last_path = "cuda"
            return
        # Bridge couldn't use the fast path — fall back to CPU bounce.
        self._upload_cpu_from_cuda(tensor)

    def _upload_cpu_from_cuda(self, tensor) -> None:
        arr = tensor.detach().cpu().contiguous().numpy()
        self._upload_cpu(arr)

    # --- Public slot ----------------------------------------------------------

    @Slot(object, bool)
    def set_frame(self, frame: Any, requested: bool = False) -> None:
        """Stage a frame for the next paint.

        Accepts:
            - np.ndarray (H, W, 3) uint8 RGB              -> CPU upload
            - torch.Tensor (H, W, 3) uint8, CPU             -> CPU upload
            - torch.Tensor (H, W, 3) uint8, CUDA            -> CUDA-GL fast path
              (auto-falls back to CPU bounce if bridge disabled)
        Anything else is dropped silently.
        """
        self._set_frame_times.append(time.perf_counter())
        if isinstance(frame, np.ndarray):
            self._pending_cpu = frame
            self._pending_cuda = None
            self._last_frame_source = frame
            self._last_input_kind = "numpy"
        elif _TORCH_AVAILABLE and isinstance(frame, torch.Tensor):
            t = frame
            if t.dtype != torch.uint8:
                t = t.to(torch.uint8)
            if not t.is_contiguous():
                t = t.contiguous()
            if t.is_cuda and self._cuda_bridge is not None and not self._cuda_bridge.is_disabled():
                self._pending_cuda = t
                self._pending_cpu = None
                # Keep the CUDA tensor reference; the GPU->host copy is
                # deferred to last_frame_rgb() so the per-frame path
                # stays on-device.
                self._last_frame_source = t
                self._last_input_kind = "cuda-t"
            else:
                # CPU tensor, or CUDA bridge unavailable — bounce to numpy.
                arr = (t.cpu() if t.is_cuda else t).numpy()
                self._pending_cpu = arr
                self._pending_cuda = None
                self._last_frame_source = arr
                self._last_input_kind = "cpu-t" if not t.is_cuda else "cuda-t-bypass"
        else:
            self._last_input_kind = f"drop:{type(frame).__name__}"
            return
        self.update()

    # --- Diagnostics ----------------------------------------------------------

    def last_paint_ms(self) -> float:
        return self._last_paint_ms

    def last_upload_path(self) -> str:
        return self._last_path

    def texture_size(self) -> tuple[int, int]:
        return self._tex_w, self._tex_h

    def cuda_bridge_state(self) -> str:
        if self._cuda_bridge is None:
            return "cuda bridge: unavailable"
        return f"cuda bridge: {self._cuda_bridge.state()}"

    # --- Perf HUD overlay -----------------------------------------------------

    def toggle_hud(self) -> None:
        self._hud_visible = not self._hud_visible
        if self._hud_label is not None:
            self._hud_label.setVisible(self._hud_visible)

    def _ensure_hud(self) -> None:
        if self._hud_label is not None:
            return
        lbl = QLabel("", self)
        lbl.setStyleSheet(
            "QLabel {"
            "  background-color: rgba(0, 0, 0, 180);"
            "  color: #00FF88;"
            "  font-family: Consolas, 'Courier New', monospace;"
            "  font-size: 9pt;"
            "  padding: 4px 6px;"
            "  border: 1px solid rgba(0, 255, 136, 60);"
            "}"
        )
        lbl.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        lbl.setVisible(self._hud_visible)
        self._hud_label = lbl
        self._position_hud()

    def _position_hud(self) -> None:
        if self._hud_label is None:
            return
        self._hud_label.adjustSize()
        # Top-right corner with an 8px inset
        self._hud_label.move(self.width() - self._hud_label.width() - 8, 8)

    def _on_queue_depths(self, frame_q: int, r_frame_q: int) -> None:
        self._qd_frame = int(frame_q)
        self._qd_rframe = int(r_frame_q)

    def _update_hud(self) -> None:
        self._ensure_hud()
        if not self._hud_visible:
            return

        in_fps = _fps_from_timeline(self._set_frame_times)
        paint_fps = _fps_from_timeline(self._paint_times)
        if self._paint_ms_samples:
            samples = list(self._paint_ms_samples)
            avg_ms = sum(samples) / len(samples)
            sorted_s = sorted(samples)
            p95_ms = sorted_s[int(0.95 * len(sorted_s))]
            max_ms = sorted_s[-1]
        else:
            avg_ms = p95_ms = max_ms = 0.0
        tex = f"{self._tex_w}x{self._tex_h}" if self._tex_w else "----"
        path = self._last_path or "--"

        text = (
            f"in   : {in_fps:5.1f} fps     "
            f"draw : {paint_fps:5.1f} fps\n"
            f"paint: {avg_ms:4.1f} ms avg   p95 {p95_ms:4.1f}   max {max_ms:4.1f}\n"
            f"tex  : {tex}    upload: {path}    input: {self._last_input_kind}\n"
            f"{self.cuda_bridge_state()}\n"
            f"q    : frame {self._qd_frame:>2}  scrub {self._qd_rframe:>2}"
        )
        self._hud_label.setText(text)
        self._position_hud()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._position_hud()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.rect().contains(event.position().toPoint()):
            self.clicked.emit()
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event):
        # angleDelta().y() > 0 == wheel rotated away from the user
        # (conventional "up") -> previous; the opposite for "down".
        delta = event.angleDelta().y()
        if delta == 0:
            super().wheelEvent(event)
            return
        self.wheel_scrolled.emit(-1 if delta > 0 else 1)
        event.accept()

    def last_frame_rgb(self) -> Optional[np.ndarray]:
        """Most recently displayed frame as HxWx3 uint8 RGB, or None.

        Materializes from a CUDA tensor on demand — Save Image is the
        only caller and runs at user-action cadence, so the GPU->host
        bounce belongs here, not in the per-frame set_frame path.

        Returned array is owned by the widget; callers should not mutate
        it (copy first if they need to).
        """
        src = self._last_frame_source
        if src is None:
            return None
        if isinstance(src, np.ndarray):
            return src
        if _TORCH_AVAILABLE and isinstance(src, torch.Tensor):
            return src.detach().cpu().numpy()
        return None
