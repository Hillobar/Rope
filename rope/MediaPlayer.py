"""Queued-decoder media player for Rope.

Replaces the previous cv2.VideoCapture + ffplay-subprocess split with a
single in-process pipeline:

    PyAV demuxer ──┬── video decoder thread ── frame queue ──> swap workers
                   └── audio decoder thread ── ring buffer ──> sounddevice

Audio output runs on the sounddevice DAC clock (sample-accurate), and the
swap loop schedules video frames against that clock instead of time.time()
so playback stays in sync even when the swap pipeline runs faster or
slower than realtime.

Video decode prefers a GPU-resident path (CUDA tensors via NVDEC), trying
PyNvVideoCodec first and torchcodec second — PyNvVideoCodec is the only
NVDEC binding with a Windows CUDA wheel, torchcodec covers Linux. If
neither loads (or both fail at runtime, e.g. FFmpeg ABI mismatch on
torchcodec, missing driver on PyNvVideoCodec), the player silently falls
back to PyAV's CPU decoder. Either path returns frames as RGB uint8 —
torch.Tensor on CUDA in the GPU path, or np.ndarray (HxWx3) in the CPU
path. swap_video accepts both shapes.
"""

import os
import sys
import threading
import queue
import time
import traceback

import numpy as np
import torch

import av

try:
    import sounddevice as sd
    _SD_AVAILABLE = True
except Exception as _sd_err:
    _SD_AVAILABLE = False
    print('[MediaPlayer] sounddevice unavailable (%s); audio disabled' % _sd_err,
          file=sys.stderr)

# PyNvVideoCodec: NVIDIA's official NVDEC binding. Ships a Windows CUDA wheel
# (which torchcodec does not), works on Linux too. The bundled FFmpeg demuxer
# is unmaintained per NVIDIA docs — for typical H.264/H.265 MP4/MKV inputs
# it's fine; weird containers fall back to torchcodec, then PyAV.
#
# Windows CUDA-12 cudart shim: PyNvVideoCodec 2.1.0's .pyd is built against
# CUDA 12 (needs cudart64_12.dll), but its loader only consults `CUDA_PATH`.
# On systems where CUDA_PATH points to v13+ (which ships cudart64_13.dll and
# in a new bin/x64 layout), the .pyd fails with a generic "DLL load failed"
# before we ever see it. Pre-register any sibling CUDA 12.x toolkit's bin
# directory so the dependency resolves; harmless if no v12 toolkit exists.
def _register_cuda12_dll_dir():
    if sys.platform != 'win32' or not hasattr(os, 'add_dll_directory'):
        return
    candidates = []
    cp = os.environ.get('CUDA_PATH')
    if cp:
        candidates.append(os.path.join(cp, 'bin'))
    root = r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA'
    if os.path.isdir(root):
        # Sort descending so 12.9 wins over 12.0 if both are installed.
        for entry in sorted(os.listdir(root), reverse=True):
            if entry.startswith('v12.'):
                candidates.append(os.path.join(root, entry, 'bin'))
    for path in candidates:
        if os.path.isfile(os.path.join(path, 'cudart64_12.dll')):
            try:
                os.add_dll_directory(path)
            except OSError:
                pass
            return  # one is enough


def _preload_pynvvideocodec_ffmpeg():
    """Windows-only: pin PyNvVideoCodec's bundled FFmpeg DLLs by absolute
    path so an unrelated FFmpeg install on the system (e.g. C:\\ffmpeg\\bin)
    can't be picked up by name lookup later. Once a DLL is loaded in the
    process by absolute path, subsequent LoadLibrary("avfilter-10.dll") etc.
    return the cached handle instead of doing a fresh search — closing the
    door on cross-version symbol mismatches like missing av_dovi_find_level.
    Load order follows the FFmpeg dependency graph (base → leaf)."""
    if sys.platform != 'win32':
        return
    import importlib.util as _ilu
    import ctypes as _ctypes
    spec = _ilu.find_spec('PyNvVideoCodec')
    if spec is None or not spec.submodule_search_locations:
        return
    pkg_dir = spec.submodule_search_locations[0]
    for name in ('avutil-59.dll', 'swresample-5.dll', 'swscale-8.dll',
                 'avcodec-61.dll', 'avformat-61.dll', 'avfilter-10.dll',
                 'avdevice-61.dll'):
        path = os.path.join(pkg_dir, name)
        if os.path.isfile(path):
            try:
                _ctypes.WinDLL(path)
            except OSError:
                # Real failures will be surfaced by PyNvVideoCodec's own
                # loader; we're only here to win a name-collision race.
                pass


_register_cuda12_dll_dir()
_preload_pynvvideocodec_ffmpeg()
try:
    import PyNvVideoCodec as _nvc  # noqa: N816 — upstream uses CamelCase package
    _NVC_AVAILABLE = True
    _NVC_LOAD_ERROR = None
except Exception as _nvc_err:
    _nvc = None
    _NVC_AVAILABLE = False
    _NVC_LOAD_ERROR = _nvc_err

# torchcodec: Meta's NVDEC binding. Linux CUDA wheels exist on the PyTorch
# index; Windows wheels are CPU-only as of 0.11.1, so on Windows the import
# usually succeeds but device='cuda' construction fails with "Unsupported
# device". The import can also fail outright on FFmpeg ABI mismatch.
try:
    from torchcodec.decoders import VideoDecoder as _TorchCodecDecoder
    _TORCHCODEC_AVAILABLE = True
    _TORCHCODEC_LOAD_ERROR = None
except Exception as _tc_err:
    _TorchCodecDecoder = None
    _TORCHCODEC_AVAILABLE = False
    _TORCHCODEC_LOAD_ERROR = _tc_err


# Bounded queue depth for decoded video frames. Larger = smoother playback
# under variable swap latency, but more VRAM/RAM pinned to upcoming frames.
# 8 frames at 1080p30 ≈ 50MB on CPU or VRAM; at 4K30 ≈ 200MB.
VIDEO_QUEUE_DEPTH = 8

# Audio ring buffer size in seconds. Large enough to absorb decode jitter
# but small enough that pause-then-seek doesn't burn through stale samples.
AUDIO_BUFFER_SECONDS = 0.5

# Default audio output configuration. PortAudio resamples on the device
# side if needed; PyAV resamples on the decode side.
AUDIO_TARGET_RATE = 48000
AUDIO_TARGET_CHANNELS = 2
AUDIO_TARGET_FORMAT = 's16'  # PyAV format name for int16 packed


class _NvcGpuDecoder:
    """PyNvVideoCodec SimpleDecoder adapter. Random-access frame fetch
    returning HxWx3 uint8 CUDA torch.Tensor — same contract as
    _TorchCodecGpuDecoder, so the rest of MediaPlayer is backend-agnostic."""

    name = 'PyNvVideoCodec'

    def __init__(self, file_path):
        # RGBP = planar RGB (3, H, W). Matches torchcodec's output layout so
        # the permute below is the same for both backends.
        self._d = _nvc.SimpleDecoder(
            file_path,
            use_device_memory=True,
            output_color_type=_nvc.OutputColorType.RGBP,
        )

    def get_frame(self, idx):
        # SimpleDecoder.__getitem__ returns a Frame exposing __dlpack__; the
        # underlying surface may be recycled by the NVDEC pool on the next
        # decode call, so we .contiguous() to allocate a fresh tensor we own.
        frame = self._d[idx]
        t = torch.from_dlpack(frame).permute(1, 2, 0).contiguous()
        # Docs don't pin the dtype for RGBP; coerce defensively.
        if t.dtype != torch.uint8:
            t = t.to(torch.uint8)
        return t


class _TorchCodecGpuDecoder:
    """torchcodec VideoDecoder adapter. Same contract as _NvcGpuDecoder."""

    name = 'torchcodec'

    def __init__(self, file_path):
        self._d = _TorchCodecDecoder(file_path, device='cuda')

    def get_frame(self, idx):
        # torchcodec returns (3, H, W) uint8 cuda; permute to HxWx3 like
        # swap_video expects. Existing surfaces are owned by torch already,
        # but .contiguous() ensures stride matches the CPU-path ndarray.
        return self._d[idx].permute(1, 2, 0).contiguous()


def _open_gpu_decoder(file_path):
    """Try PyNvVideoCodec, then torchcodec. Returns the adapter on success
    or None if no GPU decoder can be opened for this file. Logs once per
    failed backend so the user sees which fallback they're in."""
    if _NVC_AVAILABLE:
        try:
            return _NvcGpuDecoder(file_path)
        except Exception as e:
            print('[MediaPlayer] PyNvVideoCodec init failed (%s); '
                  'trying torchcodec next' % e, file=sys.stderr)
    if _TORCHCODEC_AVAILABLE:
        try:
            return _TorchCodecGpuDecoder(file_path)
        except Exception as e:
            print('[MediaPlayer] torchcodec init failed (%s); '
                  'using PyAV CPU decode' % e, file=sys.stderr)
    return None


class _AudioRing:
    """Thread-safe int16 PCM ring buffer feeding sounddevice's callback.

    Producer (audio decoder thread) calls write(); consumer (sounddevice
    callback) calls read(). The DAC clock is exposed by sounddevice
    separately — this class only handles the bytes."""

    def __init__(self, sample_rate, channels, capacity_seconds=AUDIO_BUFFER_SECONDS):
        self.sample_rate = int(sample_rate)
        self.channels = int(channels)
        self.capacity = int(self.sample_rate * capacity_seconds)
        # Layout: (capacity, channels) int16. Headroom 2x so wraparound
        # writes can be a single contiguous slice.
        self._buf = np.zeros((self.capacity * 2, self.channels), dtype=np.int16)
        self._write_pos = 0  # cumulative samples written (monotonically grows)
        self._read_pos = 0   # cumulative samples read (monotonically grows)
        self._lock = threading.Lock()

    def available_write(self):
        with self._lock:
            return self.capacity - (self._write_pos - self._read_pos)

    def write(self, samples):
        """samples: ndarray (n, channels) int16. Drops if full."""
        n = samples.shape[0]
        if n == 0:
            return
        with self._lock:
            free = self.capacity - (self._write_pos - self._read_pos)
            if free <= 0:
                return  # buffer full, drop. Callback will catch up.
            n = min(n, free)
            start = self._write_pos % self.capacity
            end = start + n
            if end <= self.capacity:
                self._buf[start:end] = samples[:n]
            else:
                first = self.capacity - start
                self._buf[start:self.capacity] = samples[:first]
                self._buf[0:end - self.capacity] = samples[first:n]
            self._write_pos += n

    def read(self, n):
        """Return n samples of int16 (n, channels). Zero-pads on underrun."""
        out = np.zeros((n, self.channels), dtype=np.int16)
        with self._lock:
            avail = self._write_pos - self._read_pos
            take = min(n, avail)
            if take > 0:
                start = self._read_pos % self.capacity
                end = start + take
                if end <= self.capacity:
                    out[:take] = self._buf[start:end]
                else:
                    first = self.capacity - start
                    out[:first] = self._buf[start:self.capacity]
                    out[first:take] = self._buf[0:end - self.capacity]
                self._read_pos += take
        return out

    def reset(self):
        with self._lock:
            self._write_pos = 0
            self._read_pos = 0


class MediaPlayer:
    """Owns one video file's decode + audio + sync state.

    Lifecycle:
        p = MediaPlayer(path)
        p.get_first_frame()           # synchronous, for initial preview
        p.start_playback(audio=True)  # spin up decode threads + audio stream
        ...
        p.get_next_frame()            # called by VideoManager.process()
        p.get_audio_position()        # used to schedule frame display
        ...
        p.stop_playback()             # pause; resumable
        p.seek(frame_number)          # works whether playing or paused
        p.close()                     # release everything
    """

    def __init__(self, file_path, prefer_gpu_decode=True):
        self.file_path = file_path

        # --- Container probe (PyAV is always used for metadata + audio) ---
        self._container = av.open(file_path)
        self._video_stream = self._container.streams.video[0]
        self._audio_stream = (self._container.streams.audio[0]
                              if self._container.streams.audio else None)

        # Frame-rate. PyAV exposes both average_rate (ideal) and base_rate.
        # Fall back to time_base inverse for files with bogus rate metadata.
        rate = (self._video_stream.average_rate
                or self._video_stream.base_rate)
        self.fps = float(rate) if rate else 30.0
        self.width = int(self._video_stream.codec_context.width)
        self.height = int(self._video_stream.codec_context.height)
        self.video_time_base = float(self._video_stream.time_base)
        # frames count: prefer container-reported, fall back to duration*fps.
        n = self._video_stream.frames or 0
        if n <= 0 and self._video_stream.duration:
            n = int(float(self._video_stream.duration)
                    * self.video_time_base * self.fps)
        if n <= 0 and self._container.duration:
            n = int((self._container.duration / av.time_base) * self.fps)
        self.video_frame_total = max(1, n)

        # --- GPU video decode probe ---
        # Adapter wraps whichever backend (PyNvVideoCodec preferred, torchcodec
        # fallback) is importable and can open this file. None = no GPU decode
        # possible for this file; the loop will use PyAV CPU instead.
        self._gpu_decoder = None
        self._gpu_decode = False
        if prefer_gpu_decode and torch.cuda.is_available():
            self._gpu_decoder = _open_gpu_decoder(file_path)
            self._gpu_decode = self._gpu_decoder is not None

        # --- Threading + queues ---
        self._video_q = queue.Queue(maxsize=VIDEO_QUEUE_DEPTH)
        self._video_thread = None
        self._audio_thread = None
        self._stop_event = threading.Event()
        # Set on every seek to make decode threads abandon in-flight work.
        self._seek_generation = 0
        self._gen_lock = threading.Lock()

        # --- Audio output state ---
        self._audio_stream_out = None
        self._audio_ring = None
        self._audio_resampler = None
        self._audio_start_dac_time = None  # sounddevice time at audio start
        self._audio_start_seek_seconds = 0.0  # PTS of first sample at start
        self._audio_enabled = False

        # Position tracking: the "next frame number to decode" — updated by
        # the decoder thread on each successful decode and reset by seek().
        self._next_decode_frame = 0
        self._next_decode_lock = threading.Lock()

    # ------- Public properties --------------------------------------------

    @property
    def has_audio(self):
        return self._audio_stream is not None and _SD_AVAILABLE

    @property
    def gpu_decode_active(self):
        return self._gpu_decode

    # ------- Synchronous single-frame access (preview / scrub) ------------

    def get_frame_at(self, frame_number):
        """Synchronous: seek + decode one frame at the given index.
        Returns (frame, pts_seconds) where frame is RGB uint8 (np.ndarray
        on CPU path, torch.Tensor HxWxC on GPU path)."""
        frame_number = max(0, min(int(frame_number), self.video_frame_total - 1))
        if self._gpu_decode and self._gpu_decoder is not None:
            try:
                tensor = self._gpu_decoder.get_frame(frame_number)
                pts = frame_number / self.fps
                return tensor, pts
            except Exception as e:
                print('[MediaPlayer] %s single-frame fetch failed (%s); '
                      'falling back to PyAV for this call'
                      % (self._gpu_decoder.name, e), file=sys.stderr)
        # PyAV path: seek to nearest preceding keyframe, decode forward.
        return self._pyav_get_frame_at(frame_number)

    def _pyav_get_frame_at(self, frame_number):
        # Open a per-call container — PyAV containers are NOT thread-safe,
        # and the background _video_decode_loop is concurrently using
        # self._container. Sharing it segfaults the ffmpeg state.
        try:
            container = av.open(self.file_path)
            stream = container.streams.video[0]
            tb = float(stream.time_base)
        except Exception:
            return np.zeros((self.height, self.width, 3), dtype=np.uint8), 0.0
        try:
            target_pts = int(round(frame_number / self.fps / tb))
            try:
                container.seek(target_pts, stream=stream,
                               any_frame=False, backward=True)
            except av.AVError:
                container.seek(0)
            last_frame = None
            last_pts = 0.0
            for packet in container.demux(stream):
                for frame in packet.decode():
                    if frame.pts is None:
                        continue
                    t = float(frame.pts) * tb
                    last_frame = frame
                    last_pts = t
                    if t * self.fps >= frame_number - 0.5:
                        rgb = frame.to_ndarray(format='rgb24')
                        return rgb, t
            if last_frame is not None:
                return last_frame.to_ndarray(format='rgb24'), last_pts
            return np.zeros((self.height, self.width, 3), dtype=np.uint8), 0.0
        finally:
            try:
                container.close()
            except Exception:
                pass

    def get_first_frame(self):
        """RGB uint8 ndarray for initial preview. Always CPU/numpy because
        the GUI's PIL/Tk path doesn't accept CUDA tensors."""
        frame, _pts = self.get_frame_at(0)
        if isinstance(frame, torch.Tensor):
            return frame.cpu().numpy()
        return frame

    # ------- Seek ---------------------------------------------------------

    def seek(self, frame_number):
        """Reposition playback to `frame_number`. Drains pending video frames
        and bumps the seek generation so any in-flight decode thread bails
        on its next iteration. Audio is repositioned on next start_playback."""
        frame_number = max(0, min(int(frame_number), self.video_frame_total - 1))
        with self._gen_lock:
            self._seek_generation += 1
        with self._next_decode_lock:
            self._next_decode_frame = frame_number
        # Drain video queue.
        while True:
            try:
                self._video_q.get_nowait()
            except queue.Empty:
                break
        # Audio: stop the stream so the next start_playback re-seeks cleanly.
        # We don't need to replay buffered audio; the user has just scrubbed.
        if self._audio_stream_out is not None:
            try:
                self._audio_stream_out.stop()
            except Exception:
                pass
        if self._audio_ring is not None:
            self._audio_ring.reset()
        self._audio_start_dac_time = None

    # ------- Playback control ---------------------------------------------

    def start_playback(self, audio_enabled=True):
        """Spin up the decode threads (and audio stream if enabled).
        Resumes from `_next_decode_frame`."""
        self._stop_event.clear()
        self._audio_enabled = bool(audio_enabled and self.has_audio)

        # Video decoder thread.
        if self._video_thread is None or not self._video_thread.is_alive():
            self._video_thread = threading.Thread(
                target=self._video_decode_loop, name='MediaPlayer-Video',
                daemon=True,
            )
            self._video_thread.start()

        # Audio decoder + output stream.
        if self._audio_enabled:
            self._open_audio_output()
            if self._audio_thread is None or not self._audio_thread.is_alive():
                self._audio_thread = threading.Thread(
                    target=self._audio_decode_loop, name='MediaPlayer-Audio',
                    daemon=True,
                )
                self._audio_thread.start()

    def stop_playback(self, _join_timeout=0.5):
        """Pause: tell decode threads to bail, stop audio stream. Position
        is preserved in `_next_decode_frame` so start_playback resumes there.

        Joins the threads with a short timeout so we don't return while a
        decoder is still mid-iteration on `self._container` — important for
        close() which then frees the container."""
        self._stop_event.set()
        with self._gen_lock:
            self._seek_generation += 1
        if self._audio_stream_out is not None:
            try:
                self._audio_stream_out.stop()
            except Exception:
                pass
        # Drain queue so put() inside the decoder thread doesn't block on
        # full queue while we're trying to shut it down.
        while True:
            try:
                self._video_q.get_nowait()
            except queue.Empty:
                break
        if self._video_thread is not None and self._video_thread.is_alive():
            self._video_thread.join(timeout=_join_timeout)
        if self._audio_thread is not None and self._audio_thread.is_alive():
            self._audio_thread.join(timeout=_join_timeout)
        self._video_thread = None
        self._audio_thread = None

    def get_next_frame(self, timeout=0.0):
        """Pop the next decoded video frame.

        Returns (frame, frame_number, pts_seconds) or None if the queue is
        empty within timeout. `frame` is torch.Tensor (HxWxC uint8 cuda) on
        the GPU path or np.ndarray (HxWx3 uint8) on the CPU path."""
        try:
            return self._video_q.get(timeout=timeout) if timeout > 0 \
                else self._video_q.get_nowait()
        except queue.Empty:
            return None

    # ------- Audio clock --------------------------------------------------

    def get_audio_position(self):
        """Current audio playback time in seconds since the file start.
        Returns None if audio is not playing — caller should fall back to
        the wall clock in that case."""
        if not self._audio_enabled or self._audio_stream_out is None:
            return None
        if self._audio_start_dac_time is None:
            return None
        try:
            now = self._audio_stream_out.time
            elapsed = now - self._audio_start_dac_time
        except Exception:
            return None
        if elapsed < 0:
            return None
        return self._audio_start_seek_seconds + elapsed

    # ------- Cleanup ------------------------------------------------------

    def close(self):
        self.stop_playback()
        if self._audio_stream_out is not None:
            try:
                self._audio_stream_out.close()
            except Exception:
                pass
            self._audio_stream_out = None
        if self._gpu_decoder is not None:
            self._gpu_decoder = None
        try:
            self._container.close()
        except Exception:
            pass

    # ====== Internal: video decode loop ==================================

    def _video_decode_loop(self):
        """Producer: keeps the video queue full from `_next_decode_frame`."""
        try:
            while not self._stop_event.is_set():
                with self._next_decode_lock:
                    start_frame = self._next_decode_frame
                if start_frame >= self.video_frame_total:
                    return
                with self._gen_lock:
                    my_gen = self._seek_generation

                # GPU path: NVDEC single-frame fetches via the adapter, in
                # order. Done frame-at-a-time so we can react to seeks promptly.
                if self._gpu_decode and self._gpu_decoder is not None:
                    self._gpu_decode_chunk(start_frame, my_gen)
                else:
                    # CPU path: PyAV demux once, decode forward until a seek
                    # bumps the generation or we hit EOF.
                    self._pyav_decode_chunk(start_frame, my_gen)
        except Exception:
            print('[MediaPlayer] video decode loop crashed:', file=sys.stderr)
            traceback.print_exc()

    def _gpu_decode_chunk(self, start_frame, my_gen):
        idx = start_frame
        while not self._stop_event.is_set():
            # Bail if a seek has happened while we were enqueuing.
            with self._gen_lock:
                if my_gen != self._seek_generation:
                    return
            if idx >= self.video_frame_total:
                return
            try:
                tensor = self._gpu_decoder.get_frame(idx)
            except Exception as e:
                print('[MediaPlayer] %s decode failed at frame %d (%s); '
                      'switching to CPU decode for the rest of this run'
                      % (self._gpu_decoder.name, idx, e), file=sys.stderr)
                self._gpu_decode = False
                self._gpu_decoder = None
                return
            pts = idx / self.fps
            # put() blocks if the queue is full — that's the back-pressure
            # signal that swap workers haven't drained yet. Wake periodically
            # so a stop or seek isn't stuck behind a full queue.
            while not self._stop_event.is_set():
                try:
                    self._video_q.put((tensor, idx, pts), timeout=0.05)
                    break
                except queue.Full:
                    with self._gen_lock:
                        if my_gen != self._seek_generation:
                            return
            with self._next_decode_lock:
                self._next_decode_frame = idx + 1
            idx += 1

    def _pyav_decode_chunk(self, start_frame, my_gen):
        target_pts = int(round(start_frame / self.fps / self.video_time_base))
        try:
            self._container.seek(target_pts, stream=self._video_stream,
                                 any_frame=False, backward=True)
        except av.AVError:
            self._container.seek(0)
        idx = start_frame
        for packet in self._container.demux(self._video_stream):
            if self._stop_event.is_set():
                return
            with self._gen_lock:
                if my_gen != self._seek_generation:
                    return
            for frame in packet.decode():
                # Re-check the generation at every frame too — without this,
                # a seek that arrives while the queue has spare capacity
                # would be ignored (put() succeeds, we never hit the
                # queue.Full branch where the existing check lived).
                if self._stop_event.is_set():
                    return
                with self._gen_lock:
                    if my_gen != self._seek_generation:
                        return
                if frame.pts is None:
                    continue
                t = float(frame.pts) * self.video_time_base
                # Skip frames before the seek target (we land on the keyframe
                # before start_frame, then walk forward until we reach it).
                if t * self.fps < start_frame - 0.5:
                    continue
                rgb = frame.to_ndarray(format='rgb24')
                while not self._stop_event.is_set():
                    try:
                        self._video_q.put((rgb, idx, t), timeout=0.05)
                        break
                    except queue.Full:
                        with self._gen_lock:
                            if my_gen != self._seek_generation:
                                return
                with self._next_decode_lock:
                    self._next_decode_frame = idx + 1
                idx += 1
                if idx >= self.video_frame_total:
                    return

    # ====== Internal: audio decode + output ==============================

    def _open_audio_output(self):
        """Create the sounddevice OutputStream and ring buffer."""
        if self._audio_stream is None:
            return
        if self._audio_ring is None:
            self._audio_ring = _AudioRing(AUDIO_TARGET_RATE,
                                          AUDIO_TARGET_CHANNELS)
        self._audio_resampler = av.AudioResampler(
            format=AUDIO_TARGET_FORMAT,
            layout='stereo',
            rate=AUDIO_TARGET_RATE,
        )

        def _callback(outdata, frames, time_info, status):
            samples = self._audio_ring.read(frames)
            outdata[:] = samples
            # Capture the DAC time of the first sample we hand off — used
            # by get_audio_position() to compute current playback time.
            if self._audio_start_dac_time is None:
                self._audio_start_dac_time = time_info.outputBufferDacTime

        if self._audio_stream_out is not None:
            try:
                self._audio_stream_out.close()
            except Exception:
                pass
        self._audio_stream_out = sd.OutputStream(
            samplerate=AUDIO_TARGET_RATE,
            channels=AUDIO_TARGET_CHANNELS,
            dtype='int16',
            callback=_callback,
            blocksize=0,  # let PortAudio pick — typically 256-1024
            latency='low',
        )
        # Anchor: position at start = current decode frame's PTS.
        with self._next_decode_lock:
            self._audio_start_seek_seconds = (self._next_decode_frame
                                              / self.fps)
        self._audio_start_dac_time = None
        self._audio_stream_out.start()

    def _audio_decode_loop(self):
        """Producer: decode audio packets and push PCM into the ring buffer.
        Seek-aware via the same _seek_generation counter as video."""
        try:
            while not self._stop_event.is_set():
                with self._gen_lock:
                    my_gen = self._seek_generation
                with self._next_decode_lock:
                    seek_seconds = self._next_decode_frame / self.fps
                self._audio_decode_chunk(seek_seconds, my_gen)
        except Exception:
            print('[MediaPlayer] audio decode loop crashed:', file=sys.stderr)
            traceback.print_exc()

    def _audio_decode_chunk(self, seek_seconds, my_gen):
        if self._audio_stream is None:
            return
        # Use a separate container handle for audio so audio seek doesn't
        # disturb the video demuxer position. PyAV containers aren't designed
        # for two simultaneous demux iterators on different streams.
        try:
            audio_container = av.open(self.file_path)
            audio_stream = audio_container.streams.audio[0]
        except Exception as e:
            print('[MediaPlayer] could not open audio container: %s' % e,
                  file=sys.stderr)
            return
        try:
            tb = float(audio_stream.time_base)
            target = int(round(seek_seconds / tb))
            try:
                audio_container.seek(target, stream=audio_stream,
                                     any_frame=False, backward=True)
            except av.AVError:
                audio_container.seek(0)
            for packet in audio_container.demux(audio_stream):
                if self._stop_event.is_set():
                    return
                with self._gen_lock:
                    if my_gen != self._seek_generation:
                        return
                for frame in packet.decode():
                    out_frames = self._audio_resampler.resample(frame)
                    for of in out_frames:
                        # PyAV layout depends on format:
                        #   packed   ('s16'):  (1, samples*channels)
                        #   planar   ('s16p'): (channels, samples)
                        # sounddevice wants    (samples, channels).
                        arr = of.to_ndarray()
                        if arr.ndim == 2 and arr.shape[0] == 1 and AUDIO_TARGET_CHANNELS > 0:
                            # packed: deinterleave by reshape.
                            total = arr.shape[1]
                            if total % AUDIO_TARGET_CHANNELS == 0:
                                arr = arr.reshape(-1, AUDIO_TARGET_CHANNELS)
                            else:
                                # Should not happen with a working resampler;
                                # fall back to mono-doubled stereo.
                                arr = np.stack([arr.flatten(), arr.flatten()], axis=1)
                        elif arr.ndim == 2 and arr.shape[0] == AUDIO_TARGET_CHANNELS:
                            arr = arr.T  # planar -> (samples, channels)
                        elif arr.ndim == 1:
                            arr = np.stack([arr, arr], axis=1)  # mono -> stereo
                        # Backpressure: if the ring is full, the callback
                        # hasn't drained yet. Sleep briefly and retry.
                        while self._audio_ring.available_write() < arr.shape[0]:
                            if self._stop_event.is_set():
                                return
                            with self._gen_lock:
                                if my_gen != self._seek_generation:
                                    return
                            time.sleep(0.005)
                        self._audio_ring.write(arr)
        finally:
            try:
                audio_container.close()
            except Exception:
                pass
