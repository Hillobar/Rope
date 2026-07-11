"""Disk-backed cache for source-face and target-media thumbnails.

Source faces store the 85x85 RGB thumbnail plus the 512-d embedding so
subsequent loads can skip the ONNX detect+recognize pipeline entirely.
Target media stores only the 50px-tall RGB thumbnail since the hot path
there is the cv2 decode/seek.

Entries are keyed by SHA1 of the absolute file path; staleness is decided
by comparing the cached mtime against the file's current mtime.
"""

import hashlib
import os

import numpy as np

_CACHE_ROOT = os.path.join(os.getcwd(), 'cache')
_FACE_DIR = os.path.join(_CACHE_ROOT, 'faces')
_MEDIA_DIR = os.path.join(_CACHE_ROOT, 'media')


def _ensure_dirs():
    os.makedirs(_FACE_DIR, exist_ok=True)
    os.makedirs(_MEDIA_DIR, exist_ok=True)


def _key(path):
    return hashlib.sha1(os.path.abspath(path).encode('utf-8')).hexdigest()


def _face_path(path):
    return os.path.join(_FACE_DIR, _key(path) + '.npz')


def _media_path(path):
    return os.path.join(_MEDIA_DIR, _key(path) + '.npz')


def _atomic_savez(target, **arrays):
    tmp = target + '.tmp'
    try:
        np.savez(tmp, **arrays)
        os.replace(tmp, target)
    except Exception:
        try:
            os.remove(tmp)
        except OSError:
            pass


def load_face(path):
    """Return (thumbnail uint8 RGB 85x85, embedding float32 (512,)) or None."""
    cache_file = _face_path(path)
    if not os.path.exists(cache_file):
        return None
    try:
        src_mtime = os.path.getmtime(path)
    except OSError:
        return None
    try:
        with np.load(cache_file) as npz:
            if float(npz['mtime']) != src_mtime:
                return None
            return npz['thumbnail'].copy(), npz['embedding'].copy()
    except Exception:
        return None


def load_media(path):
    """Return thumbnail uint8 RGB (variable WxH) or None."""
    cache_file = _media_path(path)
    if not os.path.exists(cache_file):
        return None
    try:
        src_mtime = os.path.getmtime(path)
    except OSError:
        return None
    try:
        with np.load(cache_file) as npz:
            if float(npz['mtime']) != src_mtime:
                return None
            return npz['thumbnail'].copy()
    except Exception:
        return None


def store_media(path, thumbnail):
    _ensure_dirs()
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        return
    _atomic_savez(
        _media_path(path),
        mtime=np.float64(mtime),
        thumbnail=np.ascontiguousarray(thumbnail),
    )


