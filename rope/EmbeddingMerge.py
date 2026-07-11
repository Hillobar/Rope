"""Strategies for combining multiple ArcFace-style face embeddings into one.

All strategies operate on the raw 512-d embeddings stored in
source_faces[i]['Embedding']. The combined vector is fed downstream to
calc_swapper_latent(), which does its own normalize -> emap -> normalize, so
we don't normalize on output here — keeping the result in roughly the same
scale as a single input embedding lets downstream code do its usual work
without surprise.
"""

import numpy as np


def combine(embeddings, mode):
    """Combine a list of N 1-D embeddings (length D) into a single 1-D array.

    Unrecognized modes fall back to plain mean so callers don't have to
    special-case stale persisted settings.
    """
    embs = np.asarray(embeddings, dtype=np.float32)
    if embs.ndim == 1:
        embs = embs[None, :]
    if embs.shape[0] == 1 or mode == 'Mean':
        return np.mean(embs, axis=0)
    if mode == 'Median':
        return np.median(embs, axis=0)
    if mode == 'Sph':
        return _spherical_mean(embs)
    if mode == 'Geo':
        return _geometric_median(embs)
    if mode == 'Qual':
        return _quality_weighted_mean(embs)
    return np.mean(embs, axis=0)


def _spherical_mean(embs):
    # L2-normalize each input, mean on the unit hypersphere, renormalize, then
    # rescale to the average input norm. Rescaling matters because downstream
    # code mixes results from different merge modes — keeping output magnitude
    # comparable to a plain Mean avoids surprising the swapper conditioning.
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    unit_mean = np.mean(embs / norms, axis=0)
    m = np.linalg.norm(unit_mean)
    if m == 0:
        return np.mean(embs, axis=0)
    return (unit_mean / m) * float(np.mean(norms))


def _geometric_median(embs, max_iter=50, tol=1e-6, eps=1e-8):
    # Weiszfeld iteration for the 1-median under L2. Unlike coordinate-wise
    # median, the result is a real point in embedding space rather than a
    # synthetic vector of per-axis medians from different faces. eps avoids
    # the degenerate divide-by-zero when the running estimate coincides with
    # an input point (Weiszfeld's well-known failure mode).
    y = np.mean(embs, axis=0)
    for _ in range(max_iter):
        d = np.linalg.norm(embs - y, axis=1)
        w = 1.0 / np.maximum(d, eps)
        new_y = (w[:, None] * embs).sum(axis=0) / w.sum()
        if np.linalg.norm(new_y - y) < tol:
            return new_y
        y = new_y
    return y


def _quality_weighted_mean(embs):
    # Quality proxy: ArcFace embedding magnitude correlates with face quality
    # (the relationship MagFace formalized; vanilla ArcFace exhibits it too).
    # Higher-confidence, better-aligned shots have larger ||e||, so weighting
    # by norm lets clean inputs dominate noisy ones with no extra signal.
    norms = np.linalg.norm(embs, axis=1)
    s = norms.sum()
    if s == 0:
        return np.mean(embs, axis=0)
    w = norms / s
    return (w[:, None] * embs).sum(axis=0)
