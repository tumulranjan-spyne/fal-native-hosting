"""Merge multiple per-image payloads into batched tensors for one GPU forward pass."""
from __future__ import annotations

import os

import numpy as np

_DEFAULT_MAX = 16


def max_batch_size() -> int:
    try:
        return max(1, int(os.environ.get("FAL_NATIVE_MAX_BATCH", str(_DEFAULT_MAX))))
    except ValueError:
        return _DEFAULT_MAX


def samples_to_stacked_inputs(samples: list[dict]) -> tuple[list[np.ndarray], int]:
    """
    Each sample is {"inputs": [ndarray, ...]} with the same arity and identical
    shapes per input slot (so they can be stacked on axis 0).
    """
    if not samples:
        raise ValueError("empty samples")
    cap = max_batch_size()
    if len(samples) > cap:
        raise ValueError(f"batch size {len(samples)} exceeds FAL_NATIVE_MAX_BATCH ({cap})")
    n_inputs = len(samples[0]["inputs"])
    for idx, s in enumerate(samples):
        if "inputs" not in s or len(s["inputs"]) != n_inputs:
            raise ValueError(f"sample {idx}: expected {n_inputs} inputs per sample")
    b = len(samples)
    merged: list[np.ndarray] = []
    for i in range(n_inputs):
        arrs = [np.asarray(s["inputs"][i]) for s in samples]
        shape0 = arrs[0].shape
        for idx, a in enumerate(arrs):
            if a.shape != shape0:
                raise ValueError(
                    f"input slot {i} shape mismatch: sample0 {shape0} vs sample{idx} {a.shape}"
                )
        merged.append(np.ascontiguousarray(np.stack(arrs, axis=0)))
    return merged, b
