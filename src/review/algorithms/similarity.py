"""Image similarity helpers for frame deduplication."""

from __future__ import annotations

import numpy as np


def frame_ssim(a: np.ndarray, b: np.ndarray) -> float:
    """Compute global SSIM on BT.601 luma and clamp to [-1, 1]."""
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    ga = np.dot(a[..., :3].astype(np.float64), [0.299, 0.587, 0.114])
    gb = np.dot(b[..., :3].astype(np.float64), [0.299, 0.587, 0.114])
    mu_a, mu_b = ga.mean(), gb.mean()
    var_a, var_b = ga.var(), gb.var()
    cov = float(np.mean((ga - mu_a) * (gb - mu_b)))
    num = (2 * mu_a * mu_b + c1) * (2 * cov + c2)
    den = (mu_a**2 + mu_b**2 + c1) * (var_a + var_b + c2)
    return float(np.clip(num / den, -1.0, 1.0))
