# backend/capability_core.py

from __future__ import annotations

import math
import numpy as np


def compute_ewma(values, sigma_within, lam=0.2, L=3):
    """EWMA hodnoty a konstantní kontrolní meze."""
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return None, None, None

    mean = np.mean(values)
    z = np.zeros_like(values, dtype=float)
    z[0] = mean
    for i in range(1, len(values)):
        z[i] = lam * values[i] + (1 - lam) * z[i - 1]

    # dlouhodobý rozptyl EWMA (pro konstantní meze)
    sigma_z = sigma_within * math.sqrt(lam / (2 - lam))
    UCL = mean + L * sigma_z
    LCL = mean - L * sigma_z
    return z, UCL, LCL


def compute_cusum(values, sigma_within, k_mult=0.5, h_mult=5.0):
    """Jednoduchý dvoustranný CUSUM (C+ a C-)."""
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return None, None, None, None

    target = np.mean(values)
    k = k_mult * sigma_within
    h = h_mult * sigma_within

    c_plus = np.zeros_like(values, dtype=float)
    c_minus = np.zeros_like(values, dtype=float)

    for i in range(1, len(values)):
        c_plus[i] = max(0, c_plus[i - 1] + values[i] - target - k)
        c_minus[i] = min(0, c_minus[i - 1] + values[i] - target + k)

    return c_plus, c_minus, h, -h
