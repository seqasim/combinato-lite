"""Wavelet feature extraction (ported from Combinato)."""

from __future__ import annotations

import numpy as np
import pywt

from ..config import get_config

OUT_DTYPE = np.float32
LEVEL = 4


def wavelet_features(data: np.ndarray, wavelet: str | None = None) -> np.ndarray:
    cfg = get_config()
    wave = pywt.Wavelet(wavelet or cfg.Wavelet)
    first_row = pywt.wavedec(data[0, :], wave, level=LEVEL)
    aligned = np.hstack(first_row)
    output = np.empty((data.shape[0], aligned.shape[0]), dtype=OUT_DTYPE)
    output[0, :] = aligned
    for i, row in enumerate(data[1:, :]):
        features = pywt.wavedec(row, wave, level=LEVEL)
        output[i + 1, :] = np.hstack(features)
    return output
