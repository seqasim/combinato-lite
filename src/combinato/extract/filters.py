"""Bandpass / notch filters for spike extraction (ported from Combinato)."""

from __future__ import annotations

import numpy as np
from scipy.signal import ellip, filtfilt

DETECT_LOW = 300
DETECT_HIGH = 1000
EXTRACT_LOW = 300
EXTRACT_HIGH = 3000


class DefaultFilter:
    """Elliptic bandpass filters for detection and extraction."""

    def __init__(self, timestep: float):
        self.sampling_rate = int(1.0 / timestep)
        self.timestep = timestep
        self.c_detect = ellip(
            2,
            0.1,
            40,
            (2 * timestep * DETECT_LOW, 2 * timestep * DETECT_HIGH),
            "bandpass",
        )
        self.c_extract = ellip(
            2,
            0.1,
            40,
            (2 * timestep * EXTRACT_LOW, 2 * timestep * EXTRACT_HIGH),
            "bandpass",
        )
        self.c_notch = ellip(
            2,
            0.5,
            20,
            (2 * timestep * 1999, 2 * timestep * 2001),
            "bandstop",
        )

    def filter_detect(self, x):
        b, a = self.c_detect
        return filtfilt(b, a, x)

    def filter_extract(self, x):
        b, a = self.c_extract
        return filtfilt(b, a, x)

    def filter_denoise(self, x):
        b, a = self.c_notch
        return filtfilt(b, a, x)
