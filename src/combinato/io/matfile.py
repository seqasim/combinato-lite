"""MATLAB .mat continuous-data reader."""

from __future__ import annotations

import logging

import numpy as np
from scipy.io import loadmat

logger = logging.getLogger("combinato.io.matfile")

DEFAULT_MAT_SR = 24000


def read_matfile(fname: str, scale_factor: float = 1.0):
    """
    Read continuous data from a MATLAB file.

    Expects variables ``data`` (1-D) and optionally ``sr`` (sampling rate Hz).
    Returns (fdata, atimes_ms, timestep_s).
    """
    data = loadmat(fname)
    try:
        sr = float(np.asarray(data["sr"]).ravel()[0])
        insert = "stored"
    except KeyError:
        sr = float(DEFAULT_MAT_SR)
        insert = "default"

    logger.info("Using %s sampling rate (%.1f kHz)", insert, sr / 1000.0)
    ts = 1.0 / sr
    fdata = np.asarray(data["data"]).ravel().astype(np.float64)
    if scale_factor != 1.0:
        fdata = fdata * scale_factor
    atimes = np.linspace(0, fdata.shape[0] / (sr / 1000.0), fdata.shape[0])
    return fdata, atimes, ts
