"""Threshold-based spike detection (ported from Combinato extract_spikes)."""

from __future__ import annotations

import logging

import numpy as np

from ..config import get_config
from .interpolate import align, clean, downsample, upsample

logger = logging.getLogger("combinato_lite.extract.detect")


def extract_spikes(data, times, timestep, filt, cfg=None):
    """
    Detect and extract positive/negative spikes from a continuous chunk.

    Returns [(pos_spikes, pos_times), (neg_spikes, neg_times), [(t0, t1, thr)]].
    """
    cfg = cfg or get_config()
    factor = cfg.upsampling_factor
    indices_per_spike = cfg.indices_per_spike
    pre_indices = cfg.index_maximum
    denoise = cfg.denoise
    post_indices = indices_per_spike - pre_indices

    result = []

    if denoise:
        data = filt.filter_denoise(data)

    if cfg.do_filter:
        data_detect = filt.filter_detect(data)
    else:
        data_detect = data

    data_extract = None

    noise_level = np.median(np.abs(data_detect)) / 0.6745
    threshold = cfg.threshold_factor * noise_level

    over_threshold = data_detect > threshold
    under_threshold = data_detect < -threshold

    borders = [0, 0]
    length_okay = [0, 0]
    num_spikes = [0, 0]

    borders[0] = np.diff(over_threshold).nonzero()[0]
    borders[1] = np.diff(under_threshold).nonzero()[0]

    for i in (0, 1):
        if borders[i].shape[0] % 2:
            borders[i] = borders[i][:-1]

    for i in [0, 1]:
        borders[i] = borders[i].reshape(-1, 2)
        length_okay[i] = (borders[i][:, 1] - borders[i][:, 0]) <= (
            cfg.max_spike_duration / timestep
        )
        num_spikes[i] = len(length_okay[i])

    for sign in [0, 1]:
        if num_spikes[sign] == 0:
            result.append((np.zeros((0, indices_per_spike)), np.zeros(0)))
            continue

        borders[sign] = borders[sign][length_okay[sign]]
        detect_func = np.argmin if sign == 1 else np.argmax
        maxima = [
            detect_func(data_detect[range(borders[sign][i, 0], borders[sign][i, 1])])
            + borders[sign][i, 0]
            for i in range(borders[sign].shape[0])
        ]

        if len(maxima) <= 3:
            result.append((np.zeros((0, indices_per_spike)), np.zeros(0)))
            continue

        maxima = np.array(maxima[1:-2])
        mindex = (maxima >= pre_indices + 5) & (
            maxima <= len(data) - post_indices - 5
        )
        maxima = maxima[mindex]
        if data_extract is None:
            if cfg.do_filter:
                data_extract = filt.filter_extract(data)
            else:
                data_extract = data

        extract_indices = [
            range(maxima[i] - pre_indices - 5, maxima[i] + post_indices + 5)
            for i in range(len(maxima))
        ]

        spikes = np.zeros((len(extract_indices), indices_per_spike + 10))
        for i, _ in enumerate(extract_indices):
            spikes[i] = data_extract[extract_indices[i]]

        if sign == 1:
            spikes *= -1

        spikes = upsample(spikes, factor)
        spikes, index_maximum = align(
            spikes, (pre_indices + 5) * factor, factor, factor
        )
        _, _removed = clean(spikes, index_maximum)
        timestamps = times[maxima]
        spikes, _ = downsample(
            spikes, index_maximum, factor, pre_indices, indices_per_spike
        )

        if sign == 1:
            spikes *= -1

        result.append((spikes, timestamps))

    result.append([(times[0], times[-1], threshold)])
    return result
