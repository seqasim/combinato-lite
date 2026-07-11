"""Artifact cluster scoring (ported from Combinato)."""

from __future__ import annotations

import logging

import numpy as np

from ..config import get_config

logger = logging.getLogger("combinato_lite.cluster.artifacts")
TOLERANCE = 10


def find_maxima_ratio(data, tolerance):
    up = (data[1:] > data[:-1]).nonzero()[0] + 1
    down = (data[:-1] > data[1:]).nonzero()[0]
    peaks = np.intersect1d(up, down)
    peaks = np.append(peaks, len(data))
    idx = np.diff(peaks) >= tolerance
    num = idx.sum()
    if num > 1:
        vals = np.sort(data[peaks[idx.nonzero()[0]]])
        ratio = np.abs(vals[-1] / vals[-2])
    else:
        ratio = np.inf
    return num, ratio


def max_min_ratio(data):
    return np.abs(data.max() / data.min())


def std_err_mean(data):
    return data.std(0).mean() / np.sqrt(data.shape[0])


def peak_to_peak(data):
    cut = int(data.shape[0] / 2)
    return np.ptp(data[cut:] - data[0]) / data.max()


def artifact_score(data, criteria=None):
    crit = criteria or get_config().artifact_criteria
    mean = data.mean(0)
    num_peaks, peak_ratio = find_maxima_ratio(mean, TOLERANCE)
    ratio = max_min_ratio(mean)
    std_err = std_err_mean(data)
    ptp = peak_to_peak(mean)

    score = 0
    reasons = []
    if num_peaks > crit["maxima"]:
        score += 1
        reasons.append("maxima")
    if peak_ratio < crit["maxima_1_2_ratio"]:
        score += 1
        reasons.append("maxima_1_2_ratio")
    if ratio < crit["max_min_ratio"]:
        score += 1
        reasons.append("max_min_ratio")
    if std_err > crit["sem"]:
        score += 1
        reasons.append("sem")
    if ptp > crit["ptp"]:
        score += 1
        reasons.append("ptp")
    return score, reasons, mean


def find_artifacts(spikes, sorted_idx, class_ids, invert=False):
    cfg = get_config()
    artifact_idx = np.zeros(spikes.shape[0], np.uint8)
    artifact_ids = []
    for class_id in class_ids:
        if class_id == 0:
            continue
        class_idx = sorted_idx == class_id
        class_spikes = spikes[class_idx]
        if invert:
            class_spikes = -class_spikes
        score, reasons, _ = artifact_score(class_spikes)
        if cfg.Debug:
            logger.debug("%s %s %s", class_id, score, reasons)
        artifact_idx[class_idx] = score
        if score:
            artifact_ids.append(class_id)
    return artifact_idx, artifact_ids
