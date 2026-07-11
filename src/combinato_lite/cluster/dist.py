"""Distance / template-matching helpers (ported from Combinato)."""

from __future__ import annotations

import logging

import numpy as np

from ..config import get_config
from ..constants import CLID_UNMATCHED

logger = logging.getLogger("combinato_lite.cluster.dist")


def distances_euclidean(all_spikes, templates):
    ret = np.empty((all_spikes.shape[0], templates.shape[0]))
    for i, template in enumerate(templates):
        ret[:, i] = np.sqrt(((all_spikes - template) ** 2).sum(1))
    return ret


def get_means(classes, all_spikes):
    assert classes.shape[0] == all_spikes.shape[0]
    cl_ids = np.unique(classes)
    ids, means, stds = [], [], []
    cfg = get_config()
    for clid in cl_ids:
        if clid == CLID_UNMATCHED:
            continue
        meandata = all_spikes[classes == clid]
        if meandata.shape[0]:
            ids.append(clid)
            means.append(meandata.mean(0))
            stds.append(np.sqrt(meandata.var(0).sum()))
            if cfg.Debug:
                logger.debug("class %s has stdval: %.3f", clid, stds[-1])
    if not means:
        empty = np.array([])
        return empty, empty, empty
    return np.array(ids), np.vstack(means), np.array(stds)


def template_match(spikes, sort_idx, match_idx, factor):
    cfg = get_config()
    num_samples = spikes.shape[1]
    unmatched_idx = sort_idx == CLID_UNMATCHED
    class_ids = np.unique(sort_idx[~unmatched_idx])
    if not len(class_ids):
        return

    ids, mean_array, stds = get_means(sort_idx, spikes)
    if cfg.ExcludeVariableClustersMatch and len(stds):
        median_std = np.median(stds)
        std_too_high_idx = stds > 3 * median_std
        mean_array = mean_array[~std_too_high_idx]
        ids = ids[~std_too_high_idx]
        stds = stds[~std_too_high_idx]

    if not len(ids):
        return

    all_distances = distances_euclidean(spikes[unmatched_idx], mean_array)
    all_distances[all_distances > factor * stds] = np.inf
    minimizers_idx = all_distances.argmin(1)
    minimizers = ids[minimizers_idx]
    minima = all_distances.min(1)
    minimizers[minima >= cfg.FirstMatchMaxDist * num_samples] = CLID_UNMATCHED
    sort_idx[unmatched_idx] = minimizers
    match_idx[unmatched_idx] = minimizers


def distance_groups(in1, in2, sign="pos"):
    dist = in1 - in2
    if sign == "pos":
        dist /= min(in1.max(), in2.max())
    elif sign == "neg":
        dist /= max(in1.min(), in2.min())
    else:
        raise Warning(f"Undefined sign: {sign}")
    l2_dist = np.sqrt((dist**2).sum())
    linf = np.abs(dist).max()
    return (l2_dist + 7 * linf) / 2
