"""Define clusters from SPC temperature tree (ported from Combinato)."""

from __future__ import annotations

import numpy as np

from ..config import get_config


def find_relevant_tree_points(tree, min_spikes):
    max_clusters_per_temp = get_config().MaxClustersPerTemp
    ret = []
    for shift in range(max_clusters_per_temp):
        col_idx = 5 + shift
        col = tree[:, col_idx]
        rise = (col[1:] > col[:-1]).nonzero()[0] + 1
        fall = (col[:-1] >= col[1:]).nonzero()[0]
        peaks = set(rise) & set(fall)
        if 1 in fall:
            peaks.add(1)
        for peak in peaks:
            nspk = tree[peak, col_idx]
            if nspk >= min_spikes:
                ret.append((peak, tree[peak, col_idx], shift + 1))
    return ret


def define_clusters(clu, tree):
    cfg = get_config()
    min_spikes = cfg.MinSpikesPerClusterMultiSelect
    relevant_rows = find_relevant_tree_points(tree, min_spikes)
    num_features = clu.shape[1] - 2
    idx = np.zeros(num_features, dtype=np.uint8)
    used_points = []
    current_id = 2
    max_row = 0
    for row, _, col in relevant_rows:
        row_idx = (clu[row, 2:] == col) & (idx == 0)
        if row_idx.any():
            idx[row_idx] = current_id
            current_id += 1
            p_type = "k"
            max_row = max(max_row, row)
        else:
            p_type = "r"
        used_points.append((row, col + 4, p_type))

    if len(used_points):
        row_idx = clu[max_row, 2:] == 0
        used_points.append((max_row, 4, "m"))
    else:
        row_idx = clu[1, 2:] == 0
        used_points.append((1, 4, "c"))
    idx[row_idx] = 1
    return idx, tree, used_points
