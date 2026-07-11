"""Group clusters by mean-waveform distance (ported from Combinato)."""

from __future__ import annotations

import logging

import h5py
import numpy as np

from ..config import get_config
from ..constants import CLID_UNMATCHED, GROUP_ART, GROUP_NOCLASS, TYPE_ART, TYPE_MU, TYPE_NO
from ..io.h5store import DataStore
from .dist import distance_groups

logger = logging.getLogger("combinato_lite.cluster.grouping")


def create_groups(spikes, classes, clids, sign):
    crit = get_config().MaxDistMatchGrouping
    groups = {}
    n_groups_in = len(clids) + 1
    means = np.empty((n_groups_in, spikes.shape[1]))
    nspks = np.empty(n_groups_in, int)
    dists = np.zeros((n_groups_in, n_groups_in))
    dists[:, :] = np.inf
    count = 1

    for clid in clids:
        if clid == CLID_UNMATCHED:
            continue
        count += 1
        groups[count] = [clid]
        idx = classes == clid
        nspks[count] = idx.sum()
        means[count, :] = spikes[idx].mean(0)

    for i in range(n_groups_in):
        if i not in groups:
            continue
        for j in range(i + 1, n_groups_in):
            if j not in groups:
                continue
            dists[i, j] = distance_groups(means[i, :], means[j, :], sign)

    while True:
        this_argmin = dists.argmin()
        gr1, gr2 = np.unravel_index(this_argmin, (n_groups_in, n_groups_in))
        minimum = dists[gr1, gr2]
        if minimum > crit:
            break
        logger.debug("Merging %s and %s, dist: %.4f", gr1, gr2, minimum)
        groups[gr1] += groups[gr2]
        del groups[gr2]
        nspk1 = nspks[gr1]
        nspk2 = nspks[gr2]
        nspks[gr1] = nspk1 + nspk2
        means[gr1, :] = (means[gr1, :] * nspk1 + means[gr2, :] * nspk2) / (
            nspk1 + nspk2
        )
        dists[gr2, :] = np.inf
        dists[:, gr2] = np.inf
        for i in groups.keys():
            if i < gr1:
                dists[i, gr1] = distance_groups(means[i], means[gr1], sign)
            elif i > gr2:
                dists[gr1, i] = distance_groups(means[i], means[gr1], sign)
    return groups


def group_sorting(datafname: str, sorting_fname: str, read_only: bool = False):
    """Create groups/types on a concatenated sort_cat.h5."""
    with DataStore(datafname, "r") as man, h5py.File(
        sorting_fname, "r" if read_only else "r+"
    ) as sort_fid:
        sign = sort_fid.attrs.get("sign", "pos")
        if isinstance(sign, bytes):
            sign = sign.decode("utf-8")
        idx = sort_fid["index"][:]
        spikes = man.get_spikes(sign, idx)
        classes = sort_fid["classes"][:]
        artifacts = sort_fid["artifacts"][:, :]
        group_arr = artifacts.copy().astype(np.int16)
        art_idx = (artifacts[:, 1] != 0) & (artifacts[:, 0] != 0)
        clids = artifacts[~art_idx, 0]
        group_arr[art_idx, 1] = GROUP_ART

        groups = create_groups(spikes, classes, clids, sign)
        for grid, orig_grid in enumerate(sorted(groups.keys())):
            for clid in groups[orig_grid]:
                gidx = group_arr[:, 0] == clid
                group_arr[gidx, 1] = grid + 1

        group_arr[group_arr[:, 0] == 0, 1] = GROUP_NOCLASS

        group_names = np.unique(group_arr[:, 1])
        types = np.zeros((group_names.shape[0], 2), np.int16)
        types[:, 0] = group_names
        types[:, 1] = TYPE_MU
        types[types[:, 0] == GROUP_ART, 1] = TYPE_ART
        types[types[:, 0] == GROUP_NOCLASS, 1] = TYPE_NO

        if not read_only:
            for name, data in (
                ("groups", group_arr),
                ("groups_orig", group_arr),
                ("types", types),
                ("types_orig", types),
            ):
                if name in sort_fid:
                    del sort_fid[name]
                sort_fid.create_dataset(name, data=data)
            sort_fid.flush()

    return group_arr, types
