"""Concatenate session sortings and run second-pass template match."""

from __future__ import annotations

import logging
import os
from typing import Optional

import h5py
import numpy as np

from ..config import get_config
from ..constants import CLID_UNMATCHED, SPIKE_MATCHED_2
from ..io.h5store import DataStore, write_sorting_file
from .artifacts import find_artifacts
from .dist import distances_euclidean, get_means
from .grouping import group_sorting

logger = logging.getLogger("combinato_lite.cluster.combine")

COL_CLASS = 0
COL_MATCH_TYPE = 2


def read_all_info(session_paths: list[str]):
    """Merge per-session sorting.h5 into global index/classes/matches/artifacts."""
    from ..io.h5store import SessionStore

    man_names = sorted(session_paths)
    managers = {ses: SessionStore(ses, "r") for ses in man_names}

    num = sum(managers[ses].index.shape[0] for ses in man_names)
    sorted_index = np.zeros(num, dtype=np.uint32)
    sorted_info = np.zeros((num, 4), dtype=np.int16)

    curr_idx = 0
    old_max_class = 0
    artifact_scores = [[0, 0]]

    for ses in man_names:
        man = managers[ses]
        t_size = man.index.shape[0]
        sorted_index[curr_idx : curr_idx + t_size] = man.index
        t_classes = man.classes.copy()
        idx = t_classes != 0
        t_classes[idx] += old_max_class
        t_arti = man.artifact_scores.astype(np.int16)
        if t_arti[0, 0] == CLID_UNMATCHED:
            t_arti = t_arti[1:, :]
        t_arti[:, 0] += old_max_class
        artifact_scores.append(t_arti)
        sorted_info[curr_idx : curr_idx + t_size, COL_CLASS] = t_classes
        sorted_info[curr_idx : curr_idx + t_size, COL_MATCH_TYPE] = man.matches
        curr_idx += t_size
        old_max_class = int(t_classes.max())
        man.close()

    return sorted_index, sorted_info, np.vstack(artifact_scores)


def total_match(h5path: str, all_spikes: np.ndarray):
    cfg = get_config()
    with h5py.File(h5path, "r+") as fid:
        classes = fid["classes"][:]
        ids, mean_array, stds = get_means(classes, all_spikes)
        if not len(ids):
            return fid["classes"][:], fid["matches"][:]

        unmatched_idx = (classes == CLID_UNMATCHED).nonzero()[0]
        blocksize = 50_000
        n_unmatched = unmatched_idx.shape[0]
        starts = np.arange(0, n_unmatched, blocksize)
        if not len(starts):
            starts = np.array([0])
            stops = np.array([n_unmatched])
        else:
            stops = starts + blocksize
            stops[-1] = n_unmatched

        for start, stop in zip(starts, stops):
            this_idx = unmatched_idx[start:stop]
            all_dists = distances_euclidean(all_spikes[this_idx], mean_array)
            all_dists[all_dists > cfg.SecondMatchFactor * stds] = np.inf
            minimizers_idx = all_dists.argmin(1)
            minimizers = ids[minimizers_idx]
            minima = all_dists.min(1)
            minimizers[minima >= cfg.SecondMatchMaxDist * all_spikes.shape[1]] = 0
            fid["classes"][this_idx] = minimizers
            fid["matches"][this_idx] = SPIKE_MATCHED_2
            fid["distance"][this_idx] = minima
        fid.flush()
        return fid["classes"][:], fid["matches"][:]


def combine_sessions(
    datafile: str,
    sessions: list[str],
    label: str,
    do_groups: bool = True,
) -> Optional[str]:
    """
    Concatenate session sortings into ``<label>/sort_cat.h5``.

    ``sessions`` may be session directory names or full paths.
    """
    cfg = get_config()
    basedir = os.path.dirname(datafile) or "."
    sorting_dir = os.path.join(basedir, label)
    os.makedirs(sorting_dir, exist_ok=True)
    outfname = os.path.join(sorting_dir, "sort_cat.h5")

    if not cfg.OverwriteGroups and os.path.exists(outfname):
        logger.info("%s exists already, skipping", outfname)
        return None

    # Resolve session dirs
    session_dirs = []
    for ses in sessions:
        if os.path.isdir(ses):
            session_dirs.append(ses)
        else:
            session_dirs.append(os.path.join(basedir, ses))

    sign = "neg" if "neg" in os.path.basename(session_dirs[0]) else "pos"
    sorted_index, sorted_info, artifacts = read_all_info(session_dirs)
    write_sorting_file(
        outfname,
        sorted_index,
        sorted_info[:, COL_CLASS],
        sorted_info[:, COL_MATCH_TYPE],
        artifacts,
        sign=sign,
        recheck_artifacts=cfg.RecheckArtifacts,
    )

    with DataStore(datafile, "r") as store:
        with h5py.File(outfname, "r+") as fid:
            spk_idx = fid["index"][:]
        all_spikes = store.get_spikes(sign, spk_idx)

    classes, _matches = total_match(outfname, all_spikes)

    if cfg.RecheckArtifacts:
        clids = np.unique(classes)
        rows = len(clids)
        if 0 in clids:
            artifacts = np.zeros((rows, 2), dtype=np.int64)
            artifacts[:, 0] = clids
        else:
            artifacts = np.zeros((rows + 1, 2), dtype=np.int64)
            artifacts[1:, 0] = clids
        invert = sign == "neg"
        _, art_ids = find_artifacts(all_spikes, classes, clids, invert)
        for aid in art_ids:
            artifacts[artifacts[:, 0] == aid, 1] = 1
        with h5py.File(outfname, "r+") as fid:
            fid["artifacts"][:] = artifacts

    if do_groups:
        group_sorting(datafile, outfname)

    return outfname
