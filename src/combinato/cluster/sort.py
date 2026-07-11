"""Iterative spike sorting pipeline (ported from Combinato cluster.py)."""

from __future__ import annotations

import logging
import os
from concurrent.futures import ProcessPoolExecutor
from time import strftime
from typing import Optional

import numpy as np

from ..config import get_config
from ..io.h5store import DataStore, SessionStore
from .artifacts import find_artifacts
from .define_clusters import define_clusters
from .dist import template_match
from .features import wavelet_features
from .select_features import select_features
from .spc import Clusterer, get_clusterer

logger = logging.getLogger("combinato.cluster.sort")


def handle_random_seed(seed=None) -> float:
    if seed is None:
        random_seed = np.random.random() * 2**32
        logger.info("Generated random seed: %s", random_seed)
    else:
        random_seed = float(seed)
        logger.info("Prompted random seed: %s", random_seed)
    return random_seed


def features_to_index(
    features,
    folder,
    name,
    overwrite=True,
    clusterer: Optional[Clusterer] = None,
    seed: Optional[float] = None,
):
    clusterer = clusterer or get_clusterer()
    cfg = get_config()
    if seed is None:
        seed = handle_random_seed()

    clu = None
    if not overwrite:
        try:
            clu, tree = clusterer.read_results(folder, name)
            logger.info("Read clustering results from %s", folder)
        except OSError:
            logger.info("Starting clustering in %s", folder)
            overwrite = True

    if clu is not None and features.shape[0] != clu.shape[1] - 2:
        logger.info("Read outdated clustering, restarting")
        overwrite = True

    if overwrite:
        feat_idx = select_features(features)
        logger.info("Clustering data in %s/%s", folder, name)
        clusterer.cluster(features[:, feat_idx], folder, name, seed)
        clu, tree = clusterer.read_results(folder, name)

    idx, tree, used_points = define_clusters(clu, tree)
    return idx, tree, used_points


def cluster_step(features, folder, sub_name, overwrite, clusterer, seed):
    res_idx, tree, used_points = features_to_index(
        features, folder, sub_name, overwrite, clusterer, seed
    )
    return res_idx


def iterative_sorter(
    features,
    spikes,
    n_iterations,
    name,
    overwrite=True,
    clusterer: Optional[Clusterer] = None,
    seed: Optional[float] = None,
):
    cfg = get_config()
    clusterer = clusterer or get_clusterer()
    if seed is None:
        seed = handle_random_seed()

    idx = np.zeros(features.shape[0], np.uint16)
    match_idx = np.zeros(features.shape[0], bool)

    for i in range(n_iterations):
        sub_idx = idx == 0
        sub_name = "sort_" + str(i)
        if sub_idx.sum() < cfg.MinInputSize:
            break

        res_idx = cluster_step(
            features[sub_idx], name, sub_name, overwrite, clusterer, seed
        )
        clustered_idx = res_idx > 0
        prev_idx_max = idx.max()
        res_idx[clustered_idx] += prev_idx_max
        idx[sub_idx] = res_idx

        if cfg.ReclusterClusters:
            clids = np.unique(res_idx[clustered_idx])
            for clid in clids:
                recluster_idx = idx == clid
                if recluster_idx.sum() < cfg.MinInputSizeRecluster:
                    continue
                sub_sub_name = "{}_{:02d}".format(sub_name, clid)
                recluster_res_idx = cluster_step(
                    features[recluster_idx],
                    name,
                    sub_sub_name,
                    overwrite,
                    clusterer,
                    seed,
                )
                biggest_clid = idx.max()
                recluster_res_idx[recluster_res_idx != 0] += biggest_clid
                idx[recluster_idx] = recluster_res_idx

        template_match(spikes, idx, match_idx, cfg.FirstMatchFactor)

    return idx, match_idx


def sort_spikes(spikes, folder, overwrite=False, sign="pos", clusterer=None, seed=None):
    cfg = get_config()
    n_iterations = cfg.RecursiveDepth
    all_features = wavelet_features(spikes)
    sorted_idx, match_idx = iterative_sorter(
        all_features,
        spikes,
        n_iterations,
        folder,
        overwrite=overwrite,
        clusterer=clusterer,
        seed=seed,
    )
    class_ids = np.unique(sorted_idx)
    if cfg.MarkArtifactClasses:
        invert = sign == "neg"
        _, artifact_ids = find_artifacts(spikes, sorted_idx, class_ids, invert)
    else:
        artifact_ids = []
    return sorted_idx, match_idx, artifact_ids


def sort_session(
    data_fname: str,
    session_fname: str,
    sign: str,
    overwrite: bool = False,
    clusterer=None,
    seed=None,
):
    """Sort spikes for one session; write results into session sorting.h5."""
    cfg = get_config()
    if seed is None:
        seed = handle_random_seed()

    with DataStore(data_fname, "r") as store:
        session = SessionStore(session_fname, "r+")
        idx = session.index
        spikes = store.get_spikes(sign, idx)
        sort_idx, match_idx, artifact_ids = sort_spikes(
            spikes,
            session.session_dir,
            overwrite=overwrite if overwrite is not None else cfg.overwrite,
            sign=sign,
            clusterer=clusterer,
            seed=seed,
        )

        seed_file = os.path.join(session.session_dir, "random_seed.txt")
        with open(seed_file, "a", encoding="utf-8") as f:
            f.write("{} | {}\n".format(strftime("%Y-%m-%d_%H-%M-%S"), seed))

        all_ids = np.unique(sort_idx)
        artifact_scores = np.zeros((len(all_ids), 2), np.uint8)
        artifact_scores[:, 0] = all_ids
        for cl_id in all_ids:
            row = artifact_scores[:, 0] == cl_id
            artifact_scores[row, 1] = 1 if cl_id in artifact_ids else 0

        session.update_classes(sort_idx)
        session.update_sorting_data(match_idx.astype(np.uint8), artifact_scores)
        session.close()


def _sort_helper(args):
    data_fname, sign, session_fname, seed = args
    sort_session(data_fname, session_fname, sign, overwrite=True, seed=seed)


def run_cluster_jobs(
    jobs: list[tuple[str, str, str]],
    single: bool = False,
    seed: Optional[float] = None,
):
    """
    jobs: list of (datafile, sign, session_path)
    """
    seed = handle_random_seed(seed)
    payload = [(j[0], j[1], j[2], seed) for j in jobs]
    logger.info("Starting %d clustering jobs", len(payload))
    if single or len(payload) == 1:
        for p in payload:
            _sort_helper(p)
    else:
        n = min(os.cpu_count() or 1, len(payload))
        with ProcessPoolExecutor(max_workers=n) as pool:
            list(pool.map(_sort_helper, payload))
