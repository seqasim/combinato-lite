"""Prepare clustering sessions from a spike data file."""

from __future__ import annotations

import logging
import os
from typing import Optional

import numpy as np

from ..config import get_config
from ..io.h5store import DataStore, create_session

logger = logging.getLogger("combinato.cluster.prepare")


def make_arguments(
    filename: str,
    sign: str,
    mode: str = "index",
    start: int = 0,
    stop: Optional[int] = None,
    max_nspk_session: Optional[int] = None,
    add_one: bool = False,
) -> Optional[list[np.ndarray]]:
    cfg = get_config()
    max_nspk = max_nspk_session or cfg.max_nspk_session
    with DataStore(filename, "r") as store:
        non_artifact_idx, num_spikes = store.get_non_artifact_index(sign)
        if num_spikes == 0:
            logger.info("No spikes found in %s (%s)", filename, sign)
            return None

        if mode == "time":
            times = store.get_times(sign, non_artifact_idx)
            start_idx = int(np.searchsorted(times, start))
            stop_idx = int(np.searchsorted(times, stop))
            if add_one:
                start_idx += 1
                stop_idx -= 1
        else:
            start_idx = start
            stop_idx = stop if stop is not None else int(non_artifact_idx[-1])

        starts = list(range(start_idx, stop_idx, max_nspk))
        stops = starts[1:]
        min_stop = min(stop_idx, int(non_artifact_idx[-1]))
        stops.append(min_stop)

        if len(stops) > 1 and stops[-1] - stops[-2] < max_nspk / 5:
            stops[-2] = stops[-1]
            del starts[-1], stops[-1]

        stops[-1] += 1
        return [non_artifact_idx[a:b] for a, b in zip(starts, stops)]


def prepare_sessions(
    fnames: list[str],
    sign: str = "pos",
    mode: str = "index",
    start: int = 0,
    stop: Optional[int] = None,
    max_nspk_session: Optional[int] = None,
    label: str = "sort",
    replace: bool = True,
) -> list[tuple[str, str, str]]:
    """
    Create session folders. Returns list of (datafile, sign, session_path).
    """
    ret = []
    for name in fnames:
        jobs = make_arguments(name, sign, mode, start, stop, max_nspk_session)
        if jobs is None:
            continue
        dirname = os.path.dirname(name) or "."
        for job in jobs:
            if job.shape[0] == 0:
                continue
            session_name = create_session(dirname, sign, label, job, replace)
            ret.append((name, sign, os.path.join(dirname, session_name)))
    return ret
