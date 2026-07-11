"""Multiprocess extraction pipeline using concurrent.futures."""

from __future__ import annotations

import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional

import h5py
import numpy as np

from ..config import CombinatoConfig, get_config
from ..io.h5store import DataStore
from ..io.matfile import read_matfile
from ..io.ncs import NCS_SAMPLES_PER_REC, NcsFile
from .detect import extract_spikes
from .filters import DefaultFilter

logger = logging.getLogger("combinato.extract.pipeline")

SAMPLES_PER_REC = NCS_SAMPLES_PER_REC


def _read_ncs_chunk(filename: str, start: int, stop: int, reference: Optional[str]):
    ncs = NcsFile(filename)
    data, times = ncs.read(start, stop, "both")
    fdata = np.array(data, dtype=np.float32)
    fdata *= 1e6 * ncs.header["ADBitVolts"]
    if reference is not None:
        ref = NcsFile(reference)
        ref_data = np.array(ref.read(start, stop, "data"), dtype=np.float32)
        ref_data *= 1e6 * ref.header["ADBitVolts"]
        fdata -= ref_data
    stepus = ncs.timestep * 1e6
    timerange = np.arange(0, SAMPLES_PER_REC * stepus, stepus)
    atimes = np.hstack([t + timerange for t in times]) / 1e3
    return fdata, atimes, ncs.timestep


def _read_h5_chunk(filename: str, start: int, stop: int):
    with h5py.File(filename, "r") as f:
        if f["data"].ndim != 1:
            raise ValueError("HDF5 continuous data must be 1-D under /data")
        fdata = f["data"][start:stop].ravel()
        sr = float(f["sr"][0]) if "sr" in f else 32000.0
    ts = 1.0 / sr
    atimes = np.linspace(0, fdata.shape[0] / (sr / 1000.0), fdata.shape[0])
    atimes += start / (sr / 1000.0)
    return fdata, atimes, ts


def _process_job(job: dict, cfg_dict: dict):
    """Worker: read chunk, extract spikes, return result payload."""
    from ..config import CombinatoConfig

    # asdict turns tuples into lists; coerce known tuple fields back
    fixed = dict(cfg_dict)
    for key in ("figsize", "ylim"):
        if key in fixed and isinstance(fixed[key], list):
            fixed[key] = tuple(fixed[key])
    cfg = CombinatoConfig(**fixed)
    kind = job.get("kind", "ncs")
    if kind == "ncs":
        data, times, ts = _read_ncs_chunk(
            job["filename"], job["start"], job["stop"], job.get("reference")
        )
    elif kind == "h5":
        data, times, ts = _read_h5_chunk(job["filename"], job["start"], job["stop"])
    elif kind == "mat":
        data, times, ts = read_matfile(job["filename"], job.get("scale_factor", 1.0))
    else:
        raise ValueError(f"Unknown job kind: {kind}")

    filt = DefaultFilter(ts)
    result = extract_spikes(data, times, ts, filt, cfg=cfg)
    return {
        "name": job["name"],
        "count": job["count"],
        "out_path": job["out_path"],
        "result": result,
    }


def _write_results_ordered(results: list[dict], spoints: int):
    """Write extraction results in count order into DataStore files."""
    by_name: dict[str, list] = {}
    for r in results:
        by_name.setdefault(r["name"], []).append(r)

    for name, items in by_name.items():
        items.sort(key=lambda x: x["count"])
        out_path = items[0]["out_path"]
        store = DataStore.create(out_path, spoints=spoints)
        for item in items:
            res = item["result"]
            thr = res[2][0]
            store.append(res[0][0], res[0][1], res[1][0], res[1][1], thr)
        store.close()
        logger.info("Wrote %s", out_path)


def extract_files(
    files: list[str],
    destination: str = "",
    start: Optional[int] = None,
    stop: Optional[int] = None,
    blocksize: int = 10000,
    n_workers: int = 4,
    refscheme: Optional[dict[str, str]] = None,
    cfg: Optional[CombinatoConfig] = None,
):
    """Extract spikes from Neuralynx .ncs files."""
    cfg = cfg or get_config()
    jobs = []
    for f in files:
        ncs = NcsFile(f)
        nrecs = ncs.num_recs
        del ncs
        s0 = start or 0
        s1 = min(stop, nrecs) if stop else nrecs
        if s1 % blocksize > blocksize / 2:
            laststart = s1 - blocksize
        else:
            laststart = s1
        starts = list(range(s0, laststart, blocksize))
        stops = starts[1:] + [s1]
        name = os.path.splitext(os.path.basename(f))[0]
        out_dir = os.path.join(destination, name) if destination else name
        out_path = os.path.join(out_dir, f"data_{name}.h5")
        reference = refscheme.get(f) if refscheme else None
        for i, (a, b) in enumerate(zip(starts, stops)):
            jobs.append(
                {
                    "kind": "ncs",
                    "name": name,
                    "filename": f,
                    "start": a,
                    "stop": b,
                    "count": i,
                    "out_path": out_path,
                    "reference": reference,
                }
            )
    return _run_jobs(jobs, n_workers, cfg)


def extract_h5(
    files: list[str],
    destination: str = "",
    n_workers: int = 4,
    cfg: Optional[CombinatoConfig] = None,
):
    """Extract from continuous HDF5 files with /data (and optional /sr)."""
    cfg = cfg or get_config()
    jobs = []
    chunk = 32000 * 5 * 60
    for f in files:
        with h5py.File(f, "r") as hf:
            size = hf["data"].shape[0]
        starts = list(range(0, size, chunk))
        stops = starts[1:] + [size]
        name = os.path.splitext(os.path.basename(f))[0]
        out_dir = os.path.join(destination, name) if destination else name
        out_path = os.path.join(out_dir, f"data_{name}.h5")
        for i, (a, b) in enumerate(zip(starts, stops)):
            jobs.append(
                {
                    "kind": "h5",
                    "name": name,
                    "filename": f,
                    "start": a,
                    "stop": b,
                    "count": i,
                    "out_path": out_path,
                }
            )
    return _run_jobs(jobs, n_workers, cfg)


def extract_matfile(
    matfile: str,
    destination: str = "",
    scale_factor: float = 1.0,
    cfg: Optional[CombinatoConfig] = None,
):
    """Extract from a single MATLAB file."""
    cfg = cfg or get_config()
    name = os.path.splitext(os.path.basename(matfile))[0]
    out_dir = os.path.join(destination, name) if destination else name
    out_path = os.path.join(out_dir, f"data_{name}.h5")
    jobs = [
        {
            "kind": "mat",
            "name": name,
            "filename": matfile,
            "count": 0,
            "out_path": out_path,
            "scale_factor": scale_factor,
        }
    ]
    return _run_jobs(jobs, 1, cfg)


def _run_jobs(jobs: list[dict], n_workers: int, cfg: CombinatoConfig):
    if not jobs:
        logger.warning("No extraction jobs")
        return []
    cfg_dict = cfg.to_dict()
    # artifact_criteria is fine; tuples stay as tuples in dataclass
    results = []
    n_workers = max(1, min(n_workers, len(jobs)))
    logger.info("Running %d extraction jobs with %d workers", len(jobs), n_workers)
    if n_workers == 1:
        for job in jobs:
            results.append(_process_job(job, cfg_dict))
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futs = [pool.submit(_process_job, job, cfg_dict) for job in jobs]
            for fut in as_completed(futs):
                results.append(fut.result())
    _write_results_ordered(results, spoints=cfg.indices_per_spike)
    return [r["out_path"] for r in results]
