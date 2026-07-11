"""FastAPI web GUI for manual spike-sorting curation."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ..constants import TYPE_ART, TYPE_MU, TYPE_NAMES, TYPE_NO, TYPE_SU
from ..io.h5store import DataStore, SortingStore

logger = logging.getLogger("combinato.gui")

STATIC_DIR = Path(__file__).resolve().parent / "static"

# Module-level session state for the single-user local GUI
_STATE: dict = {}


class MergeRequest(BaseModel):
    gids: list[int]
    target: Optional[int] = None


class TypeRequest(BaseModel):
    gid: int
    type: int


def cross_correlogram(times1, times2, lag, is_same):
    """Pure-NumPy cross-correlogram (replaces Cython .so)."""
    empty = np.array([], dtype=np.float64)
    if times1.shape[0] < times2.shape[0]:
        outer, inner, invert = times1, times2, True
    else:
        outer, inner, invert = times2, times1, False

    lags = []
    for i in range(outer.shape[0]):
        temp = outer[i]
        start = inner.searchsorted(temp - lag)
        stop = inner.searchsorted(temp + lag, side="right")
        if is_same:
            if i > start:
                lags.append(inner[start:i] - temp)
            if i + 1 < stop:
                lags.append(inner[i + 1 : stop] - temp)
            if i < start or i >= stop:
                lags.append(inner[start:stop] - temp)
        else:
            if start < stop:
                lags.append(inner[start:stop] - temp)

    if not lags:
        return empty
    res = np.hstack(lags)
    if invert:
        res *= -1
    return res


def create_app(datafile: str, sorting_path: str) -> FastAPI:
    app = FastAPI(title="Combinato-Lite GUI")
    _STATE["datafile"] = datafile
    _STATE["sorting_path"] = sorting_path
    _STATE["data"] = DataStore(datafile, "r")
    _STATE["sorting"] = SortingStore(sorting_path, "r+")

    if STATIC_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/")
    def index():
        index_path = STATIC_DIR / "index.html"
        if not index_path.is_file():
            raise HTTPException(404, "GUI static files missing")
        return FileResponse(index_path)

    @app.get("/api/info")
    def info():
        s = _STATE["sorting"]
        return {
            "datafile": datafile,
            "sorting": sorting_path,
            "sign": s.sign,
            "n_spikes": int(s.index.shape[0]),
            "type_names": {str(k): v for k, v in TYPE_NAMES.items()},
        }

    @app.get("/api/groups")
    def list_groups():
        s: SortingStore = _STATE["sorting"]
        data: DataStore = _STATE["data"]
        out = []
        for gid in s.get_gids():
            gid = int(gid)
            idx = s.get_cluster_index_joined(gid)
            n = int(idx.shape[0])
            gtype = int(s.get_group_type(gid))
            mean = None
            if n > 0:
                spikes = data.get_spikes(s.sign, idx)
                mean = spikes.mean(0).tolist()
            out.append(
                {
                    "gid": gid,
                    "n_spikes": n,
                    "type": gtype,
                    "type_name": TYPE_NAMES.get(gtype, str(gtype)),
                    "n_clusters": int(len(s.get_cluster_ids_by_gid(gid))),
                    "mean_waveform": mean,
                }
            )
        out.sort(key=lambda g: (-g["n_spikes"], g["gid"]))
        return out

    @app.get("/api/groups/{gid}")
    def group_detail(gid: int, max_waveforms: int = 200):
        s: SortingStore = _STATE["sorting"]
        data: DataStore = _STATE["data"]
        idx = s.get_cluster_index_joined(gid)
        if idx.size == 0:
            raise HTTPException(404, f"Group {gid} empty or missing")
        spikes = data.get_spikes(s.sign, idx)
        times = data.get_times(s.sign, idx)
        # Subsample waveforms for plotting
        if spikes.shape[0] > max_waveforms:
            sel = np.linspace(0, spikes.shape[0] - 1, max_waveforms, dtype=int)
            plot_spikes = spikes[sel]
        else:
            plot_spikes = spikes
        return {
            "gid": gid,
            "type": int(s.get_group_type(gid)),
            "n_spikes": int(idx.shape[0]),
            "mean_waveform": spikes.mean(0).tolist(),
            "waveforms": plot_spikes.tolist(),
            "times": times.tolist(),
            "cluster_ids": s.get_cluster_ids_by_gid(gid).astype(int).tolist(),
        }

    @app.get("/api/xcorr")
    def xcorr(gid_a: int, gid_b: int, lag_ms: float = 50.0, bins: int = 100):
        s: SortingStore = _STATE["sorting"]
        data: DataStore = _STATE["data"]
        idx_a = s.get_cluster_index_joined(gid_a)
        idx_b = s.get_cluster_index_joined(gid_b)
        times_a = np.sort(data.get_times(s.sign, idx_a))
        times_b = np.sort(data.get_times(s.sign, idx_b))
        # times are in ms in Combinato
        lags = cross_correlogram(times_a, times_b, lag_ms, gid_a == gid_b)
        if lags.size == 0:
            counts = [0] * bins
            edges = np.linspace(-lag_ms, lag_ms, bins + 1).tolist()
        else:
            counts, edges = np.histogram(lags, bins=bins, range=(-lag_ms, lag_ms))
            counts = counts.tolist()
            edges = edges.tolist()
        return {"counts": counts, "edges": edges, "n_lags": int(lags.size)}

    @app.post("/api/merge")
    def merge(req: MergeRequest):
        s: SortingStore = _STATE["sorting"]
        if len(req.gids) < 2:
            raise HTTPException(400, "Need at least two groups to merge")
        target = s.merge_groups(req.gids, req.target)
        return {"ok": True, "target": target}

    @app.post("/api/set_type")
    def set_type(req: TypeRequest):
        if req.type not in (TYPE_ART, TYPE_NO, TYPE_MU, TYPE_SU):
            raise HTTPException(400, f"Invalid type {req.type}")
        s: SortingStore = _STATE["sorting"]
        s.set_group_type(req.gid, req.type)
        return {"ok": True, "gid": req.gid, "type": req.type}

    @app.post("/api/reload")
    def reload():
        _STATE["sorting"].close()
        _STATE["sorting"] = SortingStore(_STATE["sorting_path"], "r+")
        return {"ok": True}

    return app


def run_gui(datafile: str, sorting_path: str, host: str = "127.0.0.1", port: int = 8765):
    app = create_app(datafile, sorting_path)
    logger.info("Open http://%s:%d  (SSH: ssh -L %d:localhost:%d user@host)", host, port, port, port)
    print(f"\n  Combinato-Lite GUI → http://{host}:{port}")
    print(f"  Remote tip: ssh -L {port}:localhost:{port} user@amarel\n")
    uvicorn.run(app, host=host, port=port, log_level="info")
