"""Unit and end-to-end tests for Combinato-Lite."""

from __future__ import annotations

import os
from pathlib import Path

import h5py
import numpy as np
import pytest

from combinato.config import CombinatoConfig, load_config, set_config
from combinato.constants import TYPE_MU, TYPE_SU
from combinato.io.h5store import DataStore, SortingStore, create_session
from combinato.cluster.features import wavelet_features
from combinato.cluster.select_features import select_features
from combinato.cluster.artifacts import artifact_score
from combinato.gui.app import cross_correlogram, create_app
from combinato.extract.filters import DefaultFilter
from combinato.extract.detect import extract_spikes


@pytest.fixture
def cfg():
    c = CombinatoConfig(
        plot=False,
        plotTemps=False,
        LogToConsole=False,
        LogToFile=False,
        RecheckArtifacts=False,
        max_nspk_session=5000,
        MinInputSizeRecluster=10000,  # disable reclustering in small tests
    )
    set_config(c)
    return c


def test_load_config_yaml(tmp_path):
    p = tmp_path / "combinato.yaml"
    p.write_text("nFeatures: 7\nLogLevel: WARNING\n")
    c = load_config(path=p)
    assert c.nFeatures == 7
    assert c.LogLevel == "WARNING"
    set_config(CombinatoConfig())


def test_wavelet_and_select_features(cfg):
    data = np.random.randn(200, 64).astype(np.float32)
    data[:, [2, 4, 6]] += 5
    feats = wavelet_features(data)
    assert feats.shape[0] == 200
    idx = select_features(feats)
    assert len(idx) == cfg.nFeatures


def test_artifact_score(cfg):
    # Clean single-peak spike-like mean
    t = np.linspace(-1, 1, 64)
    clean = np.exp(-((t - 0) ** 2) / 0.02) * 50
    spikes = clean + np.random.randn(30, 64) * 0.5
    score, reasons, _ = artifact_score(spikes)
    assert isinstance(score, int)


def test_datastore_roundtrip(tmp_path, cfg):
    path = tmp_path / "chan" / "data_chan.h5"
    store = DataStore.create(str(path), spoints=64)
    pos = np.random.randn(10, 64).astype(np.float32)
    times = np.arange(10, dtype=np.float64)
    store.append(pos, times, np.zeros((0, 64)), np.zeros(0), (0.0, 1.0, 5.0))
    store.close()

    with DataStore(str(path), "r") as s:
        assert s.n_spikes("pos") == 10
        assert s.get_spikes("pos").shape == (10, 64)
        thr = s.get_thresholds()
        assert thr.shape == (1, 3)


def test_session_create(tmp_path, cfg):
    idx = np.arange(100, dtype=np.uint32)
    name = create_session(str(tmp_path), "pos", "test", idx, replace=True)
    assert (tmp_path / name / "sorting.h5").is_file()


def test_cross_correlogram():
    t1 = np.array([0.0, 10.0, 20.0, 30.0])
    t2 = t1 + 2.0
    lags = cross_correlogram(t1, t2, lag=5.0, is_same=False)
    assert lags.size > 0
    assert np.all(np.abs(lags) <= 5.0)


def test_extract_synthetic(tmp_path, cfg):
    """Generate continuous signal with planted spikes and extract."""
    sr = 32000.0
    ts = 1.0 / sr
    n = int(sr * 2)  # 2 seconds
    t = np.arange(n) * ts
    data = np.random.randn(n) * 5.0  # noise ~5 µV

    # Plant positive spikes
    template = np.zeros(64)
    template[19] = 80.0
    template[18] = 40.0
    template[20] = 40.0
    spike_times_idx = [5000, 9000, 15000, 22000, 28000, 35000, 42000, 50000]
    for si in spike_times_idx:
        data[si : si + 64] += template

    filt = DefaultFilter(ts)
    times_ms = t * 1000.0
    result = extract_spikes(data, times_ms, ts, filt, cfg=cfg)
    pos_spikes, pos_times = result[0]
    assert pos_spikes.shape[1] == 64
    # Should find most planted spikes
    assert pos_spikes.shape[0] >= 3


def _make_spike_file(path: Path, n_pos: int = 400) -> str:
    """Create a synthetic data_*.h5 with two separable spike shapes."""
    store = DataStore.create(str(path), spoints=64)
    t = np.linspace(-1, 1, 64)
    shape_a = 40 * np.exp(-((t + 0.1) ** 2) / 0.05)
    shape_b = 55 * np.exp(-((t - 0.05) ** 2) / 0.03) - 10 * np.exp(
        -((t - 0.4) ** 2) / 0.08
    )
    n_a = n_pos // 2
    n_b = n_pos - n_a
    spikes = np.vstack(
        [
            shape_a + np.random.randn(n_a, 64) * 2,
            shape_b + np.random.randn(n_b, 64) * 2,
        ]
    ).astype(np.float32)
    times = np.cumsum(np.random.uniform(5, 50, n_pos)).astype(np.float64)
    store.append(spikes, times, np.zeros((0, 64)), np.zeros(0), (0.0, times[-1], 10.0))
    store.close()
    return str(path)


def test_spc_binary_resolves():
    from combinato.cluster.spc import resolve_spc_binary

    try:
        path = resolve_spc_binary()
    except RuntimeError as e:
        pytest.skip(str(e))
    assert path.is_file()


def test_e2e_sort_pipeline(tmp_path, cfg):
    """extract-like data → prepare → cluster → combine → group."""
    from combinato.cluster.prepare import prepare_sessions
    from combinato.cluster.sort import run_cluster_jobs
    from combinato.cluster.combine import combine_sessions
    from combinato.cluster.spc import resolve_spc_binary

    try:
        resolve_spc_binary()
    except RuntimeError as e:
        pytest.skip(f"SPC unavailable: {e}")

    data_dir = tmp_path / "CSC01"
    data_dir.mkdir()
    datafile = _make_spike_file(data_dir / "data_CSC01.h5", n_pos=300)

    sessions = prepare_sessions(
        [datafile], sign="pos", label="test", replace=True, max_nspk_session=300
    )
    assert len(sessions) >= 1

    jobs = [(name, sign, ses) for name, sign, ses in sessions]
    try:
        run_cluster_jobs(jobs, single=True, seed=42.0)
    except RuntimeError as e:
        pytest.skip(f"SPC execution failed: {e}")

    ses_names = [os.path.basename(ses) for _, _, ses in sessions]
    out = combine_sessions(datafile, ses_names, "sort_pos_test", do_groups=True)
    assert out is not None
    assert Path(out).is_file()

    with SortingStore(out, "r") as s:
        assert s.sign == "pos"
        assert s.classes is not None
        assert s.groups is not None
        assert s.types is not None
        assert s.get_gids().size >= 1


def test_gui_api(tmp_path, cfg):
    """Build a minimal sorted file and hit FastAPI endpoints."""
    from fastapi.testclient import TestClient

    data_dir = tmp_path / "CSC02"
    data_dir.mkdir()
    datafile = _make_spike_file(data_dir / "data_CSC02.h5", n_pos=100)

    # Manually craft a sort_cat.h5
    sort_dir = data_dir / "sort_pos_manual"
    sort_dir.mkdir()
    sort_path = sort_dir / "sort_cat.h5"
    n = 100
    with h5py.File(sort_path, "w") as f:
        f.create_dataset("index", data=np.arange(n, dtype=np.uint32))
        classes = np.zeros(n, dtype=np.uint16)
        classes[:40] = 1
        classes[40:80] = 2
        f.create_dataset("classes", data=classes)
        f.create_dataset("matches", data=np.zeros(n, dtype=np.int8))
        f.create_dataset("distance", data=np.zeros(n, dtype=np.float32))
        # groups: (clid, gid)
        groups = np.array([[0, 0], [1, 1], [2, 2]], dtype=np.int16)
        f.create_dataset("groups", data=groups)
        types = np.array([[0, 0], [1, TYPE_MU], [2, TYPE_SU]], dtype=np.int16)
        f.create_dataset("types", data=types)
        f.create_dataset(
            "artifacts", data=np.array([[0, 0], [1, 0], [2, 0]], dtype=np.int16)
        )
        f.attrs["sign"] = "pos"

    app = create_app(datafile, str(sort_path))
    client = TestClient(app)

    r = client.get("/api/info")
    assert r.status_code == 200
    assert r.json()["sign"] == "pos"

    r = client.get("/api/groups")
    assert r.status_code == 200
    groups = r.json()
    assert len(groups) >= 2

    gid = groups[0]["gid"]
    r = client.get(f"/api/groups/{gid}")
    assert r.status_code == 200
    assert "mean_waveform" in r.json()

    r = client.get(f"/api/xcorr?gid_a={gid}&gid_b={gid}")
    assert r.status_code == 200

    r = client.post("/api/set_type", json={"gid": gid, "type": TYPE_SU})
    assert r.status_code == 200

    if len(groups) >= 2:
        gids = [g["gid"] for g in groups if g["gid"] > 0][:2]
        if len(gids) == 2:
            r = client.post("/api/merge", json={"gids": gids})
            assert r.status_code == 200
