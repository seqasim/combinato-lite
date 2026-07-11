"""h5py-based spike / sorting stores preserving Combinato HDF5 layout."""

from __future__ import annotations

import logging
import os
from typing import Optional

import h5py
import numpy as np

from ..constants import GROUP_ART, GROUP_NOCLASS, SIGNS, TYPE_ART, TYPE_MU, TYPE_NO

logger = logging.getLogger("combinato.io.h5store")


class DataStore:
    """
    Spike data file ``data_<name>.h5``.

    Layout (Combinato-compatible):
      /pos/spikes (N, 64) float32
      /pos/times  (N,)    float64
      /neg/spikes ...
      /neg/times  ...
      /thr        (M, 3)  float64
    """

    def __init__(self, path: str, mode: str = "r"):
        self.path = path
        self._f = h5py.File(path, mode)
        self._cache: dict = {}

    def close(self):
        if self._f is not None:
            self._f.close()
            self._f = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    @staticmethod
    def create(path: str, spoints: int = 64) -> "DataStore":
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with h5py.File(path, "w") as f:
            for sign in SIGNS:
                g = f.create_group(sign)
                g.create_dataset(
                    "spikes",
                    shape=(0, spoints),
                    maxshape=(None, spoints),
                    dtype="f4",
                    chunks=True,
                )
                g.create_dataset(
                    "times",
                    shape=(0,),
                    maxshape=(None,),
                    dtype="f8",
                    chunks=True,
                )
            f.create_dataset(
                "thr",
                shape=(0, 3),
                maxshape=(None, 3),
                dtype="f8",
                chunks=True,
            )
        return DataStore(path, "r+")

    def append(self, pos_spikes, pos_times, neg_spikes, neg_times, thr_row):
        """Append one extraction chunk (Combinato OutFile.write semantics)."""
        if len(pos_spikes):
            self._append_sign("pos", pos_spikes, pos_times)
        if len(neg_spikes):
            self._append_sign("neg", neg_spikes, neg_times)
        thr = self._f["thr"]
        n = thr.shape[0]
        thr.resize((n + 1, 3))
        thr[n] = thr_row
        self._f.flush()

    def _append_sign(self, sign: str, spikes, times):
        g = self._f[sign]
        n = g["spikes"].shape[0]
        m = spikes.shape[0]
        g["spikes"].resize((n + m, spikes.shape[1]))
        g["spikes"][n:] = spikes.astype(np.float32)
        g["times"].resize((n + m,))
        g["times"][n:] = np.asarray(times, dtype=np.float64)

    def n_spikes(self, sign: str) -> int:
        if sign not in self._f:
            return 0
        return int(self._f[sign]["times"].shape[0])

    def get_spikes(self, sign: str, index=None) -> np.ndarray:
        ds = self._f[sign]["spikes"]
        if index is None:
            return ds[:]
        return ds[np.asarray(index)]

    def get_times(self, sign: str, index=None) -> np.ndarray:
        ds = self._f[sign]["times"]
        if index is None:
            return ds[:]
        return ds[np.asarray(index)]

    def get_thresholds(self) -> Optional[np.ndarray]:
        if "thr" not in self._f:
            return None
        return self._f["thr"][:]

    def get_non_artifact_index(self, sign: str) -> tuple[np.ndarray, int]:
        """
        Return indices of non-artifact spikes.

        If ``/<sign>/artifacts`` exists (uint8 mask), exclude nonzero entries;
        otherwise return all indices.
        """
        n = self.n_spikes(sign)
        if n == 0:
            return np.array([], dtype=np.uint32), 0
        art_path = f"{sign}/artifacts"
        if art_path in self._f:
            art = self._f[art_path][:]
            idx = (art == 0).nonzero()[0].astype(np.uint32)
        else:
            idx = np.arange(n, dtype=np.uint32)
        return idx, int(idx.shape[0])


class SessionStore:
    """Per-session ``sorting.h5`` inside ``sort_<sign>_<label>_<start>_<stop>/``."""

    def __init__(self, path: str, mode: str = "r+"):
        # Accept either the session directory or the sorting.h5 path
        if os.path.isdir(path):
            self.session_dir = path
            self.path = os.path.join(path, "sorting.h5")
        else:
            self.path = path
            self.session_dir = os.path.dirname(path)
        try:
            self._f = h5py.File(self.path, mode)
        except OSError:
            self._f = h5py.File(self.path, "r")
            logger.info("Opening %s read-only", self.path)

        self.index = self._f["index"][:]
        self.ident = self._f.attrs.get("ident", "")
        if isinstance(self.ident, bytes):
            self.ident = self.ident.decode("utf-8")

        self.classes = self._read_optional("classes")
        self.matches = self._read_optional("matches")
        self.artifact_scores = self._read_optional("artifact_scores")
        self.is_sorted = all(
            x is not None for x in (self.classes, self.matches, self.artifact_scores)
        )

    def _read_optional(self, name):
        if name in self._f:
            return self._f[name][:]
        return None

    def close(self):
        if self._f is not None:
            self._f.close()
            self._f = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _update_node(self, name: str, data: np.ndarray, dtype):
        arr = np.asarray(data, dtype=dtype)
        if name in self._f:
            del self._f[name]
        self._f.create_dataset(name, data=arr)
        self._f.flush()

    def update_classes(self, classes: np.ndarray):
        self._update_node("classes", classes, np.uint16)
        self.classes = np.asarray(classes, dtype=np.uint16)

    def update_sorting_data(self, matches: np.ndarray, artifact_scores: np.ndarray):
        self._update_node("matches", matches, np.uint8)
        self._update_node("artifact_scores", artifact_scores, np.uint8)
        self.matches = np.asarray(matches, dtype=np.uint8)
        self.artifact_scores = np.asarray(artifact_scores, dtype=np.uint8)
        self.is_sorted = True


def create_session(
    folder: str, sign: str, label: str, index: np.ndarray, replace: bool = False
) -> str:
    """Create a new sorting session directory and empty sorting.h5."""
    session_name = "sort_{}_{}_{:07d}_{:07d}".format(
        sign, label, int(index[0]), int(index[-1])
    )
    session_dir = os.path.join(folder, session_name)
    os.makedirs(session_dir, exist_ok=True)
    data_fname = os.path.join(session_dir, "sorting.h5")
    if os.path.exists(data_fname) and not replace:
        logger.info("Not replacing %s", data_fname)
        return session_name

    with h5py.File(data_fname, "w") as f:
        f.create_dataset("index", data=index.astype(np.uint32))
        f.attrs["ident"] = session_name
    return session_name


class SortingStore:
    """
    Concatenated sorting file ``sort_cat.h5``.

    Layout: index, classes, groups, matches, types, artifacts, distance;
    root attr ``sign``.
    """

    def __init__(self, path: str, mode: str = "r+"):
        self.path = path
        self.basedir = os.path.dirname(path)
        try:
            self._f = h5py.File(path, mode)
        except OSError:
            self._f = h5py.File(path, "r")
            logger.info("Opening %s read-only", path)

        self.index = self._f["index"][:]
        self.classes = self._f["classes"][:]
        self.groups = self._f["groups"][:] if "groups" in self._f else None
        self.types = self._f["types"][:] if "types" in self._f else None
        self.matches = self._f["matches"][:] if "matches" in self._f else None
        self.artifacts = self._f["artifacts"][:] if "artifacts" in self._f else None
        sign = self._f.attrs.get("sign", "pos")
        if isinstance(sign, bytes):
            sign = sign.decode("utf-8")
        self.sign = str(sign)

    def close(self):
        if self._f is not None:
            self._f.close()
            self._f = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def get_gids(self) -> np.ndarray:
        return np.unique(self.groups[:, 1])

    def get_cluster_ids_by_gid(self, gid: int) -> np.ndarray:
        idx = self.groups[:, 1] == gid
        return self.groups[idx, 0]

    def get_cluster_index(self, clid: int) -> np.ndarray:
        return self.index[self.classes == clid]

    def get_cluster_index_joined(self, gid: int) -> np.ndarray:
        clids = self.get_cluster_ids_by_gid(gid)
        parts = [self.get_cluster_index(clid) for clid in clids]
        if not parts:
            return np.array([], dtype=np.uint32)
        return np.sort(np.hstack(parts))

    def get_group_type(self, gid: int) -> int:
        idx = self.types[:, 0] == gid
        return int(self.types[idx, 1][0])

    def save_groups_and_types(self, groups: np.ndarray, types: np.ndarray):
        self.groups = groups
        self.types = types
        self._f["groups"][:] = groups
        if "types" in self._f:
            del self._f["types"]
        self._f.create_dataset("types", data=types)
        self._f.flush()

    def set_group_type(self, gid: int, new_type: int):
        idx = self.types[:, 0] == gid
        self.types[idx, 1] = new_type
        self._f["types"][:] = self.types
        self._f.flush()

    def merge_groups(self, gids: list[int], target_gid: Optional[int] = None) -> int:
        """Merge listed groups into target_gid (or the first). Returns target gid."""
        if not gids:
            raise ValueError("No groups to merge")
        target = int(target_gid if target_gid is not None else gids[0])
        for gid in gids:
            if gid == target:
                continue
            self.groups[self.groups[:, 1] == gid, 1] = target
        # Drop empty type rows for merged-away gids
        keep = np.isin(self.types[:, 0], np.unique(self.groups[:, 1]))
        self.types = self.types[keep]
        self.save_groups_and_types(self.groups, self.types)
        return target


def write_sorting_file(
    h5fname: str,
    sorted_index: np.ndarray,
    classes: np.ndarray,
    matches: np.ndarray,
    artifacts: np.ndarray,
    sign: str,
    recheck_artifacts: bool = True,
):
    """Create concatenated sort_cat.h5 (pre-grouping)."""
    with h5py.File(h5fname, "w") as f:
        f.create_dataset("index", data=sorted_index.astype(np.uint32))
        f.create_dataset("classes", data=classes.astype(np.uint16))
        # Placeholder per-spike groups; replaced by (clid, gid) in grouping
        f.create_dataset(
            "groups", data=np.zeros(classes.shape[0], dtype=np.int8)
        )
        f.create_dataset("matches", data=matches.astype(np.int8))
        f.create_dataset(
            "distance", data=np.zeros(sorted_index.shape[0], dtype=np.float32)
        )
        if recheck_artifacts:
            art = artifacts.copy()
            art[:, 1] = 0
            f.create_dataset("artifacts_prematch", data=artifacts)
            f.create_dataset("artifacts", data=art)
        else:
            f.create_dataset("artifacts", data=artifacts)
        f.attrs["sign"] = sign


def write_groups_and_types(
    sorting_path: str,
    group_arr: np.ndarray,
    types: np.ndarray,
):
    """Write / update groups and types on an existing sort_cat.h5."""
    with h5py.File(sorting_path, "r+") as f:
        for name, data in (
            ("groups", group_arr),
            ("groups_orig", group_arr),
            ("types", types),
            ("types_orig", types),
        ):
            if name in f:
                del f[name]
            f.create_dataset(name, data=data)
        f.flush()
