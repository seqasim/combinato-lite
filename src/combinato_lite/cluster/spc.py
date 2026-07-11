"""SPC binary backend and Clusterer interface."""

from __future__ import annotations

import logging
import os
import platform
import shutil
import subprocess
import sys
from abc import ABC, abstractmethod
from importlib import resources
from pathlib import Path
from typing import Optional

import numpy as np

from ..config import get_config

logger = logging.getLogger("combinato_lite.cluster.spc")

EXT_CL = (".dg_01", ".dg_01.lab")
EXT_TMP = (".mag", ".mst11.edges", ".param", "_tmp_data", "_cluster.run")

# Map (system, machine) -> bundled binary name
_BINARY_MAP = {
    ("Linux", "x86_64"): "cluster_linux64.exe",
    ("Linux", "amd64"): "cluster_linux64.exe",
    ("Windows", "AMD64"): "cluster_64.exe",
    ("Windows", "x86_64"): "cluster_64.exe",
    ("Darwin", "x86_64"): "cluster_mac_new.exe",
    ("Darwin", "arm64"): "cluster_mac_new.exe",  # runs under Rosetta 2
}


def _rosetta_available() -> bool:
    """Return True if Rosetta 2 appears available on Apple Silicon."""
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        return True
    # arch -x86_64 true succeeds when Rosetta is installed
    try:
        r = subprocess.run(
            ["arch", "-x86_64", "true"],
            capture_output=True,
            timeout=5,
        )
        return r.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def resolve_spc_binary(explicit: Optional[str] = None) -> Path:
    """
    Locate the SPC clustering binary for this platform.

    Raises RuntimeError with an actionable message if unavailable.
    """
    if explicit:
        p = Path(explicit)
        if not p.is_file():
            raise RuntimeError(f"SPC binary not found: {explicit}")
        return p

    key = (platform.system(), platform.machine())
    name = _BINARY_MAP.get(key)
    if name is None:
        raise RuntimeError(
            f"No SPC binary mapped for platform {key}. "
            "Supported: Linux x86_64, Windows AMD64, macOS x86_64/arm64."
        )

    if platform.system() == "Darwin" and platform.machine() == "arm64":
        if not _rosetta_available():
            raise RuntimeError(
                "Apple Silicon detected, but Rosetta 2 is not available.\n"
                "The bundled SPC binary is x86_64 and needs Rosetta 2.\n"
                "Install it with:\n"
                "  softwareupdate --install-rosetta\n"
                "Then re-run combinato."
            )
        logger.warning(
            "Using x86_64 SPC binary via Rosetta 2 on Apple Silicon. "
            "Install Rosetta if clustering fails: softwareupdate --install-rosetta"
        )

    # Prefer package data
    try:
        pkg = resources.files("combinato_lite.cluster.spc_bin")
        candidate = pkg.joinpath(name)
        # resources may return a Traversable; materialize if needed
        if hasattr(candidate, "is_file") and candidate.is_file():
            path = Path(str(candidate))
            # On some installs this is inside a zip; copy to a cache dir
            if not os.access(path, os.X_OK) or not path.exists():
                path = _materialize_binary(candidate, name)
            else:
                _ensure_executable(path)
            return path
    except Exception as exc:
        logger.debug("Package resource lookup failed: %s", exc)

    # Fallback: adjacent to this file
    local = Path(__file__).resolve().parent / "spc_bin" / name
    if local.is_file():
        _ensure_executable(local)
        return local

    raise RuntimeError(f"SPC binary {name} not found in package data")


def _materialize_binary(traversable, name: str) -> Path:
    cache = Path.home() / ".cache" / "combinato-lite" / "spc_bin"
    cache.mkdir(parents=True, exist_ok=True)
    dest = cache / name
    if not dest.exists():
        with resources.as_file(traversable) as src:
            shutil.copy2(src, dest)
        _ensure_executable(dest)
    return dest


def _ensure_executable(path: Path) -> None:
    if sys.platform == "win32":
        return
    mode = path.stat().st_mode
    path.chmod(mode | 0o111)


def _cleanup(base: str, ext: tuple[str, ...]) -> None:
    for this_ext in ext:
        name = base + this_ext
        if os.path.exists(name):
            os.remove(name)


class Clusterer(ABC):
    """Abstract clustering backend."""

    @abstractmethod
    def cluster(
        self,
        features: np.ndarray,
        folder: str,
        name: str,
        random_seed: Optional[float] = None,
    ) -> None:
        """Run clustering; write results readable by ``read_results``."""

    @abstractmethod
    def read_results(self, folder: str, name: str) -> tuple[np.ndarray, np.ndarray]:
        """Return (clu, tree) arrays."""


class SPCBinaryBackend(Clusterer):
    """Wraps the classic wave_clus / Combinato SPC text-file protocol."""

    def __init__(self, binary: Optional[str | Path] = None, cfg=None):
        self.cfg = cfg or get_config()
        self.binary = Path(binary) if binary else resolve_spc_binary()
        logger.info("SPC binary: %s", self.binary)

    def cluster(
        self,
        features: np.ndarray,
        folder: str,
        name: str,
        random_seed: Optional[float] = None,
    ) -> None:
        if not os.path.isdir(folder):
            os.mkdir(folder)

        cleanname = os.path.join(folder, name)
        _cleanup(cleanname, EXT_CL)

        data_fname = name + "_tmp_data"
        datasavename = os.path.join(folder, data_fname)
        np.savetxt(datasavename, features, newline="\n", fmt="%f")

        argument_fname = name + "_cluster.run"
        run_fname = os.path.join(folder, argument_fname)
        seed = random_seed if random_seed is not None else np.random.random() * 2**32

        with open(run_fname, "w", encoding="utf-8") as fid:
            fid.write("NumberOfPoints: %i\n" % features.shape[0])
            fid.write("DataFile: %s\n" % data_fname)
            fid.write("OutFile: %s\n" % name)
            fid.write("Dimensions: %s\n" % features.shape[1])
            fid.write("MinTemp: 0\n")
            fid.write("MaxTemp: 0.201\n")
            fid.write("TempStep: %f\n" % self.cfg.TempStep)
            fid.write("SWCycles: 100\n")
            fid.write("KNearestNeighbours: 11\n")
            fid.write("MSTree|\n")
            fid.write("DirectedGrowth|\n")
            fid.write("SaveSuscept|\n")
            fid.write("WriteLables|\n")
            fid.write("WriteCorFile~\n")
            fid.write("ForceRandomSeed: %f\n" % seed)

        out = None if self.cfg.ShowSPCOutput else subprocess.PIPE

        # On Apple Silicon, force x86_64 via arch if needed
        cmd: list[str]
        if (
            platform.system() == "Darwin"
            and platform.machine() == "arm64"
            and shutil.which("arch")
        ):
            cmd = ["arch", "-x86_64", str(self.binary), argument_fname]
        else:
            cmd = [str(self.binary), argument_fname]

        ret = subprocess.call(cmd, stdout=out, stderr=out, cwd=folder)
        if ret:
            raise RuntimeError(f"Error in SPC clustering: {name} (exit {ret})")

        _cleanup(cleanname, EXT_TMP)

    def read_results(self, folder: str, name: str) -> tuple[np.ndarray, np.ndarray]:
        tree_fname = os.path.join(folder, name + ".dg_01")
        clu_fname = os.path.join(folder, name + ".dg_01.lab")
        tree = np.loadtxt(tree_fname)
        clu = np.loadtxt(clu_fname)
        return clu, tree


def get_clusterer(backend: str = "spc", binary: Optional[str] = None, cfg=None) -> Clusterer:
    if backend == "spc":
        return SPCBinaryBackend(binary=binary, cfg=cfg)
    raise ValueError(f"Unknown clustering backend: {backend}")
