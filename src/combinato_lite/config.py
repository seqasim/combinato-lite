"""Dataclass-based configuration (replaces generated options.py)."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Optional

import yaml

# Artifact scoring thresholds (ported from Combinato artifact_criteria)
DEFAULT_ARTIFACT_CRITERIA: dict[str, float] = {
    "maxima": 5,
    "maxima_1_2_ratio": 2,
    "max_min_ratio": 1.5,
    "sem": 4,
    "ptp": 1,
}


@dataclass
class CombinatoConfig:
    """All tunable Combinato parameters with sensible defaults."""

    # Clustering
    MaxClustersPerTemp: int = 5
    MinSpikesPerClusterMultiSelect: int = 15
    RecursiveDepth: int = 1
    ReclusterClusters: bool = True
    MinInputSizeRecluster: int = 2000
    FirstMatchFactor: float = 0.75
    SecondMatchFactor: float = 3.0
    MaxDistMatchGrouping: float = 1.8
    MinInputSize: int = 15
    TempStep: float = 0.01
    MarkArtifactClasses: bool = True
    FractionOfBiggestCluster: float = 0.05
    nFeatures: int = 10
    Wavelet: str = "haar"
    ShowSPCOutput: bool = False
    RecheckArtifacts: bool = True
    ExcludeVariableClustersMatch: bool = True
    FirstMatchMaxDist: float = 4.0
    SecondMatchMaxDist: float = 20.0
    OverwriteGroups: bool = True
    overwrite: bool = True
    feature_factor: float = 3.0
    Debug: bool = False

    # Extraction
    threshold_factor: float = 5.0
    max_spike_duration: float = 0.0015
    indices_per_spike: int = 64
    index_maximum: int = 19
    upsampling_factor: int = 3
    denoise: bool = True
    do_filter: bool = True
    max_nspk_session: int = 20000

    # Plotting (optional; used by combine)
    plot: bool = True
    plotTemps: bool = False
    figsize: tuple[float, float] = (2.0, 2.0)
    dpi: int = 100
    ylim: tuple[float, float] = (-200.0, 200.0)
    linewidth: float = 0.4

    # Logging
    LogLevel: str = "INFO"
    LogToConsole: bool = True
    LogToFile: bool = False
    LogDir: str = ""
    LogFormat: str = "%(asctime)s [%(levelname)-7s] %(name)s: %(message)s"
    LogDateFormat: str = "%Y-%m-%d %H:%M:%S"

    # Artifact criteria
    artifact_criteria: dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_ARTIFACT_CRITERIA)
    )

    def update(self, updates: dict[str, Any]) -> CombinatoConfig:
        """Return a copy with selected fields updated."""
        data = asdict(self)
        for key, value in updates.items():
            if key == "artifact_criteria" and isinstance(value, dict):
                data["artifact_criteria"] = {**data["artifact_criteria"], **value}
            elif key in data:
                data[key] = value
        return CombinatoConfig(**data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_CONFIG: Optional[CombinatoConfig] = None


def _parse_env_overrides() -> dict[str, Any]:
    """Read COMBINATO_* environment variables into a dict of overrides."""
    known = {f.name for f in fields(CombinatoConfig)}
    out: dict[str, Any] = {}
    for key, value in os.environ.items():
        if not key.startswith("COMBINATO_"):
            continue
        name = key[len("COMBINATO_") :]
        if name not in known:
            continue
        # Best-effort type coercion via YAML
        try:
            out[name] = yaml.safe_load(value)
        except Exception:
            out[name] = value
    return out


def load_config(
    path: Optional[str | Path] = None,
    overrides: Optional[dict[str, Any]] = None,
) -> CombinatoConfig:
    """
    Load config from defaults, optional YAML file, env vars, then explicit overrides.

    Search order for YAML (first found wins if path is None):
      1. $COMBINATO_CONFIG
      2. ./combinato_lite.yaml
      3. ./combinato.yaml
    """
    cfg = CombinatoConfig()

    yaml_path: Optional[Path] = None
    if path is not None:
        yaml_path = Path(path)
    else:
        env_path = os.environ.get("COMBINATO_CONFIG")
        if env_path and Path(env_path).is_file():
            yaml_path = Path(env_path)
        else:
            for candidate in ("combinato_lite.yaml", "combinato.yaml"):
                if Path(candidate).is_file():
                    yaml_path = Path(candidate)
                    break

    if yaml_path is not None and yaml_path.is_file():
        with open(yaml_path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        if not isinstance(data, dict):
            raise ValueError(f"Config file {yaml_path} must contain a mapping")
        cfg = cfg.update(data)

    env_overrides = _parse_env_overrides()
    if env_overrides:
        cfg = cfg.update(env_overrides)

    if overrides:
        cfg = cfg.update(overrides)

    global _CONFIG
    _CONFIG = cfg
    return cfg


def get_config() -> CombinatoConfig:
    """Return the active config, loading defaults if needed."""
    global _CONFIG
    if _CONFIG is None:
        _CONFIG = load_config()
    return _CONFIG


def set_config(cfg: CombinatoConfig) -> None:
    """Replace the active config (mainly for tests)."""
    global _CONFIG
    _CONFIG = cfg
