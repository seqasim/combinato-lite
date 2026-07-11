"""Combinato-Lite: lightweight, portable Combinato spike sorting."""

from .constants import (
    CLID_UNMATCHED,
    GROUP_ART,
    GROUP_NOCLASS,
    SIGNS,
    SPIKE_CLUST,
    SPIKE_MATCHED,
    SPIKE_MATCHED_2,
    TYPE_ALL,
    TYPE_ART,
    TYPE_MU,
    TYPE_NAMES,
    TYPE_NO,
    TYPE_NON_NOISE,
    TYPE_SU,
)
from .config import CombinatoConfig, get_config, load_config

__version__ = "0.1.0"
__all__ = [
    "CombinatoConfig",
    "get_config",
    "load_config",
    "CLID_UNMATCHED",
    "GROUP_ART",
    "GROUP_NOCLASS",
    "SIGNS",
    "SPIKE_CLUST",
    "SPIKE_MATCHED",
    "SPIKE_MATCHED_2",
    "TYPE_ALL",
    "TYPE_ART",
    "TYPE_MU",
    "TYPE_NAMES",
    "TYPE_NO",
    "TYPE_NON_NOISE",
    "TYPE_SU",
    "__version__",
]
