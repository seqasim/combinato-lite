"""Logging setup driven by CombinatoConfig."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from .config import CombinatoConfig, get_config


def configure_logging(cfg: CombinatoConfig | None = None) -> None:
    """Configure root logger for combinato-* modules."""
    cfg = cfg or get_config()
    level = getattr(logging, str(cfg.LogLevel).upper(), logging.INFO)
    root = logging.getLogger("combinato")
    root.handlers.clear()
    root.setLevel(level)
    root.propagate = False

    formatter = logging.Formatter(cfg.LogFormat, datefmt=cfg.LogDateFormat)

    if cfg.LogToConsole:
        sh = logging.StreamHandler(sys.stderr)
        sh.setFormatter(formatter)
        sh.setLevel(level)
        root.addHandler(sh)

    if cfg.LogToFile:
        log_dir = Path(cfg.LogDir) if cfg.LogDir else Path.cwd()
        log_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_dir / "combinato.log", mode="a")
        fh.setFormatter(formatter)
        fh.setLevel(level)
        root.addHandler(fh)
