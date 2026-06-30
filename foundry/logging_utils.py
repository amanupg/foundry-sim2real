"""Per-run logging and artifact layout.

Every stage logs start/end and writes intermediate artifacts to a per-run folder.
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

from .config import RUNS_DIR


def make_run_dir(name: str | None = None) -> Path:
    import uuid
    ts = time.strftime("%Y%m%d-%H%M%S")
    rid = name or uuid.uuid4().hex[:8]
    d = RUNS_DIR / f"{ts}-{rid}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "renders").mkdir(exist_ok=True)
    return d


def get_logger(run_dir: Path) -> logging.Logger:
    """Return the run-specific logger AND attach its file handler to the
    parent `foundry` logger so that stage loggers (foundry.critique,
    foundry.generate_mesh, etc.) are captured in this run's run.log."""
    logger = logging.getLogger(f"foundry.{run_dir.name}")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s", "%H:%M:%S")
    fh = logging.FileHandler(run_dir / "run.log", encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)

    # Route stage loggers (foundry.preprocess, foundry.critique, ...) into
    # this run's file + stdout by attaching the same handlers to the parent.
    parent = logging.getLogger("foundry")
    parent.setLevel(logging.INFO)
    # Avoid duplicate handlers on the parent across runs.
    parent.handlers = [fh, sh]
    parent.propagate = False
    return logger


def write_json(run_dir: Path, name: str, data: object) -> Path:
    p = run_dir / name
    p.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    return p


def read_json(run_dir: Path, name: str) -> object:
    return json.loads((run_dir / name).read_text(encoding="utf-8"))
