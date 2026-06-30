"""Central configuration and environment loading for Foundry.

Reads credentials from environment only. If a credential is missing, the
relevant stage falls back to mock mode and logs clearly. Never blocks.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from repo root if present (never committed; gitignored).
ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


def _get(*names: str) -> str | None:
    """Return the first non-empty env var from names, else None."""
    for n in names:
        v = os.environ.get(n, "").strip()
        if v:
            return v
    return None


# --- Credentials ------------------------------------------------------------
REPLICATE_API_TOKEN: str | None = _get("REPLICATE_API_TOKEN", "REPLICATE_API_KEY")
OPENROUTER_API_KEY: str | None = _get("OPENROUTER_API_KEY", "ANTHROPIC_API_KEY")

# --- VLM (OpenRouter) -------------------------------------------------------
# Deviation from original spec (Anthropic): using OpenRouter per user request.
VLM_MODEL: str = _get("VLM_MODEL") or "meta-llama/llama-3.2-11b-vision-instruct"
VLM_FALLBACK_MODEL: str = _get("VLM_FALLBACK_MODEL") or "google/gemma-3-12b-it"
OPENROUTER_BASE: str = "https://openrouter.ai/api/v1/chat/completions"

# --- Pipeline tunables ------------------------------------------------------
MAX_ATTEMPTS: int = int(os.environ.get("FOUNDRY_MAX_ATTEMPTS", "3"))
DEFAULT_DENSITY_KG_M3: float = float(os.environ.get("FOUNDRY_DENSITY", "500.0"))
DEFAULT_REAL_DIM_M: float = float(os.environ.get("FOUNDRY_REAL_DIM_M", "0.2"))
PYBULLET_FRAMES: int = int(os.environ.get("FOUNDRY_PB_FRAMES", "240"))

# --- Paths ------------------------------------------------------------------
RUNS_DIR: Path = ROOT / "runs"
SAMPLES_DIR: Path = ROOT / "samples"
STATIC_DIR: Path = ROOT / "static"


def ensure_dirs() -> None:
    for d in (RUNS_DIR, SAMPLES_DIR, STATIC_DIR):
        d.mkdir(parents=True, exist_ok=True)


def mode_status() -> dict:
    """Surface which stages are live vs mock, for UI banners."""
    return {
        "replicate_available": bool(REPLICATE_API_TOKEN),
        "vlm_available": bool(OPENROUTER_API_KEY),
        "vlm_model": VLM_MODEL,
        "max_attempts": MAX_ATTEMPTS,
    }
