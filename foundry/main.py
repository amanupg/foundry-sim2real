"""Foundry FastAPI app.

Single service: FastAPI backend + static single-page frontend (plain HTML +
three.js via CDN). Start with: uvicorn main:app --reload
"""
from __future__ import annotations

import logging
import shutil
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .config import ROOT, RUNS_DIR, SAMPLES_DIR, STATIC_DIR, ensure_dirs, mode_status
from .controller import run_pipeline
from .logging_utils import read_json

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("foundry.main")

app = FastAPI(title="Foundry", version="0.1.0",
              description="Image-to-physics-ready-asset pipeline with agentic critique-repair loop.")

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

ensure_dirs()

# Serve generated run artifacts (meshes, renders, zips) under /runs.
app.mount("/runs", StaticFiles(directory=str(RUNS_DIR)), name="runs")
# Serve static frontend under /static.
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "version": "0.1.0", "mode": mode_status()}


@app.get("/api/status")
def api_status() -> dict:
    return mode_status()


@app.post("/api/run")
async def api_run(image: UploadFile = File(...),
                  density: float = 500.0,
                  real_world_longest_dim_meters: float = 0.2) -> JSONResponse:
    """Run the full pipeline on an uploaded image."""
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(400, "Upload must be an image.")
    ensure_dirs()
    run_dir = RUNS_DIR / "_uploads"
    run_dir.mkdir(exist_ok=True)
    in_path = run_dir / f"input_{image.filename or 'upload.png'}"
    with in_path.open("wb") as f:
        shutil.copyfileobj(image.file, f)
    log.info("uploaded %s (%d bytes)", in_path.name, in_path.stat().st_size)
    summary = run_pipeline(in_path, density=density,
                           real_world_longest_dim_meters=real_world_longest_dim_meters)
    return JSONResponse(summary)


@app.get("/api/runs")
def list_runs() -> list[str]:
    if not RUNS_DIR.exists():
        return []
    return sorted([d.name for d in RUNS_DIR.iterdir() if d.is_dir() and d.name != "_uploads"],
                  reverse=True)


@app.get("/api/runs/{run_id}")
def get_run(run_id: str) -> JSONResponse:
    d = RUNS_DIR / run_id
    if not d.is_dir():
        raise HTTPException(404, "run not found")
    try:
        return JSONResponse(read_json(d, "summary.json"))  # type: ignore
    except FileNotFoundError:
        raise HTTPException(404, "summary not found")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/samples")
def samples_note() -> dict:
    note = (
        "Works best on a single isolated rigid object photographed against a "
        "plain-ish background (e.g. a mug, shoe, toy, or chair). Scale is a "
        "user-provided estimate; the back of the object is inferred."
    )
    return {"note": note, "good_examples": ["mug", "shoe", "toy", "chair"]}
