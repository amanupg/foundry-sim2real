"""Stage 2: generate_mesh(clean_image_path) -> raw_mesh_path.

Calls hosted 3D model on Replicate. ABSTRACTED behind one function so the
specific model is swappable. Fallback chain: TRELLIS -> Hunyuan3D-2 ->
Stable Fast 3D -> TripoSR -> mock.

Model slugs change over time, so we verify availability programmatically
and fall back down the list. If all fail, mock mode (trimesh cube).
"""
from __future__ import annotations

import logging
import shutil
from pathlib import Path

import trimesh

from ..config import REPLICATE_API_TOKEN

log = logging.getLogger("foundry.generate_mesh")

# Candidate Replicate model slugs, in priority order. Verified at runtime.
# tencent/hunyuan3d-2 confirmed live (version b1b9449a1277) with image input.
CANDIDATE_MODELS: list[str] = [
    "tencent/hunyuan3d-2",
    "camenduru/hunyuan3d-2",
    "cjwbw/hunyuan3d-2",
    "stability-ai/stable-fast-3d",
    "camenduru/stable-fast-3d",
    "camenduru/triposr",
]

# Per-model input parameter overrides (verified via Replicate API schemas).
MODEL_INPUTS: dict[str, dict] = {
    "tencent/hunyuan3d-2": {
        "steps": 50,
        "guidance_scale": 5.5,
        "octree_resolution": 256,
        "remove_background": False,  # we already preprocess
    },
    "stability-ai/stable-fast-3d": {
        "remove_background": False,
    },
}

# Which candidate index to try next on retry_generation fix.
_next_model_idx = 0


def _mock_mesh(run_dir: Path) -> Path:
    """Return a trimesh cube as a mock mesh (glb)."""
    log.warning("generate_mesh: using MOCK mesh (trimesh cube)")
    mesh = trimesh.creation.box(extents=[0.15, 0.15, 0.15])
    out = run_dir / "raw_mesh.glb"
    mesh.export(out)
    return out


def _verify_and_run(clean_image_path: Path, run_dir: Path, start_idx: int) -> Path | None:
    """Try candidates from start_idx, verify each via Replicate API, run if available."""
    if not REPLICATE_API_TOKEN:
        log.warning("generate_mesh: no REPLICATE_API_TOKEN; mock mode")
        return None
    try:
        import replicate
        client = replicate.Client(api_token=REPLICATE_API_TOKEN)
    except Exception as e:
        log.warning("generate_mesh: replicate client init failed (%s); mock mode", e)
        return None

    for idx in range(start_idx, len(CANDIDATE_MODELS)):
        slug = CANDIDATE_MODELS[idx]
        try:
            # Verify the model exists and get its version id.
            model = client.models.get(slug)
            version = model.latest_version
            if version is None:
                log.info("generate_mesh: %s has no latest_version; skipping", slug)
                continue
            log.info("generate_mesh: attempting %s (version %s)", slug, version.id[:12])

            input_params = dict(MODEL_INPUTS.get(slug, {}))
            with open(clean_image_path, "rb") as fh:
                input_params["image"] = fh
                # Use predictions.create with explicit version id (replicate 1.x).
                # Retry on 429 rate limit with backoff.
                pred = None
                for attempt in range(3):
                    try:
                        pred = client.predictions.create(
                            version=version.id, input=input_params
                        )
                        break
                    except Exception as e:
                        msg = str(e)
                        if "429" in msg or "throttled" in msg:
                            wait = 15 * (attempt + 1)
                            log.warning("generate_mesh: rate limited; waiting %ds", wait)
                            import time as _t
                            _t.sleep(wait)
                            continue
                        raise
                if pred is None:
                    log.warning("generate_mesh: %s rate-limit retries exhausted", slug)
                    continue
                pred.wait()
                if pred.status != "succeeded":
                    log.warning("generate_mesh: %s prediction %s: %s",
                                slug, pred.status, str(pred.error)[:120])
                    continue
                output = pred.output

            path = _download_output(output, run_dir)
            if path:
                log.info("generate_mesh: success with %s -> %s", slug, path.name)
                global _next_model_idx
                _next_model_idx = idx + 1
                return path
        except Exception as e:
            msg = str(e)
            if "402" in msg or "insufficient credit" in msg.lower():
                log.warning("generate_mesh: insufficient Replicate credit; mock mode for all candidates")
                return None
            log.warning("generate_mesh: %s failed (%s); trying next", slug, _short(msg))
            continue
    return None


def _short(s: str, n: int = 120) -> str:
    return s if len(s) <= n else s[:n] + "..."


def _download_output(output, run_dir: Path) -> Path | None:
    """Replicate output may be a URL, a FileOutput, or a list. Normalize to a local .glb."""
    try:
        # Handle FileOutput objects (replicate >= 0.4)
        if hasattr(output, "read"):
            data = output.read()
            out = run_dir / "raw_mesh.glb"
            out.write_bytes(data)
            return out
        # Handle URL string
        if isinstance(output, str) and output.startswith("http"):
            import httpx
            r = httpx.get(output, timeout=120.0, follow_redirects=True)
            r.raise_for_status()
            out = run_dir / "raw_mesh.glb"
            out.write_bytes(r.content)
            return out
        # Handle list of outputs
        if isinstance(output, list) and output:
            return _download_output(output[0], run_dir)
        # Handle dict with url (Hunyuan3D-2 returns {"mesh": "<url>"})
        if isinstance(output, dict):
            for k in ("mesh", "url", "output"):
                if k in output:
                    return _download_output(output[k], run_dir)
        log.warning("generate_mesh: unrecognized output type %s", type(output).__name__)
    except Exception as e:
        log.warning("generate_mesh: download failed (%s)", e)
    return None


def generate_mesh(clean_image_path: Path, run_dir: Path, force_mock: bool = False,
                  model_idx: int = 0) -> Path:
    """Return local path to raw mesh (.glb). Mock cube if no key or all candidates fail."""
    log.info("generate_mesh start (force_mock=%s, model_idx=%d)", force_mock, model_idx)
    if force_mock:
        return _mock_mesh(run_dir)
    path = _verify_and_run(clean_image_path, run_dir, model_idx)
    if path is None:
        return _mock_mesh(run_dir)
    # Normalize to .glb in run_dir
    if path.suffix.lower() == ".obj":
        m = trimesh.load(path)
        out = run_dir / "raw_mesh.glb"
        m.export(out)
        return out
    return path


def reset_model_chain() -> None:
    """Reset the fallback chain pointer (called at start of each run)."""
    global _next_model_idx
    _next_model_idx = 0
