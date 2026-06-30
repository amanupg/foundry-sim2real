# AGENTS.md

Repo-specific guidance for OpenCode sessions working on Foundry.

## What this is

Image-to-physics-ready-asset pipeline. Upload a photo of a single object ->
get a simulation-ready 3D asset (mesh + collision + URDF with mass/inertia),
validated by an agentic critique-repair loop (vision model compares renders to
the original photo, retries with adjusted params up to 3 attempts).

## Commands

```bash
# venv is Python 3.11 (system python is 3.14 - wheels for coacd/rembg/pybullet
# are unreliable on 3.14; always use the .venv)
source .venv/bin/activate
uvicorn foundry.main:app --reload          # http://127.0.0.1:8000

# run the pipeline headlessly (no server)
python -c "from pathlib import Path; from foundry.controller import run_pipeline; print(run_pipeline(Path('samples/sample_mug.png')))"

# tests: there is no test suite. Verify by running the pipeline on samples/sample_mug.png
# and checking that runs/<id>/asset_pack.zip exists and contains object.urdf + meshes/.
```

## Architecture

Single FastAPI service (`foundry/main.py`) serving a static single-page frontend
(`static/index.html`, three.js via CDN). No Node build step.

- `foundry/stages/` - each stage is a pure function with a clear contract:
  preprocess -> generate_mesh -> process_mesh -> estimate_physics ->
  make_collision -> export_urdf -> validate_in_sim -> critique.
- `foundry/controller.py` - the critique-repair loop wrapping stages 2-8.
  Runs up to MAX_ATTEMPTS (default 3). On a failing verdict, applies a concrete
  parameter change based on `suggested_fix` before the next attempt.
- `foundry/config.py` - all env loading; `mode_status()` tells UI which stages
  are live vs mock.
- Per-run artifacts go to `runs/<timestamp-id>/` (gitignored).

## Credentials and mock mode

Read from env only (`.env` via python-dotenv, gitignored). If a key is missing,
that stage falls back to mock mode and logs clearly; never blocks.

- `REPLICATE_API_TOKEN` (also accepts `REPLICATE_API_KEY`) - hosted 3D generation.
- `OPENROUTER_API_KEY` (also accepts `ANTHROPIC_API_KEY`) - VLM critique.

**Deviation from original spec:** the vision critique uses OpenRouter
(`meta-llama/llama-3.2-11b-vision-instruct`, fallback `google/gemma-3-12b-it`)
instead of the Anthropic Messages API. The 90b llama vision model does not exist
on OpenRouter; only the 11b variant is available. Same JSON verdict contract.

## Hard-won gotchas (these took debugging)

- **PyBullet cannot load `.glb` meshes.** URDFs must reference `.obj` (or `.stl`).
  `export_urdf` writes both `visual.glb` (for the three.js viewer) and
  `visual.obj`/`collision.obj` (for pybullet). If you see "invalid mesh filename
  extension '.glb'" or "Cannot load URDF file", this is why.
- **`pybullet.setAdditionalSearchPath` replaces, it does not append.** Load
  `plane.urdf` by absolute path from `pybullet_data.getDataPath()`, do not rely
  on a search path set for the object URDF.
- **PyBullet offscreen render:** use `ER_TINY_RENDERER` (CPU, no GL context).
  Pass `physicsClientId=` to every camera call (`computeViewMatrixFromYawPitchRoll`,
  `computeProjectionMatrixFOV`, `getCameraImage`) or it silently uses the wrong
  client and the fallback renderer kicks in. `getProjectionMatrix` does not exist;
  use `computeProjectionMatrixFOV`.
- **VLM models accept only 1 image.** `critique.py` stitches the original photo +
  4 renders into a single composite PNG before sending.
- **VLM JSON parsing:** strip code fences, regex-extract `{.*}`, and if still
  unparseable treat as a soft-fail verdict (do not crash the loop).
- **Replicate 1.x client API:** use `client.predictions.create(version=<id>,
  input=...)` then `.wait()`. `version.predict()` and `client.run(slug, ...)`
  both 404. Hunyuan3D-2 returns `{"mesh": "<url>"}` (a dict), not a direct URL.
- **Replicate rate limits:** free tier is 6 predictions/min with burst 1.
  `generate_mesh` retries 429s with 15s backoff. A 402 (insufficient credit)
  bails to mock mode for the whole run.
- **trimesh:** `apply_transform` takes a 4x4 matrix; `apply_rotation` does not
  exist. `mesh.moment_inertia` may be None or contain NaNs for non-watertight
  meshes - fall back to `trimesh.inertia.box_inertia`.
- **macOS 26 SDK broke pybullet's bundled zlib** (`#define fdopen` clash).
  pybullet must be built from a patched source tree. The wheel in this repo's
  venv was built once; if `.venv` is recreated, see the patch in
  `examples/ThirdPartyLibs/zlib/zutil.h` (remove the `fdopen` macro).

## Conventions

- No emojis anywhere (code, comments, logs, UI).
- Every stage logs start/end and writes intermediate artifacts to the per-run folder.
- Fallback rules are explicit and non-blocking: coacd fails -> convex hull;
  pybullet render fails -> PIL point-cloud projection; VLM fails -> soft-fail
  verdict; any stage exception is caught and degrades gracefully.
- Commit at each checkpoint. Build order is strict: mock pipeline must fully run
  before integrating any real API.
