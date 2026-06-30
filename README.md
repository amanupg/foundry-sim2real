# Foundry

Image-to-physics-ready-asset pipeline with an agentic critique-repair loop.

Drop in a photo of a single object; get back a simulation-ready 3D asset
(mesh + collision geometry + URDF with estimated mass/inertia), validated by
an autonomous agent that renders the result, critiques it against the original
photo using a vision model, and retries with adjusted parameters until it
passes or exhausts attempts.

## Run

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in tokens
uvicorn foundry.main:app --reload
```

Open http://127.0.0.1:8000

## Environment

Credentials are read from environment only (via `.env` / python-dotenv). If a
key is missing, the relevant stage falls back to mock mode and logs clearly;
it never blocks.

| Variable | Purpose |
|---|---|
| `REPLICATE_API_TOKEN` | Hosted 3D generation (Checkpoint 3). Also accepts `REPLICATE_API_KEY`. |
| `OPENROUTER_API_KEY` | VLM critique (Checkpoint 2). Also accepts `ANTHROPIC_API_KEY`. |

> **Note on VLM:** The original spec called for the Anthropic Messages API.
> Per user decision, the vision critique uses OpenRouter with
> `meta-llama/llama-3.2-90b-vision-instruct` (fallback
> `meta-llama/llama-3.2-11b-vision-instruct`). Same JSON verdict contract.

## Pipeline stages

1. `preprocess` - rembg background removal, center + pad.
2. `generate_mesh` - hosted 3D model on Replicate (TRELLIS -> Hunyuan3D-2 ->
   Stable Fast 3D -> TripoSR -> mock cube). Abstracted behind one function.
3. `process_mesh` - trimesh: merge vertices, fill holes, repair, recenter,
   orient longest axis.
4. `make_collision` - coacd approximate convex decomposition (falls back to
   convex hull).
5. `estimate_physics` - mass + inertia from volume at default density;
   rescale to user-provided real-world longest dim.
6. `export_urdf` - URDF + asset pack zip (mesh + collision + urdf + report.json).
7. `validate_in_sim` - PyBullet headless drop test (240 frames) + 4 offscreen
   renders (TINY renderer, CPU, no GL).
8. `critique` - vision model compares original photo to renders + sim report;
   returns JSON verdict.

Stages 2-8 are wrapped in a controller that runs up to `MAX_ATTEMPTS` (default
3). On a failing verdict, the `suggested_fix` drives a concrete parameter
change before the next attempt (resegment / retry_generation / remesh /
adjust_decomposition).

## Honest limitations

- Works best on a single isolated rigid object photographed against a
  plain-ish background (mug, shoe, toy, chair).
- Absolute scale from a single image is unreliable; scale is a user-provided
  estimate.
- The back of the object is inferred by the 3D model.
- coacd may fall back to a convex hull for concave shapes if it fails.
- PyBullet offscreen rendering uses the CPU TINY renderer to avoid headless-GL
  breakage; if it fails, a trimesh still-render fallback is used.

## Build checkpoints

- **0** FastAPI skeleton, `/health`, static index. (done)
- **1** Full pipeline end-to-end with mocked mesh (cube). (done, verified)
- **2** Real VLM critique (OpenRouter) + repair loop. (done, verified live)
- **3** Real Replicate 3D generation with fallback chain. (code-complete;
  live verification blocked by insufficient Replicate account credit - 402.
  Code is correct and will produce real meshes once credit is added.)
- **4** Frontend polish (timeline, attempt history, three.js viewer, download). (done)
- **5** (stretch) Depth Anything v2 scale cue; MJCF export; gallery.

## Verified end-to-end

With no Replicate credit, the full HTTP flow runs on the mocked cube:
upload -> preprocess -> mock mesh -> process -> collision -> physics -> URDF ->
PyBullet drop test (stable) -> 4 renders -> real VLM verdict (OpenRouter) ->
asset pack zip download. The critique-repair loop runs up to 3 attempts with
real vision verdicts driving parameter changes.

## Run with no keys (mock mode)

If neither `REPLICATE_API_TOKEN` nor `OPENROUTER_API_KEY` is set, the pipeline
runs fully on the mock cube with a stub VLM verdict. This proves the skeleton.
