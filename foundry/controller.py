"""The agentic critique-repair loop controller.

Wraps stages 2-8 in a controller that runs up to MAX_ATTEMPTS. The loop is
the product: each attempt, its verdict, the rendered views, and the fix
applied are surfaced to the frontend.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

from . import stages
from .config import DEFAULT_DENSITY_KG_M3, DEFAULT_REAL_DIM_M, MAX_ATTEMPTS, ROOT
from .logging_utils import get_logger, make_run_dir, write_json


def _rel(p: Path) -> str:
    """Path relative to repo ROOT, for URL serving under /runs/..."""
    try:
        return str(Path(p).relative_to(ROOT))
    except ValueError:
        return str(p)

log = logging.getLogger("foundry.controller")


def run_pipeline(image_path: Path,
                 density: float = DEFAULT_DENSITY_KG_M3,
                 real_world_longest_dim_meters: float = DEFAULT_REAL_DIM_M) -> dict:
    """Run the full critique-repair loop. Returns a run summary dict."""
    run_dir = make_run_dir()
    logger = get_logger(run_dir)
    logger.info("=== RUN START === input=%s", image_path.name)

    # Stage 1: preprocess (always once first; resegment re-runs it tighter).
    clean_path = stages.preprocess.preprocess(image_path, run_dir, tighter=False)

    # Reset the model fallback chain at the start of each run.
    stages.generate_mesh.reset_model_chain()

    attempts: list[dict] = []
    best: dict | None = None
    best_score = -1.0

    # Tunables that the repair loop adjusts.
    tighter_resegment = False
    repair_aggressive = False
    coacd_threshold = 0.05
    model_idx = 0

    for attempt_num in range(1, MAX_ATTEMPTS + 1):
        logger.info("=== ATTEMPT %d/%d ===", attempt_num, MAX_ATTEMPTS)
        attempt: dict = {"attempt": attempt_num, "fix_applied": None}
        try:
            # Apply fix from previous verdict.
            if attempt_num > 1 and attempts:
                prev_verdict = attempts[-1].get("verdict", {})
                fix = prev_verdict.get("suggested_fix", "remesh")
                attempt["fix_applied"] = fix
                if fix == "resegment":
                    tighter_resegment = True
                    clean_path = stages.preprocess.preprocess(image_path, run_dir, tighter=True)
                elif fix == "retry_generation":
                    model_idx = max(model_idx + 1, stages.generate_mesh._next_model_idx)
                elif fix == "remesh":
                    repair_aggressive = True
                elif fix == "adjust_decomposition":
                    coacd_threshold = max(0.01, coacd_threshold / 2.0)

            # Stage 2: generate mesh (mock until checkpoint 3).
            raw_mesh = stages.generate_mesh.generate_mesh(
                clean_path, run_dir, model_idx=model_idx
            )
            # Stage 3: process mesh.
            processed, mesh_report = stages.process_mesh.process_mesh(
                raw_mesh, run_dir, repair_aggressive=repair_aggressive
            )
            # Stage 5: estimate physics (rescales processed mesh in place).
            mass, inertia, scale_note, processed = stages.estimate_physics.estimate_physics(
                processed, run_dir, density=density,
                real_world_longest_dim_meters=real_world_longest_dim_meters
            )
            # Stage 4: collision (after rescale so hulls match scaled mesh).
            collision = stages.make_collision.make_collision(
                processed, run_dir, threshold=coacd_threshold
            )
            # Stage 6: export URDF + asset pack.
            urdf_path, zip_path = stages.export_urdf.export_urdf(
                processed, collision, run_dir, mass, inertia, scale_note, mesh_report
            )
            # Stage 7: validate in sim.
            sim_report, renders = stages.validate_in_sim.validate_in_sim(
                urdf_path, processed, run_dir
            )
            # Stage 8: critique.
            verdict = stages.critique.critique(
                clean_path, renders, sim_report, run_dir
            )

            attempt.update({
                "raw_mesh": _rel(raw_mesh),
                "processed": _rel(processed),
                "mesh_report": mesh_report,
                "mass": mass,
                "inertia": inertia.tolist(),
                "scale_note": scale_note,
                "sim_report": sim_report,
                "renders": [_rel(r) for r in renders],
                "urdf": _rel(urdf_path),
                "zip": _rel(zip_path),
                "verdict": verdict,
                "reasoning": _reasoning(attempt_num, verdict, attempts),
            })
            attempts.append(attempt)
            write_json(run_dir, "attempts.json", attempts)

            # Score this attempt for best-selection.
            score = _score(verdict, sim_report)
            if score > best_score:
                best_score = score
                best = attempt

            if verdict.get("overall_pass"):
                logger.info("=== PASS on attempt %d ===", attempt_num)
                break

        except Exception as e:
            logger.exception("=== ATTEMPT %d FAILED: %s ===", attempt_num, e)
            attempt["error"] = str(e)
            attempts.append(attempt)
            write_json(run_dir, "attempts.json", attempts)
            # Continue to next attempt rather than aborting.

    fully_passed = bool(best and best.get("verdict", {}).get("overall_pass"))
    summary = {
        "run_id": run_dir.name,
        "run_dir": str(run_dir.relative_to(Path.cwd())) if Path.cwd() in run_dir.parents or run_dir == Path.cwd() else str(run_dir),
        "attempts": attempts,
        "best_attempt": best,
        "fully_passed": fully_passed,
        "total_attempts": len(attempts),
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    write_json(run_dir, "summary.json", summary)
    logger.info("=== RUN END === attempts=%d, passed=%s", len(attempts), fully_passed)
    return summary


def _reasoning(attempt_num: int, verdict: dict, prev: list) -> str:
    """Human-readable loop reasoning for the frontend."""
    if attempt_num == 1:
        return "Initial generation and validation pass."
    fix = prev[-1].get("verdict", {}).get("suggested_fix", "remesh") if prev else "remesh"
    parts = {
        "resegment": "Re-ran background removal with tighter settings.",
        "retry_generation": "Switched to the next 3D generation model candidate.",
        "remesh": "Increased mesh repair aggressiveness / smoothing.",
        "adjust_decomposition": "Refined collision decomposition (finer hulls).",
        "none": "No fix needed.",
    }
    base = parts.get(fix, "Applied a parameter change.")
    v = verdict or {}
    return (
        f"Attempt {attempt_num}: {base} "
        f"Verdict: resembles={v.get('resembles_input')}, "
        f"physics_ok={v.get('physics_ok')}, pass={v.get('overall_pass')}."
    )


def _score(verdict: dict, sim_report: dict) -> float:
    """Score an attempt: higher is better. Used to pick best if not fully passing."""
    score = 0.0
    if verdict.get("overall_pass"):
        score += 2.0
    if verdict.get("resembles_input"):
        score += 1.0
    if verdict.get("physics_ok"):
        score += 1.0
    if sim_report.get("stable"):
        score += 0.5
    score += float(verdict.get("confidence", 0.0)) * 0.5
    return score
