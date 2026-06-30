"""Stage 8: critique(original_clean_image_path, render_paths, sim_report) -> verdict.

Send the original object image plus rendered views to a vision model via
OpenRouter. Strict instruction to return ONLY JSON. Parse robustly: strip
code fences, regex-extract the JSON object, and if still unparseable, treat
as a soft fail verdict and continue the loop.

Deviation from spec: uses OpenRouter + Llama 3.2 vision instead of Anthropic
Claude (per user request). Same JSON contract.
"""
from __future__ import annotations

import base64
import json
import logging
import re
from pathlib import Path

import httpx

from ..config import OPENROUTER_API_KEY, OPENROUTER_BASE, VLM_MODEL, VLM_FALLBACK_MODEL

log = logging.getLogger("foundry.critique")

VERDICT_SCHEMA_KEYS = {
    "resembles_input", "geometry_defects", "physics_ok",
    "overall_pass", "suggested_fix", "confidence",
}
VALID_FIXES = {"none", "resegment", "retry_generation", "remesh", "adjust_decomposition"}

SOFT_FAIL_VERDICT = {
    "resembles_input": False,
    "geometry_defects": ["vlm_unparseable_or_unavailable"],
    "physics_ok": False,
    "overall_pass": False,
    "suggested_fix": "remesh",
    "confidence": 0.0,
    "_soft_fail": True,
}

SYSTEM_PROMPT = (
    "You are a 3D asset QA critic. You are given an original photo of an object "
    "and several rendered views of a reconstructed 3D mesh that underwent a "
    "physics drop-test. Compare the renders to the original photo and the "
    "physics report. Respond with ONLY a single JSON object, no prose, no code "
    "fences, matching exactly this schema:\n"
    '{"resembles_input": bool, "geometry_defects": [str], "physics_ok": bool, '
    '"overall_pass": bool, "suggested_fix": one of '
    '["none","resegment","retry_generation","remesh","adjust_decomposition"], '
    '"confidence": float}\n'
    "Do not include any text before or after the JSON object."
)


def _b64_image(path: Path) -> str:
    data = Path(path).read_bytes()
    return "data:image/png;base64," + base64.b64encode(data).decode("ascii")


def _strip_and_extract(text: str) -> dict | None:
    """Strip code fences, regex-extract the JSON object, parse."""
    if not text:
        return None
    # Remove code fences.
    cleaned = re.sub(r"```(?:json)?", "", text).strip()
    # Try direct parse first.
    try:
        obj = json.loads(cleaned)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    # Regex-extract the first {...} block.
    m = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
    return None


def _validate_verdict(obj: dict) -> dict:
    """Ensure all keys present with correct types; coerce/fix where possible."""
    v = dict(SOFT_FAIL_VERDICT)
    v.pop("_soft_fail", None)
    v["resembles_input"] = bool(obj.get("resembles_input", False))
    gd = obj.get("geometry_defects", [])
    if isinstance(gd, list):
        v["geometry_defects"] = [str(x) for x in gd]
    else:
        v["geometry_defects"] = [str(gd)]
    v["physics_ok"] = bool(obj.get("physics_ok", False))
    v["overall_pass"] = bool(obj.get("overall_pass", False))
    fix = str(obj.get("suggested_fix", "remesh")).lower().strip()
    v["suggested_fix"] = fix if fix in VALID_FIXES else "remesh"
    try:
        v["confidence"] = float(obj.get("confidence", 0.0))
    except Exception:
        v["confidence"] = 0.0
    return v


def _call_openrouter(images_b64: list[str], sim_report: dict) -> str | None:
    """Call OpenRouter chat completions with vision. Returns raw text or None."""
    if not OPENROUTER_API_KEY:
        log.warning("critique: no OPENROUTER_API_KEY; soft-fail verdict")
        return None

    content = [{"type": "text", "text": SYSTEM_PROMPT}]
    content.append({
        "type": "text",
        "text": f"Physics sim report: {json.dumps(sim_report)}"
    })
    for i, img in enumerate(images_b64):
        content.append({
            "type": "image_url",
            "image_url": {"url": img}
        })

    for model in (VLM_MODEL, VLM_FALLBACK_MODEL):
        try:
            log.info("critique: calling OpenRouter model %s", model)
            resp = httpx.post(
                OPENROUTER_BASE,
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": content}],
                    "max_tokens": 600,
                    "temperature": 0.2,
                },
                timeout=120.0,
            )
            resp.raise_for_status()
            data = resp.json()
            text = data["choices"][0]["message"]["content"]
            log.info("critique: %s responded (%d chars)", model, len(text or ""))
            return text
        except Exception as e:
            log.warning("critique: model %s failed (%s)", model, _short(str(e)))
    return None


def _short(s: str, n: int = 160) -> str:
    return s if len(s) <= n else s[:n] + "..."


def critique(original_clean_image_path: Path, render_paths: list[Path],
             sim_report: dict, run_dir: Path) -> dict:
    log.info("critique start")
    # If no renders and no VLM key, soft-fail immediately.
    if not render_paths and not OPENROUTER_API_KEY:
        log.warning("critique: no renders and no VLM key; soft-fail")
        v = dict(SOFT_FAIL_VERDICT)
        v["geometry_defects"] = ["no_renders_and_no_vlm"]
        return v

    images: list[str] = []
    # Always include the original clean image.
    if original_clean_image_path and original_clean_image_path.exists():
        images.append(_b64_image(original_clean_image_path))
    for rp in render_paths:
        if rp and Path(rp).exists():
            images.append(_b64_image(rp))

    if not images:
        v = dict(SOFT_FAIL_VERDICT)
        v["geometry_defects"] = ["no_images_available"]
        return v

    raw = _call_openrouter(images, sim_report)
    if raw is None:
        return dict(SOFT_FAIL_VERDICT)

    obj = _strip_and_extract(raw)
    if obj is None:
        log.warning("critique: could not parse VLM response; soft-fail")
        v = dict(SOFT_FAIL_VERDICT)
        v["geometry_defects"] = ["vlm_unparseable"]
        v["_raw_response"] = raw[:500]
        return v

    verdict = _validate_verdict(obj)
    verdict["_raw_response"] = raw[:500]
    log.info("critique end -> pass=%s, fix=%s, conf=%.2f",
             verdict["overall_pass"], verdict["suggested_fix"], verdict["confidence"])
    return verdict
