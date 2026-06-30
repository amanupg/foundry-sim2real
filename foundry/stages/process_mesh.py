"""Stage 3: process_mesh(raw_mesh_path) -> processed_mesh_path, report.

Load in trimesh; merge vertices; fill holes; check watertightness; attempt
repair; recenter to origin; normalize orientation (longest axis heuristics);
return cleaned mesh plus a report dict.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import trimesh

log = logging.getLogger("foundry.process_mesh")


def process_mesh(raw_mesh_path: Path, run_dir: Path, repair_aggressive: bool = False) -> tuple[Path, dict]:
    log.info("process_mesh start (aggressive=%s) %s", repair_aggressive, raw_mesh_path.name)
    mesh = trimesh.load(raw_mesh_path, force="mesh")
    if mesh is None or len(mesh.faces) == 0:
        log.warning("process_mesh: empty mesh, substituting cube")
        mesh = trimesh.creation.box(extents=[0.15, 0.15, 0.15])

    # Merge vertices
    mesh.merge_vertices()
    # Fill holes
    try:
        mesh.fill_holes()
    except Exception as e:
        log.warning("process_mesh: fill_holes failed (%s)", e)

    watertight_before = bool(mesh.is_watertight)
    if not watertight_before:
        try:
            if repair_aggressive:
                mesh.process(validate=True)
                try:
                    trimesh.repair.fill_holes(mesh)
                except Exception:
                    pass
            else:
                trimesh.repair.fill_holes(mesh)
        except Exception as e:
            log.warning("process_mesh: repair failed (%s)", e)

    watertight_after = bool(mesh.is_watertight)

    # Recenter to origin
    mesh.apply_translation(-mesh.centroid)

    # Normalize orientation: rotate so longest axis aligns to X, next to Y.
    try:
        ext = mesh.bounding_box.extents
        order = np.argsort(ext)[::-1]  # longest first
        if list(order) != [0, 1, 2]:
            # Build rotation that maps current axes to longest-first order.
            R = np.eye(3)[order]
            mesh.apply_rotation(R)
    except Exception as e:
        log.warning("process_mesh: orientation normalize failed (%s)", e)

    out = run_dir / "processed.glb"
    mesh.export(out)

    report = {
        "vertex_count": int(len(mesh.vertices)),
        "face_count": int(len(mesh.faces)),
        "watertight": watertight_after,
        "watertight_before_repair": watertight_before,
        "bbox_dims": [float(x) for x in mesh.bounding_box.extents],
        "volume": float(mesh.volume) if watertight_after else None,
    }
    log.info("process_mesh end -> %s (verts=%d, watertight=%s)",
             out.name, report["vertex_count"], report["watertight"])
    return out, report
