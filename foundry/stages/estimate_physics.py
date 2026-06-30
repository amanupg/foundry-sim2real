"""Stage 5: estimate_physics(processed_mesh) -> mass, inertia, scale_note.

Assume a default density (expose as parameter). Compute mass and inertia
tensor from trimesh volume. Rescale mesh to user-provided real-world longest
dim. Label scale as user-provided estimate.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import trimesh

from ..config import DEFAULT_DENSITY_KG_M3, DEFAULT_REAL_DIM_M

log = logging.getLogger("foundry.estimate_physics")


def estimate_physics(processed_mesh_path: Path, run_dir: Path,
                     density: float = DEFAULT_DENSITY_KG_M3,
                     real_world_longest_dim_meters: float = DEFAULT_REAL_DIM_M
                     ) -> tuple[float, np.ndarray, str, Path]:
    log.info("estimate_physics start (density=%s, dim_m=%s)", density, real_world_longest_dim_meters)
    mesh = trimesh.load(processed_mesh_path, force="mesh")

    # Rescale to real-world longest dimension.
    current_longest = float(max(mesh.bounding_box.extents))
    if current_longest > 0:
        scale = real_world_longest_dim_meters / current_longest
        mesh.apply_scale(scale)
    else:
        scale = 1.0

    # Re-export the rescaled mesh (overwrites processed.glb).
    rescaled_path = run_dir / "processed.glb"
    mesh.export(rescaled_path)

    # Compute mass from volume (requires watertight; fallback to convex hull volume).
    try:
        vol = float(mesh.volume)
    except Exception:
        vol = float(mesh.convex_hull.volume)
    mass = vol * density

    # Inertia tensor about center of mass.
    try:
        inertia = mesh.moment_inertia
        if inertia is None or np.any(np.isnan(inertia)):
            raise ValueError("invalid inertia")
    except Exception:
        # Fallback: inertia of a box with same bounding box dims and mass.
        ext = mesh.bounding_box.extents
        inertia = trimesh.inertia.box_inertia(float(mass), ext)

    scale_note = (
        f"Absolute scale from a single image is unreliable. Mesh rescaled so "
        f"longest axis = {real_world_longest_dim_meters} m (user-provided estimate). "
        f"Mass computed at density {density} kg/m^3 from volume {vol:.6e} m^3."
    )
    log.info("estimate_physics end -> mass=%.4f kg", mass)
    return float(mass), np.array(inertia, dtype=float), scale_note, rescaled_path
