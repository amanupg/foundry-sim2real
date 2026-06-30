"""Stage 4: make_collision(processed_mesh_path) -> collision_path.

Run coacd approximate convex decomposition for collision hulls. If coacd
fails, fall back to the convex hull from trimesh.
"""
from __future__ import annotations

import logging
from pathlib import Path

import trimesh

log = logging.getLogger("foundry.make_collision")


def make_collision(processed_mesh_path: Path, run_dir: Path, threshold: float = 0.05) -> Path:
    log.info("make_collision start (threshold=%s) %s", threshold, processed_mesh_path.name)
    mesh = trimesh.load(processed_mesh_path, force="mesh")
    collision_dir = run_dir / "collision"
    collision_dir.mkdir(exist_ok=True)

    hulls: list[trimesh.Trimesh] = []
    try:
        import coacd
        coacd_mesh = coacd.Mesh(mesh.vertices, mesh.faces)
        parts = coacd.run_coacd(coacd_mesh, threshold=threshold, max_convex_hull=64)
        for verts, faces in parts:
            hulls.append(trimesh.Trimesh(vertices=verts, faces=faces))
        log.info("make_collision: coacd produced %d hulls", len(hulls))
    except Exception as e:
        log.warning("make_collision: coacd failed (%s); using convex hull", e)
        hulls = [mesh.convex_hull]

    if not hulls:
        log.warning("make_collision: no hulls produced; using convex hull")
        hulls = [mesh.convex_hull]

    # Export as a single GLB containing all hull meshes.
    scene = trimesh.Scene(hulls)
    out = collision_dir / "collision.glb"
    scene.export(out)
    log.info("make_collision end -> %s (%d hulls)", out.name, len(hulls))
    return out
