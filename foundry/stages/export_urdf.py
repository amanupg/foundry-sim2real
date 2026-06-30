"""Stage 6: export_urdf(processed_mesh_path, collision_path, mass, inertia)
            -> urdf_path, asset_pack_dir.

Write a valid URDF referencing the visual mesh and collision hulls with the
computed inertial block. Bundle mesh + collision + urdf + report.json into an
asset pack directory; zip it.
"""
from __future__ import annotations

import json
import logging
import shutil
import zipfile
from pathlib import Path

import numpy as np
import trimesh

log = logging.getLogger("foundry.export_urdf")


def _format_inertia(I: np.ndarray) -> str:
    """Format a 3x3 inertia tensor into URDF <inertia> attributes."""
    I = np.asarray(I, dtype=float)
    if I.shape != (3, 3):
        I = np.eye(3) * float(I[0, 0] if I.ndim == 2 and I.size else 1.0)
    # Ensure symmetric
    I = 0.5 * (I + I.T)
    return (
        f'ixx="{I[0,0]:.9e}" ixy="{I[0,1]:.9e}" ixz="{I[0,2]:.9e}" '
        f'iyy="{I[1,1]:.9e}" iyz="{I[1,2]:.9e}" izz="{I[2,2]:.9e}"'
    )


def export_urdf(processed_mesh_path: Path, collision_path: Path, run_dir: Path,
                mass: float, inertia, scale_note: str, report: dict) -> tuple[Path, Path]:
    log.info("export_urdf start")
    pack_dir = run_dir / "asset_pack"
    pack_dir.mkdir(exist_ok=True)
    meshes_dir = pack_dir / "meshes"
    meshes_dir.mkdir(exist_ok=True)

    # Copy visual mesh (glb for three.js viewer, obj for pybullet).
    visual_glb = meshes_dir / "visual.glb"
    shutil.copy2(processed_mesh_path, visual_glb)
    visual_obj = meshes_dir / "visual.obj"
    try:
        mesh_v = trimesh.load(processed_mesh_path, force="mesh")
        mesh_v.export(visual_obj)
    except Exception as e:
        log.warning("export_urdf: visual.obj export failed (%s)", e)
        mesh_v = trimesh.load(processed_mesh_path, force="mesh")
        mesh_v.export(visual_obj)

    # Copy collision mesh(es) as obj (pybullet needs obj/stl, not glb).
    collision_obj = meshes_dir / "collision.obj"
    try:
        mesh_c = trimesh.load(collision_path, force="mesh")
        mesh_c.export(collision_obj)
    except Exception as e:
        log.warning("export_urdf: collision.obj export failed (%s)", e)
        # Fallback: convex hull of visual.
        mesh_c = trimesh.load(processed_mesh_path, force="mesh").convex_hull
        mesh_c.export(collision_obj)

    inertia_str = _format_inertia(np.asarray(inertia, dtype=float))

    urdf = f"""<?xml version="1.0"?>
<robot name="foundry_object">
  <link name="base">
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <mesh filename="meshes/visual.obj"/>
      </geometry>
      <material name="foundry_orange">
        <color rgba="0.85 0.45 0.15 1.0"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <mesh filename="meshes/collision.obj"/>
      </geometry>
    </collision>
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="{mass:.9e}"/>
      <inertia {inertia_str}/>
    </inertial>
  </link>
</robot>
"""
    urdf_path = pack_dir / "object.urdf"
    urdf_path.write_text(urdf, encoding="utf-8")

    # Write report.json alongside.
    full_report = dict(report)
    full_report["mass_kg"] = float(mass)
    full_report["inertia"] = np.asarray(inertia, dtype=float).tolist()
    full_report["scale_note"] = scale_note
    (pack_dir / "report.json").write_text(
        json.dumps(full_report, indent=2, default=str), encoding="utf-8"
    )

    # Zip the asset pack.
    zip_path = run_dir / "asset_pack.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in pack_dir.rglob("*"):
            if f.is_file():
                zf.write(f, f.relative_to(pack_dir))
    log.info("export_urdf end -> %s, %s", urdf_path.name, zip_path.name)
    return urdf_path, zip_path
