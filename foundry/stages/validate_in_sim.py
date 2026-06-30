"""Stage 7: validate_in_sim(asset_pack_dir) -> sim_report, render_paths.

Load the URDF into PyBullet headless; run a drop test (spawn above a ground
plane, step ~240 frames); record whether it settles vs explodes/sinks/jitters;
render the object from 4 fixed camera angles using PyBullet's offscreen
camera (TINY renderer, CPU, no GL context needed). Return sim_report and
render image paths.

Fallback: if PyBullet offscreen render fails, use trimesh still render.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from ..config import PYBULLET_FRAMES

log = logging.getLogger("foundry.validate_in_sim")


def _render_pybullet(client_id, urdf_path: Path, run_dir: Path) -> list[Path]:
    """Render 4 views using PyBullet's offscreen TINY renderer (CPU, no GL)."""
    import pybullet as p
    renders: list[Path] = []
    angles = [(0, -25), (90, -25), (180, -25), (270, -25)]
    for i, (yaw, pitch) in enumerate(angles):
        try:
            view = p.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=[0, 0, 0.08],
                distance=0.35,
                yaw=yaw,
                pitch=pitch,
                roll=0,
                upAxisIndex=2,
                physicsClientId=client_id,
            )
            proj = p.computeProjectionMatrixFOV(
                fov=50, aspect=1.0, nearVal=0.01, farVal=10.0
            )
            w, h, px, _, _ = p.getCameraImage(
                256, 256, view, proj, renderer=p.ER_TINY_RENDERER,
                physicsClientId=client_id,
            )
            arr = np.array(px, dtype=np.uint8).reshape(h, w, 4)
            from PIL import Image
            out = run_dir / "renders" / f"view_{i}.png"
            Image.fromarray(arr[:, :, :3]).save(out)
            renders.append(out)
        except Exception as e:
            log.warning("validate_in_sim: pybullet render %d failed (%s)", i, e)
    return renders


def _render_trimesh_fallback(processed_mesh_path: Path, run_dir: Path) -> list[Path]:
    """Fallback: render simple projected silhouettes of the mesh with PIL (no GL)."""
    renders: list[Path] = []
    try:
        import trimesh
        import numpy as np
        from PIL import Image, ImageDraw
        mesh = trimesh.load(processed_mesh_path, force="mesh")
        verts = np.array(mesh.vertices)
        # Normalize verts to [-1, 1] for projection.
        vmin, vmax = verts.min(axis=0), verts.max(axis=0)
        span = (vmax - vmin).max() or 1.0
        norm = (verts - (vmin + vmax) / 2.0) / (span / 2.0)
        angles = [(0, -30), (90, -30), (180, -30), (270, -30)]
        for i, (yaw, pitch) in enumerate(angles):
            try:
                import math
                cy, sy = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
                cp, sp = math.cos(math.radians(pitch)), math.sin(math.radians(pitch))
                R = np.array([[cp, 0, sp], [sy * sp, cp, -sy * cp],
                              [-cy * sp, sy, cy * cp]])
                proj = norm @ R.T
                img = Image.new("RGB", (256, 256), (11, 13, 18))
                d = ImageDraw.Draw(img)
                pts = [((p[0] * 0.4 + 0.5) * 256, (-p[1] * 0.4 + 0.5) * 256) for p in proj]
                d.point(pts, fill=(140, 160, 200))
                out = run_dir / "renders" / f"view_{i}.png"
                img.save(out)
                renders.append(out)
            except Exception as e:
                log.warning("validate_in_sim: fallback render %d failed (%s)", i, e)
    except Exception as e:
        log.warning("validate_in_sim: trimesh render fallback failed (%s)", e)
    return renders


def validate_in_sim(urdf_path: Path, processed_mesh_path: Path, run_dir: Path,
                    frames: int = PYBULLET_FRAMES) -> tuple[dict, list[Path]]:
    log.info("validate_in_sim start (frames=%d)", frames)
    import pybullet as p
    import pybullet_data
    client_id = p.connect(p.DIRECT)  # headless
    plane_urdf = str(Path(pybullet_data.getDataPath()) / "plane.urdf")
    sim_report: dict = {"stable": False, "settle_time": None, "anomalies": []}
    renders: list[Path] = []
    try:
        p.setAdditionalSearchPath(str(urdf_path.parent), physicsClientId=client_id)
        p.setGravity(0, 0, -9.81, physicsClientId=client_id)
        p.loadURDF(plane_urdf, physicsClientId=client_id)

        # Load object URDF; spawn above ground.
        body = p.loadURDF(str(urdf_path), [0, 0, 0.3], physicsClientId=client_id)
        start_pos, _ = p.getBasePositionAndOrientation(body, physicsClientId=client_id)

        max_velocity = 0.0
        settled = False
        settle_time = None
        prev_z = start_pos[2]
        jitter_count = 0
        for step in range(frames):
            p.stepSimulation(physicsClientId=client_id)
            pos, _ = p.getBasePositionAndOrientation(body, physicsClientId=client_id)
            vel, _ = p.getBaseVelocity(body, physicsClientId=client_id)
            speed = float(np.linalg.norm(vel))
            max_velocity = max(max_velocity, speed)
            dz = pos[2] - prev_z
            if step > 30 and speed < 0.01 and abs(dz) < 1e-4:
                if not settled:
                    settled = True
                    settle_time = step
            elif step > 60 and speed > 0.05:
                jitter_count += 1
            prev_z = pos[2]

        final_pos, _ = p.getBasePositionAndOrientation(body, physicsClientId=client_id)
        # Anomaly detection.
        anomalies: list[str] = []
        if max_velocity > 5.0:
            anomalies.append("explosive_motion")
        if final_pos[2] < -0.5:
            anomalies.append("sank_through_ground")
        if jitter_count > 30:
            anomalies.append("persistent_jitter")
        if not settled and not anomalies:
            anomalies.append("did_not_settle_in_time")

        sim_report = {
            "stable": bool(settled and not anomalies),
            "settle_time": settle_time,
            "max_velocity": max_velocity,
            "final_z": float(final_pos[2]),
            "anomalies": anomalies,
        }

        # Render views.
        renders = _render_pybullet(client_id, urdf_path, run_dir)
        if not renders:
            log.warning("validate_in_sim: no pybullet renders; trimesh fallback")
            renders = _render_trimesh_fallback(processed_mesh_path, run_dir)
    except Exception as e:
        log.exception("validate_in_sim: error (%s); degrading", e)
        sim_report["anomalies"].append(f"sim_error: {e}")
        renders = _render_trimesh_fallback(processed_mesh_path, run_dir)
    finally:
        try:
            p.disconnect(physicsClientId=client_id)
        except Exception:
            pass

    log.info("validate_in_sim end -> stable=%s, %d renders", sim_report["stable"], len(renders))
    return sim_report, renders
