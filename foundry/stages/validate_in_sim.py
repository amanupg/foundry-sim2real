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
    angles = [(0, -30), (90, -30), (180, -30), (270, -30)]
    for i, (yaw, pitch) in enumerate(angles):
        try:
            view = p.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=[0, 0, 0.05],
                distance=0.6,
                yaw=yaw,
                pitch=pitch,
                roll=0,
                upAxisIndex=2,
            )
            proj = p.getProjectionMatrix(
                fov=45, aspect=1.0, nearVal=0.01, farVal=10.0
            )
            w, h, px, _, _ = p.getCameraImage(
                256, 256, view, proj, renderer=p.ER_TINY_RENDERER
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
    """Fallback: render the mesh with trimesh's built-in scene viewer (still PNG)."""
    renders: list[Path] = []
    try:
        import trimesh
        mesh = trimesh.load(processed_mesh_path, force="mesh")
        angles = [(0, -30), (90, -30), (180, -30), (270, -30)]
        for i, (yaw, pitch) in enumerate(angles):
            scene = trimesh.Scene(mesh)
            scene.camera_transform = scene.camera.look_at(
                mesh.bounds, rotation=np.radians([pitch, yaw, 0])
            )
            out = run_dir / "renders" / f"view_{i}.png"
            try:
                png = scene.save_image(resolution=[256, 256])
                out.write_bytes(png)
                renders.append(out)
            except Exception:
                # Last resort: render depthless silhouette.
                continue
    except Exception as e:
        log.warning("validate_in_sim: trimesh render fallback failed (%s)", e)
    return renders


def validate_in_sim(urdf_path: Path, processed_mesh_path: Path, run_dir: Path,
                    frames: int = PYBULLET_FRAMES) -> tuple[dict, list[Path]]:
    log.info("validate_in_sim start (frames=%d)", frames)
    import pybullet as p
    client_id = p.connect(p.DIRECT)  # headless
    sim_report: dict = {"stable": False, "settle_time": None, "anomalies": []}
    renders: list[Path] = []
    try:
        p.setAdditionalSearchPath(str(urdf_path.parent), physicsClientId=client_id)
        p.setGravity(0, 0, -9.81, physicsClientId=client_id)
        p.loadURDF("plane.urdf", physicsClientId=client_id)

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
