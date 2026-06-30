"""Stage 1: preprocess(image_path) -> clean_image_path.

Run rembg to remove background, center and pad the object on a neutral
background, return the cleaned image. Falls back to a copy if rembg fails.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from PIL import Image

log = logging.getLogger("foundry.preprocess")


def _remove_bg(image_path: Path) -> Image.Image:
    """Remove background with rembg. Returns RGBA image."""
    try:
        from rembg import remove
        data = image_path.read_bytes()
        out = remove(data)
        import io
        return Image.open(io.BytesIO(out)).convert("RGBA")
    except Exception as e:
        log.warning("rembg failed (%s); falling back to plain copy", e)
        return Image.open(image_path).convert("RGBA")


def _center_pad(img: Image.Image, bg=(235, 235, 235, 255), pad_frac: float = 0.1) -> Image.Image:
    """Crop to alpha bbox, center on square neutral canvas with padding."""
    arr = np.array(img)
    if arr.shape[2] == 4:
        alpha = arr[:, :, 3]
    else:
        alpha = np.full(arr.shape[:2], 255, dtype=np.uint8)
    ys, xs = np.where(alpha > 20)
    if len(xs) == 0 or len(ys) == 0:
        return img.convert("RGB")
    x0, x1 = xs.min(), xs.max() + 1
    y0, y1 = ys.min(), ys.max() + 1
    cropped = img.crop((x0, y0, x1, y1))
    w, h = cropped.size
    side = int(max(w, h) * (1 + pad_frac * 2))
    canvas = Image.new("RGBA", (side, side), bg)
    canvas.paste(cropped, ((side - w) // 2, (side - h) // 2), cropped if cropped.mode == "RGBA" else None)
    return canvas.convert("RGB")


def preprocess(image_path: Path, run_dir: Path, tighter: bool = False) -> Path:
    """Returns path to cleaned image on neutral background."""
    log.info("preprocess start (tighter=%s) %s", tighter, image_path.name)
    image_path = Path(image_path)
    img = _remove_bg(image_path)
    pad = 0.05 if tighter else 0.12
    clean = _center_pad(img, pad_frac=pad)
    out = run_dir / "clean.png"
    clean.save(out)
    log.info("preprocess end -> %s", out.name)
    return out
