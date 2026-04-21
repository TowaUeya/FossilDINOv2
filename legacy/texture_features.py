"""旧実験コード。本研究の主張には使わない。"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from src.prefilter_common import save_yaml
from src.utils.io import ensure_dir, set_seed, setup_logging


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build optional specimen-level texture/color descriptors from rendered views.")
    p.add_argument("--renders", type=Path, required=True, help="Directory that contains specimen render subdirectories")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--color_space", choices=["rgb", "lab", "hsv"], default="lab")
    p.add_argument("--bins", type=int, default=16)
    p.add_argument("--pool", choices=["mean", "median"], default="mean")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    cmax = np.max(rgb, axis=-1)
    cmin = np.min(rgb, axis=-1)
    delta = cmax - cmin

    h = np.zeros_like(cmax)
    nz = delta > 1e-12
    idx = nz & (cmax == r)
    h[idx] = np.mod((g[idx] - b[idx]) / delta[idx], 6.0)
    idx = nz & (cmax == g)
    h[idx] = (b[idx] - r[idx]) / delta[idx] + 2.0
    idx = nz & (cmax == b)
    h[idx] = (r[idx] - g[idx]) / delta[idx] + 4.0
    h = h / 6.0

    s = np.zeros_like(cmax)
    nonzero = cmax > 1e-12
    s[nonzero] = delta[nonzero] / cmax[nonzero]
    v = cmax
    return np.stack([h, s, v], axis=-1)


def _rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    def srgb_to_linear(x: np.ndarray) -> np.ndarray:
        return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)

    rgb_lin = srgb_to_linear(rgb)
    m = np.array(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ],
        dtype=np.float64,
    )
    xyz = np.einsum("...c,dc->...d", rgb_lin, m)
    x, y, z = xyz[..., 0] / 0.95047, xyz[..., 1] / 1.0, xyz[..., 2] / 1.08883

    eps = (6.0 / 29.0) ** 3
    k = (29.0 / 3.0) ** 2 / 3.0

    def f(t: np.ndarray) -> np.ndarray:
        return np.where(t > eps, np.cbrt(t), k * t + 4.0 / 29.0)

    fx, fy, fz = f(x), f(y), f(z)
    L = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b = 200.0 * (fy - fz)

    # normalize to [0,1] range for histogram bins
    Ln = np.clip(L / 100.0, 0.0, 1.0)
    an = np.clip((a + 128.0) / 255.0, 0.0, 1.0)
    bn = np.clip((b + 128.0) / 255.0, 0.0, 1.0)
    return np.stack([Ln, an, bn], axis=-1)


def _convert_color_space(rgb01: np.ndarray, color_space: str) -> np.ndarray:
    if color_space == "rgb":
        return rgb01
    if color_space == "hsv":
        return _rgb_to_hsv(rgb01)
    return _rgb_to_lab(rgb01)


def _image_hist_descriptor(img_path: Path, color_space: str, bins: int) -> np.ndarray:
    img = np.asarray(Image.open(img_path).convert("RGB"), dtype=np.float64) / 255.0
    mask = np.mean(img, axis=-1) < 0.985  # white background を弱める
    if not np.any(mask):
        mask = np.ones(img.shape[:2], dtype=bool)

    pix = _convert_color_space(img[mask], color_space=color_space)
    feats: list[np.ndarray] = []
    for ci in range(3):
        hist, _ = np.histogram(np.clip(pix[:, ci], 0.0, 1.0), bins=bins, range=(0.0, 1.0), density=False)
        hist = hist.astype(np.float64)
        hist = hist / max(hist.sum(), 1.0)
        feats.append(hist)
    out = np.concatenate(feats, axis=0)
    norm = np.linalg.norm(out)
    return (out / norm).astype(np.float32) if norm > 0 else out.astype(np.float32)


def _is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"}


def _collect_specimen_images(renders_root: Path) -> list[tuple[str, list[Path]]]:
    """Collect specimen id + image list from a renders directory.

    Supports both of these common layouts:
      1) renders/<specimen_id>/<view>.png
      2) renders/<specimen_id>/**/<view>.png
    """
    if not renders_root.exists():
        return []

    specimens: list[tuple[str, list[Path]]] = []
    for child in sorted(renders_root.iterdir()):
        if not child.is_dir():
            continue
        images = sorted([p for p in child.rglob("*") if _is_image_file(p)])
        if images:
            specimens.append((child.relative_to(renders_root).as_posix(), images))
    return specimens


def main() -> None:
    args = parse_args()
    setup_logging()
    set_seed(args.seed)
    ensure_dir(args.out)

    specimen_images = _collect_specimen_images(args.renders)
    ids: list[str] = []
    pooled: list[np.ndarray] = []

    for specimen_id, images in specimen_images:
        per_view = np.stack([_image_hist_descriptor(p, args.color_space, args.bins) for p in images], axis=0)
        feat = np.median(per_view, axis=0) if args.pool == "median" else np.mean(per_view, axis=0)
        norm = np.linalg.norm(feat)
        if norm > 0:
            feat = feat / norm
        pooled.append(feat.astype(np.float32))
        ids.append(specimen_id)

    arr = np.stack(pooled, axis=0) if pooled else np.zeros((0, int(args.bins) * 3), dtype=np.float32)
    np.save(args.out / "texture_features.npy", arr)
    (args.out / "ids.txt").write_text("\n".join(ids), encoding="utf-8")
    save_yaml(
        args.out / "texture_feature_config.yaml",
        {
            "renders": str(args.renders),
            "out": str(args.out),
            "color_space": args.color_space,
            "bins": int(args.bins),
            "pool": args.pool,
            "n_specimens": len(ids),
            "feature_dim": int(arr.shape[1]) if arr.ndim == 2 else 0,
        },
    )


if __name__ == "__main__":
    main()
