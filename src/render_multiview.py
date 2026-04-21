from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import open3d as o3d
from tqdm import tqdm

from src.utils.geometry import fibonacci_sphere_points, load_geometry, normalize_geometry
from src.utils.io import ensure_dir, list_mesh_files, set_seed, setup_logging

LOGGER = logging.getLogger(__name__)


def render_specimen(
    renderer: o3d.visualization.rendering.OffscreenRenderer,
    mesh_path: Path,
    input_root: Path,
    out_dir: Path,
    views: int,
    size: int,
    light_direction: tuple[float, float, float],
    light_color: tuple[float, float, float],
    light_intensity: float,
) -> bool:
    mesh_rel = mesh_path.relative_to(input_root)
    sid = mesh_rel.stem
    specimen_out_dir = out_dir / mesh_rel.parent
    ensure_dir(specimen_out_dir)
    try:
        geom = normalize_geometry(load_geometry(mesh_path))
    except Exception as e:
        LOGGER.exception("Failed to load/normalize %s: %s", mesh_path, e)
        return False

    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultLit"
    mat.base_color = (0.8, 0.8, 0.8, 1.0)
    if isinstance(geom, o3d.geometry.PointCloud):
        mat.point_size = 3.0

    scene = renderer.scene
    scene.clear_geometry()
    scene.add_geometry("specimen", geom, mat)
    scene.scene.set_sun_light(light_direction, light_color, light_intensity)
    scene.scene.enable_sun_light(True)

    bbox = o3d.geometry.AxisAlignedBoundingBox([-0.5, -0.5, -0.5], [0.5, 0.5, 0.5])
    center = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    up = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    camera_positions = fibonacci_sphere_points(views, radius=2.0)
    ok = True
    for i, eye in enumerate(camera_positions):
        try:
            renderer.setup_camera(60.0, center, eye, up)
            img = renderer.render_to_image()
            out_path = specimen_out_dir / f"{sid}_view{i:02d}.png"
            o3d.io.write_image(str(out_path), img)
        except Exception as e:
            ok = False
            LOGGER.exception("Render failed %s view %d: %s", mesh_path, i, e)
    return ok


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render multi-view PNG images from 3D meshes/point clouds.")
    parser.add_argument("--in", dest="input_dir", type=Path, required=True, help="Input directory with .ply/.obj/.stl/.off")
    parser.add_argument("--out", dest="output_dir", type=Path, required=True, help="Output directory for rendered PNGs")
    parser.add_argument("--views", type=int, default=12)
    parser.add_argument("--size", type=int, default=384)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging()
    set_seed(args.seed)

    ensure_dir(args.output_dir)
    mesh_files = list_mesh_files(args.input_dir)
    if not mesh_files:
        LOGGER.warning("No mesh files found in %s", args.input_dir)
        return

    renderer = o3d.visualization.rendering.OffscreenRenderer(args.size, args.size)
    renderer.scene.set_background([1.0, 1.0, 1.0, 1.0])

    success = 0
    for mesh_path in tqdm(mesh_files, desc="Rendering"):
        if render_specimen(
            renderer=renderer,
            mesh_path=mesh_path,
            input_root=args.input_dir,
            out_dir=args.output_dir,
            views=args.views,
            size=args.size,
            light_direction=(0.577, -0.577, -0.577),
            light_color=(1.0, 1.0, 1.0),
            light_intensity=50000,
        ):
            success += 1

    LOGGER.info("Rendered %d/%d specimens", success, len(mesh_files))


if __name__ == "__main__":
    main()
