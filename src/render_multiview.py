from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

import numpy as np
import open3d as o3d
from tqdm import tqdm

from src.utils.geometry import fibonacci_sphere_points, load_geometry, normalize_geometry
from src.utils.io import ensure_dir, list_mesh_files, set_seed, setup_logging

LOGGER = logging.getLogger(__name__)


def _make_material_for_appearance(
    geom: o3d.geometry.Geometry,
    appearance: str,
) -> o3d.visualization.rendering.MaterialRecord:
    """Build material for rendering appearance modes.

    gray_lit uses a fixed gray material to suppress specimen color/texture differences
    for shape-only rendering. color_lit keeps original vertex colors/textures when
    available for optional appearance-aware rendering.
    """
    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultLit"

    if appearance == "gray_lit":
        mat.base_color = (0.8, 0.8, 0.8, 1.0)
    elif appearance == "color_lit":
        if isinstance(geom, o3d.geometry.TriangleMesh):
            has_vertex_colors = geom.has_vertex_colors()
            textures = getattr(geom, "textures", [])
            num_textures = len(textures) if textures is not None else 0
            if num_textures > 0:
                try:
                    mat.albedo_img = textures[0]
                except Exception:
                    LOGGER.debug("Failed to assign albedo texture; continuing without albedo_img.", exc_info=True)

            if not has_vertex_colors and num_textures == 0:
                mat.base_color = (0.8, 0.8, 0.8, 1.0)
        elif isinstance(geom, o3d.geometry.PointCloud):
            if not geom.has_colors():
                mat.base_color = (0.8, 0.8, 0.8, 1.0)
    else:
        raise ValueError(f"Unsupported appearance: {appearance}")

    if isinstance(geom, o3d.geometry.PointCloud):
        mat.point_size = 3.0
    return mat


def _compute_bbox_fill_ratio(image: o3d.geometry.Image, bg_threshold: int = 245) -> float:
    image_np = np.asarray(image)
    if image_np.ndim != 3 or image_np.shape[2] < 3:
        return 0.0

    rgb = image_np[..., :3]
    non_bg_mask = np.any(rgb < bg_threshold, axis=2)
    non_bg_indices = np.argwhere(non_bg_mask)
    if non_bg_indices.size == 0:
        return 0.0

    ymin, xmin = non_bg_indices.min(axis=0)
    ymax, xmax = non_bg_indices.max(axis=0)
    bbox_area = float((ymax - ymin + 1) * (xmax - xmin + 1))
    image_area = float(image_np.shape[0] * image_np.shape[1])
    return bbox_area / image_area if image_area > 0 else 0.0


def _autotune_camera_radius(
    renderer: o3d.visualization.rendering.OffscreenRenderer,
    center: np.ndarray,
    up: np.ndarray,
    fov_deg: float,
    target_fill_min: float,
    target_fill_max: float,
    initial_radius: float = 2.0,
    min_radius: float = 0.25,
    max_radius: float = 8.0,
    max_iter: int = 12,
) -> tuple[float, float, int]:
    direction = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    def render_fill(radius: float) -> float:
        eye = direction * radius
        renderer.setup_camera(fov_deg, center, eye, up)
        preview = renderer.render_to_image()
        return _compute_bbox_fill_ratio(preview)

    lo = min_radius
    hi = max_radius
    current_radius = float(np.clip(initial_radius, lo, hi))
    current_fill = render_fill(current_radius)
    best_radius = current_radius
    best_fill = current_fill
    best_gap = 0.0 if target_fill_min <= best_fill <= target_fill_max else min(
        abs(best_fill - target_fill_min),
        abs(best_fill - target_fill_max),
    )

    if target_fill_min <= current_fill <= target_fill_max:
        return best_radius, best_fill, 0

    for it in range(1, max_iter + 1):
        if current_fill < target_fill_min:
            hi = current_radius
        else:
            lo = current_radius
        current_radius = (lo + hi) * 0.5
        current_fill = render_fill(current_radius)

        gap = 0.0 if target_fill_min <= current_fill <= target_fill_max else min(
            abs(current_fill - target_fill_min),
            abs(current_fill - target_fill_max),
        )
        if gap < best_gap:
            best_radius = current_radius
            best_fill = current_fill
            best_gap = gap

        if target_fill_min <= current_fill <= target_fill_max:
            return current_radius, current_fill, it

    return best_radius, best_fill, max_iter


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
    appearance: str,
    auto_zoom: bool,
    target_fill_min: float,
    target_fill_max: float,
) -> tuple[bool, float, float]:
    mesh_rel = mesh_path.relative_to(input_root)
    sid = mesh_rel.stem
    specimen_out_dir = out_dir / mesh_rel.parent
    ensure_dir(specimen_out_dir)
    try:
        geom = normalize_geometry(load_geometry(mesh_path))
    except Exception as e:
        LOGGER.exception("Failed to load/normalize %s: %s", mesh_path, e)
        return False, 0.0, float("nan")

    if appearance == "color_lit":
        if isinstance(geom, o3d.geometry.TriangleMesh):
            has_vertex_colors = geom.has_vertex_colors()
            has_triangle_uvs = geom.has_triangle_uvs()
            textures = getattr(geom, "textures", [])
            num_textures = len(textures) if textures is not None else 0
            LOGGER.debug(
                "Color appearance for %s: has_vertex_colors=%s has_triangle_uvs=%s num_textures=%d",
                mesh_rel.as_posix(),
                has_vertex_colors,
                has_triangle_uvs,
                num_textures,
            )
            if not has_vertex_colors and num_textures == 0:
                LOGGER.warning(
                    "color_lit requested but no vertex colors/textures were detected for %s; "
                    "falling back to gray material.",
                    mesh_rel.as_posix(),
                )
        elif isinstance(geom, o3d.geometry.PointCloud):
            has_point_colors = geom.has_colors()
            LOGGER.debug(
                "Color appearance for %s: has_point_colors=%s",
                mesh_rel.as_posix(),
                has_point_colors,
            )
            if not has_point_colors:
                LOGGER.warning(
                    "color_lit requested but no point colors were detected for %s; "
                    "falling back to gray material.",
                    mesh_rel.as_posix(),
                )

    mat = _make_material_for_appearance(geom=geom, appearance=appearance)

    scene = renderer.scene
    scene.clear_geometry()
    scene.add_geometry("specimen", geom, mat)
    scene.scene.set_sun_light(light_direction, light_color, light_intensity)
    scene.scene.enable_sun_light(True)

    center = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    fov_deg = 60.0
    final_radius = 2.0
    preview_fill_ratio = float("nan")

    if auto_zoom:
        final_radius, preview_fill_ratio, iters = _autotune_camera_radius(
            renderer=renderer,
            center=center,
            up=up,
            fov_deg=fov_deg,
            target_fill_min=target_fill_min,
            target_fill_max=target_fill_max,
        )
        LOGGER.info(
            "Auto-zoom specimen=%s fill_ratio=%.4f final_radius=%.4f iterations=%d target=[%.2f, %.2f]",
            mesh_rel.as_posix(),
            preview_fill_ratio,
            final_radius,
            iters,
            target_fill_min,
            target_fill_max,
        )

    camera_positions = fibonacci_sphere_points(views, radius=final_radius)
    ok = True
    for i, eye in enumerate(camera_positions):
        try:
            renderer.setup_camera(fov_deg, center, eye, up)
            img = renderer.render_to_image()
            out_path = specimen_out_dir / f"{sid}_view{i:02d}.png"
            o3d.io.write_image(str(out_path), img)
        except Exception as e:
            ok = False
            LOGGER.exception("Render failed %s view %d: %s", mesh_path, i, e)
    return ok, preview_fill_ratio, final_radius


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render multi-view PNG images from 3D meshes/point clouds.")
    parser.add_argument("--in", dest="input_dir", type=Path, required=True, help="Input directory with .ply/.obj/.stl/.off")
    parser.add_argument("--out", dest="output_dir", type=Path, required=True, help="Output directory for rendered PNGs")
    parser.add_argument("--views", type=int, default=12)
    parser.add_argument("--size", type=int, default=384)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--appearance",
        choices=["gray_lit", "color_lit"],
        default="gray_lit",
        help=(
            "Rendering appearance. "
            "gray_lit: fixed gray material with fixed lighting for shape-only rendering. "
            "color_lit: use original vertex colors/textures when available with fixed lighting."
        ),
    )
    parser.add_argument("--auto-zoom", action="store_true", help="Automatically tune camera radius per specimen.")
    parser.add_argument("--target-fill-min", type=float, default=0.35, help="Minimum target preview fill ratio.")
    parser.add_argument("--target-fill-max", type=float, default=0.55, help="Maximum target preview fill ratio.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging()
    set_seed(args.seed)
    if not 0 < args.target_fill_min < args.target_fill_max < 1:
        raise ValueError("--target-fill-min / --target-fill-max must satisfy 0 < min < max < 1")

    ensure_dir(args.output_dir)
    LOGGER.info("Rendering appearance: %s", args.appearance)
    mesh_files = list_mesh_files(args.input_dir)
    if not mesh_files:
        LOGGER.warning("No mesh files found in %s", args.input_dir)
        return

    renderer = o3d.visualization.rendering.OffscreenRenderer(args.size, args.size)
    renderer.scene.set_background([1.0, 1.0, 1.0, 1.0])
    zoom_report_path = args.output_dir / "auto_zoom_report.csv"
    zoom_rows: list[dict[str, str | float | bool]] = []

    success = 0
    for mesh_path in tqdm(mesh_files, desc="Rendering"):
        ok, fill_ratio, final_radius = render_specimen(
            renderer=renderer,
            mesh_path=mesh_path,
            input_root=args.input_dir,
            out_dir=args.output_dir,
            views=args.views,
            size=args.size,
            light_direction=(0.577, -0.577, -0.577),
            light_color=(1.0, 1.0, 1.0),
            light_intensity=50000,
            appearance=args.appearance,
            auto_zoom=args.auto_zoom,
            target_fill_min=args.target_fill_min,
            target_fill_max=args.target_fill_max,
        )
        zoom_rows.append(
            {
                "specimen": mesh_path.relative_to(args.input_dir).as_posix(),
                "ok": ok,
                "auto_zoom": args.auto_zoom,
                "preview_fill_ratio": fill_ratio,
                "final_radius": final_radius,
                "appearance": args.appearance,
            }
        )
        if ok:
            success += 1

    with zoom_report_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["specimen", "ok", "auto_zoom", "preview_fill_ratio", "final_radius", "appearance"],
        )
        writer.writeheader()
        writer.writerows(zoom_rows)
    LOGGER.info("Auto-zoom report written: %s", zoom_report_path)
    LOGGER.info("Rendered %d/%d specimens", success, len(mesh_files))


if __name__ == "__main__":
    main()
