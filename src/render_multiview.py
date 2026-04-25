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


def _compute_bbox_fill_ratio(
    depth_image: o3d.geometry.Image | None = None,
    image: o3d.geometry.Image | None = None,
    bg_threshold: int = 245,
) -> float:
    if depth_image is not None:
        depth_np = np.asarray(depth_image)
        if depth_np.ndim == 3 and depth_np.shape[2] == 1:
            depth_np = depth_np[..., 0]
        if depth_np.ndim != 2:
            return 0.0

        finite_mask = np.isfinite(depth_np)
        positive_mask = depth_np > 0
        valid_mask = finite_mask & positive_mask
        if not np.any(valid_mask):
            return 0.0

        bg_depth = float(np.max(depth_np[valid_mask]))
        fg_mask = valid_mask & (depth_np < (bg_depth - 1e-6))
        if not np.any(fg_mask):
            return 0.0

        fg_indices = np.argwhere(fg_mask)
        ymin, xmin = fg_indices.min(axis=0)
        ymax, xmax = fg_indices.max(axis=0)
        bbox_area = float((ymax - ymin + 1) * (xmax - xmin + 1))
        image_area = float(depth_np.shape[0] * depth_np.shape[1])
        return bbox_area / image_area if image_area > 0 else 0.0

    if image is None:
        return 0.0

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
    log_prefix: str = "",
) -> tuple[float, float, int]:
    direction = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    depth_fallback_logged = False

    def render_fill(radius: float) -> float:
        nonlocal depth_fallback_logged
        eye = direction * radius
        renderer.setup_camera(fov_deg, center, eye, up)
        preview = renderer.render_to_image()
        try:
            preview_depth = renderer.render_to_depth_image()
            return _compute_bbox_fill_ratio(depth_image=preview_depth)
        except Exception:
            if not depth_fallback_logged:
                LOGGER.warning(
                    "Depth preview failed for auto-zoom%s; fallback to RGB thresholding only.",
                    f" ({log_prefix})" if log_prefix else "",
                    exc_info=True,
                )
                depth_fallback_logged = True
            return _compute_bbox_fill_ratio(image=preview)

    lo = min_radius
    hi = max_radius
    current_radius = float(np.clip(initial_radius, lo, hi))
    current_fill = render_fill(current_radius)
    trace: list[tuple[int, float, float]] = [(0, current_radius, current_fill)]
    best_radius = current_radius
    best_fill = current_fill
    best_gap = 0.0 if target_fill_min <= best_fill <= target_fill_max else min(
        abs(best_fill - target_fill_min),
        abs(best_fill - target_fill_max),
    )

    if target_fill_min <= current_fill <= target_fill_max:
        trace_text = ", ".join([f"iter={it}:radius={rad:.4f},fill_ratio={fill:.4f}" for it, rad, fill in trace])
        LOGGER.info("Auto-zoom trace%s %s", f" ({log_prefix})" if log_prefix else "", trace_text)
        return best_radius, best_fill, 0

    for it in range(1, max_iter + 1):
        if current_fill < target_fill_min:
            hi = current_radius
        else:
            lo = current_radius
        current_radius = (lo + hi) * 0.5
        current_fill = render_fill(current_radius)
        trace.append((it, current_radius, current_fill))

        gap = 0.0 if target_fill_min <= current_fill <= target_fill_max else min(
            abs(current_fill - target_fill_min),
            abs(current_fill - target_fill_max),
        )
        if gap < best_gap:
            best_radius = current_radius
            best_fill = current_fill
            best_gap = gap

        if target_fill_min <= current_fill <= target_fill_max:
            trace_text = ", ".join([f"iter={trace_it}:radius={rad:.4f},fill_ratio={fill:.4f}" for trace_it, rad, fill in trace])
            LOGGER.info("Auto-zoom trace%s %s", f" ({log_prefix})" if log_prefix else "", trace_text)
            return current_radius, current_fill, it

    trace_text = ", ".join([f"iter={it}:radius={rad:.4f},fill_ratio={fill:.4f}" for it, rad, fill in trace])
    LOGGER.info("Auto-zoom trace%s %s", f" ({log_prefix})" if log_prefix else "", trace_text)
    return best_radius, best_fill, max_iter


def _render_scale_views(
    renderer: o3d.visualization.rendering.OffscreenRenderer,
    specimen_out_dir: Path,
    sid: str,
    scale_name: str,
    views: int,
    center: np.ndarray,
    up: np.ndarray,
    fov_deg: float,
    final_radius: float,
    multiscale: bool,
) -> bool:
    camera_positions = fibonacci_sphere_points(views, radius=final_radius)
    ok = True
    for i, eye in enumerate(camera_positions):
        try:
            renderer.setup_camera(fov_deg, center, eye, up)
            img = renderer.render_to_image()
            if multiscale:
                out_path = specimen_out_dir / f"{sid}_{scale_name}_view{i:02d}.png"
            else:
                out_path = specimen_out_dir / f"{sid}_view{i:02d}.png"
            o3d.io.write_image(str(out_path), img)
        except Exception as e:
            ok = False
            LOGGER.exception("Render failed specimen=%s scale=%s view=%d: %s", sid, scale_name, i, e)
    return ok


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
    multiscale_zoom: bool,
    loose_fill_min: float,
    loose_fill_max: float,
    up_fill_min: float,
    up_fill_max: float,
) -> tuple[bool, list[dict[str, str | float | bool | int]]]:
    mesh_rel = mesh_path.relative_to(input_root)
    sid = mesh_rel.stem
    specimen_out_dir = out_dir / mesh_rel.parent
    ensure_dir(specimen_out_dir)
    try:
        geom = normalize_geometry(load_geometry(mesh_path))
    except Exception as e:
        LOGGER.exception("Failed to load/normalize %s: %s", mesh_path, e)
        return False, []

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
    if multiscale_zoom:
        views_per_scale = views // 2
        scale_configs = [
            {"name": "loose", "views": views_per_scale, "target_fill_min": loose_fill_min, "target_fill_max": loose_fill_max},
            {"name": "up", "views": views_per_scale, "target_fill_min": up_fill_min, "target_fill_max": up_fill_max},
        ]
    else:
        scale_configs = [
            {"name": "single", "views": views, "target_fill_min": target_fill_min, "target_fill_max": target_fill_max},
        ]

    ok = True
    zoom_rows: list[dict[str, str | float | bool | int]] = []
    for scale_cfg in scale_configs:
        scale_name = str(scale_cfg["name"])
        scale_views = int(scale_cfg["views"])
        scale_target_fill_min = float(scale_cfg["target_fill_min"])
        scale_target_fill_max = float(scale_cfg["target_fill_max"])

        final_radius = 2.0
        preview_fill_ratio = float("nan")
        if auto_zoom:
            final_radius, preview_fill_ratio, iters = _autotune_camera_radius(
                renderer=renderer,
                center=center,
                up=up,
                fov_deg=fov_deg,
                target_fill_min=scale_target_fill_min,
                target_fill_max=scale_target_fill_max,
                log_prefix=f"{mesh_rel.as_posix()}:{scale_name}",
            )
            LOGGER.info(
                "Auto-zoom specimen=%s scale=%s fill_ratio=%.4f final_radius=%.4f iterations=%d target=[%.2f, %.2f]",
                mesh_rel.as_posix(),
                scale_name,
                preview_fill_ratio,
                final_radius,
                iters,
                scale_target_fill_min,
                scale_target_fill_max,
            )

        scale_ok = _render_scale_views(
            renderer=renderer,
            specimen_out_dir=specimen_out_dir,
            sid=sid,
            scale_name=scale_name,
            views=scale_views,
            center=center,
            up=up,
            fov_deg=fov_deg,
            final_radius=final_radius,
            multiscale=multiscale_zoom,
        )
        ok = ok and scale_ok
        zoom_rows.append(
            {
                "specimen": mesh_rel.as_posix(),
                "ok": scale_ok,
                "auto_zoom": auto_zoom,
                "appearance": appearance,
                "scale": scale_name,
                "scale_view_count": scale_views,
                "preview_fill_ratio": preview_fill_ratio,
                "final_radius": final_radius,
                "target_fill_min": scale_target_fill_min,
                "target_fill_max": scale_target_fill_max,
            }
        )

    return ok, zoom_rows


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
    parser.add_argument(
        "--multiscale-zoom",
        action="store_true",
        help=(
            "Render two auto-zoom scales per specimen: loose and up. "
            "When enabled, --views must be divisible by the number of scales, "
            "and each scale gets views / num_scales views."
        ),
    )
    parser.add_argument(
        "--loose-fill-min",
        type=float,
        default=0.35,
        help="Minimum target bbox fill ratio for the loose scale when --multiscale-zoom is enabled.",
    )
    parser.add_argument(
        "--loose-fill-max",
        type=float,
        default=0.55,
        help="Maximum target bbox fill ratio for the loose scale when --multiscale-zoom is enabled.",
    )
    parser.add_argument(
        "--up-fill-min",
        type=float,
        default=0.65,
        help="Minimum target bbox fill ratio for the up scale when --multiscale-zoom is enabled.",
    )
    parser.add_argument(
        "--up-fill-max",
        type=float,
        default=0.85,
        help="Maximum target bbox fill ratio for the up scale when --multiscale-zoom is enabled.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging()
    set_seed(args.seed)
    if not 0 < args.target_fill_min < args.target_fill_max < 1:
        raise ValueError("--target-fill-min / --target-fill-max must satisfy 0 < min < max < 1")
    if args.multiscale_zoom:
        if not args.auto_zoom:
            raise ValueError("--multiscale-zoom requires --auto-zoom")
        if args.views % 2 != 0:
            raise ValueError("--multiscale-zoom requires --views to be divisible by 2")
        if not 0 < args.loose_fill_min < args.loose_fill_max < 1:
            raise ValueError("--loose-fill-min / --loose-fill-max must satisfy 0 < min < max < 1")
        if not 0 < args.up_fill_min < args.up_fill_max < 1:
            raise ValueError("--up-fill-min / --up-fill-max must satisfy 0 < min < max < 1")
        if not args.loose_fill_max < args.up_fill_min:
            raise ValueError("--multiscale-zoom requires --loose-fill-max < --up-fill-min")

    ensure_dir(args.output_dir)
    LOGGER.info("Rendering appearance: %s", args.appearance)
    mesh_files = list_mesh_files(args.input_dir)
    if not mesh_files:
        LOGGER.warning("No mesh files found in %s", args.input_dir)
        return

    renderer = o3d.visualization.rendering.OffscreenRenderer(args.size, args.size)
    renderer.scene.set_background([1.0, 1.0, 1.0, 1.0])
    zoom_report_path = args.output_dir / "auto_zoom_report.csv"
    zoom_rows: list[dict[str, str | float | bool | int]] = []

    success = 0
    for mesh_path in tqdm(mesh_files, desc="Rendering"):
        ok, specimen_zoom_rows = render_specimen(
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
            multiscale_zoom=args.multiscale_zoom,
            loose_fill_min=args.loose_fill_min,
            loose_fill_max=args.loose_fill_max,
            up_fill_min=args.up_fill_min,
            up_fill_max=args.up_fill_max,
        )
        zoom_rows.extend(specimen_zoom_rows)
        if ok:
            success += 1

    with zoom_report_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "specimen",
                "ok",
                "auto_zoom",
                "appearance",
                "scale",
                "scale_view_count",
                "preview_fill_ratio",
                "final_radius",
                "target_fill_min",
                "target_fill_max",
            ],
        )
        writer.writeheader()
        writer.writerows(zoom_rows)
    LOGGER.info("Auto-zoom report written: %s", zoom_report_path)
    if args.multiscale_zoom and args.auto_zoom:
        radii_by_specimen: dict[str, dict[str, float]] = {}
        for row in zoom_rows:
            specimen = str(row["specimen"])
            scale = str(row["scale"])
            final_radius = float(row["final_radius"])
            radii_by_specimen.setdefault(specimen, {})[scale] = final_radius

        same_radius_specimens = []
        different_radius_count = 0
        for specimen, scales in radii_by_specimen.items():
            if "loose" in scales and "up" in scales:
                if np.isclose(scales["loose"], scales["up"]):
                    same_radius_specimens.append(specimen)
                else:
                    different_radius_count += 1

        LOGGER.info(
            "Multiscale radius check: loose/up different for %d specimens, equal for %d specimens.",
            different_radius_count,
            len(same_radius_specimens),
        )
        if same_radius_specimens:
            LOGGER.warning(
                "loose/up final_radius are equal for specimens: %s",
                ", ".join(same_radius_specimens[:20]),
            )
    LOGGER.info("Rendered %d/%d specimens", success, len(mesh_files))


if __name__ == "__main__":
    main()
