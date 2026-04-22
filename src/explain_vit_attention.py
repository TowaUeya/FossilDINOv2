from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from src.utils.explain import attention_rollout, grad_attention_rollout, to_patch_heatmap
from src.utils.io import ensure_dir, group_renders_by_specimen, list_image_files, load_ids
from src.utils.vision import build_transform, forward_embedding, load_dinov2_model, load_image_tensor, resolve_device

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Explain DINOv2 attention for embedding formation")
    p.add_argument("--renders", type=Path, required=True)
    p.add_argument("--features", type=Path, required=False, default=None)
    p.add_argument("--emb", type=Path, required=True)
    p.add_argument("--ids", type=Path, required=True)
    p.add_argument("--clusters", type=Path, required=False, default=None)
    p.add_argument(
        "--specimen_id",
        type=str,
        required=False,
        default=None,
        help="Target specimen ID. If omitted, all specimen IDs found in both renders and --ids are processed.",
    )
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--model", type=str, default="dinov2_vits14")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--crop-size", type=int, default=224)
    p.add_argument("--layers", type=str, choices=("all", "last"), default="all")
    p.add_argument(
        "--num-show",
        type=int,
        default=6,
        help="Number of views to visualize. If larger than available views, all available views are shown.",
    )
    return p.parse_args()


def _specimen_output_dir(base_out: Path, specimen_id: str) -> Path:
    parts = [p for p in PurePosixPath(specimen_id).parts if p not in ("", ".", "..")]
    if not parts:
        raise ValueError(f"Invalid specimen_id for output path: {specimen_id}")
    return base_out.joinpath(*parts)


def _collect_blocks(model: torch.nn.Module) -> list[torch.nn.Module]:
    blocks = getattr(model, "blocks", None)
    if blocks is None:
        raise RuntimeError("Model has no blocks attribute for attention rollout")
    out = [b for b in blocks if hasattr(b, "attn")]
    if not out:
        raise RuntimeError("Model blocks do not expose attention modules")
    return out


def _select_blocks_for_rollout(blocks: list[torch.nn.Module], layers: str) -> list[torch.nn.Module]:
    if layers == "last":
        return [blocks[-1]]
    return blocks


def _reset_block_attn_cache(blocks: list[torch.nn.Module]) -> None:
    for blk in blocks:
        blk.attn._last_attn_map = None
        blk.attn._last_attn_grad = None


def _unwrap_qkv(qkv: torch.Tensor, x: torch.Tensor, num_heads: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    bsz, n_tokens, dim = x.shape
    head_dim = dim // num_heads
    qkv = qkv.reshape(bsz, n_tokens, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    return q, k, v


def _make_attention_forward_wrapper(attn_obj: torch.nn.Module, original_forward: Any):
    del original_forward

    def wrapped_forward(*args: Any, **kwargs: Any) -> torch.Tensor:
        if len(args) > 0:
            x = args[0]
        else:
            x = kwargs.get("x", None)
        if x is None:
            raise ValueError("Attention wrapper received no input tensor.")

        attn_mask = kwargs.get("attn_mask", None)
        if attn_mask is not None and not getattr(attn_obj, "_warned_attn_mask_ignored", False):
            LOGGER.warning(
                "attn_mask was passed to wrapped attention, but explicit mask application is "
                "not implemented in explain wrapper. Continuing without mask."
            )
            attn_obj._warned_attn_mask_ignored = True

        bsz, n_tokens, dim = x.shape
        num_heads = int(getattr(attn_obj, "num_heads", 1))

        qkv = attn_obj.qkv(x)
        q, k, v = _unwrap_qkv(qkv, x, num_heads)

        q_norm = getattr(attn_obj, "q_norm", None)
        if q_norm is not None:
            q = q_norm(q)
        k_norm = getattr(attn_obj, "k_norm", None)
        if k_norm is not None:
            k = k_norm(k)

        scale = getattr(attn_obj, "scale", None)
        if scale is None:
            scale = (dim // num_heads) ** -0.5

        attn = (q @ k.transpose(-2, -1)) * scale
        attn = attn.softmax(dim=-1)

        attn_drop = getattr(attn_obj, "attn_drop", None)
        if attn_drop is not None:
            attn = attn_drop(attn)

        attn_obj._last_attn_map = attn
        attn_obj._last_attn_grad = None
        if attn.requires_grad:
            attn.retain_grad()

            def _save_grad(grad: torch.Tensor) -> None:
                attn_obj._last_attn_grad = grad

            attn.register_hook(_save_grad)

        x_out = (attn @ v).transpose(1, 2).reshape(bsz, n_tokens, dim)

        proj = getattr(attn_obj, "proj", None)
        if proj is not None:
            x_out = proj(x_out)
        proj_drop = getattr(attn_obj, "proj_drop", None)
        if proj_drop is not None:
            x_out = proj_drop(x_out)

        return x_out

    return wrapped_forward


def _install_attention_wrappers(blocks: list[torch.nn.Module]) -> list[tuple[torch.nn.Module, Any, object]]:
    restore_state: list[tuple[torch.nn.Module, Any, object]] = []
    for blk in blocks:
        attn = blk.attn
        original_forward = attn.forward
        had_fused = hasattr(attn, "fused_attn")
        old_fused = getattr(attn, "fused_attn", None)
        if had_fused:
            attn.fused_attn = False
        attn.forward = _make_attention_forward_wrapper(attn, original_forward)
        restore_state.append((attn, original_forward, old_fused if had_fused else _MISSING))
    return restore_state


def _restore_attention_wrappers(restore_state: list[tuple[torch.nn.Module, Any, object]]) -> None:
    for attn, original_forward, old_fused in restore_state:
        attn.forward = original_forward
        if old_fused is _MISSING:
            continue
        attn.fused_attn = old_fused


def _resolve_num_patches(model: torch.nn.Module) -> int | None:
    patch_embed = getattr(model, "patch_embed", None)
    if patch_embed is None:
        return None
    num_patches = getattr(patch_embed, "num_patches", None)
    if isinstance(num_patches, int) and num_patches > 0:
        return num_patches
    return None


def _cls_to_patch_tokens(cls_to_tokens: torch.Tensor, num_patches: int | None, image_path: Path) -> torch.Tensor:
    if num_patches is None:
        return cls_to_tokens

    available = cls_to_tokens.shape[-1]
    if available == num_patches:
        return cls_to_tokens

    if available > num_patches:
        LOGGER.warning(
            "Extra non-patch tokens detected for %s (available=%d, patches=%d). Taking last patch tokens.",
            image_path,
            available,
            num_patches,
        )
        return cls_to_tokens[..., -num_patches:]

    raise RuntimeError(
        f"Token count ({available}) smaller than expected patch count ({num_patches}) for {image_path}."
    )


def _infer_grid_size(n_patches: int, image_path: Path) -> int:
    grid = int(math.sqrt(n_patches))
    if grid * grid != n_patches:
        raise RuntimeError(
            f"Patch count is not a perfect square for {image_path}: {n_patches}. "
            "Please verify token filtering / image-size / patch-size settings."
        )
    return grid


_MISSING = object()


def main() -> None:
    args = parse_args()
    ensure_dir(args.out)

    device = resolve_device(args.device)
    model = load_dinov2_model(args.model, device)
    transform = build_transform(args.image_size, args.crop_size)

    blocks = _collect_blocks(model)
    rollout_blocks = _select_blocks_for_rollout(blocks, args.layers)
    restore_state = _install_attention_wrappers(rollout_blocks)
    num_patches = _resolve_num_patches(model)

    render_files = list_image_files(args.renders)
    grouped = group_renders_by_specimen(render_files, root_dir=args.renders)
    ids = load_ids(args.ids)
    embs = np.load(args.emb)
    sid_to_idx = {sid: i for i, sid in enumerate(ids)}
    if args.num_show < 1:
        raise ValueError("--num-show must be >= 1")

    if args.specimen_id is not None:
        target_specimen_ids = [args.specimen_id]
    else:
        target_specimen_ids = [sid for sid in ids if sid in grouped]

    if not target_specimen_ids:
        raise RuntimeError("No common specimen IDs found between --ids and --renders.")

    n_ok = 0
    n_skip = 0
    try:
        specimen_iter = tqdm(target_specimen_ids, desc="Specimens", unit="specimen")
        for specimen_id in specimen_iter:
            if specimen_id not in grouped:
                LOGGER.warning("Skipping %s: specimen_id not found in renders.", specimen_id)
                n_skip += 1
                continue
            if specimen_id not in sid_to_idx:
                LOGGER.warning("Skipping %s: specimen_id not found in ids.", specimen_id)
                n_skip += 1
                continue

            z_specimen = torch.from_numpy(embs[sid_to_idx[specimen_id]]).to(device).float()
            z_specimen = F.normalize(z_specimen, dim=0).detach()

            image_paths = grouped[specimen_id]
            n_show = min(args.num_show, len(image_paths))

            fig_w = max(4.0 * n_show, 8.0)
            fig_h = 8.0
            fig_roll, axs_roll = plt.subplots(2, n_show, figsize=(fig_w, fig_h), squeeze=False)
            fig_grad, axs_grad = plt.subplots(2, n_show, figsize=(fig_w, fig_h), squeeze=False)

            success_cols = 0
            view_iter = tqdm(
                enumerate(image_paths[:n_show]),
                total=n_show,
                desc=f"Views ({specimen_id})",
                unit="view",
                leave=False,
            )
            for col, ip in view_iter:
                x = load_image_tensor(ip, transform).unsqueeze(0).to(device)
                x.requires_grad_(True)
                _reset_block_attn_cache(rollout_blocks)

                for p in model.parameters():
                    p.requires_grad_(False)

                model.zero_grad(set_to_none=True)
                z_view = forward_embedding(model, x, enable_grad=True)
                z_view = F.normalize(z_view, dim=-1)
                score = F.cosine_similarity(z_view, z_specimen.unsqueeze(0), dim=-1).sum()
                score.backward()

                attn_maps: list[torch.Tensor] = []
                attn_grads: list[torch.Tensor] = []
                for blk in rollout_blocks:
                    attn_map = getattr(blk.attn, "_last_attn_map", None)
                    if attn_map is None:
                        continue

                    attn_grad = getattr(blk.attn, "_last_attn_grad", None)
                    if attn_grad is None and getattr(attn_map, "grad", None) is not None:
                        attn_grad = attn_map.grad
                    if attn_grad is None:
                        attn_grad = torch.zeros_like(attn_map)

                    attn_maps.append(attn_map)
                    attn_grads.append(attn_grad)

                if not attn_maps:
                    LOGGER.warning("No attention tensor extracted for view: %s. Skipping this view.", ip)
                    continue

                roll = attention_rollout(attn_maps)
                grad_roll = grad_attention_rollout(attn_maps, attn_grads)

                roll_tokens = _cls_to_patch_tokens(roll[0], num_patches, ip)
                grad_tokens = _cls_to_patch_tokens(grad_roll[0], num_patches, ip)
                grid = _infer_grid_size(int(roll_tokens.shape[-1]), ip)

                heat = to_patch_heatmap(roll_tokens, grid)
                gheat = to_patch_heatmap(grad_tokens, grid)

                img = plt.imread(ip)
                axs_roll[0, col].imshow(img)
                axs_roll[0, col].set_title(Path(ip).name)
                axs_roll[0, col].axis("off")
                axs_roll[1, col].imshow(img)
                axs_roll[1, col].imshow(heat, cmap="jet", alpha=0.45, extent=(0, img.shape[1], img.shape[0], 0))
                axs_roll[1, col].axis("off")

                axs_grad[0, col].imshow(img)
                axs_grad[0, col].set_title(Path(ip).name)
                axs_grad[0, col].axis("off")
                axs_grad[1, col].imshow(img)
                axs_grad[1, col].imshow(gheat, cmap="jet", alpha=0.45, extent=(0, img.shape[1], img.shape[0], 0))
                axs_grad[1, col].axis("off")
                success_cols += 1

            if success_cols == 0:
                LOGGER.warning(
                    "Skipping %s: no valid attention map extracted for any view. ",
                    specimen_id,
                )
                plt.close(fig_roll)
                plt.close(fig_grad)
                n_skip += 1
                continue

            fig_roll.tight_layout()
            fig_grad.tight_layout()

            specimen_out = _specimen_output_dir(args.out, specimen_id)
            ensure_dir(specimen_out)
            fig_roll.savefig(specimen_out / "attention_rollout.png", dpi=220)
            fig_grad.savefig(specimen_out / "grad_rollout_similarity_to_specimen.png", dpi=220)
            plt.close(fig_roll)
            plt.close(fig_grad)
            n_ok += 1
            LOGGER.info("Saved ViT attention explanations for %s to %s", specimen_id, specimen_out)
            specimen_iter.set_postfix(success=n_ok, skipped=n_skip)
    finally:
        _restore_attention_wrappers(restore_state)

    if n_ok == 0:
        raise RuntimeError(
            "No valid attention maps were saved for any specimen. Please check timm version/model attention outputs."
        )

    LOGGER.info("Completed ViT attention explanation. success=%d skipped=%d", n_ok, n_skip)


if __name__ == "__main__":
    main()
