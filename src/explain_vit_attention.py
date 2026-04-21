from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from src.utils.explain import attention_rollout, grad_attention_rollout, to_patch_heatmap
from src.utils.io import ensure_dir, group_renders_by_specimen, list_image_files, load_ids, specimen_id_from_render
from src.utils.vision import build_transform, forward_embedding, load_dinov2_model, load_image_tensor, resolve_device


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Explain DINOv2 attention for embedding formation")
    p.add_argument("--renders", type=Path, required=True)
    p.add_argument("--features", type=Path, required=False, default=None)
    p.add_argument("--emb", type=Path, required=True)
    p.add_argument("--ids", type=Path, required=True)
    p.add_argument("--clusters", type=Path, required=False, default=None)
    p.add_argument("--specimen_id", type=str, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--model", type=str, default="dinov2_vits14")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--crop-size", type=int, default=224)
    return p.parse_args()


def _collect_attn_modules(model: torch.nn.Module) -> list[torch.nn.Module]:
    blocks = getattr(model, "blocks", None)
    if blocks is None:
        raise RuntimeError("Model has no blocks attribute for attention rollout")
    return [b.attn for b in blocks if hasattr(b, "attn")]


def main() -> None:
    args = parse_args()
    ensure_dir(args.out)

    device = resolve_device(args.device)
    model = load_dinov2_model(args.model, device)
    transform = build_transform(args.image_size, args.crop_size)

    render_files = list_image_files(args.renders)
    grouped = group_renders_by_specimen(render_files, root_dir=args.renders)
    if args.specimen_id not in grouped:
        raise ValueError(f"specimen_id not found: {args.specimen_id}")

    ids = load_ids(args.ids)
    embs = np.load(args.emb)
    sid_to_idx = {sid: i for i, sid in enumerate(ids)}
    z_specimen = torch.from_numpy(embs[sid_to_idx[args.specimen_id]]).to(device).float()
    z_specimen = F.normalize(z_specimen, dim=0).detach()

    image_paths = grouped[args.specimen_id]
    attn_modules = _collect_attn_modules(model)
    n_show = min(6, len(image_paths))

    fig_roll, axs_roll = plt.subplots(2, n_show, figsize=(3 * n_show, 6))
    fig_grad, axs_grad = plt.subplots(2, n_show, figsize=(3 * n_show, 6))

    for col, ip in enumerate(image_paths[:n_show]):
        x = load_image_tensor(ip, transform).unsqueeze(0).to(device)
        attn_maps: list[torch.Tensor] = []
        grads: list[torch.Tensor] = []

        def fwd_hook(_m, _in, out):
            a = out[1] if isinstance(out, tuple) else out
            attn_maps.append(a)
            a.retain_grad()

        hooks = [m.register_forward_hook(fwd_hook) for m in attn_modules]
        for p in model.parameters():
            p.requires_grad_(False)

        z_view = forward_embedding(model, x)
        z_view = F.normalize(z_view.squeeze(0), dim=0)
        score = F.cosine_similarity(z_view.unsqueeze(0), z_specimen.unsqueeze(0)).mean()
        model.zero_grad(set_to_none=True)
        score.backward()

        for a in attn_maps:
            grads.append(a.grad if a.grad is not None else torch.zeros_like(a))
        for h in hooks:
            h.remove()

        roll = attention_rollout(attn_maps)
        grad_roll = grad_attention_rollout(attn_maps, grads)

        n_tokens = int(np.sqrt(roll.shape[-1]))
        heat = to_patch_heatmap(roll[0], n_tokens)
        gheat = to_patch_heatmap(grad_roll[0], n_tokens)

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

    fig_roll.tight_layout()
    fig_grad.tight_layout()
    fig_roll.savefig(args.out / "attention_rollout.png", dpi=220)
    fig_grad.savefig(args.out / "grad_rollout_similarity_to_specimen.png", dpi=220)
    fig_roll.savefig(args.out / "attention_rollout_contact_sheet.png", dpi=220)
    fig_grad.savefig(args.out / "grad_rollout_contact_sheet.png", dpi=220)


if __name__ == "__main__":
    main()
