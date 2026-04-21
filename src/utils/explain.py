from __future__ import annotations

from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F


def _row_norm(attn: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    denom = attn.sum(dim=-1, keepdim=True).clamp_min(eps)
    return attn / denom


def attention_rollout(attn_maps: list[torch.Tensor], discard_cls_to_cls: bool = True) -> torch.Tensor:
    # attn_maps: list[(B,H,T,T)]
    if not attn_maps:
        raise ValueError("attn_maps is empty")

    b, _, t, _ = attn_maps[0].shape
    eye = torch.eye(t, device=attn_maps[0].device).unsqueeze(0).expand(b, t, t)
    rollout = eye.clone()

    for attn in attn_maps:
        a = attn.mean(dim=1)
        a_hat = _row_norm(a + eye)
        rollout = a_hat @ rollout

    if discard_cls_to_cls:
        return rollout[:, 0, 1:]
    return rollout[:, 0, :]


def grad_attention_rollout(
    attn_maps: list[torch.Tensor],
    grads: list[torch.Tensor],
) -> torch.Tensor:
    if len(attn_maps) != len(grads):
        raise ValueError("attn_maps and grads must have same length")
    if not attn_maps:
        raise ValueError("attn_maps is empty")

    b, _, t, _ = attn_maps[0].shape
    eye = torch.eye(t, device=attn_maps[0].device).unsqueeze(0).expand(b, t, t)
    rollout = eye.clone()

    for attn, grad in zip(attn_maps, grads):
        w = torch.relu(grad)
        a = (attn * w).mean(dim=1)
        a_hat = _row_norm(a + eye)
        rollout = a_hat @ rollout

    return rollout[:, 0, 1:]


def to_patch_heatmap(cls_to_patch: torch.Tensor, n_patches_side: int) -> np.ndarray:
    heat = cls_to_patch.reshape(n_patches_side, n_patches_side)
    heat = heat.detach().cpu().numpy()
    heat = heat - heat.min()
    maxv = heat.max()
    if maxv > 0:
        heat = heat / maxv
    return heat


def cosine_scalar(z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
    return F.cosine_similarity(z_a, z_b, dim=-1).mean()
