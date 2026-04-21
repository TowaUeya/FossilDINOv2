from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from src.utils.io import ensure_dir, group_renders_by_specimen, list_image_files, set_seed, setup_logging
from src.utils.vision import build_transform, load_image_tensor, resolve_device

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract DINOv2 features from rendered images.")
    parser.add_argument("--renders", type=Path, required=True, help="Directory containing render PNG files")
    parser.add_argument("--out", type=Path, required=True, help="Directory to save per-specimen [V,D] features")
    parser.add_argument("--model", type=str, default="dinov2_vits14")
    parser.add_argument("--device", type=str, default="auto", help="cuda/cpu/auto")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--crop-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_model(model_name: str, device: torch.device) -> torch.nn.Module:
    model = torch.hub.load("facebookresearch/dinov2", model_name)
    model.eval()
    model.to(device)
    return model


def main() -> None:
    args = parse_args()
    setup_logging()
    set_seed(args.seed)
    ensure_dir(args.out)

    render_files = list_image_files(args.renders)
    if not render_files:
        LOGGER.warning("No render images found: %s", args.renders)
        return

    grouped = group_renders_by_specimen(render_files, root_dir=args.renders)
    transform = build_transform(args.image_size, args.crop_size)
    device = resolve_device(args.device)
    model = load_model(args.model, device)
    LOGGER.info("Using device=%s model=%s", device, args.model)

    for sid, image_paths in tqdm(grouped.items(), desc="Extracting"):
        feats = []
        batch_tensors = []
        valid_paths = []

        for ip in image_paths:
            try:
                batch_tensors.append(load_image_tensor(ip, transform))
                valid_paths.append(ip)
            except Exception as e:
                LOGGER.exception("Failed to read image %s: %s", ip, e)

        if not batch_tensors:
            LOGGER.warning("No valid images for specimen %s", sid)
            continue

        with torch.inference_mode():
            for idx in range(0, len(batch_tensors), args.batch_size):
                bt = torch.stack(batch_tensors[idx : idx + args.batch_size]).to(device)
                out = model(bt)
                feats.append(out.detach().cpu().numpy())

        feature_arr = np.concatenate(feats, axis=0)
        out_path = args.out / f"{sid}.npy"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, feature_arr.astype(np.float32))

    LOGGER.info("Feature extraction finished for %d specimens", len(grouped))


if __name__ == "__main__":
    main()
