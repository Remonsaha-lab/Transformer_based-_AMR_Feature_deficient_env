#!/usr/bin/env python3
"""Generate project-20 semantic sidecars for DIODE RGB images."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFile

from generate_mask2former_labels import (
    ADE_TO_PROJECT20,
    PROJECT20_COLORS,
    colorize_labels,
    make_ade_palette,
    semantic_confidence_map,
)


def remap_to_project20(ade_labels: np.ndarray) -> np.ndarray:
    labels = np.full_like(ade_labels, 19, dtype=np.int32)
    for ade_id, project_id in ADE_TO_PROJECT20.items():
        labels[ade_labels == ade_id] = project_id
    return labels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--model-name", default="facebook/mask2former-swin-large-ade-semantic")
    parser.add_argument("--device", choices=["cuda", "cpu"], default=None)
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    images = sorted(
        path for path in args.input_root.rglob("*.png")
        if "_depth" not in path.name and "_semantic" not in path.name)
    if args.limit > 0:
        images = images[:args.limit]
    if not images:
        raise SystemExit(f"No RGB PNG files found below {args.input_root}")

    from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

    processor = AutoImageProcessor.from_pretrained(args.model_name)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(args.model_name).eval().to(device)
    palette = make_ade_palette(len(PROJECT20_COLORS))
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    completed = 0
    for image_path in images:
        label_path = image_path.with_name(f"{image_path.stem}_semantic.npy")
        confidence_path = image_path.with_name(f"{image_path.stem}_semantic_confidence.npy")
        if args.skip_existing and label_path.is_file() and confidence_path.is_file():
            continue
        image = Image.open(image_path).convert("RGB")
        target_size = (image.height, image.width)
        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.inference_mode():
            outputs = model(**inputs)
        ade_labels = processor.post_process_semantic_segmentation(
            outputs, target_sizes=[target_size])[0].cpu().numpy()
        labels = remap_to_project20(ade_labels)
        confidence = semantic_confidence_map(outputs, target_size)
        np.save(label_path, labels)
        np.save(confidence_path, confidence)
        Image.fromarray(colorize_labels(labels, PROJECT20_COLORS)).save(
            image_path.with_name(f"{image_path.stem}_semantic.png"))
        completed += 1
        print(f"[{completed}/{len(images)}] {image_path}")
        if device == "cuda":
            torch.cuda.empty_cache()

    print(f"Generated sidecars for {completed} images under {args.input_root}.")


if __name__ == "__main__":
    main()
