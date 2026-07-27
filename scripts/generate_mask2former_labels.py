#!/usr/bin/env python3
"""Generate Mask2Former semantic labels for the custom RGB-D scenes.

It uses the ADE20K semantic Mask2Former checkpoint. The default output uses a
combined indoor/outdoor target taxonomy suitable for the full DIODE dataset.
The existing outdoor-only taxonomy, full ADE20K, and the old 20-class project
mapping remain available as optional schemes.

  semantic_labels_mask2former_indoor_outdoor/<scene_id>_semantic.npy  int32 class IDs
  semantic_labels_mask2former_indoor_outdoor/<scene_id>_semantic.png  colour preview

Example:
  python generate_mask2former_labels.py --limit 1

The default checkpoint is comparatively large.  It is intended for checking
Mask2Former quality, rather than being a fast replacement for SegFormer.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFile


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = SCRIPT_DIR.parent / "Depth-Anything-V2" / "RGBD_data_custome"


# Project's compact labels.  They are retained as an optional output scheme
# for the existing RGB-D point-cloud pipeline.
PROJECT20_CLASS_NAMES = [
    "wall", "floor", "ceiling", "ground", "road", "sky", "building",
    "tree", "vegetation", "water", "mountain", "vehicle", "person",
    "chair", "table", "sofa", "cabinet", "window", "door", "other",
]

PROJECT20_COLORS = np.array([
    [120, 120, 120], [180, 120, 120], [120, 180, 180], [80, 160, 80],
    [128, 64, 128], [70, 130, 180], [70, 70, 70], [0, 128, 0],
    [107, 142, 35], [0, 191, 255], [139, 137, 137], [220, 20, 60],
    [255, 0, 0], [255, 160, 0], [204, 153, 51], [196, 128, 128],
    [150, 100, 80], [0, 255, 255], [190, 153, 153], [0, 0, 0],
], dtype=np.uint8)

# Outdoor target classes. Pixel value 255 is reserved for unmapped ADE20K
# classes. These IDs are retained in the combined taxonomy so old outdoor
# labels remain compatible.
OUTDOOR_IGNORE_ID = 255
OUTDOOR_CLASS_NAMES = [
    "wall", "building", "sky", "road", "sidewalk_path",
    "ground_soil", "grass_field", "tree", "vegetation_flower", "water",
    "mountain_rock", "vehicle", "person", "fence_railing", "bridge_pier",
    "pole_streetlight", "sign", "traffic_light",
]
OUTDOOR_CLASS_COLORS = np.array([
    [145, 110, 85], [90, 90, 90], [70, 130, 180], [128, 64, 128],
    [244, 35, 232], [130, 95, 60], [107, 142, 35], [0, 110, 0],
    [40, 170, 60], [0, 191, 255], [139, 137, 137], [220, 20, 60],
    [255, 0, 0], [190, 153, 153], [150, 100, 80], [153, 153, 153],
    [255, 255, 0], [250, 170, 30],
], dtype=np.uint8)

# ADE20K zero-based label IDs produced by the selected Mask2Former checkpoint.
ADE_TO_OUTDOOR = {
    0: 0,                         # wall
    1: 1, 25: 1, 48: 1, 79: 1, 84: 1,  # building / house / skyscraper / hovel / tower
    2: 2,                         # sky
    6: 3,                         # road
    11: 4, 52: 4,                 # sidewalk / path
    13: 5, 91: 5, 94: 5,          # earth / dirt track / land
    9: 6, 29: 6,                  # grass / field
    4: 7, 72: 7,                  # tree / palm
    17: 8, 66: 8,                 # plant / flower
    21: 9, 26: 9, 60: 9, 104: 9, 109: 9, 113: 9, 128: 9,  # water classes
    16: 10, 34: 10, 68: 10,       # mountain / rock / hill
    20: 11, 76: 11, 80: 11, 83: 11, 90: 11, 102: 11, 103: 11,
    116: 11, 122: 11, 127: 11,    # car, boat, bus, truck, airplane, van, ship, etc.
    12: 12,                       # person
    32: 13, 38: 13,               # fence / railing
    61: 14, 140: 14,              # bridge / pier
    87: 15, 93: 15,               # streetlight / pole
    43: 16, 123: 16,              # signboard / trade name
    136: 17,                      # traffic light
}

# Outdoor IDs (0-17) are retained exactly as above. Indoor-specific classes
# are appended. ADE20K has one ``wall`` category, so wall is intentionally
# shared by indoor and outdoor scenes instead of guessing its context.
INDOOR_OUTDOOR_CLASS_NAMES = OUTDOOR_CLASS_NAMES + [
    "floor", "ceiling", "bed", "window", "door", "table",
    "curtain_blind", "chair", "sofa", "cabinet_storage", "shelf_bookcase",
    "desk", "lamp_light", "mirror", "rug", "books", "kitchen_fixture",
    "appliance", "bathroom_fixture", "picture_decor", "indoor_furnishing",
]

INDOOR_OUTDOOR_CLASS_COLORS = np.vstack((
    OUTDOOR_CLASS_COLORS,
    np.array([
        [180, 120, 120], [120, 180, 180], [150, 80, 140], [0, 255, 255],
        [100, 50, 30], [204, 153, 51], [180, 150, 200], [255, 160, 0],
        [196, 128, 128], [80, 140, 140], [160, 120, 70], [130, 90, 50],
        [255, 230, 120], [180, 220, 230], [220, 80, 180], [110, 80, 40],
        [230, 140, 60], [100, 180, 220], [120, 220, 220], [220, 180, 120],
        [170, 170, 170],
    ], dtype=np.uint8),
))

# ADE20K zero-based labels for the indoor classes above. Classes already
# represented by outdoor targets (wall, person, and water) stay shared.
ADE_TO_INDOOR_OUTDOOR = {
    **ADE_TO_OUTDOOR,
    3: 18,                         # floor
    5: 19,                         # ceiling
    7: 20,                         # bed
    8: 21, 130: 21, 141: 21, 143: 21, 147: 21,  # window / screens / glass
    14: 22, 58: 22,                # door / screen door
    15: 23, 64: 23, 70: 23,        # table / coffee table / countertop
    18: 24, 63: 24,                # curtain / blind
    19: 25, 30: 25, 31: 25, 75: 25, 110: 25,  # chair variants
    23: 26, 97: 26,                # sofa / ottoman
    10: 27, 35: 27, 44: 27, 55: 27, 99: 27,    # storage furniture
    62: 28,                        # bookcase
    33: 29,                        # desk
    36: 30, 81: 30, 82: 30, 85: 30, 134: 30,   # lamps and lights
    27: 31,                        # mirror
    28: 32,                        # rug
    67: 33,                        # book
    45: 34, 56: 34, 71: 34, 73: 34, 77: 34, 118: 34,  # kitchen fixtures
    49: 35, 50: 35, 74: 35, 89: 35, 107: 35, 124: 35, 129: 35,  # appliances
    51: 36, 57: 36, 65: 36, 78: 36, 105: 36, 145: 36, 146: 36,  # bathroom
    22: 37, 100: 37, 132: 37, 144: 37, 148: 37, 149: 37,  # decor
    39: 38, 40: 38, 41: 38, 54: 38, 69: 38,
    92: 38, 98: 38, 108: 38, 111: 38, 112: 38, 115: 38, 119: 38,
    120: 38, 125: 38, 126: 38, 131: 38, 137: 38, 138: 38, 142: 38,
}

ADE_TO_PROJECT20: dict[int, int] = {}
for ade_id in [0, 32, 38, 42, 93, 95]:
    ADE_TO_PROJECT20[ade_id] = 0
for ade_id in [3, 28]:
    ADE_TO_PROJECT20[ade_id] = 1
for ade_id in [5, 86, 106]:
    ADE_TO_PROJECT20[ade_id] = 2
for ade_id in [9, 13, 29, 46, 94, 101]:
    ADE_TO_PROJECT20[ade_id] = 3
for ade_id in [6, 11, 52, 53, 54, 59, 91, 96, 121]:
    ADE_TO_PROJECT20[ade_id] = 4
ADE_TO_PROJECT20[2] = 5
for ade_id in [1, 25, 48, 51, 61, 79, 84, 88, 114, 140]:
    ADE_TO_PROJECT20[ade_id] = 6
for ade_id in [4, 72]:
    ADE_TO_PROJECT20[ade_id] = 7
for ade_id in [17, 66]:
    ADE_TO_PROJECT20[ade_id] = 8
for ade_id in [21, 26, 60, 104, 109, 113, 128]:
    ADE_TO_PROJECT20[ade_id] = 9
for ade_id in [16, 34, 68]:
    ADE_TO_PROJECT20[ade_id] = 10
for ade_id in [20, 76, 80, 83, 90, 102, 103, 116, 127]:
    ADE_TO_PROJECT20[ade_id] = 11
ADE_TO_PROJECT20[12] = 12
for ade_id in [19, 30, 31, 69, 75, 110]:
    ADE_TO_PROJECT20[ade_id] = 13
for ade_id in [15, 33, 45, 56, 64, 70, 73, 77]:
    ADE_TO_PROJECT20[ade_id] = 14
for ade_id in [23, 97]:
    ADE_TO_PROJECT20[ade_id] = 15
for ade_id in [10, 24, 35, 44, 50, 62, 71, 99, 107, 118, 129]:
    ADE_TO_PROJECT20[ade_id] = 16
ADE_TO_PROJECT20[8] = 17
for ade_id in [14, 58]:
    ADE_TO_PROJECT20[ade_id] = 18


def remap_to_project20(ade_labels: np.ndarray) -> np.ndarray:
    """Map 150 ADE20K labels to the project's 20 broad point-cloud classes."""
    labels = np.full_like(ade_labels, 19, dtype=np.int32)
    for ade_id, project_id in ADE_TO_PROJECT20.items():
        labels[ade_labels == ade_id] = project_id
    return labels


def remap_to_outdoor(ade_labels: np.ndarray) -> np.ndarray:
    """Keep only outdoor ADE20K categories; label all other pixels as ignore."""
    labels = np.full_like(ade_labels, OUTDOOR_IGNORE_ID, dtype=np.int32)
    for ade_id, outdoor_id in ADE_TO_OUTDOOR.items():
        labels[ade_labels == ade_id] = outdoor_id
    return labels


def remap_to_indoor_outdoor(ade_labels: np.ndarray) -> np.ndarray:
    """Map ADE20K predictions to the shared DIODE indoor/outdoor taxonomy."""
    labels = np.full_like(ade_labels, OUTDOOR_IGNORE_ID, dtype=np.int32)
    for ade_id, class_id in ADE_TO_INDOOR_OUTDOOR.items():
        labels[ade_labels == ade_id] = class_id
    return labels


def make_ade_palette(num_classes: int) -> np.ndarray:
    """Return a stable, high-contrast palette for the full ADE20K label map."""
    # A deterministic palette makes the same ADE class use the same colour in
    # every image, without adding an OpenCV dependency.
    rng = np.random.default_rng(2026)
    return rng.integers(45, 256, size=(num_classes, 3), dtype=np.uint8)


def colorize_labels(labels: np.ndarray, palette: np.ndarray, ignore_id: int | None = None) -> np.ndarray:
    preview = palette[np.clip(labels, 0, len(palette) - 1)]
    if ignore_id is not None:
        preview[labels == ignore_id] = [0, 0, 0]
    return preview


def colorize_panoptic_segments(segment_ids: np.ndarray) -> np.ndarray:
    """Create a stable colour preview where every panoptic segment is distinct."""
    max_id = int(segment_ids.max()) if segment_ids.size else 0
    palette = make_ade_palette(max_id + 1)
    preview = palette[segment_ids]
    preview[segment_ids == 0] = [0, 0, 0]
    return preview


def semantic_confidence_map(outputs, target_size: tuple[int, int]) -> np.ndarray:
    """Return confidence of the winning ADE20K class at each output pixel.

    Mask2Former predicts query class probabilities and query masks.  Combining
    them gives per-class semantic logits; softmax over classes yields a useful
    normalized per-pixel confidence in the range [0, 1].
    """
    class_probabilities = outputs.class_queries_logits.softmax(dim=-1)[..., :-1]
    mask_probabilities = outputs.masks_queries_logits.sigmoid()
    class_logits = torch.einsum("bqc,bqhw->bchw", class_probabilities, mask_probabilities)
    class_logits = F.interpolate(class_logits, size=target_size, mode="bilinear", align_corners=False)
    # Avoid materialising a second full (classes x height x width) softmax
    # tensor: full-resolution RGB-D images can be large.
    winning_logit = class_logits.amax(dim=1)
    log_normalizer = torch.logsumexp(class_logits, dim=1)
    confidence = (winning_logit - log_normalizer).exp()[0]
    return confidence.float().cpu().numpy()


def save_confidence_preview(confidence: np.ndarray, path: Path) -> None:
    """Save a grayscale confidence preview; white is high confidence."""
    preview = np.clip(confidence * 255, 0, 255).astype(np.uint8)
    Image.fromarray(preview, mode="L").save(path)


def load_scenes(data_root: Path, intrinsics_json: Path | None) -> list[dict]:
    intrinsics_json = intrinsics_json or data_root / "scene_camera_intrinsics.json"
    if intrinsics_json.is_file():
        data = json.loads(intrinsics_json.read_text())
        return [
            {"scene_id": scene["scene_id"], "image_path": data_root / scene["image"]["path"]}
            for scene in data["scenes"]
        ]

    rgb_dir = data_root / "RGB_image"
    return [
        {"scene_id": image_path.name, "image_path": image_path}
        for image_path in sorted(rgb_dir.iterdir()) if image_path.is_file()
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Mask2Former semantic and panoptic labels for custom RGB-D scenes."
    )
    parser.add_argument(
        "--data_root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Dataset root containing RGB_image and scene_camera_intrinsics.json.",
    )
    parser.add_argument("--intrinsics_json", type=Path, default=None)
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=None,
        help="Output directory. Default depends on --label_scheme.",
    )
    parser.add_argument(
        "--model_name",
        default="facebook/mask2former-swin-large-ade-semantic",
        help="Hugging Face Mask2Former semantic-segmentation checkpoint.",
    )
    parser.add_argument(
        "--panoptic_model_name",
        default="facebook/mask2former-swin-large-coco-panoptic",
        help="Hugging Face Mask2Former panoptic-segmentation checkpoint.",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=["semantic", "panoptic"],
        default=["semantic", "panoptic"],
        help="Tasks to run. Default: semantic panoptic.",
    )
    parser.add_argument("--limit", type=int, default=-1, help="Process at most this many scenes.")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument(
        "--label_scheme",
        choices=["indoor_outdoor", "outdoor", "ade150", "project20"],
        default="indoor_outdoor",
        help=("indoor_outdoor keeps shared indoor and outdoor targets (default); outdoor keeps "
              "only outdoor targets; ade150 keeps all ADE20K classes; project20 uses the old "
              "point-cloud mapping."),
    )
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()

    default_dir = {
        "indoor_outdoor": "semantic_labels_mask2former_indoor_outdoor",
        "outdoor": "semantic_labels_mask2former_outdoor",
        "ade150": "semantic_labels_mask2former_ade150",
        "project20": "semantic_labels_mask2former",
    }[args.label_scheme]
    semantic_dir = args.out_dir or args.data_root / default_dir
    panoptic_dir = args.data_root / "panoptic_labels_mask2former_coco"
    if "semantic" in args.tasks:
        semantic_dir.mkdir(parents=True, exist_ok=True)
    if "panoptic" in args.tasks:
        panoptic_dir.mkdir(parents=True, exist_ok=True)

    scenes = load_scenes(args.data_root, args.intrinsics_json)
    if args.limit > 0:
        scenes = scenes[:args.limit]
    if args.skip_existing:
        scenes = [
            scene for scene in scenes
            if not (
                ("semantic" not in args.tasks or (
                    (semantic_dir / f"{scene['scene_id']}_semantic.npy").is_file()
                    and (semantic_dir / f"{scene['scene_id']}_semantic_confidence.npy").is_file()
                ))
                and ("panoptic" not in args.tasks or (
                    (panoptic_dir / f"{scene['scene_id']}_panoptic.npy").is_file()
                    and (panoptic_dir / f"{scene['scene_id']}_panoptic_confidence.npy").is_file()
                ))
            )
        ]
    if not scenes:
        print("No scenes to process.")
        return

    try:
        from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: transformers. Install it with:\n"
            "  pip install 'transformers>=4.26' torch pillow\n"
            "Mask2Former may also require timm: pip install timm"
        ) from exc

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda was selected, but CUDA is not available.")

    print(f"Found {len(scenes)} scenes.")
    semantic_processor = semantic_model = None
    panoptic_processor = panoptic_model = None
    if "semantic" in args.tasks:
        print(f"Loading semantic Mask2Former: {args.model_name}")
        semantic_processor = AutoImageProcessor.from_pretrained(args.model_name)
        semantic_model = Mask2FormerForUniversalSegmentation.from_pretrained(args.model_name).eval().to(args.device)

        ignore_id = None
        if args.label_scheme == "ade150":
            id2label = {int(key): value for key, value in semantic_model.config.id2label.items()}
            class_names = [id2label[key] for key in range(max(id2label) + 1)]
            palette = make_ade_palette(len(class_names))
        elif args.label_scheme == "outdoor":
            class_names, palette, ignore_id = OUTDOOR_CLASS_NAMES, OUTDOOR_CLASS_COLORS, OUTDOOR_IGNORE_ID
        elif args.label_scheme == "indoor_outdoor":
            class_names, palette, ignore_id = (
                INDOOR_OUTDOOR_CLASS_NAMES,
                INDOOR_OUTDOOR_CLASS_COLORS,
                OUTDOOR_IGNORE_ID,
            )
        else:
            class_names, palette = PROJECT20_CLASS_NAMES, PROJECT20_COLORS
        label_metadata = {str(key): value for key, value in enumerate(class_names)}
        if ignore_id is not None:
            label_metadata[str(ignore_id)] = "ignore_indoor_or_unmapped"
        (semantic_dir / "class_labels.json").write_text(json.dumps(label_metadata, indent=2))

    if "panoptic" in args.tasks:
        print(f"Loading panoptic Mask2Former: {args.panoptic_model_name}")
        panoptic_processor = AutoImageProcessor.from_pretrained(args.panoptic_model_name)
        panoptic_model = Mask2FormerForUniversalSegmentation.from_pretrained(args.panoptic_model_name).eval().to(args.device)

    ImageFile.LOAD_TRUNCATED_IMAGES = True
    started = time.time()
    for index, scene in enumerate(scenes, start=1):
        scene_id = scene["scene_id"]
        try:
            image = Image.open(scene["image_path"]).convert("RGB")
            # target_sizes uses (height, width), whereas PIL uses (width, height).
            target_size = (image.height, image.width)
            if "semantic" in args.tasks:
                inputs = semantic_processor(images=image, return_tensors="pt").to(args.device)
                with torch.inference_mode():
                    outputs = semantic_model(**inputs)
                ade_labels = semantic_processor.post_process_semantic_segmentation(outputs, target_sizes=[target_size])[0].cpu().numpy()
                confidence = semantic_confidence_map(outputs, target_size)
                if args.label_scheme == "ade150":
                    labels = ade_labels.astype(np.int32)
                elif args.label_scheme == "outdoor":
                    labels = remap_to_outdoor(ade_labels)
                elif args.label_scheme == "indoor_outdoor":
                    labels = remap_to_indoor_outdoor(ade_labels)
                else:
                    labels = remap_to_project20(ade_labels)
                np.save(semantic_dir / f"{scene_id}_semantic.npy", labels)
                Image.fromarray(colorize_labels(labels, palette, ignore_id)).save(semantic_dir / f"{scene_id}_semantic.png")
                np.save(semantic_dir / f"{scene_id}_semantic_confidence.npy", confidence)
                save_confidence_preview(confidence, semantic_dir / f"{scene_id}_semantic_confidence.png")

            if "panoptic" in args.tasks:
                inputs = panoptic_processor(images=image, return_tensors="pt").to(args.device)
                with torch.inference_mode():
                    outputs = panoptic_model(**inputs)
                panoptic = panoptic_processor.post_process_panoptic_segmentation(outputs, target_sizes=[target_size])[0]
                segment_ids = panoptic["segmentation"].cpu().numpy().astype(np.int32)
                np.save(panoptic_dir / f"{scene_id}_panoptic.npy", segment_ids)
                Image.fromarray(colorize_panoptic_segments(segment_ids)).save(panoptic_dir / f"{scene_id}_panoptic.png")
                segments = []
                for segment in panoptic["segments_info"]:
                    item = dict(segment)
                    item["label"] = panoptic_model.config.id2label[item["label_id"]]
                    segments.append(item)
                (panoptic_dir / f"{scene_id}_panoptic_segments.json").write_text(json.dumps(segments, indent=2))
                confidence = np.zeros(segment_ids.shape, dtype=np.float32)
                for segment in segments:
                    confidence[segment_ids == int(segment["id"])] = float(segment["score"])
                np.save(panoptic_dir / f"{scene_id}_panoptic_confidence.npy", confidence)
                save_confidence_preview(confidence, panoptic_dir / f"{scene_id}_panoptic_confidence.png")
            print(f"  [{index:3d}/{len(scenes)}] completed {scene_id}")
        except Exception as exc:
            print(f"  [{index:3d}/{len(scenes)}] WARNING: skipped {scene_id}: {exc}")
        finally:
            if args.device == "cuda":
                torch.cuda.empty_cache()

    print(f"Done in {time.time() - started:.1f}s.")
    if "semantic" in args.tasks:
        print(f"Semantic labels: {semantic_dir}")
    if "panoptic" in args.tasks:
        print(f"Panoptic labels: {panoptic_dir}")


if __name__ == "__main__":
    main()
