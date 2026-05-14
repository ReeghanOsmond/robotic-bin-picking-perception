from pathlib import Path
import argparse
import csv
import json
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / "scripts"))

from src.bop_io import (
    load_json,
    load_rgb,
    load_depth,
    load_mask,
    get_camera_intrinsics,
    get_depth_scale,
)

from analyze_scene_objects import (
    compute_object_features,
    add_pickability_scores,
)


DEFAULT_EXAMPLES = [
    (0, 0),
    (0, 1),
    (0, 2),
    (5, 0),
    (5, 1),
]


COLORBLIND_FRIENDLY_COLORS = [
    (230, 159, 0),    # orange
    (213, 94, 0),     # vermillion
    (0, 158, 115),    # green
    (204, 121, 167),  # purple
    (240, 228, 66),   # yellow
    (90, 90, 90),     # dark gray
    (166, 118, 29),   # brown
    (102, 17, 0),     # dark red-brown
    (128, 128, 128),  # gray
    (255, 180, 120),  # light orange
]


class PickabilityRanker(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.network(x)


def ensure_2d_mask(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask)

    if mask.ndim == 3:
        mask = np.squeeze(mask)

    if mask.ndim != 2:
        raise ValueError(f"Expected 2D mask, got shape {mask.shape}")

    return mask.astype(bool)


def parse_object_index(mask_path: Path) -> int:
    return int(mask_path.stem.split("_")[-1])


def get_color(index: int) -> tuple[int, int, int]:
    return COLORBLIND_FRIENDLY_COLORS[index % len(COLORBLIND_FRIENDLY_COLORS)]


def text_color_for_background(color: tuple[int, int, int]) -> tuple[int, int, int]:
    r, g, b = color
    luminance = 0.299 * r + 0.587 * g + 0.114 * b

    if luminance > 140:
        return (0, 0, 0)

    return (255, 255, 255)


def safe_float(value, default=np.nan) -> float:
    if value is None:
        return default

    value = str(value).strip()

    if value == "":
        return default

    try:
        return float(value)
    except ValueError:
        return default


def load_class_mapping(mapping_path: Path) -> dict[int, int]:
    """
    Returns:
        YOLO class index -> original dataset part_id
    """
    if not mapping_path.exists():
        return {}

    with mapping_path.open("r") as f:
        mapping = json.load(f)

    class_index_to_part_id = mapping.get("class_index_to_part_id", {})

    return {
        int(class_index): int(part_id)
        for class_index, part_id in class_index_to_part_id.items()
    }


def load_pickability_ranker(model_path: Path, device: torch.device):
    if not model_path.exists():
        raise FileNotFoundError(
            f"Pickability ranker not found: {model_path}\n"
            "Run this first:\n"
            "python scripts\\train_pickability_ranker.py"
        )

    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(model_path, map_location=device)

    feature_columns = checkpoint["feature_columns"]
    medians = np.array(checkpoint["medians"], dtype=np.float32)
    mean = np.array(checkpoint["mean"], dtype=np.float32)
    std = np.array(checkpoint["std"], dtype=np.float32)

    model = PickabilityRanker(input_dim=len(feature_columns)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, feature_columns, medians, mean, std


def apply_preprocessor(
    x: np.ndarray,
    medians: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    x = np.where(np.isnan(x), medians, x)
    x = (x - mean) / std
    return x.astype(np.float32)


def build_feature_matrix(features: list[dict], feature_columns: list[str]) -> np.ndarray:
    rows = []

    for feature in features:
        values = []

        for column in feature_columns:
            values.append(safe_float(feature.get(column), default=np.nan))

        rows.append(values)

    return np.array(rows, dtype=np.float32)


def predict_ranker_scores(
    ranker_model: PickabilityRanker,
    features: list[dict],
    feature_columns: list[str],
    medians: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    device: torch.device,
    score_key: str,
    rank_key: str,
    top_key: str,
) -> list[dict]:
    if not features:
        return features

    x_raw = build_feature_matrix(features, feature_columns)
    x = apply_preprocessor(x_raw, medians, mean, std)

    with torch.no_grad():
        x_tensor = torch.tensor(x, dtype=torch.float32).to(device)
        predictions = ranker_model(x_tensor).detach().cpu().numpy().reshape(-1)

    for feature, prediction in zip(features, predictions):
        feature[score_key] = float(prediction)

    features = sorted(
        features,
        key=lambda feature: feature[score_key],
        reverse=True,
    )

    for rank, feature in enumerate(features, start=1):
        feature[rank_key] = rank
        feature[top_key] = 1 if rank == 1 else 0

    return features


def load_scene_inputs(
    dataset_root: Path,
    scene_id: int,
    image_id: int,
):
    scene_dir = dataset_root / "val" / f"{scene_id:06d}"

    rgb_path = scene_dir / "rgb_realsense" / f"{image_id:06d}.png"
    depth_path = scene_dir / "depth_realsense" / f"{image_id:06d}.png"
    camera_path = scene_dir / "scene_camera_realsense.json"
    scene_gt_path = scene_dir / "scene_gt_realsense.json"
    scene_gt_info_path = scene_dir / "scene_gt_info_realsense.json"
    mask_dir = scene_dir / "mask_visib_realsense"

    rgb = load_rgb(rgb_path)
    depth = load_depth(depth_path)
    scene_camera = load_json(camera_path)
    scene_gt = load_json(scene_gt_path)
    scene_gt_info = load_json(scene_gt_info_path)

    camera_k = get_camera_intrinsics(scene_camera, image_id)
    depth_scale = get_depth_scale(scene_camera, image_id)

    return {
        "scene_dir": scene_dir,
        "rgb": rgb,
        "depth": depth,
        "camera_k": camera_k,
        "depth_scale": depth_scale,
        "rgb_path": rgb_path,
        "mask_dir": mask_dir,
        "scene_gt": scene_gt,
        "scene_gt_info": scene_gt_info,
    }


def build_ground_truth_candidates(
    scene_data: dict,
    image_id: int,
) -> tuple[list[dict], dict[int, np.ndarray], list[dict]]:
    depth = scene_data["depth"]
    camera_k = scene_data["camera_k"]
    depth_scale = scene_data["depth_scale"]
    mask_dir = scene_data["mask_dir"]
    scene_gt = scene_data["scene_gt"]
    scene_gt_info = scene_data["scene_gt_info"]
    rgb = scene_data["rgb"]

    image_key = str(image_id)

    mask_files = sorted(mask_dir.glob(f"{image_id:06d}_*.png"))

    features = []
    masks_by_instance = {}
    gt_instances = []

    for mask_path in mask_files:
        object_index = parse_object_index(mask_path)

        if object_index >= len(scene_gt[image_key]):
            continue

        part_id = int(scene_gt[image_key][object_index]["obj_id"])

        if (
            image_key in scene_gt_info
            and object_index < len(scene_gt_info[image_key])
        ):
            visible_fraction = float(
                scene_gt_info[image_key][object_index].get("visib_fract", 1.0)
            )
        else:
            visible_fraction = 1.0

        mask = load_mask(mask_path)
        mask = ensure_2d_mask(mask)

        if not np.any(mask):
            continue

        feature, _ = compute_object_features(
            object_index=object_index,
            object_id=part_id,
            visible_fraction=visible_fraction,
            mask=mask,
            depth=depth,
            camera_k=camera_k,
            depth_scale=depth_scale,
        )

        feature["part_id"] = part_id
        feature["instance"] = object_index
        feature["source"] = "dataset_mask"

        features.append(feature)
        masks_by_instance[object_index] = mask

        gt_instances.append(
            {
                "instance": object_index,
                "part_id": part_id,
                "mask": mask,
            }
        )

    features = add_pickability_scores(features, rgb.shape)

    for feature in features:
        feature["part_id"] = feature["object_id"]
        feature["instance"] = feature["object_index"]
        feature["heuristic_rank"] = feature["pick_rank"]
        feature["heuristic_pickability_score"] = feature["pickability_score"]
        feature["is_heuristic_top_pick"] = 1 if feature["heuristic_rank"] == 1 else 0

    return features, masks_by_instance, gt_instances


def polygon_to_mask(
    polygon_xy: np.ndarray,
    image_shape: tuple[int, int, int],
) -> np.ndarray:
    height, width = image_shape[:2]

    mask = np.zeros((height, width), dtype=np.uint8)

    if polygon_xy is None or len(polygon_xy) < 3:
        return ensure_2d_mask(mask)

    points = np.round(polygon_xy).astype(np.int32)
    cv2.fillPoly(mask, [points], 255)

    return ensure_2d_mask(mask)


def predict_yolo_instances(
    yolo_model: YOLO,
    image_path: Path,
    image_shape: tuple[int, int, int],
    class_index_to_part_id: dict[int, int],
    imgsz: int,
    conf: float,
    iou: float,
    max_det: int,
    device: str,
) -> list[dict]:
    results = yolo_model.predict(
        source=str(image_path),
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        max_det=max_det,
        device=device,
        retina_masks=True,
        verbose=False,
    )

    result = results[0]

    if result.masks is None or result.boxes is None:
        return []

    masks_xy = result.masks.xy
    classes = result.boxes.cls.detach().cpu().numpy().astype(int)
    confidences = result.boxes.conf.detach().cpu().numpy()

    instances = []

    for index, polygon_xy in enumerate(masks_xy):
        class_index = int(classes[index])
        confidence = float(confidences[index])
        part_id = class_index_to_part_id.get(class_index, class_index)

        mask = polygon_to_mask(
            polygon_xy=np.asarray(polygon_xy),
            image_shape=image_shape,
        )

        if not np.any(mask):
            continue

        instances.append(
            {
                "segmentation_rank": len(instances) + 1,
                "class_index": class_index,
                "part_id": part_id,
                "confidence": confidence,
                "mask": mask,
                "mask_pixels": int(mask.sum()),
            }
        )

    instances = sorted(
        instances,
        key=lambda item: item["confidence"],
        reverse=True,
    )

    for index, instance in enumerate(instances, start=1):
        instance["segmentation_rank"] = index

    return instances


def mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    mask_a = ensure_2d_mask(mask_a)
    mask_b = ensure_2d_mask(mask_b)

    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()

    if union == 0:
        return 0.0

    return float(intersection / union)


def attach_best_ground_truth_matches(
    yolo_instances: list[dict],
    gt_instances: list[dict],
) -> None:
    for yolo_instance in yolo_instances:
        best_iou = 0.0
        best_gt_instance = None
        best_gt_part_id = None

        for gt_instance in gt_instances:
            iou_value = mask_iou(
                yolo_instance["mask"],
                gt_instance["mask"],
            )

            if iou_value > best_iou:
                best_iou = iou_value
                best_gt_instance = gt_instance["instance"]
                best_gt_part_id = gt_instance["part_id"]

        yolo_instance["matched_gt_instance"] = best_gt_instance
        yolo_instance["matched_gt_part_id"] = best_gt_part_id
        yolo_instance["matched_gt_iou"] = best_iou


def build_ranker_features_from_yolo_instances(
    instances: list[dict],
    depth: np.ndarray,
    camera_k: np.ndarray,
    depth_scale: float,
    visibility_proxy: str,
) -> list[dict]:
    """
    Convert YOLO-predicted masks into object-candidate features.

    visibility_proxy:
        confidence -> use YOLO confidence as a proxy for visible_fraction
        one        -> set visible_fraction = 1.0 for every predicted object
    """
    features = []

    for index, instance in enumerate(instances):
        part_id = int(instance["part_id"])
        mask = ensure_2d_mask(instance["mask"])

        if visibility_proxy == "confidence":
            visible_fraction = float(instance["confidence"])
        elif visibility_proxy == "one":
            visible_fraction = 1.0
        else:
            raise ValueError(f"Unknown visibility proxy: {visibility_proxy}")

        feature, _ = compute_object_features(
            object_index=index,
            object_id=part_id,
            visible_fraction=visible_fraction,
            mask=mask,
            depth=depth,
            camera_k=camera_k,
            depth_scale=depth_scale,
        )

        feature["part_id"] = part_id
        feature["instance"] = index
        feature["source"] = "yolo_mask"
        feature["yolo_class_index"] = int(instance["class_index"])
        feature["segmentation_confidence"] = float(instance["confidence"])
        feature["segmentation_rank"] = int(instance["segmentation_rank"])
        feature["segmentation_mask_pixels"] = int(instance["mask_pixels"])
        feature["matched_gt_instance"] = instance.get("matched_gt_instance")
        feature["matched_gt_part_id"] = instance.get("matched_gt_part_id")
        feature["matched_gt_iou"] = instance.get("matched_gt_iou", 0.0)

        features.append(feature)

    return features


def compute_mask_center(mask: np.ndarray) -> tuple[int, int]:
    mask = ensure_2d_mask(mask)

    ys, xs = np.where(mask)

    if len(xs) == 0:
        return 20, 20

    center_x = float(xs.mean())
    center_y = float(ys.mean())

    x = int(round(center_x))
    y = int(round(center_y))

    height, width = mask.shape[:2]

    x = int(np.clip(x, 18, width - 18))
    y = int(np.clip(y, 18, height - 18))

    if mask[y, x]:
        return x, y

    distances = (xs - center_x) ** 2 + (ys - center_y) ** 2
    closest_index = int(np.argmin(distances))

    x = int(xs[closest_index])
    y = int(ys[closest_index])

    x = int(np.clip(x, 18, width - 18))
    y = int(np.clip(y, 18, height - 18))

    return x, y


def draw_badge(
    image: np.ndarray,
    center: tuple[int, int],
    number: int,
    color: tuple[int, int, int],
    is_top: bool = False,
) -> None:
    x, y = center

    if is_top:
        radius = 18
        font_scale = 0.65
    else:
        radius = 13
        font_scale = 0.50

    cv2.circle(image, (x, y), radius + 4, (0, 0, 0), -1)
    cv2.circle(image, (x, y), radius + 2, (255, 255, 255), -1)
    cv2.circle(image, (x, y), radius, color, -1)

    text = str(number)
    text_color = text_color_for_background(color)

    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2

    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
    text_width, text_height = text_size

    text_x = int(x - text_width / 2)
    text_y = int(y + text_height / 2)

    cv2.putText(
        image,
        text,
        (text_x, text_y),
        font,
        font_scale,
        text_color,
        thickness,
        cv2.LINE_AA,
    )


def draw_contours(
    image: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int],
    is_top: bool,
) -> None:
    mask = ensure_2d_mask(mask)
    mask_uint8 = mask.astype(np.uint8) * 255

    contours, _ = cv2.findContours(
        mask_uint8,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if not contours:
        return

    if is_top:
        cv2.drawContours(image, contours, -1, (255, 255, 255), 4)
        cv2.drawContours(image, contours, -1, color, 2)
    else:
        cv2.drawContours(image, contours, -1, (255, 255, 255), 2)
        cv2.drawContours(image, contours, -1, color, 1)


def draw_ranked_overlay(
    image_rgb: np.ndarray,
    ranked_features: list[dict],
    mask_lookup: dict[int, np.ndarray],
    rank_key: str,
    title_prefix: str,
) -> np.ndarray:
    output = image_rgb.copy()

    for feature in ranked_features:
        rank = int(feature[rank_key])
        instance = int(feature["instance"])

        if instance not in mask_lookup:
            continue

        color = get_color(rank - 1)
        mask = ensure_2d_mask(mask_lookup[instance])

        if rank == 1:
            alpha = 0.52
        elif rank <= 10:
            alpha = 0.30
        else:
            alpha = 0.10

        color_array = np.array(color, dtype=np.uint8)

        output[mask] = (
            (1.0 - alpha) * output[mask] + alpha * color_array
        ).astype(np.uint8)

    for feature in ranked_features:
        rank = int(feature[rank_key])
        instance = int(feature["instance"])

        if instance not in mask_lookup:
            continue

        color = get_color(rank - 1)
        mask = ensure_2d_mask(mask_lookup[instance])

        draw_contours(
            image=output,
            mask=mask,
            color=color,
            is_top=(rank == 1),
        )

    for feature in ranked_features:
        rank = int(feature[rank_key])

        if rank > 10:
            continue

        instance = int(feature["instance"])

        if instance not in mask_lookup:
            continue

        color = get_color(rank - 1)
        mask = ensure_2d_mask(mask_lookup[instance])

        center = compute_mask_center(mask)

        draw_badge(
            image=output,
            center=center,
            number=rank,
            color=color,
            is_top=(rank == 1),
        )

    return output


def save_three_way_comparison_figure(
    image_rgb: np.ndarray,
    heuristic_features: list[dict],
    known_mask_model_features: list[dict],
    integrated_features: list[dict],
    gt_masks_by_instance: dict[int, np.ndarray],
    yolo_instances: list[dict],
    output_path: Path,
    scene_id: int,
    image_id: int,
) -> None:
    yolo_masks_by_instance = {
        index: instance["mask"]
        for index, instance in enumerate(yolo_instances)
    }

    heuristic_overlay = draw_ranked_overlay(
        image_rgb=image_rgb,
        ranked_features=heuristic_features,
        mask_lookup=gt_masks_by_instance,
        rank_key="heuristic_rank",
        title_prefix="Heuristic",
    )

    known_mask_overlay = draw_ranked_overlay(
        image_rgb=image_rgb,
        ranked_features=known_mask_model_features,
        mask_lookup=gt_masks_by_instance,
        rank_key="known_mask_rank",
        title_prefix="Known Mask Model",
    )

    integrated_overlay = draw_ranked_overlay(
        image_rgb=image_rgb,
        ranked_features=integrated_features,
        mask_lookup=yolo_masks_by_instance,
        rank_key="integrated_rank",
        title_prefix="Integrated",
    )

    height, width = image_rgb.shape[:2]

    fig = plt.figure(figsize=(12, 24))
    grid = fig.add_gridspec(3, 1, hspace=0.12)

    axes = [
        fig.add_subplot(grid[0, 0]),
        fig.add_subplot(grid[1, 0]),
        fig.add_subplot(grid[2, 0]),
    ]

    heuristic_top = heuristic_features[0] if heuristic_features else None
    known_top = known_mask_model_features[0] if known_mask_model_features else None
    integrated_top = integrated_features[0] if integrated_features else None

    titles = []

    if heuristic_top is not None:
        titles.append(
            f"1. Heuristic Target using Dataset Masks | "
            f"Pick #1: Part {heuristic_top['part_id']}, Instance {heuristic_top['instance']}, "
            f"Score {heuristic_top['heuristic_pickability_score']:.3f}"
        )
    else:
        titles.append("1. Heuristic Target using Dataset Masks | No candidates")

    if known_top is not None:
        titles.append(
            f"2. Learned Ranker using Dataset Masks | "
            f"Pick #1: Part {known_top['part_id']}, Instance {known_top['instance']}, "
            f"Score {known_top['known_mask_predicted_score']:.3f}"
        )
    else:
        titles.append("2. Learned Ranker using Dataset Masks | No candidates")

    if integrated_top is not None:
        matched_text = ""
        if integrated_top.get("matched_gt_instance") is not None:
            matched_text = (
                f", matched GT instance {integrated_top['matched_gt_instance']} "
                f"(IoU {integrated_top['matched_gt_iou']:.2f})"
            )

        titles.append(
            f"3. Integrated YOLO Segmentation + Ranker | "
            f"Pick #1: Part {integrated_top['part_id']}, YOLO Candidate {integrated_top['instance']}"
            f"{matched_text}, Score {integrated_top['integrated_predicted_score']:.3f}"
        )
    else:
        titles.append("3. Integrated YOLO Segmentation + Ranker | No candidates")

    overlays = [
        heuristic_overlay,
        known_mask_overlay,
        integrated_overlay,
    ]

    for ax, overlay, title in zip(axes, overlays, titles):
        ax.imshow(overlay)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.axis("off")
        ax.set_xlim(0, width)
        ax.set_ylim(height, 0)
        ax.set_aspect("equal", adjustable="box")

    fig.suptitle(
        f"Three-Way Pick Candidate Comparison: Scene {scene_id:06d}, Image {image_id:06d}",
        fontsize=16,
        fontweight="bold",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close()


def get_feature_by_instance(features: list[dict], instance: int) -> dict | None:
    for feature in features:
        if int(feature["instance"]) == int(instance):
            return feature

    return None


def save_score_comparison_plot(
    heuristic_features: list[dict],
    known_mask_model_features: list[dict],
    integrated_features: list[dict],
    output_path: Path,
    scene_id: int,
    image_id: int,
    top_n: int = 10,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Plot 1: dataset masks, same candidate order as heuristic top-N.
    heuristic_top = sorted(
        heuristic_features,
        key=lambda feature: int(feature["heuristic_rank"]),
    )[:top_n]

    labels_gt = []
    heuristic_scores = []
    known_model_scores = []

    for feature in heuristic_top:
        instance = int(feature["instance"])
        known_feature = get_feature_by_instance(known_mask_model_features, instance)

        labels_gt.append(f"P{feature['part_id']}\nI{instance}")
        heuristic_scores.append(float(feature["heuristic_pickability_score"]))

        if known_feature is None:
            known_model_scores.append(np.nan)
        else:
            known_model_scores.append(float(known_feature["known_mask_predicted_score"]))

    # Plot 2: integrated YOLO top-N with matched GT heuristic score where available.
    integrated_top = sorted(
        integrated_features,
        key=lambda feature: int(feature["integrated_rank"]),
    )[:top_n]

    labels_integrated = []
    integrated_scores = []
    matched_heuristic_scores = []

    for feature in integrated_top:
        yolo_instance = int(feature["instance"])
        matched_gt_instance = feature.get("matched_gt_instance")
        matched_iou = safe_float(feature.get("matched_gt_iou"), default=0.0)

        if matched_gt_instance is None:
            label = f"Y{yolo_instance}\nGT n/a"
            matched_score = np.nan
        else:
            label = f"Y{yolo_instance}\nGT {matched_gt_instance}\nIoU {matched_iou:.2f}"
            matched_feature = get_feature_by_instance(heuristic_features, int(matched_gt_instance))

            if matched_feature is None:
                matched_score = np.nan
            else:
                matched_score = float(matched_feature["heuristic_pickability_score"])

        labels_integrated.append(label)
        integrated_scores.append(float(feature["integrated_predicted_score"]))
        matched_heuristic_scores.append(matched_score)

    fig = plt.figure(figsize=(15, 10))
    grid = fig.add_gridspec(2, 1, hspace=0.35)

    ax_gt = fig.add_subplot(grid[0, 0])
    ax_integrated = fig.add_subplot(grid[1, 0])

    x_gt = np.arange(len(labels_gt))
    bar_width = 0.38

    ax_gt.bar(
        x_gt - bar_width / 2,
        heuristic_scores,
        width=bar_width,
        label="Heuristic target score",
    )

    ax_gt.bar(
        x_gt + bar_width / 2,
        known_model_scores,
        width=bar_width,
        label="Known-mask model score",
    )

    ax_gt.set_xticks(x_gt)
    ax_gt.set_xticklabels(labels_gt)
    ax_gt.set_ylim(0, 1.05)
    ax_gt.set_ylabel("Score")
    ax_gt.set_title(
        "Dataset Masks: Heuristic Target vs Learned Ranker",
        fontweight="bold",
    )
    ax_gt.grid(True, axis="y", alpha=0.3)
    ax_gt.legend()

    x_int = np.arange(len(labels_integrated))

    ax_integrated.bar(
        x_int - bar_width / 2,
        matched_heuristic_scores,
        width=bar_width,
        label="Matched GT heuristic score",
    )

    ax_integrated.bar(
        x_int + bar_width / 2,
        integrated_scores,
        width=bar_width,
        label="Integrated YOLO + ranker score",
    )

    ax_integrated.set_xticks(x_int)
    ax_integrated.set_xticklabels(labels_integrated)
    ax_integrated.set_ylim(0, 1.05)
    ax_integrated.set_ylabel("Score")
    ax_integrated.set_title(
        "YOLO-Predicted Masks: Integrated Ranker vs Matched Ground-Truth Candidate",
        fontweight="bold",
    )
    ax_integrated.grid(True, axis="y", alpha=0.3)
    ax_integrated.legend()

    fig.suptitle(
        f"Score Comparison: Scene {scene_id:06d}, Image {image_id:06d}",
        fontsize=16,
        fontweight="bold",
    )

    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close()


def make_summary_rows(
    heuristic_features: list[dict],
    known_mask_model_features: list[dict],
    integrated_features: list[dict],
) -> list[dict]:
    rows = []

    for feature in sorted(heuristic_features, key=lambda f: int(f["heuristic_rank"])):
        rows.append(
            {
                "source": "heuristic_dataset_masks",
                "rank": int(feature["heuristic_rank"]),
                "part_id": feature["part_id"],
                "instance": feature["instance"],
                "score": feature["heuristic_pickability_score"],
                "matched_gt_instance": feature["instance"],
                "matched_gt_part_id": feature["part_id"],
                "matched_gt_iou": 1.0,
                "segmentation_confidence": "",
            }
        )

    for feature in sorted(known_mask_model_features, key=lambda f: int(f["known_mask_rank"])):
        rows.append(
            {
                "source": "ranker_dataset_masks",
                "rank": int(feature["known_mask_rank"]),
                "part_id": feature["part_id"],
                "instance": feature["instance"],
                "score": feature["known_mask_predicted_score"],
                "matched_gt_instance": feature["instance"],
                "matched_gt_part_id": feature["part_id"],
                "matched_gt_iou": 1.0,
                "segmentation_confidence": "",
            }
        )

    for feature in sorted(integrated_features, key=lambda f: int(f["integrated_rank"])):
        rows.append(
            {
                "source": "integrated_yolo_ranker",
                "rank": int(feature["integrated_rank"]),
                "part_id": feature["part_id"],
                "instance": feature["instance"],
                "score": feature["integrated_predicted_score"],
                "matched_gt_instance": feature.get("matched_gt_instance"),
                "matched_gt_part_id": feature.get("matched_gt_part_id"),
                "matched_gt_iou": feature.get("matched_gt_iou"),
                "segmentation_confidence": feature.get("segmentation_confidence"),
            }
        )

    return rows


def save_summary_csv(rows: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        return

    fieldnames = [
        "source",
        "rank",
        "part_id",
        "instance",
        "score",
        "matched_gt_instance",
        "matched_gt_part_id",
        "matched_gt_iou",
        "segmentation_confidence",
    ]

    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_top_pick_summary(
    heuristic_features: list[dict],
    known_mask_model_features: list[dict],
    integrated_features: list[dict],
) -> None:
    heuristic_top = heuristic_features[0] if heuristic_features else None
    known_top = known_mask_model_features[0] if known_mask_model_features else None
    integrated_top = integrated_features[0] if integrated_features else None

    if heuristic_top is not None:
        print(
            "Heuristic top pick: "
            f"part={heuristic_top['part_id']}, "
            f"instance={heuristic_top['instance']}, "
            f"score={heuristic_top['heuristic_pickability_score']:.3f}"
        )

    if known_top is not None:
        print(
            "Known-mask model top pick: "
            f"part={known_top['part_id']}, "
            f"instance={known_top['instance']}, "
            f"score={known_top['known_mask_predicted_score']:.3f}"
        )

    if integrated_top is not None:
        match_text = "no GT match"

        if integrated_top.get("matched_gt_instance") is not None:
            match_text = (
                f"matched_gt_instance={integrated_top['matched_gt_instance']}, "
                f"IoU={integrated_top['matched_gt_iou']:.3f}"
            )

        print(
            "Integrated YOLO+ranker top pick: "
            f"part={integrated_top['part_id']}, "
            f"candidate={integrated_top['instance']}, "
            f"score={integrated_top['integrated_predicted_score']:.3f}, "
            f"seg_conf={integrated_top['segmentation_confidence']:.3f}, "
            f"{match_text}"
        )


def run_one_example(
    scene_id: int,
    image_id: int,
    dataset_root: Path,
    yolo_model: YOLO,
    ranker_model: PickabilityRanker,
    feature_columns: list[str],
    medians: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    class_index_to_part_id: dict[int, int],
    torch_device: torch.device,
    yolo_device: str,
    imgsz: int,
    conf: float,
    iou: float,
    max_det: int,
    visibility_proxy: str,
    output_dir: Path,
    report_dir: Path,
) -> None:
    scene_data = load_scene_inputs(
        dataset_root=dataset_root,
        scene_id=scene_id,
        image_id=image_id,
    )

    image_rgb = scene_data["rgb"]
    depth = scene_data["depth"]
    camera_k = scene_data["camera_k"]
    depth_scale = scene_data["depth_scale"]
    image_path = scene_data["rgb_path"]

    heuristic_features, gt_masks_by_instance, gt_instances = build_ground_truth_candidates(
        scene_data=scene_data,
        image_id=image_id,
    )

    # Known-mask model: same dataset masks, but ranked by learned PyTorch model.
    known_mask_model_features = [
        dict(feature)
        for feature in heuristic_features
    ]

    known_mask_model_features = predict_ranker_scores(
        ranker_model=ranker_model,
        features=known_mask_model_features,
        feature_columns=feature_columns,
        medians=medians,
        mean=mean,
        std=std,
        device=torch_device,
        score_key="known_mask_predicted_score",
        rank_key="known_mask_rank",
        top_key="known_mask_top_pick",
    )

    # Integrated model: YOLO predicted masks, then geometry features, then ranker.
    yolo_instances = predict_yolo_instances(
        yolo_model=yolo_model,
        image_path=image_path,
        image_shape=image_rgb.shape,
        class_index_to_part_id=class_index_to_part_id,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        max_det=max_det,
        device=yolo_device,
    )

    attach_best_ground_truth_matches(
        yolo_instances=yolo_instances,
        gt_instances=gt_instances,
    )

    integrated_features = build_ranker_features_from_yolo_instances(
        instances=yolo_instances,
        depth=depth,
        camera_k=camera_k,
        depth_scale=depth_scale,
        visibility_proxy=visibility_proxy,
    )

    integrated_features = predict_ranker_scores(
        ranker_model=ranker_model,
        features=integrated_features,
        feature_columns=feature_columns,
        medians=medians,
        mean=mean,
        std=std,
        device=torch_device,
        score_key="integrated_predicted_score",
        rank_key="integrated_rank",
        top_key="integrated_top_pick",
    )

    output_stem = f"scene_{scene_id:06d}_image_{image_id:06d}_three_way"

    comparison_path = output_dir / f"{output_stem}_visual_comparison.png"
    score_plot_path = output_dir / f"{output_stem}_score_comparison.png"
    summary_csv_path = report_dir / f"{output_stem}_summary.csv"

    save_three_way_comparison_figure(
        image_rgb=image_rgb,
        heuristic_features=heuristic_features,
        known_mask_model_features=known_mask_model_features,
        integrated_features=integrated_features,
        gt_masks_by_instance=gt_masks_by_instance,
        yolo_instances=yolo_instances,
        output_path=comparison_path,
        scene_id=scene_id,
        image_id=image_id,
    )

    save_score_comparison_plot(
        heuristic_features=heuristic_features,
        known_mask_model_features=known_mask_model_features,
        integrated_features=integrated_features,
        output_path=score_plot_path,
        scene_id=scene_id,
        image_id=image_id,
        top_n=10,
    )

    summary_rows = make_summary_rows(
        heuristic_features=heuristic_features,
        known_mask_model_features=known_mask_model_features,
        integrated_features=integrated_features,
    )

    save_summary_csv(
        rows=summary_rows,
        output_path=summary_csv_path,
    )

    print(f"\nScene {scene_id:06d}, image {image_id:06d}")
    print(f"GT candidates: {len(heuristic_features)}")
    print(f"YOLO candidates: {len(yolo_instances)}")
    print(f"Saved visual comparison: {comparison_path}")
    print(f"Saved score comparison: {score_plot_path}")
    print(f"Saved summary CSV: {summary_csv_path}")

    print_top_pick_summary(
        heuristic_features=heuristic_features,
        known_mask_model_features=known_mask_model_features,
        integrated_features=integrated_features,
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset-root", type=Path, default=Path("data/xyzibd"))

    parser.add_argument(
        "--yolo-model-path",
        type=Path,
        default=Path(
            "runs/segment/outputs/segmentation_training/yolo_segmentation/weights/best.pt"
        ),
    )

    parser.add_argument(
        "--ranker-model-path",
        type=Path,
        default=Path("outputs/models/pickability_ranker.pt"),
    )

    parser.add_argument(
        "--class-mapping",
        type=Path,
        default=Path("data/segmentation_yolo/class_mapping.json"),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/integrated_yolo_ranker/figures"),
    )

    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path("outputs/integrated_yolo_ranker/reports"),
    )

    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.50)
    parser.add_argument("--max-det", type=int, default=80)
    parser.add_argument("--yolo-device", type=str, default="0")

    parser.add_argument(
        "--visibility-proxy",
        type=str,
        default="confidence",
        choices=["confidence", "one"],
        help=(
            "How to approximate visible_fraction for YOLO-predicted masks. "
            "'confidence' uses YOLO confidence as a proxy. "
            "'one' sets every predicted mask to visible_fraction=1.0."
        ),
    )

    args = parser.parse_args()

    torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Three-way comparison: heuristic vs known-mask ranker vs integrated YOLO+ranker")
    print(f"YOLO model: {args.yolo_model_path}")
    print(f"Ranker model: {args.ranker_model_path}")
    print(f"Class mapping: {args.class_mapping}")
    print(f"Dataset root: {args.dataset_root}")
    print(f"imgsz={args.imgsz}, conf={args.conf}, iou={args.iou}")
    print(f"visibility_proxy={args.visibility_proxy}")

    print("\nCUDA check:")
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    yolo_model = YOLO(str(args.yolo_model_path))

    ranker_model, feature_columns, medians, mean, std = load_pickability_ranker(
        model_path=args.ranker_model_path,
        device=torch_device,
    )

    class_index_to_part_id = load_class_mapping(args.class_mapping)

    for scene_id, image_id in DEFAULT_EXAMPLES:
        run_one_example(
            scene_id=scene_id,
            image_id=image_id,
            dataset_root=args.dataset_root,
            yolo_model=yolo_model,
            ranker_model=ranker_model,
            feature_columns=feature_columns,
            medians=medians,
            mean=mean,
            std=std,
            class_index_to_part_id=class_index_to_part_id,
            torch_device=torch_device,
            yolo_device=args.yolo_device,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            max_det=args.max_det,
            visibility_proxy=args.visibility_proxy,
            output_dir=args.output_dir,
            report_dir=args.report_dir,
        )

    print("\nThree-way comparison complete.")
    print(f"Figures: {args.output_dir}")
    print(f"CSVs: {args.report_dir}")


if __name__ == "__main__":
    main()