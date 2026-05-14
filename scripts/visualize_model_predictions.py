from pathlib import Path
import argparse
import csv
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

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
    find_scene_dir,
    find_first_image_id,
    find_visible_mask_files,
    parse_object_index,
    get_annotation_for_object,
    compute_object_features,
    add_pickability_scores,
    draw_mask_overlay,
    draw_mask_contour,
    compute_badge_center,
    get_rank_color,
    matplotlib_text_color_for_background,
)


DEFAULT_EXAMPLES = [
    (0, 0),
    (0, 1),
    (0, 2),
    (5, 0),
    (5, 1),
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


def load_trained_ranker(model_path: Path, device: torch.device):
    """
    Load trained ranker model and preprocessing metadata.

    This script expects the model to already exist.
    If it does not, run:
        python scripts\\train_pickability_ranker.py
    """
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}\n"
            "Run this first:\n"
            "python scripts\\train_pickability_ranker.py"
        )

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


def predict_model_scores(
    model: PickabilityRanker,
    features: list[dict],
    feature_columns: list[str],
    medians: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    device: torch.device,
) -> list[dict]:
    if not features:
        return features

    x_raw = build_feature_matrix(features, feature_columns)
    x = apply_preprocessor(x_raw, medians, mean, std)

    with torch.no_grad():
        x_tensor = torch.tensor(x, dtype=torch.float32).to(device)
        predictions = model(x_tensor).detach().cpu().numpy().reshape(-1)

    for feature, prediction in zip(features, predictions):
        feature["predicted_pickability_score"] = float(prediction)

    features = sorted(
        features,
        key=lambda feature: feature["predicted_pickability_score"],
        reverse=True,
    )

    for rank, feature in enumerate(features, start=1):
        feature["predicted_rank"] = rank
        feature["predicted_top_pick"] = 1 if rank == 1 else 0

    return features


def load_scene_candidates(
    dataset_root: Path,
    scene_id: int | None,
    image_id: int | None,
) -> tuple[np.ndarray, list[dict], dict[int, np.ndarray], Path, int]:
    scene_dir = find_scene_dir(dataset_root, scene_id)

    if image_id is None:
        image_id = find_first_image_id(scene_dir)

    rgb_path = scene_dir / "rgb_realsense" / f"{image_id:06d}.png"
    depth_path = scene_dir / "depth_realsense" / f"{image_id:06d}.png"

    scene_camera = load_json(scene_dir / "scene_camera_realsense.json")
    scene_gt = load_json(scene_dir / "scene_gt_realsense.json")
    scene_gt_info = load_json(scene_dir / "scene_gt_info_realsense.json")

    camera_k = get_camera_intrinsics(scene_camera, image_id)
    depth_scale = get_depth_scale(scene_camera, image_id)

    rgb = load_rgb(rgb_path)
    depth = load_depth(depth_path)

    mask_files = find_visible_mask_files(scene_dir, image_id)

    features = []
    masks_by_object_index = {}

    for mask_path in mask_files:
        object_index = parse_object_index(mask_path)

        object_id, visible_fraction = get_annotation_for_object(
            scene_gt=scene_gt,
            scene_gt_info=scene_gt_info,
            image_id=image_id,
            object_index=object_index,
        )

        mask = load_mask(mask_path)
        masks_by_object_index[object_index] = mask

        feature, _ = compute_object_features(
            object_index=object_index,
            object_id=object_id,
            visible_fraction=visible_fraction,
            mask=mask,
            depth=depth,
            camera_k=camera_k,
            depth_scale=depth_scale,
        )

        feature["part_id"] = feature["object_id"]
        feature["instance"] = feature["object_index"]

        features.append(feature)

    # This adds the heuristic target score and rank.
    features = add_pickability_scores(features, rgb.shape)

    for feature in features:
        feature["part_id"] = feature["object_id"]
        feature["instance"] = feature["object_index"]
        feature["heuristic_rank"] = feature["pick_rank"]
        feature["heuristic_pickability_score"] = feature["pickability_score"]

    return rgb, features, masks_by_object_index, scene_dir, image_id


def text_color_for_background(color: tuple[int, int, int]) -> tuple[int, int, int]:
    r, g, b = color
    luminance = 0.299 * r + 0.587 * g + 0.114 * b

    if luminance > 140:
        return (0, 0, 0)

    return (255, 255, 255)


def draw_rank_badge_local(
    image: np.ndarray,
    center: tuple[int, int],
    rank: int,
    color: tuple[int, int, int],
    is_top_pick: bool = False,
) -> None:
    """
    Draw rank label.

    The top-ranked object gets a larger badge.
    """
    x, y = center

    if is_top_pick:
        radius = 18
        font_scale = 0.65
    else:
        radius = 14
        font_scale = 0.55

    cv2.circle(image, (x, y), radius + 4, (0, 0, 0), -1)
    cv2.circle(image, (x, y), radius + 2, (255, 255, 255), -1)
    cv2.circle(image, (x, y), radius, color, -1)

    text = str(rank)
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


def draw_rank_overlay(
    rgb: np.ndarray,
    features: list[dict],
    masks_by_object_index: dict[int, np.ndarray],
    rank_key: str,
    top_key: str,
    score_key: str,
) -> np.ndarray:
    """
    Draw one ranking overlay.

    rank_key:
        "heuristic_rank" or "predicted_rank"

    top_key:
        "is_top_pick" or "predicted_top_pick"

    score_key:
        "heuristic_pickability_score" or "predicted_pickability_score"
    """
    annotated_image = rgb.copy()
    max_badges = 10

    # Draw masks.
    for feature in features:
        rank = int(feature[rank_key])
        object_index = int(feature["object_index"])

        if object_index not in masks_by_object_index:
            continue

        mask = masks_by_object_index[object_index]
        color = get_rank_color(rank)

        if rank == 1:
            alpha = 0.50
        elif rank <= max_badges:
            alpha = 0.30
        else:
            alpha = 0.10

        annotated_image = draw_mask_overlay(
            image=annotated_image,
            mask=mask,
            color=color,
            alpha=alpha,
        )

    # Draw contours.
    for feature in features:
        rank = int(feature[rank_key])
        object_index = int(feature["object_index"])

        if object_index not in masks_by_object_index:
            continue

        mask = masks_by_object_index[object_index]
        color = get_rank_color(rank)

        if rank == 1:
            mask_uint8 = mask.astype(np.uint8) * 255
            contours, _ = cv2.findContours(
                mask_uint8,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )

            if contours:
                cv2.drawContours(annotated_image, contours, -1, (255, 255, 255), 4)
                cv2.drawContours(annotated_image, contours, -1, color, 2)
        else:
            draw_mask_contour(annotated_image, mask, color)

    # Draw rank badges.
    for feature in features:
        rank = int(feature[rank_key])

        if rank > max_badges:
            continue

        object_index = int(feature["object_index"])

        if object_index not in masks_by_object_index:
            continue

        mask = masks_by_object_index[object_index]
        color = get_rank_color(rank)

        badge_center = compute_badge_center(
            feature=feature,
            image_shape=annotated_image.shape,
            mask=mask,
        )

        draw_rank_badge_local(
            image=annotated_image,
            center=badge_center,
            rank=rank,
            color=color,
            is_top_pick=(rank == 1 or int(feature.get(top_key, 0)) == 1),
        )

    return annotated_image


def draw_comparison_table(table_ax, features: list[dict], max_rows: int = 5) -> None:
    table_ax.axis("off")

    features_for_table = sorted(
        features,
        key=lambda feature: int(feature["predicted_rank"]),
    )

    columns = [
        "Pred Rank",
        "Target Rank",
        "Part ID",
        "Instance",
        "Pred Score",
        "Target Score",
        "Visible Pixels",
    ]

    rows = []

    for feature in features_for_table[:max_rows]:
        rows.append(
            [
                str(feature["predicted_rank"]),
                str(feature["heuristic_rank"]),
                str(feature["part_id"]),
                str(feature["instance"]),
                f"{feature['predicted_pickability_score']:.3f}",
                f"{feature['heuristic_pickability_score']:.3f}",
                str(feature["visible_pixels"]),
            ]
        )

    table = table_ax.table(
        cellText=rows,
        colLabels=columns,
        loc="center",
        cellLoc="center",
        colLoc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.55)

    for col_index in range(len(columns)):
        cell = table[(0, col_index)]
        cell.set_facecolor("#f0f0f0")
        cell.set_text_props(weight="bold", color="black")

    for row_index, feature in enumerate(features_for_table[:max_rows], start=1):
        rank = int(feature["predicted_rank"])
        color = get_rank_color(rank)
        color_normalized = tuple(np.array(color, dtype=np.float32) / 255.0)

        rank_cell = table[(row_index, 0)]
        rank_cell.set_facecolor(color_normalized)
        rank_cell.get_text().set_color(matplotlib_text_color_for_background(color))
        rank_cell.get_text().set_weight("bold")

    table_ax.set_title(
        "Top 5 Model-Predicted Pick Candidates",
        fontsize=13,
        fontweight="bold",
        pad=8,
    )


def draw_ground_truth_vs_model_overlay(
    rgb: np.ndarray,
    features: list[dict],
    masks_by_object_index: dict[int, np.ndarray],
    output_path: Path,
) -> None:
    """
    Save a target-vs-model image.

    Left:
    - heuristic target ranking

    Right:
    - model-predicted ranking

    Bottom:
    - table of top 5 model-predicted candidates
    """
    target_image = draw_rank_overlay(
        rgb=rgb,
        features=features,
        masks_by_object_index=masks_by_object_index,
        rank_key="heuristic_rank",
        top_key="is_top_pick",
        score_key="heuristic_pickability_score",
    )

    model_image = draw_rank_overlay(
        rgb=rgb,
        features=features,
        masks_by_object_index=masks_by_object_index,
        rank_key="predicted_rank",
        top_key="predicted_top_pick",
        score_key="predicted_pickability_score",
    )

    image_height, image_width = rgb.shape[:2]

    fig = plt.figure(figsize=(16, 11))
    grid = fig.add_gridspec(
        2,
        2,
        height_ratios=[4.0, 1.15],
        width_ratios=[1.0, 1.0],
        hspace=0.18,
        wspace=0.06,
    )

    target_ax = fig.add_subplot(grid[0, 0])
    model_ax = fig.add_subplot(grid[0, 1])
    table_ax = fig.add_subplot(grid[1, :])

    heuristic_top = sorted(features, key=lambda f: int(f["heuristic_rank"]))[0]
    predicted_top = sorted(features, key=lambda f: int(f["predicted_rank"]))[0]

    target_ax.imshow(target_image)
    target_ax.set_title(
        f"Heuristic Target: Pick #1 = Part {heuristic_top['part_id']}, Instance {heuristic_top['instance']}",
        fontsize=13,
        fontweight="bold",
    )
    target_ax.axis("off")

    model_ax.imshow(model_image)
    model_ax.set_title(
        f"Model Prediction: Pick #1 = Part {predicted_top['part_id']}, Instance {predicted_top['instance']}",
        fontsize=13,
        fontweight="bold",
    )
    model_ax.axis("off")

    for ax in [target_ax, model_ax]:
        ax.set_xlim(0, image_width)
        ax.set_ylim(image_height, 0)
        ax.set_aspect("equal", adjustable="box")

    draw_comparison_table(table_ax, features, max_rows=5)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close()


def save_prediction_csv(features: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    preferred_columns = [
        "predicted_rank",
        "heuristic_rank",
        "part_id",
        "instance",
        "predicted_pickability_score",
        "heuristic_pickability_score",
        "visible_pixels",
        "valid_3d_points",
        "visible_fraction",
        "depth_median",
        "extent_x",
        "extent_y",
        "extent_z",
        "centroid_x",
        "centroid_y",
        "centroid_z",
    ]

    all_columns = sorted({key for feature in features for key in feature.keys()})
    fieldnames = [col for col in preferred_columns if col in all_columns]
    fieldnames += [col for col in all_columns if col not in fieldnames]

    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(features)


def run_one_example(
    dataset_root: Path,
    model,
    feature_columns,
    medians,
    mean,
    std,
    device: torch.device,
    scene_id: int,
    image_id: int,
    output_figure_dir: Path,
    output_report_dir: Path,
) -> None:
    rgb, features, masks_by_object_index, scene_dir, resolved_image_id = load_scene_candidates(
        dataset_root=dataset_root,
        scene_id=scene_id,
        image_id=image_id,
    )

    features = predict_model_scores(
        model=model,
        features=features,
        feature_columns=feature_columns,
        medians=medians,
        mean=mean,
        std=std,
        device=device,
    )

    output_stem = f"scene_{scene_dir.name}_image_{resolved_image_id:06d}"

    figure_path = output_figure_dir / f"{output_stem}_target_vs_model.png"
    csv_path = output_report_dir / f"{output_stem}_target_vs_model.csv"

    draw_ground_truth_vs_model_overlay(
        rgb=rgb,
        features=features,
        masks_by_object_index=masks_by_object_index,
        output_path=figure_path,
    )

    save_prediction_csv(features, csv_path)

    print(f"Saved target-vs-model figure: {figure_path}")
    print(f"Saved target-vs-model CSV: {csv_path}")

    heuristic_top = sorted(features, key=lambda f: int(f["heuristic_rank"]))[0]
    predicted_top = sorted(features, key=lambda f: int(f["predicted_rank"]))[0]

    match = (
        int(heuristic_top["object_index"]) == int(predicted_top["object_index"])
    )

    print(
        f"Heuristic target best: part_id={heuristic_top['part_id']}, "
        f"instance={heuristic_top['instance']}, "
        f"score={heuristic_top['heuristic_pickability_score']:.3f}"
    )

    print(
        f"Model predicted best: part_id={predicted_top['part_id']}, "
        f"instance={predicted_top['instance']}, "
        f"score={predicted_top['predicted_pickability_score']:.3f}"
    )

    print(f"Top-pick match: {match}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("data/xyzibd"))
    parser.add_argument("--model-path", type=Path, default=Path("outputs/models/pickability_ranker.pt"))
    parser.add_argument(
        "--output-figure-dir",
        type=Path,
        default=Path("outputs/figures/model_predictions"),
    )
    parser.add_argument(
        "--output-report-dir",
        type=Path,
        default=Path("outputs/reports/model_predictions"),
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    model, feature_columns, medians, mean, std = load_trained_ranker(
        model_path=args.model_path,
        device=device,
    )

    for scene_id, image_id in DEFAULT_EXAMPLES:
        print(f"\nVisualizing target vs model for scene {scene_id}, image {image_id}")

        run_one_example(
            dataset_root=args.dataset_root,
            model=model,
            feature_columns=feature_columns,
            medians=medians,
            mean=mean,
            std=std,
            device=device,
            scene_id=scene_id,
            image_id=image_id,
            output_figure_dir=args.output_figure_dir,
            output_report_dir=args.output_report_dir,
        )


if __name__ == "__main__":
    main()