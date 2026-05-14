from pathlib import Path
import argparse
import csv
import json
import sys
from collections import defaultdict

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / "scripts"))

from run_integrated_yolo_ranker import (
    load_scene_inputs,
    build_ground_truth_candidates,
    predict_yolo_instances,
    attach_best_ground_truth_matches,
    build_ranker_features_from_yolo_instances,
    predict_ranker_scores,
    load_pickability_ranker,
    load_class_mapping,
)


def get_scene_dirs(dataset_root: Path, scene_ids: list[int] | None) -> list[Path]:
    val_dir = dataset_root / "val"

    if scene_ids:
        scene_dirs = []

        for scene_id in scene_ids:
            scene_dir = val_dir / f"{scene_id:06d}"

            if not scene_dir.exists():
                raise FileNotFoundError(f"Scene directory not found: {scene_dir}")

            scene_dirs.append(scene_dir)

        return scene_dirs

    return sorted([path for path in val_dir.iterdir() if path.is_dir()])


def get_image_ids(scene_dir: Path, max_images_per_scene: int | None) -> list[int]:
    rgb_dir = scene_dir / "rgb_realsense"
    image_ids = sorted([int(path.stem) for path in rgb_dir.glob("*.png")])

    if max_images_per_scene is not None and max_images_per_scene > 0:
        image_ids = image_ids[:max_images_per_scene]

    return image_ids


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


def safe_int_or_none(value):
    if value is None:
        return None

    if isinstance(value, float) and np.isnan(value):
        return None

    value = str(value).strip()

    if value == "":
        return None

    try:
        return int(float(value))
    except ValueError:
        return None


def rank_features(
    features: list[dict],
    rank_key: str,
    score_key: str,
) -> list[dict]:
    if not features:
        return []

    if rank_key in features[0]:
        return sorted(
            features,
            key=lambda feature: (
                safe_float(feature.get(rank_key), default=np.inf),
                safe_float(feature.get("instance"), default=np.inf),
            ),
        )

    return sorted(
        features,
        key=lambda feature: (
            safe_float(feature.get(score_key), default=-np.inf),
            safe_float(feature.get("instance"), default=np.inf),
        ),
        reverse=True,
    )


def build_target_lookup(
    target_features: list[dict],
    target_score_key: str,
) -> dict[int, dict]:
    lookup = {}

    for feature in target_features:
        instance = safe_int_or_none(feature.get("instance"))

        if instance is None:
            continue

        lookup[instance] = {
            "feature": feature,
            "score": safe_float(feature.get(target_score_key), default=0.0),
            "part_id": safe_int_or_none(feature.get("part_id")),
        }

    return lookup


def dcg_at_k(relevances: list[float], k: int) -> float:
    dcg = 0.0

    for rank_index, relevance in enumerate(relevances[:k], start=1):
        dcg += float(relevance) / np.log2(rank_index + 1)

    return float(dcg)


def ndcg_at_k(
    target_scores_by_instance: dict[int, float],
    integrated_matched_instances: list[int | None],
    k: int,
) -> float:
    used_instances = set()
    predicted_relevances = []

    for matched_instance in integrated_matched_instances:
        if matched_instance is None:
            relevance = 0.0
        elif matched_instance in used_instances:
            # Avoid giving repeated credit for duplicate YOLO masks matching the same GT object.
            relevance = 0.0
        else:
            relevance = target_scores_by_instance.get(matched_instance, 0.0)
            used_instances.add(matched_instance)

        predicted_relevances.append(relevance)

    ideal_relevances = sorted(
        target_scores_by_instance.values(),
        reverse=True,
    )

    dcg = dcg_at_k(predicted_relevances, k)
    ideal_dcg = dcg_at_k(ideal_relevances, k)

    if ideal_dcg <= 0:
        return 0.0

    return float(dcg / ideal_dcg)


def compute_integrated_vs_target_metrics(
    scene_id: str,
    image_id: str,
    target_name: str,
    target_features: list[dict],
    target_score_key: str,
    target_rank_key: str,
    integrated_features: list[dict],
    integrated_score_key: str,
    integrated_rank_key: str,
    iou_match_threshold: float,
) -> dict:
    target_order = rank_features(
        features=target_features,
        rank_key=target_rank_key,
        score_key=target_score_key,
    )

    integrated_order = rank_features(
        features=integrated_features,
        rank_key=integrated_rank_key,
        score_key=integrated_score_key,
    )

    target_lookup = build_target_lookup(
        target_features=target_features,
        target_score_key=target_score_key,
    )

    target_instances = [
        safe_int_or_none(feature.get("instance"))
        for feature in target_order
    ]
    target_instances = [item for item in target_instances if item is not None]

    integrated_matched_instances = [
        safe_int_or_none(feature.get("matched_gt_instance"))
        for feature in integrated_order
    ]

    integrated_matched_ious = [
        safe_float(feature.get("matched_gt_iou"), default=0.0)
        for feature in integrated_order
    ]

    target_top_instance = target_instances[0] if target_instances else None
    integrated_top_matched_instance = integrated_matched_instances[0] if integrated_matched_instances else None

    target_top_score = (
        target_lookup[target_top_instance]["score"]
        if target_top_instance in target_lookup
        else 0.0
    )

    integrated_top_target_score = (
        target_lookup[integrated_top_matched_instance]["score"]
        if integrated_top_matched_instance in target_lookup
        else 0.0
    )

    top1_match = int(
        target_top_instance is not None
        and integrated_top_matched_instance is not None
        and target_top_instance == integrated_top_matched_instance
    )

    target_top3 = set(target_instances[:3])
    target_top5 = set(target_instances[:5])

    integrated_top3 = set(
        item for item in integrated_matched_instances[:3]
        if item is not None
    )

    integrated_top5 = set(
        item for item in integrated_matched_instances[:5]
        if item is not None
    )

    target_denominator_3 = max(1, min(3, len(target_instances)))
    target_denominator_5 = max(1, min(5, len(target_instances)))

    top3_overlap_count = len(target_top3.intersection(integrated_top3))
    top5_overlap_count = len(target_top5.intersection(integrated_top5))

    target_top1_in_integrated_top3 = int(
        target_top_instance is not None
        and target_top_instance in integrated_top3
    )

    integrated_top1_in_target_top3 = int(
        integrated_top_matched_instance is not None
        and integrated_top_matched_instance in target_top3
    )

    integrated_top1_in_target_top5 = int(
        integrated_top_matched_instance is not None
        and integrated_top_matched_instance in target_top5
    )

    if target_top_instance is None:
        predicted_rank_of_target_top = np.nan
        reciprocal_rank = 0.0
    elif target_top_instance in integrated_matched_instances:
        predicted_rank_of_target_top = integrated_matched_instances.index(target_top_instance) + 1
        reciprocal_rank = 1.0 / predicted_rank_of_target_top
    else:
        predicted_rank_of_target_top = np.nan
        reciprocal_rank = 0.0

    target_scores_by_instance = {
        instance: values["score"]
        for instance, values in target_lookup.items()
    }

    regret = target_top_score - integrated_top_target_score

    matched_unique_instances = set(
        safe_int_or_none(feature.get("matched_gt_instance"))
        for feature in integrated_features
        if safe_float(feature.get("matched_gt_iou"), default=0.0) >= iou_match_threshold
    )
    matched_unique_instances.discard(None)

    false_positive_count = sum(
        1
        for feature in integrated_features
        if safe_float(feature.get("matched_gt_iou"), default=0.0) < iou_match_threshold
    )

    missed_target_count = len(set(target_instances).difference(matched_unique_instances))

    matched_ious_above_threshold = [
        safe_float(feature.get("matched_gt_iou"), default=0.0)
        for feature in integrated_features
        if safe_float(feature.get("matched_gt_iou"), default=0.0) >= iou_match_threshold
    ]

    if matched_ious_above_threshold:
        mean_matched_iou = float(np.mean(matched_ious_above_threshold))
        median_matched_iou = float(np.median(matched_ious_above_threshold))
    else:
        mean_matched_iou = 0.0
        median_matched_iou = 0.0

    if integrated_order:
        integrated_top_iou = safe_float(integrated_order[0].get("matched_gt_iou"), default=0.0)
        integrated_top_score = safe_float(integrated_order[0].get(integrated_score_key), default=np.nan)
        integrated_top_part_id = safe_int_or_none(integrated_order[0].get("part_id"))
    else:
        integrated_top_iou = 0.0
        integrated_top_score = np.nan
        integrated_top_part_id = None

    return {
        "scene_id": scene_id,
        "image_id": image_id,
        "target_name": target_name,
        "target_candidate_count": len(target_features),
        "integrated_candidate_count": len(integrated_features),
        "candidate_count_difference": len(integrated_features) - len(target_features),
        "target_top_instance": target_top_instance,
        "integrated_top_matched_instance": integrated_top_matched_instance,
        "target_top_score": target_top_score,
        "integrated_top_target_score": integrated_top_target_score,
        "integrated_top_score": integrated_top_score,
        "integrated_top_part_id": integrated_top_part_id,
        "integrated_top_iou": integrated_top_iou,
        "top1_match": top1_match,
        "target_top1_in_integrated_top3": target_top1_in_integrated_top3,
        "integrated_top1_in_target_top3": integrated_top1_in_target_top3,
        "integrated_top1_in_target_top5": integrated_top1_in_target_top5,
        "top3_overlap_count": top3_overlap_count,
        "top3_overlap_fraction": top3_overlap_count / target_denominator_3,
        "top5_overlap_count": top5_overlap_count,
        "top5_overlap_fraction": top5_overlap_count / target_denominator_5,
        "predicted_rank_of_target_top": predicted_rank_of_target_top,
        "reciprocal_rank": reciprocal_rank,
        "ndcg_at_3": ndcg_at_k(
            target_scores_by_instance=target_scores_by_instance,
            integrated_matched_instances=integrated_matched_instances,
            k=3,
        ),
        "ndcg_at_5": ndcg_at_k(
            target_scores_by_instance=target_scores_by_instance,
            integrated_matched_instances=integrated_matched_instances,
            k=5,
        ),
        "regret": regret,
        "false_positive_count_at_iou_threshold": false_positive_count,
        "missed_target_count_at_iou_threshold": missed_target_count,
        "mean_matched_iou_at_threshold": mean_matched_iou,
        "median_matched_iou_at_threshold": median_matched_iou,
    }


def summarize_metrics(rows: list[dict], target_name: str) -> dict:
    target_rows = [
        row for row in rows
        if row["target_name"] == target_name
    ]

    if not target_rows:
        raise ValueError(f"No rows found for target: {target_name}")

    def mean(column: str) -> float:
        values = [
            safe_float(row[column])
            for row in target_rows
            if not np.isnan(safe_float(row[column]))
        ]

        if not values:
            return float("nan")

        return float(np.mean(values))

    def median(column: str) -> float:
        values = [
            safe_float(row[column])
            for row in target_rows
            if not np.isnan(safe_float(row[column]))
        ]

        if not values:
            return float("nan")

        return float(np.median(values))

    return {
        "target_name": target_name,
        "image_count": len(target_rows),
        "mean_target_candidate_count": mean("target_candidate_count"),
        "mean_integrated_candidate_count": mean("integrated_candidate_count"),
        "mean_candidate_count_difference": mean("candidate_count_difference"),
        "top1_agreement": mean("top1_match"),
        "target_top1_in_integrated_top3": mean("target_top1_in_integrated_top3"),
        "integrated_top1_in_target_top3": mean("integrated_top1_in_target_top3"),
        "integrated_top1_in_target_top5": mean("integrated_top1_in_target_top5"),
        "mean_top3_overlap_fraction": mean("top3_overlap_fraction"),
        "mean_top5_overlap_fraction": mean("top5_overlap_fraction"),
        "mean_reciprocal_rank": mean("reciprocal_rank"),
        "mean_predicted_rank_of_target_top": mean("predicted_rank_of_target_top"),
        "mean_ndcg_at_3": mean("ndcg_at_3"),
        "mean_ndcg_at_5": mean("ndcg_at_5"),
        "mean_regret": mean("regret"),
        "median_regret": median("regret"),
        "max_regret": float(np.max([safe_float(row["regret"]) for row in target_rows])),
        "mean_integrated_top_iou": mean("integrated_top_iou"),
        "mean_false_positive_count_at_iou_threshold": mean("false_positive_count_at_iou_threshold"),
        "mean_missed_target_count_at_iou_threshold": mean("missed_target_count_at_iou_threshold"),
        "mean_matched_iou_at_threshold": mean("mean_matched_iou_at_threshold"),
    }


def save_csv(rows: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        raise ValueError("No rows to save.")

    fieldnames = list(rows[0].keys())

    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_json(data: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w") as f:
        json.dump(data, f, indent=2)


def make_summary_plot(
    summaries: list[dict],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    labels = [
        "Integrated vs\nHeuristic",
        "Integrated vs\nKnown-Mask Model",
    ]

    metric_keys = [
        "top1_agreement",
        "target_top1_in_integrated_top3",
        "integrated_top1_in_target_top3",
        "mean_top3_overlap_fraction",
        "mean_reciprocal_rank",
        "mean_ndcg_at_3",
    ]

    metric_names = [
        "Top-1\nmatch",
        "Target #1 in\nintegrated top 3",
        "Integrated #1 in\ntarget top 3",
        "Top-3\noverlap",
        "MRR",
        "NDCG@3",
    ]

    values = np.array(
        [
            [summary[key] for key in metric_keys]
            for summary in summaries
        ],
        dtype=np.float32,
    )

    fig = plt.figure(figsize=(16, 10))
    grid = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.25)

    ax_metrics = fig.add_subplot(grid[0, :])
    ax_counts = fig.add_subplot(grid[1, 0])
    ax_text = fig.add_subplot(grid[1, 1])

    x = np.arange(len(metric_names))
    bar_width = 0.36

    ax_metrics.bar(
        x - bar_width / 2,
        values[0],
        width=bar_width,
        label=labels[0],
    )

    ax_metrics.bar(
        x + bar_width / 2,
        values[1],
        width=bar_width,
        label=labels[1],
    )

    ax_metrics.set_xticks(x)
    ax_metrics.set_xticklabels(metric_names)
    ax_metrics.set_ylim(0, 1.05)
    ax_metrics.set_ylabel("Metric value")
    ax_metrics.set_title("Integrated Pipeline Top-K Ranking Metrics", fontweight="bold")
    ax_metrics.grid(True, axis="y", alpha=0.3)
    ax_metrics.legend()

    count_labels = [
        "Mean YOLO\nextra/missing\ncandidates",
        "Mean false\npositives",
        "Mean missed\ntargets",
    ]

    count_values = [
        summaries[0]["mean_candidate_count_difference"],
        summaries[0]["mean_false_positive_count_at_iou_threshold"],
        summaries[0]["mean_missed_target_count_at_iou_threshold"],
    ]

    ax_counts.bar(count_labels, count_values)
    ax_counts.set_title("Segmentation Candidate-Set Effects", fontweight="bold")
    ax_counts.set_ylabel("Mean count per image")
    ax_counts.grid(True, axis="y", alpha=0.3)

    ax_text.axis("off")
    ax_text.set_title("Summary", fontweight="bold", loc="left")

    lines = []

    for summary in summaries:
        lines.extend(
            [
                f"{summary['target_name']}:",
                f"  images: {summary['image_count']}",
                f"  top-1 agreement: {summary['top1_agreement']:.4f}",
                f"  target #1 in integrated top 3: {summary['target_top1_in_integrated_top3']:.4f}",
                f"  integrated #1 in target top 3: {summary['integrated_top1_in_target_top3']:.4f}",
                f"  top-3 overlap: {summary['mean_top3_overlap_fraction']:.4f}",
                f"  NDCG@3: {summary['mean_ndcg_at_3']:.4f}",
                f"  mean regret: {summary['mean_regret']:.4f}",
                "",
            ]
        )

    lines.extend(
        [
            "Note:",
            "Metrics compare the integrated YOLO-mask",
            "pipeline to reference rankings. These are not",
            "real robot grasp-success metrics.",
        ]
    )

    ax_text.text(
        0.0,
        0.98,
        "\n".join(lines),
        va="top",
        fontsize=10.5,
        family="monospace",
    )

    fig.suptitle(
        "Integrated YOLO Segmentation + Pickability Ranker Evaluation",
        fontsize=16,
        fontweight="bold",
    )

    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close()


def print_summary(summaries: list[dict]) -> None:
    print("\nIntegrated pipeline summary:")

    for summary in summaries:
        print(f"\nTarget: {summary['target_name']}")
        print(f"Images: {summary['image_count']}")
        print(f"Mean target candidates: {summary['mean_target_candidate_count']:.2f}")
        print(f"Mean integrated candidates: {summary['mean_integrated_candidate_count']:.2f}")
        print(f"Mean candidate count difference: {summary['mean_candidate_count_difference']:.2f}")
        print(f"Top-1 agreement: {summary['top1_agreement']:.4f}")
        print(f"Target #1 in integrated top 3: {summary['target_top1_in_integrated_top3']:.4f}")
        print(f"Integrated #1 in target top 3: {summary['integrated_top1_in_target_top3']:.4f}")
        print(f"Integrated #1 in target top 5: {summary['integrated_top1_in_target_top5']:.4f}")
        print(f"Top-3 overlap: {summary['mean_top3_overlap_fraction']:.4f}")
        print(f"Top-5 overlap: {summary['mean_top5_overlap_fraction']:.4f}")
        print(f"MRR: {summary['mean_reciprocal_rank']:.4f}")
        print(f"NDCG@3: {summary['mean_ndcg_at_3']:.4f}")
        print(f"NDCG@5: {summary['mean_ndcg_at_5']:.4f}")
        print(f"Mean regret: {summary['mean_regret']:.4f}")
        print(f"Mean integrated top IoU: {summary['mean_integrated_top_iou']:.4f}")
        print(f"Mean false positives: {summary['mean_false_positive_count_at_iou_threshold']:.2f}")
        print(f"Mean missed targets: {summary['mean_missed_target_count_at_iou_threshold']:.2f}")


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

    parser.add_argument("--scene-ids", type=int, nargs="*", default=None)
    parser.add_argument("--max-images-per-scene", type=int, default=25)

    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.50)
    parser.add_argument("--max-det", type=int, default=80)
    parser.add_argument("--yolo-device", type=str, default="0")
    parser.add_argument("--iou-match-threshold", type=float, default=0.50)

    parser.add_argument(
        "--visibility-proxy",
        type=str,
        default="confidence",
        choices=["confidence", "one"],
    )

    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("outputs/integrated_yolo_ranker/reports/integrated_comparison_image_metrics.csv"),
    )

    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("outputs/integrated_yolo_ranker/reports/integrated_comparison_summary.json"),
    )

    parser.add_argument(
        "--output-figure",
        type=Path,
        default=Path("outputs/integrated_yolo_ranker/figures/integrated_comparison_summary.png"),
    )

    args = parser.parse_args()

    torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Scoring integrated YOLO segmentation + pickability ranker")
    print(f"YOLO model: {args.yolo_model_path}")
    print(f"Ranker model: {args.ranker_model_path}")
    print(f"Dataset root: {args.dataset_root}")
    print(f"imgsz={args.imgsz}, conf={args.conf}, iou={args.iou}")
    print(f"IoU match threshold: {args.iou_match_threshold}")
    print(f"Max images per scene: {args.max_images_per_scene}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    yolo_model = YOLO(str(args.yolo_model_path))

    ranker_model, feature_columns, medians, mean, std = load_pickability_ranker(
        model_path=args.ranker_model_path,
        device=torch_device,
    )

    class_index_to_part_id = load_class_mapping(args.class_mapping)

    scene_dirs = get_scene_dirs(
        dataset_root=args.dataset_root,
        scene_ids=args.scene_ids,
    )

    all_metric_rows = []

    for scene_dir in scene_dirs:
        scene_id = int(scene_dir.name)
        image_ids = get_image_ids(
            scene_dir=scene_dir,
            max_images_per_scene=args.max_images_per_scene,
        )

        print(f"\nProcessing scene {scene_id:06d}: {len(image_ids)} images")

        for image_id in image_ids:
            try:
                scene_data = load_scene_inputs(
                    dataset_root=args.dataset_root,
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

                yolo_instances = predict_yolo_instances(
                    yolo_model=yolo_model,
                    image_path=image_path,
                    image_shape=image_rgb.shape,
                    class_index_to_part_id=class_index_to_part_id,
                    imgsz=args.imgsz,
                    conf=args.conf,
                    iou=args.iou,
                    max_det=args.max_det,
                    device=args.yolo_device,
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
                    visibility_proxy=args.visibility_proxy,
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

                row_vs_heuristic = compute_integrated_vs_target_metrics(
                    scene_id=f"{scene_id:06d}",
                    image_id=f"{image_id:06d}",
                    target_name="heuristic_dataset_masks",
                    target_features=heuristic_features,
                    target_score_key="heuristic_pickability_score",
                    target_rank_key="heuristic_rank",
                    integrated_features=integrated_features,
                    integrated_score_key="integrated_predicted_score",
                    integrated_rank_key="integrated_rank",
                    iou_match_threshold=args.iou_match_threshold,
                )

                row_vs_known_model = compute_integrated_vs_target_metrics(
                    scene_id=f"{scene_id:06d}",
                    image_id=f"{image_id:06d}",
                    target_name="known_mask_ranker",
                    target_features=known_mask_model_features,
                    target_score_key="known_mask_predicted_score",
                    target_rank_key="known_mask_rank",
                    integrated_features=integrated_features,
                    integrated_score_key="integrated_predicted_score",
                    integrated_rank_key="integrated_rank",
                    iou_match_threshold=args.iou_match_threshold,
                )

                all_metric_rows.append(row_vs_heuristic)
                all_metric_rows.append(row_vs_known_model)

                print(
                    f"  image {image_id:06d}: "
                    f"GT={len(heuristic_features)}, YOLO={len(yolo_instances)}, "
                    f"vs heuristic top1={row_vs_heuristic['top1_match']}, "
                    f"vs known-mask top1={row_vs_known_model['top1_match']}"
                )

            except Exception as error:
                print(f"  image {image_id:06d}: skipped because {error}")

    if not all_metric_rows:
        raise ValueError("No metric rows were generated.")

    summaries = [
        summarize_metrics(all_metric_rows, "heuristic_dataset_masks"),
        summarize_metrics(all_metric_rows, "known_mask_ranker"),
    ]

    save_csv(all_metric_rows, args.output_csv)

    summary_payload = {
        "settings": {
            "yolo_model_path": str(args.yolo_model_path),
            "ranker_model_path": str(args.ranker_model_path),
            "dataset_root": str(args.dataset_root),
            "imgsz": args.imgsz,
            "conf": args.conf,
            "iou": args.iou,
            "max_det": args.max_det,
            "iou_match_threshold": args.iou_match_threshold,
            "visibility_proxy": args.visibility_proxy,
            "max_images_per_scene": args.max_images_per_scene,
        },
        "summaries": summaries,
        "note": (
            "These metrics compare the integrated YOLO-mask pipeline to reference "
            "rankings from the heuristic and the known-mask learned ranker. They "
            "do not measure real robot grasp-success."
        ),
    }

    save_json(summary_payload, args.output_json)

    make_summary_plot(
        summaries=summaries,
        output_path=args.output_figure,
    )

    print_summary(summaries)

    print("\nSaved outputs:")
    print(f"Per-image metrics CSV: {args.output_csv}")
    print(f"Summary JSON: {args.output_json}")
    print(f"Summary figure: {args.output_figure}")


if __name__ == "__main__":
    main()