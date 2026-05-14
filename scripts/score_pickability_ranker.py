from pathlib import Path
import argparse
import csv
import json
import math
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np


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


def safe_str(value) -> str:
    if value is None:
        return ""
    return str(value).strip()


def load_csv_rows(csv_path: Path) -> list[dict]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Could not find CSV: {csv_path}")

    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        raise ValueError(f"No rows found in CSV: {csv_path}")

    return rows


def find_first_available_column(rows: list[dict], candidates: list[str]) -> str:
    available = set(rows[0].keys())

    for candidate in candidates:
        if candidate in available:
            return candidate

    raise KeyError(
        f"None of these columns were found: {candidates}\n"
        f"Available columns: {sorted(available)}"
    )


def group_key(row: dict) -> tuple[str, str]:
    return safe_str(row["scene_id"]), safe_str(row["image_id"])


def group_rows_by_image(rows: list[dict]) -> dict[tuple[str, str], list[dict]]:
    groups = defaultdict(list)

    for row in rows:
        groups[group_key(row)].append(row)

    return dict(groups)


def get_candidate_id(row: dict) -> str:
    """
    Instance is the specific visible object copy in an image.
    Fall back to object_index if needed.
    """
    if "instance" in row:
        return safe_str(row["instance"])

    if "object_index" in row:
        return safe_str(row["object_index"])

    raise KeyError("Could not find candidate ID column: expected instance or object_index.")


def rank_group_by_score(
    group: list[dict],
    score_column: str,
) -> list[dict]:
    return sorted(
        group,
        key=lambda row: (
            safe_float(row[score_column], default=-np.inf),
            get_candidate_id(row),
        ),
        reverse=True,
    )


def rank_group_by_rank_column(
    group: list[dict],
    rank_column: str,
) -> list[dict]:
    return sorted(
        group,
        key=lambda row: (
            safe_float(row[rank_column], default=np.inf),
            get_candidate_id(row),
        ),
    )


def get_ranked_group(
    group: list[dict],
    score_column: str,
    rank_column: str | None,
) -> list[dict]:
    if rank_column is not None and rank_column in group[0]:
        values = [safe_float(row.get(rank_column), default=np.nan) for row in group]

        if not np.isnan(values).all():
            return rank_group_by_rank_column(group, rank_column)

    return rank_group_by_score(group, score_column)


def dcg_at_k(relevances: list[float], k: int) -> float:
    score = 0.0

    for rank_index, relevance in enumerate(relevances[:k], start=1):
        score += relevance / math.log2(rank_index + 1)

    return float(score)


def ndcg_at_k(
    target_relevance_by_candidate: dict[str, float],
    predicted_order_ids: list[str],
    k: int,
) -> float:
    predicted_relevances = [
        target_relevance_by_candidate[candidate_id]
        for candidate_id in predicted_order_ids
        if candidate_id in target_relevance_by_candidate
    ]

    ideal_relevances = sorted(
        target_relevance_by_candidate.values(),
        reverse=True,
    )

    dcg = dcg_at_k(predicted_relevances, k)
    ideal_dcg = dcg_at_k(ideal_relevances, k)

    if ideal_dcg <= 0:
        return 0.0

    return float(dcg / ideal_dcg)


def compute_image_level_metrics(
    rows: list[dict],
    target_score_column: str,
    predicted_score_column: str,
    target_rank_column: str | None,
    predicted_rank_column: str | None,
) -> list[dict]:
    groups = group_rows_by_image(rows)
    image_metrics = []

    for (scene_id, image_id), group in groups.items():
        if len(group) < 1:
            continue

        target_order = get_ranked_group(
            group=group,
            score_column=target_score_column,
            rank_column=target_rank_column,
        )

        predicted_order = get_ranked_group(
            group=group,
            score_column=predicted_score_column,
            rank_column=predicted_rank_column,
        )

        target_ids = [get_candidate_id(row) for row in target_order]
        predicted_ids = [get_candidate_id(row) for row in predicted_order]

        target_top_id = target_ids[0]
        predicted_top_id = predicted_ids[0]

        top1_match = int(target_top_id == predicted_top_id)

        predicted_top3 = set(predicted_ids[:3])
        target_top3 = set(target_ids[:3])

        target_top1_in_predicted_top3 = int(target_top_id in predicted_top3)
        predicted_top1_in_target_top3 = int(predicted_top_id in target_top3)

        top3_denominator = max(1, min(3, len(group)))
        top3_overlap_count = len(target_top3.intersection(predicted_top3))
        top3_overlap_fraction = top3_overlap_count / top3_denominator

        top5_denominator = max(1, min(5, len(group)))
        predicted_top5 = set(predicted_ids[:5])
        target_top5 = set(target_ids[:5])
        top5_overlap_count = len(target_top5.intersection(predicted_top5))
        top5_overlap_fraction = top5_overlap_count / top5_denominator

        predicted_position_of_target_top = predicted_ids.index(target_top_id) + 1
        reciprocal_rank = 1.0 / predicted_position_of_target_top

        target_score_by_id = {
            get_candidate_id(row): safe_float(row[target_score_column])
            for row in group
        }

        predicted_score_by_id = {
            get_candidate_id(row): safe_float(row[predicted_score_column])
            for row in group
        }

        target_top_score = target_score_by_id[target_top_id]
        predicted_top_target_score = target_score_by_id[predicted_top_id]
        predicted_top_model_score = predicted_score_by_id[predicted_top_id]

        # Regret measures how much target score was lost by choosing the model's top pick
        # instead of the heuristic target's top pick.
        regret = target_top_score - predicted_top_target_score

        image_metrics.append(
            {
                "scene_id": scene_id,
                "image_id": image_id,
                "candidate_count": len(group),
                "target_top_instance": target_top_id,
                "predicted_top_instance": predicted_top_id,
                "target_top_score": target_top_score,
                "predicted_top_target_score": predicted_top_target_score,
                "predicted_top_model_score": predicted_top_model_score,
                "top1_match": top1_match,
                "target_top1_in_predicted_top3": target_top1_in_predicted_top3,
                "predicted_top1_in_target_top3": predicted_top1_in_target_top3,
                "top3_overlap_count": top3_overlap_count,
                "top3_overlap_fraction": top3_overlap_fraction,
                "top5_overlap_count": top5_overlap_count,
                "top5_overlap_fraction": top5_overlap_fraction,
                "predicted_rank_of_target_top": predicted_position_of_target_top,
                "reciprocal_rank": reciprocal_rank,
                "regret": regret,
                "ndcg_at_3": ndcg_at_k(
                    target_relevance_by_candidate=target_score_by_id,
                    predicted_order_ids=predicted_ids,
                    k=3,
                ),
                "ndcg_at_5": ndcg_at_k(
                    target_relevance_by_candidate=target_score_by_id,
                    predicted_order_ids=predicted_ids,
                    k=5,
                ),
            }
        )

    return sorted(
        image_metrics,
        key=lambda row: (row["scene_id"], row["image_id"]),
    )


def compute_row_level_metrics(
    rows: list[dict],
    target_score_column: str,
    predicted_score_column: str,
) -> dict:
    target_scores = np.array(
        [safe_float(row[target_score_column]) for row in rows],
        dtype=np.float32,
    )

    predicted_scores = np.array(
        [safe_float(row[predicted_score_column]) for row in rows],
        dtype=np.float32,
    )

    valid = ~np.isnan(target_scores) & ~np.isnan(predicted_scores)

    target_scores = target_scores[valid]
    predicted_scores = predicted_scores[valid]

    if len(target_scores) == 0:
        raise ValueError("No valid target/predicted score pairs found.")

    mae = float(np.mean(np.abs(target_scores - predicted_scores)))
    mse = float(np.mean((target_scores - predicted_scores) ** 2))
    rmse = float(np.sqrt(mse))

    target_mean = float(np.mean(target_scores))
    ss_res = float(np.sum((target_scores - predicted_scores) ** 2))
    ss_tot = float(np.sum((target_scores - target_mean) ** 2))

    if ss_tot <= 0:
        r2 = 0.0
    else:
        r2 = 1.0 - ss_res / ss_tot

    if len(target_scores) > 1:
        correlation = float(np.corrcoef(target_scores, predicted_scores)[0, 1])
    else:
        correlation = 0.0

    return {
        "row_count": int(len(target_scores)),
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2": float(r2),
        "pearson_correlation": correlation,
    }


def summarize_image_metrics(image_metrics: list[dict]) -> dict:
    if not image_metrics:
        raise ValueError("No image metrics to summarize.")

    def mean_of(column: str) -> float:
        return float(np.mean([safe_float(row[column]) for row in image_metrics]))

    def median_of(column: str) -> float:
        return float(np.median([safe_float(row[column]) for row in image_metrics]))

    return {
        "image_count": int(len(image_metrics)),
        "mean_candidate_count": mean_of("candidate_count"),
        "top1_agreement": mean_of("top1_match"),
        "target_top1_in_predicted_top3": mean_of("target_top1_in_predicted_top3"),
        "predicted_top1_in_target_top3": mean_of("predicted_top1_in_target_top3"),
        "mean_top3_overlap_fraction": mean_of("top3_overlap_fraction"),
        "mean_top5_overlap_fraction": mean_of("top5_overlap_fraction"),
        "mean_reciprocal_rank": mean_of("reciprocal_rank"),
        "mean_predicted_rank_of_target_top": mean_of("predicted_rank_of_target_top"),
        "mean_regret": mean_of("regret"),
        "median_regret": median_of("regret"),
        "max_regret": float(np.max([safe_float(row["regret"]) for row in image_metrics])),
        "mean_ndcg_at_3": mean_of("ndcg_at_3"),
        "mean_ndcg_at_5": mean_of("ndcg_at_5"),
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


def make_score_summary_figure(
    image_metrics: list[dict],
    row_summary: dict,
    image_summary: dict,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    top_metrics = {
        "Top-1 match": image_summary["top1_agreement"],
        "Target #1 in\nmodel top 3": image_summary["target_top1_in_predicted_top3"],
        "Top-3 overlap": image_summary["mean_top3_overlap_fraction"],
        "MRR": image_summary["mean_reciprocal_rank"],
        "NDCG@3": image_summary["mean_ndcg_at_3"],
    }

    regrets = np.array(
        [safe_float(row["regret"]) for row in image_metrics],
        dtype=np.float32,
    )

    predicted_rank_of_target_top = np.array(
        [safe_float(row["predicted_rank_of_target_top"]) for row in image_metrics],
        dtype=np.float32,
    )

    top3_overlap = np.array(
        [safe_float(row["top3_overlap_fraction"]) for row in image_metrics],
        dtype=np.float32,
    )

    fig = plt.figure(figsize=(15, 10))
    grid = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.25)

    ax_bar = fig.add_subplot(grid[0, 0])
    ax_rank_hist = fig.add_subplot(grid[0, 1])
    ax_overlap_hist = fig.add_subplot(grid[1, 0])
    ax_text = fig.add_subplot(grid[1, 1])

    ax_bar.bar(list(top_metrics.keys()), list(top_metrics.values()))
    ax_bar.set_ylim(0, 1.05)
    ax_bar.set_ylabel("Score")
    ax_bar.set_title("Image-Level Ranking Metrics", fontweight="bold")
    ax_bar.grid(True, axis="y", alpha=0.3)

    ax_rank_hist.hist(
        predicted_rank_of_target_top,
        bins=np.arange(1, np.nanmax(predicted_rank_of_target_top) + 2) - 0.5,
        rwidth=0.85,
    )
    ax_rank_hist.set_xlabel("Model-predicted rank of heuristic #1")
    ax_rank_hist.set_ylabel("Image count")
    ax_rank_hist.set_title("Where Did the Model Rank the Target Best Pick?", fontweight="bold")
    ax_rank_hist.grid(True, axis="y", alpha=0.3)

    ax_overlap_hist.hist(
        top3_overlap,
        bins=np.linspace(0, 1, 5),
        rwidth=0.85,
    )
    ax_overlap_hist.set_xlabel("Top-3 overlap fraction")
    ax_overlap_hist.set_ylabel("Image count")
    ax_overlap_hist.set_title("Model Top-3 vs Heuristic Top-3 Overlap", fontweight="bold")
    ax_overlap_hist.grid(True, axis="y", alpha=0.3)

    ax_text.axis("off")
    ax_text.set_title("Summary", fontweight="bold", loc="left")

    summary_lines = [
        f"Rows: {row_summary['row_count']}",
        f"Images: {image_summary['image_count']}",
        f"Mean candidates/image: {image_summary['mean_candidate_count']:.2f}",
        "",
        "Row-level score fit:",
        f"MAE: {row_summary['mae']:.4f}",
        f"RMSE: {row_summary['rmse']:.4f}",
        f"R²: {row_summary['r2']:.4f}",
        f"Pearson r: {row_summary['pearson_correlation']:.4f}",
        "",
        "Image-level ranking:",
        f"Top-1 agreement: {image_summary['top1_agreement']:.4f}",
        f"Target #1 in model top 3: {image_summary['target_top1_in_predicted_top3']:.4f}",
        f"Top-3 overlap: {image_summary['mean_top3_overlap_fraction']:.4f}",
        f"Mean reciprocal rank: {image_summary['mean_reciprocal_rank']:.4f}",
        f"NDCG@3: {image_summary['mean_ndcg_at_3']:.4f}",
        "",
        "Regret:",
        f"Mean regret: {image_summary['mean_regret']:.4f}",
        f"Median regret: {image_summary['median_regret']:.4f}",
        f"Max regret: {image_summary['max_regret']:.4f}",
    ]

    ax_text.text(
        0.0,
        0.98,
        "\n".join(summary_lines),
        va="top",
        fontsize=11,
        family="monospace",
    )

    fig.suptitle(
        "Learned Pickability Ranker: Validation Score Summary",
        fontsize=16,
        fontweight="bold",
    )

    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close()


def print_summary(row_summary: dict, image_summary: dict) -> None:
    print("\nRow-level score fit:")
    print(f"Rows: {row_summary['row_count']}")
    print(f"MAE: {row_summary['mae']:.4f}")
    print(f"RMSE: {row_summary['rmse']:.4f}")
    print(f"R²: {row_summary['r2']:.4f}")
    print(f"Pearson correlation: {row_summary['pearson_correlation']:.4f}")

    print("\nImage-level ranking metrics:")
    print(f"Images: {image_summary['image_count']}")
    print(f"Mean candidates/image: {image_summary['mean_candidate_count']:.2f}")
    print(f"Top-1 agreement: {image_summary['top1_agreement']:.4f}")
    print(f"Heuristic #1 in model top 3: {image_summary['target_top1_in_predicted_top3']:.4f}")
    print(f"Model #1 in heuristic top 3: {image_summary['predicted_top1_in_target_top3']:.4f}")
    print(f"Mean top-3 overlap fraction: {image_summary['mean_top3_overlap_fraction']:.4f}")
    print(f"Mean top-5 overlap fraction: {image_summary['mean_top5_overlap_fraction']:.4f}")
    print(f"Mean reciprocal rank: {image_summary['mean_reciprocal_rank']:.4f}")
    print(f"Mean predicted rank of heuristic #1: {image_summary['mean_predicted_rank_of_target_top']:.4f}")
    print(f"NDCG@3: {image_summary['mean_ndcg_at_3']:.4f}")
    print(f"NDCG@5: {image_summary['mean_ndcg_at_5']:.4f}")
    print(f"Mean regret: {image_summary['mean_regret']:.4f}")
    print(f"Median regret: {image_summary['median_regret']:.4f}")
    print(f"Max regret: {image_summary['max_regret']:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictions-csv",
        type=Path,
        default=Path("outputs/reports/pickability_ranker_predictions.csv"),
    )
    parser.add_argument(
        "--output-image-metrics-csv",
        type=Path,
        default=Path("outputs/reports/pickability_ranker_image_metrics.csv"),
    )
    parser.add_argument(
        "--output-summary-json",
        type=Path,
        default=Path("outputs/reports/pickability_ranker_score_summary.json"),
    )
    parser.add_argument(
        "--output-figure",
        type=Path,
        default=Path("outputs/figures/training/pickability_ranker_score_summary.png"),
    )
    args = parser.parse_args()

    rows = load_csv_rows(args.predictions_csv)

    target_score_column = find_first_available_column(
        rows,
        ["target_score", "heuristic_pickability_score", "pickability_score"],
    )

    predicted_score_column = find_first_available_column(
        rows,
        ["predicted_score", "predicted_pickability_score"],
    )

    target_rank_column = None
    for candidate in ["heuristic_rank", "pick_rank"]:
        if candidate in rows[0]:
            target_rank_column = candidate
            break

    predicted_rank_column = None
    if "predicted_rank" in rows[0]:
        predicted_rank_column = "predicted_rank"

    image_metrics = compute_image_level_metrics(
        rows=rows,
        target_score_column=target_score_column,
        predicted_score_column=predicted_score_column,
        target_rank_column=target_rank_column,
        predicted_rank_column=predicted_rank_column,
    )

    row_summary = compute_row_level_metrics(
        rows=rows,
        target_score_column=target_score_column,
        predicted_score_column=predicted_score_column,
    )

    image_summary = summarize_image_metrics(image_metrics)

    full_summary = {
        "input_csv": str(args.predictions_csv),
        "target_score_column": target_score_column,
        "predicted_score_column": predicted_score_column,
        "target_rank_column": target_rank_column,
        "predicted_rank_column": predicted_rank_column,
        "row_level": row_summary,
        "image_level": image_summary,
        "note": (
            "These scores compare the model to the heuristic pickability target. "
            "They do not measure real robot grasp-success accuracy."
        ),
    }

    save_csv(image_metrics, args.output_image_metrics_csv)
    save_json(full_summary, args.output_summary_json)
    make_score_summary_figure(
        image_metrics=image_metrics,
        row_summary=row_summary,
        image_summary=image_summary,
        output_path=args.output_figure,
    )

    print_summary(row_summary, image_summary)

    print("\nSaved outputs:")
    print(f"Image-level metrics CSV: {args.output_image_metrics_csv}")
    print(f"Score summary JSON: {args.output_summary_json}")
    print(f"Score summary figure: {args.output_figure}")


if __name__ == "__main__":
    main()