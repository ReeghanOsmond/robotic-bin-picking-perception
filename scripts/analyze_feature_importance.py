from pathlib import Path
import argparse
import csv
import json
import random
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


TARGET_COLUMN = "heuristic_pickability_score"


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


def load_csv_rows(csv_path: Path) -> list[dict]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Could not find CSV: {csv_path}")

    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        raise ValueError(f"No rows found in CSV: {csv_path}")

    return rows


def load_checkpoint(model_path: Path, device: torch.device):
    if not model_path.exists():
        raise FileNotFoundError(
            f"Could not find trained model: {model_path}\n"
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


def group_key(row: dict) -> tuple[str, str]:
    return str(row["scene_id"]), str(row["image_id"])


def split_by_image_group(
    rows: list[dict],
    val_fraction: float,
    seed: int,
) -> tuple[list[dict], list[dict]]:
    """
    Use the same image-group split style as training so candidates from the
    same image do not appear in both train and validation.
    """
    groups = sorted({group_key(row) for row in rows})

    if len(groups) < 2:
        raise ValueError("Need at least two scene/image groups.")

    rng = random.Random(seed)
    rng.shuffle(groups)

    val_count = max(1, int(round(len(groups) * val_fraction)))
    val_groups = set(groups[:val_count])

    train_rows = []
    val_rows = []

    for row in rows:
        if group_key(row) in val_groups:
            val_rows.append(row)
        else:
            train_rows.append(row)

    if not train_rows or not val_rows:
        raise ValueError("Train/validation split failed. Adjust val_fraction.")

    return train_rows, val_rows


def build_feature_matrix(
    rows: list[dict],
    feature_columns: list[str],
) -> np.ndarray:
    matrix = []

    for row in rows:
        values = []

        for column in feature_columns:
            values.append(safe_float(row.get(column), default=np.nan))

        matrix.append(values)

    return np.array(matrix, dtype=np.float32)


def build_target_vector(rows: list[dict]) -> np.ndarray:
    targets = []

    for row in rows:
        target = safe_float(row.get(TARGET_COLUMN), default=np.nan)

        if np.isnan(target):
            target = safe_float(row.get("pickability_score"), default=np.nan)

        targets.append(target)

    targets = np.array(targets, dtype=np.float32)

    if np.isnan(targets).any():
        raise ValueError(
            "Target contains NaNs. Rebuild outputs/reports/pick_candidate_features.csv."
        )

    return targets.reshape(-1)


def apply_preprocessor(
    x: np.ndarray,
    medians: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    x = np.where(np.isnan(x), medians, x)
    x = (x - mean) / std
    return x.astype(np.float32)


def predict_scores(
    model: PickabilityRanker,
    x: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    model.eval()

    with torch.no_grad():
        x_tensor = torch.tensor(x, dtype=torch.float32).to(device)
        predictions = model(x_tensor).detach().cpu().numpy().reshape(-1)

    return predictions


def group_indices_by_image(rows: list[dict]) -> dict[tuple[str, str], list[int]]:
    groups = defaultdict(list)

    for index, row in enumerate(rows):
        groups[group_key(row)].append(index)

    return dict(groups)


def compute_ranking_metrics(
    rows: list[dict],
    target_scores: np.ndarray,
    predicted_scores: np.ndarray,
) -> dict:
    groups = group_indices_by_image(rows)

    top1_matches = []
    target_top1_in_predicted_top3 = []
    top3_overlap_fractions = []
    reciprocal_ranks = []
    ndcg_at_3_values = []

    for indices in groups.values():
        indices = list(indices)

        target_group = target_scores[indices]
        predicted_group = predicted_scores[indices]

        target_order = np.argsort(target_group)[::-1]
        predicted_order = np.argsort(predicted_group)[::-1]

        target_top_local = int(target_order[0])
        predicted_top_local = int(predicted_order[0])

        top1_matches.append(1 if target_top_local == predicted_top_local else 0)

        predicted_top3 = set(int(i) for i in predicted_order[:3])
        target_top3 = set(int(i) for i in target_order[:3])

        target_top1_in_predicted_top3.append(
            1 if target_top_local in predicted_top3 else 0
        )

        denominator = max(1, min(3, len(indices)))
        top3_overlap_fractions.append(
            len(target_top3.intersection(predicted_top3)) / denominator
        )

        target_top_predicted_rank = int(np.where(predicted_order == target_top_local)[0][0]) + 1
        reciprocal_ranks.append(1.0 / target_top_predicted_rank)

        ndcg_at_3_values.append(
            compute_ndcg_at_k(
                target_scores=target_group,
                predicted_order=predicted_order,
                k=3,
            )
        )

    return {
        "top1_agreement": float(np.mean(top1_matches)),
        "target_top1_in_predicted_top3": float(np.mean(target_top1_in_predicted_top3)),
        "mean_top3_overlap_fraction": float(np.mean(top3_overlap_fractions)),
        "mean_reciprocal_rank": float(np.mean(reciprocal_ranks)),
        "mean_ndcg_at_3": float(np.mean(ndcg_at_3_values)),
    }


def compute_ndcg_at_k(
    target_scores: np.ndarray,
    predicted_order: np.ndarray,
    k: int,
) -> float:
    predicted_relevance = target_scores[predicted_order[:k]]
    ideal_relevance = np.sort(target_scores)[::-1][:k]

    dcg = 0.0
    ideal_dcg = 0.0

    for rank_index, relevance in enumerate(predicted_relevance, start=1):
        dcg += float(relevance) / np.log2(rank_index + 1)

    for rank_index, relevance in enumerate(ideal_relevance, start=1):
        ideal_dcg += float(relevance) / np.log2(rank_index + 1)

    if ideal_dcg <= 0:
        return 0.0

    return float(dcg / ideal_dcg)


def compute_all_metrics(
    rows: list[dict],
    target_scores: np.ndarray,
    predicted_scores: np.ndarray,
) -> dict:
    mae = float(np.mean(np.abs(target_scores - predicted_scores)))
    mse = float(np.mean((target_scores - predicted_scores) ** 2))
    rmse = float(np.sqrt(mse))

    if len(target_scores) > 1:
        pearson = float(np.corrcoef(target_scores, predicted_scores)[0, 1])
    else:
        pearson = 0.0

    target_mean = float(np.mean(target_scores))
    ss_res = float(np.sum((target_scores - predicted_scores) ** 2))
    ss_tot = float(np.sum((target_scores - target_mean) ** 2))

    if ss_tot <= 0:
        r2 = 0.0
    else:
        r2 = float(1.0 - ss_res / ss_tot)

    ranking_metrics = compute_ranking_metrics(
        rows=rows,
        target_scores=target_scores,
        predicted_scores=predicted_scores,
    )

    metrics = {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "pearson_correlation": pearson,
        "r2": r2,
    }

    metrics.update(ranking_metrics)

    return metrics


def permutation_importance(
    model: PickabilityRanker,
    val_rows: list[dict],
    val_x_raw: np.ndarray,
    val_y: np.ndarray,
    feature_columns: list[str],
    medians: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    device: torch.device,
    repeats: int,
    seed: int,
) -> tuple[list[dict], dict]:
    rng = np.random.default_rng(seed)

    val_x = apply_preprocessor(val_x_raw, medians, mean, std)
    baseline_predictions = predict_scores(model, val_x, device)

    baseline_metrics = compute_all_metrics(
        rows=val_rows,
        target_scores=val_y,
        predicted_scores=baseline_predictions,
    )

    importance_rows = []

    for feature_index, feature_name in enumerate(feature_columns):
        repeat_results = []

        for repeat_index in range(repeats):
            permuted_x_raw = val_x_raw.copy()

            shuffled_values = permuted_x_raw[:, feature_index].copy()
            rng.shuffle(shuffled_values)

            permuted_x_raw[:, feature_index] = shuffled_values
            permuted_x = apply_preprocessor(permuted_x_raw, medians, mean, std)

            permuted_predictions = predict_scores(model, permuted_x, device)

            permuted_metrics = compute_all_metrics(
                rows=val_rows,
                target_scores=val_y,
                predicted_scores=permuted_predictions,
            )

            repeat_results.append(
                {
                    "mae_increase": permuted_metrics["mae"] - baseline_metrics["mae"],
                    "rmse_increase": permuted_metrics["rmse"] - baseline_metrics["rmse"],
                    "r2_drop": baseline_metrics["r2"] - permuted_metrics["r2"],
                    "pearson_drop": baseline_metrics["pearson_correlation"] - permuted_metrics["pearson_correlation"],
                    "top1_drop": baseline_metrics["top1_agreement"] - permuted_metrics["top1_agreement"],
                    "top3_contains_target_drop": baseline_metrics["target_top1_in_predicted_top3"] - permuted_metrics["target_top1_in_predicted_top3"],
                    "top3_overlap_drop": baseline_metrics["mean_top3_overlap_fraction"] - permuted_metrics["mean_top3_overlap_fraction"],
                    "ndcg_at_3_drop": baseline_metrics["mean_ndcg_at_3"] - permuted_metrics["mean_ndcg_at_3"],
                }
            )

        importance_row = {
            "feature": feature_name,
            "mae_increase_mean": mean_metric(repeat_results, "mae_increase"),
            "mae_increase_std": std_metric(repeat_results, "mae_increase"),
            "rmse_increase_mean": mean_metric(repeat_results, "rmse_increase"),
            "rmse_increase_std": std_metric(repeat_results, "rmse_increase"),
            "r2_drop_mean": mean_metric(repeat_results, "r2_drop"),
            "r2_drop_std": std_metric(repeat_results, "r2_drop"),
            "pearson_drop_mean": mean_metric(repeat_results, "pearson_drop"),
            "pearson_drop_std": std_metric(repeat_results, "pearson_drop"),
            "top1_drop_mean": mean_metric(repeat_results, "top1_drop"),
            "top1_drop_std": std_metric(repeat_results, "top1_drop"),
            "top3_contains_target_drop_mean": mean_metric(repeat_results, "top3_contains_target_drop"),
            "top3_contains_target_drop_std": std_metric(repeat_results, "top3_contains_target_drop"),
            "top3_overlap_drop_mean": mean_metric(repeat_results, "top3_overlap_drop"),
            "top3_overlap_drop_std": std_metric(repeat_results, "top3_overlap_drop"),
            "ndcg_at_3_drop_mean": mean_metric(repeat_results, "ndcg_at_3_drop"),
            "ndcg_at_3_drop_std": std_metric(repeat_results, "ndcg_at_3_drop"),
        }

        importance_rows.append(importance_row)

        print(
            f"{feature_name:>20s} | "
            f"MAE +{importance_row['mae_increase_mean']:.5f} | "
            f"top1 drop {importance_row['top1_drop_mean']:.5f} | "
            f"NDCG@3 drop {importance_row['ndcg_at_3_drop_mean']:.5f}"
        )

    importance_rows = sorted(
        importance_rows,
        key=lambda row: row["mae_increase_mean"],
        reverse=True,
    )

    return importance_rows, baseline_metrics


def mean_metric(rows: list[dict], key: str) -> float:
    return float(np.mean([row[key] for row in rows]))


def std_metric(rows: list[dict], key: str) -> float:
    return float(np.std([row[key] for row in rows]))


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


def make_importance_plot(
    importance_rows: list[dict],
    baseline_metrics: dict,
    output_path: Path,
    top_n: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    top_rows = importance_rows[:top_n]

    features = [row["feature"] for row in top_rows]

    mae_importance = np.array(
        [row["mae_increase_mean"] for row in top_rows],
        dtype=np.float32,
    )

    mae_std = np.array(
        [row["mae_increase_std"] for row in top_rows],
        dtype=np.float32,
    )

    top1_drop = np.array(
        [row["top1_drop_mean"] for row in top_rows],
        dtype=np.float32,
    )

    ndcg_drop = np.array(
        [row["ndcg_at_3_drop_mean"] for row in top_rows],
        dtype=np.float32,
    )

    fig = plt.figure(figsize=(15, 10))
    grid = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)

    ax_mae = fig.add_subplot(grid[0, 0])
    ax_top1 = fig.add_subplot(grid[0, 1])
    ax_ndcg = fig.add_subplot(grid[1, 0])
    ax_text = fig.add_subplot(grid[1, 1])

    y_positions = np.arange(len(features))

    ax_mae.barh(y_positions, mae_importance, xerr=mae_std)
    ax_mae.set_yticks(y_positions)
    ax_mae.set_yticklabels(features)
    ax_mae.invert_yaxis()
    ax_mae.set_xlabel("MAE increase after permutation")
    ax_mae.set_title("Feature Importance by Score Error", fontweight="bold")
    ax_mae.grid(True, axis="x", alpha=0.3)

    ax_top1.barh(y_positions, top1_drop)
    ax_top1.set_yticks(y_positions)
    ax_top1.set_yticklabels(features)
    ax_top1.invert_yaxis()
    ax_top1.set_xlabel("Top-1 agreement drop")
    ax_top1.set_title("Feature Importance by Top-1 Ranking", fontweight="bold")
    ax_top1.grid(True, axis="x", alpha=0.3)

    ax_ndcg.barh(y_positions, ndcg_drop)
    ax_ndcg.set_yticks(y_positions)
    ax_ndcg.set_yticklabels(features)
    ax_ndcg.invert_yaxis()
    ax_ndcg.set_xlabel("NDCG@3 drop")
    ax_ndcg.set_title("Feature Importance by Top-3 Ranking Quality", fontweight="bold")
    ax_ndcg.grid(True, axis="x", alpha=0.3)

    ax_text.axis("off")
    ax_text.set_title("Baseline Validation Metrics", fontweight="bold", loc="left")

    summary_lines = [
        f"MAE: {baseline_metrics['mae']:.4f}",
        f"RMSE: {baseline_metrics['rmse']:.4f}",
        f"R²: {baseline_metrics['r2']:.4f}",
        f"Pearson r: {baseline_metrics['pearson_correlation']:.4f}",
        "",
        f"Top-1 agreement: {baseline_metrics['top1_agreement']:.4f}",
        f"Target #1 in model top 3: {baseline_metrics['target_top1_in_predicted_top3']:.4f}",
        f"Top-3 overlap: {baseline_metrics['mean_top3_overlap_fraction']:.4f}",
        f"MRR: {baseline_metrics['mean_reciprocal_rank']:.4f}",
        f"NDCG@3: {baseline_metrics['mean_ndcg_at_3']:.4f}",
        "",
        "Permutation importance:",
        "Each feature is shuffled across validation",
        "candidates. Larger metric degradation means",
        "the model relied more on that feature.",
    ]

    ax_text.text(
        0.0,
        0.95,
        "\n".join(summary_lines),
        va="top",
        fontsize=11,
        family="monospace",
    )

    fig.suptitle(
        "Learned Pickability Ranker: Permutation Feature Importance",
        fontsize=16,
        fontweight="bold",
    )

    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close()


def print_top_features(importance_rows: list[dict], top_n: int) -> None:
    print("\nTop features by MAE increase:")
    for row in importance_rows[:top_n]:
        print(
            f"{row['feature']:>20s} | "
            f"MAE increase: {row['mae_increase_mean']:.5f} ± {row['mae_increase_std']:.5f} | "
            f"top1 drop: {row['top1_drop_mean']:.5f} | "
            f"NDCG@3 drop: {row['ndcg_at_3_drop_mean']:.5f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path("outputs/reports/pick_candidate_features.csv"),
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("outputs/models/pickability_ranker.pt"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("outputs/reports/feature_importance.csv"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("outputs/reports/feature_importance_summary.json"),
    )
    parser.add_argument(
        "--output-figure",
        type=Path,
        default=Path("outputs/figures/training/feature_importance.png"),
    )
    parser.add_argument("--val-fraction", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--top-n", type=int, default=12)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    model, feature_columns, medians, mean, std = load_checkpoint(
        model_path=args.model_path,
        device=device,
    )

    rows = load_csv_rows(args.input_csv)

    _, val_rows = split_by_image_group(
        rows=rows,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )

    val_x_raw = build_feature_matrix(val_rows, feature_columns)
    val_y = build_target_vector(val_rows)

    print(f"Validation candidate rows: {len(val_rows)}")
    print(f"Validation image groups: {len(group_indices_by_image(val_rows))}")
    print(f"Features: {len(feature_columns)}")
    print(f"Permutation repeats per feature: {args.repeats}")

    importance_rows, baseline_metrics = permutation_importance(
        model=model,
        val_rows=val_rows,
        val_x_raw=val_x_raw,
        val_y=val_y,
        feature_columns=feature_columns,
        medians=medians,
        mean=mean,
        std=std,
        device=device,
        repeats=args.repeats,
        seed=args.seed,
    )

    summary = {
        "input_csv": str(args.input_csv),
        "model_path": str(args.model_path),
        "validation_candidate_rows": len(val_rows),
        "validation_image_groups": len(group_indices_by_image(val_rows)),
        "feature_count": len(feature_columns),
        "repeats": args.repeats,
        "baseline_metrics": baseline_metrics,
        "top_features_by_mae_increase": importance_rows[: args.top_n],
        "note": (
            "Permutation importance measures how much validation performance "
            "degrades when one feature is shuffled. This compares the learned "
            "model against the heuristic pickability target, not real robot "
            "grasp-success labels."
        ),
    }

    save_csv(importance_rows, args.output_csv)
    save_json(summary, args.output_json)

    make_importance_plot(
        importance_rows=importance_rows,
        baseline_metrics=baseline_metrics,
        output_path=args.output_figure,
        top_n=args.top_n,
    )

    print_top_features(importance_rows, top_n=args.top_n)

    print("\nSaved outputs:")
    print(f"Feature importance CSV: {args.output_csv}")
    print(f"Feature importance JSON: {args.output_json}")
    print(f"Feature importance figure: {args.output_figure}")


if __name__ == "__main__":
    main()