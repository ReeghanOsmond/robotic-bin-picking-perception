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
from torch.utils.data import DataLoader, TensorDataset


FEATURE_COLUMNS = [
    "visible_pixels",
    "valid_3d_points",
    "visible_fraction",
    "depth_min",
    "depth_median",
    "depth_max",
    "extent_x",
    "extent_y",
    "extent_z",
    "centroid_x",
    "centroid_y",
    "centroid_z",
    "pixel_center_x",
    "pixel_center_y",
    "bbox_x_min",
    "bbox_x_max",
    "bbox_y_min",
    "bbox_y_max",
    "part_id",
]

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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
        raise FileNotFoundError(f"Could not find feature CSV: {csv_path}")

    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        raise ValueError(f"No rows found in CSV: {csv_path}")

    return rows


def group_key(row: dict) -> tuple[str, str]:
    return str(row["scene_id"]), str(row["image_id"])


def split_by_image_group(
    rows: list[dict],
    val_fraction: float,
    seed: int,
) -> tuple[list[dict], list[dict]]:
    groups = sorted({group_key(row) for row in rows})

    if len(groups) < 2:
        raise ValueError("Need at least two scene/image groups for train/validation split.")

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

    return train_rows, val_rows


def group_indices_by_image(rows: list[dict]) -> dict[tuple[str, str], list[int]]:
    groups = defaultdict(list)

    for index, row in enumerate(rows):
        groups[group_key(row)].append(index)

    return dict(groups)


def build_feature_matrix(rows: list[dict], feature_columns: list[str]) -> np.ndarray:
    matrix = []

    for row in rows:
        matrix.append([safe_float(row.get(column), default=np.nan) for column in feature_columns])

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
        raise ValueError("Target column contains NaNs. Rebuild the candidate feature CSV.")

    return targets.reshape(-1, 1)


def fit_preprocessor(train_x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    medians = np.nanmedian(train_x, axis=0)
    medians = np.where(np.isnan(medians), 0.0, medians)

    imputed = np.where(np.isnan(train_x), medians, train_x)

    mean = imputed.mean(axis=0)
    std = imputed.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)

    return medians.astype(np.float32), mean.astype(np.float32), std.astype(np.float32)


def apply_preprocessor(
    x: np.ndarray,
    medians: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    x = np.where(np.isnan(x), medians, x)
    x = (x - mean) / std
    return x.astype(np.float32)


def train_model(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    device: torch.device,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    patience: int,
) -> tuple[PickabilityRanker, dict]:
    train_dataset = TensorDataset(
        torch.tensor(train_x, dtype=torch.float32),
        torch.tensor(train_y, dtype=torch.float32),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    model = PickabilityRanker(input_dim=train_x.shape[1]).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-4,
    )

    loss_fn = nn.SmoothL1Loss()

    val_x_tensor = torch.tensor(val_x, dtype=torch.float32).to(device)
    val_y_tensor = torch.tensor(val_y, dtype=torch.float32).to(device)

    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0

    history = {
        "train_loss": [],
        "val_loss": [],
    }

    for epoch in range(1, epochs + 1):
        model.train()
        batch_losses = []

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            predictions = model(batch_x)
            loss = loss_fn(predictions, batch_y)
            loss.backward()
            optimizer.step()

            batch_losses.append(float(loss.detach().cpu().item()))

        train_loss = float(np.mean(batch_losses))

        model.eval()
        with torch.no_grad():
            val_predictions = model(val_x_tensor)
            val_loss = float(loss_fn(val_predictions, val_y_tensor).detach().cpu().item())

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if epoch == 1 or epoch % 25 == 0:
            print(
                f"Epoch {epoch:04d} | "
                f"train_loss={train_loss:.5f} | "
                f"val_loss={val_loss:.5f}"
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch}.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history


def predict_scores(model: PickabilityRanker, x: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()

    with torch.no_grad():
        x_tensor = torch.tensor(x, dtype=torch.float32).to(device)
        predictions = model(x_tensor).detach().cpu().numpy().reshape(-1)

    return predictions


def compute_predicted_ranks(rows: list[dict], predicted_scores: np.ndarray) -> np.ndarray:
    predicted_ranks = np.zeros(len(rows), dtype=np.int32)
    groups = group_indices_by_image(rows)

    for indices in groups.values():
        scores = predicted_scores[indices]
        order = np.argsort(scores)[::-1]

        for rank_position, local_index in enumerate(order, start=1):
            global_index = indices[int(local_index)]
            predicted_ranks[global_index] = rank_position

    return predicted_ranks


def evaluate_rows(
    rows: list[dict],
    target_scores: np.ndarray,
    predicted_scores: np.ndarray,
) -> dict:
    target_scores = target_scores.reshape(-1)
    predicted_scores = predicted_scores.reshape(-1)

    mae = float(np.mean(np.abs(target_scores - predicted_scores)))
    mse = float(np.mean((target_scores - predicted_scores) ** 2))

    groups = group_indices_by_image(rows)

    top1_matches = []
    reciprocal_ranks = []

    for indices in groups.values():
        indices = list(indices)

        target_group = target_scores[indices]
        predicted_group = predicted_scores[indices]

        true_top_local_index = int(np.argmax(target_group))
        pred_order = np.argsort(predicted_group)[::-1]

        predicted_top_local_index = int(pred_order[0])
        top1_matches.append(1 if predicted_top_local_index == true_top_local_index else 0)

        true_rank_position = int(np.where(pred_order == true_top_local_index)[0][0]) + 1
        reciprocal_ranks.append(1.0 / true_rank_position)

    return {
        "row_count": int(len(rows)),
        "image_group_count": int(len(groups)),
        "mae": mae,
        "mse": mse,
        "top1_agreement": float(np.mean(top1_matches)),
        "mean_reciprocal_rank": float(np.mean(reciprocal_ranks)),
    }


def save_training_curve(history: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    epochs = np.arange(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(9, 6))
    plt.plot(epochs, history["train_loss"], label="train loss")
    plt.plot(epochs, history["val_loss"], label="validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Smooth L1 loss")
    plt.title("Pickability Ranker Training Curve", fontweight="bold")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def save_final_validation_plot(
    history: dict,
    val_rows: list[dict],
    val_y: np.ndarray,
    predicted_scores: np.ndarray,
    metrics: dict,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    target_scores = val_y.reshape(-1)
    predicted_scores = predicted_scores.reshape(-1)

    heuristic_ranks = np.array(
        [safe_float(row.get("heuristic_rank"), default=np.nan) for row in val_rows],
        dtype=np.float32,
    )

    predicted_ranks = compute_predicted_ranks(val_rows, predicted_scores).astype(np.float32)

    epochs = np.arange(1, len(history["train_loss"]) + 1)

    fig = plt.figure(figsize=(15, 10))
    grid = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.28)

    ax_loss = fig.add_subplot(grid[0, 0])
    ax_score = fig.add_subplot(grid[0, 1])
    ax_rank = fig.add_subplot(grid[1, 0])
    ax_text = fig.add_subplot(grid[1, 1])

    ax_loss.plot(epochs, history["train_loss"], label="train loss")
    ax_loss.plot(epochs, history["val_loss"], label="validation loss")
    ax_loss.set_title("Training Curve", fontweight="bold")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Smooth L1 loss")
    ax_loss.grid(True, alpha=0.3)
    ax_loss.legend()

    ax_score.scatter(target_scores, predicted_scores, alpha=0.7, s=35)
    min_score = min(float(target_scores.min()), float(predicted_scores.min()))
    max_score = max(float(target_scores.max()), float(predicted_scores.max()))
    ax_score.plot([min_score, max_score], [min_score, max_score], linestyle="--", linewidth=1)
    ax_score.set_title("Validation: Target vs Predicted Score", fontweight="bold")
    ax_score.set_xlabel("Heuristic target score")
    ax_score.set_ylabel("Model predicted score")
    ax_score.grid(True, alpha=0.3)

    valid_rank_mask = ~np.isnan(heuristic_ranks)

    ax_rank.scatter(
        heuristic_ranks[valid_rank_mask],
        predicted_ranks[valid_rank_mask],
        alpha=0.7,
        s=35,
    )

    if valid_rank_mask.any():
        max_rank = max(
            float(np.nanmax(heuristic_ranks[valid_rank_mask])),
            float(np.nanmax(predicted_ranks[valid_rank_mask])),
        )
        ax_rank.plot([1, max_rank], [1, max_rank], linestyle="--", linewidth=1)

    ax_rank.set_title("Validation: Heuristic Rank vs Predicted Rank", fontweight="bold")
    ax_rank.set_xlabel("Heuristic rank")
    ax_rank.set_ylabel("Predicted rank")
    ax_rank.invert_xaxis()
    ax_rank.invert_yaxis()
    ax_rank.grid(True, alpha=0.3)

    ax_text.axis("off")
    ax_text.set_title("Validation Metrics", fontweight="bold", loc="left")

    metric_lines = [
        f"Rows: {metrics.get('row_count', 'n/a')}",
        f"Image groups: {metrics.get('image_group_count', 'n/a')}",
        f"MAE: {metrics.get('mae', float('nan')):.4f}",
        f"MSE: {metrics.get('mse', float('nan')):.4f}",
        f"Top-1 agreement: {metrics.get('top1_agreement', float('nan')):.4f}",
        f"Mean reciprocal rank: {metrics.get('mean_reciprocal_rank', float('nan')):.4f}",
        "",
        "Model:",
        "PyTorch MLP ranker trained on",
        "object-level geometry/depth features.",
        "",
        "Target:",
        "Heuristic pickability score.",
        "",
        "Limitation:",
        "Not trained on real robot grasp-success labels.",
    ]

    ax_text.text(
        0.0,
        0.95,
        "\n".join(metric_lines),
        va="top",
        fontsize=11,
        family="monospace",
    )

    fig.suptitle(
        "Learned Pickability Ranker: Final Validation Results",
        fontsize=16,
        fontweight="bold",
    )

    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close()


def add_prediction_columns(
    rows: list[dict],
    target_scores: np.ndarray,
    predicted_scores: np.ndarray,
) -> list[dict]:
    output_rows = []

    for row, target, prediction in zip(rows, target_scores.reshape(-1), predicted_scores.reshape(-1)):
        new_row = dict(row)
        new_row["target_score"] = float(target)
        new_row["predicted_score"] = float(prediction)
        output_rows.append(new_row)

    groups = group_indices_by_image(output_rows)

    for indices in groups.values():
        predicted = np.array([output_rows[index]["predicted_score"] for index in indices])
        order = np.argsort(predicted)[::-1]

        for rank_position, local_index in enumerate(order, start=1):
            global_index = indices[int(local_index)]
            output_rows[global_index]["predicted_rank"] = rank_position
            output_rows[global_index]["predicted_top_pick"] = 1 if rank_position == 1 else 0

    return output_rows


def save_predictions(rows: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = sorted({key for row in rows for key in row.keys()})

    preferred = [
        "scene_id",
        "image_id",
        "part_id",
        "instance",
        "heuristic_rank",
        "predicted_rank",
        "is_top_pick",
        "predicted_top_pick",
        "target_score",
        "predicted_score",
        "visible_pixels",
        "valid_3d_points",
        "visible_fraction",
        "depth_median",
        "extent_x",
        "extent_y",
        "extent_z",
    ]

    ordered = [col for col in preferred if col in fieldnames]
    ordered += [col for col in fieldnames if col not in ordered]

    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ordered)
        writer.writeheader()
        writer.writerows(rows)


def save_training_artifacts(
    model: PickabilityRanker,
    feature_columns: list[str],
    medians: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    metrics: dict,
    history: dict,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "pickability_ranker.pt"
    metadata_path = output_dir / "pickability_ranker_metadata.json"
    metrics_path = output_dir / "pickability_ranker_metrics.json"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "feature_columns": feature_columns,
            "medians": medians.tolist(),
            "mean": mean.tolist(),
            "std": std.tolist(),
        },
        model_path,
    )

    metadata = {
        "feature_columns": feature_columns,
        "model_type": "PyTorch MLP regression ranker",
        "target": TARGET_COLUMN,
        "note": (
            "This model learns to approximate the heuristic pickability score. "
            "It is not trained on real robot grasp-success labels."
        ),
    }

    with metadata_path.open("w") as f:
        json.dump(metadata, f, indent=2)

    full_metrics = dict(metrics)
    full_metrics["training_history"] = history

    with metrics_path.open("w") as f:
        json.dump(full_metrics, f, indent=2)

    print(f"Saved model: {model_path}")
    print(f"Saved metadata: {metadata_path}")
    print(f"Saved metrics: {metrics_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path("outputs/reports/pick_candidate_features.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/models"),
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=Path("outputs/figures/training"),
    )
    parser.add_argument(
        "--predictions-csv",
        type=Path,
        default=Path("outputs/reports/pickability_ranker_predictions.csv"),
    )
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--val-fraction", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    rows = load_csv_rows(args.input_csv)

    available_features = [
        column for column in FEATURE_COLUMNS
        if column in rows[0]
    ]

    if not available_features:
        raise ValueError("None of the requested feature columns were found in the CSV.")

    print(f"Rows loaded: {len(rows)}")
    print(f"Features used: {len(available_features)}")

    train_rows, val_rows = split_by_image_group(
        rows=rows,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )

    train_x_raw = build_feature_matrix(train_rows, available_features)
    val_x_raw = build_feature_matrix(val_rows, available_features)

    train_y = build_target_vector(train_rows)
    val_y = build_target_vector(val_rows)

    medians, mean, std = fit_preprocessor(train_x_raw)

    train_x = apply_preprocessor(train_x_raw, medians, mean, std)
    val_x = apply_preprocessor(val_x_raw, medians, mean, std)

    print(f"Train rows: {len(train_rows)}")
    print(f"Validation rows: {len(val_rows)}")

    model, history = train_model(
        train_x=train_x,
        train_y=train_y,
        val_x=val_x,
        val_y=val_y,
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        patience=args.patience,
    )

    val_predictions = predict_scores(model, val_x, device)
    val_metrics = evaluate_rows(val_rows, val_y, val_predictions)

    print("\nValidation metrics:")
    for key, value in val_metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

    prediction_rows = add_prediction_columns(
        rows=val_rows,
        target_scores=val_y,
        predicted_scores=val_predictions,
    )

    save_predictions(prediction_rows, args.predictions_csv)

    save_training_curve(
        history=history,
        output_path=args.figure_dir / "pickability_training_curve.png",
    )

    save_final_validation_plot(
        history=history,
        val_rows=val_rows,
        val_y=val_y,
        predicted_scores=val_predictions,
        metrics=val_metrics,
        output_path=args.figure_dir / "pickability_final_validation.png",
    )

    save_training_artifacts(
        model=model,
        feature_columns=available_features,
        medians=medians,
        mean=mean,
        std=std,
        metrics=val_metrics,
        history=history,
        output_dir=args.output_dir,
    )

    print("\nTraining complete.")
    print(f"Saved predictions: {args.predictions_csv}")
    print(f"Saved training curve: {args.figure_dir / 'pickability_training_curve.png'}")
    print(f"Saved validation plot: {args.figure_dir / 'pickability_final_validation.png'}")


if __name__ == "__main__":
    main()