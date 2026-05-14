from pathlib import Path
import argparse
import json
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
import yaml


COLORBLIND_FRIENDLY_COLORS = [
    (230, 159, 0),
    (213, 94, 0),
    (0, 158, 115),
    (204, 121, 167),
    (240, 228, 66),
    (90, 90, 90),
    (166, 118, 29),
    (102, 17, 0),
]


def load_yaml(path: Path) -> dict:
    with path.open("r") as f:
        return yaml.safe_load(f)


def get_color(index: int) -> tuple[int, int, int]:
    return COLORBLIND_FRIENDLY_COLORS[index % len(COLORBLIND_FRIENDLY_COLORS)]


def parse_yolo_segmentation_label(label_path: Path) -> list[dict]:
    objects = []

    if not label_path.exists():
        return objects

    with label_path.open("r") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    for line in lines:
        values = line.split()

        if len(values) < 7:
            continue

        class_id = int(float(values[0]))
        coords = [float(value) for value in values[1:]]

        if len(coords) % 2 != 0:
            continue

        points = np.array(coords, dtype=np.float32).reshape(-1, 2)

        objects.append(
            {
                "class_id": class_id,
                "points_norm": points,
            }
        )

    return objects


def polygon_to_pixels(points_norm: np.ndarray, width: int, height: int) -> np.ndarray:
    points = points_norm.copy()
    points[:, 0] *= width
    points[:, 1] *= height
    points = np.round(points).astype(np.int32)
    return points


def draw_clean_overlay(
    image_rgb: np.ndarray,
    objects: list[dict],
    class_names: dict,
) -> np.ndarray:
    overlay = image_rgb.copy()

    height, width = image_rgb.shape[:2]

    for index, obj in enumerate(objects):
        class_id = obj["class_id"]
        color = get_color(class_id)

        points = polygon_to_pixels(
            points_norm=obj["points_norm"],
            width=width,
            height=height,
        )

        if points.shape[0] < 3:
            continue

        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, [points], 255)

        color_array = np.array(color, dtype=np.uint8)
        mask_bool = mask > 0

        overlay[mask_bool] = (
            0.70 * overlay[mask_bool] + 0.30 * color_array
        ).astype(np.uint8)

        cv2.polylines(
            overlay,
            [points],
            isClosed=True,
            color=(255, 255, 255),
            thickness=2,
        )

        cv2.polylines(
            overlay,
            [points],
            isClosed=True,
            color=color,
            thickness=1,
        )

        # Small number only. No long text on the object.
        moments = cv2.moments(points)

        if moments["m00"] != 0:
            cx = int(moments["m10"] / moments["m00"])
            cy = int(moments["m01"] / moments["m00"])
        else:
            cx = int(points[:, 0].mean())
            cy = int(points[:, 1].mean())

        cx = int(np.clip(cx, 16, width - 16))
        cy = int(np.clip(cy, 16, height - 16))

        radius = 11

        cv2.circle(overlay, (cx, cy), radius + 3, (0, 0, 0), -1)
        cv2.circle(overlay, (cx, cy), radius + 1, (255, 255, 255), -1)
        cv2.circle(overlay, (cx, cy), radius, color, -1)

        label = str(index + 1)

        text_size, _ = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            1,
        )

        tx = int(cx - text_size[0] / 2)
        ty = int(cy + text_size[1] / 2)

        cv2.putText(
            overlay,
            label,
            (tx, ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    return overlay


def make_preview_figure(
    image_path: Path,
    label_path: Path,
    class_names: dict,
    output_path: Path,
) -> None:
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    objects = parse_yolo_segmentation_label(label_path)

    overlay = draw_clean_overlay(
        image_rgb=image_rgb,
        objects=objects,
        class_names=class_names,
    )

    fig = plt.figure(figsize=(14, 7))
    grid = fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.05)

    raw_ax = fig.add_subplot(grid[0, 0])
    label_ax = fig.add_subplot(grid[0, 1])

    raw_ax.imshow(image_rgb)
    raw_ax.set_title("Raw Image", fontsize=13, fontweight="bold")
    raw_ax.axis("off")

    label_ax.imshow(overlay)
    label_ax.set_title(f"Clean Label Overlay ({len(objects)} objects)", fontsize=13, fontweight="bold")
    label_ax.axis("off")

    height, width = image_rgb.shape[:2]

    for ax in [raw_ax, label_ax]:
        ax.set_xlim(0, width)
        ax.set_ylim(height, 0)
        ax.set_aspect("equal", adjustable="box")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("data/segmentation_yolo"))
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/segmentation_label_previews"))
    args = parser.parse_args()

    dataset_yaml = load_yaml(args.dataset_root / "dataset.yaml")
    class_names = dataset_yaml["names"]

    image_dir = args.dataset_root / "images" / args.split
    label_dir = args.dataset_root / "labels" / args.split

    image_paths = sorted(image_dir.glob("*.png"))

    if not image_paths:
        raise FileNotFoundError(f"No images found in: {image_dir}")

    rng = random.Random(args.seed)
    rng.shuffle(image_paths)

    selected_paths = image_paths[: args.count]

    for image_path in selected_paths:
        label_path = label_dir / f"{image_path.stem}.txt"
        output_path = args.output_dir / args.split / f"{image_path.stem}_clean_labels.png"

        make_preview_figure(
            image_path=image_path,
            label_path=label_path,
            class_names=class_names,
            output_path=output_path,
        )

        print(f"Saved clean label preview: {output_path}")


if __name__ == "__main__":
    main()