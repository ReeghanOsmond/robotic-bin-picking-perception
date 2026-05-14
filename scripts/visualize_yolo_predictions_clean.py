from pathlib import Path
import argparse
import json

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from ultralytics import YOLO


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


def ensure_2d_mask(mask: np.ndarray) -> np.ndarray:
    """
    Ensure mask is a 2D boolean array with shape H x W.
    """
    mask = np.asarray(mask)

    if mask.ndim == 3:
        mask = np.squeeze(mask)

    if mask.ndim != 2:
        raise ValueError(f"Expected 2D mask, got shape {mask.shape}")

    return mask.astype(bool)


def load_json(path: Path) -> dict:
    with path.open("r") as f:
        return json.load(f)


def get_color(index: int) -> tuple[int, int, int]:
    return COLORBLIND_FRIENDLY_COLORS[index % len(COLORBLIND_FRIENDLY_COLORS)]


def text_color_for_background(color: tuple[int, int, int]) -> tuple[int, int, int]:
    r, g, b = color
    luminance = 0.299 * r + 0.587 * g + 0.114 * b

    if luminance > 140:
        return (0, 0, 0)

    return (255, 255, 255)


def parse_object_index(mask_path: Path) -> int:
    """
    BOP mask filenames usually look like:
    000000_000003.png

    The second number is the object instance index in scene_gt.
    """
    return int(mask_path.stem.split("_")[-1])


def load_class_mapping(mapping_path: Path) -> dict[int, int]:
    """
    Returns mapping:
        YOLO class index -> original dataset part_id
    """
    if not mapping_path.exists():
        return {}

    mapping = load_json(mapping_path)
    class_index_to_part_id = mapping.get("class_index_to_part_id", {})

    return {
        int(class_index): int(part_id)
        for class_index, part_id in class_index_to_part_id.items()
    }


def load_scene_ground_truth(
    dataset_root: Path,
    scene_id: int,
    image_id: int,
) -> tuple[np.ndarray, list[dict], Path]:
    scene_dir = dataset_root / "val" / f"{scene_id:06d}"

    image_path = scene_dir / "rgb_realsense" / f"{image_id:06d}.png"
    mask_dir = scene_dir / "mask_visib_realsense"
    scene_gt_path = scene_dir / "scene_gt_realsense.json"

    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    scene_gt = load_json(scene_gt_path)
    image_key = str(image_id)

    if image_key not in scene_gt:
        raise KeyError(f"Image ID {image_id} not found in {scene_gt_path}")

    mask_files = sorted(mask_dir.glob(f"{image_id:06d}_*.png"))

    instances = []

    for mask_path in mask_files:
        object_index = parse_object_index(mask_path)

        if object_index >= len(scene_gt[image_key]):
            continue

        part_id = int(scene_gt[image_key][object_index]["obj_id"])

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        if mask is None:
            continue

        mask_bool = ensure_2d_mask(mask > 0)

        if not np.any(mask_bool):
            continue

        instances.append(
            {
                "panel_index": len(instances) + 1,
                "instance": object_index,
                "part_id": part_id,
                "mask": mask_bool,
                "mask_pixels": int(mask_bool.sum()),
            }
        )

    return image_rgb, instances, image_path


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


def predict_yolo_masks(
    model: YOLO,
    image_path: Path,
    image_shape: tuple[int, int, int],
    class_index_to_part_id: dict[int, int],
    imgsz: int,
    conf: float,
    iou: float,
    max_det: int,
    device: str,
) -> list[dict]:
    results = model.predict(
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

    predicted_instances = []

    masks_xy = result.masks.xy
    classes = result.boxes.cls.detach().cpu().numpy().astype(int)
    confidences = result.boxes.conf.detach().cpu().numpy()

    for index, polygon_xy in enumerate(masks_xy):
        class_index = int(classes[index])
        confidence = float(confidences[index])
        part_id = class_index_to_part_id.get(class_index, class_index)

        mask_bool = polygon_to_mask(
            polygon_xy=np.asarray(polygon_xy),
            image_shape=image_shape,
        )

        mask_bool = ensure_2d_mask(mask_bool)

        if not np.any(mask_bool):
            continue

        predicted_instances.append(
            {
                "panel_index": len(predicted_instances) + 1,
                "class_index": class_index,
                "part_id": part_id,
                "confidence": confidence,
                "mask": mask_bool,
                "mask_pixels": int(mask_bool.sum()),
            }
        )

    predicted_instances = sorted(
        predicted_instances,
        key=lambda item: item["confidence"],
        reverse=True,
    )

    for new_index, item in enumerate(predicted_instances, start=1):
        item["panel_index"] = new_index

    return predicted_instances


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
) -> None:
    x, y = center
    radius = 13

    cv2.circle(image, (x, y), radius + 4, (0, 0, 0), -1)
    cv2.circle(image, (x, y), radius + 2, (255, 255, 255), -1)
    cv2.circle(image, (x, y), radius, color, -1)

    text = str(number)
    text_color = text_color_for_background(color)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.50
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


def draw_mask_contours(
    image: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int],
    is_prediction: bool,
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

    if is_prediction:
        cv2.drawContours(image, contours, -1, (255, 255, 255), 3)
        cv2.drawContours(image, contours, -1, color, 2)
    else:
        cv2.drawContours(image, contours, -1, (255, 255, 255), 2)
        cv2.drawContours(image, contours, -1, color, 1)


def draw_overlay(
    image_rgb: np.ndarray,
    instances: list[dict],
    is_prediction: bool,
) -> np.ndarray:
    output = image_rgb.copy()

    for index, instance in enumerate(instances):
        color = get_color(index)
        mask = ensure_2d_mask(instance["mask"])

        alpha = 0.32 if is_prediction else 0.26

        color_array = np.array(color, dtype=np.uint8)
        mask_bool = mask.astype(bool)

        output[mask_bool] = (
            (1.0 - alpha) * output[mask_bool] + alpha * color_array
        ).astype(np.uint8)

    for index, instance in enumerate(instances):
        color = get_color(index)
        mask = ensure_2d_mask(instance["mask"])

        draw_mask_contours(
            image=output,
            mask=mask,
            color=color,
            is_prediction=is_prediction,
        )

    for index, instance in enumerate(instances):
        color = get_color(index)
        mask = ensure_2d_mask(instance["mask"])

        center = compute_mask_center(mask)

        draw_badge(
            image=output,
            center=center,
            number=instance["panel_index"],
            color=color,
        )

    return output


def save_comparison_figure(
    image_rgb: np.ndarray,
    gt_instances: list[dict],
    predicted_instances: list[dict],
    output_path: Path,
    scene_id: int,
    image_id: int,
) -> None:
    gt_overlay = draw_overlay(
        image_rgb=image_rgb,
        instances=gt_instances,
        is_prediction=False,
    )

    pred_overlay = draw_overlay(
        image_rgb=image_rgb,
        instances=predicted_instances,
        is_prediction=True,
    )

    height, width = image_rgb.shape[:2]

    fig = plt.figure(figsize=(12, 16))

    grid = fig.add_gridspec(
        2,
        1,
        hspace=0.10,
    )

    gt_ax = fig.add_subplot(grid[0, 0])
    pred_ax = fig.add_subplot(grid[1, 0])

    gt_ax.imshow(gt_overlay)
    gt_ax.set_title(
        f"Ground-Truth Visible Masks ({len(gt_instances)} objects)",
        fontsize=13,
        fontweight="bold",
    )
    gt_ax.axis("off")

    pred_ax.imshow(pred_overlay)
    pred_ax.set_title(
        f"YOLO Predicted Masks ({len(predicted_instances)} objects)",
        fontsize=13,
        fontweight="bold",
    )
    pred_ax.axis("off")

    for ax in [gt_ax, pred_ax]:
        ax.set_xlim(0, width)
        ax.set_ylim(height, 0)
        ax.set_aspect("equal", adjustable="box")

    fig.suptitle(
        f"Segmentation Preview: Scene {scene_id:06d}, Image {image_id:06d}",
        fontsize=16,
        fontweight="bold",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close()


def run_one_example(
    model: YOLO,
    dataset_root: Path,
    class_index_to_part_id: dict[int, int],
    scene_id: int,
    image_id: int,
    output_dir: Path,
    imgsz: int,
    conf: float,
    iou: float,
    max_det: int,
    device: str,
) -> None:
    image_rgb, gt_instances, image_path = load_scene_ground_truth(
        dataset_root=dataset_root,
        scene_id=scene_id,
        image_id=image_id,
    )

    predicted_instances = predict_yolo_masks(
        model=model,
        image_path=image_path,
        image_shape=image_rgb.shape,
        class_index_to_part_id=class_index_to_part_id,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        max_det=max_det,
        device=device,
    )

    output_path = output_dir / f"scene_{scene_id:06d}_image_{image_id:06d}_yolo_gt_vs_prediction.png"

    save_comparison_figure(
        image_rgb=image_rgb,
        gt_instances=gt_instances,
        predicted_instances=predicted_instances,
        output_path=output_path,
        scene_id=scene_id,
        image_id=image_id,
    )

    print(f"Saved YOLO prediction preview: {output_path}")
    print(f"  GT masks: {len(gt_instances)}")
    print(f"  Predicted masks: {len(predicted_instances)}")


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data/xyzibd"),
    )

    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path(
            "runs/segment/outputs/segmentation_training/yolo_segmentation/weights/best.pt"
        ),
    )

    parser.add_argument(
        "--class-mapping",
        type=Path,
        default=Path("data/segmentation_yolo/class_mapping.json"),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/segmentation_prediction_previews"),
    )

    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.50)
    parser.add_argument("--max-det", type=int, default=80)
    parser.add_argument("--device", type=str, default="0")

    args = parser.parse_args()

    print("Segmentation prediction preview")
    print(f"Model path: {args.model_path}")
    print(f"Dataset root: {args.dataset_root}")
    print(f"Class mapping: {args.class_mapping}")
    print(f"Output dir: {args.output_dir}")
    print(f"Confidence threshold: {args.conf}")
    print(f"IoU threshold: {args.iou}")
    print(f"Max detections: {args.max_det}")

    print("\nCUDA check:")
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    model = YOLO(str(args.model_path))

    class_index_to_part_id = load_class_mapping(args.class_mapping)

    for scene_id, image_id in DEFAULT_EXAMPLES:
        print(f"\nRunning scene {scene_id}, image {image_id}")

        run_one_example(
            model=model,
            dataset_root=args.dataset_root,
            class_index_to_part_id=class_index_to_part_id,
            scene_id=scene_id,
            image_id=image_id,
            output_dir=args.output_dir,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            max_det=args.max_det,
            device=args.device,
        )


if __name__ == "__main__":
    main()