from pathlib import Path
import argparse
import json
import random
import shutil

import cv2
import numpy as np
import yaml


def load_json(path: Path) -> dict:
    with path.open("r") as f:
        return json.load(f)


def find_existing_file(scene_dir: Path, candidates: list[str]) -> Path:
    for candidate in candidates:
        path = scene_dir / candidate
        if path.exists():
            return path

    raise FileNotFoundError(
        f"Could not find any of these files in {scene_dir}: {candidates}"
    )


def find_existing_dir(scene_dir: Path, candidates: list[str]) -> Path:
    for candidate in candidates:
        path = scene_dir / candidate
        if path.exists() and path.is_dir():
            return path

    raise FileNotFoundError(
        f"Could not find any of these folders in {scene_dir}: {candidates}"
    )


def get_scene_dirs(dataset_root: Path, source_split: str) -> list[Path]:
    split_dir = dataset_root / source_split

    if not split_dir.exists():
        raise FileNotFoundError(f"Could not find split directory: {split_dir}")

    scene_dirs = sorted([p for p in split_dir.iterdir() if p.is_dir()])

    if not scene_dirs:
        raise FileNotFoundError(f"No scene folders found in: {split_dir}")

    return scene_dirs


def parse_object_index(mask_path: Path) -> int:
    """
    Mask filenames usually look like:
    000000_000003.png

    The second number is the object instance index in scene_gt.
    """
    return int(mask_path.stem.split("_")[-1])


def collect_image_records(dataset_root: Path, source_split: str) -> list[dict]:
    records = []

    scene_dirs = get_scene_dirs(dataset_root, source_split)

    for scene_dir in scene_dirs:
        rgb_dir = find_existing_dir(
            scene_dir,
            ["rgb_realsense", "rgb", "gray"],
        )

        mask_visib_dir = find_existing_dir(
            scene_dir,
            ["mask_visib_realsense", "mask_visib"],
        )

        scene_gt_path = find_existing_file(
            scene_dir,
            ["scene_gt_realsense.json", "scene_gt.json"],
        )

        scene_gt = load_json(scene_gt_path)

        rgb_files = sorted(rgb_dir.glob("*.png"))

        for rgb_path in rgb_files:
            image_id = int(rgb_path.stem)

            mask_files = sorted(mask_visib_dir.glob(f"{image_id:06d}_*.png"))

            if not mask_files:
                continue

            image_key = str(image_id)

            if image_key not in scene_gt:
                continue

            records.append(
                {
                    "scene_dir": scene_dir,
                    "scene_name": scene_dir.name,
                    "image_id": image_id,
                    "rgb_path": rgb_path,
                    "mask_files": mask_files,
                    "scene_gt": scene_gt,
                }
            )

    if not records:
        raise ValueError("No usable image records found.")

    return records


def collect_part_ids(records: list[dict]) -> list[int]:
    part_ids = set()

    for record in records:
        image_key = str(record["image_id"])
        scene_gt = record["scene_gt"]

        for mask_path in record["mask_files"]:
            object_index = parse_object_index(mask_path)

            if object_index >= len(scene_gt[image_key]):
                continue

            part_id = int(scene_gt[image_key][object_index]["obj_id"])
            part_ids.add(part_id)

    return sorted(part_ids)


def split_records(
    records: list[dict],
    val_fraction: float,
    seed: int,
    max_images: int | None,
) -> tuple[list[dict], list[dict]]:
    records = list(records)

    rng = random.Random(seed)
    rng.shuffle(records)

    if max_images is not None:
        records = records[:max_images]

    val_count = max(1, int(round(len(records) * val_fraction)))

    val_records = records[:val_count]
    train_records = records[val_count:]

    if not train_records:
        raise ValueError("No training records after split. Reduce val_fraction or increase images.")

    if not val_records:
        raise ValueError("No validation records after split. Increase val_fraction.")

    return train_records, val_records


def mask_to_polygon(
    mask: np.ndarray,
    min_area: float,
    epsilon_fraction: float,
) -> list[tuple[float, float]] | None:
    """
    Convert a binary mask to a single polygon.

    For this first version, use the largest external contour.
    This avoids splitting one visible object into multiple instances.
    """
    mask_uint8 = (mask > 0).astype(np.uint8) * 255

    contours, _ = cv2.findContours(
        mask_uint8,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(contour))

    if area < min_area:
        return None

    perimeter = cv2.arcLength(contour, closed=True)
    epsilon = epsilon_fraction * perimeter

    approx = cv2.approxPolyDP(
        contour,
        epsilon=epsilon,
        closed=True,
    )

    if approx.shape[0] < 3:
        return None

    points = approx.reshape(-1, 2)

    return [(float(x), float(y)) for x, y in points]


def make_yolo_label_line(
    class_index: int,
    polygon: list[tuple[float, float]],
    image_width: int,
    image_height: int,
) -> str | None:
    normalized_values = []

    for x, y in polygon:
        x_norm = np.clip(x / image_width, 0.0, 1.0)
        y_norm = np.clip(y / image_height, 0.0, 1.0)

        normalized_values.append(x_norm)
        normalized_values.append(y_norm)

    if len(normalized_values) < 6:
        return None

    values = [str(class_index)]
    values.extend([f"{value:.6f}" for value in normalized_values])

    return " ".join(values)


def convert_record(
    record: dict,
    output_root: Path,
    split_name: str,
    part_id_to_class_index: dict[int, int],
    min_area: float,
    epsilon_fraction: float,
) -> int:
    scene_name = record["scene_name"]
    image_id = record["image_id"]
    rgb_path = record["rgb_path"]
    scene_gt = record["scene_gt"]
    mask_files = record["mask_files"]

    image = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)

    if image is None:
        raise FileNotFoundError(f"Could not read image: {rgb_path}")

    image_height, image_width = image.shape[:2]

    output_stem = f"scene_{scene_name}_image_{image_id:06d}"

    output_image_path = output_root / "images" / split_name / f"{output_stem}.png"
    output_label_path = output_root / "labels" / split_name / f"{output_stem}.txt"

    output_image_path.parent.mkdir(parents=True, exist_ok=True)
    output_label_path.parent.mkdir(parents=True, exist_ok=True)

    label_lines = []

    image_key = str(image_id)

    for mask_path in mask_files:
        object_index = parse_object_index(mask_path)

        if object_index >= len(scene_gt[image_key]):
            continue

        part_id = int(scene_gt[image_key][object_index]["obj_id"])
        class_index = part_id_to_class_index[part_id]

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        if mask is None:
            continue

        polygon = mask_to_polygon(
            mask=mask,
            min_area=min_area,
            epsilon_fraction=epsilon_fraction,
        )

        if polygon is None:
            continue

        label_line = make_yolo_label_line(
            class_index=class_index,
            polygon=polygon,
            image_width=image_width,
            image_height=image_height,
        )

        if label_line is not None:
            label_lines.append(label_line)

    if not label_lines:
        return 0

    shutil.copy2(rgb_path, output_image_path)

    with output_label_path.open("w") as f:
        f.write("\n".join(label_lines))
        f.write("\n")

    return len(label_lines)


def write_dataset_yaml(
    output_root: Path,
    class_index_to_part_id: dict[int, int],
) -> None:
    names = {
        class_index: f"part_{part_id}"
        for class_index, part_id in class_index_to_part_id.items()
    }

    dataset_yaml = {
        "path": str(output_root.resolve()).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "names": names,
    }

    yaml_path = output_root / "dataset.yaml"

    with yaml_path.open("w") as f:
        yaml.safe_dump(dataset_yaml, f, sort_keys=False)

    print(f"Saved dataset YAML: {yaml_path}")


def write_class_mapping(
    output_root: Path,
    part_id_to_class_index: dict[int, int],
) -> None:
    class_index_to_part_id = {
        class_index: part_id
        for part_id, class_index in part_id_to_class_index.items()
    }

    mapping = {
        "part_id_to_class_index": {
            str(part_id): class_index
            for part_id, class_index in part_id_to_class_index.items()
        },
        "class_index_to_part_id": {
            str(class_index): part_id
            for class_index, part_id in class_index_to_part_id.items()
        },
        "note": (
            "YOLO class indices are zero-based and contiguous. "
            "part_id is the dataset part type / class label."
        ),
    }

    mapping_path = output_root / "class_mapping.json"

    with mapping_path.open("w") as f:
        json.dump(mapping, f, indent=2)

    print(f"Saved class mapping: {mapping_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("data/xyzibd"))
    parser.add_argument("--source-split", type=str, default="val")
    parser.add_argument("--output-root", type=Path, default=Path("data/segmentation_yolo"))
    parser.add_argument("--val-fraction", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--min-area", type=float, default=30.0)
    parser.add_argument("--epsilon-fraction", type=float, default=0.002)
    args = parser.parse_args()

    print("Collecting image records...")
    records = collect_image_records(
        dataset_root=args.dataset_root,
        source_split=args.source_split,
    )

    part_ids = collect_part_ids(records)

    if not part_ids:
        raise ValueError("No part IDs found.")

    part_id_to_class_index = {
        part_id: index
        for index, part_id in enumerate(part_ids)
    }

    class_index_to_part_id = {
        index: part_id
        for part_id, index in part_id_to_class_index.items()
    }

    print(f"Found {len(records)} usable images.")
    print(f"Found {len(part_ids)} part classes: {part_ids}")

    train_records, val_records = split_records(
        records=records,
        val_fraction=args.val_fraction,
        seed=args.seed,
        max_images=args.max_images,
    )

    print(f"Train images: {len(train_records)}")
    print(f"Val images: {len(val_records)}")

    if args.output_root.exists():
        print(f"Clearing existing output folder: {args.output_root}")
        shutil.rmtree(args.output_root)

    total_train_labels = 0
    total_val_labels = 0

    for record in train_records:
        total_train_labels += convert_record(
            record=record,
            output_root=args.output_root,
            split_name="train",
            part_id_to_class_index=part_id_to_class_index,
            min_area=args.min_area,
            epsilon_fraction=args.epsilon_fraction,
        )

    for record in val_records:
        total_val_labels += convert_record(
            record=record,
            output_root=args.output_root,
            split_name="val",
            part_id_to_class_index=part_id_to_class_index,
            min_area=args.min_area,
            epsilon_fraction=args.epsilon_fraction,
        )

    write_dataset_yaml(
        output_root=args.output_root,
        class_index_to_part_id=class_index_to_part_id,
    )

    write_class_mapping(
        output_root=args.output_root,
        part_id_to_class_index=part_id_to_class_index,
    )

    print("\nConversion complete.")
    print(f"Output root: {args.output_root}")
    print(f"Train labels: {total_train_labels}")
    print(f"Val labels: {total_val_labels}")
    print(f"Dataset YAML: {args.output_root / 'dataset.yaml'}")


if __name__ == "__main__":
    main()