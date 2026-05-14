from pathlib import Path
import argparse
import csv
import sys

import numpy as np

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
    find_visible_mask_files,
    parse_object_index,
    get_annotation_for_object,
    compute_object_features,
    add_pickability_scores,
)


def get_scene_dirs(dataset_root: Path, scene_ids: list[int] | None) -> list[Path]:
    val_dir = dataset_root / "val"

    if not val_dir.exists():
        raise FileNotFoundError(f"Validation directory not found: {val_dir}")

    if scene_ids:
        scene_dirs = []

        for scene_id in scene_ids:
            scene_dir = val_dir / f"{scene_id:06d}"

            if not scene_dir.exists():
                raise FileNotFoundError(f"Scene directory not found: {scene_dir}")

            scene_dirs.append(scene_dir)

        return scene_dirs

    return sorted([p for p in val_dir.iterdir() if p.is_dir()])


def get_image_ids(scene_dir: Path, max_images_per_scene: int | None) -> list[int]:
    rgb_dir = scene_dir / "rgb_realsense"
    rgb_files = sorted(rgb_dir.glob("*.png"))

    image_ids = [int(path.stem) for path in rgb_files]

    if max_images_per_scene is not None:
        image_ids = image_ids[:max_images_per_scene]

    return image_ids


def process_image(scene_dir: Path, image_id: int) -> list[dict]:
    scene_camera = load_json(scene_dir / "scene_camera_realsense.json")
    scene_gt = load_json(scene_dir / "scene_gt_realsense.json")
    scene_gt_info = load_json(scene_dir / "scene_gt_info_realsense.json")

    rgb_path = scene_dir / "rgb_realsense" / f"{image_id:06d}.png"
    depth_path = scene_dir / "depth_realsense" / f"{image_id:06d}.png"

    rgb = load_rgb(rgb_path)
    depth = load_depth(depth_path)

    camera_k = get_camera_intrinsics(scene_camera, image_id)
    depth_scale = get_depth_scale(scene_camera, image_id)

    mask_files = find_visible_mask_files(scene_dir, image_id)

    features = []

    for mask_path in mask_files:
        object_index = parse_object_index(mask_path)

        object_id, visible_fraction = get_annotation_for_object(
            scene_gt=scene_gt,
            scene_gt_info=scene_gt_info,
            image_id=image_id,
            object_index=object_index,
        )

        mask = load_mask(mask_path)

        feature, _ = compute_object_features(
            object_index=object_index,
            object_id=object_id,
            visible_fraction=visible_fraction,
            mask=mask,
            depth=depth,
            camera_k=camera_k,
            depth_scale=depth_scale,
        )

        features.append(feature)

    features = add_pickability_scores(features, rgb.shape)

    rows = []

    for feature in features:
        row = dict(feature)

        row["scene_id"] = scene_dir.name
        row["image_id"] = f"{image_id:06d}"

        # More readable aliases.
        row["part_id"] = row["object_id"]
        row["instance"] = row["object_index"]

        # These are the first training targets.
        # They come from the current heuristic, not from real robot grasp trials.
        row["heuristic_rank"] = row["pick_rank"]
        row["heuristic_pickability_score"] = row["pickability_score"]
        row["is_top_pick"] = 1 if row["pick_rank"] == 1 else 0

        rows.append(row)

    return rows


def save_rows(rows: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        raise ValueError("No rows to save.")

    preferred_columns = [
        "scene_id",
        "image_id",
        "part_id",
        "instance",
        "heuristic_rank",
        "is_top_pick",
        "heuristic_pickability_score",
        "pickability_score",
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
        "area_score",
        "visibility_score",
        "closeness_score",
        "depth_coverage_score",
        "center_score",
        "geometry_score",
        "pca_axis_1_x",
        "pca_axis_1_y",
        "pca_axis_1_z",
        "bbox_x_min",
        "bbox_x_max",
        "bbox_y_min",
        "bbox_y_max",
        "object_id",
        "object_index",
    ]

    all_columns = sorted({key for row in rows for key in row.keys()})
    fieldnames = [col for col in preferred_columns if col in all_columns]
    fieldnames += [col for col in all_columns if col not in fieldnames]

    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("data/xyzibd"))
    parser.add_argument("--output", type=Path, default=Path("outputs/reports/pick_candidate_features.csv"))
    parser.add_argument("--scene-ids", type=int, nargs="*", default=None)
    parser.add_argument("--max-images-per-scene", type=int, default=25)
    args = parser.parse_args()

    scene_dirs = get_scene_dirs(args.dataset_root, args.scene_ids)

    all_rows = []

    for scene_dir in scene_dirs:
        image_ids = get_image_ids(scene_dir, args.max_images_per_scene)

        print(f"Processing scene {scene_dir.name}: {len(image_ids)} images")

        for image_id in image_ids:
            try:
                rows = process_image(scene_dir, image_id)
                all_rows.extend(rows)
                print(f"  image {image_id:06d}: {len(rows)} candidates")
            except Exception as error:
                print(f"  image {image_id:06d}: skipped because {error}")

    save_rows(all_rows, args.output)

    print("\nCandidate feature dataset complete.")
    print(f"Rows saved: {len(all_rows)}")
    print(f"Output: {args.output}")

    scene_count = len(set(row["scene_id"] for row in all_rows))
    image_count = len(set((row["scene_id"], row["image_id"]) for row in all_rows))
    candidate_count = len(all_rows)
    top_pick_count = int(np.sum([row["is_top_pick"] for row in all_rows]))

    print("\nSummary:")
    print(f"Scenes: {scene_count}")
    print(f"Images: {image_count}")
    print(f"Object candidates: {candidate_count}")
    print(f"Top-pick labels: {top_pick_count}")


if __name__ == "__main__":
    main()