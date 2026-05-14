from pathlib import Path
import argparse
import csv
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.bop_io import (
    load_json,
    load_rgb,
    load_depth,
    load_mask,
    get_camera_intrinsics,
    get_depth_scale,
)
from src.geometry import backproject_depth_to_points, compute_pca_orientation


# Colorblind-friendly, high-contrast palette.
# Avoids blue/cyan because the scene already contains a blue bin.
COLORBLIND_FRIENDLY_COLORS = [
    (230, 159, 0),    # orange
    (213, 94, 0),     # vermillion
    (0, 158, 115),    # green
    (204, 121, 167),  # purple
    (240, 228, 66),   # yellow
    (90, 90, 90),     # dark gray
    (166, 118, 29),   # brown
    (102, 17, 0),     # dark red-brown
]


def find_scene_dir(dataset_root: Path, scene_id: int | None) -> Path:
    val_dir = dataset_root / "val"

    if scene_id is not None:
        scene_dir = val_dir / f"{scene_id:06d}"
        if not scene_dir.exists():
            raise FileNotFoundError(f"Scene directory not found: {scene_dir}")
        return scene_dir

    scene_dirs = sorted([p for p in val_dir.iterdir() if p.is_dir()])
    if not scene_dirs:
        raise FileNotFoundError(f"No validation scenes found in: {val_dir}")

    return scene_dirs[0]


def find_first_image_id(scene_dir: Path) -> int:
    rgb_dir = scene_dir / "rgb_realsense"
    rgb_files = sorted(rgb_dir.glob("*.png"))

    if not rgb_files:
        raise FileNotFoundError(f"No RGB images found in: {rgb_dir}")

    return int(rgb_files[0].stem)


def find_visible_mask_files(scene_dir: Path, image_id: int) -> list[Path]:
    mask_dir = scene_dir / "mask_visib_realsense"
    mask_files = sorted(mask_dir.glob(f"{image_id:06d}_*.png"))

    if not mask_files:
        raise FileNotFoundError(
            f"No visible masks found for image {image_id:06d} in {mask_dir}"
        )

    return mask_files


def parse_object_index(mask_path: Path) -> int:
    """
    BOP mask filenames usually look like:
    000000_000003.png

    The second number is the object instance index in scene_gt.
    """
    return int(mask_path.stem.split("_")[-1])


def get_annotation_for_object(
    scene_gt: dict,
    scene_gt_info: dict,
    image_id: int,
    object_index: int,
) -> tuple[int | None, float | None]:
    """
    object_id is the part/class ID.
    object_index is the specific visible object instance in this image.
    """
    image_key = str(image_id)

    object_id = None
    visible_fraction = None

    if image_key in scene_gt and object_index < len(scene_gt[image_key]):
        object_id = scene_gt[image_key][object_index].get("obj_id")

    if image_key in scene_gt_info and object_index < len(scene_gt_info[image_key]):
        visible_fraction = scene_gt_info[image_key][object_index].get("visib_fract")

    return object_id, visible_fraction


def project_point(point_3d: np.ndarray, camera_k: np.ndarray) -> tuple[int, int] | None:
    x, y, z = point_3d

    if z <= 0:
        return None

    fx = camera_k[0, 0]
    fy = camera_k[1, 1]
    cx = camera_k[0, 2]
    cy = camera_k[1, 2]

    u = int(round((fx * x / z) + cx))
    v = int(round((fy * y / z) + cy))

    return u, v


def mask_pixel_center(mask: np.ndarray) -> tuple[float, float]:
    ys, xs = np.where(mask)
    return float(xs.mean()), float(ys.mean())


def compute_object_features(
    object_index: int,
    object_id: int | None,
    visible_fraction: float | None,
    mask: np.ndarray,
    depth: np.ndarray,
    camera_k: np.ndarray,
    depth_scale: float,
) -> tuple[dict, np.ndarray]:
    points = backproject_depth_to_points(
        depth=depth,
        mask=mask,
        camera_k=camera_k,
        depth_scale=depth_scale,
    )

    ys, xs = np.where(mask)

    if len(xs) == 0:
        raise ValueError("Mask has zero visible pixels.")

    bbox_x_min = int(xs.min())
    bbox_x_max = int(xs.max())
    bbox_y_min = int(ys.min())
    bbox_y_max = int(ys.max())

    pixel_center_x, pixel_center_y = mask_pixel_center(mask)

    feature = {
        "object_index": object_index,
        "object_id": object_id,
        "visible_fraction": visible_fraction if visible_fraction is not None else 1.0,
        "visible_pixels": int(mask.sum()),
        "valid_3d_points": int(points.shape[0]),
        "bbox_x_min": bbox_x_min,
        "bbox_x_max": bbox_x_max,
        "bbox_y_min": bbox_y_min,
        "bbox_y_max": bbox_y_max,
        "pixel_center_x": pixel_center_x,
        "pixel_center_y": pixel_center_y,
        "centroid_x": np.nan,
        "centroid_y": np.nan,
        "centroid_z": np.nan,
        "depth_min": np.nan,
        "depth_median": np.nan,
        "depth_max": np.nan,
        "extent_x": np.nan,
        "extent_y": np.nan,
        "extent_z": np.nan,
        "pca_axis_1_x": np.nan,
        "pca_axis_1_y": np.nan,
        "pca_axis_1_z": np.nan,
    }

    if points.shape[0] >= 3:
        centroid, axes, eigenvalues = compute_pca_orientation(points)

        feature["centroid_x"] = float(centroid[0])
        feature["centroid_y"] = float(centroid[1])
        feature["centroid_z"] = float(centroid[2])

        feature["depth_min"] = float(points[:, 2].min())
        feature["depth_median"] = float(np.median(points[:, 2]))
        feature["depth_max"] = float(points[:, 2].max())

        extents = np.ptp(points, axis=0)
        feature["extent_x"] = float(extents[0])
        feature["extent_y"] = float(extents[1])
        feature["extent_z"] = float(extents[2])

        feature["pca_axis_1_x"] = float(axes[0, 0])
        feature["pca_axis_1_y"] = float(axes[1, 0])
        feature["pca_axis_1_z"] = float(axes[2, 0])

    return feature, points


def add_pickability_scores(features: list[dict], image_shape: tuple[int, int, int]) -> list[dict]:
    """
    Interpretable pickability score for visible industrial parts.

    This is not a grasp planner. It ranks visible objects based on:
    - visible object area
    - visibility fraction / low occlusion
    - valid depth coverage
    - closeness to camera
    - central accessibility
    - basic geometry confidence
    """
    if not features:
        return features

    height, width = image_shape[:2]

    visible_pixels = np.array([f["visible_pixels"] for f in features], dtype=np.float32)
    max_visible_pixels = max(float(visible_pixels.max()), 1.0)

    depth_medians = np.array(
        [
            f["depth_median"] if not np.isnan(f["depth_median"]) else np.nan
            for f in features
        ],
        dtype=np.float32,
    )

    valid_depths = depth_medians[~np.isnan(depth_medians)]

    if len(valid_depths) > 1:
        depth_min = float(valid_depths.min())
        depth_max = float(valid_depths.max())
        depth_range = max(depth_max - depth_min, 1e-6)
    else:
        depth_min = 0.0
        depth_max = 1.0
        depth_range = 1.0

    for feature in features:
        # 1. Visible area score.
        # Use square root so one huge object does not dominate too aggressively.
        area_score = np.sqrt(feature["visible_pixels"] / max_visible_pixels)
        area_score = float(np.clip(area_score, 0.0, 1.0))

        # 2. Visibility score.
        # Higher visible fraction means less occluded and more reliable.
        visibility_score = float(np.clip(feature["visible_fraction"], 0.0, 1.0))

        # 3. Valid depth coverage.
        # Rewards objects where most mask pixels have valid 3D depth.
        if feature["visible_pixels"] > 0:
            depth_coverage_score = feature["valid_3d_points"] / feature["visible_pixels"]
        else:
            depth_coverage_score = 0.0

        depth_coverage_score = float(np.clip(depth_coverage_score, 0.0, 1.0))

        # 4. Closeness score.
        # Smaller median depth means closer to camera, often more accessible in top-down bin picking.
        if np.isnan(feature["depth_median"]):
            closeness_score = 0.0
        else:
            closeness_score = (depth_max - feature["depth_median"]) / depth_range
            closeness_score = float(np.clip(closeness_score, 0.0, 1.0))

        # 5. Center accessibility.
        # Objects near the image center are often easier to inspect and less likely to be cut off.
        center_dx = (feature["pixel_center_x"] - width / 2) / max(width / 2, 1)
        center_dy = (feature["pixel_center_y"] - height / 2) / max(height / 2, 1)
        center_distance = float(np.sqrt(center_dx**2 + center_dy**2))
        center_score = 1.0 - min(center_distance, 1.0)

        # 6. Geometry confidence.
        # Rewards objects with enough 3D points and non-degenerate 3D extent.
        if (
            feature["valid_3d_points"] < 50
            or np.isnan(feature["extent_x"])
            or np.isnan(feature["extent_y"])
            or np.isnan(feature["extent_z"])
        ):
            geometry_score = 0.0
        else:
            extents = np.array(
                [feature["extent_x"], feature["extent_y"], feature["extent_z"]],
                dtype=np.float32,
            )

            max_extent = float(np.nanmax(extents))
            min_extent = float(np.nanmin(extents))

            if max_extent <= 0:
                geometry_score = 0.0
            else:
                # Penalize almost-flat/noisy/degenerated point clouds, but do not require perfect shape.
                extent_ratio = min_extent / max_extent
                geometry_score = float(np.clip(0.5 + extent_ratio, 0.0, 1.0))

        pickability_score = (
            0.30 * area_score
            + 0.25 * visibility_score
            + 0.20 * closeness_score
            + 0.10 * depth_coverage_score
            + 0.10 * center_score
            + 0.05 * geometry_score
        )

        feature["area_score"] = float(area_score)
        feature["visibility_score"] = float(visibility_score)
        feature["closeness_score"] = float(closeness_score)
        feature["depth_coverage_score"] = float(depth_coverage_score)
        feature["center_score"] = float(center_score)
        feature["geometry_score"] = float(geometry_score)
        feature["pickability_score"] = float(pickability_score)

    features = sorted(features, key=lambda f: f["pickability_score"], reverse=True)

    for rank, feature in enumerate(features, start=1):
        feature["pick_rank"] = rank

    return features


def save_point_cloud(points: np.ndarray, output_path: Path) -> None:
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(str(output_path), point_cloud)


def get_rank_color(rank: int) -> tuple[int, int, int]:
    return COLORBLIND_FRIENDLY_COLORS[(rank - 1) % len(COLORBLIND_FRIENDLY_COLORS)]


def text_color_for_background(color: tuple[int, int, int]) -> tuple[int, int, int]:
    """
    Choose black or white text depending on background luminance.
    """
    r, g, b = color
    luminance = 0.299 * r + 0.587 * g + 0.114 * b

    if luminance > 140:
        return (0, 0, 0)

    return (255, 255, 255)


def matplotlib_text_color_for_background(color: tuple[int, int, int]) -> str:
    text_color = text_color_for_background(color)
    if text_color == (0, 0, 0):
        return "black"
    return "white"


def draw_mask_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int],
    alpha: float,
) -> np.ndarray:
    """
    Blend a colored mask into an RGB image.
    """
    output = image.copy()
    color_array = np.array(color, dtype=np.uint8)

    output[mask] = (
        (1.0 - alpha) * output[mask] + alpha * color_array
    ).astype(np.uint8)

    return output


def draw_mask_contour(
    image: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int],
) -> None:
    """
    Draw thin high-contrast contours around a mask.
    """
    mask_uint8 = mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(
        mask_uint8,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if not contours:
        return

    # Thin white halo, then thinner colored outline.
    cv2.drawContours(image, contours, -1, (255, 255, 255), 2)
    cv2.drawContours(image, contours, -1, color, 1)


def compute_badge_center(
    feature: dict,
    image_shape: tuple[int, int, int],
    mask: np.ndarray | None = None,
) -> tuple[int, int]:
    """
    Place the rank badge at the middle of the visible piece.

    The primary position is the centroid of all visible mask pixels.
    If that point falls outside the mask because the shape is irregular,
    snap to the closest visible mask pixel.
    """
    height, width = image_shape[:2]

    if mask is not None and np.any(mask):
        ys, xs = np.where(mask)

        center_x = float(xs.mean())
        center_y = float(ys.mean())

        x = int(round(center_x))
        y = int(round(center_y))

        x = int(np.clip(x, 0, width - 1))
        y = int(np.clip(y, 0, height - 1))

        if mask[y, x]:
            return (
                int(np.clip(x, 18, width - 18)),
                int(np.clip(y, 18, height - 18)),
            )

        # If the centroid is outside the visible region, use the visible mask pixel
        # closest to the centroid.
        distances = (xs - center_x) ** 2 + (ys - center_y) ** 2
        closest_index = int(np.argmin(distances))

        x = int(xs[closest_index])
        y = int(ys[closest_index])

        return (
            int(np.clip(x, 18, width - 18)),
            int(np.clip(y, 18, height - 18)),
        )

    # Fallback if the mask is missing, tiny, or invalid.
    x = int(round(feature["pixel_center_x"]))
    y = int(round(feature["pixel_center_y"]))

    x = int(np.clip(x, 18, width - 18))
    y = int(np.clip(y, 18, height - 18))

    return x, y


def draw_rank_badge(
    image: np.ndarray,
    center: tuple[int, int],
    rank: int,
    color: tuple[int, int, int],
) -> None:
    """
    Draw a compact numbered badge instead of a full text label.
    """
    x, y = center
    radius = 14

    cv2.circle(image, (x, y), radius + 4, (0, 0, 0), -1)
    cv2.circle(image, (x, y), radius + 2, (255, 255, 255), -1)
    cv2.circle(image, (x, y), radius, color, -1)

    text = str(rank)
    text_color = text_color_for_background(color)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
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


def draw_arrow_with_halo(
    image: np.ndarray,
    start: tuple[int, int],
    end: tuple[int, int],
    color: tuple[int, int, int],
) -> None:
    """
    Draw an orientation arrow with a light halo for visibility.

    This arrow shows the PCA dominant visible 3D direction.
    It is not a grasp direction or robot motion direction.
    """
    cv2.arrowedLine(
        image,
        start,
        end,
        (255, 255, 255),
        4,
        tipLength=0.22,
    )

    cv2.arrowedLine(
        image,
        start,
        end,
        color,
        2,
        tipLength=0.22,
    )


def draw_ranked_results_table(
    table_ax,
    features: list[dict],
    max_rows: int = 5,
) -> None:
    """
    Draw top-ranked pick-candidate information under the images.

    Part ID is the dataset object/class ID.
    Instance is the specific visible copy of that part in this scene.
    """
    table_ax.axis("off")

    columns = [
        "Rank",
        "Part ID",
        "Instance",
        "Score",
        "Visible Pixels",
        "Median Depth",
        "Visible Fraction",
    ]

    rows = []

    for feature in features[:max_rows]:
        depth_median = feature["depth_median"]

        if np.isnan(depth_median):
            depth_text = "n/a"
        else:
            depth_text = f"{depth_median:.1f}"

        rows.append(
            [
                str(feature["pick_rank"]),
                str(feature["object_id"]),
                str(feature["object_index"]),
                f"{feature['pickability_score']:.3f}",
                str(feature["visible_pixels"]),
                depth_text,
                f"{feature['visible_fraction']:.2f}",
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

    # Header styling.
    for col_index in range(len(columns)):
        cell = table[(0, col_index)]
        cell.set_facecolor("#f0f0f0")
        cell.set_text_props(weight="bold", color="black")

    # Rank color styling.
    for row_index, feature in enumerate(features[:max_rows], start=1):
        rank = int(feature["pick_rank"])
        color = get_rank_color(rank)
        color_normalized = tuple(np.array(color, dtype=np.float32) / 255.0)

        rank_cell = table[(row_index, 0)]
        rank_cell.set_facecolor(color_normalized)
        rank_cell.get_text().set_color(matplotlib_text_color_for_background(color))
        rank_cell.get_text().set_weight("bold")

    table_ax.set_title(
        "Top 5 Pick Candidates",
        fontsize=13,
        fontweight="bold",
        pad=8,
    )


def draw_results_overlay(
    rgb: np.ndarray,
    masks_by_object_index: dict[int, np.ndarray],
    features: list[dict],
    camera_k: np.ndarray,
    output_path: Path,
) -> None:
    """
    Create a clean portfolio-style visualization.

    Top left:
    - raw scene

    Top right:
    - annotated scene with masks, thin contours, numbered rank badges, and PCA arrows

    Bottom:
    - top 5 ranked object information
    """
    max_badges = 10
    max_arrows = 5

    raw_image = rgb.copy()
    annotated_image = rgb.copy()

    # Blend masks first.
    for feature in features:
        rank = int(feature["pick_rank"])
        object_index = int(feature["object_index"])

        if object_index not in masks_by_object_index:
            continue

        mask = masks_by_object_index[object_index]
        color = get_rank_color(rank)

        alpha = 0.30 if rank <= max_badges else 0.12

        annotated_image = draw_mask_overlay(
            image=annotated_image,
            mask=mask,
            color=color,
            alpha=alpha,
        )

    # Draw thin contours after overlays.
    for feature in features:
        rank = int(feature["pick_rank"])
        object_index = int(feature["object_index"])

        if object_index not in masks_by_object_index:
            continue

        mask = masks_by_object_index[object_index]
        color = get_rank_color(rank)

        draw_mask_contour(annotated_image, mask, color)

    # Draw arrows first so number badges appear above them.
    for feature in features:
        rank = int(feature["pick_rank"])

        if rank > max_arrows:
            continue

        if np.isnan(feature["centroid_z"]):
            continue

        centroid = np.array(
            [
                feature["centroid_x"],
                feature["centroid_y"],
                feature["centroid_z"],
            ],
            dtype=np.float32,
        )

        axis = np.array(
            [
                feature["pca_axis_1_x"],
                feature["pca_axis_1_y"],
                feature["pca_axis_1_z"],
            ],
            dtype=np.float32,
        )

        if np.any(np.isnan(axis)):
            continue

        object_extents = np.array(
            [
                feature["extent_x"],
                feature["extent_y"],
                feature["extent_z"],
            ],
            dtype=np.float32,
        )

        if np.any(np.isnan(object_extents)):
            arrow_length = 35.0
        else:
            arrow_length = float(np.clip(0.35 * np.nanmax(object_extents), 25.0, 80.0))

        axis_endpoint = centroid + axis * arrow_length

        p1 = project_point(centroid, camera_k)
        p2 = project_point(axis_endpoint, camera_k)

        if p1 is not None and p2 is not None:
            color = get_rank_color(rank)
            draw_arrow_with_halo(annotated_image, p1, p2, color)

    # Draw badges last so they sit on top of arrows and masks.
    for feature in features:
        rank = int(feature["pick_rank"])

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

        draw_rank_badge(
            image=annotated_image,
            center=badge_center,
            rank=rank,
            color=color,
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

    raw_ax = fig.add_subplot(grid[0, 0])
    annot_ax = fig.add_subplot(grid[0, 1])
    table_ax = fig.add_subplot(grid[1, :])

    raw_ax.imshow(raw_image)
    raw_ax.set_title("Raw Scene", fontsize=14, fontweight="bold")
    raw_ax.axis("off")

    annot_ax.imshow(annotated_image)
    annot_ax.set_title("Annotated Scene", fontsize=14, fontweight="bold")
    annot_ax.axis("off")

    # Force both image panels to use the same limits and aspect ratio.
    for ax in [raw_ax, annot_ax]:
        ax.set_xlim(0, image_width)
        ax.set_ylim(image_height, 0)
        ax.set_aspect("equal", adjustable="box")

    draw_ranked_results_table(
        table_ax=table_ax,
        features=features,
        max_rows=5,
    )

    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close()


def save_csv(features: list[dict], output_path: Path) -> None:
    if not features:
        return

    fieldnames = list(features[0].keys())

    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(features)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("data/xyzibd"))
    parser.add_argument("--scene-id", type=int, default=None)
    parser.add_argument("--image-id", type=int, default=None)
    args = parser.parse_args()

    scene_dir = find_scene_dir(args.dataset_root, args.scene_id)

    image_id = args.image_id
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

    output_pcd_dir = PROJECT_ROOT / "outputs" / "pointclouds"
    output_pcd_dir.mkdir(parents=True, exist_ok=True)

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

        feature, points = compute_object_features(
            object_index=object_index,
            object_id=object_id,
            visible_fraction=visible_fraction,
            mask=mask,
            depth=depth,
            camera_k=camera_k,
            depth_scale=depth_scale,
        )

        features.append(feature)

        if points.shape[0] > 0:
            pcd_path = output_pcd_dir / (
                f"scene_{scene_dir.name}_image_{image_id:06d}_object_{object_index:06d}.ply"
            )
            save_point_cloud(points, pcd_path)

    features = add_pickability_scores(features, rgb.shape)

    output_fig_dir = PROJECT_ROOT / "outputs" / "figures"
    output_report_dir = PROJECT_ROOT / "outputs" / "reports"
    output_fig_dir.mkdir(parents=True, exist_ok=True)
    output_report_dir.mkdir(parents=True, exist_ok=True)

    output_stem = f"scene_{scene_dir.name}_image_{image_id:06d}"

    overlay_path = output_fig_dir / f"{output_stem}_pickability_overlay.png"
    csv_path = output_report_dir / f"{output_stem}_pickability_summary.csv"

    draw_results_overlay(
        rgb=rgb,
        masks_by_object_index=masks_by_object_index,
        features=features,
        camera_k=camera_k,
        output_path=overlay_path,
    )

    save_csv(features, csv_path)

    print("Processed all visible objects.")
    print(f"Scene directory: {scene_dir}")
    print(f"Image ID: {image_id}")
    print(f"Objects found: {len(features)}")
    print(f"Saved overlay: {overlay_path}")
    print(f"Saved CSV: {csv_path}")

    print("\nTop pick candidates:")
    for feature in features[:5]:
        depth_text = "n/a" if np.isnan(feature["depth_median"]) else f"{feature['depth_median']:.2f}"
        print(
            f"Rank {feature['pick_rank']}: "
            f"part_id={feature['object_id']}, "
            f"instance={feature['object_index']}, "
            f"score={feature['pickability_score']:.3f}, "
            f"visible_pixels={feature['visible_pixels']}, "
            f"depth_median={depth_text}, "
            f"visible_fraction={feature['visible_fraction']:.3f}"
        )


if __name__ == "__main__":
    main()