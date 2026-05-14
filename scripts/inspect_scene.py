from pathlib import Path
import argparse
import sys

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


def find_scene_dir(dataset_root, scene_id=None):
    val_dir = dataset_root / "val"

    if scene_id is not None:
        scene_dir = val_dir / f"{scene_id:06d}"
        if not scene_dir.exists():
            raise FileNotFoundError(f"Scene directory not found: {scene_dir}")
        return scene_dir

    scene_dirs = sorted([p for p in val_dir.iterdir() if p.is_dir()])

    if not scene_dirs:
        raise FileNotFoundError(f"No scenes found in: {val_dir}")

    return scene_dirs[0]


def find_first_image_id(scene_dir):
    rgb_dir = scene_dir / "rgb_realsense"
    rgb_files = sorted(rgb_dir.glob("*.png"))

    if not rgb_files:
        raise FileNotFoundError(f"No RGB files found in: {rgb_dir}")

    return int(rgb_files[0].stem)


def find_first_visible_mask(scene_dir, image_id):
    mask_dir = scene_dir / "mask_visib_realsense"
    mask_files = sorted(mask_dir.glob(f"{image_id:06d}_*.png"))

    if not mask_files:
        raise FileNotFoundError(
            f"No visible object masks found for image {image_id:06d} in {mask_dir}"
        )

    return mask_files[0]


def make_mask_overlay(rgb, mask):
    overlay = rgb.copy()
    red = np.array([255, 0, 0], dtype=np.uint8)
    overlay[mask] = (0.6 * overlay[mask] + 0.4 * red).astype(np.uint8)
    return overlay


def save_point_cloud(points, output_path):
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(str(output_path), point_cloud)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data/xyzibd"),
        help="Path to extracted dataset root.",
    )
    parser.add_argument("--scene-id", type=int, default=None)
    parser.add_argument("--image-id", type=int, default=None)
    args = parser.parse_args()

    dataset_root = args.dataset_root
    scene_dir = find_scene_dir(dataset_root, args.scene_id)

    image_id = args.image_id
    if image_id is None:
        image_id = find_first_image_id(scene_dir)

    rgb_path = scene_dir / "rgb_realsense" / f"{image_id:06d}.png"
    depth_path = scene_dir / "depth_realsense" / f"{image_id:06d}.png"
    mask_path = find_first_visible_mask(scene_dir, image_id)
    camera_path = scene_dir / "scene_camera_realsense.json"

    rgb = load_rgb(rgb_path)
    depth = load_depth(depth_path)
    mask = load_mask(mask_path)

    scene_camera = load_json(camera_path)
    camera_k = get_camera_intrinsics(scene_camera, image_id)
    depth_scale = get_depth_scale(scene_camera, image_id)

    points = backproject_depth_to_points(
        depth=depth,
        mask=mask,
        camera_k=camera_k,
        depth_scale=depth_scale,
    )

    print("Loaded scene successfully.")
    print(f"Scene directory: {scene_dir}")
    print(f"Image ID: {image_id}")
    print(f"RGB: {rgb_path}")
    print(f"Depth: {depth_path}")
    print(f"Mask: {mask_path}")
    print(f"Camera intrinsics:\n{camera_k}")
    print(f"Depth scale: {depth_scale}")
    print(f"Extracted point cloud shape: {points.shape}")

    if points.shape[0] >= 3:
        centroid, axes, eigenvalues = compute_pca_orientation(points)
        print(f"Centroid: {centroid}")
        print(f"PCA axes:\n{axes}")
        print(f"PCA eigenvalues: {eigenvalues}")
    else:
        print("Not enough points for PCA orientation.")

    output_fig_dir = PROJECT_ROOT / "outputs" / "figures"
    output_pcd_dir = PROJECT_ROOT / "outputs" / "pointclouds"
    output_fig_dir.mkdir(parents=True, exist_ok=True)
    output_pcd_dir.mkdir(parents=True, exist_ok=True)

    overlay = make_mask_overlay(rgb, mask)

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(rgb)
    plt.title("RGB")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(depth, cmap="gray")
    plt.title("Depth")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(overlay)
    plt.title("Visible Object Mask")
    plt.axis("off")

    plt.tight_layout()

    figure_path = output_fig_dir / "scene_inspection.png"
    plt.savefig(figure_path, dpi=200)
    plt.close()

    print(f"Saved figure: {figure_path}")

    if points.shape[0] > 0:
        pcd_path = output_pcd_dir / "object_points.ply"
        save_point_cloud(points, pcd_path)
        print(f"Saved point cloud: {pcd_path}")


if __name__ == "__main__":
    main()