from pathlib import Path
import json
import cv2
import numpy as np


def load_json(path):
    path = Path(path)
    with path.open("r") as f:
        return json.load(f)


def load_rgb(path):
    path = Path(path)
    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)

    if image_bgr is None:
        raise FileNotFoundError(f"Could not load RGB image: {path}")

    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def load_depth(path):
    path = Path(path)
    depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)

    if depth is None:
        raise FileNotFoundError(f"Could not load depth image: {path}")

    return depth.astype(np.float32)


def load_mask(path):
    path = Path(path)
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    if mask is None:
        raise FileNotFoundError(f"Could not load mask: {path}")

    return mask > 0


def get_camera_intrinsics(scene_camera, image_id):
    entry = scene_camera[str(image_id)]
    cam_k = np.array(entry["cam_K"], dtype=np.float32).reshape(3, 3)
    return cam_k


def get_depth_scale(scene_camera, image_id):
    entry = scene_camera[str(image_id)]
    return float(entry.get("depth_scale", 1.0))