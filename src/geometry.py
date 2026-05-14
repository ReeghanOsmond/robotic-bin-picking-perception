import numpy as np


def ensure_2d_depth(depth: np.ndarray) -> np.ndarray:
    """
    Ensure depth is a 2D float32 array with shape H x W.

    Some image-loading paths can return H x W x 1 or H x W x 3.
    For this project, depth should be a single-channel image.
    """
    depth = np.asarray(depth)

    if depth.ndim == 3:
        if depth.shape[2] == 1:
            depth = depth[:, :, 0]
        else:
            # If depth was accidentally loaded with multiple channels,
            # use the first channel. Depth PNG channels should be identical
            # or only the first channel should be meaningful.
            depth = depth[:, :, 0]

    if depth.ndim != 2:
        raise ValueError(f"Expected 2D depth image, got shape {depth.shape}")

    return depth.astype(np.float32)


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


def backproject_depth_to_points(
    depth: np.ndarray,
    mask: np.ndarray,
    camera_k: np.ndarray,
    depth_scale: float = 1.0,
) -> np.ndarray:
    """
    Convert masked depth pixels into a 3D point cloud using camera intrinsics.

    Parameters
    ----------
    depth:
        H x W depth image.

    mask:
        H x W boolean object mask.

    camera_k:
        3 x 3 camera intrinsic matrix.

    depth_scale:
        Dataset-provided depth scale.

    Returns
    -------
    points:
        N x 3 array of 3D points.
    """
    depth = ensure_2d_depth(depth)
    mask = ensure_2d_mask(mask)

    if depth.shape != mask.shape:
        raise ValueError(
            f"Depth and mask must have same shape. "
            f"Got depth={depth.shape}, mask={mask.shape}"
        )

    ys, xs = np.where(mask)

    if len(xs) == 0:
        return np.empty((0, 3), dtype=np.float32)

    z = depth[ys, xs].astype(np.float32).reshape(-1)
    z = z * float(depth_scale)

    valid = np.isfinite(z) & (z > 0)

    xs = xs.reshape(-1)[valid].astype(np.float32)
    ys = ys.reshape(-1)[valid].astype(np.float32)
    z = z[valid].astype(np.float32)

    if len(z) == 0:
        return np.empty((0, 3), dtype=np.float32)

    fx = float(camera_k[0, 0])
    fy = float(camera_k[1, 1])
    cx = float(camera_k[0, 2])
    cy = float(camera_k[1, 2])

    x = (xs - cx) * z / fx
    y = (ys - cy) * z / fy

    points = np.stack([x, y, z], axis=1)

    return points.astype(np.float32)


def compute_pca_orientation(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute centroid and principal axes for a point cloud.

    Returns
    -------
    centroid:
        3D centroid of the point cloud.

    eigenvectors:
        Principal axes ordered from largest to smallest variance.

    eigenvalues:
        Variances along each principal axis, ordered largest to smallest.
    """
    points = np.asarray(points, dtype=np.float32)

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected N x 3 point cloud, got shape {points.shape}")

    if points.shape[0] < 3:
        raise ValueError("Need at least 3 valid 3D points for PCA.")

    centroid = points.mean(axis=0)
    centered = points - centroid

    covariance = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)

    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    return centroid.astype(np.float32), eigenvectors.astype(np.float32), eigenvalues.astype(np.float32)