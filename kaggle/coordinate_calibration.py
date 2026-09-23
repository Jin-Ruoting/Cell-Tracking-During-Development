"""Bounded image-feature coordinate calibration; no graph or label lookup.

The central feature plus six directional differences are inspired by the
public x138 coordinate-refinement description. The fitted ridge regressor,
trilinear sampling at final graph positions, and export policy are separate.
"""
from __future__ import annotations

OFFSETS = ((0, 0, 0), (-1, 0, 0), (1, 0, 0), (0, -1, 0),
           (0, 1, 0), (0, 0, -1), (0, 0, 1))
RIDGE_L2 = 0.01
MAX_SHIFT_UM = 2.0
MAX_TRAIN_RESIDUAL_UM = 4.0


def sample_features(feature, coordinates):
    """Sample C×Z×Y×X at fractional Z/Y/X positions and six neighbors."""
    import torch
    from torch.nn import functional as F

    if feature.ndim != 4 or coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError("Invalid feature or coordinate shape")
    if len(coordinates) == 0:
        return feature.new_empty((0, feature.shape[0] * 7))
    points = torch.as_tensor(coordinates, device=feature.device, dtype=torch.float32)
    if not torch.isfinite(points).all() or not torch.isfinite(feature).all():
        raise ValueError("Nonfinite feature or coordinate")
    points = points[:, None, :] + torch.tensor(OFFSETS, device=feature.device)[None]
    sizes = torch.tensor(feature.shape[-3:], device=feature.device)
    points = points.clamp(min=0)
    points = torch.minimum(points, sizes - 1)
    grid = (2 * points / (sizes - 1).clamp(min=1) - 1)[..., [2, 1, 0]]
    sampled = F.grid_sample(feature[None].float(), grid.reshape(1, 1, 1, -1, 3),
                            mode="bilinear", padding_mode="border", align_corners=True)
    blocks = sampled[0, :, 0, 0].T.reshape(len(coordinates), 7, feature.shape[0])
    return torch.cat((blocks[:, 0], (blocks[:, 1:] - blocks[:, :1]).flatten(1)), dim=1)


def fit_ridge(blocks):
    """Equal movie weights; training-only standardization and fixed L2."""
    import numpy as np

    if not blocks or any(len(x) == 0 or len(x) != len(y) for x, y in blocks):
        raise ValueError("Every training movie needs valid matched examples")
    x = np.concatenate([a for a, _ in blocks]).astype(np.float64)
    y = np.concatenate([b for _, b in blocks]).astype(np.float64)
    if x.ndim != 2 or y.shape != (len(x), 3) or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Invalid training arrays")
    weights = np.concatenate([np.full(len(a), 1 / (len(blocks) * len(a))) for a, _ in blocks])
    mean = np.sum(x * weights[:, None], axis=0)
    scale = np.sqrt(np.sum((x - mean) ** 2 * weights[:, None], axis=0)).clip(1e-4)
    target_mean = np.sum(y * weights[:, None], axis=0)
    x = (x - mean) / scale
    covariance = x.T @ (weights[:, None] * x)
    coefficient = np.linalg.solve(covariance + RIDGE_L2 * np.eye(x.shape[1]),
                                  x.T @ (weights[:, None] * (y - target_mean)))
    model = {"mean": mean, "scale": scale, "coefficient": coefficient, "intercept": target_mean}
    if not all(np.isfinite(value).all() for value in model.values()):
        raise ValueError("Nonfinite fitted regressor")
    return model


def calibrated_positions(model, features, positions, voxel_scale, image_shape):
    """Return integer positions; final quantized movement never exceeds 2 um."""
    import numpy as np

    original = np.asarray(positions, dtype=np.float64)
    scale = np.asarray(voxel_scale, dtype=np.float64)
    shape = np.asarray(image_shape, dtype=np.int64)
    if (original.shape != (len(features), 3) or scale.shape != (3,) or shape.shape != (3,)
            or not np.isfinite(original).all() or not (original == np.rint(original)).all()
            or not np.isfinite(scale).all() or (scale <= 0).any() or (shape <= 0).any()
            or (original < 0).any() or (original >= shape).any()):
        raise ValueError("Invalid original voxel coordinates or geometry")
    x = np.asarray(features, dtype=np.float64)
    delta = ((x - model["mean"]) / model["scale"]) @ model["coefficient"] + model["intercept"]
    if not np.isfinite(delta).all():
        raise ValueError("Nonfinite predicted displacement")
    norm = np.linalg.norm(delta, axis=1)
    delta *= np.minimum(1, MAX_SHIFT_UM / np.maximum(norm, 1e-12))[:, None]
    integer_delta = np.rint(delta / scale).astype(np.int64)
    rounded_norm = np.linalg.norm(integer_delta * scale, axis=1)
    too_far = rounded_norm > MAX_SHIFT_UM
    integer_delta[too_far] = np.trunc(
        integer_delta[too_far] * (MAX_SHIFT_UM / rounded_norm[too_far])[:, None]).astype(np.int64)
    result = np.clip(original.astype(np.int64) + integer_delta, 0, shape - 1)
    if (np.linalg.norm((result - original) * scale, axis=1) > MAX_SHIFT_UM + 1e-8).any():
        raise AssertionError("Final coordinate displacement exceeded the frozen bound")
    return result
