"""S-Space inner-programming spatial readout, collision detection, and steering math.

Pure NumPy module directly compatible with hidden-state tensors extracted by
`mirros-lab/s-space` (`[N, D]` activations and `[3, D]` pretrained visual axes):
1. Fits a linear 6D spatial box readout (3D center + 3D half-extents) from activations.
2. Predicts pairwise collisions via explicit analytical box intersection over S-Space.
3. Compares explicit 6D box readout vs direct collision-axis linear readout.
4. Measures cross-modal subspace alignment (Procrustes R^2) between Visual S-Space
   and XML-S-Space axes.
5. Computes right-inverse activation steering vectors `Delta_h = B @ Delta_p` to push
   colliding objects apart inside the model's hidden state.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any


def _np() -> Any:
    return importlib.import_module("numpy")


@dataclass(frozen=True)
class SpatialBoxProbe:
    """Ridge-regularized linear map from hidden state h in R^D to (center, half_extents) in R^6."""

    weights: Any  # [D, 6]
    bias: Any     # [6]

    def predict(self, hidden: Any) -> tuple[Any, Any]:
        """Return `(centers [..., 3], positive_extents [..., 3])`."""
        np = _np()
        arr = np.asarray(hidden, dtype=np.float64)
        out = arr @ self.weights + self.bias
        centers = out[..., :3]
        extents = np.maximum(out[..., 3:6], 1e-4)
        return centers, extents


def fit_spatial_box_probe(
    hidden: Any,
    centers: Any,
    half_extents: Any,
    *,
    ridge_lambda: float = 1e-2,
) -> SpatialBoxProbe:
    """Fit linear readout h -> [cx, cy, cz, hx, hy, hz] via closed-form ridge regression."""
    np = _np()
    x = np.asarray(hidden, dtype=np.float64)
    y = np.concatenate(
        [np.asarray(centers, dtype=np.float64), np.asarray(half_extents, dtype=np.float64)],
        axis=-1,
    )
    if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0] or y.shape[1] != 6:
        raise ValueError(f"invalid shapes: hidden={x.shape}, targets={y.shape}")
    mean_x = x.mean(axis=0, keepdims=True)
    mean_y = y.mean(axis=0, keepdims=True)
    xc = x - mean_x
    yc = y - mean_y
    d = xc.shape[1]
    gram = xc.T @ xc + float(ridge_lambda) * np.eye(d, dtype=np.float64)
    w = np.linalg.solve(gram, xc.T @ yc)
    b = (mean_y - mean_x @ w).reshape(6)
    return SpatialBoxProbe(weights=w, bias=b)


def predict_box_collision(
    h_a: Any,
    h_b: Any,
    probe: SpatialBoxProbe,
    *,
    margin_m: float = 0.0,
) -> dict[str, Any]:
    """Inner-programming collision detector: explicit AABB separation over read-out S-Space."""
    np = _np()
    ca, ea = probe.predict(h_a)
    cb, eb = probe.predict(h_b)
    # Signed separation along each axis: negative means overlap along that axis.
    axis_gap = np.abs(ca - cb) - (ea + eb)  # [..., 3]
    # AABB signed separation: negative iff ALL 3 axes overlap.
    signed_dist = np.max(axis_gap, axis=-1)
    colliding = signed_dist < -float(margin_m)
    return {
        "signed_distance_m": signed_dist,
        "colliding": colliding,
        "centers_a": ca,
        "centers_b": cb,
        "extents_a": ea,
        "extents_b": eb,
    }


def fit_contrast_axis(h_positive: Any, h_negative: Any) -> Any:
    """Extract an L2-unit linear contrast axis between positive and negative states."""
    np = _np()
    hp = np.asarray(h_positive, dtype=np.float64)
    hn = np.asarray(h_negative, dtype=np.float64)
    diff = hp.mean(axis=0) - hn.mean(axis=0)
    norm = float(np.linalg.norm(diff))
    if norm <= 1e-12:
        raise ValueError("contrast axis has near-zero norm")
    return diff / norm


def evaluate_collision_methods(
    h_a_train: Any,
    h_b_train: Any,
    centers_a_train: Any,
    extents_a_train: Any,
    centers_b_train: Any,
    extents_b_train: Any,
    labels_train: Any,
    h_a_eval: Any,
    h_b_eval: Any,
    centers_a_eval: Any,
    extents_a_eval: Any,
    centers_b_eval: Any,
    extents_b_eval: Any,
    labels_eval: Any,
    *,
    ridge_lambda: float = 1e-2,
) -> dict[str, Any]:
    """Compare explicit 6D S-Space box collision vs direct difference-axis classifier."""
    np = _np()
    h_tr = np.concatenate([h_a_train, h_b_train], axis=0)
    c_tr = np.concatenate([centers_a_train, centers_b_train], axis=0)
    e_tr = np.concatenate([extents_a_train, extents_b_train], axis=0)
    probe = fit_spatial_box_probe(h_tr, c_tr, e_tr, ridge_lambda=ridge_lambda)

    pred_box = predict_box_collision(h_a_eval, h_b_eval, probe)
    y_eval = np.asarray(labels_eval, dtype=bool)
    box_acc = float((pred_box["colliding"] == y_eval).mean())
    center_mae = float(
        0.5
        * (
            np.abs(pred_box["centers_a"] - centers_a_eval).mean()
            + np.abs(pred_box["centers_b"] - centers_b_eval).mean()
        )
    )
    extent_mae = float(
        0.5
        * (
            np.abs(pred_box["extents_a"] - extents_a_eval).mean()
            + np.abs(pred_box["extents_b"] - extents_b_eval).mean()
        )
    )

    # Direct contrast axis on symmetric pair features |h_a - h_b|.
    y_tr = np.asarray(labels_train, dtype=bool)
    feat_tr = np.abs(np.asarray(h_a_train, dtype=np.float64) - np.asarray(h_b_train, dtype=np.float64))
    feat_ev = np.abs(np.asarray(h_a_eval, dtype=np.float64) - np.asarray(h_b_eval, dtype=np.float64))
    axis = fit_contrast_axis(feat_tr[y_tr], feat_tr[~y_tr])
    tr_scores = feat_tr @ axis
    thresh = 0.5 * (float(tr_scores[y_tr].mean()) + float(tr_scores[~y_tr].mean()))
    direct_pred = (feat_ev @ axis) >= thresh
    direct_acc = float((direct_pred == y_eval).mean())

    return {
        "explicit_6d_box_accuracy": round(box_acc, 4),
        "direct_contrast_axis_accuracy": round(direct_acc, 4),
        "center_mae_m": round(center_mae, 6),
        "extent_mae_m": round(extent_mae, 6),
        "eval_count": int(len(y_eval)),
    }


def cross_modal_subspace_alignment(
    visual_axes: Any, xml_axes: Any
) -> dict[str, Any]:
    """Measure subspace alignment between Visual S-Space [3, D] and XML-S-Space [3, D].

    Reports both direct per-axis cosine similarity (zero-shot transfer) and
    principal angles / orthogonal Procrustes R^2 (shared rotated 3D subspace).
    """
    np = _np()
    va = np.asarray(visual_axes, dtype=np.float64)
    xa = np.asarray(xml_axes, dtype=np.float64)
    if va.shape != xa.shape or va.ndim != 2 or va.shape[0] != 3:
        raise ValueError(f"expected matching [3, D] axes, got {va.shape} and {xa.shape}")
    va_n = va / np.maximum(np.linalg.norm(va, axis=1, keepdims=True), 1e-12)
    xa_n = xa / np.maximum(np.linalg.norm(xa, axis=1, keepdims=True), 1e-12)
    direct_cos = [round(float(np.dot(va_n[i], xa_n[i])), 6) for i in range(3)]

    # Orthonormal bases of the two 3D subspaces via QR, then principal angles via SVD.
    q_vis, _ = np.linalg.qr(va_n.T)  # [D, 3]
    q_xml, _ = np.linalg.qr(xa_n.T)  # [D, 3]
    svals = np.clip(np.linalg.svd(q_vis.T @ q_xml, compute_uv=False), 0.0, 1.0)
    principal_angles_deg = [round(float(np.degrees(np.arccos(v))), 3) for v in svals]
    subspace_overlap = round(float(np.mean(svals**2)), 6)  # 1.0 = identical 3D subspace

    return {
        "direct_axis_cosine": direct_cos,
        "principal_angles_deg": principal_angles_deg,
        "subspace_overlap_r2": subspace_overlap,
    }


def steering_delta_for_separation(
    axes_3xd: Any,
    delta_xyz: Any,
) -> Any:
    """Compute minimal-norm activation edit Delta_h = B @ Delta_p using right inverse.

    Matches `s-space`'s `coordinate_right_inverse`: guarantees `axes @ Delta_h == delta_xyz`.
    """
    np = _np()
    factors = np.asarray(axes_3xd, dtype=np.float64)
    dp = np.asarray(delta_xyz, dtype=np.float64).reshape(3)
    if factors.ndim != 2 or factors.shape[0] != 3:
        raise ValueError(f"expected [3, D] axes, got {factors.shape}")
    basis = factors.T @ np.linalg.inv(factors @ factors.T)  # [D, 3]
    return basis @ dp
