"""Mathematical utility functions for active inference.

All functions are pure NumPy — no PyTorch dependency.
Variable naming follows paper notation (see docs/references/paper-notation.md).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Numerical primitives
# ---------------------------------------------------------------------------

def log_stable(arr: NDArray[np.float64]) -> NDArray[np.float64]:
    """Logarithm with epsilon guard against log(0)."""
    return np.log(arr + 1e-16)


def softmax(dist: NDArray[np.float64], precision: float = 1.0) -> NDArray[np.float64]:
    """Numerically stable softmax with configurable precision (inverse temperature)."""
    shifted = dist - dist.max(axis=0)
    exp_vals = np.exp(precision * shifted)
    return exp_vals / np.sum(exp_vals, axis=0)


def norm_dist(dist: NDArray[np.float64]) -> NDArray[np.float64]:
    """Normalize a categorical distribution (or matrix columns)."""
    if dist.ndim == 1:
        total = dist.sum()
        return dist / total if total > 0 else dist
    return dist / dist.sum(axis=0, keepdims=True)


def norm_counts(counts: NDArray[np.float64]) -> NDArray[np.float64]:
    """Normalize count matrix (columns sum to 1)."""
    result = np.zeros_like(counts)
    for c in range(counts.shape[1]):
        col_sum = counts[:, c].sum()
        if col_sum > 0:
            result[:, c] = counts[:, c] / col_sum
    return result


# ---------------------------------------------------------------------------
# Normalization (maps to/from [-1, 1])
# ---------------------------------------------------------------------------

def normalize(
    x: NDArray[np.float64] | float,
    limits: tuple[float, float],
    rng: bool = True,
) -> NDArray[np.float64]:
    """Map value from [lo, hi] to [-1, 1] (rng=True) or [0, 1] (rng=False)."""
    lo, hi = limits
    x_norm = (np.asarray(x, dtype=np.float64) - lo) / (hi - lo)
    if rng:
        x_norm = x_norm * 2.0 - 1.0
    return x_norm


def denormalize(
    x: NDArray[np.float64] | float,
    limits: tuple[float, float],
    rng: bool = True,
) -> NDArray[np.float64]:
    """Map value from [-1, 1] (rng=True) or [0, 1] (rng=False) back to [lo, hi]."""
    lo, hi = limits
    x_arr = np.asarray(x, dtype=np.float64)
    x_denorm = (x_arr + 1.0) / 2.0 if rng else x_arr
    return x_denorm * (hi - lo) + lo


# ---------------------------------------------------------------------------
# Forward kinematics (pure NumPy, angles in normalized space)
# ---------------------------------------------------------------------------

def forward_kinematics(
    angles_norm: NDArray[np.float64],
    lengths_norm: NDArray[np.float64] | tuple[float, ...],
    norm_polar: tuple[float, float] = (-180.0, 180.0),
) -> NDArray[np.float64]:
    """Compute end-effector position from normalized joint angles.

    Args:
        angles_norm: Joint angles in [-1, 1] (normalized from degrees).
        lengths_norm: Link lengths in normalized Cartesian space.
        norm_polar: Polar normalization range (degrees).

    Returns:
        End-effector position in normalized Cartesian space, shape (2,).
    """
    angles_deg = denormalize(angles_norm, norm_polar)
    angles_rad = np.deg2rad(angles_deg)
    lens = np.asarray(lengths_norm, dtype=np.float64)

    # Cumulative angles (each joint adds to the chain)
    cum_angles = np.cumsum(angles_rad)

    # End-effector = sum of link vectors
    x = np.sum(lens * np.cos(cum_angles))
    y = np.sum(lens * np.sin(cum_angles))
    return np.array([x, y])


def forward_kinematics_all(
    angles_norm: NDArray[np.float64],
    lengths_norm: NDArray[np.float64] | tuple[float, ...],
    norm_polar: tuple[float, float] = (-180.0, 180.0),
) -> NDArray[np.float64]:
    """Compute ALL joint positions (base + each joint endpoint).

    Returns:
        Positions array of shape (n_joints+1, 2), starting from base (0,0).
    """
    angles_deg = denormalize(angles_norm, norm_polar)
    angles_rad = np.deg2rad(angles_deg)
    lens = np.asarray(lengths_norm, dtype=np.float64)
    n = len(lens)

    cum_angles = np.cumsum(angles_rad)
    positions = np.zeros((n + 1, 2))
    for i in range(n):
        positions[i + 1] = positions[i] + lens[i] * np.array(
            [np.cos(cum_angles[i]), np.sin(cum_angles[i])]
        )
    return positions


def analytical_jacobian(
    angles_norm: NDArray[np.float64],
    lengths_norm: NDArray[np.float64] | tuple[float, ...],
    norm_polar: tuple[float, float] = (-180.0, 180.0),
) -> NDArray[np.float64]:
    """Analytical Jacobian of forward kinematics w.r.t. NORMALIZED angles.

    For a 3-joint planar arm, the Jacobian maps normalized angle changes
    to end-effector velocity. Includes the chain rule through the
    denormalization (angle_rad = π * angle_norm).

    Returns:
        Jacobian matrix of shape (2, n_joints).
    """
    angles_deg = denormalize(angles_norm, norm_polar)
    angles_rad = np.deg2rad(angles_deg)
    lens = np.asarray(lengths_norm, dtype=np.float64)
    n = len(lens)

    cum_angles = np.cumsum(angles_rad)

    # d(angle_rad)/d(angle_norm) = (hi - lo) / 2 * π / 180
    # For norm_polar = (-180, 180): = 360/2 * π/180 = π
    scale = (norm_polar[1] - norm_polar[0]) / 2.0 * np.pi / 180.0

    J = np.zeros((2, n))
    for k in range(n):
        # Sum over joints >= k (chain rule for cumulative angles)
        sx = 0.0
        sy = 0.0
        for i in range(k, n):
            sx += -lens[i] * np.sin(cum_angles[i])
            sy += lens[i] * np.cos(cum_angles[i])
        J[0, k] = scale * sx
        J[1, k] = scale * sy

    return J


# ---------------------------------------------------------------------------
# Bayesian model comparison & evidence accumulation
# ---------------------------------------------------------------------------

def bmc(
    probs: NDArray[np.float64],
    log_evidence: NDArray[np.float64],
    weight: float,
    gain_prior: float,
    gain_evidence: float,
) -> NDArray[np.float64]:
    """Bayesian model comparison.

    Combines prior preferences (gain_prior) with accumulated evidence
    (gain_evidence) via softmax.
    """
    E = -log_stable(probs) * gain_prior - log_evidence * gain_evidence
    return softmax(-E, weight)


def acc_log_evidence(
    eta: NDArray[np.float64],
    Eta_m: NDArray[np.float64],
    mu: NDArray[np.float64],
    pi: float,
    pi_m: float,
    p: float,
) -> NDArray[np.float64]:
    """Accumulate log evidence for Bayesian model comparison.

    Computes precision-weighted evidence for each dynamics model.

    Args:
        eta: Weighted prediction, shape (dim,).
        Eta_m: Individual model predictions, shape (n_models, dim).
        mu: Velocity belief, shape (dim,).
        pi: Observation precision.
        pi_m: Model precision.
        p: Prior precision.

    Returns:
        Log evidence increment for each model, shape (n_models,).
    """
    # Map to [0, 1] range for balanced computation
    eta_rng = normalize(eta, (-1.0, 1.0), rng=False)
    Eta_m_rng = normalize(Eta_m, (-1.0, 1.0), rng=False)
    mu_rng = normalize(mu, (-1.0, 1.0), rng=False)

    p_m = p - pi + pi_m

    Mu_m_rng = (p * mu_rng - pi * eta_rng + pi_m * Eta_m_rng) / p_m
    L_t = (
        p_m * Mu_m_rng**2
        - p * mu_rng**2
        + pi * eta_rng**2
        - pi_m * Eta_m_rng**2
    )

    # Sum over spatial dimensions, result shape (n_models,)
    return np.sum(L_t, axis=tuple(range(1, L_t.ndim)))
