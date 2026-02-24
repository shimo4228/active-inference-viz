"""Continuous predictive coding units (pure NumPy, no autograd).

Reimplements simulation/unit.py from priorelli/embodied-decisions.
PyTorch autograd is replaced with analytical Jacobians for:
  - Forward kinematics: closed-form 2×3 Jacobian
  - Dynamics functions: trivial constant gradients (-λI or 0)
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from active_inference_viz.model.math_utils import (
    acc_log_evidence,
    analytical_jacobian,
    forward_kinematics,
)

if TYPE_CHECKING:
    from active_inference_viz.model.config import SimConfig

# Type aliases
DynFunc = Callable[[NDArray[np.float64]], NDArray[np.float64]]
DynJacFunc = Callable[[NDArray[np.float64]], NDArray[np.float64]]


# ---------------------------------------------------------------------------
# Dynamics functions (NumPy equivalents of brain.py's f_0, f_t1, f_t2)
# ---------------------------------------------------------------------------

def make_dynamics_stay() -> tuple[DynFunc, DynJacFunc]:
    """Stay dynamics: f(x) = 0, J(x) = 0."""
    def f(x: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.zeros_like(x)

    def jac(_x: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.zeros((len(_x), len(_x)))

    return f, jac


def make_dynamics_target(
    target_norm: NDArray[np.float64], lam: float,
) -> tuple[DynFunc, DynJacFunc]:
    """Target dynamics: f(x) = λ(target - x), J(x) = -λI."""
    target = np.asarray(target_norm, dtype=np.float64)

    def f(x: NDArray[np.float64]) -> NDArray[np.float64]:
        return (target - x) * lam

    def jac(x: NDArray[np.float64]) -> NDArray[np.float64]:
        return -lam * np.eye(len(x))

    return f, jac


# ---------------------------------------------------------------------------
# Predictive coding unit
# ---------------------------------------------------------------------------

class Unit:
    """Hierarchical predictive coding unit with analytical gradients.

    Maintains beliefs x = [x_pos, x_vel] (generalized coordinates).
    Replaces PyTorch autograd with explicit Jacobian computations.
    """

    def __init__(
        self,
        dim: tuple[int, int],
        pi_eta_x: float,
        pi_x: float,
        p_x: float,
        lr: float,
        F_m: list[DynFunc],
        F_m_jac: list[DynJacFunc],
        v: NDArray[np.float64],
        L: NDArray[np.float64],
        has_parent: bool = False,
    ) -> None:
        n_orders, n_dim = dim

        # Hidden states: [position, velocity]
        self.x = np.zeros((n_orders, n_dim))

        # Precisions
        self.pi_eta_x = pi_eta_x
        self.pi_x = pi_x
        self.p_x = p_x

        # Dynamics models and their Jacobians
        self.F_m = F_m
        self.F_m_jac = F_m_jac

        # Shared arrays with Discrete (v = o_ext, L = L_ext)
        self.v = v
        self.L = L

        # Learning rate
        self.lr = lr

        # Whether this unit has a parent unit providing g()
        self.has_parent = has_parent

        # Prediction errors and gradients (accumulated each step)
        self.eps_eta_x = np.zeros_like(self.x)
        self.grad_o = np.zeros_like(self.x)
        self.grad_x = np.zeros_like(self.x[0])
        self.eps_x = np.zeros_like(self.x[1])

        # Dynamics predictions
        self.pred_x = np.zeros_like(self.x[1])
        self.Preds_x = np.zeros((len(F_m), len(self.x[1])))

    def step_likelihood_prior(
        self, prior_input: NDArray[np.float64],
    ) -> None:
        """Likelihood step for unit WITHOUT parent (internal unit).

        Prior input is a constant (normalized initial angles).
        """
        self.eps_eta_x[0] = (self.x[0] - prior_input) * self.pi_eta_x

    def step_likelihood_parent(
        self,
        parent: Unit,
        cfg: SimConfig,
    ) -> None:
        """Likelihood step for unit WITH parent (external unit).

        Computes prediction error from forward kinematics and
        backpropagates gradient to parent via analytical Jacobian.
        """
        # Predict hand position from parent's joint angles
        parent_x0 = parent.x[0].copy()
        pred_eta_x = forward_kinematics(
            parent_x0,
            np.array(cfg.lengths_norm),
            cfg.norm_polar,
        )

        # Prior prediction error
        self.eps_eta_x[0] = (self.x[0] - pred_eta_x) * self.pi_eta_x

        # Backpropagate gradient to parent via analytical Jacobian.
        # In PyTorch: eps.backward(eps) computes VJP = -π * J^T @ ε
        # where π = precision, J = Jacobian of prediction function.
        J = analytical_jacobian(parent_x0, np.array(cfg.lengths_norm), cfg.norm_polar)
        parent.grad_o[0] += -self.pi_eta_x * (J.T @ self.eps_eta_x[0])

    def step_dynamics(self) -> None:
        """Compute dynamics prediction errors and accumulate evidence."""
        x0 = self.x[0].copy()

        # Compute predictions for each dynamics model
        for i, f in enumerate(self.F_m):
            self.Preds_x[i] = f(x0)

        # Weighted prediction
        self.pred_x = self.v @ self.Preds_x

        # Dynamics prediction error
        self.eps_x = (self.x[1] - self.pred_x) * self.pi_x

        # Gradient of dynamics w.r.t. x0
        # grad_x = eps_x^T @ (d(eps_x)/dx) where eps_x = (x1 - pred) * pi_x
        # d(eps_x)/dx = -pi_x * d(pred)/dx = -pi_x * Σ v_i * J_fi
        J_weighted = np.zeros((len(x0), len(x0)))
        for i, (jac_fn, vi) in enumerate(zip(self.F_m_jac, self.v)):
            J_weighted += vi * jac_fn(x0)

        # Vector-Jacobian product: eps_x @ (-pi_x * J_weighted)
        self.grad_x = -self.pi_x * (self.eps_x @ J_weighted)

        # Accumulate log evidence for Bayesian model comparison
        self.L += acc_log_evidence(
            self.pred_x, self.Preds_x, self.x[1],
            self.pi_x, self.pi_x, self.p_x,
        )

    def update(self, dt: float) -> None:
        """Integrate beliefs via gradient descent on free energy."""
        # Free energy gradient
        dF_dx = np.zeros_like(self.x)
        dF_dx += self.grad_o + self.eps_eta_x
        dF_dx[0] += self.grad_x
        dF_dx[1] += self.eps_x

        # Shift operator: [x_vel, 0]
        x_shifted = np.zeros_like(self.x)
        if self.x.shape[0] > 1:
            x_shifted[:-1] = self.x[1:]

        # Belief update
        x_dot = x_shifted - dF_dx
        self.x += dt * x_dot * self.lr

        # Reset accumulated errors
        self.eps_eta_x[:] = 0.0
        self.grad_o[:] = 0.0


# ---------------------------------------------------------------------------
# Observation unit
# ---------------------------------------------------------------------------

class Obs:
    """Observation unit that computes sensory prediction errors.

    For proprioceptive obs: g(int.x) = int.x[0]  (joint angles)
    For visual obs: g(ext.x) = ext.x  (position + velocity)
    """

    def __init__(
        self,
        dim: tuple[int, ...],
        pi_o: float,
        obs_type: str,
        lr_a: float | None = None,
    ) -> None:
        self.o = np.zeros(dim)
        self.pi_o = pi_o
        self.obs_type = obs_type  # "prop" or "vis"
        self.eps_o = np.zeros(dim)

        # Actions (only for proprioceptive)
        self.actions: NDArray[np.float64] | None = None
        self.lr_a = lr_a
        if lr_a is not None:
            self.actions = np.zeros(dim)

    def step(self, parent: Unit) -> None:
        """Compute prediction error and propagate gradient to parent.

        The VJP from PyTorch's backward(eps) includes a -precision factor:
        grad = -π_o * J^T @ ε_o. For identity prediction functions (g_prop, g_vis),
        J = I, so grad = -π_o * ε_o.
        """
        if self.obs_type == "prop":
            # g_prop: extract position order from internal unit
            pred = parent.x[0].copy()
            self.eps_o = (self.o - pred) * self.pi_o
            parent.grad_o[0] += -self.pi_o * self.eps_o
        elif self.obs_type == "vis":
            # g_vis: identity mapping from external unit
            pred = parent.x.copy()
            self.eps_o = (self.o - pred) * self.pi_o
            parent.grad_o += -self.pi_o * self.eps_o

    def update(self, dt: float) -> None:
        """Update actions (proprioceptive only)."""
        if self.actions is not None and self.lr_a is not None:
            a_dot = -dt * self.eps_o
            self.actions += dt * a_dot * self.lr_a
