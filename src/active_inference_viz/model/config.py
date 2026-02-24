"""Simulation configuration and result dataclasses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass(frozen=True)
class SimConfig:
    """Immutable simulation configuration. Hashable for @st.cache_data."""

    # --- Time ---
    dt: float = 0.3
    n_tau: int = 30
    n_cues: int = 15
    n_wait: int = 6

    # --- Targets (pixel space) ---
    t1_pos: tuple[float, float] = (-250.0, 400.0)
    t2_pos: tuple[float, float] = (250.0, 400.0)

    # --- Arm morphology ---
    lengths: tuple[float, float, float] = (250.0, 150.0, 50.0)
    start_angles: tuple[float, float, float] = (35.0, 150.0, 0.0)

    # --- Continuous — internal unit precisions ---
    pi_eta_x_int: float = 0.0
    pi_x_int: float = 1.0
    p_x_int: float = 2.0
    pi_prop: float = 1.0
    lr_int: float = 1.0

    # --- Continuous — external unit precisions ---
    pi_eta_x_ext: float = 0.45
    pi_x_ext: float = 1.0
    p_x_ext: float = 2.0
    pi_vis: float = 1.0
    lr_ext: float = 1.0
    lambda_ext: float = 0.5

    # --- Action ---
    lr_a: float = 1.0

    # --- Discrete inference ---
    n_policy: int = 1
    gain_prior: float = 1.0
    gain_evidence: float = 5.0
    k_h: float = 0.0
    w_c: float = 1.2
    w_h: float = 1.0

    # --- Learning ---
    omega_d: float = 1.0
    eta_d: float = 0.0
    omega_a_ext: float = 1.0
    eta_a_ext: float = 0.0
    omega_a_cue: float = 1.0
    eta_a_cue: float = 0.0
    alpha_c: float = 0.6
    alpha_h: float = 0.8

    # --- Cue sequence ---
    cue_sequence: tuple[int, ...] = (0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0)

    # --- Derived properties ---

    @property
    def n_joints(self) -> int:
        return len(self.lengths)

    @property
    def n_steps(self) -> int:
        return self.n_cues * self.n_tau + self.n_tau * self.n_wait

    @property
    def norm_polar(self) -> tuple[float, float]:
        return (-180.0, 180.0)

    @property
    def norm_cart(self) -> tuple[float, float]:
        s = sum(self.lengths)
        return (-s, s)

    @property
    def lengths_norm(self) -> tuple[float, ...]:
        """Link lengths normalized to [-1, 1] (Cartesian range)."""
        lo, hi = self.norm_cart
        return tuple((2.0 * (l - lo) / (hi - lo)) - 1.0 for l in self.lengths)

    @property
    def t1_pos_norm(self) -> tuple[float, float]:
        lo, hi = self.norm_cart
        return tuple((2.0 * (x - lo) / (hi - lo)) - 1.0 for x in self.t1_pos)  # type: ignore[return-value]

    @property
    def t2_pos_norm(self) -> tuple[float, float]:
        lo, hi = self.norm_cart
        return tuple((2.0 * (x - lo) / (hi - lo)) - 1.0 for x in self.t2_pos)  # type: ignore[return-value]

    @property
    def start_angles_norm(self) -> tuple[float, ...]:
        lo, hi = self.norm_polar
        return tuple((2.0 * (a - lo) / (hi - lo)) - 1.0 for a in self.start_angles)


@dataclass
class SimResult:
    """Output of a single trial simulation."""

    config: SimConfig

    # Kinematics — actual
    angles: NDArray[np.float64]  # (n_steps, n_joints) degrees
    pos: NDArray[np.float64]  # (n_steps, n_joints+1, 2) pixel coords

    # Kinematics — estimated (brain beliefs)
    est_angles: NDArray[np.float64]  # (n_steps, n_joints) degrees
    est_pos: NDArray[np.float64]  # (n_steps, 2) pixel coords

    # Discrete beliefs
    states: NDArray[np.float64]  # (n_steps, 2) P(T1), P(T2)
    causes: NDArray[np.float64]  # (n_steps, 3) o_ext beliefs
    cues: NDArray[np.float64]  # (n_cues, 2) cue presentations

    # Dynamics
    vel: NDArray[np.float64]  # (n_steps, 2) actual velocity
    est_vel: NDArray[np.float64]  # (n_steps, 2) believed velocity
    F_m: NDArray[np.float64]  # (n_steps, 3, 2) dynamics model predictions
    L_ext: NDArray[np.float64]  # (n_steps, 3) log evidence

    # Likelihoods (final)
    A_cue: NDArray[np.float64]  # (2, 2)
    A_ext: NDArray[np.float64]  # (3, 2)
