"""Brain: connects discrete inference with continuous predictive coding.

Reimplements simulation/brain.py from priorelli/embodied-decisions.
Architecture: internal unit (joints) → external unit (hand) ←→ discrete (targets).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from active_inference_viz.model.config import SimConfig
from active_inference_viz.model.continuous import (
    Obs,
    Unit,
    make_dynamics_stay,
    make_dynamics_target,
)
from active_inference_viz.model.discrete import Discrete
from active_inference_viz.model.math_utils import denormalize, normalize


class Brain:
    """Active inference brain combining discrete and continuous inference."""

    def __init__(self, cfg: SimConfig) -> None:
        self.cfg = cfg

        # --- Discrete module ---
        self.discrete = Discrete(cfg)

        # --- Dynamics functions ---
        f_stay, j_stay = make_dynamics_stay()
        f_t1, j_t1 = make_dynamics_target(
            np.array(cfg.t1_pos_norm), cfg.lambda_ext,
        )
        f_t2, j_t2 = make_dynamics_target(
            np.array(cfg.t2_pos_norm), cfg.lambda_ext,
        )

        # --- Internal unit (joint angles) ---
        f_int_stay, j_int_stay = make_dynamics_stay()
        self.int_unit = Unit(
            dim=(2, cfg.n_joints),
            pi_eta_x=cfg.pi_eta_x_int,
            pi_x=cfg.pi_x_int,
            p_x=cfg.p_x_int,
            lr=cfg.lr_int,
            F_m=[f_int_stay],
            F_m_jac=[j_int_stay],
            v=np.zeros(1),
            L=np.zeros(1),
            has_parent=False,
        )

        # --- External unit (hand position) ---
        self.ext_unit = Unit(
            dim=(2, 2),  # 2 orders, 2D position
            pi_eta_x=cfg.pi_eta_x_ext,
            pi_x=cfg.pi_x_ext,
            p_x=cfg.p_x_ext,
            lr=cfg.lr_ext,
            F_m=[f_t1, f_t2, f_stay],
            F_m_jac=[j_t1, j_t2, j_stay],
            v=self.discrete.o_ext,  # Shared reference
            L=self.discrete.L_ext,  # Shared reference
            has_parent=True,
        )

        # --- Observation units ---
        self.prop = Obs(
            dim=(cfg.n_joints,),
            pi_o=cfg.pi_prop,
            obs_type="prop",
            lr_a=cfg.lr_a,
        )
        self.vis = Obs(
            dim=(2, 2),
            pi_o=cfg.pi_vis,
            obs_type="vis",
        )

        # Prior input for internal unit (normalized initial angles)
        self._int_prior = np.array(cfg.start_angles_norm)

    def init_belief(
        self,
        angles_deg: NDArray[np.float64],
        pos_pixel: NDArray[np.float64],
    ) -> None:
        """Initialize beliefs from actual body state."""
        cfg = self.cfg

        # Internal unit: belief = actual joint angles
        int_start_norm = normalize(angles_deg, cfg.norm_polar)
        self.int_unit.x[0] = int_start_norm
        self.int_unit.x[1] = 0.0

        # External unit: belief = actual hand position
        ext_start_norm = normalize(pos_pixel, cfg.norm_cart)
        self.ext_unit.x[0] = ext_start_norm
        self.ext_unit.x[1] = 0.0

        # Reset discrete beliefs
        self.discrete.o_ext[:] = [0.0, 0.0, 1.0]

        # Reset observations and actions
        self.prop.o[:] = 0.0
        self.vis.o[:] = 0.0
        if self.prop.actions is not None:
            self.prop.actions[:] = 0.0

    def inference_step(
        self,
        obs_prop: NDArray[np.float64],
        obs_vis: NDArray[np.float64],
        cue: NDArray[np.float64],
        step: int,
    ) -> NDArray[np.float64]:
        """Run one inference step.

        Args:
            obs_prop: Normalized joint angle observations, shape (n_joints,).
            obs_vis: Normalized visual observations, shape (2, 2).
            cue: Current cue vector, shape (2,).
            step: Current simulation step index.

        Returns:
            Action output in degrees (angular velocity for body update).
        """
        cfg = self.cfg

        # --- Discrete step (every n_tau steps, during cue period) ---
        if (step + 1) % cfg.n_tau == 0 and step < cfg.n_steps - cfg.n_tau * cfg.n_wait:
            self.discrete.step(cue)

        # --- Set observations ---
        self.prop.o = obs_prop.copy()
        self.vis.o = obs_vis.copy()

        # --- Message passing: likelihood ---
        self.int_unit.step_likelihood_prior(self._int_prior)
        self.ext_unit.step_likelihood_parent(self.int_unit, cfg)

        # --- Message passing: observations ---
        self.prop.step(self.int_unit)
        self.vis.step(self.ext_unit)

        # --- Message passing: dynamics ---
        self.int_unit.step_dynamics()
        self.ext_unit.step_dynamics()

        # --- Update beliefs ---
        self.int_unit.update(cfg.dt)
        self.ext_unit.update(cfg.dt)
        self.prop.update(cfg.dt)
        self.vis.update(cfg.dt)

        # --- Return action (denormalized to degrees) ---
        if self.prop.actions is not None:
            return denormalize(self.prop.actions, cfg.norm_polar)
        return np.zeros(cfg.n_joints)
