"""Simulation runner: executes a trial and returns SimResult.

Body model: the arm tracks the brain's believed angles with a simple
spring-like lag. This creates the prediction-error loop that drives
active inference while being faithful to the theory without needing
a full physics engine (Pymunk).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from active_inference_viz.model.brain import Brain
from active_inference_viz.model.config import SimConfig, SimResult
from active_inference_viz.model.math_utils import (
    denormalize,
    forward_kinematics_all,
    normalize,
)

# How fast the body tracks the brain's belief (higher = faster tracking)
BODY_TRACKING_GAIN = 8.0


def _generate_cue(cfg: SimConfig, cue_idx: int) -> NDArray[np.float64]:
    """Generate a one-hot cue vector from the cue sequence."""
    cue = np.zeros(2)
    if cue_idx < len(cfg.cue_sequence):
        cue[cfg.cue_sequence[cue_idx]] = 1.0
    return cue


def run_trial(cfg: SimConfig) -> SimResult:
    """Execute a single trial simulation.

    The simulation loop:
    1. Brain receives observations from the body
    2. Brain runs inference (discrete + continuous)
    3. Body tracks toward brain's believed angles (spring model)
    4. The tracking lag creates prediction errors that drive the loop

    Args:
        cfg: Immutable simulation configuration.

    Returns:
        SimResult with all logged data for visualization.
    """
    n_steps = cfg.n_steps
    n_joints = cfg.n_joints
    lengths_norm = np.array(cfg.lengths_norm)

    # --- Initialize arrays ---
    angles = np.zeros((n_steps, n_joints))
    est_angles = np.zeros((n_steps, n_joints))
    pos = np.zeros((n_steps, n_joints + 1, 2))
    est_pos = np.zeros((n_steps, 2))
    states = np.zeros((n_steps, 2))
    causes = np.zeros((n_steps, 3))
    cues_log = np.zeros((cfg.n_cues, 2))
    vel = np.zeros((n_steps, 2))
    est_vel = np.zeros((n_steps, 2))
    F_m = np.zeros((n_steps, 3, 2))
    L_ext = np.zeros((n_steps, 3))

    # --- Initialize body state (actual joint angles) ---
    actual_angles = np.array(cfg.start_angles, dtype=np.float64)
    actual_angles_norm = normalize(actual_angles, cfg.norm_polar)

    # Compute initial hand position
    init_all_pos = forward_kinematics_all(actual_angles_norm, lengths_norm, cfg.norm_polar)
    hand_pos = denormalize(init_all_pos[-1], cfg.norm_cart)
    prev_hand_pos = hand_pos.copy()

    # --- Initialize brain ---
    brain = Brain(cfg)
    brain.init_belief(actual_angles, hand_pos)

    # --- Current cue ---
    current_cue = np.zeros(2)
    cue_idx = 0

    # --- Simulation loop ---
    for step in range(n_steps):
        # Update cue at cue presentation boundaries
        cue_step = step // cfg.n_tau
        if step % cfg.n_tau == 0 and cue_step < cfg.n_cues:
            current_cue = _generate_cue(cfg, cue_idx)
            cues_log[cue_idx] = current_cue
            cue_idx += 1

        # --- Generate observations from actual body ---
        all_pos = forward_kinematics_all(actual_angles_norm, lengths_norm, cfg.norm_polar)
        hand_pos_norm = all_pos[-1]
        hand_pos = denormalize(hand_pos_norm, cfg.norm_cart)
        hand_vel = (hand_pos - prev_hand_pos) / max(cfg.dt, 1e-8)
        hand_vel_norm = normalize(hand_vel, cfg.norm_cart)

        obs_prop = actual_angles_norm.copy()
        obs_vis = np.array([hand_pos_norm, hand_vel_norm])

        # --- Brain inference ---
        brain.inference_step(obs_prop, obs_vis, current_cue, step)

        # --- Body tracks brain's belief (spring-like dynamics) ---
        believed_angles_norm = brain.int_unit.x[0].copy()
        actual_angles_norm += (believed_angles_norm - actual_angles_norm) * BODY_TRACKING_GAIN * cfg.dt
        actual_angles = denormalize(actual_angles_norm, cfg.norm_polar)

        prev_hand_pos = hand_pos.copy()

        # --- Log data ---
        # "angles" = actual body angles; "est_angles" = brain's belief
        angles[step] = actual_angles
        est_angles[step] = denormalize(brain.int_unit.x[0], cfg.norm_polar)

        # Compute believed arm positions for visualization
        believed_all_pos = forward_kinematics_all(
            brain.int_unit.x[0], lengths_norm, cfg.norm_polar,
        )
        pos[step, 0] = np.array([0.0, 0.0])  # Base
        for j in range(n_joints):
            pos[step, j + 1] = denormalize(believed_all_pos[j + 1], cfg.norm_cart)

        est_pos[step] = denormalize(brain.ext_unit.x[0], cfg.norm_cart)
        states[step] = brain.discrete.prior
        causes[step] = brain.discrete.o_ext
        vel[step] = hand_vel
        est_vel[step] = denormalize(brain.ext_unit.x[1], cfg.norm_cart)
        F_m[step] = denormalize(brain.ext_unit.Preds_x, cfg.norm_cart)
        L_ext[step] = brain.discrete.L_ext

    return SimResult(
        config=cfg,
        angles=angles,
        pos=pos,
        est_angles=est_angles,
        est_pos=est_pos,
        states=states,
        causes=causes,
        cues=cues_log,
        vel=vel,
        est_vel=est_vel,
        F_m=F_m,
        L_ext=L_ext,
        A_cue=brain.discrete.A_cue.copy(),
        A_ext=brain.discrete.A_ext.copy(),
    )
