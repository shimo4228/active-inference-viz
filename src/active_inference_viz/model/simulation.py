"""Simulation runner: executes a trial and returns SimResult.

Simplified body model (no Pymunk physics): actions directly update joint angles.
This captures the core active inference dynamics faithfully.
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


def _generate_cue(cfg: SimConfig, cue_idx: int) -> NDArray[np.float64]:
    """Generate a one-hot cue vector from the cue sequence."""
    cue = np.zeros(2)
    if cue_idx < len(cfg.cue_sequence):
        cue[cfg.cue_sequence[cue_idx]] = 1.0
    return cue


def run_trial(cfg: SimConfig) -> SimResult:
    """Execute a single trial simulation.

    Args:
        cfg: Immutable simulation configuration.

    Returns:
        SimResult with all logged data for visualization.
    """
    n_steps = cfg.n_steps
    n_joints = cfg.n_joints

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

    # --- Initialize body state ---
    actual_angles = np.array(cfg.start_angles, dtype=np.float64)
    prev_hand_pos = np.zeros(2)

    # Compute initial positions
    angles_norm = normalize(actual_angles, cfg.norm_polar)
    all_pos = forward_kinematics_all(angles_norm, np.array(cfg.lengths_norm), cfg.norm_polar)
    hand_pos = denormalize(all_pos[-1], cfg.norm_cart)
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
        elif cue_step >= cfg.n_cues:
            # Wait period: no new cues, maintain last belief
            pass

        # --- Generate observations ---
        angles_norm = normalize(actual_angles, cfg.norm_polar)
        all_pos = forward_kinematics_all(
            angles_norm, np.array(cfg.lengths_norm), cfg.norm_polar,
        )
        hand_pos_norm = all_pos[-1]
        hand_pos = denormalize(hand_pos_norm, cfg.norm_cart)

        # Compute velocity
        hand_vel = (hand_pos - prev_hand_pos) / max(cfg.dt, 1e-8)
        hand_vel_norm = normalize(hand_vel, cfg.norm_cart)

        # Proprioceptive observation (normalized angles)
        obs_prop = angles_norm.copy()

        # Visual observation [position, velocity] (normalized)
        obs_vis = np.array([hand_pos_norm, hand_vel_norm])

        # --- Brain inference ---
        action_deg = brain.inference_step(obs_prop, obs_vis, current_cue, step)

        # --- Update body state ---
        actual_angles += action_deg
        # Clamp to valid range
        actual_angles = np.clip(actual_angles, -180.0, 180.0)

        prev_hand_pos = hand_pos.copy()

        # --- Log data ---
        angles[step] = actual_angles
        est_angles[step] = denormalize(brain.int_unit.x[0], cfg.norm_polar)
        # Denormalize all positions to pixel space
        pos[step, 0] = np.array([0.0, 0.0])  # Base
        for j in range(n_joints):
            pos[step, j + 1] = denormalize(all_pos[j + 1], cfg.norm_cart)
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
