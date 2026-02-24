"""2D arm visualization with trajectory and targets (Plotly)."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from active_inference_viz.model.config import SimResult
from active_inference_viz.viz import theme


def create_arm_figure(result: SimResult, step: int) -> go.Figure:
    """Create the arm view figure for a given simulation step.

    Shows:
    - 3-joint arm links and joints
    - Target positions (colored by belief strength)
    - Hand trajectory trail up to current step
    - Estimated hand position
    """
    cfg = result.config
    fig = go.Figure()

    # --- Targets ---
    t1 = np.array(cfg.t1_pos)
    t2 = np.array(cfg.t2_pos)
    belief = result.states[step]

    fig.add_trace(go.Scatter(
        x=[t1[0]], y=[t1[1]],
        mode="markers+text",
        marker=dict(size=20, color=theme.TARGET_1, opacity=0.3 + 0.7 * belief[0]),
        text=[f"T1 ({belief[0]:.2f})"],
        textposition="top center",
        name="Target 1",
    ))
    fig.add_trace(go.Scatter(
        x=[t2[0]], y=[t2[1]],
        mode="markers+text",
        marker=dict(size=20, color=theme.TARGET_2, opacity=0.3 + 0.7 * belief[1]),
        text=[f"T2 ({belief[1]:.2f})"],
        textposition="top center",
        name="Target 2",
    ))

    # --- Trajectory trail ---
    trail_start = max(0, step - 100)
    trail = result.pos[trail_start:step + 1, -1, :]  # Hand positions
    if len(trail) > 1:
        fig.add_trace(go.Scatter(
            x=trail[:, 0], y=trail[:, 1],
            mode="lines",
            line=dict(color=theme.TRAJECTORY, width=2, dash="dot"),
            opacity=0.6,
            name="Trajectory",
        ))

    # --- Arm links ---
    arm_pos = result.pos[step]  # (n_joints+1, 2)
    fig.add_trace(go.Scatter(
        x=arm_pos[:, 0], y=arm_pos[:, 1],
        mode="lines+markers",
        line=dict(color=theme.ARM_LINK, width=4),
        marker=dict(
            size=[8, 10, 8, 12],
            color=[theme.ARM_JOINT, theme.ARM_JOINT, theme.ARM_JOINT, theme.HAND],
        ),
        name="Arm",
    ))

    # --- Estimated position ---
    est = result.est_pos[step]
    fig.add_trace(go.Scatter(
        x=[est[0]], y=[est[1]],
        mode="markers",
        marker=dict(size=10, color=theme.HAND, symbol="x", line=dict(width=2)),
        name="Estimated",
    ))

    fig.update_layout(**theme.arm_layout(cfg.norm_cart))
    fig.update_layout(title=f"Step {step} / {cfg.n_steps}")

    return fig
