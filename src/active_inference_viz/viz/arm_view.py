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
    - Estimated hand position (brain's ext unit belief)
    """
    cfg = result.config
    fig = go.Figure()

    # --- Workspace range (based on arm reach + targets) ---
    max_reach = sum(cfg.lengths)
    t1 = np.array(cfg.t1_pos)
    t2 = np.array(cfg.t2_pos)
    x_lo = min(-max_reach, t1[0], t2[0]) - 50
    x_hi = max(max_reach, t1[0], t2[0]) + 50
    y_lo = min(-max_reach, 0) - 50
    y_hi = max(max_reach, t1[1], t2[1]) + 50

    # --- Targets ---
    belief = result.states[step]

    fig.add_trace(go.Scatter(
        x=[t1[0]], y=[t1[1]],
        mode="markers+text",
        marker=dict(
            size=25, color=theme.TARGET_1,
            opacity=0.3 + 0.7 * float(belief[0]),
            line=dict(width=2, color=theme.TARGET_1),
        ),
        text=[f"T1 ({belief[0]:.2f})"],
        textposition="top center",
        textfont=dict(size=14, color=theme.TARGET_1),
        name="Target 1",
    ))
    fig.add_trace(go.Scatter(
        x=[t2[0]], y=[t2[1]],
        mode="markers+text",
        marker=dict(
            size=25, color=theme.TARGET_2,
            opacity=0.3 + 0.7 * float(belief[1]),
            line=dict(width=2, color=theme.TARGET_2),
        ),
        text=[f"T2 ({belief[1]:.2f})"],
        textposition="top center",
        textfont=dict(size=14, color=theme.TARGET_2),
        name="Target 2",
    ))

    # --- Hand trajectory trail ---
    trail_start = max(0, step - 200)
    trail = result.pos[trail_start:step + 1, -1, :]
    if len(trail) > 1:
        fig.add_trace(go.Scatter(
            x=trail[:, 0].tolist(),
            y=trail[:, 1].tolist(),
            mode="lines",
            line=dict(color=theme.TRAJECTORY, width=2, dash="dot"),
            opacity=0.5,
            name="Trajectory",
            showlegend=True,
        ))

    # --- Arm links ---
    arm_pos = result.pos[step]  # (n_joints+1, 2)
    n_pts = arm_pos.shape[0]
    sizes = [10] * (n_pts - 1) + [16]  # Larger marker for hand
    colors = [theme.ARM_JOINT] * (n_pts - 1) + [theme.HAND]

    fig.add_trace(go.Scatter(
        x=arm_pos[:, 0].tolist(),
        y=arm_pos[:, 1].tolist(),
        mode="lines+markers",
        line=dict(color=theme.ARM_LINK, width=6),
        marker=dict(size=sizes, color=colors),
        name="Arm",
    ))

    # --- Estimated hand position (ext unit belief) ---
    est = result.est_pos[step]
    fig.add_trace(go.Scatter(
        x=[float(est[0])], y=[float(est[1])],
        mode="markers",
        marker=dict(
            size=12, color=theme.HAND, symbol="x",
            line=dict(width=2, color="white"),
        ),
        name="Brain belief (ext)",
    ))

    # --- Layout ---
    fig.update_layout(
        **theme.LAYOUT_DEFAULTS,
        xaxis=dict(
            title="X (px)", range=[x_lo, x_hi],
            scaleanchor="y", scaleratio=1,
            gridcolor="#E8E8E8", zeroline=True, zerolinecolor="#CCC",
        ),
        yaxis=dict(
            title="Y (px)", range=[y_lo, y_hi],
            gridcolor="#E8E8E8", zeroline=True, zerolinecolor="#CCC",
        ),
        showlegend=True,
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.8)"),
        height=520,
        title=dict(text=f"Arm View — Step {step} / {cfg.n_steps - 1}", font=dict(size=16)),
    )

    return fig
