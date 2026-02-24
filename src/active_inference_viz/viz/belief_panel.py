"""Belief panel: P(T1)/P(T2) time series with cue indicators."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from active_inference_viz.model.config import SimResult
from active_inference_viz.viz import theme


def create_belief_figure(result: SimResult, step: int) -> go.Figure:
    """Create the belief panel showing target probabilities over time.

    Shows:
    - P(T1) and P(T2) as time series up to current step
    - Cue presentation markers (vertical bands)
    - External cause beliefs (o_ext)
    - Current step indicator
    """
    cfg = result.config
    fig = go.Figure()

    steps = np.arange(step + 1)

    # --- Cue presentation bands ---
    for i in range(min(cfg.n_cues, len(result.cues))):
        cue_start = i * cfg.n_tau
        cue_end = (i + 1) * cfg.n_tau
        if cue_start > step:
            break
        cue = result.cues[i]
        color = theme.CUE_T1 if cue[0] > 0.5 else theme.CUE_T2 if cue[1] > 0.5 else "rgba(200,200,200,0.2)"
        fig.add_vrect(
            x0=cue_start, x1=min(cue_end, step),
            fillcolor=color, opacity=0.3,
            layer="below", line_width=0,
        )

    # --- P(T1) and P(T2) ---
    fig.add_trace(go.Scatter(
        x=steps, y=result.states[:step + 1, 0],
        mode="lines",
        line=dict(color=theme.TARGET_1, width=2.5),
        name="P(T1)",
    ))
    fig.add_trace(go.Scatter(
        x=steps, y=result.states[:step + 1, 1],
        mode="lines",
        line=dict(color=theme.TARGET_2, width=2.5),
        name="P(T2)",
    ))

    # --- External causes (o_ext) as dashed lines ---
    fig.add_trace(go.Scatter(
        x=steps, y=result.causes[:step + 1, 0],
        mode="lines",
        line=dict(color=theme.TARGET_1, width=1, dash="dash"),
        name="o_ext[T1]",
        opacity=0.6,
    ))
    fig.add_trace(go.Scatter(
        x=steps, y=result.causes[:step + 1, 1],
        mode="lines",
        line=dict(color=theme.TARGET_2, width=1, dash="dash"),
        name="o_ext[T2]",
        opacity=0.6,
    ))
    fig.add_trace(go.Scatter(
        x=steps, y=result.causes[:step + 1, 2],
        mode="lines",
        line=dict(color=theme.STAY, width=1, dash="dash"),
        name="o_ext[Stay]",
        opacity=0.6,
    ))

    # --- Current step indicator ---
    fig.add_vline(x=step, line_dash="solid", line_color="black", line_width=1)

    layout = theme.belief_layout()
    layout["xaxis"]["range"] = [0, cfg.n_steps]
    fig.update_layout(**layout)
    fig.update_layout(title="Discrete Beliefs")

    return fig
