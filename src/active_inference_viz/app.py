"""Active Inference Visualizer — Streamlit Dashboard.

Interactive visualization of embodied decision-making via active inference
(Priorelli et al., 2025).
"""

from __future__ import annotations

import streamlit as st

from active_inference_viz.model.config import SimConfig
from active_inference_viz.model.simulation import run_trial
from active_inference_viz.viz.arm_view import create_arm_figure
from active_inference_viz.viz.belief_panel import create_belief_figure

st.set_page_config(
    page_title="Active Inference Visualizer",
    page_icon="🧠",
    layout="wide",
)


@st.cache_data
def cached_run_trial(cfg: SimConfig):  # noqa: ANN201
    """Run simulation with caching (SimConfig is frozen → hashable)."""
    return run_trial(cfg)


def main() -> None:
    st.title("Active Inference Visualizer")
    st.caption(
        "Interactive exploration of embodied decision-making "
        "(Priorelli et al., 2025)"
    )

    # ----------------------------------------------------------------
    # Sidebar: Parameters
    # ----------------------------------------------------------------
    with st.sidebar:
        st.header("Parameters")

        st.subheader("Simulation")
        n_cues = st.slider("Number of cues", 3, 30, 15)
        n_tau = st.slider("Steps per cue (n_tau)", 10, 60, 30)
        n_wait = st.slider("Wait periods after cues", 1, 10, 6)
        dt = st.slider("Integration timestep (dt)", 0.05, 1.0, 0.3, step=0.05)

        st.subheader("Targets")
        t1_x = st.slider("T1 x position", -400, 0, -250)
        t2_x = st.slider("T2 x position", 0, 400, 250)
        t_y = st.slider("Target y position", 100, 500, 400)

        st.subheader("Discrete Inference")
        gain_evidence = st.slider("Evidence gain", 0.1, 20.0, 5.0, step=0.1)
        alpha_c = st.slider("Cue reliability (alpha_c)", 0.5, 1.0, 0.6, step=0.01)
        w_c = st.slider("State precision (w_c)", 0.1, 5.0, 1.2, step=0.1)

        st.subheader("Continuous Inference")
        pi_eta_x_ext = st.slider("External prior precision", 0.0, 2.0, 0.45, step=0.05)
        lambda_ext = st.slider("Dynamics gain (lambda)", 0.1, 2.0, 0.5, step=0.1)

        st.subheader("Cue Sequence")
        cue_mode = st.selectbox(
            "Cue pattern",
            ["Default (paper)", "All T1", "All T2", "Alternating", "Random 50/50"],
        )

    # ----------------------------------------------------------------
    # Build cue sequence
    # ----------------------------------------------------------------
    if cue_mode == "All T1":
        cue_seq = tuple([0] * n_cues)
    elif cue_mode == "All T2":
        cue_seq = tuple([1] * n_cues)
    elif cue_mode == "Alternating":
        cue_seq = tuple(i % 2 for i in range(n_cues))
    elif cue_mode == "Random 50/50":
        import numpy as np
        rng = np.random.default_rng(42)
        cue_seq = tuple(int(x) for x in rng.choice([0, 1], size=n_cues))
    else:
        # Default paper sequence (padded/truncated to n_cues)
        default = (0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0)
        cue_seq = tuple(default[i] if i < len(default) else 0 for i in range(n_cues))

    # ----------------------------------------------------------------
    # Build config and run simulation
    # ----------------------------------------------------------------
    cfg = SimConfig(
        dt=dt,
        n_tau=n_tau,
        n_cues=n_cues,
        n_wait=n_wait,
        t1_pos=(float(t1_x), float(t_y)),
        t2_pos=(float(t2_x), float(t_y)),
        gain_evidence=gain_evidence,
        alpha_c=alpha_c,
        w_c=w_c,
        pi_eta_x_ext=pi_eta_x_ext,
        lambda_ext=lambda_ext,
        cue_sequence=cue_seq,
    )

    result = cached_run_trial(cfg)

    # ----------------------------------------------------------------
    # Step slider
    # ----------------------------------------------------------------
    step = st.slider(
        "Simulation step",
        0,
        cfg.n_steps - 1,
        cfg.n_steps - 1,
        key="step_slider",
    )

    # ----------------------------------------------------------------
    # Main panels
    # ----------------------------------------------------------------
    col1, col2 = st.columns([3, 2])

    with col1:
        arm_fig = create_arm_figure(result, step)
        st.plotly_chart(arm_fig, use_container_width=True)

    with col2:
        # Info cards
        belief = result.states[step]
        c1, c2, c3 = st.columns(3)
        c1.metric("P(T1)", f"{belief[0]:.3f}")
        c2.metric("P(T2)", f"{belief[1]:.3f}")
        leading = "T1" if belief[0] > belief[1] else "T2" if belief[1] > belief[0] else "Tied"
        c3.metric("Leading", leading)

    # Belief panel (full width)
    belief_fig = create_belief_figure(result, step)
    st.plotly_chart(belief_fig, use_container_width=True)

    # ----------------------------------------------------------------
    # Footer
    # ----------------------------------------------------------------
    st.divider()
    st.caption(
        "Based on: Priorelli et al. (2025) 'Embodied decisions as active inference', "
        "PLOS Computational Biology. "
        "Mathematical core reimplemented in pure NumPy."
    )


if __name__ == "__main__":
    main()
