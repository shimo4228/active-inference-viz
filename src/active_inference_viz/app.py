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
    """Run simulation with caching (SimConfig is frozen -> hashable)."""
    return run_trial(cfg)


def main() -> None:
    st.title("Active Inference Visualizer")

    # ----------------------------------------------------------------
    # How to use
    # ----------------------------------------------------------------
    with st.expander("**How to use this tool**", expanded=False):
        st.markdown("""
**What you're seeing:**
A 3-joint arm that decides which of two targets to reach, based on noisy sensory cues.
This implements the "embodied decisions as active inference" model (Priorelli et al., 2025).

**Two coupled processes run simultaneously:**
1. **Discrete inference** (bottom panel): The brain accumulates evidence from cues
   to decide which target is correct — P(T1) vs P(T2).
2. **Continuous inference** (top panel): The arm moves toward the believed target,
   driven by predictive coding. The arm position reflects the brain's belief.

**How to explore:**
- Move the **step slider** to scrub through the simulation timestep by timestep
- Change **cue pattern** to see how different evidence streams affect the decision
- Increase **evidence gain** to see faster/stronger commitment
- Change **cue reliability** (alpha_c) to make cues more/less informative
- Change **dynamics gain** (lambda) to speed up/slow down reaching
        """)

    # ----------------------------------------------------------------
    # Sidebar: Parameters
    # ----------------------------------------------------------------
    with st.sidebar:
        st.header("Parameters")

        st.subheader("Cue Sequence")
        cue_mode = st.selectbox(
            "Cue pattern",
            ["Default (paper)", "All T1", "All T2", "Alternating", "Random 50/50"],
        )
        n_cues = st.slider("Number of cues", 3, 30, 15)

        st.subheader("Discrete Inference")
        gain_evidence = st.slider("Evidence gain", 0.1, 20.0, 5.0, step=0.1)
        alpha_c = st.slider("Cue reliability (alpha_c)", 0.5, 1.0, 0.6, step=0.01)
        w_c = st.slider("State precision (w_c)", 0.1, 5.0, 1.2, step=0.1)

        st.subheader("Continuous Inference")
        pi_eta_x_ext = st.slider("Prior precision (ext)", 0.0, 2.0, 0.45, step=0.05)
        lambda_ext = st.slider("Dynamics gain (lambda)", 0.1, 2.0, 0.5, step=0.1)

        with st.expander("Advanced"):
            n_tau = st.slider("Steps per cue (n_tau)", 10, 60, 30)
            n_wait = st.slider("Wait periods", 1, 10, 6)
            dt = st.slider("Timestep (dt)", 0.05, 1.0, 0.3, step=0.05)
            t1_x = st.slider("T1 x position", -400, 0, -250)
            t2_x = st.slider("T2 x position", 0, 400, 250)
            t_y = st.slider("Target y position", 100, 500, 400)

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
        default = (0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0)
        cue_seq = tuple(default[i] if i < len(default) else 0 for i in range(n_cues))

    # ----------------------------------------------------------------
    # Build config and run simulation
    # ----------------------------------------------------------------
    # Read advanced params (use defaults if expander not opened)
    _n_tau = n_tau if "n_tau" in dir() else 30
    _n_wait = n_wait if "n_wait" in dir() else 6
    _dt = dt if "dt" in dir() else 0.3
    _t1_x = t1_x if "t1_x" in dir() else -250
    _t2_x = t2_x if "t2_x" in dir() else 250
    _t_y = t_y if "t_y" in dir() else 400

    cfg = SimConfig(
        dt=_dt,
        n_tau=_n_tau,
        n_cues=n_cues,
        n_wait=_n_wait,
        t1_pos=(float(_t1_x), float(_t_y)),
        t2_pos=(float(_t2_x), float(_t_y)),
        gain_evidence=gain_evidence,
        alpha_c=alpha_c,
        w_c=w_c,
        pi_eta_x_ext=pi_eta_x_ext,
        lambda_ext=lambda_ext,
        cue_sequence=cue_seq,
    )

    result = cached_run_trial(cfg)

    # ----------------------------------------------------------------
    # Step slider (prominent, full width)
    # ----------------------------------------------------------------
    step = st.slider(
        "Simulation step",
        0,
        cfg.n_steps - 1,
        0,
        key="step_slider",
        help="Scrub through the simulation. Cues are presented in the first part; "
             "the arm reaches during the wait period.",
    )

    # ----------------------------------------------------------------
    # Info cards
    # ----------------------------------------------------------------
    belief = result.states[step]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("P(Target 1)", f"{belief[0]:.3f}")
    c2.metric("P(Target 2)", f"{belief[1]:.3f}")
    leading = "T1" if belief[0] > belief[1] + 0.01 else "T2" if belief[1] > belief[0] + 0.01 else "Undecided"
    c3.metric("Decision", leading)
    cue_phase = step < cfg.n_cues * cfg.n_tau
    c4.metric("Phase", "Cue accumulation" if cue_phase else "Reaching")

    # ----------------------------------------------------------------
    # Main panels
    # ----------------------------------------------------------------
    arm_fig = create_arm_figure(result, step)
    st.plotly_chart(arm_fig, use_container_width=True)

    belief_fig = create_belief_figure(result, step)
    st.plotly_chart(belief_fig, use_container_width=True)

    # ----------------------------------------------------------------
    # Footer
    # ----------------------------------------------------------------
    st.divider()
    st.caption(
        "Based on: Priorelli et al. (2025) 'Embodied decisions as active inference', "
        "PLOS Computational Biology. "
        "Math core reimplemented in pure NumPy (no PyTorch)."
    )


if __name__ == "__main__":
    main()
