"""Color palette and Plotly layout configuration."""

from __future__ import annotations

# --- Color palette ---
TARGET_1 = "#E74C3C"  # Red
TARGET_2 = "#3498DB"  # Blue
STAY = "#95A5A6"  # Gray
ARM_LINK = "#2C3E50"  # Dark navy
ARM_JOINT = "#34495E"  # Slightly lighter
HAND = "#E67E22"  # Orange
TRAJECTORY = "#27AE60"  # Green
BACKGROUND = "#FAFAFA"
CUE_T1 = "#FADBD8"  # Light red
CUE_T2 = "#D6EAF8"  # Light blue

# --- Plotly layout defaults ---
LAYOUT_DEFAULTS = dict(
    font=dict(family="Inter, sans-serif", size=12),
    paper_bgcolor=BACKGROUND,
    plot_bgcolor="white",
    margin=dict(l=60, r=20, t=40, b=40),
)


def arm_layout(norm_cart: tuple[float, float]) -> dict:
    """Layout for the arm view panel."""
    lo, hi = norm_cart
    return {
        **LAYOUT_DEFAULTS,
        "xaxis": dict(
            title="X",
            range=[lo * 1.1, hi * 1.1],
            scaleanchor="y",
            scaleratio=1,
            gridcolor="#E8E8E8",
        ),
        "yaxis": dict(
            title="Y",
            range=[lo * 0.5, hi * 1.2],
            gridcolor="#E8E8E8",
        ),
        "showlegend": True,
        "legend": dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.8)"),
        "height": 500,
    }


def belief_layout() -> dict:
    """Layout for the belief panel."""
    return {
        **LAYOUT_DEFAULTS,
        "xaxis": dict(title="Step", gridcolor="#E8E8E8"),
        "yaxis": dict(title="Probability", range=[-0.05, 1.05], gridcolor="#E8E8E8"),
        "showlegend": True,
        "legend": dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.8)"),
        "height": 300,
    }
