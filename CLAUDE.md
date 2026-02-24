# Active Inference Visualizer

Interactive Streamlit dashboard for visualizing embodied decision-making via active inference (Priorelli et al., 2025).

## Tech Stack

- **Language**: Python 3.13+
- **Package Manager**: uv + pyproject.toml
- **Framework**: Streamlit
- **Computation**: NumPy, SciPy (pure NumPy reimplementation of PyTorch original)
- **Visualization**: Plotly (interactive), Matplotlib (static export)

## Directory Structure

```
src/active_inference_viz/
├── model/          # Mathematical core (config, math_utils, discrete, continuous, brain, simulation)
├── viz/            # Visualization components (theme, arm_view, belief_panel)
├── scenarios/      # Experiment presets
├── tutorial/       # Educational content
└── app.py          # Streamlit entry point

tests/              # pytest test suite
docs/
├── references/     # Paper notation, original code map, tool survey
└── MATH-REFERENCE.md
```

## Build / Test / Run

```bash
# Run app
uv run streamlit run src/active_inference_viz/app.py

# Tests
uv run pytest --cov=src --cov-report=term-missing

# Lint & type check
uv run ruff check src/ tests/
uv run mypy src/
```

## Conventions

- **Immutable configs**: SimConfig is a frozen dataclass
- **Variable naming**: Follows paper notation where possible (see docs/references/paper-notation.md)
- **No PyTorch**: All gradients computed analytically (Jacobian for 3-joint arm)
- **Cache strategy**: `@st.cache_data` on `run_trial()` since SimConfig is hashable

## Status

- Phase 0: Setup (complete)
- Phase 1: Math model core (in progress)
- Phase 2: MVP visualization (planned)
