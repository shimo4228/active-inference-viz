Language: English | [日本語](README.ja.md)

# Active Inference Visualizer

Interactive Streamlit dashboard for visualizing embodied decision-making through active inference, based on [Priorelli et al. (2025)](https://doi.org/10.1016/j.neunet.2025.107249). This project provides a pure NumPy reimplementation of the original PyTorch code, making the mathematical foundations transparent and accessible.

## Features

- **Interactive simulation** of active inference agents performing reaching tasks
- **Real-time visualization** of belief dynamics, free energy, and arm kinematics
- **Pure NumPy implementation** -- all gradients computed analytically (no PyTorch dependency)
- **Educational walkthroughs** explaining the math behind active inference
- **Experiment presets** for exploring different parameter configurations

## Tech Stack

| Category | Tool |
|----------|------|
| Language | Python 3.13+ |
| Package Manager | [uv](https://docs.astral.sh/uv/) + pyproject.toml |
| Web Framework | Streamlit |
| Computation | NumPy, SciPy |
| Visualization | Plotly (interactive), Matplotlib (static export) |
| Linting | Ruff, mypy (strict mode) |
| Testing | pytest + pytest-cov |

## Installation

```bash
# Clone the repository
git clone https://github.com/shimo4228/active-inference-viz.git
cd active-inference-viz

# Install dependencies (requires uv)
uv sync
```

## Usage

```bash
# Launch the dashboard
uv run streamlit run src/active_inference_viz/app.py

# Run tests
uv run pytest --cov=src --cov-report=term-missing

# Lint & type check
uv run ruff check src/ tests/
uv run mypy src/
```

## Project Structure

```
src/active_inference_viz/
├── model/          # Mathematical core
│   ├── config.py       # SimConfig (frozen dataclass)
│   ├── math_utils.py   # Linear algebra helpers
│   ├── discrete.py     # Discrete state inference
│   ├── continuous.py   # Continuous state inference
│   ├── brain.py        # Agent brain (belief updating)
│   └── simulation.py   # Trial runner
├── viz/            # Visualization components
│   ├── theme.py        # Color scheme & styling
│   ├── arm_view.py     # 3-joint arm rendering
│   └── belief_panel.py # Belief distribution plots
├── scenarios/      # Experiment presets
├── tutorial/       # Educational content
└── app.py          # Streamlit entry point

tests/              # pytest test suite
docs/
├── references/     # Paper notation, original code map
└── MATH-REFERENCE.md
```

## Status

| Phase | Description | Status |
|-------|-------------|--------|
| Phase 0 | Project setup | Complete |
| Phase 1 | Math model core | In progress |
| Phase 2 | MVP visualization | Planned |

## License

MIT
