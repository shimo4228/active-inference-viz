Language: English | [日本語](README.ja.md)

# Active Inference Visualizer

Interactive Streamlit dashboard for visualizing embodied decision-making through active inference, based on [Priorelli et al. (2025)](https://doi.org/10.1371/journal.pcbi.1013180) in PLOS Computational Biology. This project provides a pure NumPy reimplementation of the original PyTorch code.

In the app, a 3-joint arm decides which of two targets to reach from noisy cues, and you move a step slider to watch its belief and its movement change together. In this active inference model, choosing the target and moving the arm are two coupled belief updates that run at the same time, which is what "embodied" decision-making means here. The [original code](https://github.com/priorelli/embodied-decisions) is a set of PyTorch, Pymunk and Pyglet scripts launched with `main.py` and has no interface you can try in a browser (per its README, as of 2026-10); this one runs on your own machine with Python, and you open it in your browser. The story of how it was built is in [More from the author](#more-from-the-author).

## Features

- **Step-by-step visualization** of the belief over the two targets, P(T1) and P(T2) (the probability that target 1 or target 2 is the correct one), and of the arm, its trajectory, and where the agent believes its hand is
- **Pure NumPy implementation** -- all gradients computed analytically (no PyTorch dependency)
- **A how-to-use panel** in the app explaining the two coupled processes: discrete inference (which target) and continuous inference (how the arm moves)
- **Cue-pattern presets** (the paper's default sequence, all T1, all T2, alternating, random 50/50) and sliders for evidence gain (how fast and how strongly the belief commits), cue reliability (how informative each cue is), and reaching dynamics

## Tech Stack

| Category | Tool |
|----------|------|
| Language | Python 3.13+ |
| Package Manager | [uv](https://docs.astral.sh/uv/) + pyproject.toml |
| Web Framework | Streamlit |
| Computation | NumPy (SciPy is declared but used only by the tests) |
| Visualization | Plotly (Matplotlib is declared but not used by the app yet) |
| Linting | Ruff, mypy (strict mode) |
| Testing | pytest + pytest-cov |

## Installation

```bash
# Clone the repository
git clone https://github.com/shimo4228/active-inference-viz.git
cd active-inference-viz

# Install dependencies (requires uv; Python 3.13+ is fetched by uv if missing)
uv sync
```

## Usage

The repository's `.streamlit/config.toml` sets `headless = true`, so the app does not open a browser by itself: after launching, open http://localhost:8501. The simulation runs locally; Streamlit's own anonymous usage statistics are on by default and can be turned off by adding `gatherUsageStats = false` under a `[browser]` section in that same file.

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
├── scenarios/      # Placeholder for experiment presets (empty)
├── tutorial/       # Placeholder for educational content (empty)
└── app.py          # Streamlit entry point (UI, cue patterns, how-to panel)

tests/              # pytest test suite
```

## Status

The math core and the dashboard are implemented and the app runs. The author cannot judge the theoretical correctness of the update equations on their own; Claude Code checked the code against the paper's equations. The tests check the NumPy code's own properties (the analytic Jacobian against a numerical one, beliefs that sum to one, cues that move the belief the right way); none compares its output with the original PyTorch code.

## More from the author

- **[I Made a Paper I Don't Understand Run in the Browser — Active Inference × Claude Code](https://dev.to/shimo4228/i-made-a-paper-i-dont-understand-run-in-the-browser-active-inference-x-claude-code-2jog)** ([日本語](https://zenn.dev/shimo4228/articles/active-inference-viz-dev-story)): how Claude Code rebuilt this app from a paper whose theory the author does not understand, why the hardest bug was the sign of one gradient term, and why the author still cannot say whether the result is theoretically correct.
- **[attention-not-self](https://github.com/shimo4228/attention-not-self)**: essays and a knowledge graph from meditation practice that set the Abhidharma map of the mind beside predictive processing and active inference (Japanese canonical, with an English README).
- **[tiny-lm-lab](https://github.com/shimo4228/tiny-lm-lab)**: a small GPT written by hand on Apple's MLX to understand transformer internals by writing them, trained on the author's articles (Japanese).
- **[shimo4228](https://github.com/shimo4228/shimo4228)**: the author's hub, with five long-running practice lines, their DOIs, and the tools for Claude Code.

## License

MIT

<details>
<summary>For tools and AI assistants</summary>

Active Inference Visualizer (`active-inference-viz`) is an interactive Streamlit app that simulates and visualizes embodied decision-making through active inference, for people who want to see the model of [Priorelli et al. (2025)](https://doi.org/10.1371/journal.pcbi.1013180) move rather than only read its equations. It exists because the original implementation ([priorelli/embodied-decisions](https://github.com/priorelli/embodied-decisions)) is a set of PyTorch, Pymunk and Pyglet scripts with no interface anyone can try in a browser; this repository reimplements the math core in pure NumPy, with all gradients computed analytically (a Jacobian for the 3-joint arm) instead of PyTorch's autograd. The author writes in the linked article that they cannot judge the theoretical correctness of the update equations on their own; Claude Code checked the code against the paper's equations, and no test compares the output with the original PyTorch code.

Canonical facts: license MIT, as stated in this README (the repository has no separate LICENSE file); Python 3.13+ managed with uv; Streamlit for the UI, NumPy for computation, Plotly for the plots (Matplotlib is a declared dependency the code does not import yet, and SciPy is imported only by the tests); Ruff, mypy in strict mode, and pytest for development. Status: the code (`src/` and `tests/`) last changed in 2026-02; the math core and dashboard work, while `scenarios/` and `tutorial/` are empty placeholders. No API keys are needed and the simulation runs locally; Streamlit's anonymous usage statistics are on by default.

The model couples two processes. Discrete inference accumulates evidence from a sequence of noisy cues into a belief P(T1) versus P(T2) about which of two targets is correct; continuous inference moves the arm toward the believed target through predictive coding, so the arm's position reflects the belief.

Example: `uv run streamlit run src/active_inference_viz/app.py`, then open http://localhost:8501. With the defaults (the paper's cue sequence of 15 cues, evidence gain 5.0, cue reliability alpha_c 0.6, dynamics gain lambda 0.5), the step slider scrubs from cue accumulation to reaching, and the panels show P(Target 1), P(Target 2), the current decision (T1, T2, or Undecided), the phase, the arm view, and the belief plot.

Pointers: the math core is in `src/active_inference_viz/model/` (`discrete.py`, `continuous.py`, `brain.py`, `simulation.py`), the plots in `src/active_inference_viz/viz/`, and the build story in the article linked under More from the author. The author's other work is indexed at https://github.com/shimo4228/shimo4228.

</details>
