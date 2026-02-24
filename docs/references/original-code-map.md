# Original Code Map — priorelli/embodied-decisions

Repository: https://github.com/priorelli/embodied-decisions

## File → Class → Method Mapping

### config.py
- Simulation parameters (all uppercase constants)
- Key params: N_CUES, T_CUE, DT, SIGMA_ETA, SIGMA_X, KAPPA, LAMBDA

### model/brain.py → Brain
- `__init__()`: Initialize discrete + continuous components
- `step()`: One timestep of combined inference
- Connects discrete decisions to continuous motor control

### model/inference.py → Discrete inference
- Evidence accumulation
- EFE (Expected Free Energy) computation
- Policy selection via softmax
- Likelihood learning (A-matrix update)

### model/unit.py → Unit (Predictive coding unit)
- `__init__()`: Generalized coordinates (mu, mu_dot)
- `step_prior()`: Update dynamics prediction error
- `step_likelihood()`: Update sensory prediction error (uses PyTorch autograd → we replace with analytical Jacobian)

### model/body.py → Body (Physical simulation)
- Forward kinematics: joint angles → hand position
- Pymunk physics integration (we replace with pure NumPy)

### simulation/run.py
- Trial loop: cue presentation → decision → reaching
- Data logging per timestep

## Key Config Parameters (Defaults)

| Parameter | Value | Description |
|---|---|---|
| N_TARGETS | 2 | Number of reach targets |
| N_JOINTS | 3 | Arm joints |
| L1, L2, L3 | 1.0, 0.7, 0.5 | Link lengths |
| DT | 0.01 | Integration timestep |
| T_CUE | 50 | Cue presentation steps |
| T_REACH | 200 | Reaching steps |
| SIGMA_ETA | 1.0 | Dynamics precision |
| SIGMA_X | 1.0 | Sensory precision |
| KAPPA | 1.0 | Gradient gain |
| LAMBDA | 10.0 | Decay rate |

## Original → This Project File Mapping

| Original | This Project | Notes |
|---|---|---|
| config.py | model/config.py | Frozen dataclass instead of global constants |
| model/brain.py | model/brain.py | Same structure, NumPy only |
| model/inference.py | model/discrete.py | Renamed for clarity |
| model/unit.py | model/continuous.py | Analytical Jacobian replaces autograd |
| model/body.py | model/math_utils.py | FK + Jacobian as pure functions |
| simulation/run.py | model/simulation.py | Returns SimResult dataclass |
