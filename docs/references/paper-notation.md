# Paper Notation → Code Variable Mapping

Reference: Priorelli et al. (2025) "Embodied decisions as active inference", PLOS Computational Biology.

## Core Variables

| Paper Symbol | Code Variable | Description |
|---|---|---|
| s | `s` | Hidden discrete state (target identity) |
| o | `obs` | Discrete observation (sensory cue) |
| π | `pi` / `policy` | Discrete policy |
| A | `A` | Likelihood matrix (observation model) |
| B | `B` | Transition matrix |
| C | `C` | Preference (log prior over observations) |
| D | `D` | Prior over initial states |
| E | `E` | Prior over policies (habit) |
| γ | `gamma` | Policy precision (inverse temperature) |
| G | `G` | Expected free energy |
| F | `F` | Variational free energy |

## Continuous Variables (Predictive Coding)

| Paper Symbol | Code Variable | Description |
|---|---|---|
| μ | `mu` | Belief about continuous state (generalized coords) |
| μ' | `mu_dot` | Velocity belief |
| η | `eta` | Prior attractor (target position) |
| x | `x` | Sensory state (hand position) |
| φ | `phi` | Joint angles |
| v | `v` | Joint velocities |
| ε_η | `eps_eta` | Prediction error (dynamics) |
| ε_x | `eps_x` | Prediction error (sensory) |
| Σ_η | `sigma_eta` | Dynamics precision (inverse variance) |
| Σ_x | `sigma_x` | Sensory precision |
| κ | `kappa` | Learning rate / gain |
| λ | `lam` | Decay rate for dynamics |
| J | `J` | Jacobian of forward kinematics |

## Four Core Processes

### 1. Evidence Accumulation (Discrete)
- Cue observations → update posterior over targets
- `P(s|o) ∝ A[o,:] * D` (Bayesian update)

### 2. State Inference (Discrete)
- EFE computation → policy selection → action
- `G(π) = Σ_τ [H[P(o|s)]·Q(s|π) + D_KL[Q(s|π)||P(s)]]`

### 3. Predictive Coding (Continuous)
- Belief update via gradient descent on free energy
- `μ̇ = -κ (ε_η / Σ_η + J^T ε_x / Σ_x)`

### 4. Motor Control (Continuous)
- Action = movement toward believed state
- `a = μ' (velocity belief drives action)`
