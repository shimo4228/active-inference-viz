# Existing Active Inference Tools — Survey & Differentiation

## Existing Tools

### pymdp (infer-actively/pymdp)
- **Scope**: Discrete state-space active inference (POMDP)
- **Limitation**: No continuous motor control, no hybrid discrete+continuous
- **Our differentiation**: We implement the full hybrid model including predictive coding for continuous reaching

### Active Inference Tutor
- **Scope**: Educational text/quiz format
- **Limitation**: No interactive simulation, no visualization of dynamics
- **Our differentiation**: Real-time parameter manipulation with instant visual feedback

### ActiveInferAnts (ActiveInferenceInstitute/ActiveInferAnts)
- **Scope**: Agent-based simulation framework
- **Limitation**: Generic framework, not focused on the specific embodied decision model
- **Our differentiation**: Focused deep dive into one well-published model with full mathematical transparency

### START Platform
- **Scope**: Bayesian inference teaching
- **Limitation**: General Bayesian tools, not active inference specific
- **Our differentiation**: Specific to active inference with predictive coding visualization

## Gap We Fill

No existing tool provides:
1. Interactive visualization of **hybrid discrete + continuous** active inference
2. Real-time manipulation of model parameters with instant re-simulation
3. Side-by-side comparison of different parameter settings
4. Educational decomposition of free energy into precision + complexity
5. Visualization of the full loop: cue → evidence → decision → reaching

## Target Audience

- Researchers learning active inference theory
- Students in computational neuroscience courses
- ActiveInference.org community members
- Readers of Priorelli et al. (2025) wanting interactive exploration
