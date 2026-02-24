---
name: pytorch-backward-vjp-replacement
description: "PyTorch eps.backward(eps) を手動 Jacobian に置換する際の -precision 因子。欠落すると数値発散する"
user-invocable: false
origin: auto-extracted
---

# PyTorch backward(v) → Manual VJP: The -precision Trap

**Extracted:** 2026-02-25
**Context:** PyTorch autograd を使った予測符号化/変分推論コードを NumPy に再実装するとき

## Problem

PyTorch の `tensor.backward(gradient_vector)` は Vector-Jacobian Product (VJP) を計算するが、
予測符号化コードでよく使われる `eps.backward(eps)` パターンには、直感に反する
**-precision 因子**が含まれる。これを見落とすと勾配の符号が反転し、
gradient descent が gradient ASCENT になり数値が発散する。

## The Pattern in Original PyTorch Code

```python
# Prediction error (includes precision scaling)
eps = (x - prediction) * precision

# Backpropagate — this is NOT just J^T @ eps!
eps.backward(eps)
parent.grad += leaf.grad
```

## What backward(eps) Actually Computes

`tensor.backward(v)` computes: `v^T @ (d(tensor)/d(leaf))`

For `eps = (x - g(parent)) * π`:
- `d(eps_j)/d(parent_k) = -π * J_g[j,k]`
- VJP = `eps^T @ (-π * J_g) = -π * J_g^T @ eps`

**The result is: `-precision * J^T @ eps`, NOT `+J^T @ eps`**

## Solution: Manual Replacement

```python
# WRONG — causes divergent oscillations
parent.grad_o += J.T @ eps

# CORRECT — matches PyTorch backward behavior
parent.grad_o += -precision * (J.T @ eps)
```

## Three Cases to Check

### 1. Likelihood backprop (child → parent via prediction function g)
```python
eps_eta = (child.x - g(parent.x)) * pi_eta
# VJP: -pi_eta * J_g^T @ eps_eta
parent.grad += -pi_eta * (J_g.T @ eps_eta)
```

### 2. Observation backprop (obs → parent via identity or simple g)
```python
eps_o = (observation - g(parent.x)) * pi_o
# For identity g: J = I, so VJP = -pi_o * eps_o
parent.grad += -pi_o * eps_o
```

### 3. Dynamics backprop (within same unit)
```python
eps_x = (x_vel - weighted_pred) * pi_x
# VJP: -pi_x * J_dynamics^T @ eps_x
grad_x = -pi_x * (J_dynamics.T @ eps_x)
```

## Diagnostic: How to Detect This Bug

- **Symptom**: Beliefs/states diverge to ±infinity within 50-100 steps
- **Pattern**: Values look reasonable for the first few steps, then explode exponentially
- **Root cause**: Gradient points UPHILL instead of downhill on the free energy landscape
- **Quick check**: Print the ext unit's position belief at steps 0, 30, 60 — if it grows unboundedly, VJP signs are wrong

## When to Use

- Replacing PyTorch autograd with NumPy analytical Jacobians
- Reimplementing predictive coding / active inference models
- Any variational inference code where `tensor.backward(tensor)` is used
- Deploying PyTorch research code to environments without PyTorch (Streamlit Cloud, embedded, etc.)
