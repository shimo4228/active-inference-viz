"""Discrete inference: evidence accumulation, state inference, EFE, learning.

Reimplements simulation/discrete.py from priorelli/embodied-decisions using
pure NumPy. The Discrete class maintains beliefs over two target states (T1, T2)
and performs Bayesian model comparison at each cue presentation.
"""

from __future__ import annotations

import itertools

import numpy as np
from numpy.typing import NDArray

from active_inference_viz.model.config import SimConfig
from active_inference_viz.model.math_utils import (
    bmc,
    log_stable,
    norm_counts,
    norm_dist,
    softmax,
)


class Discrete:
    """Discrete active inference over target states."""

    def __init__(self, cfg: SimConfig) -> None:
        self.cfg = cfg
        self.n_states = 2  # T1, T2

        # --- Likelihood matrices ---
        self.a_cue = np.array([[2.0, 1.8], [1.8, 2.0]])
        self.A_cue = self._get_A_cue()
        self.coincidence_cue = np.zeros_like(self.a_cue)

        self.a_ext = np.array([[0.6, 0.0], [0.0, 0.6], [0.4, 0.4]])
        self.A_ext = self._get_A_ext()
        self.coincidence_ext = np.zeros_like(self.a_ext)

        # --- Transition matrix (STAY only) ---
        self.B = np.zeros((self.n_states, self.n_states, 1))
        self.B[:, :, 0] = np.eye(self.n_states)

        # --- Preference (uniform) ---
        self.C = np.ones(self.n_states) / self.n_states

        # --- Prior ---
        self.d = np.full(self.n_states, 0.5)
        self.prior = softmax(self.d)

        # --- Policies ---
        n_actions = 1  # STAY only
        x = [n_actions] * cfg.n_policy
        self.policies = [
            np.array(p).reshape(cfg.n_policy, 1)
            for p in itertools.product(*[list(range(i)) for i in x])
        ]
        self.E = np.zeros(len(self.policies))

        # --- Entropy ---
        self.H_A = -(self.A_ext * log_stable(self.A_ext)).sum(axis=0)

        # --- External observations & evidence ---
        self.o_ext = np.array([0.0, 0.0, 1.0])  # Start: "stay" belief
        self.L_ext = np.zeros(3)

    # --- Likelihood construction ---

    def _get_A_cue(self) -> NDArray[np.float64]:
        ac = self.cfg.alpha_c
        return np.array([[ac, 1 - ac], [1 - ac, ac]])

    def _get_A_ext(self) -> NDArray[np.float64]:
        ah = self.cfg.alpha_h
        return np.array([[ah, 0.0], [0.0, ah], [1 - ah, 1 - ah]])

    # --- Inference ---

    def infer_states(
        self, cue: NDArray[np.float64], r_ext: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Infer current states from cue observation and extrinsic evidence."""
        qs_cue = self.A_cue.T @ cue
        qs_ext = self.A_ext.T @ r_ext

        log_prior = log_stable(self.prior)
        log_post_cue = log_stable(qs_cue)
        log_post_ext = log_stable(qs_ext) * self.cfg.k_h

        return softmax(log_prior + log_post_cue + log_post_ext, self.cfg.w_c)

    def compute_G(self, qs_current: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute expected free energy for each policy."""
        G = np.zeros(len(self.policies))

        for pid, policy in enumerate(self.policies):
            qs_pi_t = qs_current
            for t in range(policy.shape[0]):
                action = policy[t, 0]
                qs_pi_t = self.B[:, :, action] @ (qs_current if t == 0 else qs_pi_t)
                # KL divergence
                G[pid] += (log_stable(qs_pi_t) - log_stable(self.C)) @ qs_pi_t

        return G

    # --- Step ---

    def step(self, cue: NDArray[np.float64]) -> None:
        """One discrete inference step (called every n_tau continuous steps)."""
        # Bayesian model comparison
        self.o_ext[:] = bmc(
            self.o_ext, self.L_ext, self.cfg.w_h,
            self.cfg.gain_prior, self.cfg.gain_evidence,
        )

        # Infer current state
        qs_current = self.infer_states(cue, self.o_ext)

        # Accumulate coincidences for learning
        self.coincidence_cue += np.outer(cue, qs_current)
        self.coincidence_ext += np.outer(self.o_ext, qs_current)

        # Expected free energy → policy selection
        G = self.compute_G(qs_current)
        Q_pi = softmax(self.E - G)

        # Action posterior → next state prediction
        P_u = np.zeros(1)
        for pid, policy in enumerate(self.policies):
            P_u[int(policy[0, 0])] += Q_pi[pid]
        P_u = norm_dist(P_u)

        # Predict next state
        qs_next = np.zeros(self.n_states)
        for action_idx, prob in enumerate(P_u):
            qs_next += prob * self.B[:, :, action_idx] @ qs_current
        self.prior = qs_next

        # Reset evidence and recompute expected observations
        self.L_ext[:] = 0.0
        self.o_ext[:] = bmc(
            self.A_ext @ self.prior, self.L_ext, self.cfg.w_h,
            self.cfg.gain_prior, self.cfg.gain_evidence,
        )

    # --- Learning ---

    def learn_likelihood(self) -> None:
        """Update likelihood matrices from accumulated coincidences."""
        self.a_cue = (
            self.cfg.omega_a_cue * self.a_cue
            + self.cfg.eta_a_cue * self.coincidence_cue
        )
        self.coincidence_cue[:] = 0.0
        self.A_cue = norm_counts(self.a_cue)

        self.a_ext = (
            self.cfg.omega_a_ext * self.a_ext
            + self.cfg.eta_a_ext * self.coincidence_ext
        )
        self.coincidence_ext[:] = 0.0
        self.A_ext = norm_counts(self.a_ext)

    def learn_prior(self) -> None:
        """Update prior via Dirichlet concentration smoothing."""
        self.d = self.cfg.omega_d * self.d + self.cfg.eta_d * self.prior
        self.prior = softmax(self.d)

    # --- Reset ---

    def reset(self) -> None:
        """Reset beliefs for a new trial (keep learned parameters)."""
        self.prior = softmax(self.d)
        self.o_ext[:] = [0.0, 0.0, 1.0]
        self.L_ext[:] = 0.0
