"""Tests for discrete inference module."""

import numpy as np

from active_inference_viz.model.config import SimConfig
from active_inference_viz.model.discrete import Discrete


class TestDiscreteInit:
    def test_default_config(self) -> None:
        cfg = SimConfig()
        d = Discrete(cfg)
        assert d.n_states == 2
        assert d.A_cue.shape == (2, 2)
        assert d.A_ext.shape == (3, 2)

    def test_prior_sums_to_one(self) -> None:
        d = Discrete(SimConfig())
        np.testing.assert_allclose(d.prior.sum(), 1.0, atol=1e-10)

    def test_o_ext_sums_to_one(self) -> None:
        d = Discrete(SimConfig())
        np.testing.assert_allclose(d.o_ext.sum(), 1.0, atol=1e-10)

    def test_A_cue_columns_sum_to_one(self) -> None:
        d = Discrete(SimConfig())
        for col in range(d.A_cue.shape[1]):
            np.testing.assert_allclose(d.A_cue[:, col].sum(), 1.0, atol=1e-10)

    def test_A_ext_columns_sum_to_one(self) -> None:
        d = Discrete(SimConfig())
        for col in range(d.A_ext.shape[1]):
            np.testing.assert_allclose(d.A_ext[:, col].sum(), 1.0, atol=1e-10)


class TestInferStates:
    def test_cue_for_t1_shifts_belief(self) -> None:
        d = Discrete(SimConfig())
        cue_t1 = np.array([1.0, 0.0])
        r_ext = np.array([0.0, 0.0, 1.0])  # Neutral
        qs = d.infer_states(cue_t1, r_ext)
        np.testing.assert_allclose(qs.sum(), 1.0, atol=1e-10)
        assert qs[0] > qs[1], "Cue for T1 should increase P(T1)"

    def test_cue_for_t2_shifts_belief(self) -> None:
        d = Discrete(SimConfig())
        cue_t2 = np.array([0.0, 1.0])
        r_ext = np.array([0.0, 0.0, 1.0])
        qs = d.infer_states(cue_t2, r_ext)
        assert qs[1] > qs[0], "Cue for T2 should increase P(T2)"

    def test_neutral_cue_balanced(self) -> None:
        d = Discrete(SimConfig())
        neutral = np.array([0.5, 0.5])
        r_ext = np.array([0.0, 0.0, 1.0])
        qs = d.infer_states(neutral, r_ext)
        np.testing.assert_allclose(qs[0], qs[1], atol=0.05)


class TestDiscreteStep:
    def test_step_maintains_valid_beliefs(self) -> None:
        d = Discrete(SimConfig())
        cue = np.array([1.0, 0.0])
        d.step(cue)
        np.testing.assert_allclose(d.prior.sum(), 1.0, atol=1e-10)
        np.testing.assert_allclose(d.o_ext.sum(), 1.0, atol=1e-10)
        assert np.all(d.prior >= 0), "Beliefs must be non-negative"
        assert np.all(d.o_ext >= 0), "o_ext must be non-negative"

    def test_repeated_t1_cues_increase_t1_belief(self) -> None:
        d = Discrete(SimConfig())
        cue_t1 = np.array([1.0, 0.0])
        for _ in range(5):
            d.step(cue_t1)
        assert d.prior[0] > d.prior[1], "Repeated T1 cues should favor T1"


class TestDiscreteReset:
    def test_reset_restores_initial_state(self) -> None:
        d = Discrete(SimConfig())
        initial_o_ext = d.o_ext.copy()
        # Run some steps
        d.step(np.array([1.0, 0.0]))
        d.step(np.array([0.0, 1.0]))
        # Reset
        d.reset()
        np.testing.assert_allclose(d.o_ext, initial_o_ext)
        np.testing.assert_allclose(d.L_ext, np.zeros(3))
