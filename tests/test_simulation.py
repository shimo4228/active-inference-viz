"""Tests for simulation runner."""

import numpy as np

from active_inference_viz.model.config import SimConfig
from active_inference_viz.model.simulation import run_trial


class TestRunTrial:
    def test_returns_sim_result(self) -> None:
        cfg = SimConfig(n_cues=3, n_wait=2, n_tau=10)
        result = run_trial(cfg)
        n_steps = cfg.n_steps
        assert result.angles.shape == (n_steps, cfg.n_joints)
        assert result.states.shape == (n_steps, 2)
        assert result.causes.shape == (n_steps, 3)
        assert result.pos.shape == (n_steps, cfg.n_joints + 1, 2)

    def test_beliefs_valid(self) -> None:
        cfg = SimConfig(n_cues=3, n_wait=2, n_tau=10)
        result = run_trial(cfg)
        # Beliefs should be valid probability distributions
        for step in range(cfg.n_steps):
            np.testing.assert_allclose(
                result.states[step].sum(), 1.0, atol=1e-5,
                err_msg=f"states at step {step} don't sum to 1",
            )
            assert np.all(result.states[step] >= -1e-10), \
                f"Negative belief at step {step}"

    def test_causes_valid(self) -> None:
        cfg = SimConfig(n_cues=3, n_wait=2, n_tau=10)
        result = run_trial(cfg)
        for step in range(cfg.n_steps):
            np.testing.assert_allclose(
                result.causes[step].sum(), 1.0, atol=1e-5,
                err_msg=f"causes at step {step} don't sum to 1",
            )

    def test_cues_logged(self) -> None:
        cfg = SimConfig(n_cues=3, n_wait=2, n_tau=10, cue_sequence=(0, 1, 0))
        result = run_trial(cfg)
        assert result.cues.shape == (3, 2)
        # First cue should be for T1
        np.testing.assert_allclose(result.cues[0], [1.0, 0.0])
        # Second cue should be for T2
        np.testing.assert_allclose(result.cues[1], [0.0, 1.0])

    def test_t1_dominant_cues_shift_belief(self) -> None:
        """Predominantly T1 cues should result in T1-biased final belief."""
        cfg = SimConfig(
            n_cues=5, n_wait=2, n_tau=10,
            cue_sequence=(0, 0, 0, 0, 0),
            gain_evidence=5.0,
        )
        result = run_trial(cfg)
        # After accumulating T1 evidence, final state should favor T1
        final_state = result.states[-1]
        assert final_state[0] > final_state[1], \
            f"Expected T1 > T2 but got {final_state}"

    def test_config_preserved(self) -> None:
        cfg = SimConfig(n_cues=3, n_wait=2, n_tau=10)
        result = run_trial(cfg)
        assert result.config is cfg


class TestSimConfigHashable:
    def test_frozen_dataclass_hashable(self) -> None:
        cfg = SimConfig()
        h = hash(cfg)
        assert isinstance(h, int)

    def test_different_configs_different_hash(self) -> None:
        cfg1 = SimConfig(n_cues=5)
        cfg2 = SimConfig(n_cues=10)
        assert hash(cfg1) != hash(cfg2)
