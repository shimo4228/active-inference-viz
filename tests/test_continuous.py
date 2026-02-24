"""Tests for continuous predictive coding module."""

import numpy as np

from active_inference_viz.model.config import SimConfig
from active_inference_viz.model.continuous import (
    Unit,
    make_dynamics_stay,
    make_dynamics_target,
)


class TestDynamicsFunctions:
    def test_stay_returns_zero(self) -> None:
        f, jac = make_dynamics_stay()
        x = np.array([0.5, -0.3])
        np.testing.assert_allclose(f(x), np.zeros(2))

    def test_stay_jacobian_zero(self) -> None:
        _, jac = make_dynamics_stay()
        J = jac(np.array([0.5, -0.3]))
        np.testing.assert_allclose(J, np.zeros((2, 2)))

    def test_target_at_target_returns_zero(self) -> None:
        target = np.array([0.5, 0.5])
        f, _ = make_dynamics_target(target, 0.5)
        np.testing.assert_allclose(f(target), np.zeros(2), atol=1e-15)

    def test_target_dynamics_direction(self) -> None:
        target = np.array([1.0, 0.0])
        f, _ = make_dynamics_target(target, 0.5)
        x = np.array([0.0, 0.0])
        result = f(x)
        # Should point toward target
        assert result[0] > 0.0

    def test_target_jacobian_negative_lambda_identity(self) -> None:
        target = np.array([0.5, 0.5])
        lam = 0.5
        _, jac = make_dynamics_target(target, lam)
        J = jac(np.array([0.0, 0.0]))
        expected = -lam * np.eye(2)
        np.testing.assert_allclose(J, expected)


class TestUnit:
    def _make_simple_unit(self) -> Unit:
        """Create a simple unit for testing."""
        f_stay, j_stay = make_dynamics_stay()
        return Unit(
            dim=(2, 3),
            pi_eta_x=1.0,
            pi_x=1.0,
            p_x=2.0,
            lr=1.0,
            F_m=[f_stay],
            F_m_jac=[j_stay],
            v=np.array([1.0]),
            L=np.zeros(1),
        )

    def test_initial_state_zero(self) -> None:
        unit = self._make_simple_unit()
        np.testing.assert_allclose(unit.x, np.zeros((2, 3)))

    def test_step_likelihood_prior(self) -> None:
        unit = self._make_simple_unit()
        unit.x[0] = np.array([0.5, 0.3, 0.1])
        prior = np.array([0.0, 0.0, 0.0])
        unit.step_likelihood_prior(prior)
        # Error = (x - prior) * pi = 0.5 * 1.0
        np.testing.assert_allclose(
            unit.eps_eta_x[0], [0.5, 0.3, 0.1],
        )

    def test_step_dynamics(self) -> None:
        unit = self._make_simple_unit()
        unit.x[0] = np.array([0.1, 0.2, 0.3])
        unit.x[1] = np.array([0.0, 0.0, 0.0])
        unit.step_dynamics()
        # With f_stay, pred_x = 0, so eps_x = (x[1] - 0) * pi_x = 0
        np.testing.assert_allclose(unit.eps_x, np.zeros(3))

    def test_update_moves_belief(self) -> None:
        unit = self._make_simple_unit()
        unit.x[0] = np.array([0.5, 0.3, 0.1])
        unit.x[1] = np.array([0.0, 0.0, 0.0])
        prior = np.zeros(3)
        unit.step_likelihood_prior(prior)
        unit.step_dynamics()
        x_before = unit.x.copy()
        unit.update(dt=0.3)
        # State should have changed
        assert not np.allclose(unit.x, x_before)


class TestUnitWithTargetDynamics:
    def test_ext_unit_dynamics_evidence(self) -> None:
        """External unit with target dynamics should accumulate evidence."""
        target = np.array([0.5, 0.5])
        f_t1, j_t1 = make_dynamics_target(target, 0.5)
        f_stay, j_stay = make_dynamics_stay()

        L = np.zeros(2)
        v = np.array([0.5, 0.5])  # Equal belief in both models
        unit = Unit(
            dim=(2, 2),
            pi_eta_x=0.45,
            pi_x=1.0,
            p_x=2.0,
            lr=1.0,
            F_m=[f_t1, f_stay],
            F_m_jac=[j_t1, j_stay],
            v=v,
            L=L,
        )
        unit.x[0] = np.array([0.0, 0.0])
        unit.x[1] = np.array([0.1, 0.1])

        unit.step_dynamics()
        # L should have been modified (evidence accumulated)
        assert not np.allclose(L, np.zeros(2))
