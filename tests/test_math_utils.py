"""Tests for mathematical utility functions."""

import numpy as np
import pytest
from scipy.optimize import approx_fprime

from active_inference_viz.model.math_utils import (
    acc_log_evidence,
    analytical_jacobian,
    bmc,
    denormalize,
    forward_kinematics,
    forward_kinematics_all,
    log_stable,
    norm_counts,
    norm_dist,
    normalize,
    softmax,
)


class TestNumericalPrimitives:
    def test_log_stable_positive(self) -> None:
        result = log_stable(np.array([1.0, 2.0, 0.5]))
        expected = np.log(np.array([1.0, 2.0, 0.5]) + 1e-16)
        np.testing.assert_allclose(result, expected)

    def test_log_stable_zero(self) -> None:
        result = log_stable(np.array([0.0]))
        assert np.isfinite(result[0])

    def test_softmax_uniform(self) -> None:
        result = softmax(np.array([1.0, 1.0, 1.0]))
        np.testing.assert_allclose(result, [1 / 3, 1 / 3, 1 / 3], atol=1e-10)

    def test_softmax_precision(self) -> None:
        dist = np.array([2.0, 1.0])
        low_prec = softmax(dist, precision=0.1)
        high_prec = softmax(dist, precision=10.0)
        # Higher precision → sharper distribution
        assert high_prec[0] > low_prec[0]

    def test_softmax_sums_to_one(self) -> None:
        result = softmax(np.array([3.0, 1.0, -2.0]), precision=2.0)
        np.testing.assert_allclose(result.sum(), 1.0, atol=1e-10)

    def test_norm_dist_1d(self) -> None:
        result = norm_dist(np.array([2.0, 3.0, 5.0]))
        np.testing.assert_allclose(result, [0.2, 0.3, 0.5])
        np.testing.assert_allclose(result.sum(), 1.0)

    def test_norm_counts(self) -> None:
        counts = np.array([[2.0, 3.0], [8.0, 7.0]])
        result = norm_counts(counts)
        np.testing.assert_allclose(result[:, 0].sum(), 1.0)
        np.testing.assert_allclose(result[:, 1].sum(), 1.0)


class TestNormalization:
    def test_normalize_denormalize_roundtrip(self) -> None:
        x = np.array([100.0, -50.0, 0.0])
        limits = (-200.0, 200.0)
        normed = normalize(x, limits)
        recovered = denormalize(normed, limits)
        np.testing.assert_allclose(recovered, x, atol=1e-10)

    def test_normalize_range(self) -> None:
        limits = (-180.0, 180.0)
        assert normalize(np.float64(-180.0), limits) == pytest.approx(-1.0)
        assert normalize(np.float64(180.0), limits) == pytest.approx(1.0)
        assert normalize(np.float64(0.0), limits) == pytest.approx(0.0)

    def test_normalize_no_range(self) -> None:
        result = normalize(np.float64(0.0), (-1.0, 1.0), rng=False)
        assert result == pytest.approx(0.5)


class TestForwardKinematics:
    @pytest.fixture
    def arm_params(self) -> dict:
        lengths = (250.0, 150.0, 50.0)
        norm_cart = (-450.0, 450.0)
        norm_polar = (-180.0, 180.0)
        lengths_norm = tuple(
            (2.0 * (l - norm_cart[0]) / (norm_cart[1] - norm_cart[0])) - 1.0
            for l in lengths
        )
        return {
            "lengths_norm": np.array(lengths_norm),
            "norm_polar": norm_polar,
            "norm_cart": norm_cart,
        }

    def test_straight_arm(self, arm_params: dict) -> None:
        """All angles = 0 → arm extends along x-axis."""
        angles_norm = normalize(np.array([0.0, 0.0, 0.0]), arm_params["norm_polar"])
        pos = forward_kinematics(
            angles_norm, arm_params["lengths_norm"], arm_params["norm_polar"],
        )
        # End-effector should be at x = sum(lengths_norm), y ≈ 0
        total_len = np.sum(arm_params["lengths_norm"])
        np.testing.assert_allclose(pos[0], total_len, atol=1e-10)
        np.testing.assert_allclose(pos[1], 0.0, atol=1e-10)

    def test_all_positions_chain(self, arm_params: dict) -> None:
        """forward_kinematics_all returns base + each joint."""
        angles_norm = normalize(np.array([35.0, 150.0, 0.0]), arm_params["norm_polar"])
        all_pos = forward_kinematics_all(
            angles_norm, arm_params["lengths_norm"], arm_params["norm_polar"],
        )
        assert all_pos.shape == (4, 2)
        np.testing.assert_allclose(all_pos[0], [0.0, 0.0])

    def test_fk_matches_all_fk(self, arm_params: dict) -> None:
        """forward_kinematics should match last position of forward_kinematics_all."""
        angles_norm = normalize(np.array([35.0, 150.0, 0.0]), arm_params["norm_polar"])
        ee = forward_kinematics(
            angles_norm, arm_params["lengths_norm"], arm_params["norm_polar"],
        )
        all_pos = forward_kinematics_all(
            angles_norm, arm_params["lengths_norm"], arm_params["norm_polar"],
        )
        np.testing.assert_allclose(ee, all_pos[-1], atol=1e-12)


class TestAnalyticalJacobian:
    @pytest.fixture
    def arm_params(self) -> dict:
        lengths = (250.0, 150.0, 50.0)
        norm_cart = (-450.0, 450.0)
        lengths_norm = tuple(
            (2.0 * (l - norm_cart[0]) / (norm_cart[1] - norm_cart[0])) - 1.0
            for l in lengths
        )
        return {
            "lengths_norm": np.array(lengths_norm),
            "norm_polar": (-180.0, 180.0),
        }

    def test_jacobian_vs_numerical(self, arm_params: dict) -> None:
        """Analytical Jacobian must match numerical (finite-difference) Jacobian."""
        angles_norm = normalize(np.array([35.0, 150.0, 0.0]), arm_params["norm_polar"])

        J_analytical = analytical_jacobian(
            angles_norm, arm_params["lengths_norm"], arm_params["norm_polar"],
        )

        # Numerical Jacobian via scipy
        def fk_wrapper(a: np.ndarray) -> np.ndarray:
            return forward_kinematics(a, arm_params["lengths_norm"], arm_params["norm_polar"])

        J_numerical = np.zeros_like(J_analytical)
        for i in range(2):
            def fk_i(a: np.ndarray, idx: int = i) -> float:
                return fk_wrapper(a)[idx]
            J_numerical[i] = approx_fprime(angles_norm, fk_i, 1e-7)

        np.testing.assert_allclose(J_analytical, J_numerical, atol=1e-5)

    def test_jacobian_multiple_configs(self, arm_params: dict) -> None:
        """Jacobian matches numerical for several angle configurations."""
        configs = [
            [0.0, 0.0, 0.0],
            [90.0, 0.0, 0.0],
            [45.0, -45.0, 30.0],
            [-90.0, 90.0, -90.0],
        ]
        for angles_deg in configs:
            angles_norm = normalize(np.array(angles_deg), arm_params["norm_polar"])
            J_a = analytical_jacobian(
                angles_norm, arm_params["lengths_norm"], arm_params["norm_polar"],
            )

            def fk_wrapper(a: np.ndarray) -> np.ndarray:
                return forward_kinematics(a, arm_params["lengths_norm"], arm_params["norm_polar"])

            J_n = np.zeros_like(J_a)
            for i in range(2):
                def fk_i(a: np.ndarray, idx: int = i) -> float:
                    return fk_wrapper(a)[idx]
                J_n[i] = approx_fprime(angles_norm, fk_i, 1e-7)

            np.testing.assert_allclose(
                J_a, J_n, atol=1e-5,
                err_msg=f"Jacobian mismatch for angles {angles_deg}",
            )

    def test_jacobian_shape(self, arm_params: dict) -> None:
        angles_norm = normalize(np.array([0.0, 0.0, 0.0]), arm_params["norm_polar"])
        J = analytical_jacobian(
            angles_norm, arm_params["lengths_norm"], arm_params["norm_polar"],
        )
        assert J.shape == (2, 3)


class TestBMC:
    def test_bmc_uniform_no_evidence(self) -> None:
        probs = np.array([1 / 3, 1 / 3, 1 / 3])
        log_evidence = np.zeros(3)
        result = bmc(probs, log_evidence, 1.0, 1.0, 5.0)
        np.testing.assert_allclose(result.sum(), 1.0, atol=1e-10)
        np.testing.assert_allclose(result, probs, atol=1e-5)

    def test_bmc_evidence_shifts_belief(self) -> None:
        probs = np.array([1 / 3, 1 / 3, 1 / 3])
        log_evidence = np.array([5.0, 0.0, 0.0])
        result = bmc(probs, log_evidence, 1.0, 1.0, 5.0)
        assert result[0] > result[1]
        assert result[0] > result[2]

    def test_bmc_sums_to_one(self) -> None:
        probs = np.array([0.2, 0.5, 0.3])
        log_evidence = np.array([1.0, -1.0, 0.0])
        result = bmc(probs, log_evidence, 1.2, 1.0, 5.0)
        np.testing.assert_allclose(result.sum(), 1.0, atol=1e-10)


class TestAccLogEvidence:
    def test_output_shape(self) -> None:
        eta = np.array([0.1, 0.2])
        Eta_m = np.array([[0.3, 0.1], [0.0, 0.4], [0.0, 0.0]])
        mu = np.array([0.15, 0.25])
        result = acc_log_evidence(eta, Eta_m, mu, 1.0, 1.0, 2.0)
        assert result.shape == (3,)

    def test_identical_predictions(self) -> None:
        """When all models predict the same, evidence should be similar."""
        eta = np.array([0.5, 0.5])
        Eta_m = np.array([[0.5, 0.5], [0.5, 0.5]])
        mu = np.array([0.5, 0.5])
        result = acc_log_evidence(eta, Eta_m, mu, 1.0, 1.0, 2.0)
        np.testing.assert_allclose(result[0], result[1], atol=1e-10)
