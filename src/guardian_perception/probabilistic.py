"""Small, inspectable Gaussian state-estimation primitives for fusion experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GaussianState:
    """Constant-velocity state [forward_m, lateral_m, vx_mps, vy_mps]."""

    mean: np.ndarray
    covariance: np.ndarray


def predict_constant_velocity(state: GaussianState, dt_s: float, *, acceleration_std_mps2: float = 3.0) -> GaussianState:
    if dt_s <= 0:
        raise ValueError("dt_s must be positive")
    transition = np.array([[1, 0, dt_s, 0], [0, 1, 0, dt_s], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float)
    q = acceleration_std_mps2 ** 2
    process = q * np.array(
        [[dt_s**4 / 4, 0, dt_s**3 / 2, 0], [0, dt_s**4 / 4, 0, dt_s**3 / 2], [dt_s**3 / 2, 0, dt_s**2, 0], [0, dt_s**3 / 2, 0, dt_s**2]],
        dtype=float,
    )
    return GaussianState(transition @ state.mean, transition @ state.covariance @ transition.T + process)


def linear_update(state: GaussianState, measurement: np.ndarray, observation: np.ndarray, measurement_covariance: np.ndarray) -> GaussianState:
    """Joseph-form linear Gaussian update to keep covariance numerically stable."""

    innovation = measurement - observation @ state.mean
    innovation_covariance = observation @ state.covariance @ observation.T + measurement_covariance
    gain = state.covariance @ observation.T @ np.linalg.inv(innovation_covariance)
    identity = np.eye(len(state.mean))
    covariance = (identity - gain @ observation) @ state.covariance @ (identity - gain @ observation).T + gain @ measurement_covariance @ gain.T
    return GaussianState(state.mean + gain @ innovation, covariance)


def squared_mahalanobis(state: GaussianState, measurement: np.ndarray, observation: np.ndarray, measurement_covariance: np.ndarray) -> float:
    innovation = measurement - observation @ state.mean
    innovation_covariance = observation @ state.covariance @ observation.T + measurement_covariance
    return float(innovation.T @ np.linalg.solve(innovation_covariance, innovation))


def pdaf_weights(squared_distances: list[float], *, missed_detection_weight: float = 0.35) -> tuple[np.ndarray, float]:
    """Return normalized candidate association weights and a missed-detection weight."""

    if missed_detection_weight <= 0:
        raise ValueError("missed_detection_weight must be positive")
    likelihood = np.exp(-0.5 * np.asarray(squared_distances, dtype=float))
    normalizer = float(likelihood.sum() + missed_detection_weight)
    return likelihood / normalizer, missed_detection_weight / normalizer


def pdaf_position_update(
    state: GaussianState,
    measurements: list[np.ndarray],
    weights: np.ndarray,
    measurement_covariance: np.ndarray,
) -> GaussianState:
    """Moment-match weighted 2D position measurements, including association spread."""

    if not measurements:
        return state
    total_weight = float(weights.sum())
    if total_weight <= 0:
        return state
    normalized = weights / total_weight
    stacked = np.stack(measurements)
    mean_measurement = normalized @ stacked
    spread = sum(weight * np.outer(value - mean_measurement, value - mean_measurement) for value, weight in zip(stacked, normalized, strict=True))
    observation = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)
    # A diffuse association must have less influence than one confident radar return.
    effective_covariance = (measurement_covariance + spread) / total_weight
    return linear_update(state, mean_measurement, observation, effective_covariance)
