"""Structured result containers used by the discrete-time TU solver."""

from dataclasses import dataclass

import numpy as np


@dataclass
class ValueIterationResult:
    """Output from the effort-enabled value iteration."""

    values: np.ndarray
    effort_policy: np.ndarray
    meet: np.ndarray
    iterations: int
    error: float


@dataclass
class NoEffortValueResult:
    """Output from the no-effort counterfactual value iteration."""

    values: np.ndarray
    meet: np.ndarray
    iterations: int
    error: float


@dataclass
class SimulationResult:
    """Complete output from the outer fixed-point algorithm."""

    contributions: np.ndarray
    l_density: np.ndarray
    alpha: np.ndarray
    alpha_binary: np.ndarray
    alpha_no_effort_binary: np.ndarray
    payoffs: np.ndarray
    V: np.ndarray
    V_no_effort: np.ndarray
    deltaV: np.ndarray
    gain_upgrade: np.ndarray
    effort_policy: np.ndarray
    u: np.ndarray
    meet: np.ndarray
    pairs: float
    unmatched_individuals: float
    matched_individuals: float
    converged: bool
    cycle_detected: bool
