"""Core object-oriented solver for the discrete-time TU Shimer-Smith variant.

This module contains the full numerical algorithm and no plotting code.
"""

from __future__ import annotations

import math
import sys
from typing import Dict, Tuple

import numpy as np

from .config import ProductionFunction, TUDiscreteConfig
from .results import NoEffortValueResult, SimulationResult, ValueIterationResult


class TUDiscreteModel:
    """Solve the discrete-time transferable-utility matching model.

    The class is state-light and returns a full ``SimulationResult`` from ``run``.
    Plotting is intentionally kept in a separate module.
    """

    def __init__(self, config: TUDiscreteConfig, production_function: ProductionFunction):
        self.config = config
        self.production_function = production_function
        self._validate_config()

    def _validate_config(self) -> None:
        """Validate basic parameter restrictions before solving."""
        g = self.config.grid
        m = self.config.model
        s = self.config.solver

        if g.n <= 1:
            raise ValueError("n must be greater than 1")
        if g.distribution not in {"uniform", "normal"}:
            raise ValueError("distribution must be 'uniform' or 'normal'")
        if g.distribution == "normal" and g.sigma <= 0:
            raise ValueError("sigma must be strictly positive for normal distribution")

        if not (0.0 < m.beta < 1.0):
            raise ValueError("beta must be in (0,1)")
        if m.delta <= 0:
            raise ValueError("delta must be strictly positive")
        if m.rho < 0:
            raise ValueError("rho must be non-negative")
        if m.phi < 0:
            raise ValueError("phi must be non-negative")
        if m.eta < 0:
            raise ValueError("eta must be non-negative")
        if m.upgrade_step < 1:
            raise ValueError("upgrade_step must be >= 1")

        if not (0.0 < s.alpha_relax <= 1.0):
            raise ValueError("alpha_relax must be in (0,1]")
        if not (0.0 < s.damp_u <= 1.0):
            raise ValueError("damp_u must be in (0,1]")
        if s.v_max_iter < 1 or s.outer_max_iter < 1 or s.u_max_iter < 1:
            raise ValueError("iteration limits must be >= 1")

        denom = 1.0 - m.beta * (1.0 - m.delta)
        if denom <= 0:
            raise ValueError("Need 1 - beta*(1-delta) > 0 for finite match values")

    def compute_type_grid(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return midpoint contributions grid and associated type density."""
        g = self.config.grid
        n = g.n
        contributions = np.linspace(1 / n / 2, 1 - 1 / n / 2, n)

        if g.distribution == "uniform":
            l_density = np.ones(n)
        elif g.distribution == "normal":
            l_density = np.exp(-((contributions - g.mu) ** 2) / (2 * g.sigma ** 2)) / (
                g.sigma * math.sqrt(2 * math.pi)
            )
            trunc_mass = (
                math.erf((1 - g.mu) / (g.sigma * math.sqrt(2)))
                - math.erf((-g.mu) / (g.sigma * math.sqrt(2)))
            ) / 2
            if trunc_mass <= 0:
                raise ValueError("invalid truncation mass; check mu/sigma values")
            l_density /= trunc_mass
        else:
            sys.exit("Warning: The distribution of type should be 'uniform' or 'normal'")

        return contributions, l_density

    def compute_payoffs(self, contributions: np.ndarray) -> np.ndarray:
        """Evaluate the production function on the full type grid."""
        n = self.config.grid.n
        payoffs = np.empty((n, n))
        for i in range(n):
            x = contributions[i]
            for j in range(n):
                y = contributions[j]
                payoffs[i, j] = self.production_function(x, y)
        return payoffs

    def _match_value_matrix(self, payoffs: np.ndarray) -> np.ndarray:
        """Compute PDV of a match for each type pair (i,j)."""
        m = self.config.model
        denom = 1.0 - m.beta * (1.0 - m.delta)
        return payoffs / denom

    def _compute_meeting_and_match_terms(
        self,
        alpha: np.ndarray,
        u: np.ndarray,
        match_values: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute meet(i) and E[match value | meet, i] for all i."""
        n = self.config.grid.n
        rho = self.config.model.rho

        weighted_availability = alpha * u[np.newaxis, :]
        total_availability = np.sum(weighted_availability, axis=1)
        meet = np.minimum(1.0, rho * total_availability / n)

        safe_totals = np.where(total_availability > 0, total_availability, 1.0)
        partner_weights = weighted_availability / safe_totals[:, np.newaxis]
        partner_weights[total_availability <= 0, :] = 0.0
        ev_match_given_meet = np.sum(partner_weights * match_values, axis=1)
        return meet, ev_match_given_meet

    def solve_u_fixed_point(
        self,
        alpha: np.ndarray,
        l_density: np.ndarray,
        normalize_mass: bool = True,
    ) -> np.ndarray:
        """Solve the proxy fixed-point equation for the unmatched profile u."""
        m = self.config.model
        s = self.config.solver
        n = self.config.grid.n

        u_prev = l_density.copy()
        target_mass = np.sum(l_density) / n

        for _ in range(s.u_max_iter):
            denom = m.delta + m.rho * np.dot(alpha, u_prev) / n
            denom = np.maximum(denom, 1e-14)
            u_candidate = m.delta * l_density / denom
            u_next = (1.0 - s.damp_u) * u_prev + s.damp_u * u_candidate

            if normalize_mass:
                current_mass = np.sum(u_next) / n
                if current_mass > 0:
                    u_next *= target_mass / current_mass

            err = np.linalg.norm(u_next - u_prev)
            u_prev = u_next
            if err < s.tol_u:
                break

        return u_prev

    def estimate_matching_counts(self, alpha: np.ndarray, l_density: np.ndarray) -> Tuple[float, float, float]:
        """Estimate pairs, unmatched individuals, and matched individuals."""
        u_raw = self.solve_u_fixed_point(alpha=alpha, l_density=l_density, normalize_mass=False)
        total_individuals = float(np.sum(l_density))
        unmatched_individuals = float(np.sum(u_raw))
        matched_individuals = max(0.0, total_individuals - unmatched_individuals)
        pairs = matched_individuals / 2.0
        return pairs, unmatched_individuals, matched_individuals

    def solve_value_iteration(
        self,
        alpha: np.ndarray,
        u: np.ndarray,
        payoffs: np.ndarray,
        e_grid: np.ndarray,
        V_init: np.ndarray | None = None,
    ) -> ValueIterationResult:
        """Solve unmatched values with endogenous effort choice over ``e_grid``."""
        m = self.config.model
        s = self.config.solver
        n = self.config.grid.n

        match_values = self._match_value_matrix(payoffs)
        meet, ev_match_given_meet = self._compute_meeting_and_match_terms(alpha, u, match_values)

        p_grid = 1.0 - np.exp(-m.eta * e_grid)
        effort_cost = 0.5 * m.phi * (e_grid ** 2)

        if V_init is None:
            V = np.zeros(n)
        else:
            V = V_init.copy()

        policy_idx = np.zeros(n, dtype=int)
        final_err = np.inf
        iter_used = 0

        for it in range(s.v_max_iter):
            V_new = np.empty_like(V)
            idx_new = np.empty_like(policy_idx)

            for i in range(n):
                i_up = min(i + m.upgrade_step, n - 1)

                denom = 1.0 - m.beta * (1.0 - meet[i]) * (1.0 - p_grid)
                numer = -effort_cost + m.beta * (
                    (1.0 - meet[i]) * p_grid * V[i_up] + meet[i] * ev_match_given_meet[i]
                )

                V_i_by_e = numer / denom
                k = int(np.argmax(V_i_by_e))
                V_new[i] = V_i_by_e[k]
                idx_new[i] = k

            final_err = np.max(np.abs(V_new - V))
            V = V_new
            policy_idx = idx_new
            iter_used = it + 1

            if final_err < s.v_tol:
                break

        efforts = e_grid[policy_idx]
        return ValueIterationResult(
            values=V,
            effort_policy=efforts,
            meet=meet,
            iterations=iter_used,
            error=float(final_err),
        )

    def solve_value_no_effort(
        self,
        alpha: np.ndarray,
        u: np.ndarray,
        payoffs: np.ndarray,
        V_init: np.ndarray | None = None,
    ) -> NoEffortValueResult:
        """Solve counterfactual values when effort is fixed at zero."""
        m = self.config.model
        s = self.config.solver
        n = self.config.grid.n

        match_values = self._match_value_matrix(payoffs)
        meet, ev_match_given_meet = self._compute_meeting_and_match_terms(alpha, u, match_values)

        if V_init is None:
            V_no = np.zeros(n)
        else:
            V_no = V_init.copy()

        const = m.beta * meet * ev_match_given_meet
        coeff = m.beta * (1.0 - meet)

        final_err = np.inf
        iter_used = 0
        for it in range(s.v_max_iter):
            V_next = const + coeff * V_no
            final_err = np.max(np.abs(V_next - V_no))
            V_no = V_next
            iter_used = it + 1
            if final_err < s.v_tol:
                break

        return NoEffortValueResult(
            values=V_no,
            meet=meet,
            iterations=iter_used,
            error=float(final_err),
        )

    def _solve_no_effort_equilibrium(
        self,
        alpha_init: np.ndarray,
        l_density: np.ndarray,
        payoffs: np.ndarray,
        V_init: np.ndarray,
        verbose: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Solve the no-effort counterfactual to a self-consistent fixed point.

        Mirrors the outer loop in ``run`` but uses ``solve_value_no_effort``
        instead of ``solve_value_iteration``.

        Returns ``(alpha, u, V_no_effort, alpha_binary)``.
        """
        s = self.config.solver

        alpha = alpha_init.copy()
        V = V_init.copy()

        alpha_relax = float(s.alpha_relax)
        alpha_kappa = float(s.alpha_kappa)

        seen_binary: Dict[bytes, int] = {
            (alpha >= 0.5).astype(np.uint8).tobytes(): 0
        }

        for outer in range(1, s.outer_max_iter + 1):
            u = self.solve_u_fixed_point(alpha=alpha, l_density=l_density)

            no_effort_res = self.solve_value_no_effort(
                alpha=alpha, u=u, payoffs=payoffs, V_init=V,
            )
            V = no_effort_res.values

            alpha_target = self.update_alpha(
                V=V, payoffs=payoffs, smooth=s.smooth_alpha, kappa=alpha_kappa,
            )
            alpha_next = (1.0 - alpha_relax) * alpha + alpha_relax * alpha_target

            max_alpha_diff = float(np.max(np.abs(alpha_next - alpha)))

            if verbose and outer % 10 == 0:
                alpha_prev_binary = (alpha >= 0.5).astype(float)
                alpha_next_binary = (alpha_next >= 0.5).astype(float)
                alpha_changes = int(
                    np.count_nonzero(alpha_next_binary != alpha_prev_binary)
                )
                print(
                    f"  No-effort iter {outer:03d} | "
                    f"alpha bin changes: {alpha_changes:6d} | "
                    f"max|dalpha|={max_alpha_diff:.2e} | "
                    f"V range: [{V.min():.6f}, {V.max():.6f}] | "
                    f"V iters: {no_effort_res.iterations:4d} "
                    f"(err={no_effort_res.error:.2e})"
                )

            if max_alpha_diff < s.alpha_tol:
                alpha = alpha_next
                break

            alpha_next_binary = (alpha_next >= 0.5).astype(np.uint8)
            alpha_key = alpha_next_binary.tobytes()
            if alpha_key in seen_binary:
                if max_alpha_diff < 10.0 * s.alpha_tol:
                    if verbose and outer % 10 == 0:
                        print(
                            "  No-effort cycle detected: repeated binary set "
                            f"from iteration {seen_binary[alpha_key]}."
                        )
                    alpha = alpha_next
                    break

                alpha_relax = max(0.005, 0.5 * alpha_relax)
                alpha_kappa = max(10.0, 0.9 * alpha_kappa)
                if verbose and outer % 10 == 0:
                    print(
                        f"  No-effort binary pattern repeated "
                        f"(iter {seen_binary[alpha_key]}); "
                        f"reducing alpha_relax -> {alpha_relax:.4f}, "
                        f"alpha_kappa -> {alpha_kappa:.1f}"
                    )

            seen_binary[alpha_key] = outer
            alpha = alpha_next

        # Final consistent recomputation.
        u = self.solve_u_fixed_point(alpha=alpha, l_density=l_density)
        no_effort_res = self.solve_value_no_effort(
            alpha=alpha, u=u, payoffs=payoffs, V_init=V,
        )
        V_no_effort = no_effort_res.values

        alpha_no_effort_binary = self.update_alpha(
            V=V_no_effort, payoffs=payoffs, smooth=False, kappa=alpha_kappa,
        )

        return alpha, u, V_no_effort, alpha_no_effort_binary

    def update_alpha(
        self,
        V: np.ndarray,
        payoffs: np.ndarray,
        smooth: bool,
        kappa: float,
    ) -> np.ndarray:
        """Update the acceptance matrix using TU threshold logic."""
        match_values = self._match_value_matrix(payoffs)
        gap = match_values - (V[:, np.newaxis] + V[np.newaxis, :])

        if not smooth:
            return (gap >= 0.0).astype(float)

        z = np.clip(kappa * gap, -60.0, 60.0)
        return 1.0 / (1.0 + np.exp(-z))

    def run(self, e_grid: np.ndarray, verbose: bool = True) -> SimulationResult:
        """Run the full outer fixed-point algorithm and return all outputs."""
        s = self.config.solver
        m = self.config.model
        n = self.config.grid.n

        contributions, l_density = self.compute_type_grid()
        payoffs = self.compute_payoffs(contributions)

        alpha = np.ones((n, n), dtype=float)
        V = np.zeros(n)
        u = l_density.copy()
        meet = np.zeros(n)
        effort_policy = np.zeros(n)

        seen_binary: Dict[bytes, int] = {(alpha >= 0.5).astype(np.uint8).tobytes(): 0}
        converged = False
        cycle_detected = False

        alpha_relax = float(s.alpha_relax)
        alpha_kappa = float(s.alpha_kappa)

        for outer in range(1, s.outer_max_iter + 1):
            u = self.solve_u_fixed_point(alpha=alpha, l_density=l_density)

            value_res = self.solve_value_iteration(
                alpha=alpha,
                u=u,
                payoffs=payoffs,
                e_grid=e_grid,
                V_init=V,
            )
            V = value_res.values
            effort_policy = value_res.effort_policy
            meet = value_res.meet

            alpha_target = self.update_alpha(
                V=V,
                payoffs=payoffs,
                smooth=s.smooth_alpha,
                kappa=alpha_kappa,
            )
            alpha_next = (1.0 - alpha_relax) * alpha + alpha_relax * alpha_target

            alpha_prev_binary = (alpha >= 0.5).astype(float)
            alpha_next_binary = (alpha_next >= 0.5).astype(float)
            alpha_changes = int(np.count_nonzero(alpha_next_binary != alpha_prev_binary))
            max_alpha_diff = float(np.max(np.abs(alpha_next - alpha)))

            sample_ids = [0, n // 2, n - 1]
            effort_msg = ", ".join([f"e[{i}]={effort_policy[i]:.3f}" for i in sample_ids])

            if verbose and outer % 10 == 0:

                print(
                    f"Outer iter {outer:03d} | alpha bin changes: {alpha_changes:6d} | "
                    f"max|dalpha|={max_alpha_diff:.2e} | "
                    f"V range: [{V.min():.6f}, {V.max():.6f}] | "
                    f"V iters: {value_res.iterations:4d} (err={value_res.error:.2e}) | {effort_msg}"
                )

            if max_alpha_diff < s.alpha_tol:
                converged = True
                alpha = alpha_next
                break

            alpha_key = alpha_next_binary.astype(np.uint8).tobytes()
            if alpha_key in seen_binary:
                if max_alpha_diff < 10.0 * s.alpha_tol:
                    cycle_detected = True
                    if verbose and outer % 10 == 0:
                        print(
                            "Cycle detected in alpha update: "
                            f"repeated binary set from iteration {seen_binary[alpha_key]}."
                        )
                    alpha = alpha_next
                    break

                alpha_relax = max(0.005, 0.5 * alpha_relax)
                alpha_kappa = max(10.0, 0.9 * alpha_kappa)
                if verbose and outer % 10 == 0:
                    print(
                        f"Binary pattern repeated (iter {seen_binary[alpha_key]}) but alpha is still moving; "
                        f"reducing alpha_relax -> {alpha_relax:.4f}, alpha_kappa -> {alpha_kappa:.1f}"
                    )

            seen_binary[alpha_key] = outer
            alpha = alpha_next

        # Final recomputation for internally consistent diagnostics.
        u = self.solve_u_fixed_point(alpha=alpha, l_density=l_density)
        value_res = self.solve_value_iteration(
            alpha=alpha,
            u=u,
            payoffs=payoffs,
            e_grid=e_grid,
            V_init=V,
        )
        V = value_res.values
        effort_policy = value_res.effort_policy
        meet = value_res.meet

        if verbose and outer % 10 == 0:
            print("Solving no-effort counterfactual equilibrium...")

        alpha_no_effort, u_no_effort, V_no_effort, alpha_no_effort_binary = (
            self._solve_no_effort_equilibrium(
                alpha_init=alpha,
                l_density=l_density,
                payoffs=payoffs,
                V_init=V,
                verbose=verbose,
            )
        )

        deltaV = V - V_no_effort
        p_effort = 1.0 - np.exp(-m.eta * effort_policy)
        idx = np.arange(n)
        idx_next = np.minimum(idx + m.upgrade_step, n - 1)
        gain_upgrade = p_effort * (V[idx_next] - V)

        if verbose and outer % 10 == 0:
            print(
                f"No-effort equilibrium solved | "
                f"deltaV range: [{deltaV.min():.6f}, {deltaV.max():.6f}]"
            )

            if converged:
                print("Algorithm converged.")
            elif cycle_detected:
                print("Algorithm stopped after cycle detection.")
            else:
                print("Algorithm reached outer_max_iter without convergence.")

        alpha_binary = (alpha >= 0.5).astype(float)
        alpha_map_diff_count = int(np.count_nonzero(alpha_binary != alpha_no_effort_binary))
        pairs, unmatched_individuals, matched_individuals = self.estimate_matching_counts(
            alpha=alpha,
            l_density=l_density,
        )

        if verbose and outer % 10 == 0:
            print(f"Alpha-map differences (with effort vs no effort): {alpha_map_diff_count}")
            print(f"Estimated number of pairs: {pairs:.6f}")
            print(f"Estimated number of unmatched individuals: {unmatched_individuals:.6f}")
            print(f"Estimated number of matched individuals: {matched_individuals:.6f}")

        return SimulationResult(
            contributions=contributions,
            l_density=l_density,
            alpha=alpha,
            alpha_binary=alpha_binary,
            alpha_no_effort_binary=alpha_no_effort_binary,
            payoffs=payoffs,
            V=V,
            V_no_effort=V_no_effort,
            deltaV=deltaV,
            gain_upgrade=gain_upgrade,
            effort_policy=effort_policy,
            u=u,
            u_no_effort=u_no_effort,
            meet=meet,
            pairs=pairs,
            unmatched_individuals=unmatched_individuals,
            matched_individuals=matched_individuals,
            converged=converged,
            cycle_detected=cycle_detected,
        )
