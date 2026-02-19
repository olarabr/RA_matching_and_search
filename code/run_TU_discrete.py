import math
import os
import sys

import matplotlib
import numpy as np
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

def compute_type_grid(n, distribution="uniform", mu=0.5, sigma=0.1):
    """Midpoint grid on [0,1] and corresponding type density values."""
    contributions = np.linspace(1 / n / 2, 1 - 1 / n / 2, n)

    if distribution == "uniform":
        l_density = np.ones(n)
    elif distribution == "normal":
        if sigma <= 0:
            raise ValueError("sigma must be strictly positive for normal distribution")
        l_density = np.exp(-((contributions - mu) ** 2) / (2 * sigma ** 2)) / (
            sigma * math.sqrt(2 * math.pi)
        )
        trunc_mass = (
            math.erf((1 - mu) / (sigma * math.sqrt(2)))
            - math.erf((-mu) / (sigma * math.sqrt(2)))
        ) / 2
        if trunc_mass <= 0:
            raise ValueError("invalid truncation mass; check mu/sigma values")
        l_density /= trunc_mass
    else:
        sys.exit("Warning: The distribution of type should be 'uniform' or 'normal'")

    return contributions, l_density


def compute_payoffs(n, contributions, production_function):
    """Compute payoff matrix on the type grid."""
    payoffs = np.empty((n, n))
    for i in range(n):
        x = contributions[i]
        for j in range(n):
            y = contributions[j]
            payoffs[i, j] = production_function(x, y)
    return payoffs


def _compute_meeting_and_match_terms(alpha, u, rho, match_values, n):
    """Compute meet(i) and E[match value | meet, i] for each type i."""
    weighted_availability = alpha * u[np.newaxis, :]
    total_availability = np.sum(weighted_availability, axis=1)
    meet = np.minimum(1.0, rho * total_availability / n)

    safe_totals = np.where(total_availability > 0, total_availability, 1.0)
    partner_weights = weighted_availability / safe_totals[:, np.newaxis]
    partner_weights[total_availability <= 0, :] = 0.0
    ev_match_given_meet = np.sum(partner_weights * match_values, axis=1)
    return meet, ev_match_given_meet


def solve_u_fixed_point(
    alpha,
    delta,
    rho,
    l_density,
    n,
    tol_u=1e-10,
    max_iter_u=20000,
    damp_u=1.0,
    normalize_mass=True,
):
    """
    Proxy fixed point:
        u = delta * l_density / (delta + rho * dot(alpha, u) / n)
    """
    if damp_u <= 0 or damp_u > 1:
        raise ValueError("damp_u must be in (0,1]")

    u_prev = l_density.copy()
    target_mass = np.sum(l_density) / n

    for _ in range(max_iter_u):
        denom = delta + rho * np.dot(alpha, u_prev) / n
        denom = np.maximum(denom, 1e-14)
        u_candidate = delta * l_density / denom
        u_next = (1 - damp_u) * u_prev + damp_u * u_candidate

        if normalize_mass:
            current_mass = np.sum(u_next) / n
            if current_mass > 0:
                u_next *= target_mass / current_mass

        err = np.linalg.norm(u_next - u_prev)
        u_prev = u_next
        if err < tol_u:
            break

    return u_prev


def estimate_matching_counts(alpha, delta, rho, l_density, n, tol_u=1e-10):
    """
    Estimate stock counts from the unnormalized u fixed point.

    Interpreting the grid as n representative individuals:
      total individuals ~= sum(l_density)
      unmatched individuals ~= sum(u_raw)
      matched individuals = total - unmatched
      pairs = matched / 2
    """
    u_raw = solve_u_fixed_point(
        alpha=alpha,
        delta=delta,
        rho=rho,
        l_density=l_density,
        n=n,
        tol_u=tol_u,
        normalize_mass=False,
    )
    total_individuals = float(np.sum(l_density))
    unmatched_individuals = float(np.sum(u_raw))
    matched_individuals = max(0.0, total_individuals - unmatched_individuals)
    pairs = matched_individuals / 2.0
    return pairs, unmatched_individuals, matched_individuals


def solve_V_value_iteration(
    alpha,
    u,
    payoffs,
    beta,
    delta,
    rho,
    phi,
    eta,
    e_grid,
    v_tol=1e-10,
    v_max_iter=10000,
    V_init=None,
):
    """Solve the unmatched value Bellman equation with discrete effort choices."""
    n = payoffs.shape[0]
    denom = 1 - beta * (1 - delta)
    if denom <= 0:
        raise ValueError("Need 1 - beta*(1-delta) > 0 for finite match values")

    match_values = payoffs / denom

    # Meeting probabilities and conditional partner values for each i.
    meet, ev_match_given_meet = _compute_meeting_and_match_terms(alpha, u, rho, match_values, n)

    p_grid = 1.0 - np.exp(-eta * e_grid)
    effort_cost = 0.5 * phi * (e_grid ** 2)

    if V_init is None:
        V = np.zeros(n)
    else:
        V = V_init.copy()

    policy_idx = np.zeros(n, dtype=int)
    final_err = np.inf
    iter_used = 0

    for it in range(v_max_iter):
        V_new = np.empty_like(V)
        idx_new = np.empty_like(policy_idx)

        for i in range(n):
            i_up = min(i + 2, n - 1)
            ev_unmatched_next = (1 - p_grid) * V[i] + p_grid * V[i_up]
            objective = -effort_cost + beta * (
                (1 - meet[i]) * ev_unmatched_next + meet[i] * ev_match_given_meet[i]
            )
            k = int(np.argmax(objective))
            V_new[i] = objective[k]
            idx_new[i] = k

        final_err = np.max(np.abs(V_new - V))
        V = V_new
        policy_idx = idx_new
        iter_used = it + 1

        if final_err < v_tol:
            break

    efforts = e_grid[policy_idx]
    return V, efforts, meet, iter_used, final_err


def solve_V_no_effort(
    alpha,
    u,
    payoffs,
    beta,
    delta,
    rho,
    v_tol=1e-10,
    v_max_iter=10000,
    V_init=None,
):
    """
    Counterfactual value function with effort fixed at e=0 (no type upgrading).

    Bellman per type i:
        V_no(i) = beta * [(1-meet_i) * V_no(i) + meet_i * EV_match_given_meet(i)]
    """
    n = payoffs.shape[0]
    denom = 1 - beta * (1 - delta)
    if denom <= 0:
        raise ValueError("Need 1 - beta*(1-delta) > 0 for finite match values")

    match_values = payoffs / denom
    meet, ev_match_given_meet = _compute_meeting_and_match_terms(alpha, u, rho, match_values, n)

    if V_init is None:
        V_no = np.zeros(n)
    else:
        V_no = V_init.copy()

    const = beta * meet * ev_match_given_meet
    coeff = beta * (1 - meet)

    final_err = np.inf
    iter_used = 0
    for it in range(v_max_iter):
        V_next = const + coeff * V_no
        final_err = np.max(np.abs(V_next - V_no))
        V_no = V_next
        iter_used = it + 1
        if final_err < v_tol:
            break

    return V_no, meet, iter_used, final_err


def update_alpha(V, payoffs, beta, delta, smooth=False, kappa=120.0):
    """TU acceptance update based on current unmatched values.

    If smooth=False, returns a hard 0/1 best response.
    If smooth=True, returns a smoothed acceptance probability in (0,1)
    using a sigmoid with steepness kappa.
    """
    denom = 1 - beta * (1 - delta)
    if denom <= 0:
        raise ValueError("Need 1 - beta*(1-delta) > 0 for finite match values")

    match_values = payoffs / denom
    gap = match_values - (V[:, np.newaxis] + V[np.newaxis, :])

    if not smooth:
        return (gap >= 0.0).astype(float)

    z = np.clip(kappa * gap, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-z))


def plot_matching_set(alpha_binary, n, fig_name, distribution, contributions, l_density):
    """Plot matching set using the same style as run_TU.py, saving to ./figures_2/."""
    color_matching = "Blues"
    X = np.linspace(1 / n / 2, 1 - 1 / n / 2, n)
    Y = X
    levels = [0.1, 1]

    plt.figure(figsize=(10, 10))
    plt.rc("axes", labelsize=40)
    plt.rc("xtick", labelsize=30)
    plt.rc("ytick", labelsize=30)

    if distribution == "uniform":
        plt.contour(X, Y, alpha_binary, levels, colors="k")
        plt.contourf(X, Y, alpha_binary, levels, cmap=color_matching)
        plt.gca().set_aspect("equal", adjustable="box")
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.xticks(np.arange(0, 1.1, 0.2))
        plt.yticks(np.arange(0, 1.1, 0.2))
        plt.gca().xaxis.set_major_formatter(FormatStrFormatter("%g"))
        plt.gca().yaxis.set_major_formatter(FormatStrFormatter("%g"))
        plt.gca().tick_params(axis="x", pad=10)
        plt.gca().tick_params(axis="y", pad=10)
        plt.xlabel("x")
        plt.ylabel("y")

    else:
        left, width = 0.1, 0.65
        bottom, height = 0.1, 0.65
        bottom_h = left_h = left + width + 0.02
        rect_set = [left, bottom, width, height]
        rect_hist_x = [left, bottom_h, width, 0.2]
        rect_hist_y = [left_h, bottom, 0.2, height]
        ax_set = plt.axes(rect_set)
        ax_hist_x = plt.axes(rect_hist_x)
        ax_hist_y = plt.axes(rect_hist_y)

        ax_set.contour(X, Y, alpha_binary, levels, colors="k")
        ax_set.contourf(X, Y, alpha_binary, levels, cmap=color_matching)
        ax_hist_x.axis("off")
        ax_hist_y.axis("off")
        ax_hist_x.plot(contributions, l_density)
        ax_hist_x.fill_between(contributions, 0, l_density, alpha=0.2)
        ax_hist_y.plot(l_density, contributions)
        ax_hist_y.fill_betweenx(contributions, 0, l_density, alpha=0.2)
        ax_hist_x.set_xlim((0, 1))
        ax_hist_y.set_ylim((0, 1))
        ax_set.set_xlim((0, 1))
        ax_set.set_ylim((0, 1))
        ax_set.set_xticks(np.arange(0, 1.1, 0.2))
        ax_set.set_yticks(np.arange(0, 1.1, 0.2))
        ax_set.xaxis.set_major_formatter(FormatStrFormatter("%g"))
        ax_set.yaxis.set_major_formatter(FormatStrFormatter("%g"))
        ax_set.tick_params(axis="x", pad=10)
        ax_set.tick_params(axis="y", pad=10)
        ax_set.set_xlabel("x")
        ax_set.set_ylabel("y")

    out_dir = "./figures_2"
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, fig_name), bbox_inches="tight", pad_inches=0)
    plt.close()


def plot_value_diagnostics(contributions, V, V_no_effort, deltaV, gain_upgrade, u, effort_policy):
    """Plot value diagnostics and save all figures to ./figures_2/."""
    out_dir = "./figures_2"
    os.makedirs(out_dir, exist_ok=True)

    # A) V with effort vs no effort.
    plt.figure(figsize=(10, 6))
    plt.plot(contributions, V, lw=2, label="V (with effort)")
    plt.plot(contributions, V_no_effort, lw=2, linestyle="--", label="V (no effort)")
    plt.xlabel("x")
    plt.ylabel("Value")
    plt.title("Unmatched Value by Type")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.savefig(os.path.join(out_dir, "value_V_by_type.pdf"), bbox_inches="tight", pad_inches=0.05)
    plt.close()

    # B) Delta value and one-step upgrade gain.
    plt.figure(figsize=(10, 6))
    plt.plot(contributions, deltaV, lw=2, label="ΔV = V - V_no_effort")
    plt.plot(
        contributions,
        gain_upgrade,
        lw=2,
        linestyle="--",
        label="p(e*) * [V(next) - V]",
    )
    plt.xlabel("x")
    plt.ylabel("Value Difference")
    plt.title("Value Gain from Optimal Effort by Type")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.savefig(
        os.path.join(out_dir, "value_deltaV_by_type.pdf"),
        bbox_inches="tight",
        pad_inches=0.05,
    )
    plt.close()

    # C) Unmatched density.
    plt.figure(figsize=(10, 6))
    plt.plot(contributions, u, lw=2, color="tab:orange")
    plt.xlabel("x")
    plt.ylabel("u(x)")
    plt.title("Stationary Unmatched Density by Type")
    plt.grid(alpha=0.25)
    plt.savefig(
        os.path.join(out_dir, "unmatched_density_u_by_type.pdf"),
        bbox_inches="tight",
        pad_inches=0.05,
    )
    plt.close()

    # D) Optional 1xn colormap for ΔV.
    plt.figure(figsize=(10, 2.5))
    im = plt.imshow(
        deltaV[np.newaxis, :],
        aspect="auto",
        cmap="viridis",
        origin="lower",
        extent=[0, 1, 0, 1],
    )
    plt.xlabel("x")
    plt.yticks([])
    plt.title("ΔV Colormap by Type")
    cbar = plt.colorbar(im)
    cbar.set_label("ΔV")
    plt.savefig(os.path.join(out_dir, "deltaV_colormap.pdf"), bbox_inches="tight", pad_inches=0.05)
    plt.close()

    # E) Optional 1xn colormap for optimal effort.
    plt.figure(figsize=(10, 2.5))
    im = plt.imshow(
        effort_policy[np.newaxis, :],
        aspect="auto",
        cmap="plasma",
        origin="lower",
        extent=[0, 1, 0, 1],
    )
    plt.xlabel("x")
    plt.yticks([])
    plt.title("Optimal Effort Colormap by Type")
    cbar = plt.colorbar(im)
    cbar.set_label("e*")
    plt.savefig(os.path.join(out_dir, "effort_colormap.pdf"), bbox_inches="tight", pad_inches=0.05)
    plt.close()


def find_equilibrium_TU_discrete(
    n,
    beta,
    delta,
    rho,
    phi,
    eta,
    e_grid,
    production_function,
    fig_name,
    distribution="uniform",
    mu=0.5,
    sigma=0.1,
    tol_u=1e-10,
    v_tol=1e-10,
    v_max_iter=10000,
    outer_max_iter=200,
    alpha_relax=1.0,
    smooth_alpha=True,
    alpha_kappa=120.0,
    alpha_tol=1e-6,
):
    """Outer fixed point on (u, V, alpha) for the discrete-time TU model. During iteration, alpha is treated as a continuous matrix in [0,1] for numerical stability."""
    if alpha_relax <= 0 or alpha_relax > 1:
        raise ValueError("alpha_relax must be in (0,1]")
    alpha_relax = float(alpha_relax)
    alpha_kappa = float(alpha_kappa)

    contributions, l_density = compute_type_grid(n, distribution, mu, sigma)
    payoffs = compute_payoffs(n, contributions, production_function)

    alpha = np.ones((n, n), dtype=float)
    V = np.zeros(n)
    u = l_density.copy()
    meet = np.zeros(n)
    effort_policy = np.zeros(n)
    seen_binary = {(alpha >= 0.5).astype(np.uint8).tobytes(): 0}
    converged = False
    cycle_detected = False

    for outer in range(1, outer_max_iter + 1):
        u = solve_u_fixed_point(alpha, delta, rho, l_density, n, tol_u=tol_u)

        V, effort_policy, meet, v_iters, v_err = solve_V_value_iteration(
            alpha=alpha,
            u=u,
            payoffs=payoffs,
            beta=beta,
            delta=delta,
            rho=rho,
            phi=phi,
            eta=eta,
            e_grid=e_grid,
            v_tol=v_tol,
            v_max_iter=v_max_iter,
            V_init=V,
        )

        # Smoothed (optional) best response + damped update to avoid 2-cycles.
        alpha_target = update_alpha(V, payoffs, beta, delta, smooth=smooth_alpha, kappa=alpha_kappa)
        alpha_next = (1 - alpha_relax) * alpha + alpha_relax * alpha_target

        # Diagnostics: binary changes (intuition) and continuous max change (convergence).
        alpha_prev_binary = (alpha >= 0.5).astype(float)
        alpha_next_binary = (alpha_next >= 0.5).astype(float)
        alpha_changes = int(np.count_nonzero(alpha_next_binary != alpha_prev_binary))
        max_alpha_diff = float(np.max(np.abs(alpha_next - alpha)))

        sample_ids = [0, n // 2, n - 1]
        effort_msg = ", ".join([f"e[{i}]={effort_policy[i]:.3f}" for i in sample_ids])

        print(
            f"Outer iter {outer:03d} | alpha bin changes: {alpha_changes:6d} | "
            f"max|dalpha|={max_alpha_diff:.2e} | "
            f"V range: [{V.min():.6f}, {V.max():.6f}] | "
            f"V iters: {v_iters:4d} (err={v_err:.2e}) | {effort_msg}"
        )

        # Convergence on continuous alpha.
        if max_alpha_diff < alpha_tol:
            converged = True
            alpha = alpha_next
            break

        # Cycle detection on the *binary* pattern implied by alpha_next.
        alpha_key = alpha_next_binary.astype(np.uint8).tobytes()
        if alpha_key in seen_binary:
            # Repeating the *binary* pattern is common when we iterate on a continuous alpha.
            # Only treat this as a problematic cycle if the continuous alpha is not moving.
            if max_alpha_diff < 10 * alpha_tol:
                cycle_detected = True
                print(
                    f"Cycle detected in alpha update: repeated binary set from iteration {seen_binary[alpha_key]}."
                )
                alpha = alpha_next
                break
            else:
                # Still moving in continuous alpha: continue iterating, but dampen the update to stabilize.
                alpha_relax = max(0.005, 0.5 * alpha_relax)
                alpha_kappa = max(10.0, 0.9 * alpha_kappa)
                print(
                    f"Binary pattern repeated (iter {seen_binary[alpha_key]}) but alpha is still moving; "
                    f"reducing alpha_relax -> {alpha_relax:.4f}, alpha_kappa -> {alpha_kappa:.1f}"
                )

        seen_binary[alpha_key] = outer
        alpha = alpha_next

    # Recompute final objects with the final alpha for consistent diagnostics.
    u = solve_u_fixed_point(alpha, delta, rho, l_density, n, tol_u=tol_u)
    V, effort_policy, meet, _, _ = solve_V_value_iteration(
        alpha=alpha,
        u=u,
        payoffs=payoffs,
        beta=beta,
        delta=delta,
        rho=rho,
        phi=phi,
        eta=eta,
        e_grid=e_grid,
        v_tol=v_tol,
        v_max_iter=v_max_iter,
        V_init=V,
    )
    V_no_effort, _, v_no_iters, v_no_err = solve_V_no_effort(
        alpha=alpha,
        u=u,
        payoffs=payoffs,
        beta=beta,
        delta=delta,
        rho=rho,
        v_tol=v_tol,
        v_max_iter=v_max_iter,
        V_init=V,
    )
    deltaV = V - V_no_effort
    p_effort = 1.0 - np.exp(-eta * effort_policy)
    V_next = np.empty_like(V)
    V_next[:-1] = V[1:]
    V_next[-1] = V[-1]
    gain_upgrade = p_effort * (V_next - V)
    print(
        f"No-effort solve | iters: {v_no_iters:4d} (err={v_no_err:.2e}) | "
        f"deltaV range: [{deltaV.min():.6f}, {deltaV.max():.6f}]"
    )

    alpha_binary = (alpha >= 0.5).astype(float)
    plot_matching_set(alpha_binary, n, fig_name, distribution, contributions, l_density)
    plot_value_diagnostics(contributions, V, V_no_effort, deltaV, gain_upgrade, u, effort_policy)

    if converged:
        print("Algorithm converged.")
    elif cycle_detected:
        print("Algorithm stopped after cycle detection.")
    else:
        print("Algorithm reached outer_max_iter without convergence.")

    pairs, unmatched_individuals, matched_individuals = estimate_matching_counts(
        alpha=alpha,
        delta=delta,
        rho=rho,
        l_density=l_density,
        n=n,
        tol_u=tol_u,
    )
    print(f"Estimated number of pairs: {pairs:.6f}")
    print(f"Estimated number of unmatched individuals: {unmatched_individuals:.6f}")
    print(f"Estimated number of matched individuals: {matched_individuals:.6f}")

    return {
        "alpha": alpha,
        "alpha_binary": alpha_binary,
        "V": V,
        "V_no_effort": V_no_effort,
        "deltaV": deltaV,
        "gain_upgrade": gain_upgrade,
        "effort_policy": effort_policy,
        "u": u,
        "meet": meet,
        "pairs": pairs,
        "unmatched_individuals": unmatched_individuals,
        "matched_individuals": matched_individuals,
        "converged": converged,
        "cycle_detected": cycle_detected,
    }


if __name__ == "__main__":
    # Example run with requested safe parameters.
    n = 500
    beta = 0.8
    delta = 0.05
    rho = 0.6
    phi = 1
    eta = 10.0
    e_grid = np.linspace(0.0, 2.0, 51)
    distribution = "normal"
    mu = 0.5
    sigma = 0.1

    def production_function(x, y):
        return (x + y) ** 2

    fig_name = "matching_set_TU_discrete.pdf"

    find_equilibrium_TU_discrete(
        n=n,
        beta=beta,
        delta=delta,
        rho=rho,
        phi=phi,
        eta=eta,
        e_grid=e_grid,
        production_function=production_function,
        fig_name=fig_name,
        distribution=distribution,
        mu=mu,
        sigma=sigma,
        tol_u=1e-8,
        v_tol=1e-8,
        v_max_iter=10000,
        outer_max_iter=2000,
        alpha_relax=0.1,
        smooth_alpha=True,
        alpha_kappa=120.0,
        alpha_tol=1e-6,
    )

    print(f"Figure saved to ./figures_2/{fig_name}")
