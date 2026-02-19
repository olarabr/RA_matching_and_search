"""Run the object-oriented discrete-time TU model from the code_2 layout.

This script keeps the same workflow as the original single-file version:
1) solve the model,
2) print diagnostics,
3) save matching and value-related figures to ./figures_2.
"""

from pathlib import Path

import numpy as np

from tu_discrete import (
    ModelConfig,
    PlotConfig,
    SolverConfig,
    TUDiscreteConfig,
    TUDiscreteModel,
    TypeGridConfig,
    save_all_plots,
)


def production_function(x: float, y: float) -> float:
    """Example production technology."""
    return (x + y) ** 2


def build_default_config() -> TUDiscreteConfig:
    """Default run configuration (mirrors current script defaults)."""
    output_dir = Path(__file__).resolve().parent / "figures_2"
    return TUDiscreteConfig(
        grid=TypeGridConfig(
            n=500,
            distribution="uniform",
            mu=0.5,
            sigma=0.1,
        ),
        model=ModelConfig(
            beta=0.8,
            delta=0.05,
            rho=0.6,
            phi=0.5,
            eta=10.0,
            upgrade_step=10,
        ),
        solver=SolverConfig(
            tol_u=1e-8,
            v_tol=1e-8,
            v_max_iter=10000,
            outer_max_iter=5000,
            alpha_relax=0.1,
            smooth_alpha=True,
            alpha_kappa=120.0,
            alpha_tol=1e-6,
            u_max_iter=20000,
            damp_u=0.5,
        ),
        plot=PlotConfig(
            output_dir=str(output_dir),
            matching_set_filename="matching_set_TU_discrete.pdf",
        ),
    )


def main() -> None:
    """Solve model, save all figures, and print output locations."""
    config = build_default_config()
    e_grid = np.linspace(0.0, 2.0, 51)

    model = TUDiscreteModel(config=config, production_function=production_function)
    result = model.run(e_grid=e_grid, verbose=True)

    matching_path = save_all_plots(
        result=result,
        grid_config=config.grid,
        plot_config=config.plot,
    )

    distribution_path = (Path(config.plot.output_dir) / "distribution_overlays_combined.pdf").resolve()
    print(f"Distribution overlay figure saved to {distribution_path}")
    print(f"Matching-set figure saved to {matching_path}")
    print(f"Diagnostic figures saved to {config.plot.output_dir}")


if __name__ == "__main__":
    main()
