"""Configuration dataclasses for the discrete-time TU Shimer-Smith model.

The configuration is split into small pieces so model, solver, and plotting
settings are explicit and easier to maintain.
"""

from dataclasses import dataclass, field
from typing import Callable


ProductionFunction = Callable[[float, float], float]


@dataclass
class TypeGridConfig:
    """Type-grid and exogenous type-distribution settings."""

    n: int = 200
    distribution: str = "normal"
    mu: float = 0.5
    sigma: float = 0.1


@dataclass
class ModelConfig:
    """Economic parameters for the discrete-time environment."""

    beta: float = 0.96
    delta: float = 0.05
    rho: float = 0.6
    phi: float = 1.0
    eta: float = 2.0
    # Upgrade step count used when effort succeeds.
    # Kept configurable to preserve behavior from prior scripts.
    upgrade_step: int = 1


@dataclass
class SolverConfig:
    """Numerical controls for fixed-point and value-function iterations."""

    tol_u: float = 1e-8
    v_tol: float = 1e-8
    v_max_iter: int = 10000
    outer_max_iter: int = 2000
    alpha_relax: float = 0.1
    smooth_alpha: bool = True
    alpha_kappa: float = 120.0
    alpha_tol: float = 1e-6
    u_max_iter: int = 20000
    damp_u: float = 1.0


@dataclass
class PlotConfig:
    """Output plotting settings."""

    output_dir: str = "./figures_2"
    matching_set_filename: str = "matching_set_TU_discrete.pdf"


@dataclass
class TUDiscreteConfig:
    """Top-level model configuration."""

    grid: TypeGridConfig = field(default_factory=TypeGridConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    solver: SolverConfig = field(default_factory=SolverConfig)
    plot: PlotConfig = field(default_factory=PlotConfig)
