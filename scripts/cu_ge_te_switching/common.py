"""Shared helpers for the Cu-Ge-Te phase-switching study.

The study starts from the hull campaign of 2026-09-29
(``docs/archive/cu_ge_te_hull_mapping_20260929.md``) and asks which of its
polymorphs could form a switchable pair.  Every stage writes into
:data:`OUT_DIR`; the stages are separate scripts so that the GPU-bound ones can
be rerun without repeating the CPU screening.
"""
from __future__ import annotations

import logging
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
HULL_DIR = REPO_ROOT / "artifacts" / "cu_ge_te_hull"
HULL_CSV = HULL_DIR / "cu_ge_te_hull_entries.csv"
OUT_DIR = REPO_ROOT / "artifacts" / "cu_ge_te_switching"
ELEMENTS = ("Cu", "Ge", "Te")
MLIP = "orb_conserv_inf"

# Space groups whose point group is polar (1, 2, m, mm2, 4, 4mm, 3, 3m, 6, 6mm):
# the only ones that can carry a spontaneous polarisation, and so the only ones
# an electric field couples to linearly.
_POLAR_RANGES = [(1, 1), (3, 9), (25, 46), (75, 80), (99, 110), (143, 146),
                 (156, 161), (168, 173), (183, 186)]
POLAR_SPACE_GROUPS = frozenset(
    n for lo, hi in _POLAR_RANGES for n in range(lo, hi + 1)
)


def is_polar(space_group_number: int) -> bool:
    return int(space_group_number) in POLAR_SPACE_GROUPS


def setup_logging() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger("cu_ge_te_switching")


def orb_calculator(device: str = "cuda", d3: bool = False):
    """The ORB-v3 conservative calculator the hull campaign relaxed with.

    With *d3*, the same model plus a PBE D3(BJ) dispersion term (orb-models'
    ``D3SumModel``).  ORB is fitted to PBE, which barely binds the lone-pair
    and layer contacts of tellurides: zero-pressure liquids and glasses come out
    ~30% too dilute without it.  Energies with and without D3 are on different
    scales and are never mixed.
    """
    if not d3:
        from wyckoff_transformer.evaluation.hull_mlips import build_hull_calculator

        return build_hull_calculator(MLIP, device=device)
    from orb_models.forcefield import pretrained
    from orb_models.forcefield.inference.calculator import ORBCalculator
    from orb_models.forcefield.inference.d3_model import AlchemiDFTD3, D3SumModel

    from wyckoff_transformer.evaluation.hull_mlips import ORB_PRECISION

    model, adapter = pretrained.orb_v3_conservative_inf_omat(device=device, precision=ORB_PRECISION,
                                                             compile=False)
    summed = D3SumModel(model.eval(), AlchemiDFTD3(functional="PBE", damping="BJ")).to(device)
    return ORBCalculator(summed, adapter, device=device)


def relax(atoms, calculator, fmax: float = 0.005, steps: int = 2000,
          cell: bool = True):
    """Relax positions (and the cell, through a Frechet filter) with FIRE.

    Returns the relaxed atoms and whether the force criterion was reached.
    """
    from ase.filters import FrechetCellFilter
    from ase.optimize import FIRE

    atoms = atoms.copy()
    atoms.calc = calculator
    target = FrechetCellFilter(atoms) if cell else atoms
    optimizer = FIRE(target, logfile=None)
    converged = bool(optimizer.run(fmax=fmax, steps=steps))
    return atoms, converged


def frozen(images, energies):
    """Copies carrying their own energies: the images share one calculator, so
    writing them directly would stamp every frame with the last result."""
    from ase.calculators.singlepoint import SinglePointCalculator

    out = []
    for image, energy in zip(images, energies):
        copy = image.copy()
        copy.calc = SinglePointCalculator(copy, energy=float(energy))
        out.append(copy)
    return out
