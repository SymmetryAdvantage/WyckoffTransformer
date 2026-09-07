"""ASE-based structure relaxation with optional symmetry and cell constraints."""
import json
import logging
import os
from pathlib import Path
from typing import Optional

os.environ.setdefault("SPGLIB_OLD_ERROR_HANDLING", "0")

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.constraints import FixAtoms, FixSymmetry
from ase.filters import FrechetCellFilter as CellFilter
from ase.io import write
from ase.optimize import BFGS
from ase.optimize.optimize import Optimizer
import spglib

logger = logging.getLogger(__name__)

#: Stage labels, which name the CIF each stage writes.  Which of them holds the
#: structure a trial *kept* is decided on energy by the rattle stage, so
#: :mod:`wyckoff_transformer.cryspr.generator` writes that one out separately
#: rather than globbing for a stage.
WARMUP_CIF_LABEL = "1_fix-cell"
SYMMETRIC_CIF_LABEL = "2_sym_cell+pos"
FINAL_CIF_LABEL = "3_no-sym_cell+pos"
RATTLE_CIF_LABEL = "4_rattle_no-sym"

#: Stage-4 perturbation, as measured in the reconstruction study
#: (docs/cryspr_reconstruction_report.md): per-atom displacement in A, and the
#: standard deviation of the symmetrised cell strain, which is dimensionless.
RATTLE_STDEV = 0.05
RATTLE_STRAIN_STDEV = 0.01

#: Per-trial record of what the rattle stage did, written next to its CIF.
RATTLE_VERDICT_FILE = "rattle.json"

#: Energy the rattle has to win, eV/atom, before its structure replaces the
#: symmetric one.  Without a margin the stage would trade a converged symmetric
#: minimum for numerical noise; with it, 33.1% of trials in the study accepted,
#: by a median of 186 meV/atom.
RATTLE_ACCEPT_EV_PER_ATOM = 1e-3


def _get_spacegroup_info(atoms: Atoms, symprec: float) -> tuple[str, int]:
    """Return the international symbol and number from spglib."""
    try:
        dataset = spglib.get_symmetry_dataset(
            (atoms.cell.array, atoms.get_scaled_positions(), atoms.numbers),
            symprec=symprec,
        )
    except spglib.SpglibError as exc:
        logger.warning("Failed to determine symmetry via spglib: %s", exc)
        return "unknown", 0

    if dataset is None:
        return "unknown", 0

    symbol = getattr(dataset, "international", None)
    number = getattr(dataset, "number", None)
    if symbol is None:
        symbol = dataset["international"]
    if number is None:
        number = dataset["number"]
    return str(symbol), int(number)


def run_ase_relaxer(
        atoms_in: Atoms,
        calculator: Calculator,
        optimizer: type[Optimizer] = BFGS,
        cell_filter=None,
        fix_symmetry: bool = True,
        fix_fractional: bool = False,
        hydrostatic_strain: bool = False,
        symprec: float = 1e-3,
        fmax: float = 0.05,
        steps_limit: int = 500,
        wdir: Path = Path("."),
        label: Optional[str] = None,
        logfile: Optional[Path] = None,
) -> Atoms:
    """Run a single ASE relaxation pass on *atoms_in*.

    Args:
        atoms_in: Input structure; not modified in place.
        calculator: ASE Calculator to attach to the atoms.
        optimizer: Local optimisation algorithm class (default :class:`~ase.optimize.BFGS`).
        cell_filter: ASE filter class for cell relaxation; ``None`` keeps the
            cell fixed.
        fix_symmetry: Apply a :class:`~ase.constraints.FixSymmetry` constraint.
        fix_fractional: Fix all atomic positions (ions immobile).
        hydrostatic_strain: Restrict cell filter to isotropic strain only.
        symprec: Symmetry tolerance in Å used by :mod:`spglib` and
            :class:`~ase.constraints.FixSymmetry`.
        fmax: Force convergence criterion in eV/Å.
        steps_limit: Maximum number of optimisation steps.
        wdir: Directory for the output CIF file.
        label: Stage name used in the output CIF filename.  Defaults to
            ``"fix-cell"`` or ``"cell+pos"`` depending on *cell_filter*.
        logfile: Path to append optimiser output; ``None`` writes to *stderr*.

    Returns:
        Relaxed :class:`~ase.Atoms` object.
    """
    atoms = atoms_in.copy()
    full_formula = atoms.get_chemical_formula(mode="metal")
    reduced_formula = atoms.get_chemical_formula(mode="metal", empirical=True)
    atoms.calc = calculator

    spg0_symbol, spg0_number = _get_spacegroup_info(atoms, symprec=symprec)
    # Set every constraint in one call: Atoms.copy() carries the previous stage's
    # constraints over, so an unconstrained stage has to clear them explicitly.
    constraints = []
    if fix_fractional:
        constraints.append(FixAtoms(indices=list(range(len(atoms)))))
    if fix_symmetry:
        constraints.append(FixSymmetry(atoms, symprec=symprec))
    atoms.set_constraint(constraints)
    target = cell_filter(atoms, hydrostatic_strain=hydrostatic_strain) if cell_filter is not None else atoms

    E0 = atoms.get_potential_energy()
    logger.info(
        "Start relaxation: E₀ = %.5f eV, symmetry = %s (%d), fix_sym = %s, relax_cell = %s",
        E0, spg0_symbol, spg0_number, fix_symmetry, cell_filter is not None,
    )

    log_arg = str(logfile) if logfile is not None else "-"
    opt = optimizer(atoms=target, logfile=log_arg)
    opt.run(fmax=fmax, steps=steps_limit)

    if label is None:
        label = "fix-cell" if cell_filter is None else "cell+pos"
    cif_path = wdir / f"{reduced_formula}_{full_formula}_{label}.cif"
    write(filename=str(cif_path), images=atoms, format="cif")

    E1 = atoms.get_potential_energy()
    spg1_symbol, spg1_number = _get_spacegroup_info(atoms, symprec=symprec)
    cell_diff = (atoms.cell.cellpar() / atoms_in.cell.cellpar() - 1.0) * 100
    logger.info(
        "End relaxation: E₁ = %.5f eV, symmetry = %s (%d), max|F| = %.4f eV/Å",
        E1, spg1_symbol, spg1_number, abs(atoms.get_forces()).max(),
    )
    logger.debug("Cell diff (%%): %s", cell_diff)

    return atoms


def perturb(
        atoms_in: Atoms,
        rattle_stdev: float = RATTLE_STDEV,
        strain_stdev: float = RATTLE_STRAIN_STDEV,
        seed: Optional[int] = None,
) -> Atoms:
    """A copy of *atoms_in* with the atoms rattled and the cell strained.

    Both perturbations are needed and neither substitutes for the other: the
    rattle breaks site symmetry, the strain breaks the symmetry of the lattice
    itself, and a Wyckoff gene fixes both.

    Constraints are cleared before the displacement is applied, not afterwards:
    :meth:`ase.Atoms.set_positions` enforces the attached constraints, so a
    :class:`~ase.constraints.FixSymmetry` carried over from the previous stage
    would symmetrise the rattle away and leave the structure exactly where it
    was.

    Args:
        atoms_in: Structure to perturb; not modified in place.
        rattle_stdev: Standard deviation of the per-atom Cartesian
            displacement, Å.
        strain_stdev: Standard deviation of the cell strain, dimensionless.
            The drawn matrix is symmetrised, so the perturbation is a strain
            rather than a rotation.
        seed: Seed for the perturbation.  Passing one makes the stage
            reproducible, which matters because it can change the kept
            structure.

    Returns:
        The perturbed copy, with no constraints and no calculator.
    """
    atoms = atoms_in.copy()
    atoms.set_constraint([])
    atoms.calc = None

    rng = np.random.default_rng(seed)
    drawn = rng.normal(0.0, strain_stdev, size=(3, 3))
    strain = 0.5 * (drawn + drawn.T)
    atoms.set_cell(atoms.cell @ (np.eye(3) + strain), scale_atoms=True)
    atoms.rattle(stdev=rattle_stdev, seed=int(rng.integers(1, 2 ** 31 - 1)))
    return atoms


def stepwise_relax(
        atoms_in: Atoms,
        calculator: Calculator,
        optimizer: type[Optimizer] = BFGS,
        fix_symmetry: bool = True,
        release_symmetry: bool = True,
        rattle: bool = True,
        rattle_stdev: float = RATTLE_STDEV,
        strain_stdev: float = RATTLE_STRAIN_STDEV,
        rattle_accept: float = RATTLE_ACCEPT_EV_PER_ATOM,
        seed: Optional[int] = None,
        hydrostatic_strain: bool = False,
        symprec: float = 1e-3,
        fmax: float = 0.05,
        steps_limit: int = 500,
        wdir: Path = Path("."),
        logfile_prefix: str = "",
        logfile_postfix: str = "",
) -> Atoms:
    """Relax under symmetry constraints, then release them, then rattle.

    The schedule is:

    1. Symmetry-constrained: a fix-cell warm-up (so that a random PyXtal cell is
       not dragged around by badly placed atoms), then cell + positions with a
       :class:`~ase.constraints.FixSymmetry` constraint.  Skipped, apart from
       the warm-up, when *fix_symmetry* is ``False``.
    2. Unconstrained: cell + positions with no symmetry constraint, so the
       structure can relax into a lower-symmetry minimum if one is nearby, and
       so that the rattle has a converged, unconstrained baseline to be
       perturbed away from and compared against.
    3. Rattle: a finite perturbation of positions and cell, then another
       unconstrained relaxation, kept only if it wins *rattle_accept*.

    Stage 2 cannot do stage 3's job, which is why the rattle is on by default.
    A structure converged to a symmetric stationary point in stage 1 stays
    there under gradient descent: for a symmetry-invariant potential the force
    and stress components along symmetry-breaking modes vanish identically, so
    only the MLIP's own numerical asymmetry can seed a descent.  Measured over
    9990 trials of the reconstruction study, the unconstrained stage took *zero*
    optimiser steps in 78.2% of them.  A finite perturbation is what leaves the
    stationary point; it lowered the energy in 33.1% of trials, by a median of
    186 meV/atom, and recovered 99 further ground-state matches.

    Args:
        atoms_in: Input structure.
        calculator: ASE Calculator.
        optimizer: Optimisation algorithm class.
        fix_symmetry: Run the symmetry-constrained step.  When ``False`` only
            the warm-up and the unconstrained step run.
        release_symmetry: Run the unconstrained step before the rattle.  On by
            default, and it earns its place as the rattle's *baseline* rather
            than as a relaxation of its own: the perturbation then starts from a
            structure already at an unconstrained minimum, so the acceptance
            test asks whether the rattle found a better basin, not whether it
            finished work the constrained stages had left undone.  On its own it
            moves the energy by more than 1 meV/atom in 0.4% of 7387
            ``upi73i4k`` trials and leaves the spglib space group unchanged in
            398 of 398 sampled genes, and it takes zero optimiser steps in
            78.2% of trials -- which is what makes keeping it cheap.
        rattle: Run the rattle stage.
        rattle_stdev: Per-atom displacement of the perturbation, Å.
        strain_stdev: Cell strain of the perturbation, dimensionless.
        rattle_accept: Energy the rattled structure must win over the
            unperturbed one, eV/atom, before it replaces it.  Positive.
        seed: Seed for the perturbation; see :func:`perturb`.
        hydrostatic_strain: Restrict cell relaxation to isotropic strain.
        symprec: Symmetry tolerance in Å.
        fmax: Force convergence criterion in eV/Å.
        steps_limit: Maximum optimisation steps per stage.
        wdir: Directory for output CIF and log files.
        logfile_prefix: Prefix for log file names.
        logfile_postfix: Postfix for log file names.

    Returns:
        The kept :class:`~ase.Atoms`: the rattled structure when it won,
        otherwise the last relaxation stage that ran.

    Raises:
        ValueError: If no stage that relaxes the cell would run, i.e. all of
            *fix_symmetry*, *release_symmetry* and *rattle* are ``False``,
            leaving only the fix-cell warm-up.
    """
    if not fix_symmetry and not release_symmetry and not rattle:
        raise ValueError(
            "fix_symmetry=False, release_symmetry=False and rattle=False leaves "
            "only the fix-cell warm-up, which never relaxes the cell."
        )
    wdir = Path(wdir)
    wdir.mkdir(parents=True, exist_ok=True)

    atoms = atoms_in.copy()
    full_formula = atoms.get_chemical_formula(mode="metal")
    reduced_formula = atoms.get_chemical_formula(mode="metal", empirical=True)

    write(
        filename=str(wdir / f"{reduced_formula}_{full_formula}_0_initial.cif"),
        images=atoms,
        format="cif",
    )

    def logfile_for(stage: str) -> Path:
        parts = [p for p in (logfile_prefix, stage, logfile_postfix) if p]
        return wdir / ("_".join(parts) + ".log")

    shared = dict(
        calculator=calculator,
        optimizer=optimizer,
        hydrostatic_strain=hydrostatic_strain,
        symprec=symprec,
        fmax=fmax,
        steps_limit=steps_limit,
        wdir=wdir,
    )

    # Step 1: symmetry-constrained, cell fixed first and then released.
    atoms = run_ase_relaxer(
        atoms_in=atoms,
        fix_symmetry=fix_symmetry,
        cell_filter=None,
        label=WARMUP_CIF_LABEL,
        logfile=logfile_for("fix-cell"),
        **shared,
    )
    if fix_symmetry:
        atoms = run_ase_relaxer(
            atoms_in=atoms,
            fix_symmetry=True,
            cell_filter=CellFilter,
            label=SYMMETRIC_CIF_LABEL,
            logfile=logfile_for("sym_cell+positions"),
            **shared,
        )

    # Step 2: no symmetry constraint, so the structure may lower its symmetry --
    # and so that the rattle below is perturbing a converged, unconstrained
    # structure rather than a constrained one.
    if release_symmetry:
        atoms = run_ase_relaxer(
            atoms_in=atoms,
            fix_symmetry=False,
            cell_filter=CellFilter,
            label=FINAL_CIF_LABEL,
            logfile=logfile_for("no-sym_cell+positions"),
            **shared,
        )

    # Step 3: leave the stationary point by force, and keep the result only if
    # it is genuinely lower.  The margin is what makes this safe to run always:
    # a converged structure cannot be traded away for noise.
    if rattle:
        atoms = _rattle_stage(
            atoms,
            rattle_stdev=rattle_stdev,
            strain_stdev=strain_stdev,
            rattle_accept=rattle_accept,
            seed=seed,
            logfile=logfile_for("rattle_no-sym"),
            **shared,
        )

    return atoms


def _rattle_stage(
        atoms: Atoms,
        rattle_stdev: float,
        strain_stdev: float,
        rattle_accept: float,
        seed: Optional[int],
        logfile: Path,
        **shared,
) -> Atoms:
    """Perturb, re-relax, and return whichever structure has the lower energy."""
    if rattle_accept < 0:
        raise ValueError(f"rattle_accept must be non-negative, got {rattle_accept}")

    atoms.calc = shared["calculator"]
    energy_before = atoms.get_potential_energy()

    rattled = run_ase_relaxer(
        atoms_in=perturb(
            atoms,
            rattle_stdev=rattle_stdev,
            strain_stdev=strain_stdev,
            seed=seed,
        ),
        fix_symmetry=False,
        cell_filter=CellFilter,
        label=RATTLE_CIF_LABEL,
        logfile=logfile,
        **shared,
    )
    delta = (rattled.get_potential_energy() - energy_before) / len(atoms)
    accepted = delta < -rattle_accept
    logger.info(
        "Rattle stage: dE = %+.6f eV/atom, %s",
        delta, "accepted" if accepted else "rejected",
    )
    # Recorded per trial, not just logged: pool workers do not configure
    # logging, and how often the rattle wins -- and by how much -- is the
    # measurement that sets the trial schedule and justifies the stage.
    (shared["wdir"] / RATTLE_VERDICT_FILE).write_text(
        json.dumps(
            {
                "delta_ev_per_atom": delta,
                "accepted": accepted,
                "accept_threshold_ev_per_atom": rattle_accept,
                "rattle_stdev": rattle_stdev,
                "strain_stdev": strain_stdev,
                "seed": seed,
            }
        ) + "\n",
        encoding="utf-8",
    )
    return rattled if accepted else atoms
