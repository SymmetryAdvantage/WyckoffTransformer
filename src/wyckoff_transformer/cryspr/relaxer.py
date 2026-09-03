"""ASE-based structure relaxation with optional symmetry and cell constraints."""
import logging
import os
from pathlib import Path
from typing import Optional

os.environ.setdefault("SPGLIB_OLD_ERROR_HANDLING", "0")

from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.constraints import FixAtoms, FixSymmetry
from ase.filters import FrechetCellFilter as CellFilter
from ase.io import write
from ase.optimize import BFGS
from ase.optimize.optimize import Optimizer
import spglib

logger = logging.getLogger(__name__)

#: Label of the final, symmetry-free stage of :func:`stepwise_relax`.
FINAL_CIF_LABEL = "3_no-sym_cell+pos"
#: Filename suffix of the CIF that stage writes; :mod:`wyckoff_transformer.cryspr.generator`
#: globs for it to pick up each trial's relaxed structure.
FINAL_CIF_SUFFIX = f"_{FINAL_CIF_LABEL}.cif"

#: Label of the symmetry-constrained cell+positions stage, which becomes the
#: last one when ``release_symmetry=False``.
SYMMETRIC_CIF_LABEL = "2_sym_cell+pos"
SYMMETRIC_CIF_SUFFIX = f"_{SYMMETRIC_CIF_LABEL}.cif"


def final_cif_suffix(release_symmetry: bool = True) -> str:
    """Suffix of the CIF written by the last stage :func:`stepwise_relax` runs."""
    return FINAL_CIF_SUFFIX if release_symmetry else SYMMETRIC_CIF_SUFFIX


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


def stepwise_relax(
        atoms_in: Atoms,
        calculator: Calculator,
        optimizer: type[Optimizer] = BFGS,
        fix_symmetry: bool = True,
        release_symmetry: bool = True,
        hydrostatic_strain: bool = False,
        symprec: float = 1e-3,
        fmax: float = 0.05,
        steps_limit: int = 500,
        wdir: Path = Path("."),
        logfile_prefix: str = "",
        logfile_postfix: str = "",
) -> Atoms:
    """Relax under symmetry constraints first, then release them.

    The schedule is:

    1. Symmetry-constrained: a fix-cell warm-up (so that a random PyXtal cell is
       not dragged around by badly placed atoms), then cell + positions with a
       :class:`~ase.constraints.FixSymmetry` constraint.  Skipped, apart from
       the warm-up, when *fix_symmetry* is ``False``.
    2. Unconstrained: cell + positions with no symmetry constraint, so the
       structure can relax into a lower-symmetry minimum if one is nearby.

    Note that a structure converged to a symmetric stationary point in step 1
    stays there in step 2 unless something breaks the symmetry — the force and
    stress components along symmetry-breaking modes vanish exactly for an
    invariant potential, and only the MLIP's numerical asymmetry seeds a
    descent.  Step 2 mainly matters when step 1 hit *steps_limit* or when the
    symmetric point is unstable enough for that noise to grow.

    Args:
        atoms_in: Input structure.
        calculator: ASE Calculator.
        optimizer: Optimisation algorithm class.
        fix_symmetry: Run the symmetry-constrained step.  When ``False`` only
            the warm-up and the unconstrained step run.
        release_symmetry: Run the final unconstrained step.  Setting this to
            ``False`` gives the two-stage schedule, which costs a third less
            per trial.  Measured on 7387 trials of run ``upi73i4k``, the step
            it drops moves the energy by more than 1 meV/atom in 0.4% of trials
            and leaves the spglib space group unchanged in 398 of 398 sampled
            genes, so it is close to free of consequence for both energy and
            the space-group-dependent BAWL fingerprint.  It does still guard
            against structures that hit *steps_limit* under constraint.
        hydrostatic_strain: Restrict cell relaxation to isotropic strain.
        symprec: Symmetry tolerance in Å.
        fmax: Force convergence criterion in eV/Å.
        steps_limit: Maximum optimisation steps per stage.
        wdir: Directory for output CIF and log files.
        logfile_prefix: Prefix for log file names.
        logfile_postfix: Postfix for log file names.

    Returns:
        Relaxed :class:`~ase.Atoms` after the last stage that ran.

    Raises:
        ValueError: If both *fix_symmetry* and *release_symmetry* are ``False``,
            which would leave only the fix-cell warm-up.
    """
    if not fix_symmetry and not release_symmetry:
        raise ValueError(
            "fix_symmetry=False and release_symmetry=False leaves only the "
            "fix-cell warm-up, which never relaxes the cell."
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
        label="1_fix-cell",
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

    # Step 2: no symmetry constraint, so the structure may lower its symmetry.
    if release_symmetry:
        atoms = run_ase_relaxer(
            atoms_in=atoms,
            fix_symmetry=False,
            cell_filter=CellFilter,
            label=FINAL_CIF_LABEL,
            logfile=logfile_for("no-sym_cell+positions"),
            **shared,
        )

    return atoms
