"""Crystal structure generation and relaxation orchestration."""
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.io import write
from ase.neighborlist import neighbor_list
from ase.optimize import BFGS
from ase.optimize.optimize import Optimizer
from pyxtal import pyxtal
from pyxtal.tolerance import Tol_matrix

from wyckoff_transformer.cryspr.relaxer import (
    RATTLE_ACCEPT_EV_PER_ATOM,
    RATTLE_STDEV,
    RATTLE_STRAIN_STDEV,
    stepwise_relax,
    stepwise_relax_stages,
)

logger = logging.getLogger(__name__)

#: Factor scaling PyXtal's inter-atomic distance tolerances during generation.
#:
#: **What the factor multiplies.**  ``Tol_matrix(prototype="atomic")`` returns
#: ``0.5 * (r_a + r_b)`` for a pair -- the covalent-radius *mean*, not the sum.
#: So factor 1.3 asks for contacts no shorter than ``0.65 * (r_a + r_b)``, which
#: is 35% *shorter* than a covalent bond rather than 30% longer.  This comment
#: said "sum" until 2026-09-10 and the difference matters: read as the sum, 1.3
#: would forbid every real bonded crystal, and the whole tuning below would look
#: like a constraint on physical plausibility instead of the mild bias it is.
#:
#: **It biases the draw; it does not bound it.**  Measured over 600 draws of the
#: oracle cohort, 45% of the structures PyXtal *returns* have a closest contact
#: below this floor, the worst at 0.60 in tolerance units -- under PyXtal's own
#: default of 1.0.  Only 2 of 272 violations were an atom against its own
#: periodic image, so this is genuine crowding: PyXtal checks distances while
#: placing atoms rather than exhaustively on the finished cell.  Nothing
#: downstream depends on the floor holding -- the relaxation repairs the
#: contacts, landing at a median 1.82 against the ground truth's 1.83 -- but the
#: factor is a preference for loose draws, not a guarantee about them.
#:
#: **Why 1.3 rather than PyXtal's 1.0.**  What the oracle studies support for a
#: *free* lattice, which is how genes are sampled here: it draws a loose cell
#: (median 1.68x the target volume) that the variable-cell relaxation then
#: contracts to 1.34x, and that compressive annealing gives 50.8% recovery at 5
#: trials with only 1.2% generation failures
#: (docs/archive/cryspr_oracle_relaxed_cell_report.md).  The 39.8% generation
#: choking that motivated trying 1.0 appears only when the cell is pinned to the
#: ground-truth lattice, and 1.0 in that setup jams the atoms instead: 19.0%
#: recovery against the loose baseline's 50.8%.  It is also what every
#: measurement the ranking protocol rests on used, so changing it would
#: invalidate the trial schedule those numbers set.
#:
#: **It does not exclude the answer.**  All 750 ORB-relaxed reference structures
#: of the oracle cohort clear this floor, the tightest by 14% (closest contacts:
#: min 1.48, median 1.83 in tolerance units).  So the floor is not what caps
#: reconstruction; finding the right coordinates is.
#:
#: **Why it is nonetheless a knob.**  Because the factor biases the *joint*
#: draw and the bias has a direction: PyXtal places orbits one at a time and
#: rejects against what is already down, so a higher factor under-samples dense
#: configurations, which is what real crystals are.  Relaxing it was untestable
#: while relaxers collapsed on overlapped input; NEP89's ZBL core does not (0
#: failures in 18,000 draws, 227 of them below 0.8 tolerance units), so
#: ``--pyxtal-tol-factor`` exists to measure it.  The default stays 1.3 until a
#: measurement says otherwise; ``docs/pyxtal_tolerance_sweep.md`` is that
#: measurement.
DEFAULT_PYXTAL_TOL_FACTOR = 1.3

#: Built ``Tol_matrix`` objects, keyed by factor.
#:
#: A ``Tol_matrix(prototype="atomic")`` walks the whole element table to fill a
#: 100x100 radius array, which is milliseconds -- irrelevant once, and not
#: irrelevant per draw in a pool that does tens of thousands of them.  The
#: objects are read-only as this package uses them (only ``get_tol``), so one
#: per factor is safe to share; the cache is per process, and the generate stage
#: forks ``spawn`` workers, so each worker builds its own on first use.
_IADM_CACHE: dict[float, Tol_matrix] = {}


def pyxtal_tol_matrix(factor: float = DEFAULT_PYXTAL_TOL_FACTOR) -> Tol_matrix:
    """The atomic-prototype tolerance matrix scaled by *factor*, cached.

    Args:
        factor: Multiplier on ``0.5 * (r_a + r_b)``; see
            :data:`DEFAULT_PYXTAL_TOL_FACTOR` for what the number means and why
            1.3 is the default.

    Returns:
        A ``Tol_matrix``.  The *same object* for the same factor, so callers
        must not mutate it.
    """
    # Keyed on the float itself rather than a rounded string: a factor comes
    # from the command line or a default, not from arithmetic, so there are no
    # near-identical keys to collapse, and rounding would silently serve the
    # wrong matrix to anyone who did pass one.
    key = float(factor)
    if key not in _IADM_CACHE:
        _IADM_CACHE[key] = Tol_matrix(prototype="atomic", factor=key)
    return _IADM_CACHE[key]


_DEFAULT_IADM = pyxtal_tol_matrix(DEFAULT_PYXTAL_TOL_FACTOR)

# Tolerance used to *reject* collapsed structures after relaxation, i.e. a floor
# at 0.55 * (r_a + r_b).  Set below the generation factor so that a real
# structure can never be discarded by it, which is measured rather than hoped:
# the 750 oracle references sit at 1.48 and above in tolerance units, so they
# clear this one by 35% at worst.  It still catches the energy-lowering MACE
# collapse artifacts, which sit comfortably below it.  (An earlier version of
# this comment justified the gap by real materials whose "short metallic
# contacts graze the generation floor" -- nothing in the reference set comes
# within 14% of that floor, so the gap is margin, not rescue.)
_CLASH_IADM = Tol_matrix(prototype="atomic", factor=1.1)

#: Top-level package of the MACE ASE calculators, used to recognise a MACE
#: calculator without importing MACE (an optional dependency, and the backend
#: environments are mutually exclusive -- see
#: :mod:`wyckoff_transformer.cryspr.mlips`).
_MACE_PACKAGE = "mace"


def is_mace_calculator(calculator: Calculator) -> bool:
    """Return ``True`` if *calculator* is backed by MACE.

    Matched on the calculator's class hierarchy rather than on a model name,
    which is a free-form label, and rather than by importing MACE, which need
    not be installed in the environment running a different backend.  The MRO
    is walked because :func:`~wyckoff_transformer.cryspr.calculator.build_mace_calculator`
    hands back a local subclass of ``MACECalculator``, so the concrete class is
    defined in this project, not in MACE.

    Args:
        calculator: The ASE calculator used for relaxation.

    Returns:
        ``True`` if any class in the MRO comes from the ``mace`` package.
    """
    for klass in type(calculator).__mro__:
        module = getattr(klass, "__module__", "") or ""
        if module == _MACE_PACKAGE or module.startswith(_MACE_PACKAGE + "."):
            return True
    return False


def has_atomic_clash(atoms: Atoms, iadm: Tol_matrix = _CLASH_IADM) -> bool:
    """Return ``True`` if any pair of atoms is closer than the tolerance.

    Each interatomic distance (including periodic images) is compared against
    the species-pair minimum from *iadm*.  A relaxed structure that has
    collapsed into overlapping atoms (a known artifact of MACE-MP-0's spurious
    short-range energy basins, which produce unphysically low energies) is thus
    rejected on a purely physical, geometry-based criterion.

    The default *iadm* (:data:`_CLASH_IADM`, factor 1.1) is more permissive than
    the generation tolerance (:data:`_DEFAULT_IADM`, factor 1.3).  Both are
    multiples of ``0.5 * (r_a + r_b)``, the covalent-radius mean; see
    :data:`_DEFAULT_IADM`.  The gap is margin rather than rescue: no structure
    in the oracle reference set comes within 14% of even the *generation* floor,
    so nothing real is at risk from either.

    The collapse this guards against is a MACE pathology, so :func:`func_run`
    engages the guard only for MACE calculators; this function itself is
    potential-agnostic and can be called on any structure.

    Args:
        atoms: Structure to check.
        iadm: PyXtal tolerance matrix defining per-species-pair minimum distances.

    Returns:
        ``True`` if at least one interatomic distance is below tolerance.
    """
    numbers = atoms.numbers
    if len(numbers) == 0:
        return False
    unique = sorted({int(n) for n in numbers})
    cutoff = max(iadm.get_tol(a, b) for a in unique for b in unique)
    if cutoff <= 0:
        return False
    i_idx, j_idx, dists = neighbor_list("ijd", atoms, cutoff)
    tols = np.array([iadm.get_tol(int(numbers[i]), int(numbers[j]))
                     for i, j in zip(i_idx, j_idx)])
    return bool(np.any(dists < tols))


#: Sub-directory of a trial holding the pre-relaxation's stage CIFs and logs.
#:
#: A directory rather than a filename prefix: ``stepwise_relax`` names its CIFs
#: after the formula and the stage, so a pre-relaxation writing into the trial
#: directory would collide with the scoring relaxation stage for stage, and the
#: kept-CIF glob would stop being unambiguous.
PRERELAX_DIR = "prerelax"

#: Force tolerance of the pre-relaxation, eV/A.  Looser than the protocol's own
#: 0.05: what the stage delivers is a starting geometry, and converging it
#: tightly on a potential whose minimum is not the one being scored just moves
#: the structure to the wrong stationary point more precisely.
PRERELAX_FMAX = 0.1

#: Per-trial record of what the pre-relaxation did, written next to its CIFs.
#:
#: The same shape of evidence as ``rattle.json``, and for the same reason: pool
#: workers configure no logging, and what the cheap potential did to a cell is
#: not recoverable from the scoring relaxation's output.  The volume ratio is
#: the number that matters.  A pre-relaxation is supposed to *contract* a loose
#: PyXtal draw towards contact; one that expands it hands the scoring potential
#: a symmetric stationary point in an inflated cell, which its own symmetric
#: stages cannot descend out of -- only the rattle can.  Measured on one
#: Ga4Ta6 draw during development: 28.5 -> 42.3 A^3/atom under NEP89, after
#: which ORB stayed at 44.6 and the rattle recovered 17.2 against the
#: single-stage arm's 16.8.
PRERELAX_VERDICT_FILE = "prerelax.json"


def prerelax(
        atoms_in: Atoms,
        calculator: Calculator,
        wdir: Path,
        label: str = "",
        fix_symmetry: bool = True,
        release_symmetry: bool = False,
        rattle: bool = False,
        seed: Optional[int] = None,
        fmax: float = PRERELAX_FMAX,
        optimizer: type[Optimizer] = BFGS,
        strict: bool = False,
        max_expansion: Optional[float] = None,
) -> Atoms:
    """Relax *atoms_in* on a cheap potential and return the geometry.

    The first half of a two-stage trial, and on its own the fixed-symmetry
    pre-screen the wide-then-narrow protocol variant runs over ten times the
    usual number of draws.  Only the geometry is returned; the caller that wants
    the energy reads it off the returned atoms, and no caller in this package
    lets it reach the funnel.

    A failure here is *not* fatal to a two-stage trial: the whole point of the
    stage is that the scoring relaxation can do the job without it, so a cheap
    potential that raises -- an element it has no parameters for, a cell it
    cannot build a neighbour list in -- costs the pre-relaxation and nothing
    else.  The unrelaxed input is returned and the reason is logged.  A caller
    that *ranks* on this stage's energy has the opposite need and passes
    ``strict=True``: an unrelaxed draw compared against relaxed ones is a
    silently wrong comparison, where a missing draw is a visible one.

    Args:
        atoms_in: Structure to pre-relax; not modified in place.
        calculator: The cheap ASE Calculator.
        wdir: Directory for the stage CIFs and optimiser logs.
        label: Prefix for the log lines.
        fix_symmetry: Run the symmetry-constrained stage.
        release_symmetry: Run the unconstrained stage.  Off by default.
        rattle: Run the rattle stage.  Off by default.
        seed: Seed for the rattle perturbation, if it runs.
        fmax: Force convergence criterion, eV/A.
        optimizer: ASE local optimisation algorithm class.
        strict: Re-raise a relaxation failure instead of returning *atoms_in*.
        max_expansion: Discard the result and return *atoms_in* if the cheap
            potential grew the cell by more than this factor.  ``None``, the
            default, keeps whatever it produced.  See
            :data:`PRERELAX_VERDICT_FILE` for what the guard is for: PyXtal
            draws are deliberately loose and the protocol relies on
            *compressive* relaxation, so a pre-relaxation that inflates the cell
            has moved away from the answer, and the raw draw is the better start.

    Returns:
        The pre-relaxed structure, or -- unless *strict* -- *atoms_in* unchanged
        if the stage failed.

    Raises:
        Exception: Whatever the relaxation raised, when *strict*.
    """
    wdir = Path(wdir)
    from wyckoff_transformer.cryspr.relaxer import _get_spacegroup_info

    volume_before = float(atoms_in.get_volume()) if atoms_in.cell.rank == 3 else None
    try:
        relaxed = stepwise_relax(
            atoms_in=atoms_in,
            calculator=calculator,
            optimizer=optimizer,
            fix_symmetry=fix_symmetry,
            release_symmetry=release_symmetry,
            rattle=rattle,
            seed=seed,
            fmax=fmax,
            wdir=wdir,
            logfile_prefix=atoms_in.get_chemical_formula(mode="metal"),
            logfile_postfix="prerelax",
        )
    except Exception as exc:
        # Broad on purpose, and re-raised when the caller ranks on this energy:
        # a cheap potential can fail in as many ways as it has backends, and a
        # two-stage trial loses nothing by falling back to the raw draw.
        if strict:
            raise
        logger.warning("[%s] Pre-relaxation failed (%s); using the raw draw", label, exc)
        return atoms_in
    volume_after = float(relaxed.get_volume()) if relaxed.cell.rank == 3 else None
    ratio = (
        volume_after / volume_before if volume_before and volume_after else None
    )
    rejected = bool(max_expansion is not None and ratio is not None and ratio > max_expansion)
    # Every value cast explicitly: see the note in relaxer._rattle_stage about
    # MLIPs that return numpy scalars.
    verdict = {
        "volume_before": None if volume_before is None else float(volume_before),
        "volume_after": None if volume_after is None else float(volume_after),
        "volume_ratio": None if ratio is None else float(ratio),
        "max_expansion": None if max_expansion is None else float(max_expansion),
        "rejected": bool(rejected),
        "n_atoms": len(relaxed),
        "spacegroup_before": int(_get_spacegroup_info(atoms_in, symprec=1e-3)[1]),
        "spacegroup_after": int(_get_spacegroup_info(relaxed, symprec=1e-3)[1]),
        "fmax": float(fmax),
        "fix_symmetry": bool(fix_symmetry),
        "release_symmetry": bool(release_symmetry),
        "rattle": bool(rattle),
    }
    try:
        (wdir / PRERELAX_VERDICT_FILE).write_text(
            json.dumps(verdict) + "\n", encoding="utf-8"
        )
    except OSError as exc:  # a diagnostic must never cost the trial
        logger.warning("[%s] Could not write %s (%s)", label, PRERELAX_VERDICT_FILE, exc)

    if rejected:
        logger.warning(
            "[%s] Pre-relaxation grew the cell %.2fx (limit %.2fx); using the raw draw",
            label, ratio, max_expansion,
        )
        return atoms_in

    # Detached from the cheap calculator: the caller hands this to a different
    # one, and an Atoms still holding the old calc would serve its cached energy.
    relaxed = relaxed.copy()
    relaxed.calc = None
    relaxed.set_constraint([])
    return relaxed


#: Suffix of the CIF holding the structure a trial actually kept.  The rattle
#: stage is accepted on energy, so which stage's CIF is the final one is not
#: known until it has run; this one always is.
KEPT_CIF_SUFFIX = "_kept.cif"

#: Suffix of the CIF holding the structure the rattle stage was *handed*.
#:
#: Written alongside the kept one, always, because the rattle is not free of
#: consequence beyond energy and the protocol reports both readouts.  A rattle
#: is a finite symmetry-breaking perturbation: the structure it leaves is off
#: the Wyckoff orbits the gene specified -- which is what WyFormer predicts and
#: what the whole symmetry-by-design argument rests on -- and it can relax onto
#: a LeMat-Bulk entry the unrattled structure was distinct from, turning a novel
#: structure into a known one.  Neither effect is visible from the kept
#: structure alone.  Identical to the kept CIF when the rattle did not run or
#: did not win its margin.
PRERATTLE_CIF_SUFFIX = "_prerattle.cif"


def _trial_seed(id_gene: int | str, i_trial: int) -> int:
    """A stable seed per (gene, trial), so a rerun rattles the same way.

    Hashed rather than derived arithmetically: the rattle of gene 1 trial 2 and
    that of gene 2 trial 1 should be independent draws.
    """
    digest = hashlib.sha256(f"{id_gene}_{i_trial}".encode()).hexdigest()
    return int(digest[:8], 16)


def single_pyxtal(
        wyckoffgene: dict,
        iadm: Tol_matrix = _DEFAULT_IADM,
        nlimit: int = 20,
        wdir: Path = Path("."),
) -> Optional[Atoms]:
    """Generate a random crystal from a Wyckoff gene using PyXtal.

    Args:
        wyckoffgene: Dict with keys ``group`` (int), ``species`` (list of str),
            ``numIons`` (list of int), and ``sites`` (list of list of str).
        iadm: PyXtal tolerance matrix controlling minimum inter-atomic distances.
        nlimit: Maximum number of PyXtal generation attempts.
        wdir: Directory in which to save the raw generated CIF.

    Returns:
        An :class:`~ase.Atoms` object, or ``None`` if generation failed.
    """
    spg: int = wyckoffgene["group"]
    species: list[str] = wyckoffgene["species"]
    num_ions: list[int] = wyckoffgene["numIons"]
    sites: list[list] = wyckoffgene["sites"]

    try:
        candidate = pyxtal()
        candidate.from_random(
            dim=3,
            group=spg,
            species=species,
            numIons=num_ions,
            sites=sites,
            tm=iadm,
            max_count=nlimit,
        )
        atoms: Atoms = candidate.to_ase()
        formula = atoms.get_chemical_formula(mode="metal")
        wdir = Path(wdir)
        wdir.mkdir(parents=True, exist_ok=True)
        candidate.to_file(str(wdir / f"pyxtal_generated_{formula}.cif"))
        return atoms
    except Exception as exc:
        logger.warning("PyXtal generation failed: %s", exc)
        return None


def relax_trial(
        atoms_in: Atoms,
        calculator: Calculator,
        trial_dir: Path,
        label: str = "",
        fix_symmetry: bool = True,
        release_symmetry: bool = True,
        rattle: bool = True,
        rattle_stdev: float = RATTLE_STDEV,
        strain_stdev: float = RATTLE_STRAIN_STDEV,
        rattle_accept: float = RATTLE_ACCEPT_EV_PER_ATOM,
        clash_guard: bool = False,
        seed: Optional[int] = None,
        fmax: float = 0.01,
        optimizer: type[Optimizer] = BFGS,
        prerelax_calculator: Optional[Calculator] = None,
        prerelax_fmax: float = PRERELAX_FMAX,
        prerelax_release_symmetry: bool = False,
        prerelax_rattle: bool = False,
        prerelax_max_expansion: Optional[float] = None,
) -> tuple[Optional[Atoms], Optional[float]]:
    """Relax one PyXtal draw and write the CIF the relaxation kept.

    The relaxation half of :func:`func_run`, for one trial and with no PyXtal
    call of its own, so that a caller which generates structures on the CPU can
    hand them to a separate process holding the MLIP on a GPU.

    With *prerelax_calculator* the trial becomes two-stage: the same
    :func:`~wyckoff_transformer.cryspr.relaxer.stepwise_relax` schedule is run
    first on the cheap potential, into ``trial_dir / "prerelax"``, and the main
    relaxation starts from its output rather than from the raw draw.  The
    pre-relaxation's energy is never returned, so it cannot reach the funnel;
    only its geometry survives.

    Args:
        atoms_in: The generated structure to relax.
        calculator: Shared ASE Calculator.
        trial_dir: Directory for this trial's stage CIFs, logs and kept CIF.
        label: Prefix for the log lines, e.g. ``"orb-17 trial-0"``.
        fix_symmetry: Run the symmetry-constrained step.
        release_symmetry: Run the unconstrained step before the rattle.
        rattle: Run the rattle stage.
        rattle_stdev: Per-atom displacement of the perturbation, A.
        strain_stdev: Cell strain of the perturbation, dimensionless.
        rattle_accept: Energy the rattle must win to be kept, eV/atom.
        clash_guard: Discard a relaxed structure whose atoms have collapsed
            into each other (:func:`has_atomic_clash`).  A MACE pathology; see
            :func:`func_run`, which resolves the default.
        seed: Seed for the rattle perturbation.
        fmax: Force convergence criterion in eV/A.
        optimizer: ASE local optimisation algorithm class.
        prerelax_calculator: Cheap potential to relax with first.  ``None``, the
            default, is the single-stage protocol.
        prerelax_fmax: Force tolerance of the pre-relaxation, eV/A.  Looser than
            *fmax* on purpose: the pre-relaxation is a starting point, and the
            last fraction of an eV/A on a potential that is not the one being
            scored buys nothing.
        prerelax_release_symmetry: Run the unconstrained stage in the
            pre-relaxation.  Off by default, which is what keeps the
            pre-relaxation from deciding the symmetry question: a structure that
            left its space group on the cheap potential would arrive at the
            scoring one with its symmetric stages already spent.
        prerelax_rattle: Run the rattle stage in the pre-relaxation.  Off by
            default, for the same reason.
        prerelax_max_expansion: Fall back to the raw draw if the cheap potential
            grew the cell by more than this factor.  ``None`` keeps whatever it
            produced; see :func:`prerelax`.

    Returns:
        ``(atoms, energy, prerattle)``, where *prerattle* is a
        ``(atoms, energy)`` pair for the structure the rattle stage was handed
        -- equal to the kept one when the rattle did not run or did not win.
        ``(None, None, None)`` when the clash guard rejected the relaxed
        structure.

    Raises:
        Exception: Whatever the relaxation raised.  Unlike :func:`func_run`,
            which has other trials to fall back on, this reports the failure to
            its caller rather than logging it.
    """
    trial_dir = Path(trial_dir)
    trial_dir.mkdir(parents=True, exist_ok=True)
    formula = atoms_in.get_chemical_formula(mode="metal")
    logger.info("[%s] Starting relaxation", label)

    if prerelax_calculator is not None:
        atoms_in = prerelax(
            atoms_in=atoms_in,
            calculator=prerelax_calculator,
            wdir=trial_dir / PRERELAX_DIR,
            label=label,
            fix_symmetry=fix_symmetry,
            release_symmetry=prerelax_release_symmetry,
            rattle=prerelax_rattle,
            seed=seed,
            fmax=prerelax_fmax,
            optimizer=optimizer,
            max_expansion=prerelax_max_expansion,
        )

    stages = stepwise_relax_stages(
        atoms_in=atoms_in,
        calculator=calculator,
        optimizer=optimizer,
        fix_symmetry=fix_symmetry,
        release_symmetry=release_symmetry,
        rattle=rattle,
        rattle_stdev=rattle_stdev,
        strain_stdev=strain_stdev,
        rattle_accept=rattle_accept,
        seed=seed,
        fmax=fmax,
        wdir=trial_dir,
        logfile_prefix=formula,
        logfile_postfix="relax",
    )
    atoms_relaxed = stages.kept
    energy = atoms_relaxed.get_potential_energy()
    if clash_guard and has_atomic_clash(atoms_relaxed):
        logger.warning(
            "[%s] Relaxed structure has atomic clashes (E = %.5f eV); "
            "discarding as unphysical.", label, energy,
        )
        return None, None, None

    # The kept structure, whichever stage produced it.
    write(
        filename=str(trial_dir / f"{formula}{KEPT_CIF_SUFFIX}"),
        images=atoms_relaxed,
        format="cif",
    )
    # And the one the rattle was handed, so the metrics can be reported both
    # ways.  Written unconditionally, including when it is the same structure:
    # a missing file and an unrattled trial would otherwise be indistinguishable
    # downstream, which is exactly the ambiguity this pair exists to remove.
    prerattle_energy = stages.prerattle.get_potential_energy()
    write(
        filename=str(trial_dir / f"{formula}{PRERATTLE_CIF_SUFFIX}"),
        images=stages.prerattle,
        format="cif",
    )
    logger.info(
        "[%s] Done, E = %.5f eV (pre-rattle %.5f eV%s)",
        label, energy, prerattle_energy,
        ", rattle accepted" if stages.rattle_accepted else "",
    )
    return atoms_relaxed, energy, (stages.prerattle, prerattle_energy)


def func_run(
        id_gene: int | str,
        wyckoffgene: dict,
        calculator: Calculator,
        output_dir: Path = Path("."),
        model_name: str = "model",
        n_trials: int = 6,
        fix_symmetry: bool = True,
        release_symmetry: bool = True,
        rattle: bool = True,
        rattle_stdev: float = RATTLE_STDEV,
        strain_stdev: float = RATTLE_STRAIN_STDEV,
        rattle_accept: float = RATTLE_ACCEPT_EV_PER_ATOM,
        clash_guard: Optional[bool] = None,
        fmax: float = 0.01,
        optimizer: type[Optimizer] = BFGS,
) -> tuple[Optional[Atoms], Optional[str], Optional[float], Optional[float], Optional[str]]:
    """Generate and relax crystal structures for one Wyckoff gene.

    Runs *n_trials* independent PyXtal generation + MLIP relaxation cycles.
    The trial with the lowest final energy is returned.  All trial directories
    and output CIF files are written under ``output_dir / str(id_gene) /``.

    Args:
        id_gene: Identifier for this Wyckoff gene, used as a sub-directory name.
        wyckoffgene: Wyckoff gene dictionary (see :func:`single_pyxtal`).
        calculator: Shared ASE Calculator instance (not recreated per trial).
        output_dir: Root directory for all output files.
        model_name: Label used in log messages and output filenames.
        n_trials: Number of random generation + relaxation trials.
        fix_symmetry: Run the symmetry-constrained step of the relaxation.
        release_symmetry: Run the unconstrained step before the rattle.
            ``False`` drops it; see
            :func:`~wyckoff_transformer.cryspr.relaxer.stepwise_relax`.
        rattle: Run the rattle stage, which is the only one that can leave a
            symmetric stationary point.
        rattle_stdev: Per-atom displacement of the perturbation, Å.
        strain_stdev: Cell strain of the perturbation, dimensionless.
        rattle_accept: Energy the rattle must win to be kept, eV/atom.
        clash_guard: Discard a relaxed trial whose atoms have collapsed into
            each other (:func:`has_atomic_clash`).  ``None``, the default,
            engages the guard for MACE calculators only, which is where the
            collapse comes from; ``True`` or ``False`` forces it either way.
        fmax: Force convergence criterion in eV/Å.
        optimizer: ASE local optimisation algorithm class.

    Returns:
        Tuple ``(atoms, formula, energy, energy_per_atom, cif)`` for the
        lowest-energy successful trial, where *cif* is the text of the
        structure that trial kept.
        Returns ``(None, None, None, None, None)`` when all trials fail.
    """
    output_dir = Path(output_dir)
    gene_dir = output_dir / str(id_gene)
    gene_dir.mkdir(parents=True, exist_ok=True)

    # Resolved once: the calculator is shared across trials, and an unguarded
    # run should say so in the log rather than be inferred from its absence.
    if clash_guard is None:
        clash_guard = is_mace_calculator(calculator)
        logger.debug(
            "[%s-%s] Clash guard %s (auto: calculator %s MACE-backed)",
            model_name, id_gene,
            "on" if clash_guard else "off",
            "is" if clash_guard else "is not",
        )

    atoms_by_trial: dict[str, Atoms] = {}
    energy_by_trial: dict[str, float] = {}
    formula: Optional[str] = None

    for i_trial in range(n_trials):
        trial_key = f"trial-{i_trial}"
        trial_dir = gene_dir / trial_key
        trial_dir.mkdir(parents=True, exist_ok=True)

        atoms_in = single_pyxtal(wyckoffgene=wyckoffgene, nlimit=30, wdir=trial_dir)
        if atoms_in is None:
            continue

        formula = atoms_in.get_chemical_formula(mode="metal")

        try:
            atoms_relaxed, energy, _prerattle = relax_trial(
                atoms_in=atoms_in,
                calculator=calculator,
                trial_dir=trial_dir,
                label=f"{model_name}-{id_gene} {trial_key}",
                fix_symmetry=fix_symmetry,
                release_symmetry=release_symmetry,
                rattle=rattle,
                rattle_stdev=rattle_stdev,
                strain_stdev=strain_stdev,
                rattle_accept=rattle_accept,
                clash_guard=clash_guard,
                seed=_trial_seed(id_gene, i_trial),
                fmax=fmax,
                optimizer=optimizer,
            )
        except Exception as exc:
            logger.warning("[%s-%s %s] Relaxation failed: %s", model_name, id_gene, trial_key, exc)
            continue
        if atoms_relaxed is None:  # the clash guard rejected it
            continue
        atoms_by_trial[trial_key] = atoms_relaxed
        energy_by_trial[trial_key] = energy

    if not atoms_by_trial:
        logger.warning(
            "[%s-%s] All %d trials failed or produced no structure.",
            model_name, id_gene, n_trials,
        )
        return None, None, None, None, None

    lowest_key = min(energy_by_trial, key=energy_by_trial.__getitem__)

    symlink_lowest = gene_dir / "trial-lowest"
    if not symlink_lowest.exists():
        os.symlink(lowest_key, symlink_lowest, target_is_directory=True)

    lowest_dir = gene_dir / lowest_key
    # The kept CIF, not a stage CIF: with the rattle stage the last stage to
    # run is not necessarily the one whose structure was kept.
    final_cifs = sorted(lowest_dir.glob(f"*{KEPT_CIF_SUFFIX}"))
    if final_cifs:
        symlink_cif = gene_dir / "min_e_strc.cif"
        if not symlink_cif.exists():
            # Relative symlink so it survives directory moves
            cif_rel = Path("trial-lowest") / final_cifs[0].name
            os.symlink(cif_rel, symlink_cif)

    atoms = atoms_by_trial[lowest_key]
    energy = energy_by_trial[lowest_key]
    energy_per_atom = energy / len(atoms)

    cif_content: Optional[str] = final_cifs[0].read_text() if final_cifs else None

    return atoms, formula, energy, energy_per_atom, cif_content
