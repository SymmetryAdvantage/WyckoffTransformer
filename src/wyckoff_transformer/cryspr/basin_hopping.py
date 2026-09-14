"""Symmetry-constrained basin hopping on a cheap potential.

The protocol's default search over a gene's continuous degrees of freedom is
*independent draws*: PyXtal proposes a cell and a set of free coordinates from
scratch, one to three times, and each proposal is relaxed on its own.  That is
an unbiased sampler and a poor searcher -- every draw forgets what the last one
found, so a gene with many free coordinates gets a handful of random points in a
space whose dimension grows with them.  The wide-then-narrow variant answers
this with more draws; this module answers it with a *walk*.

Basin hopping alternates a finite perturbation with a local relaxation and
accepts or rejects on energy, so the search moves between adjacent minima
instead of restarting.  What makes it usable here is that the perturbation is
:func:`~wyckoff_transformer.cryspr.relaxer.symmetric_perturb`, which projects
the step onto the space group's own subspace: every structure the walk visits
sits on the Wyckoff orbits the gene specified.  The walk therefore searches
exactly the space WyFormer's prediction defines -- the free coordinates and the
allowed lattice parameters -- and nothing outside it.

That constraint is the point, not a limitation.  An unconstrained walk would
wander off the gene, and a structure off the gene is not a test of the gene.
The consequence is that a gene with **zero positional degrees of freedom** has
nothing to walk through: the projected displacement vanishes and only the cell
can move, which stages 1 and 2 of the relaxation already do.  The measured
distinct-minima counts say the same thing (1.1 to 1.3 per gene at 0 DoF from ten
independent draws), so this is a property of the gene rather than of the
searcher.

Energies come from the cheap potential and are used only to order and select.
Nothing computed here reaches the funnel; see
:mod:`wyckoff_transformer.cryspr.nep89`.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.filters import FrechetCellFilter as CellFilter
from ase.optimize import BFGS
from ase.optimize.optimize import Optimizer

from wyckoff_transformer.cryspr.relaxer import (
    SYMMETRIC_CIF_LABEL,
    WARMUP_CIF_LABEL,
    _get_spacegroup_info,
    run_ase_relaxer,
    symmetric_perturb,
)

logger = logging.getLogger(__name__)

#: Hops per starting structure.  Each one costs a local relaxation on the cheap
#: potential -- ~1 s for the cells this sees -- so twenty is seconds per trial
#: against the seconds-to-minutes a single scoring relaxation costs.
DEFAULT_STEPS = 20

#: Metropolis temperature, eV/atom.  Roughly room temperature, which is the
#: conventional choice: a walk that only ever goes downhill is a greedy descent
#: and stalls in the first basin it finds, while one that accepts everything is
#: a random sampler with extra steps.  ``0`` makes it monotonic.
DEFAULT_TEMPERATURE_EV_PER_ATOM = 0.02

#: Energy within which two visited minima are treated as the same one while the
#: walk runs, eV/atom.  A cheap pre-filter only: the caller deduplicates what it
#: keeps with ``StructureMatcher``, which is the verdict that counts.  1 meV/atom
#: is the rattle stage's own acceptance margin, i.e. the repository's standing
#: answer to "closer than this is not a different answer".
SAME_MINIMUM_EV_PER_ATOM = 1e-3

#: Displacement drawn per hop, A, *before* the projection onto the space group's
#: subspace shrinks it to about 0.68 of this.
#:
#: Deliberately six times the rattle stage's 0.05 A, which was calibrated to
#: *break* symmetry rather than to cross a barrier.  Measured over five
#: high-DoF oracle draws at 15 hops each: 0.05 A finds 2.0 minima spanning
#: 1 meV/atom -- one basin, found repeatedly -- while 0.15 A and above find 2.4
#: to 2.6 spanning 57 meV/atom, and the count saturates there.  0.3 A sits in
#: the saturated region with room to spare, and the space group survived every
#: hop at every step size tried.
BASINHOP_STDEV = 0.3

#: Cell strain drawn per hop, dimensionless.  Three times the rattle stage's,
#: for the same reason: a hop has to be able to change the cell enough to reach
#: a neighbouring basin, not merely to perturb the one it is in.
BASINHOP_STRAIN_STDEV = 0.03


@dataclass
class Minimum:
    """One relaxed structure the walk visited.

    Attributes:
        atoms: The relaxed structure, with no calculator attached.
        energy: Total energy on the cheap potential, eV.
        energy_per_atom: The same, per atom.
        spacegroup: spglib's number for it, so that the symmetry claim is
            measured at every step rather than assumed once.
        step: Which hop found it; 0 is the initial relaxation.
        accepted: Whether the walk moved to it.  A rejected minimum is still a
            minimum, and still a candidate worth relaxing on the scoring
            potential -- the acceptance test governs where the *walk* goes, not
            what is worth keeping.
    """

    atoms: Atoms
    energy: float
    energy_per_atom: float
    spacegroup: int
    step: int
    accepted: bool


@dataclass
class BasinHopResult:
    """What one walk found.

    Attributes:
        minima: Distinct minima, ascending in energy per atom.
        n_steps: Hops attempted.
        n_accepted: Hops the Metropolis test accepted.
        n_symmetry_lost: Hops whose relaxation *lost* symmetry operations the
            starting structure had.  The projection makes this impossible in
            principle, so a non-zero count means the symmetry tolerance is doing
            something unexpected and the run's premise needs checking.
        n_symmetry_gained: Hops whose relaxation found *more* operations than it
            started with, i.e. converged onto a supergroup.  Kept, not rejected:
            a structure with the operations of a supergroup still has the
            starting group's own operations, so it still satisfies the gene --
            it merely acquired accidental extra symmetry, usually while lowering
            its energy.  Measured at 122 of 123 changes over 36000 hops of the
            oracle cohort, against a single genuine loss, which is why rejecting
            on "the space group number changed" threw away the good case.
        n_failed: Hops whose relaxation raised.
    """

    minima: list[Minimum] = field(default_factory=list)
    n_steps: int = 0
    n_accepted: int = 0
    n_symmetry_lost: int = 0
    n_symmetry_gained: int = 0
    n_failed: int = 0

    @property
    def best(self) -> Optional[Minimum]:
        return self.minima[0] if self.minima else None


def symmetric_basin_hop(
    atoms_in: Atoms,
    calculator: Calculator,
    n_steps: int = DEFAULT_STEPS,
    temperature: float = DEFAULT_TEMPERATURE_EV_PER_ATOM,
    rattle_stdev: float = BASINHOP_STDEV,
    strain_stdev: float = BASINHOP_STRAIN_STDEV,
    symprec: float = 1e-3,
    fmax: float = 0.1,
    steps_limit: int = 500,
    wdir: Path = Path("."),
    seed: Optional[int] = None,
    optimizer: type[Optimizer] = BFGS,
    same_minimum: float = SAME_MINIMUM_EV_PER_ATOM,
) -> BasinHopResult:
    """Walk between the symmetry-preserving minima reachable from *atoms_in*.

    The schedule per hop is: project a random step onto the space group's
    subspace, relax under a :class:`~ase.constraints.FixSymmetry` constraint,
    then accept or reject on energy.  The starting structure is itself relaxed
    the same way first, so the walk begins at a minimum rather than at a draw.

    Every minimum visited is collected, accepted or not: the acceptance test
    decides where the walk goes next, while the caller is selecting *candidates*
    to spend the scoring potential on and a rejected basin is as good a candidate
    as an accepted one.

    Args:
        atoms_in: Starting structure, normally a PyXtal draw.
        calculator: The cheap ASE Calculator.
        n_steps: Hops to attempt.
        temperature: Metropolis temperature, eV/atom.  ``0`` accepts only
            downhill moves.
        rattle_stdev: Cartesian displacement drawn before projection, A.
        strain_stdev: Cell strain drawn before projection, dimensionless.
        symprec: Symmetry tolerance, A.
        fmax: Force convergence of each local relaxation, eV/A.
        steps_limit: Optimiser step cap per local relaxation.
        wdir: Directory for the stage CIF and the optimiser log.  Each hop
            overwrites the previous one's CIF rather than adding to it, so a
            walk costs a constant number of files.
        seed: Seed for the walk.  Every perturbation and every Metropolis draw
            comes from it, so a run is reproducible.
        optimizer: ASE local optimisation algorithm class.
        same_minimum: Energy within which two visited minima are treated as one
            while the walk runs, eV/atom.

    Returns:
        A :class:`BasinHopResult`.  Empty ``minima`` means even the initial
        relaxation failed.
    """
    wdir = Path(wdir)
    wdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    result = BasinHopResult(n_steps=n_steps)

    start_group = _get_spacegroup_info(atoms_in, symprec=symprec)[1]
    start_operations = _n_symmetry_operations(atoms_in, symprec)
    current = _relax_symmetric(
        atoms_in, calculator, symprec, fmax, steps_limit, wdir,
        optimizer, warmup=True,
    )
    if current is None:
        logger.warning("Basin hop: the initial relaxation failed; nothing to walk from")
        return result
    current_energy = float(current.get_potential_energy())
    _record(result, current, current_energy, symprec, step=0, accepted=True,
            same_minimum=same_minimum)

    for step in range(1, n_steps + 1):
        proposal = symmetric_perturb(
            current,
            rattle_stdev=rattle_stdev,
            strain_stdev=strain_stdev,
            symprec=symprec,
            seed=int(rng.integers(1, 2 ** 31 - 1)),
        )
        relaxed = _relax_symmetric(
            proposal, calculator, symprec, fmax, steps_limit, wdir, optimizer,
            warmup=False,
        )
        if relaxed is None:
            result.n_failed += 1
            continue
        energy = float(relaxed.get_potential_energy())
        group = _get_spacegroup_info(relaxed, symprec=symprec)[1]
        operations = _n_symmetry_operations(relaxed, symprec)
        if group != start_group and start_operations and operations:
            # Count operations, not group numbers.  A relaxation that converges
            # onto a *supergroup* still has every operation it started with, so
            # the structure still satisfies the gene -- it has merely acquired
            # accidental extra symmetry, and usually lowered its energy doing
            # so.  Rejecting on "the number changed" discarded exactly that
            # case: 122 of 123 changes over 36000 oracle hops were increases.
            # Only a genuine *loss* means the walk left the gene's orbits, and
            # the projection makes that impossible in principle.
            if operations < start_operations:
                result.n_symmetry_lost += 1
                logger.warning(
                    "Basin hop step %d lost symmetry: group %d (%d ops) -> %d "
                    "(%d ops); rejecting",
                    step, start_group, start_operations, group, operations,
                )
                continue
            result.n_symmetry_gained += 1
            logger.info(
                "Basin hop step %d gained symmetry: group %d (%d ops) -> %d "
                "(%d ops); keeping",
                step, start_group, start_operations, group, operations,
            )

        delta = (energy - current_energy) / len(relaxed)
        accepted = _accept(delta, temperature, rng)
        _record(result, relaxed, energy, symprec, step=step, accepted=accepted,
                same_minimum=same_minimum)
        if accepted:
            result.n_accepted += 1
            current, current_energy = relaxed, energy

    result.minima.sort(key=lambda m: m.energy_per_atom)
    logger.info(
        "Basin hop: %d/%d hops accepted, %d distinct minima, best %.4f eV/atom "
        "(%d failed, %d lost symmetry, %d gained it)",
        result.n_accepted, result.n_steps, len(result.minima),
        result.minima[0].energy_per_atom if result.minima else float("nan"),
        result.n_failed, result.n_symmetry_lost, result.n_symmetry_gained,
    )
    return result


def _n_symmetry_operations(atoms: Atoms, symprec: float) -> int:
    """How many symmetry operations spglib finds, or 0 if it cannot say.

    The order of the space group, which is what makes "did this structure lose
    symmetry?" answerable without group-subgroup tables: a supergroup has more
    operations and contains all of the original ones, a broken symmetry has
    fewer.  Comparing group *numbers* cannot distinguish the two.
    """
    import spglib

    try:
        dataset = spglib.get_symmetry_dataset(
            (atoms.cell.array, atoms.get_scaled_positions(), atoms.numbers),
            symprec=symprec,
        )
    except spglib.SpglibError:
        return 0
    if dataset is None:
        return 0
    rotations = getattr(dataset, "rotations", None)
    if rotations is None:
        rotations = dataset.get("rotations") if hasattr(dataset, "get") else None
    return 0 if rotations is None else int(len(rotations))


def _accept(delta_per_atom: float, temperature: float, rng) -> bool:
    """Metropolis on energy per atom, so the criterion is size-independent."""
    if delta_per_atom < 0:
        return True
    if temperature <= 0:
        return False
    return bool(rng.random() < math.exp(-delta_per_atom / temperature))


def _relax_symmetric(
    atoms: Atoms,
    calculator: Calculator,
    symprec: float,
    fmax: float,
    steps_limit: int,
    wdir: Path,
    optimizer: type[Optimizer],
    warmup: bool,
) -> Optional[Atoms]:
    """One symmetry-constrained local relaxation, or ``None`` if it raised.

    *warmup* adds the fix-cell pass the protocol runs before releasing the cell,
    which matters only for the first relaxation of a raw PyXtal draw: a loose
    cell dragged around by badly placed atoms is what that stage exists for, and
    a structure already at a minimum does not need it again.
    """
    shared = dict(
        calculator=calculator,
        optimizer=optimizer,
        symprec=symprec,
        fmax=fmax,
        steps_limit=steps_limit,
        wdir=wdir,
        hydrostatic_strain=False,
    )
    try:
        if warmup:
            atoms = run_ase_relaxer(
                atoms_in=atoms, fix_symmetry=True, cell_filter=None,
                label=WARMUP_CIF_LABEL, logfile=wdir / "basinhop_fix-cell.log",
                **shared,
            )
        return run_ase_relaxer(
            atoms_in=atoms, fix_symmetry=True, cell_filter=CellFilter,
            label=SYMMETRIC_CIF_LABEL, logfile=wdir / "basinhop_sym.log",
            **shared,
        )
    except Exception as exc:  # noqa: BLE001 - one bad hop must not end the walk
        logger.warning("Basin hop: a local relaxation failed (%s)", exc)
        return None


def _record(
    result: BasinHopResult,
    atoms: Atoms,
    energy: float,
    symprec: float,
    step: int,
    accepted: bool,
    same_minimum: float,
) -> None:
    """Add this minimum unless one at the same energy is already held."""
    per_atom = energy / len(atoms)
    for known in result.minima:
        if abs(known.energy_per_atom - per_atom) <= same_minimum:
            return
    kept = atoms.copy()
    kept.calc = None
    kept.set_constraint([])
    result.minima.append(
        Minimum(
            atoms=kept,
            energy=energy,
            energy_per_atom=per_atom,
            spacegroup=_get_spacegroup_info(atoms, symprec=symprec)[1],
            step=step,
            accepted=accepted,
        )
    )
