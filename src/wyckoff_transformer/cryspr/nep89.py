"""NEP89 as a cheap pre-relaxation potential, with a Lennard-Jones fallback.

NEP89 (`arXiv:2504.21286 <https://arxiv.org/abs/2504.21286>`_) is a
neuroevolution potential fitted across 89 elements.  It is two to three orders
of magnitude cheaper per force call than the graph-network foundation models the
ranking protocol scores with -- measured here, 0.8 ms for a 16-atom cell on one
CPU thread against tens to hundreds of ms for ORB -- which is what makes it
usable as a *pre*-relaxation: a stage that moves a PyXtal draw most of the way
to a minimum before the expensive potential is asked for anything.

It is deliberately **not** registered as a scoring potential.  ``e_above_hull``
is only meaningful when the structure's energy and the hull come from the same
model (see :mod:`wyckoff_transformer.evaluation.hull_mlips`), LeMat-Bulk
publishes no NEP89 hull, and NEP89's training data mixes functionals with
per-dataset energy shifts fitted during training.  Every NEP89 energy in this
package is therefore an ordering device inside one trial, never a number that
reaches the funnel.

Coverage and the fallback
-------------------------
The 89 elements are read from the model file's own header rather than hard-coded
(:func:`nep89_elements`): H through Bi plus Ac, Th, Pa, U, Np, Pu.  Within that
range Po, At, Rn, Fr and Ra are absent, as is everything above Pu.  A structure
containing any of those cannot be evaluated at all -- ``calorine`` raises rather
than extrapolating -- so :class:`Nep89WithFallback` falls back to a
species-aware Lennard-Jones potential (:class:`SpeciesLennardJones`) for the
whole structure.  Falling back per *structure* and not per *atom* is forced by
the physics: NEP89 is a many-body potential over a neighbourhood, so there is no
well-defined way to evaluate it on part of a cell.

The fallback is :class:`ScreenedMorse`: ZBL screened nuclear repulsion at short
range, blended into a Morse well at the covalent contact distance.  The ZBL core
is not an analogy -- it is the *same* function NEP89 itself uses below 1 A,
reproduced to within 0.1-0.8% on the shipped model -- so an overlapped pair is
pushed apart by the same physics whether or not NEP89 covers its chemistry.  The
Morse half supplies the cohesion ZBL lacks, since a variable-cell relaxation
under a monotonically repulsive potential would expand without bound and the
pre-relaxation's job is to *contract* a loose draw towards contact.

This replaced a species-aware Lennard-Jones, whose ``r^-12`` core reached 1.4e16
eV at 0.1 A for Si-Si against ZBL's physically correct 1.4e4 -- a magnitude that
overflows float32 and sends a quasi-Newton optimiser on an arbitrary step, on
exactly the crowded draws the fallback exists to handle.  45% of PyXtal draws
sit below the nominal generation floor and the worst measured contact was 0.19
of the covalent-radius sum, so that region is routine, not hypothetical.

The fallback remains a geometry regulariser, not a model of anything: its
energies are meaningless and its per-trial ordering close to meaningless, so a
fallback structure gains nothing from the pre-relaxation except a sane starting
geometry -- and loses nothing, because the scoring potential relaxes it
afterwards regardless.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.data import atomic_numbers, covalent_radii
from ase.stress import full_3x3_to_voigt_6_stress

logger = logging.getLogger(__name__)

#: The NEP89 model file, pinned by commit rather than by branch.
#:
#: GPUMD ships the weights as a plain-text NEP file in its own repository, and
#: ``master`` is a moving target: pinning the commit is what makes a reported
#: number reproducible, exactly as ``orb-models``'s date-stamped URL literal
#: does for the ORB checkpoint.  ``nep89_20250409`` is the release that
#: accompanies the paper and the GPUMD 4.0 distribution.
NEP89_MODEL_URL = (
    "https://raw.githubusercontent.com/brucefan1983/GPUMD/"
    "ca78f110d7aed5b7f7ee8e87b571eac2ce5fedce/"
    "potentials/nep/nep89_20250409/nep89_20250409.txt"
)

#: SHA-256 of the file at :data:`NEP89_MODEL_URL`, checked after download.
#:
#: A URL pin fixes *which* file is asked for; the digest is what makes the
#: answer verifiable.  Without it a silently re-uploaded blob would change every
#: pre-relaxation in the protocol and nothing in the manifest.
NEP89_MODEL_SHA256 = "75168ece02e840e4a32644f982b78d43cba697f5b64b4c8134ab66c7a8c28be1"

#: Human-readable identity of the weights, for ``manifest.json``.
NEP89_CHECKPOINT = "nep89_20250409 (GPUMD 4.0, arXiv:2504.21286)"

#: Cache for downloaded NEP models, beside the MACE one.
NEP_CACHE_DIR = Path.home() / ".cache" / "wyckoff_transformer" / "nep_models"


def resolve_nep89_model(
    model: Optional[Union[str, Path]] = None,
    expected_sha256: Optional[str] = NEP89_MODEL_SHA256,
) -> Path:
    """Return a local path to the NEP89 model file, downloading it if needed.

    Args:
        model: Override the pinned URL with a local path or another URL.  A
            local path is returned as given and is *not* digest-checked: a
            hand-built or fine-tuned model is a deliberate choice, and refusing
            it would make the override useless.
        expected_sha256: Digest the downloaded file must have.  ``None`` skips
            the check.

    Returns:
        Path to the local copy.

    Raises:
        FileNotFoundError: If an explicit local path does not exist.
        ValueError: If a downloaded file's digest does not match.
    """
    from .calculator import resolve_model_path

    if model is None:
        return resolve_model_path(
            NEP89_MODEL_URL,
            cache_dir=NEP_CACHE_DIR,
            expected_sha256=expected_sha256,
        )
    text = str(model)
    if text.startswith(("http://", "https://")):
        # A different URL is a different model; we have no digest for it.
        return resolve_model_path(text, cache_dir=NEP_CACHE_DIR)
    path = Path(model)
    if not path.is_file():
        raise FileNotFoundError(f"No NEP model file at {path}")
    return path


def nep89_elements(model: Optional[Union[str, Path]] = None) -> frozenset[str]:
    """Element symbols the NEP model at *model* was fitted for.

    Read from the model file's first line, which NEP writes as
    ``<model_type> <n_types> <symbol> ...``.  Parsed here rather than taken from
    ``calorine`` so that the coverage question can be answered -- and the
    fallback decided -- without importing an optional dependency, and rather
    than hard-coded so that a fine-tuned model with a different element set
    reports its own.

    Args:
        model: Model file, or ``None`` for the pinned NEP89 one (downloaded if
            it is not cached).

    Returns:
        The supported symbols.

    Raises:
        ValueError: If the header does not carry the number of types it claims.
    """
    path = resolve_nep89_model(model)
    with path.open("r", encoding="utf-8") as handle:
        header = handle.readline().split()
    if len(header) < 2:
        raise ValueError(f"{path}: first line is not a NEP header: {' '.join(header)!r}")
    try:
        n_types = int(header[1])
    except ValueError as exc:
        raise ValueError(f"{path}: {header[1]!r} is not a type count") from exc
    symbols = header[2:2 + n_types]
    if len(symbols) != n_types:
        raise ValueError(
            f"{path}: header claims {n_types} types but lists {len(symbols)}"
        )
    unknown = [s for s in symbols if s not in atomic_numbers]
    if unknown:
        raise ValueError(f"{path}: header lists non-elements {unknown}")
    return frozenset(symbols)


# --------------------------------------------------------------------------- #
# The Lennard-Jones fallback
# --------------------------------------------------------------------------- #
#: Well depth of the fallback potential, eV.
#:
#: LJ energies are never scored, so the only thing this constant decides is how
#: hard the fallback pushes -- and therefore how tightly a fallback structure is
#: relaxed before the scoring potential takes over, given that ``fmax`` is an
#: absolute force tolerance.  1 eV puts the force scale at ~10 eV/A near
#: contact, so the protocol's 0.05 eV/A leaves the geometry converged to a few
#: parts in a thousand.  Far smaller and the stage would meet ``fmax`` without
#: moving anything.
LJ_EPSILON_EV = 1.0

#: Pair cutoff, as a multiple of that pair's sigma.  2.5 sigma is the textbook
#: truncation and keeps ~1.6% of the well depth at the cut, which the shift
#: below removes.
LJ_CUTOFF_SIGMA = 2.5

#: 2**(-1/6): converts a minimum-energy separation into a LJ sigma.
_R_MIN_TO_SIGMA = 2.0 ** (-1.0 / 6.0)


class SpeciesLennardJones(Calculator):
    """Pairwise Lennard-Jones with per-species sizes, energy-shifted at the cut.

    Each pair's minimum is placed at the sum of the two covalent radii -- the
    same contact criterion PyXtal's ``Tol_matrix(prototype="atomic")`` uses for
    the generation floor -- so the potential is repulsive inside contact and
    weakly attractive outside it.  Sizes combine as
    ``sigma_ij = 2**(-1/6) (r_i + r_j)``, which is a Lorentz-Berthelot-style
    arithmetic mix on radii; the well depth is the same for every pair, since
    nothing here justifies a per-element one.

    ASE ships :class:`ase.calculators.lj.LennardJones`, which takes a *single*
    sigma and epsilon.  On a multi-element crystal that is worse than useless:
    one sigma for every pair either lets the small species overlap or holds the
    large one apart, which is the very failure the PyXtal fork in this project
    exists to fix (see ``pyproject.toml``'s ``tool.uv.sources``).

    The pair energy is shifted by its value at the cutoff, so energy and forces
    are continuous there and an optimiser cannot chase a step discontinuity.
    The forces are the analytic derivative and the stress the analytic virial,
    both verified against finite differences in the tests.

    Args:
        epsilon: Well depth, eV.
        cutoff_sigma: Pair cutoff as a multiple of that pair's sigma.
        radii: Per-element minimum-energy contact radii, A.  Defaults to ASE's
            covalent radii.
    """

    implemented_properties = ("energy", "free_energy", "forces", "stress")

    def __init__(
        self,
        epsilon: float = LJ_EPSILON_EV,
        cutoff_sigma: float = LJ_CUTOFF_SIGMA,
        radii: Optional[dict[str, float]] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")
        if cutoff_sigma <= 1.0:
            raise ValueError(
                f"cutoff_sigma must exceed 1 (the minimum sits at 2**(1/6) sigma), "
                f"got {cutoff_sigma}"
            )
        self.epsilon = float(epsilon)
        self.cutoff_sigma = float(cutoff_sigma)
        self.radii = dict(radii) if radii else {}

    def _radius(self, number: int) -> float:
        symbol = _symbol_of(number)
        if symbol in self.radii:
            return float(self.radii[symbol])
        radius = float(covalent_radii[number])
        if radius <= 0:  # ASE carries 0.2 A for elements it has no datum for
            radius = 1.0
        return radius

    def sigma_matrix(self, numbers) -> tuple[np.ndarray, dict[int, int]]:
        """Per-species-pair sigma, and the map from atomic number into it."""
        unique = sorted({int(n) for n in numbers})
        index = {number: i for i, number in enumerate(unique)}
        radii = np.array([self._radius(n) for n in unique])
        return _R_MIN_TO_SIGMA * (radii[:, None] + radii[None, :]), index

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        from ase.neighborlist import neighbor_list

        super().calculate(atoms, properties, system_changes)
        atoms = self.atoms
        numbers = atoms.numbers
        n_atoms = len(atoms)
        forces = np.zeros((n_atoms, 3))
        virial = np.zeros((3, 3))
        energy = 0.0

        if n_atoms:
            sigma, index = self.sigma_matrix(numbers)
            # One global cutoff for the neighbour search, then the pair cutoff
            # applied per pair: ASE's neighbour list takes a per-atom radius,
            # not a per-pair one, and over-collecting is cheaper than getting
            # the bookkeeping wrong.
            i_idx, j_idx, distances, vectors = neighbor_list(
                "ijdD", atoms, self.cutoff_sigma * float(sigma.max())
            )
            rows = np.array([index[int(n)] for n in numbers[i_idx]], dtype=int) \
                if len(i_idx) else np.empty(0, dtype=int)
            cols = np.array([index[int(n)] for n in numbers[j_idx]], dtype=int) \
                if len(j_idx) else np.empty(0, dtype=int)
            if len(i_idx):
                sigma_pair = sigma[rows, cols]
                cutoff = self.cutoff_sigma * sigma_pair
                keep = distances < cutoff
                i_idx, j_idx = i_idx[keep], j_idx[keep]
                distances, vectors = distances[keep], vectors[keep]
                sigma_pair, cutoff = sigma_pair[keep], cutoff[keep]

                pair_energy, pair_dedr = _lj_pair(
                    distances, sigma_pair, self.epsilon, cutoff
                )
                # Every unordered pair is listed twice (as i-j and as j-i), so
                # the energy is halved -- but the force accumulates once per
                # listing, which is once per atom of the pair, i.e. exactly the
                # derivative of the halved sum.
                energy = 0.5 * float(pair_energy.sum())
                direction = vectors / distances[:, None]
                np.add.at(forces, i_idx, pair_dedr[:, None] * direction)
                virial = 0.5 * (
                    (pair_dedr / distances)[:, None, None]
                    * vectors[:, :, None] * vectors[:, None, :]
                ).sum(axis=0)

        self.results["energy"] = energy
        self.results["free_energy"] = energy
        self.results["forces"] = forces
        volume = atoms.get_volume() if atoms.cell.rank == 3 else 0.0
        if volume > 0:
            self.results["stress"] = full_3x3_to_voigt_6_stress(virial / volume)


def _lj_pair(
    r: np.ndarray, sigma: np.ndarray, epsilon: float, cutoff: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Shifted LJ pair energy and its radial derivative.

    Returns:
        ``(energy, dE/dr)``.  The energy is shifted by its value at *cutoff*, so
        it vanishes there; the derivative is unaffected by a constant shift.
    """
    ratio6 = (sigma / r) ** 6
    ratio12 = ratio6 * ratio6
    cut6 = (sigma / cutoff) ** 6
    energy = 4.0 * epsilon * (ratio12 - ratio6 - (cut6 * cut6 - cut6))
    dedr = 4.0 * epsilon * (-12.0 * ratio12 + 6.0 * ratio6) / r
    return energy, dedr


def _symbol_of(number: int) -> str:
    from ase.data import chemical_symbols

    return chemical_symbols[int(number)]


# --------------------------------------------------------------------------- #
# The fallback: ZBL where NEP89 is ZBL, Morse where it has to bind
# --------------------------------------------------------------------------- #
#: Coefficients and exponents of the ZBL universal screening function,
#: ``phi(x) = sum_k c_k exp(-b_k x)``, from Ziegler, Biersack and Littmark
#: (1985).  These are the numbers GPUMD's ``nep`` uses, which is why the
#: fallback below reproduces NEP89's own short-range limit rather than merely
#: resembling it: measured against the shipped model, pure ZBL matches NEP89 to
#: within 0.1-0.8% at r <= 0.4 A for Si-Si, Fe-Fe, Na-Cl and U-O.
ZBL_COEFFICIENTS = (0.18175, 0.50986, 0.28022, 0.02817)
ZBL_EXPONENTS = (3.19980, 0.94229, 0.40290, 0.20162)

#: Universal screening length: ``a = 0.4685 / (Z_i^0.23 + Z_j^0.23)``, in A.
ZBL_SCREENING_CONSTANT = 0.46850
ZBL_SCREENING_POWER = 0.23

#: ``e^2 / (4 pi eps_0)`` in eV A, so that the bare Coulomb term is
#: ``COULOMB_EV_ANGSTROM * Z_i * Z_j / r``.
COULOMB_EV_ANGSTROM = 14.399645

#: Where the ZBL core is handed over to the binding term, as fractions of the
#: pair's contact distance ``r_e = r_i + r_j``.
#:
#: NEP89 itself switches over absolute radii -- its header reads ``zbl 1 2``, so
#: full ZBL below 1 A blending out by 2 A.  Absolute radii cannot be reused
#: here: a pair whose contact distance is *inside* that window (H-H at 0.62 A)
#: would have its binding region suppressed and would never bind at all.  Making
#: the window scale with ``r_e`` fixes that and reduces to NEP89's own choice for
#: a typical pair -- Si-Si gets 0.89 to 1.67 A against NEP89's 1 to 2.
ZBL_INNER_FRACTION = 0.40
ZBL_OUTER_FRACTION = 0.75

#: Morse width, as the dimensionless product ``alpha * r_e``.
#:
#: Scale-free on purpose: a fixed ``alpha`` in inverse angstroms would make the
#: well absurdly stiff for H-H relative to Cs-Cs, whose contact distances differ
#: by a factor of eight.  4.0 puts the curvature in the range Morse fits to real
#: solids take.
MORSE_ALPHA_TIMES_RE = 4.0

#: Pair cutoff, as a multiple of ``r_e``.  Morse decays exponentially, so at
#: ``2 r_e`` the surviving depth is ``e^-4``, under 2% of the well.
MORSE_CUTOFF_RE = 2.0


class ScreenedMorse(Calculator):
    """Repulsion from ZBL, cohesion from Morse: the fallback NEP89 deserves.

    The fallback exists for the five elements below Pu that NEP89 omits (Po, At,
    Rn, Fr, Ra) and everything above it.  Those are exactly the structures a
    pre-relaxation gets handed with no surrogate at all, so what happens at
    short range matters more here than anywhere else -- and a PyXtal draw really
    does arrive crowded: 45% of them sit below the nominal generation floor and
    the worst measured contact was 0.19 of the covalent-radius sum.

    A plain Lennard-Jones, which this replaces, is the wrong shape for that job.
    Its ``r^-12`` core reaches **1.4e16 eV at 0.1 A** -- a magnitude that
    overflows float32 and sends a quasi-Newton optimiser on an arbitrary step --
    and its repulsion is a fitting convenience with no physical content.  ZBL is
    neither: it is the screened nuclear repulsion, it is what NEP89 itself uses
    below 1 A (verified to 0.1-0.8% on the shipped model), and at 0.1 A it gives
    the physically right ~1.4e4 eV for Si-Si instead of twelve orders of
    magnitude more.

    The two pieces and why both are needed:

    ``ZBL`` (below ``ZBL_INNER_FRACTION * r_e``)
        Strictly repulsive, correct in magnitude, and smooth -- so an overlapped
        draw is pushed apart along a well-conditioned gradient.
    ``Morse`` (above ``ZBL_OUTER_FRACTION * r_e``)
        Supplies the well at ``r_e = r_i + r_j`` that a *variable-cell*
        relaxation needs.  ZBL alone is monotonically repulsive, so a cell
        relaxed under it would simply expand without bound; the pre-relaxation's
        whole purpose is to contract a loose draw towards contact.  Morse rather
        than the attractive half of LJ because it is bounded and exponentially
        decaying, so nothing in it can produce a pathological magnitude either.

    The handover is GPUMD's own cosine switch, which is ``C^1`` at both ends, so
    energy and forces are continuous everywhere.  Forces and stress are analytic
    and checked against finite differences in the tests.

    Its energies remain meaningless for anything but geometry -- see the module
    docstring.  What changed is that the geometry it produces is now arrived at
    through the same physics NEP89 uses, rather than through a repulsive wall
    that happens to be steep.

    Args:
        epsilon: Morse well depth, eV.  Sets how hard the fallback pulls a loose
            cell together, and therefore how tightly ``fmax`` converges it.
        radii: Per-element contact radii, A.  Defaults to ASE's covalent radii,
            whose sum is the same contact criterion PyXtal's tolerance matrix is
            built from.
    """

    implemented_properties = ("energy", "free_energy", "forces", "stress")

    def __init__(
        self,
        epsilon: float = LJ_EPSILON_EV,
        radii: Optional[dict[str, float]] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")
        self.epsilon = float(epsilon)
        self.radii = dict(radii) if radii else {}

    def _radius(self, number: int) -> float:
        symbol = _symbol_of(number)
        if symbol in self.radii:
            return float(self.radii[symbol])
        radius = float(covalent_radii[number])
        return radius if radius > 0 else 1.0

    def pair_tables(self, numbers) -> tuple[np.ndarray, dict[int, int]]:
        """Per-species-pair contact distance ``r_e``, and the index into it."""
        unique = sorted({int(n) for n in numbers})
        index = {number: i for i, number in enumerate(unique)}
        radii = np.array([self._radius(n) for n in unique])
        return radii[:, None] + radii[None, :], index

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        from ase.neighborlist import neighbor_list

        super().calculate(atoms, properties, system_changes)
        atoms = self.atoms
        numbers = atoms.numbers
        forces = np.zeros((len(atoms), 3))
        virial = np.zeros((3, 3))
        energy = 0.0

        if len(atoms):
            re_table, index = self.pair_tables(numbers)
            i_idx, j_idx, distances, vectors = neighbor_list(
                "ijdD", atoms, MORSE_CUTOFF_RE * float(re_table.max())
            )
            if len(i_idx):
                rows = np.fromiter((index[int(n)] for n in numbers[i_idx]),
                                   dtype=int, count=len(i_idx))
                cols = np.fromiter((index[int(n)] for n in numbers[j_idx]),
                                   dtype=int, count=len(j_idx))
                re = re_table[rows, cols]
                keep = distances < MORSE_CUTOFF_RE * re
                i_idx, j_idx = i_idx[keep], j_idx[keep]
                distances, vectors, re = distances[keep], vectors[keep], re[keep]
                z_i = numbers[i_idx].astype(float)
                z_j = numbers[j_idx].astype(float)

                pair_energy, pair_dedr = _screened_morse_pair(
                    distances, re, z_i, z_j, self.epsilon
                )
                # Each unordered pair is listed twice, so the energy is halved;
                # the force accumulates once per listing, which is once per atom
                # of the pair -- exactly the derivative of the halved sum.
                energy = 0.5 * float(pair_energy.sum())
                np.add.at(forces, i_idx, (pair_dedr / distances)[:, None] * vectors)
                virial = 0.5 * (
                    (pair_dedr / distances)[:, None, None]
                    * vectors[:, :, None] * vectors[:, None, :]
                ).sum(axis=0)

        self.results["energy"] = energy
        self.results["free_energy"] = energy
        self.results["forces"] = forces
        volume = atoms.get_volume() if atoms.cell.rank == 3 else 0.0
        if volume > 0:
            self.results["stress"] = full_3x3_to_voigt_6_stress(virial / volume)


def _zbl_pair(r, z_i, z_j):
    """ZBL screened nuclear repulsion and its radial derivative, eV and eV/A."""
    screening = ZBL_SCREENING_CONSTANT / (
        z_i ** ZBL_SCREENING_POWER + z_j ** ZBL_SCREENING_POWER
    )
    x = r / screening
    prefactor = COULOMB_EV_ANGSTROM * z_i * z_j
    phi = np.zeros_like(r)
    dphi = np.zeros_like(r)
    for c, b in zip(ZBL_COEFFICIENTS, ZBL_EXPONENTS):
        term = c * np.exp(-b * x)
        phi += term
        dphi += -b / screening * term
    energy = prefactor * phi / r
    dedr = prefactor * (dphi / r - phi / r ** 2)
    return energy, dedr


def _morse_pair(r, re, epsilon):
    """Morse well at *re*, shifted to vanish at the cutoff, and its derivative."""
    alpha = MORSE_ALPHA_TIMES_RE / re
    cutoff = MORSE_CUTOFF_RE * re

    def raw(distance):
        expo = np.exp(-alpha * (distance - re))
        return epsilon * ((1.0 - expo) ** 2 - 1.0), expo

    energy, expo = raw(r)
    shift, _ = raw(cutoff)
    dedr = 2.0 * epsilon * alpha * expo * (1.0 - expo)
    return energy - shift, dedr


def _switch(r, r_inner, r_outer):
    """GPUMD's cosine handover: 1 below *r_inner*, 0 above *r_outer*, C1 at both.

    Returns:
        ``(f, df/dr)``.
    """
    width = np.maximum(r_outer - r_inner, 1e-12)
    t = np.clip((r - r_inner) / width, 0.0, 1.0)
    f = 0.5 * (1.0 + np.cos(np.pi * t))
    inside = (r > r_inner) & (r < r_outer)
    dfdr = np.where(inside, -0.5 * np.pi / width * np.sin(np.pi * t), 0.0)
    return f, dfdr


def _screened_morse_pair(r, re, z_i, z_j, epsilon):
    """The blended pair energy and its radial derivative.

    ``E = f(r) V_ZBL(r) + (1 - f(r)) V_Morse(r)`` with *f* the cosine switch, so
    the ZBL core is what an overlapped pair feels and the Morse well is what a
    loose cell contracts into, with a continuous derivative between them.
    """
    zbl, dzbl = _zbl_pair(r, z_i, z_j)
    morse, dmorse = _morse_pair(r, re, epsilon)
    f, dfdr = _switch(r, ZBL_INNER_FRACTION * re, ZBL_OUTER_FRACTION * re)
    energy = f * zbl + (1.0 - f) * morse
    dedr = f * dzbl + (1.0 - f) * dmorse + dfdr * (zbl - morse)
    return energy, dedr


# --------------------------------------------------------------------------- #
# The hybrid calculator
# --------------------------------------------------------------------------- #
class Nep89WithFallback(Calculator):
    """NEP89 where it is fitted, a ZBL-cored surrogate where it is not.

    One calculator that a worker can build once and hand every trial, whatever
    the trial's chemistry.  Which potential answers is decided per structure
    from its element set: NEP89 covers 89 elements
    (:func:`nep89_elements`) and cannot be evaluated at all outside them, so a
    cell containing Po, Ra, Am or anything above Pu goes to
    :class:`ScreenedMorse` in full -- which shares NEP89's own short-range
    physics, so the handover changes which model answers without changing what
    happens to an overlapped pair.  Which one answered is recorded on
    :attr:`last_backend`, and the *set* of structures that fell back is counted
    in :attr:`n_fallback`, so a run can say how much of its pre-relaxation was
    real.

    ``calorine``'s ``CPUNEP`` keeps a C++ NEP object sized to one structure and
    only refreshes positions, cell and species in place.  Handing it a different
    atom count would write past that object's arrays, so the wrapper rebuilds it
    -- ``set_atoms``, which resets it -- whenever the symbol sequence changes.
    That costs one 15 MB model re-parse (~0.24 s) per trial and nothing per
    optimiser step, which is the split that matters: a trial is hundreds of
    steps.

    Args:
        model: NEP model file or URL; ``None`` takes the pinned NEP89.
        epsilon: Morse well depth of the fallback, eV.
        elements: Override the supported set.  For tests; the default is read
            from the model file.
        fallback: Override the fallback calculator entirely.
    """

    implemented_properties = ("energy", "free_energy", "forces", "stress")

    def __init__(
        self,
        model: Optional[Union[str, Path]] = None,
        epsilon: float = LJ_EPSILON_EV,
        elements: Optional[frozenset[str]] = None,
        fallback: Optional[Calculator] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.model_path = resolve_nep89_model(model)
        self.elements = frozenset(elements) if elements is not None else nep89_elements(
            self.model_path
        )
        self.fallback = fallback if fallback is not None else ScreenedMorse(epsilon=epsilon)
        self.last_backend: Optional[str] = None
        self.n_fallback = 0
        self.fallback_species: set[str] = set()
        self._nep = None
        self._nep_symbols: Optional[tuple[str, ...]] = None

    def supports(self, atoms: Atoms) -> bool:
        """Whether NEP89 covers every element in *atoms*."""
        return set(atoms.get_chemical_symbols()) <= self.elements

    def unsupported_species(self, atoms: Atoms) -> set[str]:
        return set(atoms.get_chemical_symbols()) - self.elements

    def _nep_calculator(self):
        if self._nep is None:
            from calorine.calculators import CPUNEP  # optional dependency

            self._nep = CPUNEP(str(self.model_path))
            logger.info("Loaded NEP89 from %s", self.model_path)
        return self._nep

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        atoms = self.atoms
        missing = self.unsupported_species(atoms)
        if missing:
            if self.last_backend != "lj" or self.fallback_species != missing:
                logger.info(
                    "NEP89 does not cover %s; falling back to %s for %s",
                    ", ".join(sorted(missing)),
                    type(self.fallback).__name__,
                    atoms.get_chemical_formula(mode="metal"),
                )
            self.last_backend = "fallback"
            self.n_fallback += 1
            self.fallback_species = missing
            self.fallback.calculate(atoms, properties, all_changes)
            self.results = dict(self.fallback.results)
            return

        nep = self._nep_calculator()
        symbols = tuple(atoms.get_chemical_symbols())
        if symbols != self._nep_symbols:
            # Rebuilds the C++ object on the next calculate(), at the right size.
            nep.set_atoms(atoms)
            self._nep_symbols = symbols
        # all_changes rather than this calculator's own diff: the inner
        # calculator's notion of "unchanged" is relative to whatever structure
        # it last saw, which is not necessarily the one we last saw.
        nep.calculate(atoms, properties, all_changes)
        self.last_backend = "nep89"
        self.results = dict(nep.results)
        self.results.setdefault("free_energy", self.results.get("energy"))

    def provenance(self) -> dict:
        """What this calculator is, for ``manifest.json``."""
        return {
            "prerelax_mlip": "nep89",
            "prerelax_checkpoint": NEP89_CHECKPOINT,
            "prerelax_model_url": NEP89_MODEL_URL,
            "prerelax_model_sha256": NEP89_MODEL_SHA256,
            "prerelax_n_elements": len(self.elements),
            "prerelax_fallback": type(self.fallback).__name__,
            "prerelax_fallback_note": (
                "ZBL screened nuclear repulsion below 0.4 r_e, blended into a "
                "Morse well at r_e; the ZBL core is the same function NEP89 "
                "uses, matched to 0.1-0.8% on the shipped model"
            ),
            "prerelax_fallback_epsilon_ev": getattr(self.fallback, "epsilon", None),
        }


def build_nep89_calculator(
    model: Optional[Union[str, Path]] = None,
    epsilon: float = LJ_EPSILON_EV,
    device: str = "cpu",
) -> Nep89WithFallback:
    """Build :class:`Nep89WithFallback`.

    Args:
        model: NEP model file or URL; ``None`` takes the pinned NEP89.
        epsilon: Morse well depth of the fallback, eV.
        device: Accepted and ignored, so that this builder has the same
            signature as the GPU-backed ones.  ``calorine``'s CPU
            implementation is what makes NEP89 cheap here -- it needs no CUDA
            context, so a pre-relaxation worker can run on a core while the
            scoring potential owns the card -- and its GPU sibling drives the
            ``gpumd`` executable through files, which is the wrong shape for
            one force call at a time.
    """
    if device and not str(device).startswith("cpu"):
        logger.info("NEP89 runs on the CPU; ignoring device=%r", device)
    return Nep89WithFallback(model=model, epsilon=epsilon)
