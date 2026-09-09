"""Template-matching starts: a training structure's geometry in place of a random draw.

``single_pyxtal`` hands a gene to PyXtal's rejection sampler, which guesses the
cell volume, the cell shape and every free Wyckoff coordinate.  The oracle
studies say what that guessing costs: the true cell *and* the true coordinates
together recover the target at every degree of freedom, while either alone
recovers ~0.17 above 10 positional DoF
(``docs/archive/pyxtal_dof_reduction_study.md``).  A gene, however, does not
come with its own coordinates -- but a training structure built on the *same
Wyckoff orbits* does, and those coordinates are a far better guess than a
uniform draw.

So: find the LeMat-Bulk structures whose anonymous Wyckoff fingerprint -- space
group plus the multiset of occupied orbits, elements ignored -- equals the
gene's, take the one whose chemical formula is closest, and start the
relaxation from its lattice and coordinates with the gene's elements written
onto its orbits.  A gene whose fingerprint no training structure shares keeps
the random start; so does one whose only candidates cannot be rebuilt.

The match is by *anonymous* fingerprint on purpose.  Requiring the elements too
would restrict templates to structures the gene is already a chemical
substitution of, which is both far rarer and precisely the case where the
generated structure is least likely to be novel.  What transfers between two
structures on the same orbits is geometry, and geometry is what the sampler is
bad at.

Three pieces:

:class:`TemplateIndex`
    LeMat-Bulk keyed by anonymous fingerprint, with the composition and the
    hull distance of every entry.  Built once from the Wyckoff cache
    (~2 minutes over 4.2M rows) and persisted as a parquet.
:func:`select_templates`
    The formula match: cheapest first by shared atoms, then an optimal
    element-substitution transport over the survivors, ties broken by hull
    distance.
:func:`template_atoms`
    The rebuild: PyXtal symmetry detection on the template, the gene's elements
    assigned to its orbits by chemical similarity, and the result handed back
    as :class:`~ase.Atoms` on the template's own lattice.
"""
from __future__ import annotations

import hashlib
import logging
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

#: Where :meth:`TemplateIndex.load` keeps the built index.  Beside the Wyckoff
#: cache it is derived from, and regenerated whenever that cache changes.
DEFAULT_INDEX_PATH = Path("cache/lemat_bulk_ehull/anonymous_wyckoff_index.parquet")

#: Columns of the index parquet.  ``immutable_id`` is the frame's index.
INDEX_COLUMNS = ("anon_hash", "letters", "composition", "energy_above_hull")

#: Columns of the Wyckoff cache that :func:`record_anonymous_hash` reads.
_ANONYMOUS_COLUMNS = (
    "spacegroup_number",
    "site_symmetries",
    "sites_enumeration_augmented",
)

#: PyXtal symmetry tolerances tried, in order, when reading a template's orbits.
#: 0.1 first because that is what the Wyckoff cache was built with, so it is the
#: tolerance whose answer the index already agrees with; the others rescue a
#: template whose CIF has drifted since.
DEFAULT_SEED_TOLERANCES = (0.1, 0.01, 0.3, 1e-3, 1e-5)

#: Largest Mendeleev-number gap, used to normalise :func:`element_distance` onto
#: ``[0, 1]``.  The numbers run 1-103.
_MENDELEEV_SPAN = 102.0

_MENDELEEV: dict[str, float] = {}

_FORMULA_TOKEN = re.compile(r"([A-Z][a-z]?)(\d+)")


# --------------------------------------------------------------------------- #
# Keys
# --------------------------------------------------------------------------- #
def anonymous_hash(fingerprint: tuple) -> int:
    """A stable 64-bit key for an anonymous Wyckoff fingerprint.

    ``hash()`` cannot be used: the fingerprint contains strings, and Python
    salts string hashing per process, so a persisted index would not survive
    the interpreter that built it.  The frozensets are sorted into a canonical
    form and digested instead.

    Args:
        fingerprint: The output of
            :func:`~wyckoff_transformer.evaluation.novelty.record_to_anonymous_fingerprint`.

    Returns:
        The digest, as an unsigned 64-bit integer.
    """
    spacegroup, variants = fingerprint
    canonical = sorted(
        tuple(sorted(((str(symmetry), int(enumeration)), int(count))
                     for (symmetry, enumeration), count in variant))
        for variant in variants
    )
    digest = hashlib.blake2b(
        repr((int(spacegroup), canonical)).encode(), digest_size=8
    ).digest()
    return int.from_bytes(digest, "big")


def record_anonymous_hash(record: Mapping) -> int:
    """:func:`anonymous_hash` of a Wyckoff-representation record."""
    from wyckoff_transformer.evaluation.novelty import record_to_anonymous_fingerprint

    return anonymous_hash(record_to_anonymous_fingerprint(record))


def letters_key(letters: Iterable[str]) -> str:
    """Canonical form of a multiset of Wyckoff letters, e.g. ``"a b d d d i"``.

    The anonymous fingerprint identifies orbits by ``(site symmetry,
    enumeration)`` and is a *set over equivalent enumerations*, so two records
    can share it while their Wyckoff letters differ by a relabelling of the
    affine normaliser.  A template's coordinates are only transferable under the
    identity relabelling, so this narrower key is what
    :func:`select_templates` actually matches on; the fingerprint is what makes
    the lookup cheap.
    """
    return " ".join(sorted(str(letter) for letter in letters))


def composition_key(composition: Mapping) -> str:
    """Canonical formula string, elements in alphabetical order, e.g. ``"Fe4Re4"``."""
    return "".join(
        f"{symbol}{int(count)}"
        for symbol, count in sorted(
            (getattr(element, "symbol", str(element)), count)
            for element, count in composition.items()
        )
    )


def parse_composition(formula: str) -> dict[str, int]:
    """Inverse of :func:`composition_key`."""
    return {symbol: int(count) for symbol, count in _FORMULA_TOKEN.findall(formula)}


# --------------------------------------------------------------------------- #
# Chemical distance
# --------------------------------------------------------------------------- #
def element_distance(first: str, second: str) -> float:
    """Chemical dissimilarity of two elements, in ``[0, 1]``.

    The normalised Mendeleev-number gap.  That ordering walks the periodic
    table down each group before moving to the next, so chemically
    interchangeable elements sit next to each other -- the alkali metals occupy
    8-12, the halogens 97-102 -- which is exactly the similarity a substitution
    on a fixed orbit needs.  Atomic number would rank Li closer to Be than to
    Na, which is the wrong answer for this purpose.

    Args:
        first: Element symbol.
        second: Element symbol.

    Returns:
        ``0.0`` for the same element, up to ``1.0`` for opposite ends of the
        table.
    """
    if first == second:
        return 0.0
    if not _MENDELEEV:
        from pymatgen.core import Element

        for number in range(1, 104):
            element = Element.from_Z(number)
            _MENDELEEV[element.symbol] = float(element.mendeleev_no)
    try:
        gap = abs(_MENDELEEV[first] - _MENDELEEV[second])
    except KeyError:  # an element outside the table: maximally dissimilar
        return 1.0
    return min(gap / _MENDELEEV_SPAN, 1.0)


def _assign(rows: Sequence[str], columns: Sequence[str]) -> tuple[np.ndarray, float]:
    """Cheapest one-to-one assignment of *rows* onto *columns* by element distance.

    Returns:
        ``(column index per row, total cost)``.
    """
    from scipy.optimize import linear_sum_assignment

    cost = np.array(
        [[element_distance(row, column) for column in columns] for row in rows],
        dtype=float,
    ).reshape(len(rows), len(columns))
    row_index, column_index = linear_sum_assignment(cost)
    order = np.empty(len(rows), dtype=int)
    order[row_index] = column_index
    return order, float(cost[row_index, column_index].sum())


def composition_distance(gene: Mapping[str, int], template: Mapping[str, int]) -> float:
    """How far a template's formula is from the gene's, per atom, in ``[0, 1]``.

    Both compositions hold the same number of atoms -- they occupy the same
    Wyckoff orbits -- so this is the cost of the cheapest one-to-one
    substitution of the template's atoms into the gene's, divided by that
    count.  Atoms of an element both share cost nothing and are cancelled
    before the assignment runs, which is what keeps it small: only the
    *difference* between the two formulas is matched.

    Args:
        gene: Element symbol -> count, for the gene.
        template: Element symbol -> count, for the candidate.

    Returns:
        ``0.0`` if the formulas are identical.

    Raises:
        ValueError: If the two hold different numbers of atoms.
    """
    total = sum(gene.values())
    if total != sum(template.values()):
        raise ValueError(
            f"compositions hold {total} and {sum(template.values())} atoms; "
            f"a template shares the gene's orbits and therefore its atom count"
        )
    if total == 0:
        return 0.0
    residual_gene: list[str] = []
    residual_template: list[str] = []
    for symbol, count in gene.items():
        surplus = count - min(count, template.get(symbol, 0))
        residual_gene.extend([symbol] * surplus)
    for symbol, count in template.items():
        surplus = count - min(count, gene.get(symbol, 0))
        residual_template.extend([symbol] * surplus)
    if not residual_gene:
        return 0.0
    _, cost = _assign(residual_gene, residual_template)
    return cost / total


def _shared_atoms(gene: Mapping[str, int], template: Mapping[str, int]) -> int:
    """Atoms the two formulas already agree on: the prefilter's ranking key."""
    return sum(min(count, template.get(symbol, 0)) for symbol, count in gene.items())


# --------------------------------------------------------------------------- #
# The index
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TemplateQuery:
    """What a gene asks the index for."""

    anon_hash: int
    letters: str
    composition: dict[str, int]

    @property
    def n_atoms(self) -> int:
        return sum(self.composition.values())


@dataclass(frozen=True)
class TemplateMatch:
    """A candidate template, ranked."""

    immutable_id: str
    composition: str
    distance: float
    energy_above_hull: float


def gene_query(gene: Mapping, fingerprinter=None) -> TemplateQuery:
    """The index lookup for one PyXtal-notation gene.

    Args:
        gene: ``group``/``species``/``numIons``/``sites``, as
            :func:`~wyckoff_transformer.cryspr.generator.single_pyxtal` takes.
        fingerprinter: A
            :class:`~wyckoff_transformer.evaluation.protocol.GeneFingerprinter`,
            built here if not supplied.  Building one loads the Wyckoff
            mappings, so a caller with many genes should keep one.

    Returns:
        The query.

    Raises:
        Exception: Whatever the fingerprinter raises for an illegal gene.
    """
    from wyckoff_transformer.evaluation.protocol import GeneFingerprinter

    if fingerprinter is None:
        fingerprinter = GeneFingerprinter()
    record = fingerprinter.record(gene)
    return TemplateQuery(
        anon_hash=record_anonymous_hash(record),
        letters=letters_key(record["wyckoff_letters"]),
        composition=dict(
            Counter(
                str(element)
                for element, multiplicity in zip(record["elements"], record["multiplicity"])
                for _ in range(int(multiplicity))
            )
        ),
    )


class TemplateIndex:
    """LeMat-Bulk keyed by anonymous Wyckoff fingerprint.

    Attributes:
        frame: One row per training structure, indexed by ``immutable_id``,
            with the columns of :data:`INDEX_COLUMNS`.
    """

    def __init__(self, frame: pd.DataFrame) -> None:
        missing = [column for column in INDEX_COLUMNS if column not in frame]
        if missing:
            raise ValueError(f"template index is missing column(s) {missing}")
        self.frame = frame
        # Grouped once: the lookup is by fingerprint and a run makes one call
        # per gene, so scanning 4.2M rows per gene is the thing to avoid.
        self._by_hash = {
            int(key): group for key, group in frame.groupby("anon_hash", sort=False)
        }

    def __len__(self) -> int:
        return len(self.frame)

    @property
    def n_fingerprints(self) -> int:
        return len(self._by_hash)

    @classmethod
    def load(
        cls,
        path: Optional[Path] = None,
        cache: Optional[Path] = None,
        splits: Optional[Sequence[str]] = None,
    ) -> "TemplateIndex":
        """Read the index parquet, building it from the Wyckoff cache if absent.

        Args:
            path: The parquet.  Defaults to :data:`DEFAULT_INDEX_PATH`.
            cache: Wyckoff cache to build from, if the parquet is not there.
            splits: Splits to index.  Defaults to the protocol's reference
                splits -- the whole corpus, which is also what novelty is
                judged against.

        Returns:
            The index.
        """
        path = Path(path) if path is not None else DEFAULT_INDEX_PATH
        if path.is_file():
            frame = pd.read_parquet(path)
            logger.info("Loaded %d template rows from %s", len(frame), path)
            return cls(frame)
        return cls(build_index_frame(cache=cache, splits=splits, out=path))

    def candidates(self, query: TemplateQuery) -> pd.DataFrame:
        """Every training structure on the gene's fingerprint *and* its orbits."""
        group = self._by_hash.get(int(query.anon_hash))
        if group is None:
            return self.frame.iloc[:0]
        return group[group["letters"].values == query.letters]

    def select(
        self,
        query: TemplateQuery,
        exclude: Iterable[str] = (),
        k: int = 4,
        prefilter: int = 64,
    ) -> list[TemplateMatch]:
        """The *k* templates whose formula is closest to the gene's.

        Ranking is two-tier because the candidate pool is not small -- a common
        fingerprint such as a single 4a orbit in Fm-3m is shared by tens of
        thousands of entries.  Every candidate is scored by the atoms its
        formula already shares with the gene, which is one dictionary pass;
        only the *prefilter* best go on to :func:`composition_distance`, which
        solves an assignment.  Ties on distance -- and identical formulas tie at
        zero often -- are broken by hull distance: of two prototypes for the
        same composition, the more stable one is the better guess at where the
        atoms sit.

        Args:
            query: From :func:`gene_query`.
            exclude: ``immutable_id``s not to return.  The oracle studies
                reconstruct a training structure and must exclude *itself*,
                which is otherwise always the closest match there is.
            k: How many to return.  More than one because a template can turn
                out not to be rebuildable, and the caller can then fall through
                to the next without another pass over the CIF export.
            prefilter: Candidates carried into the exact ranking.

        Returns:
            Matches, closest first.  Empty when the gene has no template.
        """
        candidates = self.candidates(query)
        if not len(candidates):
            return []
        excluded = set(exclude)
        ranked: list[tuple[int, float, str, str, float]] = []
        for immutable_id, formula, hull in zip(
            candidates.index.values,
            candidates["composition"].values,
            candidates["energy_above_hull"].values,
        ):
            immutable_id = str(immutable_id)
            if immutable_id in excluded:
                continue
            composition = parse_composition(str(formula))
            if sum(composition.values()) != query.n_atoms:
                # Same orbits means the same atom count, so this is a corrupt or
                # truncated index row rather than a candidate that lost a race.
                logger.debug("Template %s has %d atoms, the gene has %d; skipped",
                             immutable_id, sum(composition.values()), query.n_atoms)
                continue
            shared = _shared_atoms(query.composition, composition)
            ranked.append((-shared, float(hull), immutable_id, str(formula), float(hull)))
        if not ranked:
            return []
        ranked.sort()
        matches = []
        for _, _, immutable_id, formula, hull in ranked[:prefilter]:
            matches.append(
                TemplateMatch(
                    immutable_id=immutable_id,
                    composition=formula,
                    distance=composition_distance(
                        query.composition, parse_composition(formula)
                    ),
                    energy_above_hull=hull,
                )
            )
        matches.sort(key=lambda match: (match.distance, match.energy_above_hull,
                                        match.immutable_id))
        return matches[:k]


def build_index_frame(
    cache: Optional[Path] = None,
    splits: Optional[Sequence[str]] = None,
    out: Optional[Path] = None,
) -> pd.DataFrame:
    """Build the anonymous-fingerprint index from the Wyckoff cache.

    One pass over the cache, which holds no geometry: the fingerprint, the
    Wyckoff letters, the composition and the hull distance are all the ranking
    needs, and the coordinates are only fetched for the handful of templates
    that are actually chosen (:func:`load_template_structures`).

    Args:
        cache: Pickle of split name -> Wyckoff-representation DataFrame.
            Defaults to the protocol's reference cache.
        splits: Splits to index.  Defaults to the protocol's -- all of them.
        out: Where to write the parquet.  Not written when ``None``.

    Returns:
        The frame, indexed by ``immutable_id``.
    """
    from wyckoff_transformer.evaluation.protocol import (
        DEFAULT_REFERENCE_CACHE,
        DEFAULT_REFERENCE_SPLITS,
    )
    from wyckoff_transformer.evaluation.novelty import record_to_anonymous_fingerprint

    cache = Path(cache) if cache is not None else DEFAULT_REFERENCE_CACHE
    splits = splits if splits is not None else DEFAULT_REFERENCE_SPLITS
    if not cache.is_file():
        raise FileNotFoundError(
            f"No LeMat-Bulk gene cache at {cache}. Build it with the dataset "
            f"caching scripts, or pass --reference-cache."
        )
    frames = pd.read_pickle(cache)
    missing = [split for split in splits if split not in frames]
    if missing:
        raise KeyError(f"{cache} has no split(s) {missing}; found {sorted(frames)}")

    pieces = []
    for split in splits:
        frame = frames[split]
        columns = [frame[name].values for name in _ANONYMOUS_COLUMNS]
        hashes = np.fromiter(
            (
                anonymous_hash(
                    record_to_anonymous_fingerprint(dict(zip(_ANONYMOUS_COLUMNS, values)))
                )
                for values in zip(*columns)
            ),
            dtype=np.uint64,
            count=len(frame),
        )
        pieces.append(
            pd.DataFrame(
                {
                    "anon_hash": hashes,
                    "letters": [letters_key(letters)
                                for letters in frame["wyckoff_letters"].values],
                    "composition": [composition_key(composition)
                                    for composition in frame["composition"].values],
                    "energy_above_hull": frame["energy_above_hull"].values,
                },
                index=frame.index.astype(str),
            )
        )
        logger.info("Indexed split %s: %d rows", split, len(frame))
    index = pd.concat(pieces)
    index.index.name = "immutable_id"
    if out is not None:
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        index.to_parquet(out)
        logger.info("Wrote %d template rows to %s", len(index), out)
    return index


def load_template_structures(ids: Iterable[str], **kwargs) -> dict[str, object]:
    """The chosen templates' geometry, from the LeMat-Bulk CIF export.

    A thin alias for
    :func:`~wyckoff_transformer.evaluation.structure_novelty.load_reference_structures`,
    which streams the ~1 GB export once; call it with every candidate of every
    gene at once rather than per gene.
    """
    from wyckoff_transformer.evaluation.structure_novelty import load_reference_structures

    return load_reference_structures(ids, **kwargs)


# --------------------------------------------------------------------------- #
# The rebuild
# --------------------------------------------------------------------------- #
def gene_letters(gene: Mapping) -> list[str]:
    """The Wyckoff letters the gene occupies, one entry per orbit."""
    return [
        str(site)[-1]
        for species_sites in gene["sites"]
        for site in species_sites
    ]


def _seed_pyxtal(structure, group: int, letters: str, tolerances: Sequence[float]):
    """PyXtal symmetry detection on a template, at the first tolerance that agrees.

    The template was chosen because the *index* says it sits on the gene's
    orbits, and the index was built at ``tol=0.1``.  Re-deriving the symmetry
    here can land somewhere else -- a different space group, or the same one
    with an orbit split -- and a template whose orbits do not line up with the
    gene's cannot lend it coordinates.  So each tolerance is tried until one
    reproduces the space group and the letters the index promised.

    Returns:
        The ``pyxtal`` object, or ``None`` if no tolerance agreed.
    """
    from pyxtal import pyxtal

    for tolerance in tolerances:
        try:
            crystal = pyxtal()
            crystal.from_seed(structure, tol=tolerance)
        except Exception as exc:  # noqa: BLE001 - every tolerance is allowed to fail
            logger.debug("from_seed at tol=%g failed: %s", tolerance, exc)
            continue
        if crystal.group.number != int(group):
            continue
        if letters_key(site.wp.letter for site in crystal.atom_sites) == letters:
            return crystal
    return None


def template_atoms(
    gene: Mapping,
    structure,
    tolerances: Sequence[float] = DEFAULT_SEED_TOLERANCES,
):
    """Rebuild a gene on a template's lattice and coordinates.

    The template occupies the gene's orbits, so every free Wyckoff parameter
    the sampler would have drawn is already fixed by it; all that is left is to
    decide which of the gene's elements takes which orbit.  Orbits of the same
    Wyckoff letter are interchangeable, so within each letter the gene's
    elements are assigned to the template's by the cheapest total
    :func:`element_distance` -- an oxygen orbit in the template goes to the
    gene's oxygen, or failing that to its most oxygen-like element, rather than
    to whichever orbit the gene happens to list first.

    The lattice is the template's, unscaled.  That is the point of the method
    and also its main risk: a template of much smaller atoms hands over a cell
    that is too tight, and the relaxation then has to expand it.

    Args:
        gene: PyXtal-notation gene.
        structure: The template, as a ``pymatgen`` ``Structure``.
        tolerances: Symmetry tolerances to try, in order.

    Returns:
        The starting structure as :class:`~ase.Atoms`.

    Raises:
        ValueError: If no tolerance reproduces the gene's space group and
            orbits, or if the rebuilt structure does not hold the gene's
            composition.
    """
    letters = gene_letters(gene)
    crystal = _seed_pyxtal(structure, int(gene["group"]), letters_key(letters), tolerances)
    if crystal is None:
        raise ValueError(
            f"no symmetry tolerance reproduces space group {int(gene['group'])} "
            f"with orbits {letters_key(letters)} for this template"
        )

    wanted: list[tuple[str, str]] = [
        (str(element), str(site)[-1])
        for element, species_sites in zip(gene["species"], gene["sites"])
        for site in species_sites
    ]
    sites_by_letter: dict[str, list] = {}
    for site in crystal.atom_sites:
        sites_by_letter.setdefault(site.wp.letter, []).append(site)

    for letter, group in _group_by_letter(wanted).items():
        targets = sites_by_letter[letter]
        order, _ = _assign(group, [site.specie for site in targets])
        for element, position in zip(group, order):
            targets[position].substitute_with_single(element)

    # ``_get_formula`` recomputes ``species``/``numIons`` from the sites, but
    # only fills ``species`` when it is unset; the template's own element list
    # is still there and would be used instead.
    crystal.species = None
    crystal._get_formula()

    atoms = crystal.to_ase()
    rebuilt = Counter(atoms.get_chemical_symbols())
    expected = Counter()
    for element, count in zip(gene["species"], gene["numIons"]):
        expected[str(element)] += int(count)
    if rebuilt != expected:
        raise ValueError(
            f"rebuilt template has composition {dict(rebuilt)}, gene asks for "
            f"{dict(expected)}"
        )
    return atoms


def _group_by_letter(wanted: Sequence[tuple[str, str]]) -> dict[str, list[str]]:
    """Gene elements per Wyckoff letter, ``{"d": ["O", "P", "Rb"]}``."""
    grouped: dict[str, list[str]] = {}
    for element, letter in wanted:
        grouped.setdefault(letter, []).append(element)
    return grouped


def single_template(
    gene: Mapping,
    matches: Sequence[TemplateMatch],
    structures: Mapping[str, object],
    tolerances: Sequence[float] = DEFAULT_SEED_TOLERANCES,
) -> tuple[Optional[object], Optional[TemplateMatch], Optional[str]]:
    """The template start for one gene, falling through candidates in order.

    The counterpart of
    :func:`~wyckoff_transformer.cryspr.generator.single_pyxtal`: it returns the
    structure a trial should start from, or ``None`` when the gene has no
    usable template and the caller should fall back to a random draw.

    Args:
        gene: PyXtal-notation gene.
        matches: Candidates from :meth:`TemplateIndex.select`, closest first.
        structures: ``immutable_id`` -> ``Structure``, from
            :func:`load_template_structures`.
        tolerances: Symmetry tolerances to try per candidate.

    Returns:
        ``(atoms, match, error)``.  On success *error* is ``None``; otherwise
        *atoms* and *match* are ``None`` and *error* says why the last
        candidate was rejected (or that there were none).
    """
    if not matches:
        return None, None, "no template shares the gene's fingerprint"
    reason = "no candidate structure could be read"
    for match in matches:
        structure = structures.get(match.immutable_id)
        if structure is None:
            continue
        try:
            return template_atoms(gene, structure, tolerances), match, None
        except Exception as exc:  # noqa: BLE001 - try the next candidate
            reason = f"{match.immutable_id}: {type(exc).__name__}: {exc}"
            logger.debug("Template %s unusable: %s", match.immutable_id, exc)
    return None, None, reason
