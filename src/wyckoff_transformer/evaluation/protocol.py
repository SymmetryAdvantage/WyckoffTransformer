"""The de novo ranking protocol: a funnel from sampled genes to SUN.

The cascade is::

    sampled -> valid gene -> unique gene (keep counts)
            -> PyXtal + 1-3 trials x 4-stage CrySPR (2 symmetric, free, rattle)
            -> valid structure -> unique structure -> novel structure
            -> e_hull <= 0.1 -> e_hull <= 0

Novelty and uniqueness are both decided in two stages: the augmented Wyckoff
fingerprint first, then ``StructureMatcher`` on whatever shares it.  The
fingerprint alone is not a verdict -- two structures with the same space group
and the same elements on the same Wyckoff orbits can still be different
structures -- so the gene screen produces *candidates* for the matcher rather
than a decision, and a gene already present in LeMat-Bulk can still relax into
a novel structure.  Novelty is checked against both the sampled gene's
fingerprint and the *relaxed* structure's own, since relaxation -- the rattle
stage especially -- can move a structure off the orbit set PyXtal placed it on.

What keeps this cheap is the relaxation budget.  Trials are allotted by the
gene's positional degrees of freedom -- one where PyXtal has no free coordinate
to draw and a second trial provably cannot help, up to three where the draw has
room to miss -- and the stages are the two symmetry-constrained ones, an
unconstrained one, and the rattle, which is the only stage that can leave the
symmetric stationary point the others converge to.  See
:data:`DEFAULT_TRIAL_SCHEDULE` and
:func:`~wyckoff_transformer.cryspr.relaxer.stepwise_relax`.

Uniqueness is applied by *deduplicating* genes but *keeping their counts*, so
every rate stays per sampled gene.  A duplicate belongs once in the numerator
and once per sample in the denominator; reporting rates over the deduplicated
set instead would make uniqueness ~1.0 by construction.

Stage boundaries are deliberate.  ``screen`` and ``relax`` need only this
package; ``score`` needs LeMat-GenBench and its own dependency set, which in
practice lives in a separate environment.  Each stage writes its results to
disk so the next can start from them.
"""
from __future__ import annotations

import gzip
import json
import logging
import pickle
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Sequence

import pandas as pd

from wyckoff_transformer.evaluation.novelty import record_to_augmented_fingerprint

# wyckoff_transformer.tokenization imports torch, and preprocess_wychoffs pulls
# in sklearn/scipy, but only GeneFingerprinter needs either.  They are imported
# lazily so that the score stage -- which needs only read_screen and funnel --
# can run under LeMat-GenBench's environment, whose torch is pinned to 2.6 by
# its CUDA-specific torch-scatter wheels and cannot coexist with ours.

logger = logging.getLogger(__name__)

#: Default cache of LeMat-Bulk in the Wyckoff-gene representation, as written by
#: the dataset caching scripts.  Holds ``train``/``val``/``test`` frames with the
#: columns :func:`record_to_augmented_fingerprint` needs.
DEFAULT_REFERENCE_CACHE = Path("cache/lemat_bulk_ehull/data.pkl.gz")

#: Novelty must be judged against every LeMat-Bulk structure, not just the
#: training split: the benchmark's reference is the whole corpus.
DEFAULT_REFERENCE_SPLITS = ("train", "val", "test")

#: Thresholds the funnel reports, in eV/atom.  0.1 is the metastability
#: threshold LeMat-GenBench uses for MetaSUN; 0 is SUN.
METASTABLE_THRESHOLD = 0.1
STABLE_THRESHOLD = 0.0

#: PyXtal trials per gene, as ``<inclusive upper bound on positional DoF>:<trials>``
#: pairs with ``*`` for the last bin.
#:
#: Trials are worth what the random draw of the free coordinates costs, so the
#: budget belongs where those coordinates are.  ``p(e_hull <= 0.1)`` at 1, 2 and
#: 3 trials, over the 2423 genes of run ``upi73i4k`` with all three trials
#: surviving, paired per gene against the ORB ``e_above_hull`` of the same run::
#:
#:     positional DoF  share   1 trial  2 trials  3 trials
#:     0               0.202     0.410     0.410     0.410
#:     1-2             0.226     0.305     0.365     0.381
#:     3-5             0.283     0.215     0.336     0.374
#:     6-10            0.178     0.144     0.260     0.305
#:     >10             0.111     0.126     0.215     0.256
#:
#: At zero positional DoF a second trial changes *nothing* -- the only freedom
#: left is the cell, which stages 1 and 2 relax anyway, and 97.6% of those genes
#: have every trial agree to within 1 meV/atom.  One trial for a fifth of the
#: cohort is therefore free of consequence, and the trials it saves pay for the
#: extra trials above it.  Beyond 2 DoF a third trial still earns 3.8-4.4 points,
#: so the default takes it: 2.37 trials per gene reach 99% of a flat three-trial
#: budget's p at 0.8x its cost.
DEFAULT_TRIAL_SCHEDULE = "0:1,2:2,*:3"


@dataclass
class GeneScreen:
    """Outcome of the MLIP-free part of the cascade.

    Attributes:
        n_sampled: Number of genes read from the input.  Every rate is per this.
        valid: Indices of genes that are formally legal Wyckoff assignments.
        invalid: Indices that are not, mapped to the reason in *invalid_reason*.
        invalid_reason: Index -> why the gene was rejected.
        counts: Representative index -> how many sampled genes share its
            fingerprint.  The representative is the first occurrence.
        novel: Representative indices whose fingerprint is absent from the
            reference, so no LeMat-Bulk entry can match them.
        known: Representative indices whose fingerprint is present, i.e. the
            ones whose relaxed structure the matcher still has to rule on.
        fingerprint: Index -> augmented Wyckoff fingerprint, for valid genes.
    """

    n_sampled: int
    valid: list[int] = field(default_factory=list)
    invalid: list[int] = field(default_factory=list)
    invalid_reason: dict[int, str] = field(default_factory=dict)
    counts: dict[int, int] = field(default_factory=dict)
    novel: list[int] = field(default_factory=list)
    known: list[int] = field(default_factory=list)
    fingerprint: dict[int, tuple] = field(default_factory=dict)

    @property
    def n_unique(self) -> int:
        return len(self.counts)

    @property
    def n_sampled_novel(self) -> int:
        """Sampled genes, not representatives, whose fingerprint is novel."""
        return sum(self.counts[i] for i in self.novel)

    @property
    def n_sampled_known(self) -> int:
        return sum(self.counts[i] for i in self.known)

    def summary(self) -> dict:
        """Stage-A counts, in both representative and sampled-gene terms."""
        return {
            "sampled": self.n_sampled,
            "valid_gene": len(self.valid),
            "unique_gene": self.n_unique,
            "gene_novel": len(self.novel),
            "gene_known": len(self.known),
            "sampled_novel": self.n_sampled_novel,
            "sampled_known": self.n_sampled_known,
            "valid_gene_rate": _ratio(len(self.valid), self.n_sampled),
            "unique_gene_rate": _ratio(self.n_unique, self.n_sampled),
            "gene_novelty_rate": _ratio(len(self.novel), self.n_unique),
        }


def _ratio(numerator: int, denominator: int) -> Optional[float]:
    return numerator / denominator if denominator else None


def load_genes(path: Path) -> list[dict]:
    """Read a list of PyXtal-notation Wyckoff genes from JSON or gzipped JSON."""
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, mode="rt", encoding="utf-8") as handle:
        genes = json.load(handle)
    if not isinstance(genes, list):
        raise ValueError(f"{path} does not contain a list of Wyckoff genes")
    return genes


#: Columns :func:`record_to_augmented_fingerprint` reads.
_FINGERPRINT_COLUMNS = (
    "spacegroup_number",
    "elements",
    "site_symmetries",
    "sites_enumeration_augmented",
)


def _frame_fingerprints(frame: pd.DataFrame) -> Iterable[tuple]:
    """Fingerprint every row, without building a Series per row.

    ``DataFrame.apply(..., axis=1)`` dominates the runtime of this stage on a
    4-million-row reference; zipping the raw columns is several times faster and
    gives identical results.
    """
    columns = [frame[name].values for name in _FINGERPRINT_COLUMNS]
    for values in zip(*columns):
        yield record_to_augmented_fingerprint(dict(zip(_FINGERPRINT_COLUMNS, values)))


def load_reference_fingerprints(
    cache: Path = DEFAULT_REFERENCE_CACHE,
    splits: Sequence[str] = DEFAULT_REFERENCE_SPLITS,
    fingerprint_cache: Optional[Path] = None,
) -> set[tuple]:
    """Build the LeMat-Bulk fingerprint set that gene-novelty is judged against.

    Args:
        cache: Pickle holding a dict of split name -> DataFrame in the Wyckoff
            representation.
        splits: Which splits to include.  All of them, by default: the
            benchmark's novelty reference is the whole corpus, and excluding a
            split here would score memorised structures as novel.
        fingerprint_cache: Where to persist the computed set.  Reused verbatim
            when it exists, since the set depends only on *cache* and *splits*
            and every variant evaluation needs the same one.

    Returns:
        The set of augmented Wyckoff fingerprints.
    """
    if fingerprint_cache is not None and Path(fingerprint_cache).is_file():
        with gzip.open(fingerprint_cache, "rb") as handle:
            fingerprints = pickle.load(handle)
        logger.info("Loaded %d reference fingerprints from %s",
                    len(fingerprints), fingerprint_cache)
        return fingerprints

    cache = Path(cache)
    if not cache.is_file():
        raise FileNotFoundError(
            f"No LeMat-Bulk gene cache at {cache}. Build it with the dataset "
            f"caching scripts, or pass --reference-cache."
        )
    frames = pd.read_pickle(cache)
    missing = [s for s in splits if s not in frames]
    if missing:
        raise KeyError(f"{cache} has no split(s) {missing}; found {sorted(frames)}")

    fingerprints: set[tuple] = set()
    for split in splits:
        frame = frames[split]
        fingerprints.update(_frame_fingerprints(frame))
        logger.info("Reference split %s: %d rows, %d fingerprints so far",
                    split, len(frame), len(fingerprints))

    if fingerprint_cache is not None:
        fingerprint_cache = Path(fingerprint_cache)
        fingerprint_cache.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(fingerprint_cache, "wb") as handle:
            pickle.dump(fingerprints, handle, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("Cached %d reference fingerprints to %s",
                    len(fingerprints), fingerprint_cache)
    return fingerprints


def parse_trial_schedule(spec: str) -> tuple[tuple[float, int], ...]:
    """Parse a trial schedule into ``(dof_upper_bound, trials)`` pairs.

    Args:
        spec: Comma-separated ``dof:trials`` pairs, ordered by increasing DoF,
            e.g. ``"0:1,2:2,*:3"``.  Each bound is inclusive, and ``*`` (or
            ``inf``) stands for everything above the previous one.  A bare
            integer is accepted as a constant schedule.

    Returns:
        The pairs, with the last bound as ``inf``.

    Raises:
        ValueError: If the spec is malformed, unordered, or does not end in a
            bin that catches every remaining DoF.
    """
    spec = str(spec).strip()
    if not spec:
        raise ValueError("empty trial schedule")
    if spec.isdigit():  # "3" means three trials for every gene
        return ((float("inf"), _positive_trials(spec)),)

    pairs: list[tuple[float, int]] = []
    for part in spec.split(","):
        bound, separator, trials = part.partition(":")
        if not separator:
            raise ValueError(f"{part!r} is not a 'dof:trials' pair")
        bound = bound.strip()
        limit = (
            float("inf") if bound in ("*", "inf")
            else float(int(bound))  # reject 2.5 as a DoF bound
        )
        pairs.append((limit, _positive_trials(trials)))

    bounds = [limit for limit, _ in pairs]
    if bounds != sorted(bounds) or len(set(bounds)) != len(bounds):
        raise ValueError(f"trial schedule {spec!r} is not in increasing DoF order")
    if bounds[-1] != float("inf"):
        raise ValueError(
            f"trial schedule {spec!r} does not end in a '*' bin, so genes with "
            f"more than {bounds[-1]:.0f} positional DoF would get no trials"
        )
    return tuple(pairs)


def _positive_trials(value: str) -> int:
    trials = int(value)
    if trials < 1:
        raise ValueError(f"a gene needs at least one trial, got {trials}")
    return trials


def trials_for_dof(dof: int, schedule: Sequence[tuple[float, int]]) -> int:
    """Trials the *schedule* allots to a gene with *dof* positional DoF."""
    for limit, trials in schedule:
        if dof <= limit:
            return trials
    raise ValueError(f"schedule {schedule} has no bin for {dof} DoF")


def positional_dof(gene: dict) -> int:
    """Free continuous coordinates PyXtal has to draw for this gene.

    The sum over the gene's Wyckoff orbits of each orbit's degrees of freedom:
    the size of the space a trial samples from, and therefore what decides how
    much a second trial is worth.  Lattice parameters are deliberately excluded
    -- the fix-cell warm-up and the cell relaxation that follows it recover
    those regardless of where the draw started, which is why genes with zero
    positional DoF but a free cell show no trial-to-trial spread at all.

    Args:
        gene: A PyXtal-notation gene; ``group`` and ``sites`` are read.

    Returns:
        The number of free internal coordinates.

    Raises:
        Exception: Whatever PyXtal raises for an illegal (group, letter) pair.
    """
    from pyxtal.symmetry import Group

    number = int(gene["group"])
    group = _GROUPS.get(number)
    if group is None:
        group = _GROUPS[number] = Group(number)
    return sum(
        int(group[str(site)[-1]].get_dof())
        for species_sites in gene["sites"]
        for site in species_sites
    )


#: PyXtal ``Group`` objects are expensive to build and immutable once built.
_GROUPS: dict[int, object] = {}


class GeneFingerprinter:
    """Converts PyXtal-notation genes to augmented Wyckoff fingerprints.

    The conversion doubles as the formal validity check: an illegal
    (space group, Wyckoff letter) pair has no entry in the mappings and raises.
    """

    def __init__(self) -> None:
        from wyckoff_transformer.data import pyxtal_notation_to_sites, structure_to_sites
        from wyckoff_transformer.preprocess_wychoffs import get_augmentation_dict
        from wyckoff_transformer.tokenization import load_wyckoff_mappings

        mappings = load_wyckoff_mappings()
        self.enum_from_ss_letter = mappings.enum_from_ss_letter
        self.ss_from_letter = mappings.ss_from_letter
        self.augmentation = get_augmentation_dict()
        self._to_sites = pyxtal_notation_to_sites
        self._structure_to_sites = structure_to_sites

    def record(self, gene: dict) -> dict:
        """Wyckoff-representation record for one gene.  Raises if illegal."""
        return self._to_sites(
            gene, self.enum_from_ss_letter, self.ss_from_letter, self.augmentation
        )

    def fingerprint(self, gene: dict) -> tuple:
        """Augmented fingerprint, invariant to equivalent Wyckoff enumerations."""
        return record_to_augmented_fingerprint(self.record(gene))

    def fingerprint_structure(
        self, structure, tol: float = 0.1, a_tol: float = 5.0
    ) -> tuple:
        """Augmented fingerprint of a *relaxed* structure, not its sampled gene.

        Runs PyXtal symmetry detection (``pyxtal.from_seed`` over a tolerance
        sweep, via :func:`~wyckoff_transformer.data.structure_to_sites`) and
        fingerprints the result.  Relaxation -- the rattle stage especially --
        can lower the symmetry PyXtal imposed, so this fingerprint can differ
        from :meth:`fingerprint` of the gene the structure was drawn from.
        Raises if symmetry detection fails.
        """
        record = self._structure_to_sites(
            structure, self.enum_from_ss_letter, self.augmentation, tol=tol, a_tol=a_tol
        )
        return record_to_augmented_fingerprint(record)


def screen_genes(
    genes: Iterable[dict],
    reference: set[tuple],
    fingerprinter: Optional[GeneFingerprinter] = None,
) -> GeneScreen:
    """Run the free part of the cascade: validity, uniqueness, gene novelty.

    Args:
        genes: PyXtal-notation Wyckoff genes, in sampling order.
        reference: LeMat-Bulk fingerprints, from
            :func:`load_reference_fingerprints`.
        fingerprinter: Reused across calls; built here when omitted.

    Returns:
        A :class:`GeneScreen`.  Nothing is dropped -- invalid, duplicate and
        known genes are all retained as counts so that every downstream rate
        can be expressed per sampled gene.
    """
    fingerprinter = fingerprinter or GeneFingerprinter()
    genes = list(genes)
    screen = GeneScreen(n_sampled=len(genes))

    first_seen: dict[tuple, int] = {}
    for index, gene in enumerate(genes):
        try:
            fingerprint = fingerprinter.fingerprint(gene)
        except Exception as exc:
            screen.invalid.append(index)
            screen.invalid_reason[index] = f"{type(exc).__name__}: {exc}"
            continue

        screen.valid.append(index)
        screen.fingerprint[index] = fingerprint

        representative = first_seen.get(fingerprint)
        if representative is None:
            first_seen[fingerprint] = index
            screen.counts[index] = 1
            if fingerprint in reference:
                screen.known.append(index)
            else:
                screen.novel.append(index)
        else:
            screen.counts[representative] += 1

    logger.info(
        "Screened %d genes: %d valid, %d unique, %d gene-novel (%d sampled)",
        screen.n_sampled, len(screen.valid), screen.n_unique,
        len(screen.novel), screen.n_sampled_novel,
    )
    return screen


def funnel(
    screen: GeneScreen,
    structures: pd.DataFrame,
) -> dict:
    """Assemble the full funnel, with every rate per sampled gene.

    ``metastable`` / ``stable`` (and their ``_per_sampled_gene`` rates) count
    every unique structure at or below the hull threshold, *without* the novelty
    filter.  ``metastable_among_novel`` / ``stable_among_novel`` are the counts
    with novelty applied, and ``metasun_per_sampled_gene`` /
    ``sun_per_sampled_gene`` the corresponding rates -- that is what MetaSUN and
    SUN mean.

    ``gene_known_became_novel`` / ``gene_novel_became_known`` count how many
    unique structures crossed the novelty line under relaxation, relative to
    their *sampled gene's* novelty.  ``relaxed_fingerprint_resolved`` /
    ``relaxed_fingerprint_changed`` say how often the relaxed structure could be
    re-fingerprinted and how often that fingerprint differs from the gene's.

    Args:
        screen: Stage-A result.
        structures: One row per relaxed representative, indexed by gene index,
            with boolean columns ``has_structure``, ``valid_structure``,
            ``unique_structure``, ``novel_structure`` and a float
            ``e_above_hull``.  ``gene_known_became_novel`` and its siblings are
            reported only when ``novel_structure``, ``relaxed_fingerprint_*``
            are present.
            Columns absent from *structures* are reported as ``None`` rather
            than assumed, so a partial run stays honest about what it measured.

    Returns:
        A dict of counts and rates, suitable for JSON.
    """
    result = screen.summary()
    denominator = screen.n_sampled

    def weighted(mask: "pd.Series") -> int:
        """Sampled genes behind the representatives selected by *mask*."""
        if mask is None:
            return 0
        selected = [i for i, keep in mask.items() if bool(keep)]
        return sum(screen.counts.get(i, 0) for i in selected)

    stages = [
        ("structure", "has_structure"),
        ("valid_structure", "valid_structure"),
        ("unique_structure", "unique_structure"),
        ("novel_structure", "novel_structure"),
    ]
    surviving = None
    pre_novelty = None
    for label, column in stages:
        if column not in structures.columns:
            result[label] = None
            result[f"{label}_per_sampled_gene"] = None
            continue
        mask = structures[column].fillna(False).astype(bool)
        if surviving is not None:
            mask = mask & surviving
        surviving = mask
        if label != "novel_structure":
            pre_novelty = mask
        result[label] = int(mask.sum())
        result[f"{label}_per_sampled_gene"] = _ratio(weighted(mask), denominator)

    if "e_above_hull" in structures.columns and pre_novelty is not None:
        energies = structures["e_above_hull"]
        has_energy = energies.notna()
        novel = surviving if surviving is not None else pre_novelty
        for label, sun_label, threshold in (
            ("metastable", "metasun", METASTABLE_THRESHOLD),
            ("stable", "sun", STABLE_THRESHOLD),
        ):
            below = has_energy & (energies <= threshold)
            # metastable/stable: every unique structure below the threshold, no
            # novelty filter.  metasun/sun and *_among_novel: the same, with
            # novelty applied.
            total = pre_novelty & below
            among_novel = novel & below
            result[label] = int(total.sum())
            result[f"{label}_per_sampled_gene"] = _ratio(weighted(total), denominator)
            result[f"{label}_among_novel"] = int(among_novel.sum())
            result[f"{sun_label}_per_sampled_gene"] = _ratio(
                weighted(among_novel), denominator
            )
        # A structure the hull cannot reach -- a composition whose subspace is
        # empty, or a phase diagram that will not build -- fails both
        # thresholds, which is the conservative answer but not a visible one.
        result["no_hull_energy"] = int((pre_novelty & energies.isna()).sum())
    else:
        for key in (
            "metastable", "metastable_per_sampled_gene", "metastable_among_novel",
            "stable", "stable_per_sampled_gene", "stable_among_novel",
            "metasun_per_sampled_gene", "sun_per_sampled_gene",
            "no_hull_energy",
        ):
            result[key] = None

    # Novelty moves under relaxation.  A gene whose sampled fingerprint is in
    # LeMat-Bulk can still relax into a structure the matcher rejects (a); one
    # whose sampled fingerprint is absent can relax onto a known structure
    # through its *relaxed* fingerprint (b).  Both are counted against the
    # sampled gene's novelty, over representatives that produced a unique
    # structure.
    if "novel_structure" in structures.columns and pre_novelty is not None:
        novel_structure = structures["novel_structure"].fillna(False).astype(bool)
        gene_index = structures.index.to_series()
        known_gene = gene_index.isin(set(screen.known))
        novel_gene = gene_index.isin(set(screen.novel))
        became_novel = pre_novelty & known_gene & novel_structure
        became_known = pre_novelty & novel_gene & ~novel_structure
        result["gene_known_became_novel"] = int(became_novel.sum())
        result["gene_novel_became_known"] = int(became_known.sum())
        result["gene_known_became_novel_per_sampled_gene"] = _ratio(
            weighted(became_novel), denominator
        )
        result["gene_novel_became_known_per_sampled_gene"] = _ratio(
            weighted(became_known), denominator
        )
    else:
        for key in (
            "gene_known_became_novel", "gene_novel_became_known",
            "gene_known_became_novel_per_sampled_gene",
            "gene_novel_became_known_per_sampled_gene",
        ):
            result[key] = None

    for column in ("relaxed_fingerprint_resolved", "relaxed_fingerprint_changed"):
        result[column] = (
            int(structures[column].fillna(False).astype(bool).sum())
            if column in structures.columns
            else None
        )

    return result


def write_screen(screen: GeneScreen, path: Path) -> None:
    """Persist a :class:`GeneScreen` as JSON.

    Fingerprints are dropped: they are large, unstable across releases of the
    Wyckoff mappings, and recomputable from the genes.
    """
    payload = asdict(screen)
    payload.pop("fingerprint", None)
    payload["counts"] = {str(k): v for k, v in screen.counts.items()}
    payload["invalid_reason"] = {str(k): v for k, v in screen.invalid_reason.items()}
    payload["summary"] = screen.summary()
    Path(path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def read_screen(path: Path) -> GeneScreen:
    """Load a :class:`GeneScreen` written by :func:`write_screen`."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return GeneScreen(
        n_sampled=payload["n_sampled"],
        valid=payload["valid"],
        invalid=payload["invalid"],
        invalid_reason={int(k): v for k, v in payload["invalid_reason"].items()},
        counts={int(k): v for k, v in payload["counts"].items()},
        novel=payload["novel"],
        known=payload["known"],
    )
