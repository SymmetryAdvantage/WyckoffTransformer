"""The de novo ranking protocol: a funnel from sampled genes to SUN.

The cascade is::

    sampled -> valid gene -> unique gene (keep counts)
            -> gene-novel?  no  -> count into the denominator, stop
                            yes -> PyXtal + 1 trial x 2-stage CrySPR
            -> valid structure -> BAWL-unique -> BAWL-novel
            -> e_hull <= 0.1 -> e_hull <= 0

Two properties make it cheap.  First, the filters that need no potential run
before the one that does: a gene that is not novel cannot contribute to SUN or
MetaSUN by definition, so it is counted into the denominator and never relaxed.
Second, the relaxation budget is one PyXtal trial and two relaxation stages
rather than three trials and three stages -- 4.5x less work per gene for about
30% more genes at equal statistical power.

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
            reference; these are the ones worth relaxing.
        known: Representative indices whose fingerprint is present.
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


class GeneFingerprinter:
    """Converts PyXtal-notation genes to augmented Wyckoff fingerprints.

    The conversion doubles as the formal validity check: an illegal
    (space group, Wyckoff letter) pair has no entry in the mappings and raises.
    """

    def __init__(self) -> None:
        from wyckoff_transformer.data import pyxtal_notation_to_sites
        from wyckoff_transformer.preprocess_wychoffs import get_augmentation_dict
        from wyckoff_transformer.tokenization import load_wyckoff_mappings

        mappings = load_wyckoff_mappings()
        self.enum_from_ss_letter = mappings.enum_from_ss_letter
        self.ss_from_letter = mappings.ss_from_letter
        self.augmentation = get_augmentation_dict()
        self._to_sites = pyxtal_notation_to_sites

    def record(self, gene: dict) -> dict:
        """Wyckoff-representation record for one gene.  Raises if illegal."""
        return self._to_sites(
            gene, self.enum_from_ss_letter, self.ss_from_letter, self.augmentation
        )

    def fingerprint(self, gene: dict) -> tuple:
        """Augmented fingerprint, invariant to equivalent Wyckoff enumerations."""
        return record_to_augmented_fingerprint(self.record(gene))


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

    Args:
        screen: Stage-A result.
        structures: One row per relaxed representative, indexed by gene index,
            with boolean columns ``has_structure``, ``valid_structure``,
            ``bawl_unique``, ``bawl_novel`` and a float ``e_above_hull``.
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
        ("bawl_unique", "bawl_unique"),
        ("bawl_novel", "bawl_novel"),
    ]
    surviving = None
    for label, column in stages:
        if column not in structures.columns:
            result[label] = None
            result[f"{label}_per_sampled_gene"] = None
            continue
        mask = structures[column].fillna(False).astype(bool)
        if surviving is not None:
            mask = mask & surviving
        surviving = mask
        result[label] = int(mask.sum())
        result[f"{label}_per_sampled_gene"] = _ratio(weighted(mask), denominator)

    if "e_above_hull" in structures.columns and surviving is not None:
        energies = structures["e_above_hull"]
        for label, threshold in (
            ("metastable", METASTABLE_THRESHOLD),
            ("stable", STABLE_THRESHOLD),
        ):
            mask = surviving & energies.notna() & (energies <= threshold)
            result[label] = int(mask.sum())
            result[f"{label}_per_sampled_gene"] = _ratio(weighted(mask), denominator)
        # The headline: MetaSUN counts everything at or below 0.1 eV/atom that
        # also survived uniqueness and novelty.
        result["metasun_per_sampled_gene"] = result["metastable_per_sampled_gene"]
        result["sun_per_sampled_gene"] = result["stable_per_sampled_gene"]
    else:
        for key in (
            "metastable", "metastable_per_sampled_gene",
            "stable", "stable_per_sampled_gene",
            "metasun_per_sampled_gene", "sun_per_sampled_gene",
        ):
            result[key] = None

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
