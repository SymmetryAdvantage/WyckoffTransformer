"""128-bit canonical keys for Wyckoff genes, and a tensor table to look them up in.

:func:`~wyckoff_transformer.evaluation.novelty.record_to_augmented_fingerprint`
answers "are these the same gene" exactly, and its answer is a nested
``frozenset`` -- which means the reference set is 4.8M Python objects, ~17 GB
resident and minutes to load.  This module answers the same question with a pair
of 64-bit integers, so the reference is a sorted ``int64`` tensor of 77 MB that
``torch.searchsorted`` queries in microseconds.

**The key is exactly equivalent to the fingerprint, not an approximation of it.**
The fingerprint is a set of multisets:

    (space group, {{ (element, site symmetry, enumeration) x count } for each
                   equivalent enumeration })

so the key is built the same way round: hash each variant's *sorted* multiset,
then hash the *sorted, deduplicated* list of those variant digests together with
the space group.  Sorting canonicalises the multiset, deduplicating and sorting
canonicalises the set, and neither step assumes anything about the augmentations
-- in particular **not** that they form a group.  A canonical *representative*
(the minimum over the orbit) would have needed that assumption; hashing the whole
set does not.

Equality of keys therefore implies equality of fingerprints up to a hash
collision, whose probability over the 4,826,004 distinct fingerprints of
``lemat_bulk_fmax1_stress`` is about 4e-26.
``tests/test_gene_hash.py`` pins the equivalence against the fingerprint itself
rather than trusting the argument.

What is a tensor and what is not:

* the reference table, the membership test and cohort uniqueness -- tensors;
* the per-gene key -- ``hashlib`` over a canonical byte encoding, in Python.
  A thousand-gene cohort takes milliseconds and the 5.3M-row reference build is
  a one-off, so there is nothing yet to gain from vectorising it, and a
  hand-written 64-bit mix would be one more thing that has to be right.

The encoding is deliberately **dataset-independent**: an element enters as its
atomic number and a site symmetry as its own UTF-8 bytes, never as a tokeniser
id.  Two models trained on different datasets number their tokens differently,
and a table keyed by one model's ids would silently answer a different question
for another's.
"""
from __future__ import annotations

import gc
import hashlib
import logging
import struct
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import torch

from wyckoff_transformer.evaluation.novelty import augmented_variants

logger = logging.getLogger(__name__)

#: Bumped whenever the encoding changes, so a cached table built by an older
#: version is refused rather than silently mixed with new keys.
GENE_HASH_VERSION = 1

#: Domain separation, so a variant digest can never be read as a gene key.
_SITE_TAG = b"wyckoff-site\x00"
_VARIANT_TAG = b"wyckoff-variant\x00"
_GENE_TAG = b"wyckoff-gene\x00"

_DIGEST_BYTES = 16

#: The file a built reference table is cached as.
GENE_KEY_TABLE_FILE = "gene_keys.npz"


def _atomic_number(element) -> int:
    """An element as its atomic number, whatever object it arrived as.

    Records built by ``pyxtal_notation_to_sites`` carry pymatgen ``Element``;
    a processor restored from JSON carries symbols; older caches carried the
    ``"Element Fe"`` repr.  All three denote the same atom and must produce the
    same key, or a gene would be novel against a reference that holds it.
    """
    number = getattr(element, "Z", None)
    if isinstance(number, (int, np.integer)):
        return int(number)
    from pymatgen.core import Element  # noqa: PLC0415

    text = str(element)
    if text.startswith("Element "):
        text = text[len("Element "):]
    return int(Element(text).Z)


def _site_digest(element, site_symmetry, enumeration) -> bytes:
    hasher = hashlib.blake2b(digest_size=_DIGEST_BYTES)
    hasher.update(_SITE_TAG)
    hasher.update(struct.pack("<q", _atomic_number(element)))
    symmetry = site_symmetry if isinstance(site_symmetry, bytes) else str(site_symmetry).encode()
    hasher.update(struct.pack("<I", len(symmetry)))
    hasher.update(symmetry)
    hasher.update(struct.pack("<q", int(enumeration)))
    return hasher.digest()


def _variant_digest(elements, site_symmetries, enumeration) -> bytes:
    """One equivalent enumeration, as a multiset of sites.

    Sites are hashed individually and the digests sorted, which is what makes
    the result independent of the order the sites happen to be listed in.  The
    count of each repeated site is folded in, so two occupied orbits of the same
    kind do not collapse into one -- ``Counter``, not ``set``, exactly as the
    fingerprint has it.
    """
    digests = sorted(
        _site_digest(element, symmetry, enum)
        for element, symmetry, enum in zip(elements, site_symmetries, enumeration)
    )
    hasher = hashlib.blake2b(digest_size=_DIGEST_BYTES)
    hasher.update(_VARIANT_TAG)
    hasher.update(struct.pack("<I", len(digests)))
    for digest in digests:
        hasher.update(digest)
    return hasher.digest()


def gene_key(record) -> tuple[int, int]:
    """The canonical 128-bit key of one Wyckoff record, as a pair of ``int64``.

    Args:
        record: A mapping with ``spacegroup_number``, ``elements``,
            ``site_symmetries_augmented`` and ``sites_enumeration_augmented`` -- what
            ``GeneFingerprinter.record`` produces and what
            :func:`~wyckoff_transformer.evaluation.novelty.record_to_augmented_fingerprint`
            reads.

    Returns:
        ``(low, high)``, both signed so they fit a ``torch.int64`` tensor.
    """
    elements = record["elements"]
    variants = sorted({
        _variant_digest(elements, symmetries, enumeration)
        for symmetries, enumeration in augmented_variants(record)
    })
    hasher = hashlib.blake2b(digest_size=_DIGEST_BYTES)
    hasher.update(_GENE_TAG)
    hasher.update(struct.pack("<H", GENE_HASH_VERSION))
    hasher.update(struct.pack("<q", int(record["spacegroup_number"])))
    hasher.update(struct.pack("<I", len(variants)))
    for digest in variants:
        hasher.update(digest)
    digest = hasher.digest()
    return (
        int.from_bytes(digest[:8], "little", signed=True),
        int.from_bytes(digest[8:], "little", signed=True),
    )


def gene_keys(records: Iterable) -> torch.Tensor:
    """``[N, 2]`` ``int64`` keys for a sequence of records, in their own order."""
    keys = [gene_key(record) for record in records]
    if not keys:
        return torch.empty((0, 2), dtype=torch.int64)
    return torch.tensor(keys, dtype=torch.int64)


# --------------------------------------------------------------------------- #
# Uniqueness
# --------------------------------------------------------------------------- #
def unique_representatives(keys: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Collapse a cohort onto one representative per distinct gene.

    Args:
        keys: ``[N, 2]`` from :func:`gene_keys`.

    Returns:
        ``(representative, counts)``.  *representative* is the index of the
        first row carrying each distinct key, ascending; *counts* is how many
        rows carried it, aligned with *representative*.  First occurrence, not
        an arbitrary one, so the choice matches
        :func:`~wyckoff_transformer.evaluation.protocol.screen_genes` and a
        cohort's representative does not move when the sort changes.
    """
    if keys.numel() == 0:
        empty = torch.empty(0, dtype=torch.int64)
        return empty, empty
    _, inverse, counts = torch.unique(keys, dim=0, return_inverse=True, return_counts=True)
    positions = torch.arange(keys.shape[0], dtype=torch.int64)
    first = torch.full((int(counts.shape[0]),), keys.shape[0], dtype=torch.int64)
    first.scatter_reduce_(0, inverse, positions, reduce="amin")
    order = torch.argsort(first)
    return first[order], counts[order]


# --------------------------------------------------------------------------- #
# Novelty
# --------------------------------------------------------------------------- #
class GeneKeyTable:
    """A sorted table of gene keys with a vectorised membership test.

    The low words are sorted and **unique** -- checked when the table is built,
    because the whole point of the second word is to make a false positive
    impossible, and a duplicated low word would hide one behind the other in a
    single ``searchsorted``.  With them unique, membership is one
    ``searchsorted`` plus two comparisons, and is exact.
    """

    def __init__(self, low: torch.Tensor, high: torch.Tensor, meta: Optional[dict] = None) -> None:
        if low.shape != high.shape or low.dim() != 1:
            raise ValueError("low and high must be matching 1-D tensors")
        self.low = low.to(torch.int64).contiguous()
        self.high = high.to(torch.int64).contiguous()
        self.meta = dict(meta or {})

    def __len__(self) -> int:
        return int(self.low.shape[0])

    @classmethod
    def from_keys(cls, keys: torch.Tensor, meta: Optional[dict] = None) -> "GeneKeyTable":
        """Build from ``[N, 2]`` keys, deduplicating and sorting them."""
        if keys.numel() == 0:
            return cls(torch.empty(0, dtype=torch.int64), torch.empty(0, dtype=torch.int64), meta)
        keys = torch.unique(keys.to(torch.int64), dim=0)
        order = torch.argsort(keys[:, 0])
        low, high = keys[order, 0].contiguous(), keys[order, 1].contiguous()
        if low.numel() > 1 and bool((low[1:] == low[:-1]).any()):
            raise RuntimeError(
                "Two distinct gene keys share a low word. At 64 bits over a few million "
                "genes this has probability ~1e-6, so it is far more likely that the "
                "encoding is wrong than that this is the collision. Do not work around "
                "it by widening the comparison."
            )
        return cls(low, high, meta)

    def to(self, device) -> "GeneKeyTable":
        return GeneKeyTable(self.low.to(device), self.high.to(device), self.meta)

    def contains(self, keys: torch.Tensor) -> torch.Tensor:
        """``[N]`` boolean: is each of ``[N, 2]`` *keys* in the table?"""
        if keys.dim() != 2 or keys.shape[1] != 2:
            raise ValueError(f"keys must be [N, 2], got {tuple(keys.shape)}")
        if len(self) == 0 or keys.shape[0] == 0:
            return torch.zeros(keys.shape[0], dtype=torch.bool, device=keys.device)
        low = keys[:, 0].to(self.low.device, torch.int64).contiguous()
        high = keys[:, 1].to(self.high.device, torch.int64).contiguous()
        position = torch.searchsorted(self.low, low)
        in_range = position < len(self)
        safe = position.clamp(max=len(self) - 1)
        hit = in_range & (self.low[safe] == low) & (self.high[safe] == high)
        return hit.to(keys.device)

    # ------------------------------------------------------------------ #
    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            low=self.low.numpy(),
            high=self.high.numpy(),
            version=np.array(GENE_HASH_VERSION),
            meta=np.array(repr(self.meta)),
        )
        return path

    @classmethod
    def load(cls, path: Path) -> "GeneKeyTable":
        with np.load(Path(path), allow_pickle=False) as payload:
            version = int(payload["version"])
            if version != GENE_HASH_VERSION:
                raise ValueError(
                    f"{path} holds version {version} keys and this is version "
                    f"{GENE_HASH_VERSION}. Rebuild it: keys from two encodings cannot "
                    "be compared, and a gene would be scored novel against a reference "
                    "that holds it.")
            meta = {}
            if "meta" in payload:
                import ast  # noqa: PLC0415

                try:
                    meta = ast.literal_eval(str(payload["meta"]))
                except (ValueError, SyntaxError):
                    meta = {}
            return cls(
                torch.from_numpy(payload["low"]),
                torch.from_numpy(payload["high"]),
                meta,
            )


#: Columns :func:`record_to_augmented_fingerprint` and :func:`gene_key` read.
#:
#: ``site_symmetries_augmented`` rather than ``site_symmetries``: a relabelling
#: can change the oriented symbol, so the symbol has to travel with the index.
#: See ``docs/wyckoff_augmentation_audit.md``.
_RECORD_COLUMNS = (
    "spacegroup_number",
    "elements",
    "site_symmetries_augmented",
    "sites_enumeration_augmented",
)


def keys_from_frame(frame) -> torch.Tensor:
    """Keys for every row of a Wyckoff-representation DataFrame.

    Zips the raw columns rather than going through ``DataFrame.apply``, for the
    reason given in ``evaluation/protocol._frame_fingerprints``: on a
    four-million-row reference, building a Series per row dominates.
    """
    columns = [frame[name].values for name in _RECORD_COLUMNS]
    keys = [gene_key(dict(zip(_RECORD_COLUMNS, values))) for values in zip(*columns)]
    if not keys:
        return torch.empty((0, 2), dtype=torch.int64)
    return torch.tensor(keys, dtype=torch.int64)


def build_reference_table(
    cache: Path,
    splits: Sequence[str],
    output: Optional[Path] = None,
) -> GeneKeyTable:
    """Key every row of the reference archive and build the table.

    One streaming pass over the same cache
    :func:`~wyckoff_transformer.evaluation.protocol.load_reference_fingerprints`
    reads.  The result is two orders of magnitude smaller than the fingerprint
    set it replaces, so unlike that set it is worth keeping on a GPU.
    """
    from wyckoff_transformer.dataset_cache import (  # noqa: PLC0415
        cache_exists, iter_splits, resolve_cache)

    cache = resolve_cache(cache)
    if not cache_exists(cache):
        raise FileNotFoundError(f"No Wyckoff gene cache in {cache}")

    chunks, rows = [], 0
    # A split at a time, and only the four columns a key is made of: the
    # reference is ~25 GB in full and the keys are ~16 bytes a row, so reading
    # the whole thing is what made this stage the memory ceiling it does not
    # need to be.
    for split, frame in iter_splits(cache, list(splits), columns=_RECORD_COLUMNS):
        chunks.append(keys_from_frame(frame))
        rows += len(frame)
        logger.info("Keyed split %s: %d rows", split, len(frame))
        del frame
        gc.collect()
    keys = torch.cat(chunks) if chunks else torch.empty((0, 2), dtype=torch.int64)

    table = GeneKeyTable.from_keys(keys, meta={
        "cache": str(cache), "splits": list(splits), "rows": rows,
        "version": GENE_HASH_VERSION,
    })
    logger.info("%d distinct gene keys from %d rows", len(table), rows)
    if output is not None:
        table.save(output)
        logger.info("Saved to %s", output)
    return table


def default_key_table_path(cache: Path, splits: Sequence[str]) -> Path:
    """Beside the reference it was built from, named for the splits it covers."""
    from wyckoff_transformer.dataset_cache import as_cache_dir  # noqa: PLC0415

    stem = Path(GENE_KEY_TABLE_FILE).stem
    suffix = "" if tuple(splits) == ("train", "val", "test") else "_" + "_".join(splits)
    return as_cache_dir(cache) / f"{stem}{suffix}.npz"
