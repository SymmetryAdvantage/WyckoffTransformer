"""The LeMat-Bulk reference that structure novelty and uniqueness are judged against.

Novelty is decided in two stages, by :class:`~wyckoff_transformer.evaluation.novelty.NoveltyFilter`:
the augmented Wyckoff fingerprint first, then ``StructureMatcher`` on the
reference entries that share it.  The fingerprint is a necessary condition that
costs nothing -- a generated structure whose fingerprint appears nowhere in
LeMat-Bulk cannot be one of its entries -- and the matcher resolves the rest,
where a shared fingerprint means only that two structures have the same space
group and the same elements on the same Wyckoff orbits, not the same geometry.

That split is what makes the reference tractable.  LeMat-Bulk has 4.2M entries
and ``NoveltyFilter`` wants a ``Structure`` per row, which is far too much to
hold; but only the entries whose fingerprint collides with something we
generated can ever reach the matcher.  For a 2500-gene run that is a few hundred
fingerprints, so the reference is built per run: one streaming pass over the
Wyckoff cache to find the colliding ``immutable_id``s, then one chunked pass
over the CIF export to read just those structures.

This replaces the earlier BAWL path, which compared BAWL hashes against a
parquet of *augmented Wyckoff* fingerprints and so reported every structure as
novel.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Optional, Sequence

import pandas as pd

logger = logging.getLogger(__name__)

#: LeMat-Bulk export carrying ``immutable_id`` and ``cif``.  The Wyckoff cache
#: holds no geometry, so the structures the matcher needs come from here.
DEFAULT_LEMAT_CIF_CSV = Path("data/lemat-bulk/lemat_pbe.csv.gz")

#: Rows per chunk while scanning that export.  It is ~1 GB gzipped.
DEFAULT_CHUNKSIZE = 50_000


def collect_reference_ids(
    fingerprints: Iterable[tuple],
    cache: Optional[Path] = None,
    splits: Optional[Sequence[str]] = None,
) -> dict[tuple, list[str]]:
    """Find the LeMat-Bulk entries whose fingerprint is one of *fingerprints*.

    One streaming pass over the Wyckoff cache.  The cache is kept out of the
    return value: only the ids matter, and holding 4.2M fingerprints to answer
    a few hundred questions is what this function exists to avoid.

    Args:
        fingerprints: The fingerprints to look for, from
            :meth:`~wyckoff_transformer.evaluation.protocol.GeneFingerprinter.fingerprint`.
        cache: Pickle of split name -> DataFrame in the Wyckoff representation,
            indexed by ``immutable_id``.  Defaults to the protocol's.
        splits: Which splits count as known.  Defaults to all of them.

    Returns:
        Fingerprint -> the ``immutable_id``s carrying it.  Fingerprints with no
        LeMat-Bulk entry are absent, not empty.
    """
    from wyckoff_transformer.evaluation.protocol import (
        _FINGERPRINT_COLUMNS,
        DEFAULT_REFERENCE_CACHE,
        DEFAULT_REFERENCE_SPLITS,
    )
    from wyckoff_transformer.evaluation.novelty import record_to_augmented_fingerprint

    cache = Path(cache) if cache is not None else DEFAULT_REFERENCE_CACHE
    splits = splits if splits is not None else DEFAULT_REFERENCE_SPLITS

    wanted = set(fingerprints)
    if not wanted:
        return {}
    if not cache.is_file():
        raise FileNotFoundError(
            f"No LeMat-Bulk gene cache at {cache}. Build it with the dataset "
            f"caching scripts, or pass --reference-cache."
        )

    frames = pd.read_pickle(cache)
    missing = [s for s in splits if s not in frames]
    if missing:
        raise KeyError(f"{cache} has no split(s) {missing}; found {sorted(frames)}")

    hits: dict[tuple, list[str]] = {}
    for split in splits:
        frame = frames[split]
        columns = [frame[name].values for name in _FINGERPRINT_COLUMNS]
        for immutable_id, values in zip(frame.index.values, zip(*columns)):
            fingerprint = record_to_augmented_fingerprint(
                dict(zip(_FINGERPRINT_COLUMNS, values))
            )
            if fingerprint in wanted:
                hits.setdefault(fingerprint, []).append(str(immutable_id))
        logger.info(
            "Scanned reference split %s (%d rows): %d of %d fingerprints matched so far",
            split, len(frame), len(hits), len(wanted),
        )
    return hits


def load_reference_structures(
    ids: Iterable[str],
    lemat_cif_csv: Optional[Path] = None,
    chunksize: int = DEFAULT_CHUNKSIZE,
) -> dict[str, "object"]:
    """Read the named LeMat-Bulk structures out of the CIF export.

    Args:
        ids: ``immutable_id``s to read.
        lemat_cif_csv: The export.  Defaults to :data:`DEFAULT_LEMAT_CIF_CSV`.
        chunksize: Rows per chunk; the file is ~1 GB gzipped.

    Returns:
        ``immutable_id`` -> ``pymatgen`` ``Structure``.  Ids the export does not
        carry, and CIFs that fail to parse, are absent: a reference entry we
        cannot read is one we cannot match against, and dropping it can only
        make a structure look *more* novel, never less.
    """
    from pymatgen.core import Structure

    lemat_cif_csv = (
        Path(lemat_cif_csv) if lemat_cif_csv is not None else DEFAULT_LEMAT_CIF_CSV
    )
    wanted = set(ids)
    if not wanted:
        return {}
    if not lemat_cif_csv.is_file():
        raise FileNotFoundError(
            f"No LeMat-Bulk CIF export at {lemat_cif_csv}. Structure novelty "
            f"needs the reference geometries; pass --lemat-cif-csv."
        )

    structures: dict[str, object] = {}
    for chunk in pd.read_csv(
        lemat_cif_csv, chunksize=chunksize, usecols=["immutable_id", "cif"]
    ):
        for immutable_id, cif in zip(chunk["immutable_id"], chunk["cif"]):
            if immutable_id not in wanted:
                continue
            try:
                structures[immutable_id] = Structure.from_str(cif, fmt="cif")
            except Exception as exc:
                logger.warning("Reference %s: unreadable CIF (%s)", immutable_id, exc)
        if len(structures) >= len(wanted):
            break

    if len(structures) < len(wanted):
        logger.warning(
            "Read %d of %d reference structures from %s",
            len(structures), len(wanted), lemat_cif_csv,
        )
    return structures


def build_novelty_reference(
    fingerprints: Iterable[tuple],
    cache: Optional[Path] = None,
    splits: Optional[Sequence[str]] = None,
    lemat_cif_csv: Optional[Path] = None,
    chunksize: int = DEFAULT_CHUNKSIZE,
) -> pd.DataFrame:
    """Assemble the reference frame :class:`NoveltyFilter` expects.

    Only fingerprints that actually occur in LeMat-Bulk contribute rows.  A
    generated structure whose fingerprint is absent from the frame is novel
    without any matching, which is exactly the cheap first stage.

    Args:
        fingerprints: Fingerprints of the generated genes, i.e. every
            fingerprint that could possibly collide.
        cache: Wyckoff-representation cache; see :func:`collect_reference_ids`.
        splits: Splits to treat as known.
        lemat_cif_csv: CIF export; see :func:`load_reference_structures`.
        chunksize: Rows per chunk while scanning it.

    Returns:
        A frame indexed by ``immutable_id`` with ``fingerprint`` and
        ``structure`` columns.  Empty when nothing collides.
    """
    hits = collect_reference_ids(fingerprints, cache=cache, splits=splits)
    if not hits:
        logger.info("No generated fingerprint occurs in LeMat-Bulk; nothing to match.")
        return pd.DataFrame(columns=["fingerprint", "structure"])

    ids = [i for candidates in hits.values() for i in candidates]
    logger.info(
        "%d fingerprints collide with LeMat-Bulk, over %d reference entries",
        len(hits), len(ids),
    )
    structures = load_reference_structures(
        ids, lemat_cif_csv=lemat_cif_csv, chunksize=chunksize
    )

    rows = [
        {"immutable_id": i, "fingerprint": fingerprint, "structure": structures[i]}
        for fingerprint, candidates in hits.items()
        for i in candidates
        if i in structures
    ]
    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(columns=["fingerprint", "structure"])
    return frame.set_index("immutable_id")
