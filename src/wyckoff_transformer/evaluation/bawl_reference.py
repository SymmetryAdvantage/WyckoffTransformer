"""The LeMat-Bulk BAWL fingerprint set that structure novelty is judged against.

This is data, not logic: LeMat-GenBench ships a parquet of every distinct BAWL
fingerprint in LeMat-Bulk (4.7M of them), and novelty is membership in that set.
Reading it directly replaces the benchmark's ``ReferenceFingerprintDatabase``,
along with its fallback path that looks for a ``.pkl`` the distribution does not
contain.

The parquet is ~272 MB and holds a single ``values`` column.  Loading it costs
about as much as reading it from disk, so the set is cached to a compact pickle
on first use, exactly as the gene-level reference is.
"""
from __future__ import annotations

import gzip
import logging
import pickle
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

#: Column holding the fingerprints in LeMat-GenBench's parquet.
FINGERPRINT_COLUMN = "values"

#: Where the parquet sits inside a LeMat-GenBench checkout.
GENBENCH_RELATIVE_PATH = Path("data/augmented_fingerprints/unique_fingerprints.parquet")


def load_bawl_reference(
    parquet: Path,
    cache: Optional[Path] = None,
) -> set[str]:
    """Load the set of known BAWL fingerprints.

    Args:
        parquet: ``unique_fingerprints.parquet``.  In a LeMat-GenBench checkout
            this is at :data:`GENBENCH_RELATIVE_PATH`; copy it somewhere stable
            if you would rather not depend on that checkout at all.
        cache: Optional gzipped-pickle cache, written on first load and reused
            afterwards.

    Returns:
        Every distinct BAWL fingerprint in LeMat-Bulk.
    """
    if cache is not None and Path(cache).is_file():
        with gzip.open(cache, "rb") as handle:
            fingerprints = pickle.load(handle)
        logger.info("Loaded %d BAWL fingerprints from %s", len(fingerprints), cache)
        return fingerprints

    parquet = Path(parquet)
    if not parquet.is_file():
        raise FileNotFoundError(
            f"No BAWL reference fingerprints at {parquet}. It ships with "
            f"LeMat-GenBench at {GENBENCH_RELATIVE_PATH}; pass "
            f"--bawl-reference to point at a copy."
        )

    import pyarrow.parquet as pq

    table = pq.read_table(parquet, columns=[FINGERPRINT_COLUMN])
    fingerprints = set(table.column(FINGERPRINT_COLUMN).to_pylist())
    logger.info("Loaded %d BAWL fingerprints from %s", len(fingerprints), parquet)

    if cache is not None:
        cache = Path(cache)
        cache.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(cache, "wb") as handle:
            pickle.dump(fingerprints, handle, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("Cached BAWL fingerprints to %s", cache)
    return fingerprints
