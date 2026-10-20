"""The PBE hull reference, built from LeMat-Bulk on the HuggingFace Hub.

``wyformer-gene-screen`` compares predicted formation energies with the hull of
the per-row table ``scripts/process_lemat.py`` writes: ``full_formula``,
``chemsys`` and ``energy_corrected`` -- the raw PBE/PBE+U total energy, the name
notwithstanding -- indexed by ``immutable_id``. Outside a checkout with the data
store there is no such table, so this builds the same one from the dataset's
parquet shards. It reads only three columns, about 6% of the 2.8 GB, and caches
the result; the first build takes about a minute.

    reference = load_hf_reference("kazeevn/LeMat-Bulk-fixed")
"""
from __future__ import annotations

import logging
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from wyckoff_transformer.energy_fields import EnergyField, EnergySource

logger = logging.getLogger(__name__)

DEFAULT_REPO = "kazeevn/LeMat-Bulk-fixed"
#: The config whose rows the hull is built from, as in docs/lemat_bulk_pipeline.md.
CONFIG = "compatible_pbe"
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "wyckoff_transformer" / "hull_references"

#: What the formation energies of a reference built from each known revision mean.
#: Each must equal the manifest field of the table the data-store pipeline builds
#: from the same revision; ``test_hf_reference`` pins the two together. A revision
#: not listed here has unknown provenance and is refused by the compatibility check.
KNOWN_REVISIONS = {
    (DEFAULT_REPO, "bf8f0b3112a0c522ccd7f90da9748e9240ee80a5"): EnergyField(
        quantity="formation_energy",
        extent="per_atom",
        source=EnergySource(method="dft", dft_settings="PBE_MP", correction="none"),
        reference="lemat_bulk_fixed_pbe@bf8f0b31",
    ),
}
#: The revision a bare repo name resolves to, so a moved ``main`` does not
#: silently change the hull.
DEFAULT_REVISIONS = {repo: revision for repo, revision in KNOWN_REVISIONS}

_COLUMNS = ("immutable_id", "species_at_sites", "energy")


def parse_spec(spec: str) -> tuple[str, str | None]:
    """Split ``repo[@revision]``; a bare repo takes its pinned revision, if it has one."""
    repo, _, revision = spec.partition("@")
    return repo, revision or DEFAULT_REVISIONS.get(repo)


def reference_field(repo: str, revision: str) -> EnergyField | None:
    """The formation-energy field of a reference built from *revision*, if known.

    A commit may be abbreviated, as git allows, to seven characters or more.
    """
    for (known_repo, known_revision), field in KNOWN_REVISIONS.items():
        if (known_repo == repo and revision is not None and len(revision) >= 7
                and known_revision.startswith(revision)):
            return field
    return None


def _formula_and_chemsys(species: list[str]) -> tuple[str, str]:
    counts = Counter(species)
    formula = " ".join(f"{element}{count}" for element, count in sorted(counts.items()))
    return formula, "-".join(sorted(counts))


def _read_shard(filesystem, path: str) -> pd.DataFrame:
    import pyarrow.parquet as pq

    # pre_buffer coalesces the column chunks into a few large range requests; without
    # it a shard costs one round trip per page.
    with filesystem.open(path, "rb", block_size=1 << 20, cache_type="none") as handle:
        table = pq.ParquetFile(handle, pre_buffer=True).read(columns=list(_COLUMNS))
    frame = table.to_pandas()
    formulas = [_formula_and_chemsys(species) for species in frame["species_at_sites"]]
    return pd.DataFrame({
        "immutable_id": frame["immutable_id"],
        "full_formula": [formula for formula, _ in formulas],
        "chemsys": [chemsys for _, chemsys in formulas],
        "energy_corrected": frame["energy"],
    })


def load_hf_reference(
    repo: str = DEFAULT_REPO,
    revision: str | None = None,
    cache_dir: Path | None = None,
    workers: int = 17,
) -> pd.DataFrame:
    """The hull reference table of *repo* at *revision*, indexed by ``immutable_id``.

    Args:
        revision: A branch, tag or commit. Resolved to a commit before anything is
            read, so the cache is keyed by what was actually read.
        cache_dir: Where the built table is kept; ``~/.cache/wyckoff_transformer/
            hull_references`` by default.
        workers: Shards read in parallel. Each read is latency-bound, not
            bandwidth-bound, so this is about round trips, not cores.
    """
    from huggingface_hub import HfApi, HfFileSystem

    sha = HfApi().dataset_info(repo, revision=revision).sha
    cache_dir = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
    cached = cache_dir / f"{repo.replace('/', '__')}@{sha}.parquet"
    if cached.is_file():
        return pd.read_parquet(cached).set_index("immutable_id")

    filesystem = HfFileSystem()
    shards = sorted(filesystem.glob(f"datasets/{repo}@{sha}/{CONFIG}/*.parquet"))
    if not shards:
        raise FileNotFoundError(f"{repo}@{sha} has no {CONFIG}/*.parquet shards")
    logger.info("Building the hull reference from %d shards of %s@%s", len(shards), repo, sha[:8])
    with ThreadPoolExecutor(max_workers=workers) as pool:
        frames = list(pool.map(lambda path: _read_shard(filesystem, path), shards))
    reference = pd.concat(frames, ignore_index=True)
    reference = reference.dropna(subset=["energy_corrected"])

    cache_dir.mkdir(parents=True, exist_ok=True)
    partial = cached.with_suffix(".partial")
    reference.to_parquet(partial, index=False)
    partial.replace(cached)
    logger.info("Cached %d reference rows at %s", len(reference), cached)
    return reference.set_index("immutable_id")
