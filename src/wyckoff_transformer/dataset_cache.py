"""The on-disk dataset cache: one Parquet file per split.

A dataset cache is the directory ``cache_root() / <dataset>``.  The Wyckoff
records live in ``train.parquet``, ``val.parquet`` and ``test.parquet`` beside
the tokenised ``tensors/``, the ``tokenisers/`` and whatever else is derived
from them.

Until 2026-09-22 the records were a single ``data.pkl.gz``: a pickle of
``{split: DataFrame}``.  That format is still **read** -- every cache already on
disk is in it -- but nothing writes it any more.  Three reasons:

* ``pickle.load`` executes whatever the file says.  Caches are copied between
  machines, pulled from the store and shared with collaborators, so the cache
  was a code-execution channel for anyone who could write to any of them.
* It is slow and all-or-nothing.  ``lemat_bulk_fmax1_stress`` takes 83 s to
  unpickle 5.3M rows, and a caller that wants one split or three columns pays
  all of it: the post-training evaluation used to load 399 MB to read ``test``.
  The same data as Parquet reads in 0.3 s, one split in 0.04 s, and a few
  columns in 0.1 s.
* Nothing outside this repository can read it.  Parquet opens in pandas,
  polars, DuckDB and Arrow.

The cost is about 8% more disk: the energy columns are float64 noise, which
compresses no better here than in gzip, and Parquet's per-column framing is not
free.  The read speed is worth it.

Columns hold Python objects that Parquet has no types for -- ``pymatgen``
``Element``, ``Counter``, the ``frozenset`` of augmentation variants.  They are
stored as strings, maps and nested lists, and the *container* each one came from
is recorded in the file's schema metadata, so a load returns the same objects
the cache was built from.  See :func:`column_spec`.

A set is written **sorted**, so one cache's bytes follow its contents rather
than the insertion history of the frame it came from.  One consequence is worth
knowing: the tokeniser lays a row's augmentation variants out in the order the
set yields them, so re-tokenising a cache converted from a pickle puts the same
variants in a different order than the pickle would have.  Training draws a
variant uniformly at random (``cascade/dataset.py``), so this changes no
distribution; the novelty fingerprints and gene keys are order-independent by
construction -- ``record_to_augmented_fingerprint`` builds a frozenset and
``gene_key`` sorts; and the token ids are unaffected, because
``Tokenizer.from_token_set`` numbers ``sorted(all_tokens)``.  The tensors are
simply not bit-identical.  An already tokenised ``tensors/`` file is untouched
by any of this.
"""
from __future__ import annotations

import gc
import gzip
import json
import subprocess
import logging
import pickle
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pymatgen.core import Element

from wyckoff_transformer.dataset_manifest import warn_if_obsolete_dataset
from wyckoff_transformer.paths import cache_root, resolve_store_path

logger = logging.getLogger(__name__)

#: The splits a dataset is cached in, in the order they are reported.
SPLITS = ("train", "val", "test")

#: The superseded single-file format, still read when no Parquet is present.
LEGACY_CACHE_NAME = "data.pkl.gz"

#: Where the column specifications ride in the Parquet schema.
METADATA_KEY = b"wyformer_dataset_cache"

#: Bumped when a change makes an older file unreadable, which has not happened.
FORMAT_VERSION = 1

#: The field an unnamed DataFrame index is stored under.
INDEX_FIELD = "__index__"

#: Rows encoded and written at a time.  The whole frame is already in memory;
#: this bounds the *encoded* copy, which is what made the LeMat-Bulk migration
#: need a big machine.
WRITE_CHUNK_ROWS = 200_000

#: Rows per Parquet row group.  Small enough that a reader can skip most of a
#: file on a predicate, large enough that the per-group overhead is noise.
ROW_GROUP_ROWS = 100_000




class CacheFormatError(ValueError):
    """A cache file does not hold what this module wrote."""


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def dataset_cache_dir(dataset: str) -> Path:
    """The cache directory of *dataset*, whether or not it exists yet."""
    return cache_root() / dataset


def split_path(cache: Path, split: str) -> Path:
    """The Parquet file holding *split* of the cache directory *cache*."""
    return Path(cache) / f"{split}.parquet"


def legacy_path(cache: Path) -> Path:
    """The superseded pickle of the cache directory *cache*."""
    return Path(cache) / LEGACY_CACHE_NAME


def as_cache_dir(cache: str | Path) -> Path:
    """The cache directory *cache* names, without resolving it into the store.

    A directory is the cache itself.  A path ending in ``data.pkl.gz`` names the
    superseded file, and its directory is returned instead: the two spell the
    same cache, and a default or a ``--reference-cache`` written before the
    format changed must keep working.
    """
    cache = Path(cache)
    return cache.parent if cache.name == LEGACY_CACHE_NAME else cache


def resolve_cache(cache: str | Path) -> Path:
    """The directory *cache* names, re-rooted into the store."""
    return as_cache_dir(resolve_store_path(Path(cache)))


def available_splits(cache: str | Path) -> tuple[str, ...]:
    """Which splits *cache* holds, Parquet first, then the superseded pickle."""
    cache = resolve_cache(cache)
    present = {path.stem for path in cache.glob("*.parquet")}
    if present:
        return tuple(split for split in SPLITS if split in present) + tuple(
            sorted(present.difference(SPLITS)))
    if legacy_path(cache).is_file():
        with gzip.open(legacy_path(cache), "rb") as handle:
            frames = pickle.load(handle)
        return tuple(split for split in SPLITS if split in frames) + tuple(
            sorted(set(frames).difference(SPLITS)))
    return ()


def cache_exists(cache: str | Path) -> bool:
    """Whether *cache* holds readable records in either format."""
    cache = resolve_cache(cache)
    return any(cache.glob("*.parquet")) or legacy_path(cache).is_file()


# ---------------------------------------------------------------------------
# Column specifications
# ---------------------------------------------------------------------------
#
# A specification is either a scalar name -- "str", "int", "float", "element" --
# or a container ``{"container": ..., "item": ...}``, recursively.  It is JSON
# so it can ride in the Parquet metadata and be read by anything.

_SCALAR_ARROW = {
    "str": pa.string(),
    "element": pa.string(),
    "int": pa.int32(),
    "float": pa.float64(),
}

_CONTAINERS = {"list": list, "tuple": tuple, "frozenset": frozenset, "set": set}


def _first_present(values: Iterable) -> Any:
    """The first value that is neither ``None`` nor NaN, or ``None``."""
    for value in values:
        if value is None:
            continue
        if isinstance(value, float) and value != value:
            continue
        return value
    return None


def column_spec(values: pd.Series) -> Optional[dict | str]:
    """How the objects in *values* are stored, or ``None`` if Arrow knows.

    Inferred from the first row that has a value, because a column holds one
    kind of record throughout -- it is built one way, by
    :func:`wyckoff_transformer.data.structure_to_sites`.  A column of scalars
    returns ``None``: Arrow's own inference is right for those, and recording a
    specification for them would only be a second place to be wrong.
    """
    sample = _first_present(values.to_numpy())
    if isinstance(sample, Counter) or (isinstance(sample, dict) and sample):
        key = _scalar_spec(_first_present(sample.keys()))
        if key is None:
            raise CacheFormatError(f"Cannot store a mapping keyed by {type(_first_present(sample.keys()))}")
        return {"container": "counter", "key": key}
    if isinstance(sample, (list, tuple, set, frozenset, np.ndarray)):
        container = type(sample).__name__
        if isinstance(sample, np.ndarray):
            container = "list"
        if container not in _CONTAINERS:
            raise CacheFormatError(f"Cannot store a {container} column")
        return {"container": container, "item": _item_spec(sample)}
    return None


def _item_spec(container) -> dict | str:
    """The specification of what *container* holds, defaulting to ``int``."""
    sample = _first_present(container)
    nested = None
    if isinstance(sample, (list, tuple, set, frozenset, np.ndarray)):
        nested = column_spec(pd.Series([sample], dtype=object))
    if nested is not None:
        return nested
    scalar = _scalar_spec(sample)
    # An empty container carries no type, and an empty list round-trips through
    # any of them, so the default only has to be a valid one.
    return "int" if scalar is None else scalar


def _scalar_spec(value) -> Optional[str]:
    if isinstance(value, Element):
        return "element"
    if isinstance(value, str):
        return "str"
    if isinstance(value, (bool, np.bool_)):
        return None
    if isinstance(value, (int, np.integer)):
        return "int"
    if isinstance(value, (float, np.floating)):
        return "float"
    return None


def _scalar_arrow_type(values: pd.Series) -> pa.DataType:
    """The Arrow type of a column Arrow can infer, without encoding it twice."""
    if values.dtype != object:
        return pa.from_numpy_dtype(values.dtype)
    return pa.array(values.to_numpy()).type


def _arrow_type(spec: dict | str) -> pa.DataType:
    if isinstance(spec, str):
        return _SCALAR_ARROW[spec]
    if spec["container"] == "counter":
        return pa.map_(_arrow_type(spec["key"]), pa.int64())
    return pa.list_(_arrow_type(spec["item"]))


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------

def _encode_value(value, spec: dict | str):
    if value is None:
        return None
    if isinstance(spec, str):
        if spec == "element":
            return str(value)
        if spec == "int":
            return int(value)
        return value
    if spec["container"] == "counter":
        return [(_encode_value(key, spec["key"]), int(count)) for key, count in value.items()]
    item = spec["item"]
    values = _ordered(value) if spec["container"] in ("set", "frozenset") else value
    if item == "str":
        return list(values)
    return [_encode_value(each, item) for each in values]


def _ordered(values):
    """A set's elements in a fixed order, so the bytes follow the contents.

    A set has no order to preserve -- ``pickle`` does not round-trip one
    either -- and writing it as it happens to iterate would make one cache's
    file depend on the insertion history of the frame it was written from.
    Sorting is only possible when the elements compare, which for the one set
    column there is -- the augmentation variants, tuples of integers -- they do.
    """
    try:
        return sorted(values)
    except TypeError:
        return list(values)


def _encode_column(values: pd.Series, spec: Optional[dict | str]) -> pa.Array:
    if spec is None:
        return pa.array(values.to_numpy())
    return pa.array([_encode_value(value, spec) for value in values.to_numpy()],
                    type=_arrow_type(spec))


# ---------------------------------------------------------------------------
# Decoding
# ---------------------------------------------------------------------------

#: ``Element`` by symbol.  A load turns tens of millions of symbols back into
#: elements, and a plain dict of the whole table is the cheapest lookup there
#: is -- an enum call, or a dict with ``__missing__``, both cost more per hit
#: than the rest of the decode put together.
_ELEMENTS = {element.symbol: element for element in Element}


def _decode_value(value, spec: dict | str):
    """Restore one value.  The general case; :func:`_decode_column` is faster."""
    if value is None:
        return None
    if isinstance(spec, str):
        return _ELEMENTS[value] if spec == "element" else value
    if spec["container"] == "counter":
        key = spec["key"]
        return Counter({_decode_value(name, key): count for name, count in value})
    item = spec["item"]
    decoded = value if item in ("str", "int") else [_decode_value(each, item) for each in value]
    return _CONTAINERS[spec["container"]](decoded)


def _decode_column(column: pa.ChunkedArray, spec: Optional[dict | str]) -> list:
    """Restore one column's Python objects.

    The shapes the cache actually holds get a loop of their own, because the
    general recursion costs a Python call per value and there are tens of
    millions of them.  ``to_pylist`` already returns the right thing for a list
    of strings or integers, which is most of the columns; those are returned
    untouched.
    """
    if spec is None:
        return column
    rows = column.to_pylist()
    if isinstance(spec, str):
        return [_decode_value(row, spec) for row in rows]

    container = spec["container"]
    if container == "counter":
        if spec["key"] == "element":
            return [None if row is None else Counter({_ELEMENTS[name]: count for name, count in row})
                    for row in rows]
        if spec["key"] == "str":
            return [None if row is None else Counter(dict(row)) for row in rows]
        return [_decode_value(row, spec) for row in rows]

    item = spec["item"]
    build = _CONTAINERS[container]
    if item in ("str", "int", "float"):
        # to_pylist already produced a list of exactly these.
        if container == "list":
            return rows
        return [None if row is None else build(row) for row in rows]
    if item == "element":
        return [None if row is None else build([_ELEMENTS[name] for name in row])
                for row in rows]
    if isinstance(item, dict) and item["container"] in _CONTAINERS \
            and item["item"] in ("str", "int", "float"):
        inner = _CONTAINERS[item["container"]]
        return [None if row is None else build(map(inner, row)) for row in rows]
    return [_decode_value(row, spec) for row in rows]


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------

def _package_version() -> Optional[str]:
    try:
        return metadata.version("wyckoff-transformer")
    except Exception:                                   # noqa: BLE001 - never fail a write over this
        return None


def _git_state() -> tuple[Optional[str], Optional[bool]]:
    """``(commit, dirty)`` for the checkout this code was imported from.

    ``(None, None)`` when it was not imported from one -- an installed wheel has
    no git to ask, and a build record that invented a commit would be worse than
    one that admits it does not know.
    """
    here = Path(__file__).resolve().parent
    try:
        commit = subprocess.run(
            ["git", "-C", str(here), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10, check=True).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(here), "status", "--porcelain"],
            capture_output=True, text=True, timeout=30, check=True).stdout
    except (OSError, subprocess.SubprocessError):
        return None, None
    return (commit or None), bool(status.strip())


def provenance(tool: str, **options) -> dict:
    """A build record to hand :func:`save_cache`: what wrote this split, and how.

    The options a caller passes are the ones that change what lands in the
    file -- a cap, a tolerance, an ordering -- not the ones that only change how
    long it takes, such as the worker count. A reader comparing two caches wants
    to know whether they hold the same thing.

    ``dirty`` says the checkout had uncommitted changes, so ``commit`` does not
    fully describe the code that ran. It is recorded rather than refused:
    rebuilding a cache is not a result, and a five-hour job should not die over
    an unstaged file.
    """
    commit, dirty = _git_state()
    return {
        "tool": tool,
        "built": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "version": _package_version(),
        "commit": commit,
        "dirty": dirty,
        "options": dict(options),
    }


def _frame_metadata(frame: pd.DataFrame, specs: Mapping[str, Optional[dict | str]],
                    build: Optional[dict] = None) -> dict:
    index_field = frame.index.name or INDEX_FIELD
    if frame.index.name is not None and frame.index.name in frame.columns:
        raise CacheFormatError(
            f"The index is named {frame.index.name!r}, which is also a column; "
            "one of them would overwrite the other.")
    record = {
        "version": FORMAT_VERSION,
        "index_field": index_field,
        "index_name": frame.index.name,
        "columns": {name: spec for name, spec in specs.items() if spec is not None},
    }
    if build is not None:
        try:
            json.dumps(build)
        except (TypeError, ValueError) as exc:
            raise CacheFormatError(
                f"The build record does not survive JSON: {exc}. It rides in the "
                "file's schema metadata, so it has to be plain data.") from exc
        record["build"] = build
    return record


def _encode_chunk(frame: pd.DataFrame, specs, index_field: str, schema: pa.Schema) -> pa.Table:
    arrays = [pa.array(frame.index.to_numpy())]
    names = [index_field]
    for name in frame.columns:
        arrays.append(_encode_column(frame[name], specs[name]))
        names.append(name)
    return pa.Table.from_arrays(arrays, names=names).cast(schema)


def save_split(frame: pd.DataFrame, cache: str | Path, split: str,
               build: Optional[dict] = None) -> Path:
    """Write one split of *cache*, replacing whatever was there.

    The file appears whole or not at all: it is written beside its final name
    and renamed, so an interrupted rebuild leaves the previous cache readable
    rather than a truncated file that fails halfway through a training run.

    Args:
        frame: The split's records.
        cache: The cache directory to write into.
        split: Which split this is.
        build: What built it, from :func:`provenance`.  It rides in the file's
            own schema metadata rather than a file beside it, so a split copied
            to another machine takes its provenance with it, and splits built at
            different times each say so.  Nothing enforces it -- a split written
            without one simply records nothing, which is what
            :func:`build_info` then reports.
    """
    cache = resolve_cache(cache)
    cache.mkdir(parents=True, exist_ok=True)
    target = split_path(cache, split)
    specs = {name: column_spec(frame[name]) for name in frame.columns}
    file_metadata = _frame_metadata(frame, specs, build)
    index_field = file_metadata["index_field"]

    fields = [pa.field(index_field, _scalar_arrow_type(frame.index.to_series()))]
    for name in frame.columns:
        spec = specs[name]
        arrow_type = _arrow_type(spec) if spec is not None else _scalar_arrow_type(frame[name])
        fields.append(pa.field(name, arrow_type))
    schema = pa.schema(fields, metadata={METADATA_KEY: json.dumps(file_metadata).encode()})

    temporary = target.with_suffix(".parquet.partial")
    collecting = gc.isenabled()
    gc.disable()          # See the note in _read_parquet_split.
    try:
        with pq.ParquetWriter(temporary, schema, compression="zstd") as writer:
            for start in range(0, len(frame), WRITE_CHUNK_ROWS):
                chunk = frame.iloc[start:start + WRITE_CHUNK_ROWS]
                writer.write_table(_encode_chunk(chunk, specs, index_field, schema),
                                   row_group_size=ROW_GROUP_ROWS)
        temporary.replace(target)
    finally:
        if collecting:
            gc.enable()
        temporary.unlink(missing_ok=True)
    logger.info("Wrote %d rows to %s (%.1f MB)", len(frame), target,
                target.stat().st_size / 1e6)
    return target


def save_cache(frames: Mapping[str, pd.DataFrame], cache: str | Path,
               build: Optional[dict] = None) -> Path:
    """Write every split of *frames* into the cache directory *cache*.

    *build* is recorded in each split; pass a different one per split with
    :func:`save_split` when they were not built together.
    """
    cache = resolve_cache(cache)
    for split, frame in frames.items():
        save_split(frame, cache, split, build)
    if legacy_path(cache).is_file():
        logger.warning(
            "%s is now stale: %s is read in preference to it. Delete it once "
            "nothing on this machine still wants the old format.",
            legacy_path(cache), ", ".join(f"{split}.parquet" for split in frames))
    return cache


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------

def _read_metadata(schema: pa.Schema, path: Path) -> dict:
    raw = (schema.metadata or {}).get(METADATA_KEY)
    if raw is None:
        raise CacheFormatError(
            f"{path} has no {METADATA_KEY.decode()} metadata, so the Python types "
            "of its columns are unknown. It was not written by this module.")
    metadata = json.loads(raw)
    if metadata["version"] > FORMAT_VERSION:
        raise CacheFormatError(
            f"{path} is format version {metadata['version']}; this code reads "
            f"up to {FORMAT_VERSION}.")
    return metadata


def _read_parquet_split(path: Path, columns: Optional[Sequence[str]]) -> pd.DataFrame:
    parquet = pq.ParquetFile(path)
    metadata = _read_metadata(parquet.schema_arrow, path)
    index_field = metadata["index_field"]
    if columns is not None:
        wanted = [index_field] + [name for name in columns if name != index_field]
        missing = [name for name in wanted if name not in parquet.schema_arrow.names]
        if missing:
            raise KeyError(f"{path} has no column(s) {missing}; it holds "
                           f"{[n for n in parquet.schema_arrow.names if n != index_field]}")
    else:
        wanted = None
    table = parquet.read(columns=wanted)
    specs = metadata["columns"]
    index = pd.Index(table.column(index_field).to_pandas())
    # Assigned rather than passed: pd.Index keeps the Series name when given None.
    index.name = metadata["index_name"]
    data = {}
    # Decoding allocates one small container per value, which is exactly the
    # workload generational collection is worst at: it halves the time here and
    # collects nothing, since none of these objects are in a cycle.
    collecting = gc.isenabled()
    gc.disable()
    try:
        for name in table.schema.names:
            if name == index_field:
                continue
            decoded = _decode_column(table.column(name), specs.get(name))
            data[name] = (decoded.to_pandas() if isinstance(decoded, pa.ChunkedArray)
                          else pd.Series(decoded, dtype=object))
    finally:
        if collecting:
            gc.enable()
    frame = pd.DataFrame(data, copy=False)
    frame.index = index
    # The caller's order, not the file's: a caller that names columns gets them
    # in the order it named, as the superseded format's loc[] did.
    return frame if columns is None else frame.loc[:, list(columns)]


def load_legacy_cache(cache: str | Path) -> dict[str, pd.DataFrame]:
    """Every split of the superseded ``data.pkl.gz``, whatever Parquet is there.

    The fallback path, and what ``scripts/migrate_cache_to_parquet.py``
    converts from.  Reading it runs whatever the file says, which is most of
    why it was replaced, so it is only ever reached for a cache that was built
    before 2026-09-22.
    """
    path = legacy_path(resolve_cache(cache))
    logger.warning(
        "Reading the superseded %s. Nothing writes this format any more; "
        "convert the cache with scripts/migrate_cache_to_parquet.py.", path)
    with gzip.open(path, "rb") as handle:
        return pickle.load(handle)


def _select(frame: pd.DataFrame, columns: Optional[Sequence[str]], where: Path) -> pd.DataFrame:
    if columns is None:
        return frame
    missing = [name for name in columns if name not in frame.columns]
    if missing:
        raise KeyError(f"{where} has no column(s) {missing}; it holds {sorted(frame.columns)}")
    return frame.loc[:, list(columns)]


def build_info(cache: str | Path, split: Optional[str] = None):
    """What built *cache*, as :func:`provenance` recorded it.

    Args:
        cache: A cache directory, or the superseded ``data.pkl.gz`` inside one.
        split: One split's record; every split's, keyed by name, if omitted.

    Returns:
        The build record, or ``None`` where there is none -- a split written
        before the record existed, or converted from the superseded format,
        whose build options are genuinely not knowable.  ``None`` means "not
        recorded", never "built with the defaults".

    Raises:
        KeyError: If *split* is not in the cache.
        FileNotFoundError: If *cache* holds no records at all.
    """
    cache = resolve_cache(cache)
    if split is None:
        splits = available_splits(cache)
        if not splits:
            raise FileNotFoundError(f"No dataset cache in {cache}.")
        return {name: build_info(cache, name) for name in splits}
    path = split_path(cache, split)
    if not path.is_file():
        if split in available_splits(cache):
            return None            # the superseded format records nothing
        raise KeyError(f"{cache} has no split {split!r}; it holds "
                       f"{list(available_splits(cache))}")
    return _read_metadata(pq.ParquetFile(path).schema_arrow, path).get("build")


def load_split(
    cache: str | Path,
    split: str,
    columns: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """One split of *cache*, from Parquet if it is there and the pickle if not.

    Args:
        cache: A cache directory, or the superseded ``data.pkl.gz`` inside one.
        split: Which split to read.
        columns: Read only these.  Parquet reads only the bytes they occupy, so
            asking for what you need is worth it: four gene columns of
            LeMat-Bulk's training split take 0.1 s against 14 s for all of them.
            The superseded format has to be read whole either way.

    Raises:
        FileNotFoundError: If *cache* holds no records in either format.
        KeyError: If it holds no *split*.
    """
    cache = resolve_cache(cache)
    warn_if_obsolete_dataset(cache, "reading its cache")
    path = split_path(cache, split)
    if path.is_file():
        return _read_parquet_split(path, columns)
    if legacy_path(cache).is_file():
        frames = load_legacy_cache(cache)
        if split not in frames:
            raise KeyError(f"{legacy_path(cache)} has no split {split!r}; "
                           f"it holds {sorted(frames)}")
        return _select(frames[split], columns, legacy_path(cache))
    if any(cache.glob("*.parquet")):
        raise KeyError(f"{cache} has no split {split!r}; it holds "
                       f"{list(available_splits(cache))}")
    raise FileNotFoundError(
        f"No dataset cache in {cache}. Build it with wyformer-cache-dataset.")


def iter_splits(
    cache: str | Path,
    splits: Optional[Sequence[str]] = None,
    columns: Optional[Sequence[str]] = None,
) -> Iterator[tuple[str, pd.DataFrame]]:
    """Each split of *cache* in turn, holding as few at once as the format allows.

    What to loop over when a caller reads split after split and keeps only a
    summary -- the fingerprint set, the gene key table, the template index.
    Parquet reads one split at a time and never holds two.  The superseded
    pickle is a single stream, so it is read once and its frames handed out and
    dropped as they are consumed; calling :func:`load_split` per split would
    unpickle the whole 399 MB file once per split instead.

    Args:
        cache: A cache directory, or the superseded ``data.pkl.gz`` inside one.
        splits: Which splits to yield, in this order; every one present, by
            default.
        columns: Read only these.

    Raises:
        FileNotFoundError: If *cache* holds no records in either format.
        KeyError: If a named split or column is missing.
    """
    cache = resolve_cache(cache)
    warn_if_obsolete_dataset(cache, "reading its cache")
    if any(cache.glob("*.parquet")) or not legacy_path(cache).is_file():
        for split in splits if splits is not None else available_splits(cache):
            yield split, load_split(cache, split, columns)
        return

    frames = load_legacy_cache(cache)
    wanted = tuple(splits) if splits is not None else tuple(
        split for split in SPLITS if split in frames)
    missing = [split for split in wanted if split not in frames]
    if missing:
        raise KeyError(f"{legacy_path(cache)} has no split(s) {missing}; "
                       f"it holds {sorted(frames)}")
    for split in wanted:
        # Popped, so a consumer that drops its frame drops the last reference:
        # holding all three at once is what made this the memory ceiling.
        yield split, _select(frames.pop(split), columns, legacy_path(cache))


def load_cache(
    cache: str | Path,
    splits: Optional[Sequence[str]] = None,
    columns: Optional[Sequence[str]] = None,
) -> dict[str, pd.DataFrame]:
    """The splits of *cache*, keyed by name.

    Args:
        cache: A cache directory, or the superseded ``data.pkl.gz`` inside one.
        splits: Which splits to read; every one present, by default.  Naming
            them is how a caller stops paying for the splits it does not use --
            a named split that is missing is an error, an absent one that was
            not asked for is not.
        columns: Read only these, in every split.

    Raises:
        FileNotFoundError: If *cache* holds no records in either format.
        KeyError: If a named split or column is missing.
    """
    cache = resolve_cache(cache)
    if splits is None and not cache_exists(cache):
        raise FileNotFoundError(
            f"No dataset cache in {cache}. Build it with wyformer-cache-dataset.")
    return dict(iter_splits(cache, splits, columns))
