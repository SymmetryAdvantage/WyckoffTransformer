#!/usr/bin/env python3
"""Pull the experimental-provenance flags that LeMat-Bulk does not carry.

LeMat-Bulk's ``compatible_pbe`` schema has 24 fields and none of them says
whether an entry descends from an experimentally observed structure: the only
provenance is the ``immutable_id`` prefix, which names the source database
(``mp``, ``agm``, ``oqmd``) and not the evidence. Materials Project does carry
it, as ``SummaryDoc.theoretical`` -- false exactly when the material has an
ICSD entry -- and ``database_IDs``, which lists those ICSD ids.

This script takes the ``mp-`` ids out of a LeMat-Bulk table, asks MP for those
two fields, and writes them keyed on ``material_id``. It does not touch the
structures, so nothing is re-tokenised; the result joins onto any cache or CSV by
``immutable_id``.

Default source is the energy CSV rather than the tokenised cache, because the
cache is filtered (converged forces, tokenisable Wyckoff sets) in a way that
costs most of the MP rows: 138,931 in ``lemat_pbe_ehull.csv.gz`` against 31,794
in ``cache/lemat_bulk_ehull``. The experimentally-backed formulas are the scarce
side of this problem, so the wider source is worth the extra API calls.

Needs ``MP_API_KEY`` in the environment or in ``.env``, and the ``mp`` extra::

    uv sync --extra mp
    uv run python scripts/pull_mp_provenance.py
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

DEFAULT_SOURCE = Path("data/lemat-bulk/lemat_pbe_ehull.csv.gz")
DEFAULT_OUT = Path("data/mp_provenance.csv.gz")
CHUNK = 1000


def mp_ids(source: Path) -> list[str]:
    """Every ``mp-`` immutable id in a LeMat-Bulk table.

    Accepts either the energy CSV, where the ids are the ``immutable_id``
    column, or a tokenised cache pickle, where they are the index of each
    split's frame.
    """
    if source.name.endswith(".pkl.gz"):
        data = pd.read_pickle(source)
        frames = data.values() if isinstance(data, dict) else [data]
        ids = pd.Index([])
        for frame in frames:
            ids = ids.append(frame.index)
    else:
        ids = pd.Index(pd.read_csv(source, usecols=["immutable_id"])["immutable_id"])
    mp = [i for i in ids.unique() if str(i).startswith("mp-")]
    return sorted(mp)


def fetch(ids: list[str], api_key: str) -> pd.DataFrame:
    """``material_id``, ``theoretical`` and the ICSD ids, for ids MP still has."""
    from mp_api.client import MPRester

    rows = []
    with MPRester(api_key) as mpr:
        for start in range(0, len(ids), CHUNK):
            chunk = ids[start:start + CHUNK]
            docs = mpr.materials.summary.search(
                material_ids=chunk,
                fields=["material_id", "theoretical", "database_IDs"],
            )
            for doc in docs:
                icsd = (doc.database_IDs or {}).get("icsd", []) or []
                rows.append({
                    "material_id": str(doc.material_id),
                    "theoretical": bool(doc.theoretical),
                    "n_icsd": len(icsd),
                })
            print(f"  {min(start + CHUNK, len(ids))}/{len(ids)}", flush=True)
    return pd.DataFrame(rows).set_index("material_id")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE,
                        help="LeMat-Bulk energy CSV or tokenised cache to take the mp- ids from")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    try:
        import dotenv
        dotenv.load_dotenv(".env")
    except ImportError:
        pass
    api_key = os.environ.get("MP_API_KEY")
    if not api_key:
        raise SystemExit("MP_API_KEY is not set; put it in .env or the environment")

    ids = mp_ids(args.source)
    print(f"{len(ids)} mp- ids in {args.source}")
    table = fetch(ids, api_key)
    missing = len(ids) - len(table)
    print(f"MP returned {len(table)}; {missing} ids no longer resolve")
    print(f"experimental (theoretical=False): {(~table['theoretical']).sum()}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
