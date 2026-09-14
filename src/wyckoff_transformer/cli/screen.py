"""Rank chemical formulas by how likely they are to lower the convex hull.

Takes a trained ensemble and a list of formulas, and answers, for each: where is
the floor under this composition, how sure are we, what would it have to beat,
and is that low enough to contradict something somebody has already made.

The formulas need not be in any database -- that is the point. A formula nobody
has computed has no provenance to supply, and the model's location head never
reads provenance, so it answers the same way it answers for a well-studied one.

Ranking is on Wren's uncertainty-adjusted criterion by default: a candidate is
scored on ``location + sigma - hull`` rather than ``location - hull``. With
millions of candidates a point-estimate ranking selects for the largest positive
errors, and the adjustment was worth fifteen points of precision in the one
prospective campaign anyone has published.

Example::

    wyformer-screen --ensemble runs/formula_energy/ensemble.pt \\
        --formulas candidates.txt --reference data/lemat-bulk/lemat_pbe_ehull.csv.gz \\
        --top 500 --out shortlist.csv
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List, Optional

import pandas as pd
import torch

from wyckoff_transformer.formula_energy import screen as screening
from wyckoff_transformer.formula_energy import train as T

logger = logging.getLogger(__name__)

#: Columns the displacement bound and the hull lookup need from the reference.
REFERENCE_COLUMNS = ("immutable_id", "full_formula", "chemsys", "energy_corrected")


def read_formulas(path: Optional[Path], inline: Optional[List[str]]) -> List[str]:
    """One formula per line from a file, or whatever was passed on the command line."""
    if inline:
        return list(inline)
    if path is None:
        raise SystemExit("Give --formulas or --formula")
    lines = [line.strip() for line in path.read_text().splitlines()]
    formulas = [line for line in lines if line and not line.startswith("#")]
    if not formulas:
        raise SystemExit(f"{path} holds no formulas")
    return formulas


def load_reference(path: Path, protected_only: bool = False) -> pd.DataFrame:
    """The hull entries, as :mod:`~wyckoff_transformer.formula_energy.screen` wants them."""
    frame = pd.read_csv(path, usecols=list(REFERENCE_COLUMNS), low_memory=False)
    frame["energy_corrected"] = pd.to_numeric(frame["energy_corrected"], errors="coerce")
    frame = frame.dropna(subset=["full_formula", "chemsys", "energy_corrected"])
    return frame.set_index("immutable_id")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ensemble", type=Path, required=True,
                        help="saved by formula_energy.train.save_ensemble")
    parser.add_argument("--formulas", type=Path, help="file with one formula per line")
    parser.add_argument("--formula", nargs="+", help="formulas given inline")
    parser.add_argument("--reference", type=Path, required=True,
                        help="LeMat-Bulk energy CSV defining the hull to beat")
    parser.add_argument("--out", type=Path, default=Path("shortlist.csv"))
    parser.add_argument("--top", type=int, default=None, help="keep only this many")
    parser.add_argument("--no-margin", action="store_true",
                        help="rank on the point estimate, without Wren's uncertainty adjustment")
    parser.add_argument("--displacement-filter", action="store_true",
                        help="also compute L(X); slow, so run it on a shortlist")
    parser.add_argument("--device", type=torch.device, default=None)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO,
                        format="%(asctime)s %(message)s")

    device = args.device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    formulas = read_formulas(args.formulas, args.formula)
    logger.info("%d formulas, device %s", len(formulas), device)

    reference = load_reference(args.reference)
    hull, kept = [], []
    for formula in formulas:
        try:
            hull.append(screening.hull_energy_per_atom(reference, formula))
            kept.append(formula)
        except (ValueError, KeyError) as problem:
            logger.warning("skipping %s: %s", formula, problem)
    if not kept:
        raise SystemExit("No formula had a reference hull to be measured against")

    models, _, _feature_names = T.load_ensemble(args.ensemble, device)
    data = T.prepare_formulas(kept, hull=hull).to(device)
    prediction = T.predict(models, data).drop(columns=["target"])
    frame = screening.shortlist(prediction, margin=not args.no_margin, top=args.top)

    if args.displacement_filter:
        logger.info("computing L(X) for %d candidates", len(frame))
        frame = screening.apply_displacement_filter(frame, reference)

    frame.to_csv(args.out)
    print(frame.head(20).to_string())
    print(f"\nwrote {len(frame)} rows to {args.out}")


if __name__ == "__main__":
    main()
