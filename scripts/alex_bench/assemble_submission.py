"""Assemble the benchmark submission from an evaluated arm's DiffCSP++ structures.

The arm was produced by ``scripts/platforms/zeus/run_alex_bench_setting.sh`` with a budget
a little above the submission size. Genes are taken in the arm's own order -- lowest
predicted e_hull first for fire-control, pool order otherwise -- and each is submitted as
its DiffCSP++ structure with the same seeded rattle the evaluation applied
(:func:`wyckoff_transformer.diffcsp_bridge.rattle_start`), so the structures submitted are
the structures that were evaluated. A gene whose start failed, or failed
:func:`~wyckoff_transformer.diffcsp_bridge.check_start`, is skipped; no MLIP output is
consulted for the selection. The ORB evaluation of the chosen subset is reported, as the
expected score, in ``manifest.json``.

    .venv/bin/python scripts/alex_bench/assemble_submission.py ARM_DIR OUT_DIR [--n 10000]
"""
import argparse
import json
import os
import subprocess
from pathlib import Path

import pandas as pd
from ase.io import read, write
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.cif import CifWriter

from wyckoff_transformer.diffcsp_bridge import check_start, rattle_start
from wyckoff_transformer.evaluation.protocol import (
    METASTABLE_THRESHOLD,
    STABLE_THRESHOLD,
    load_genes,
    read_screen,
)
from wyckoff_transformer.formula_energy.prefilter import wilson_interval


def _order(arm_dir: Path, n_genes: int) -> list[int]:
    """Gene indices in the order the arm ranks them."""
    arm = json.loads((arm_dir / "arm.json").read_text())
    if arm.get("regressor"):
        predicted = pd.read_csv(arm_dir / "predicted_e_hull.csv").set_index("index")
        return list(predicted["predicted_e_hull"].sort_values(kind="stable").index)
    return list(range(n_genes))


def _expected(structures: pd.DataFrame, chosen: list[int], n: int) -> dict:
    """The ORB evaluation of the chosen genes: what the submission should score."""
    rows = structures.loc[structures.index.intersection(chosen)]
    base = (rows["valid_structure"].eq(True) & rows["unique_structure"].eq(True)
            & rows["novel_structure"].eq(True))
    out = {}
    for name, threshold in (("msun", METASTABLE_THRESHOLD), ("sun", STABLE_THRESHOLD)):
        hits = int((base & (rows["e_above_hull"] <= threshold)).sum())
        low, high = wilson_interval(hits, n)
        out[name] = {"hits": hits, "rate": hits / n, "low": low, "high": high}
    out["valid"] = int(rows["valid_structure"].eq(True).sum())
    out["novel"] = int((rows["valid_structure"].eq(True) & rows["novel_structure"].eq(True)).sum())
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("arm_dir", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--n", type=int, default=10_000)
    args = parser.parse_args()

    genes = load_genes(args.arm_dir / "engaged_genes.json.gz")
    protocol = args.arm_dir / "protocol"
    # Without a protocol run (sampling only, as in the container) every engaged gene is
    # its own representative: fire-control and fire-discipline screen for uniqueness.
    screen_path = protocol / "screen.json"
    representatives = (set(read_screen(screen_path).counts) if screen_path.is_file()
                       else set(range(len(genes))))
    frames = {int(a.info["gene"]): a
              for a in read(str(args.arm_dir / "diffcsp" / "starts.extxyz"), index=":")}

    chosen, skipped, rattled = [], {}, {}
    for index in _order(args.arm_dir, len(genes)):
        if len(chosen) == args.n:
            break
        if index not in representatives:
            skipped[index] = "duplicate gene"
            continue
        if index not in frames:
            skipped[index] = "no DiffCSP++ structure"
            continue
        check = check_start(frames[index], genes[index])
        if check.error:
            skipped[index] = check.error
            continue
        # Judged again after the rattle: a start that just clears the distance floor
        # can be pushed below it, and what is checked must be what is submitted.
        rattled[index] = rattle_start(frames[index], index, 0)
        check = check_start(rattled[index], genes[index])
        if check.error:
            skipped[index] = f"after the rattle: {check.error}"
            continue
        chosen.append(index)
    if len(chosen) < args.n:
        raise SystemExit(f"Only {len(chosen)} usable genes for a submission of {args.n}; "
                         "re-run the arm with a larger budget")

    cif_dir = args.out_dir / "cifs"
    cif_dir.mkdir(parents=True, exist_ok=True)
    submitted, records = [], []
    for rank, index in enumerate(chosen):
        atoms = rattled[index]
        atoms.info = {"submission_id": rank, "gene": index}
        name = f"{rank:05d}"
        CifWriter(AseAtomsAdaptor.get_structure(atoms)).write_file(cif_dir / f"{name}.cif")
        submitted.append(atoms)
        records.append({"submission_id": name, "gene_index": index,
                        "formula": atoms.get_chemical_formula(mode="metal"),
                        "n_atoms": len(atoms), "spacegroup_gene": genes[index]["group"],
                        "gene": json.dumps(genes[index])})
    write(str(args.out_dir / "structures.extxyz"), submitted, format="extxyz")
    manifest = pd.DataFrame(records)
    predicted = args.arm_dir / "predicted_e_hull.csv"
    if predicted.is_file():
        manifest = manifest.merge(
            pd.read_csv(predicted)[["index", "predicted_e_hull"]],
            left_on="gene_index", right_on="index", how="left").drop(columns="index")
    manifest.to_csv(args.out_dir / "manifest.csv", index=False)

    structures_path = protocol / "structures.csv"
    pool_path = args.arm_dir.parent / "pool" / "pool.json"
    record = {
        "arm_dir": str(args.arm_dir),
        "arm": json.loads((args.arm_dir / "arm.json").read_text()),
        "pool": json.loads(pool_path.read_text()) if pool_path.is_file() else None,
        "n_submitted": len(chosen),
        "skipped": pd.Series(skipped).value_counts().to_dict() if skipped else {},
        "rattle": "cryspr.relaxer.perturb, seed _trial_seed(gene_index, 0)",
        # Only when the arm was evaluated: sampling alone has no ORB numbers to report.
        "expected_orb_score": (
            _expected(pd.read_csv(structures_path).set_index("index"), chosen, len(chosen))
            if structures_path.is_file() else None),
        "wyformer_commit": os.environ.get("WYFORMER_REVISION") or subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
            cwd=Path(__file__).resolve().parent).stdout.strip() or None,
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps({k: record[k] for k in ("n_submitted", "skipped", "expected_orb_score")},
                     indent=1))


if __name__ == "__main__":
    main()
