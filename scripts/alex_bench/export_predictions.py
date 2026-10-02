"""Turn DiffCSP++ bench-harness predictions into protocol starts.

Run with DiffCSPNew's interpreter -- it needs nothing but pickle, pymatgen and ase, so
the pymatgen objects in ``pred.pkl`` are unpickled by the venv that pickled them:

    /home/kna/DiffCSPNew/.venv/bin/python scripts/alex_bench/export_predictions.py \
        BENCHSET.pkl PRED.pkl OUT.extxyz OUT.csv

Writes one extxyz frame per structure, tagged ``gene`` (the index in the protocol's gene
file) and ``trial``, and one CSV row per (gene, trial) attempted -- including the genes
PyXtal could not initialise (absent from ``pred.pkl``) and the predictions DiffCSP++
returned as ``None`` -- so that a failure is counted, not lost.
"""
import argparse
import csv
import pickle

from ase.io import write
from pymatgen.io.ase import AseAtomsAdaptor


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("benchset")
    parser.add_argument("pred")
    parser.add_argument("out_extxyz")
    parser.add_argument("out_csv")
    parser.add_argument("--trials", type=int, default=1, help="As given to gen_init.py")
    args = parser.parse_args()

    with open(args.benchset, "rb") as handle:
        entries = pickle.load(handle)
    with open(args.pred, "rb") as handle:
        preds = pickle.load(handle)
    by_key = {(p["entry"], p["trial"]): p["pred"] for p in preds}

    frames, rows = [], []
    for entry, record in enumerate(entries):
        gene = int(record["gene_index"])
        for trial in range(args.trials):
            row = {"index": gene, "trial": trial, "status": "ok", "error": ""}
            if (entry, trial) not in by_key:
                row.update(status="failed", error="pyxtal could not initialise the gene")
            elif by_key[(entry, trial)] is None:
                row.update(status="failed", error="diffcsp++ returned no structure")
            else:
                atoms = AseAtomsAdaptor.get_atoms(by_key[(entry, trial)])
                atoms.info = {"gene": gene, "trial": trial}
                frames.append(atoms)
            rows.append(row)
    write(args.out_extxyz, frames, format="extxyz")
    with open(args.out_csv, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["index", "trial", "status", "error"])
        writer.writeheader()
        writer.writerows(rows)
    n_ok = sum(row["status"] == "ok" for row in rows)
    print(f"{n_ok}/{len(rows)} (gene, trial) pairs have a structure -> {args.out_extxyz}")


if __name__ == "__main__":
    main()
