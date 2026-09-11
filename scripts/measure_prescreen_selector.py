"""Is the pre-screen's ranking any good?  Relax the candidates it rejected.

The wide-then-narrow arm spends its scoring budget on the ``k`` pre-screened
candidates NEP89 ranked best, and its +6.5 points of reconstruction are the
evidence that the ranking works.  That evidence is indirect: the arm's own
outputs only ever contain ORB energies for the candidates NEP89 *chose*, so any
correlation computed from them is conditioned on the ranker's own decision and
attenuated by the restricted range that conditioning creates.

This closes the loop by relaxing the candidates the pre-screen threw away -- the
ones marked ``rejected``, meaning distinct from every kept structure but over
budget -- for a subsample of genes.  With ORB energies for the *whole* distinct
pool, three things become measurable rather than inferred:

*Recall.*  How often NEP89's top-k contains the candidate ORB ranks first.  This
is the quantity the arm's design assumes is high, and the only honest test of it.

*Regret.*  How much energy is left on the table by taking NEP89's pick instead
of ORB's best, in ORB's own units.

*Headroom.*  What an oracle selector would have scored on the same pool, which
bounds what any better surrogate could buy and says whether the arm is limited
by the ranking or by the candidate pool.

Only the rejected candidates are relaxed: the selected ones already have ORB
energies in the arm's ``relaxations.csv``, and re-relaxing them would cost the
same again for numbers that are already on disk.
"""
from __future__ import annotations

import argparse
import logging
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger("selector")

_CALC = None


def _worker(device: str, debug: bool) -> None:
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[key] = "1"
    logging.basicConfig(level=logging.DEBUG if debug else logging.WARNING)
    global _CALC
    from wyckoff_transformer.evaluation.hull_mlips import build_hull_calculator

    _CALC = build_hull_calculator("orb_conserv_inf", device=device)


def _relax_one(payload):
    """The protocol's own four-stage schedule on one rejected candidate."""
    import io

    from ase.io import read
    from wyckoff_transformer.cryspr.generator import _trial_seed, relax_trial

    gene, trial, xyz, wdir = payload
    row = {"index": gene, "trial": trial, "status": "failed",
           "energy_per_atom_orb": None, "error": None}
    started = time.time()
    try:
        atoms = read(io.StringIO(xyz), format="extxyz")
        relaxed, energy, _prerattle = relax_trial(
            atoms_in=atoms, calculator=_CALC, trial_dir=Path(wdir),
            label=f"gene {gene} trial {trial}", seed=_trial_seed(gene, trial),
            fmax=0.05,
        )
        if relaxed is not None:
            row["status"] = "ok"
            row["energy_per_atom_orb"] = energy / len(relaxed)
    except Exception as exc:  # noqa: BLE001 - one candidate must not stop the sweep
        row["error"] = f"{type(exc).__name__}: {exc}"
    row["seconds"] = round(time.time() - started, 2)
    return row


def dof_bin(dof) -> str:
    dof = -1 if pd.isna(dof) else int(dof)
    if dof <= 0:
        return "0"
    return "1-2" if dof <= 2 else "3-5" if dof <= 5 else "6-10" if dof <= 10 else ">10"


def choose_genes(selection: pd.DataFrame, per_bin: int, min_distinct: int, seed: int):
    """Genes with a pool worth ranking, stratified by DoF.

    A gene whose distinct pool is no larger than its budget had nothing to
    select *between*, so including it would dilute the measurement with genes
    where recall is 1 by construction.
    """
    per_gene = selection.groupby("index").first()
    per_gene = per_gene[per_gene["n_distinct"] > per_gene["budget"]]
    per_gene = per_gene[per_gene["n_distinct"] >= min_distinct]
    per_gene["bin"] = per_gene["dof_positional"].map(dof_bin)
    rng = np.random.default_rng(seed)
    picked = []
    for label in ("0", "1-2", "3-5", "6-10", ">10"):
        group = per_gene[per_gene["bin"] == label]
        take = min(len(group), per_bin)
        if take:
            picked.append(group.sample(n=take, random_state=int(rng.integers(1 << 30))))
    return pd.concat(picked) if picked else per_gene.iloc[:0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--arm", type=Path,
                        default=Path("generated/nep89_protocol_variants/oracle/wide"))
    parser.add_argument("--out", type=Path,
                        default=Path("generated/nep89_protocol_variants/selector"))
    parser.add_argument("--per-bin", type=int, default=15,
                        help="Genes per DoF bin whose rejected candidates are relaxed.")
    parser.add_argument("--min-distinct", type=int, default=4)
    parser.add_argument("--devices", type=str, default="cuda:0")
    parser.add_argument("--workers-per-device", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args.out.mkdir(parents=True, exist_ok=True)

    selection = pd.read_csv(args.arm / "prescreen_selection.csv")
    genes = choose_genes(selection, args.per_bin, args.min_distinct, args.seed)
    logger.info("%d genes chosen; pools of %.1f distinct on average",
                len(genes), genes["n_distinct"].mean())

    rejected = selection[
        (selection["verdict"] == "rejected")
        & selection["index"].isin(set(genes.index))
    ]
    logger.info("relaxing %d rejected candidates", len(rejected))

    import io

    from ase.io import read as ase_read
    from ase.io import write as ase_write

    frames = {}
    for atoms in ase_read(str(args.arm / "prescreen_all.extxyz"), index=":",
                          format="extxyz"):
        key = (int(atoms.info["gene"]), int(atoms.info["trial"]))
        frames[key] = atoms

    payloads = []
    for row in rejected.itertuples():
        key = (int(row.index), int(row.trial))
        atoms = frames.get(key)
        if atoms is None:
            continue
        buf = io.StringIO()
        ase_write(buf, atoms, format="extxyz")
        payloads.append((key[0], key[1], buf.getvalue(),
                         str(args.out / "cryspr" / str(key[0]) / f"trial-{key[1]}")))

    slots = [d.strip() for d in args.devices.split(",") if d.strip()]
    slots = [d for d in slots for _ in range(args.workers_per_device)]
    ctx = multiprocessing.get_context("spawn")
    rows, started = [], time.time()
    with ProcessPoolExecutor(max_workers=len(slots), mp_context=ctx,
                             initializer=_worker,
                             initargs=(slots[0], args.debug)) as pool:
        futures = [pool.submit(_relax_one, p) for p in payloads]
        for done, future in enumerate(as_completed(futures), start=1):
            rows.append(future.result())
            if done % 50 == 0 or done == len(futures):
                rate = (time.time() - started) / done
                logger.info("relaxed %d/%d (%.1f s each, %.0f min left)",
                            done, len(futures), rate,
                            rate * (len(futures) - done) / 60)
    out = args.out / "rejected_relaxed.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    logger.info("wrote %s", out)
    genes.to_csv(args.out / "genes.csv")


if __name__ == "__main__":
    main()
