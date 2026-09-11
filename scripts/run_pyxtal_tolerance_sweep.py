"""Does a more permissive PyXtal distance floor reconstruct more structures?

`single_pyxtal` draws under `Tol_matrix(prototype="atomic", factor=1.3)`, a
floor at `0.65 * (r_a + r_b)`.  PyXtal enforces it by placing Wyckoff orbits one
at a time and rejecting a placement against the orbits already down, so the
factor does not bound the finished structure -- 45% of returned draws are below
it -- but it does bias the *joint* draw away from dense configurations, which is
what real crystals are.  The floor existed because relaxers collapsed on
overlapped input; NEP89's ZBL core does not (0 failures in 18,000 draws of the
`nep89_protocol_variants` cohort, 227 of them under 0.8 tolerance units), so the
factor is now testable.

One arm per factor, over the 750-gene DoF-stratified oracle cohort with an
ORB-relaxed reference each, and `--n-trials 3` so the trial schedule cannot
confound the comparison.  Every arm runs the same three stages:

    generate --pyxtal-tol-factor f  ->  prescreen (NEP89)  ->  relax (ORB)

Stages:

    run       the three protocol stages, per arm, strictly serially
    contacts  closest contact and cell volume of every draw, per arm
    score     StructureMatcher against each gene's reference, per arm
    report    the tables and docs/pyxtal_tolerance_sweep.md

The confound this has to answer as well as the headline: PyXtal draws the cell
too, and a permissive floor may simply let it accept *smaller* cells.  If it
does, the arms differ in cell volume as well as in coordinate sampling and the
comparison is not a clean test of the sampling-bias hypothesis -- so `contacts`
measures the drawn volume against the reference's, and `report` prints it next
to the reconstruction rate whatever it says.

Usage:

    uv run python scripts/run_pyxtal_tolerance_sweep.py run --pyxtal-cores 16 \
        --prescreen-cores 16 --devices cuda:0 --workers-per-device 2
    uv run python scripts/run_pyxtal_tolerance_sweep.py contacts
    uv run python scripts/run_pyxtal_tolerance_sweep.py score --cores 16
    uv run python scripts/run_pyxtal_tolerance_sweep.py report
"""
from __future__ import annotations

import argparse
import logging
import multiprocessing
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

# The scoring half of this study is the same measurement the NEP89 variant sweep
# makes, against the same references, so it is imported rather than restated:
# two implementations of "did StructureMatcher fit" would be two definitions of
# reconstruction, and the numbers here are meant to sit beside those.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_nep89_protocol_variants import (
    DOF_BINS,
    _match_one,
    paired_table,
)

logger = logging.getLogger("pyxtal-tol-sweep")

#: Where the cohort lives.  Prepared by ``run_nep89_protocol_variants --stage
#: genes``; not regenerated here, because a different draw of 750 genes would
#: make these numbers incomparable with that study's.
COHORT_ROOT = Path("generated/nep89_protocol_variants")

DEFAULT_ROOT = Path("generated/pyxtal_tolerance_sweep")

#: The tolerance factors, control first.
#:
#: 1.3 is the shipped default and the control.  1.0 is PyXtal's own default.
#: 0.7, 0.4 and 0.1 are past anything defensible as a physical floor -- 0.1
#: permits a 0.05*(r_a+r_b) contact, i.e. essentially none -- and that is the
#: point: if rejection-sampling bias is what caps reconstruction, removing the
#: rejection almost entirely has to show it, and a geometric ladder localises
#: where the effect starts rather than only whether it exists.
FACTORS = (1.3, 1.0, 0.7, 0.4, 0.1)

#: Trials per gene, flat.
#:
#: The protocol's own schedule spends 1 trial on a zero-DoF gene and 2 on the
#: rest, which would give the factors different budgets in different DoF bins --
#: and the factor can only matter where there are free coordinates to draw.  A
#: flat 3 makes the per-bin rates comparable across arms.
N_TRIALS = "3"


def arm_name(factor: float) -> str:
    """The arm's directory name, e.g. ``f1.3``.

    The factor is in the path because two runs of this study are otherwise
    indistinguishable on disk: nothing in a CIF or a trial log says which floor
    it was drawn under.
    """
    return f"f{factor:g}"


#: The tolerance matrix contacts are *measured* in, independent of the arm.
#:
#: factor 1.0, i.e. ``0.5 * (r_a + r_b)``, so a reported number is a physical
#: ratio and not a ratio to whatever floor that arm happened to draw under.
#: Measuring each arm in its own units would make every arm's floor 1.0 by
#: construction and hide the whole effect.
MEASURE_FACTOR = 1.0


# --------------------------------------------------------------------------- #
# run
# --------------------------------------------------------------------------- #
def stage_run(args) -> None:
    """Run every arm's three protocol stages, one subprocess at a time.

    Strictly serial, including across arms: each stage takes either 16 CPU
    workers or a GPU, and two pools at once on a shared machine would make the
    per-draw and per-relaxation seconds -- which are part of the result --
    meaningless.
    """
    gene_file = COHORT_ROOT / "oracle_genes.json.gz"
    screen = COHORT_ROOT / "oracle" / "_shared" / "screen" / "screen.json"
    if not gene_file.is_file():
        raise FileNotFoundError(f"No gene file at {gene_file}")

    for factor in args.factors:
        out = args.root / arm_name(factor)
        out.mkdir(parents=True, exist_ok=True)
        # The screen is a pure function of the gene file and the LeMat-Bulk
        # reference, so it is identical for every arm and copied rather than
        # recomputed; nothing in this study depends on it beyond the gene list
        # it hands `generate`.
        target = out / "screen.json"
        if not target.is_file() and screen.is_file():
            target.write_bytes(screen.read_bytes())
            logger.info("[%s] adopted %s", arm_name(factor), screen)

        common = [
            sys.executable, "-m", "wyckoff_transformer.cli.protocol",
            str(gene_file), "--output-dir", str(out), "--n-trials", N_TRIALS,
        ]
        if args.limit is not None:
            common += ["--limit", str(args.limit)]
        stages = [
            ["--stage", "generate",
             "--pyxtal-tol-factor", f"{factor:g}",
             "--pyxtal-cores", str(args.pyxtal_cores)],
            ["--stage", "prescreen", "--cores", str(args.prescreen_cores)],
            ["--stage", "relax", "--relax-from", "prescreen",
             "--devices", args.devices,
             "--workers-per-device", str(args.workers_per_device)],
        ]
        for stage in stages:
            command = common + stage
            logger.info("[%s] %s", arm_name(factor), " ".join(command[3:]))
            started = time.time()
            result = subprocess.run(command, check=False)
            logger.info("[%s] %s exited %d after %.0f s", arm_name(factor),
                        stage[1], result.returncode, time.time() - started)
            if result.returncode != 0:
                raise SystemExit(f"arm {arm_name(factor)} stage {stage[1]} failed")


# --------------------------------------------------------------------------- #
# contacts
# --------------------------------------------------------------------------- #
def _contact_rows(payload) -> list[dict]:
    """Closest contact and cell volume of every draw in one arm's extxyz.

    A worker per arm rather than per structure: the frames are read
    sequentially out of one file, so splitting finer would mean re-reading it.
    """
    import os

    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[key] = "1"
    arm, path = payload
    from ase.io import iread
    from ase.neighborlist import neighbor_list
    from pyxtal.tolerance import Tol_matrix

    tm = Tol_matrix(prototype="atomic", factor=MEASURE_FACTOR)
    rows = []
    for atoms in iread(path, format="extxyz", index=":"):
        numbers = atoms.numbers
        unique = sorted({int(n) for n in numbers})
        tol_max = max(tm.get_tol(a, b) for a in unique for b in unique)
        # A cutoff at the tolerance itself only answers "is anything below the
        # floor", and a loose draw has nothing inside it at all -- which came
        # back as an infinite ratio for every zero-DoF gene on the first
        # attempt.  The distribution is the point here, so the cutoff is grown
        # until the nearest distinct pair is actually inside it.
        cutoff = 2.5 * tol_max
        while cutoff <= 40.0:
            first, second, dist = neighbor_list("ijd", atoms, cutoff)
            # Self-image pairs are governed by the cell rather than by
            # placement, and PyXtal does not check them for a multiplicity-1
            # orbit at all, so the headline number excludes them -- as
            # `pyxtal_generation_audit` does -- and they are counted separately
            # instead of being silently folded in.
            distinct = first != second
            if distinct.any():
                break
            cutoff *= 2
        tols = np.array([tm.get_tol(int(numbers[a]), int(numbers[b]))
                         for a, b in zip(first, second)]) if len(dist) else np.zeros(0)
        ratio = dist / tols if len(dist) else np.zeros(0)
        rows.append({
            "arm": arm,
            "index": int(atoms.info.get("gene", -1)),
            "trial": int(atoms.info.get("trial", -1)),
            "n_atoms": len(atoms),
            "volume": float(atoms.get_volume()),
            "min_ratio": float(np.min(ratio[distinct])) if distinct.any() else np.inf,
            "min_ratio_with_images": float(np.min(ratio)) if len(ratio) else np.inf,
        })
    return rows


def _reference_volumes(index: pd.DataFrame) -> pd.DataFrame:
    """Volume per atom of each gene's ORB-relaxed reference.

    Per atom rather than per cell: PyXtal draws the conventional cell of the
    gene's space group, which need not have the reference's atom count if the
    reference's symmetry was detected at a different setting, and a per-cell
    ratio would then compare different numbers of atoms.
    """
    from ase.io import read as ase_read

    rows = []
    for row in index.itertuples():
        try:
            reference = ase_read(row.reference_cif)
            rows.append({
                "index": int(row.index),
                "ref_volume_per_atom": float(reference.get_volume()) / len(reference),
            })
        except Exception as exc:  # noqa: BLE001 - a missing reference is a NaN
            logger.warning("gene %s: unreadable reference (%s)", row.index, exc)
    return pd.DataFrame(rows)


def stage_contacts(args) -> None:
    """Per-draw contact ratio and cell volume for every arm -> ``contacts.csv``."""
    index = pd.read_csv(COHORT_ROOT / "oracle_index.csv")
    payloads = []
    for factor in args.factors:
        path = args.root / arm_name(factor) / "pyxtal.extxyz"
        if not path.is_file():
            logger.warning("[%s] no pyxtal.extxyz; skipping", arm_name(factor))
            continue
        payloads.append((arm_name(factor), str(path)))

    ctx = multiprocessing.get_context("spawn")
    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=min(args.cores, len(payloads)),
                             mp_context=ctx) as pool:
        futures = {pool.submit(_contact_rows, p): p[0] for p in payloads}
        for future in as_completed(futures):
            arm_rows = future.result()
            logger.info("[%s] measured %d draws", futures[future], len(arm_rows))
            rows += arm_rows

    frame = pd.DataFrame(rows)
    frame = frame.merge(_reference_volumes(index), on="index", how="left")
    frame = frame.merge(index[["index", "dof_positional", "dof_bin"]],
                        on="index", how="left")
    frame["volume_per_atom"] = frame["volume"] / frame["n_atoms"]
    frame["volume_ratio"] = frame["volume_per_atom"] / frame["ref_volume_per_atom"]
    frame.to_csv(args.root / "contacts.csv", index=False)
    logger.info("%d draws -> %s", len(frame), args.root / "contacts.csv")
    print(contact_summary(frame).to_string())


def contact_summary(contacts: pd.DataFrame) -> pd.DataFrame:
    """What each factor actually drew: contacts, and the cell it drew them in."""
    rows = []
    for arm, group in contacts.groupby("arm", sort=False):
        finite = group["min_ratio"].replace(np.inf, np.nan)
        rows.append({
            "arm": arm,
            "draws": len(group),
            "min_contact_p1": float(finite.quantile(0.01)),
            "min_contact_median": float(finite.median()),
            "min_contact_min": float(finite.min()),
            "below_1.0": float((finite < 1.0).mean()),
            "below_0.8": float((finite < 0.8).mean()),
            "below_0.5": float((finite < 0.5).mean()),
            "vol_ratio_median": float(group["volume_ratio"].median(skipna=True)),
            "vol_ratio_p10": float(group["volume_ratio"].quantile(0.10)),
            "vol_ratio_p90": float(group["volume_ratio"].quantile(0.90)),
        })
    return pd.DataFrame(rows).set_index("arm")


# --------------------------------------------------------------------------- #
# score
# --------------------------------------------------------------------------- #
def stage_score(args) -> None:
    """Match every arm's kept structure against its gene's reference."""
    index = pd.read_csv(COHORT_ROOT / "oracle_index.csv")
    if args.limit is not None:
        index = index[index["index"] < args.limit]
    ctx = multiprocessing.get_context("spawn")
    frames = []
    for factor in args.factors:
        arm = arm_name(factor)
        out = args.root / arm
        structures_path = out / "structures.csv"
        if not structures_path.is_file():
            logger.warning("[%s] no structures.csv; skipping", arm)
            continue
        structures = pd.read_csv(structures_path, index_col="index")
        payloads = [
            (int(row.index), str(out / "cifs" / f"{row.index}.cif"), row.reference_cif)
            for row in index.itertuples()
            if (out / "cifs" / f"{row.index}.cif").is_file()
        ]
        logger.info("[%s] matching %d kept structures", arm, len(payloads))
        matches = []
        with ProcessPoolExecutor(max_workers=args.cores, mp_context=ctx) as pool:
            futures = [pool.submit(_match_one, p) for p in payloads]
            for done, future in enumerate(as_completed(futures), start=1):
                matches.append(future.result())
                if done % 200 == 0 or done == len(futures):
                    logger.info("[%s] matched %d/%d", arm, done, len(futures))

        frame = index.merge(pd.DataFrame(matches), on="index", how="left")
        frame["matched"] = frame["matched"].eq(True)
        for column in ("has_structure", "energy_per_atom", "n_relaxed", "n_drawn",
                       "relax_seconds", "pyxtal_seconds", "prescreen_seconds",
                       "n_prescreened"):
            frame[column] = (
                structures[column].reindex(frame["index"]).to_numpy()
                if column in structures.columns else np.nan
            )
        frame["arm"] = arm
        frame["tol_factor"] = factor
        frame["delta_e_per_atom"] = frame["energy_per_atom"] - frame["e_ref_per_atom"]
        frames.append(frame)

    if not frames:
        raise SystemExit("no arm has any structures to score")
    scored = pd.concat(frames, ignore_index=True)
    scored.to_csv(args.root / "scored.csv", index=False)
    logger.info("%d arm-genes -> %s", len(scored), args.root / "scored.csv")
    print(summarise(scored).to_string())


def summarise(scored: pd.DataFrame) -> pd.DataFrame:
    """Reconstruction rate and cost per arm, overall and per DoF bin."""
    rows = []
    for arm, group in scored.groupby("arm", sort=False):
        row = {
            "arm": arm,
            "genes": len(group),
            "has_structure": float(group["has_structure"].eq(True).mean()),
            "matched": float(group["matched"].mean()),
            "median_dE_meV": 1000 * float(group["delta_e_per_atom"].median(skipna=True)),
            "at_or_below_ref": float((group["delta_e_per_atom"] <= 1e-3).mean()),
            "draws_per_gene": float(group["n_drawn"].mean(skipna=True)),
            "orb_trials_per_gene": float(group["n_relaxed"].mean(skipna=True)),
            "pyxtal_seconds_per_gene": float(group["pyxtal_seconds"].mean(skipna=True)),
            "prescreen_seconds_per_gene": float(group["prescreen_seconds"].mean(skipna=True)),
            "orb_seconds_per_gene": float(group["relax_seconds"].mean(skipna=True)),
        }
        for label in DOF_BINS:
            sub = group[group["dof_bin"].astype(str) == label]
            row[f"matched_{label}"] = float(sub["matched"].mean()) if len(sub) else np.nan
        rows.append(row)
    return pd.DataFrame(rows).set_index("arm")


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #
def cost_table(root: Path, factors) -> pd.DataFrame:
    """PyXtal and NEP89 cost per arm, straight from the per-trial logs.

    Read from the logs rather than from ``structures.csv``: a draw that failed
    has no gene-level row to average into, and the failure rate is the number
    the permissive arms are supposed to improve.
    """
    rows = []
    for factor in factors:
        arm = arm_name(factor)
        out = root / arm
        row = {"arm": arm, "tol_factor": factor}
        draws = out / "pyxtal.csv"
        if draws.is_file():
            frame = pd.read_csv(draws)
            counts = frame["status"].value_counts()
            row.update({
                "draws": len(frame),
                "draw_failed": int(counts.get("failed", 0)),
                "draw_timeout": int(counts.get("timeout", 0)),
                "pyxtal_cpu_hours": float(frame["seconds"].sum(skipna=True)) / 3600,
                "pyxtal_s_median": float(frame["seconds"].median(skipna=True)),
                "pyxtal_s_p99": float(frame["seconds"].quantile(0.99)),
            })
        pre = out / "prescreen.csv"
        if pre.is_file():
            frame = pd.read_csv(pre)
            row.update({
                "prescreen_n": len(frame),
                "prescreen_failed": int((frame["status"] != "ok").sum()),
                "prescreen_s_median": float(frame["seconds"].median(skipna=True)),
                "prescreen_cpu_hours": float(frame["seconds"].sum(skipna=True)) / 3600,
            })
            if "volume_ratio" in frame:
                row["prescreen_vol_ratio_median"] = float(
                    frame["volume_ratio"].median(skipna=True)
                )
        relax = out / "relaxations.csv"
        if relax.is_file():
            frame = pd.read_csv(relax)
            row.update({
                "orb_n": len(frame),
                "orb_failed": int((frame["status"] != "ok").sum()),
                "orb_gpu_hours": float(frame["seconds"].sum(skipna=True)) / 3600,
            })
        rows.append(row)
    return pd.DataFrame(rows).set_index("arm")


def stage_report(args) -> None:
    scored = pd.read_csv(args.root / "scored.csv")
    contacts = pd.read_csv(args.root / "contacts.csv")
    summary = summarise(scored)
    contact = contact_summary(contacts)
    cost = cost_table(args.root, args.factors)
    baseline = arm_name(args.factors[0])
    paired = paired_table(scored, baseline=baseline)

    print(summary.to_string())
    print()
    print(contact.to_string())
    print()
    print(cost.to_string())
    print()
    print(paired.to_string() if len(paired) else "no paired comparison")

    out = Path(args.out)
    out.write_text("\n".join([
        "<!-- tables regenerated by scripts/run_pyxtal_tolerance_sweep.py --stage report -->",
        "",
        "## Reconstruction",
        "",
        summary.to_markdown(floatfmt=".4f"),
        "",
        "## Paired against the control",
        "",
        paired.to_markdown(floatfmt=".4g") if len(paired) else "_none_",
        "",
        "## What was drawn",
        "",
        contact.to_markdown(floatfmt=".4f"),
        "",
        "## Cost",
        "",
        cost.to_markdown(floatfmt=".4f"),
        "",
    ]) + "\n", encoding="utf-8")
    logger.info("tables -> %s", out)


# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=("run", "contacts", "score", "report"))
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--factors", type=float, nargs="+", default=list(FACTORS),
                        help="Tolerance factors; the first is the control.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only the first N genes, for a smoke test.")
    parser.add_argument("--pyxtal-cores", type=int, default=16)
    parser.add_argument("--prescreen-cores", type=int, default=16)
    parser.add_argument("--cores", type=int, default=16,
                        help="Workers for the contacts and score stages.")
    parser.add_argument("--devices", type=str, default="cuda:0")
    parser.add_argument("--workers-per-device", type=int, default=2)
    parser.add_argument("--out", type=Path,
                        default=Path("docs/pyxtal_tolerance_sweep_tables.md"),
                        help="Where --stage report writes its tables.")
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    args = build_parser().parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    {"run": stage_run, "contacts": stage_contacts,
     "score": stage_score, "report": stage_report}[args.stage](args)


if __name__ == "__main__":
    main()
