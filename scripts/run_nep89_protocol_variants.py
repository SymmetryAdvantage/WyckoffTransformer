"""Evaluate the two NEP89 variants of the de novo ranking protocol.

Three arms of the same protocol, over the same genes, differing only in how the
starting structures reach the scoring potential:

``baseline``
    The published protocol.  One to three PyXtal draws per gene by positional
    DoF, each relaxed by the four-stage ORB schedule.
``two_stage``
    Identical, except that every trial is pre-relaxed on NEP89 under the
    symmetry-constrained schedule first, and ORB starts from that geometry.
``wide``
    Ten times as many PyXtal draws, all relaxed on NEP89 under fixed symmetry,
    deduplicated with ``StructureMatcher``, and the schedule's *usual* number of
    lowest-energy survivors handed to the full ORB schedule.  The number of ORB
    relaxations is therefore the same as ``baseline``'s; only the choice of
    start differs.

Two cohorts, because they answer different questions:

**The oracle sample** (``--cohort oracle``) is 750 DoF-stratified LeMat-Bulk
structures with an ORB-relaxed reference each, prepared by
``scripts/oracle_reconstruction.py``.  Every gene has a known answer, so the
readout is *reconstruction*: does the arm's kept structure match the reference
under ``StructureMatcher``, and how far above it in energy is it.  That is the
only cohort on which "did the extra starts find a better basin" can be answered
directly rather than inferred from a rate.

**A W&B run's genes** (``--cohort wandb``) are WyFormer samples with no known
answer, so the readout is the funnel itself -- MetaSUN per sampled gene, and the
CPU-seconds it cost.  Run through ``wyformer-protocol-wandb``, which logs each
arm back to the run.

Stages::

    genes    write the oracle gene file and the index that maps it to references
    run      run one arm (or all) through the wyformer-protocol CLI
    score    match every arm's kept structures against the references
    report   aggregate into RESULTS.md

CPU by default: on this host the GPUs are contended and NEP89 is a CPU
potential, so an arm that mixes them wants ``--devices`` for the ORB half only.
"""
from __future__ import annotations

import argparse
import gzip
import json
import logging
import multiprocessing
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("nep89-variants")

#: Prepared by ``scripts/oracle_reconstruction.py --stage prepare``: 750
#: LeMat-Bulk structures, 150 per positional-DoF bin, each with its gene and an
#: ORB-relaxed reference.
ORACLE_ROOT = Path("generated/oracle_reconstruction")

DEFAULT_ROOT = Path("generated/nep89_protocol_variants")

#: DoF bins, as everywhere else in this repository's CrySPR studies.
DOF_BINS = ["0", "1-2", "3-5", "6-10", ">10"]

#: The three arms, as the flags that distinguish them.  Everything not named
#: here is the protocol's default, which is the point: an arm differs from the
#: baseline in exactly the listed arguments.
ARMS: dict[str, dict] = {
    "baseline": {
        "stages": ("screen", "generate", "relax"),
        "generate": (),
        "relax": (),
    },
    "two_stage": {
        "stages": ("screen", "generate", "relax"),
        "generate": (),
        "relax": ("--prerelax-mlip", "nep89"),
    },
    # The same arm with the expansion guard, because NEP89 grows a loose PyXtal
    # cell in 28% of draws (13% by more than 20%, 5% by more than 50%, 99th
    # percentile 2.18x; measured over 300 of this cohort's own draws).  ORB
    # cannot descend out of an inflated *symmetric* cell, so for those draws the
    # unguarded arm hands it a worse start than the raw draw was.  Whether that
    # costs reconstructions is exactly what running both arms answers.
    "two_stage_guarded": {
        "stages": ("screen", "generate", "relax"),
        "generate": (),
        "relax": ("--prerelax-mlip", "nep89", "--prerelax-max-expansion", "1.0"),
    },
    "wide": {
        "stages": ("screen", "generate", "prescreen", "relax"),
        "generate": ("--trial-multiplier", "10"),
        "relax": ("--relax-from", "prescreen"),
    },
    # NEP89 does the whole job and ORB only refines.  The cheap potential runs
    # the protocol's own four-stage schedule on every trial -- symmetric stages,
    # unconstrained stage, rattle -- and the single lowest-energy winner is the
    # one structure the scoring potential ever sees.  One ORB relaxation per
    # gene against the baseline's 2.4, so the arm is a bet that NEP89's ordering
    # of a gene's own trials is good enough to pick the winner: measured at
    # median within-gene Spearman 0.77 and 2x the chance rate, with a heavy tail.
    "nep_first": {
        "stages": ("screen", "generate", "prescreen", "relax"),
        "generate": (),
        "prescreen": (
            "--prescreen-release-symmetry", "--prescreen-rattle",
            "--prescreen-select", "1",
        ),
        "relax": ("--relax-from", "prescreen"),
    },
    # A search rather than a wider draw.  From each of the usual trials, a
    # symmetry-constrained walk between adjacent minima on NEP89 -- every step
    # projected onto the gene's own space group -- and the schedule's usual
    # number of survivors go to ORB.  The comparison against `wide` is the one
    # worth making: both spend the baseline's ORB budget on a NEP89-chosen
    # candidate set, and they differ only in how that set was built.
    "basinhop": {
        "stages": ("screen", "generate", "basinhop", "relax"),
        "generate": (),
        "relax": ("--relax-from", "basinhop"),
    },
}

#: Stages of an arm that score rather than relax.  Run separately, because the
#: oracle cohort scores against its references instead.
SCORE_STAGE = "score"


def dof_bin(dof: int) -> str:
    if dof <= 0:
        return "0"
    if dof <= 2:
        return "1-2"
    if dof <= 5:
        return "3-5"
    if dof <= 10:
        return "6-10"
    return ">10"


# --------------------------------------------------------------------------- #
# genes
# --------------------------------------------------------------------------- #
def stage_genes(root: Path, oracle_root: Path, per_bin: Optional[int], seed: int) -> None:
    """Write the oracle gene file, and the index that ties it to the references.

    The protocol takes a flat JSON list of PyXtal-notation genes and reports by
    list position, so the mapping from position back to ``immutable_id`` has to
    be written down here or the reconstruction cannot be scored at all.

    A subsample is stratified by DoF bin, not uniform: the arms are expected to
    differ most where the draw has the most room to miss, and a uniform draw of
    a stratified pool would reintroduce LeMat-Bulk's own DoF distribution.
    """
    references = pd.read_csv(oracle_root / "references.csv")
    if "dof_bin" not in references:
        references["dof_bin"] = references["dof_positional"].map(dof_bin)

    if per_bin is not None:
        rng = np.random.default_rng(seed)
        picked = []
        for label in DOF_BINS:
            group = references[references["dof_bin"] == label]
            take = min(len(group), per_bin)
            if take:
                picked.append(group.sample(n=take, random_state=int(rng.integers(1 << 30))))
        references = pd.concat(picked, ignore_index=True)
    references = references.sort_values(["dof_bin", "immutable_id"]).reset_index(drop=True)

    genes = [json.loads(text) for text in references["gene_json"]]
    root.mkdir(parents=True, exist_ok=True)
    gene_file = root / "oracle_genes.json.gz"
    with gzip.open(gene_file, "wt", encoding="utf-8") as handle:
        json.dump(genes, handle)

    index = references[[
        "immutable_id", "spacegroup", "dof_positional", "dof_bin",
        "n_conventional_atoms", "e_ref_per_atom",
    ]].copy()
    index.insert(0, "index", range(len(index)))
    index["reference_cif"] = [
        str(oracle_root / "structures" / immutable_id / "reference" / "relaxed.cif")
        for immutable_id in index["immutable_id"]
    ]
    missing = [p for p in index["reference_cif"] if not Path(p).is_file()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} of {len(index)} references have no relaxed.cif "
            f"(first: {missing[0]}). Run scripts/oracle_reconstruction.py "
            f"--stage prepare first."
        )
    index.to_csv(root / "oracle_index.csv", index=False)
    logger.info(
        "%d oracle genes -> %s; per bin %s",
        len(genes), gene_file, index["dof_bin"].value_counts().to_dict(),
    )


# --------------------------------------------------------------------------- #
# run
# --------------------------------------------------------------------------- #
def _hardware_flags(args) -> list[str]:
    if args.devices:
        return ["--devices", args.devices, "--workers-per-device", str(args.workers_per_device)]
    return ["--cores", str(args.cores)]


#: Files a downstream arm can take from an upstream one instead of recomputing.
#:
#: ``screen.json`` because it is a pure function of the gene file and the
#: LeMat-Bulk reference, identical for every arm, and costs two minutes of
#: unpickling each time.
SHARED_SCREEN = ("screen.json",)

#: The PyXtal draws, which are shared for a much stronger reason than cost.
#:
#: ``single_pyxtal`` takes no seed -- PyXtal draws from the global RNG -- so two
#: arms that each run ``generate`` relax *different* random structures, and the
#: comparison between them then mixes the effect being measured with draw noise.
#: The trial-spread study is precisely the measurement of how large that noise
#: is: p(e_hull <= 0.1) moves by 6 to 13 points between the first and third trial
#: of the same gene.  Copying the draws makes baseline vs two-stage a paired
#: comparison at the level of the individual structure, where the only
#: difference is the pre-relaxation.
#:
#: Only between arms that draw the same number of them: the wide arm's
#: ``--trial-multiplier 10`` is a different draw budget, so it necessarily makes
#: its own and is comparable only per gene.
SHARED_DRAWS = ("pyxtal.extxyz", "pyxtal.csv")


def _multiplier_of(spec: dict) -> str:
    flags = list(spec["generate"])
    if "--trial-multiplier" in flags:
        return flags[flags.index("--trial-multiplier") + 1]
    return "1"


def _adopt_shared(
    args, arm: str, spec: dict, out: Path, cohort: str = "oracle"
) -> list[str]:
    """Copy what another arm already computed, and say which stages to skip.

    Returns:
        The stage names *not* to run, because their output was copied in.
    """
    shared_root = args.root / cohort / "_shared"
    groups = [
        (shared_root / "screen", SHARED_SCREEN, "screen"),
        (shared_root / f"draws_x{_multiplier_of(spec)}", SHARED_DRAWS, "generate"),
    ]
    skip = []
    for directory, names, stage in groups:
        if stage not in spec["stages"]:
            continue
        if all((directory / name).is_file() for name in names):
            for name in names:
                target = out / name
                source = directory / name
                if not target.is_file() or target.stat().st_size != source.stat().st_size:
                    target.write_bytes(source.read_bytes())
            logger.info("[%s] adopted %s from %s", arm, ", ".join(names), directory)
            skip.append(stage)
    return skip


def _publish_shared(
    args, arm: str, spec: dict, out: Path, cohort: str = "oracle"
) -> None:
    """Make this arm's screen and draws available to the arms after it."""
    shared_root = args.root / cohort / "_shared"
    for directory, names in (
        (shared_root / "screen", SHARED_SCREEN),
        (shared_root / f"draws_x{_multiplier_of(spec)}", SHARED_DRAWS),
    ):
        if not all((out / name).is_file() for name in names):
            continue
        directory.mkdir(parents=True, exist_ok=True)
        for name in names:
            if not (directory / name).is_file():
                (directory / name).write_bytes((out / name).read_bytes())
                logger.info("[%s] published %s -> %s", arm, name, directory)


def stage_run(args, arm: str) -> None:
    """Run one arm of the oracle cohort through the ``wyformer-protocol`` CLI.

    A subprocess per stage rather than an in-process call: the stages are what
    ships, each one's argument handling is part of what is being evaluated, and
    a crash in one arm then cannot take the study with it.  ``--resume`` is on
    inside the protocol, so re-running this is cheap and picks up where it
    stopped.

    The screen and the PyXtal draws are taken from whichever arm computed them
    first rather than recomputed -- see :data:`SHARED_DRAWS` for why that is a
    correctness requirement and not an economy.
    """
    spec = ARMS[arm]
    out = args.root / "oracle" / arm
    out.mkdir(parents=True, exist_ok=True)
    gene_file = args.root / "oracle_genes.json.gz"
    if not gene_file.is_file():
        raise FileNotFoundError(f"No gene file at {gene_file}; run --stage genes first.")

    skip = [] if args.no_share else _adopt_shared(args, arm, spec, out)

    common = [
        sys.executable, "-m", "wyckoff_transformer.cli.protocol",
        str(gene_file), "--output-dir", str(out),
    ]
    if args.limit is not None:
        common += ["--limit", str(args.limit)]

    for stage in spec["stages"]:
        if stage in skip:
            logger.info("[%s] stage %s: adopted, not run", arm, stage)
            continue
        command = list(common) + ["--stage", stage]
        if stage == "generate":
            command += ["--pyxtal-cores", str(args.pyxtal_cores)] + list(spec["generate"])
        elif stage in ("prescreen", "basinhop"):
            # No multiplier here on purpose: the pre-screen selects the
            # *unmultiplied* schedule back down, which is what makes the arm's
            # ORB cost equal to the baseline's.  Always CPU, whatever --devices
            # says: NEP89 runs on the CPU, so a GPU slot here would just be a
            # worker with an idle card attached.
            command += ["--cores", str(args.prescreen_cores or args.cores)]
            command += list(spec.get(stage, ()))
        elif stage == "relax":
            command += _hardware_flags(args) + list(spec["relax"])
        logger.info("[%s] %s", arm, " ".join(command[3:]))
        started = time.time()
        result = subprocess.run(command, check=False)
        logger.info("[%s] stage %s exited %d after %.0f s",
                    arm, stage, result.returncode, time.time() - started)
        if result.returncode != 0:
            raise SystemExit(f"arm {arm} stage {stage} failed with {result.returncode}")

    if not args.no_share:
        _publish_shared(args, arm, spec, out)


# --------------------------------------------------------------------------- #
# wandb
# --------------------------------------------------------------------------- #
def stage_wandb(args) -> None:
    """Run every arm over one W&B run's genes, and score the funnel.

    The cohort is generated **once**, by the first arm, and the rest run with
    ``--skip-generate`` on a copy of that gene file.  Generating per arm would
    confound the comparison with sampling noise: the arms differ by a few points
    of MetaSUN at most, and two independent 1000-gene draws from the same model
    differ by about as much on their own.

    An arm with a pre-screen is run in two invocations rather than one, because
    ``wyformer-protocol-wandb`` gives every stage the same hardware and the two
    halves want different hardware: the pre-screen is NEP89 on CPU cores, the
    relaxation is the scoring potential on whatever ``--devices`` says.

    Uploading is off by default.  Every arm would write its funnel into the same
    ``protocol/`` keys of the same run summary, so the last one to finish would
    be the only one recorded -- which is worse than not recording at all.  Pass
    ``--upload-arm <name>`` to log exactly one.
    """
    from wyckoff_transformer.cli.protocol_wandb import GENES_FILE

    root = args.root / "wandb"
    root.mkdir(parents=True, exist_ok=True)
    shared_genes = root / GENES_FILE

    for position, arm in enumerate(args.arms):
        spec = ARMS[arm]
        out = root / arm
        out.mkdir(parents=True, exist_ok=True)
        target = out / GENES_FILE
        have_cohort = shared_genes.is_file()
        if have_cohort:
            if not target.is_file() or target.read_bytes() != shared_genes.read_bytes():
                target.write_bytes(shared_genes.read_bytes())
        elif position:
            raise SystemExit(
                f"no shared cohort at {shared_genes}; run the first arm first"
            )

        skip = [] if args.no_share else _adopt_shared(
            args, arm, spec, out, cohort="wandb"
        )
        stages = [s for s in list(spec["stages"]) + [SCORE_STAGE] if s not in skip]
        for stage in skip:
            logger.info("[wandb/%s] stage %s: adopted, not run", arm, stage)
        cheap = [s for s in stages if s in ("screen", "generate", "prescreen")]
        expensive = [s for s in stages if s not in cheap]
        # One invocation unless the arm has a pre-screen, whose hardware differs.
        groups = (
            [(cheap, ["--cores", str(args.prescreen_cores or args.cores)]),
             (expensive, _hardware_flags(args))]
            if "prescreen" in cheap
            else [(stages, _hardware_flags(args))]
        )

        groups = [(group, hardware) for group, hardware in groups if group]
        for offset, (group, hardware) in enumerate(groups):
            command = [
                sys.executable, "-m", "wyckoff_transformer.cli.protocol_wandb",
                args.wandb_run, "--output-dir", str(out),
                "--n-genes", str(args.n_genes),
                "--stages", ",".join(group),
                "--pyxtal-cores", str(args.pyxtal_cores),
            ] + hardware
            command += list(spec["generate"]) + list(spec["relax"])
            command += list(spec.get("prescreen", ())) + list(spec.get("basinhop", ()))
            for condition in args.condition or ():
                command += ["--condition", condition]
            # Only the very last group of the uploading arm writes back: an
            # earlier group has no funnel.json to upload yet.
            if arm != args.upload_arm or offset != len(groups) - 1:
                command += ["--no-upload"]
            if args.limit is not None:
                command += ["--limit", str(args.limit)]
            if have_cohort or offset:
                command += ["--skip-generate"]

            logger.info("[wandb/%s] %s", arm, " ".join(command[3:]))
            started = time.time()
            result = subprocess.run(command, check=False)
            logger.info("[wandb/%s] stages %s exited %d after %.0f s",
                        arm, ",".join(group), result.returncode, time.time() - started)
            if result.returncode != 0:
                raise SystemExit(f"wandb arm {arm} failed with {result.returncode}")
            have_cohort = True

        if not shared_genes.is_file() and target.is_file():
            shared_genes.write_bytes(target.read_bytes())
            logger.info("cohort of %d genes shared from arm %s", args.n_genes, arm)
        if not args.no_share:
            _publish_shared(args, arm, spec, out, cohort="wandb")


# --------------------------------------------------------------------------- #
# score
# --------------------------------------------------------------------------- #
def _match_one(payload) -> dict:
    """Does this arm's kept structure match the reference?  One gene."""
    for key, value in {
        "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
    }.items():
        os.environ[key] = value
    index, kept_cif, reference_cif = payload
    from pymatgen.analysis.structure_matcher import StructureMatcher
    from pymatgen.core import Structure

    row = {"index": index, "matched": False, "match_error": None, "rms": None}
    try:
        kept = Structure.from_file(kept_cif)
        reference = Structure.from_file(reference_cif)
        matcher = StructureMatcher()
        row["matched"] = bool(matcher.fit(reference, kept))
        distance = matcher.get_rms_dist(reference, kept)
        row["rms"] = float(distance[0]) if distance is not None else None
    except Exception as exc:  # noqa: BLE001 - one bad CIF must not stop the sweep
        row["match_error"] = f"{type(exc).__name__}: {exc}"
    return row


def stage_score(args) -> None:
    """Match every arm's kept structures against the ORB-relaxed references.

    The oracle readout, and the reason this cohort exists: ``matched`` is a
    verdict per gene rather than a rate over an unknown population, so two arms
    can be compared *per gene* -- which is a paired comparison and therefore far
    more powerful than the funnel's unpaired rates at this cohort size.
    """
    index = pd.read_csv(args.root / "oracle_index.csv")
    if args.limit is not None:
        # Otherwise a smoke test's un-run genes count as failures and every rate
        # is a fraction of a cohort that was never attempted.
        index = index[index["index"] < args.limit]
    ctx = multiprocessing.get_context("spawn")
    frames = []
    for arm in args.arms:
        out = args.root / "oracle" / arm
        structures_path = out / "structures.csv"
        if not structures_path.is_file():
            logger.warning("[%s] no structures.csv; skipping", arm)
            continue
        structures = pd.read_csv(structures_path, index_col="index")
        payloads = []
        for row in index.itertuples():
            kept = out / "cifs" / f"{row.index}.cif"
            if kept.is_file():
                payloads.append((int(row.index), str(kept), row.reference_cif))
        logger.info("[%s] matching %d kept structures", arm, len(payloads))

        matches = []
        with ProcessPoolExecutor(max_workers=args.cores, mp_context=ctx) as pool:
            futures = [pool.submit(_match_one, p) for p in payloads]
            for done, future in enumerate(as_completed(futures), start=1):
                matches.append(future.result())
                if done % 100 == 0 or done == len(futures):
                    logger.info("[%s] matched %d/%d", arm, done, len(futures))

        frame = index.merge(pd.DataFrame(matches), on="index", how="left")
        # eq(True) rather than fillna(False).astype(bool): the merge leaves an
        # object column, and fillna on one is deprecated-and-downcasting.
        frame["matched"] = frame["matched"].eq(True)
        for column in (
            "has_structure", "energy_per_atom", "n_relaxed", "n_drawn",
            "relax_seconds", "pyxtal_seconds", "prescreen_seconds", "n_prescreened",
            "n_trials",
        ):
            frame[column] = (
                structures[column].reindex(frame["index"]).to_numpy()
                if column in structures.columns else np.nan
            )
        frame["arm"] = arm
        frame["delta_e_per_atom"] = frame["energy_per_atom"] - frame["e_ref_per_atom"]
        frames.append(frame)

    if not frames:
        raise SystemExit("no arm has any structures to score")
    scored = pd.concat(frames, ignore_index=True)
    scored.to_csv(args.root / "oracle_scored.csv", index=False)
    logger.info("scored %d arm-genes -> %s", len(scored), args.root / "oracle_scored.csv")
    print(summarise(scored).to_string())


def prerelax_volume_report(root: Path, arms) -> pd.DataFrame:
    """What the cheap potential did to each cell, from every ``prerelax.json``.

    The failure mode the two-stage arm has to be checked for: a pre-relaxation
    that *expands* a loose PyXtal draw rather than contracting it leaves the
    scoring potential at a symmetric stationary point in an inflated cell, and
    gradient descent under a symmetry constraint cannot leave one -- so the
    inflation survives every stage but the rattle.  A rate is the only honest way
    to report it, since one bad trial proves nothing and a systematic bias
    invalidates the arm.
    """
    rows = []
    for arm in arms:
        for path in sorted((root / "oracle" / arm).glob("cryspr/*/trial-*/**/prerelax.json")):
            try:
                verdict = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            verdict["arm"] = arm
            rows.append(verdict)
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    summary = []
    for arm, group in frame.groupby("arm", sort=False):
        ratio = group["volume_ratio"].dropna()
        summary.append({
            "arm": arm,
            "trials": len(group),
            "median_volume_ratio": float(ratio.median()) if len(ratio) else np.nan,
            "expanded": float((ratio > 1.0).mean()) if len(ratio) else np.nan,
            "expanded_over_20pct": float((ratio > 1.2).mean()) if len(ratio) else np.nan,
            "contracted_under_80pct": float((ratio < 0.8).mean()) if len(ratio) else np.nan,
            "spacegroup_kept": float(
                (group["spacegroup_before"] == group["spacegroup_after"]).mean()
            ),
        })
    return pd.DataFrame(summary).set_index("arm")


def summarise(scored: pd.DataFrame) -> pd.DataFrame:
    """Per-arm reconstruction rate, energy gap and cost, overall and per DoF bin."""
    rows = []
    for arm, group in scored.groupby("arm", sort=False):
        row = {
            "arm": arm,
            "genes": len(group),
            "has_structure": float(group["has_structure"].eq(True).mean()),
            "matched": float(group["matched"].mean()),
            "median_dE_meV": 1000 * float(group["delta_e_per_atom"].median(skipna=True)),
            # A comparison against NaN is already False, which is the answer we
            # want for a gene with no structure.
            "at_or_below_ref": float((group["delta_e_per_atom"] <= 1e-3).mean()),
            "orb_seconds_per_gene": float(group["relax_seconds"].mean(skipna=True)),
            "prescreen_seconds_per_gene": float(
                group["prescreen_seconds"].mean(skipna=True)
            ) if "prescreen_seconds" in group else np.nan,
            "orb_trials_per_gene": float(group["n_relaxed"].mean(skipna=True)),
            "draws_per_gene": float(group["n_drawn"].mean(skipna=True)),
        }
        for label in DOF_BINS:
            sub = group[group["dof_bin"] == label]
            row[f"matched_{label}"] = float(sub["matched"].mean()) if len(sub) else np.nan
        rows.append(row)
    return pd.DataFrame(rows).set_index("arm")


def paired_table(scored: pd.DataFrame, baseline: str = "baseline") -> pd.DataFrame:
    """Per-gene wins and losses against *baseline*, which is the powerful test.

    Both arms see the same gene, so the comparison is paired and the discordant
    pairs are the whole signal: an arm that recovers 40 genes the baseline
    missed and loses 10 it found is a real improvement even when the two rates
    differ by a couple of points on 750 genes.  McNemar's exact statistic is the
    binomial tail on those discordant pairs.
    """
    from scipy.stats import binomtest

    wide = scored.pivot_table(index="index", columns="arm", values="matched")
    rows = []
    for arm in wide.columns:
        if arm == baseline or baseline not in wide.columns:
            continue
        pair = wide[[baseline, arm]].dropna()
        won = int(((~pair[baseline].astype(bool)) & pair[arm].astype(bool)).sum())
        lost = int((pair[baseline].astype(bool) & (~pair[arm].astype(bool))).sum())
        test = binomtest(won, won + lost, 0.5) if won + lost else None
        rows.append({
            "arm": arm,
            "n_paired": len(pair),
            "baseline_matched": int(pair[baseline].astype(bool).sum()),
            "arm_matched": int(pair[arm].astype(bool).sum()),
            "won": won,
            "lost": lost,
            "mcnemar_p": float(test.pvalue) if test is not None else np.nan,
        })
    return pd.DataFrame(rows).set_index("arm") if rows else pd.DataFrame()


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #
def stage_report(args) -> None:
    scored = pd.read_csv(args.root / "oracle_scored.csv")
    summary = summarise(scored)
    paired = paired_table(scored)

    lines = [
        "# NEP89 protocol variants: results",
        "",
        "Generated by `scripts/run_nep89_protocol_variants.py --stage report`.",
        "",
        "## Oracle cohort",
        "",
        (
            f"{scored['index'].nunique()} DoF-stratified LeMat-Bulk genes with "
            "an ORB-relaxed reference each. `matched` is `StructureMatcher` at "
            "pymatgen's defaults against that reference."
        ),
        "",
        summary.to_markdown(floatfmt=".4f"),
        "",
    ]
    if len(paired):
        lines += [
            "### Paired against the baseline",
            "",
            (
                "Both arms see the same gene, so only the discordant pairs "
                "carry information; `mcnemar_p` is the exact binomial tail "
                "on them."
            ),
            "",
            paired.to_markdown(floatfmt=".4f"),
            "",
        ]

    volumes = prerelax_volume_report(args.root, args.arms)
    if len(volumes):
        lines += [
            "### What the pre-relaxation did to the cell",
            "",
            "A pre-relaxation is meant to contract a loose PyXtal draw towards "
            "contact. One that expands it leaves ORB at a symmetric stationary "
            "point in an inflated cell, which its own symmetric stages cannot "
            "descend out of -- only the rattle can.",
            "",
            volumes.to_markdown(floatfmt=".4f"),
            "",
        ]

    prescreens = {}
    for arm in args.arms:
        path = args.root / "oracle" / arm / "prescreen.csv"
        if not path.is_file():
            continue
        frame = pd.read_csv(path)
        ok = frame[frame["status"] == "ok"]
        selection = args.root / "oracle" / arm / "prescreen_selection.csv"
        verdicts = (
            pd.read_csv(selection)["verdict"].value_counts().to_dict()
            if selection.is_file() else {}
        )
        prescreens[arm] = {
            "draws": len(frame),
            "ok": len(ok),
            "median_volume_ratio": float(ok["volume_ratio"].median(skipna=True)),
            "expanded": float((ok["volume_ratio"] > 1.0).mean()),
            "backend_lj": int((ok["backend"] == "lj").sum()),
            "seconds_per_draw": float(ok["seconds"].mean(skipna=True)),
            **{f"verdict_{k}": v for k, v in verdicts.items()},
        }
    if prescreens:
        lines += [
            "### The pre-screen",
            "",
            "`verdict_duplicate` is the share of the widened draw that landed on "
            "a structure another draw of the same gene had already found -- what "
            "the extra starts actually bought.",
            "",
            pd.DataFrame(prescreens).T.to_markdown(floatfmt=".4f"),
            "",
        ]

    funnels = {}
    for cohort in ("oracle", "wandb"):
        for arm in args.arms:
            path = args.root / cohort / arm / "funnel.json"
            if path.is_file():
                funnels[f"{cohort}/{arm}"] = json.loads(path.read_text(encoding="utf-8"))
    if funnels:
        frame = pd.DataFrame(funnels).T
        keep = [
            c for c in (
                "sampled", "structure_per_sampled_gene", "valid_structure_per_sampled_gene",
                "unique_structure_per_sampled_gene", "novel_structure_per_sampled_gene",
                "metastable_per_sampled_gene", "metasun_per_sampled_gene",
                "sun_per_sampled_gene",
            ) if c in frame.columns
        ]
        lines += [
            "## Funnels",
            "",
            frame[keep].to_markdown(floatfmt=".4f"),
            "",
        ]

    rattle = {}
    for cohort in ("oracle", "wandb"):
        for arm in args.arms:
            path = args.root / cohort / arm / "funnel.json"
            if not path.is_file():
                continue
            f = json.loads(path.read_text(encoding="utf-8"))
            if f.get("prerattle_metasun_per_sampled_gene") is None:
                continue
            rattle[f"{cohort}/{arm}"] = {
                "metasun": f.get("metasun_per_sampled_gene"),
                "metasun_prerattle": f.get("prerattle_metasun_per_sampled_gene"),
                "novel": f.get("novel_structure_per_sampled_gene"),
                "novel_prerattle": f.get("prerattle_novel_structure_per_sampled_gene"),
                "moved_off_gene": f.get("rattle_moved_off_gene"),
                "novel_became_known": f.get("rattle_novel_became_known"),
                "known_became_novel": f.get("rattle_known_became_novel"),
                "metasun_lost": f.get("rattle_metasun_lost"),
                "metasun_gained": f.get("rattle_metasun_gained"),
                "lowered_energy": f.get("rattle_lowered_energy"),
            }
    if rattle:
        lines += [
            "## What the rattle stage costs and buys",
            "",
            "The rattle lowers ORB's energy, which is why it is on by default. "
            "It also discards the Wyckoff orbits WyFormer predicted -- and for a "
            "Wyckoff generative model those orbits *are* the prediction -- and it "
            "can relax a novel structure onto a known one. `_prerattle` columns "
            "are the same metric on the structure the rattle stage was handed; "
            "the counts are genes that crossed, in both directions.",
            "",
            pd.DataFrame(rattle).T.to_markdown(floatfmt=".4f"),
            "",
        ]

    manifests = {}
    for cohort in ("oracle", "wandb"):
        for arm in args.arms:
            path = args.root / cohort / arm / "manifest.json"
            if path.is_file():
                manifests[f"{cohort}/{arm}"] = json.loads(path.read_text(encoding="utf-8"))
    if manifests:
        lines += ["## Manifests", "", "```json",
                  json.dumps(manifests, indent=2, default=str), "```", ""]

    (args.root / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    logger.info("wrote %s", args.root / "RESULTS.md")
    print("\n".join(lines[:40]))


# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--stage", choices=("genes", "run", "wandb", "score", "report", "all"),
        default="all",
        help="'all' is the oracle cohort end to end; 'wandb' is the other cohort "
             "and is deliberately not in it, since it needs a run id.",
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--oracle-root", type=Path, default=ORACLE_ROOT)
    parser.add_argument(
        "--arms", type=str, default=",".join(ARMS),
        help="Comma-separated subset of the arms to run and report.",
    )
    parser.add_argument(
        "--per-bin", type=int, default=None,
        help="Subsample the oracle references to this many genes per DoF bin. "
             "Omit for all 150.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Subsample seed.")
    parser.add_argument(
        "--cores", type=int, default=16,
        help="CPU workers. 16 is the established figure for MLIP work on zeus: "
             "24 physical cores, shared machine.",
    )
    parser.add_argument("--pyxtal-cores", type=int, default=16)
    parser.add_argument(
        "--prescreen-cores", type=int, default=None,
        help="CPU workers for the pre-screen. Defaults to --cores. Always CPU: "
             "NEP89 is a CPU potential, so --devices does not apply to it.",
    )
    parser.add_argument(
        "--devices", type=str, default=None,
        help="Run the relaxations on GPU instead, e.g. 'cuda:0'.",
    )
    parser.add_argument("--workers-per-device", type=int, default=2)
    wandb_group = parser.add_argument_group("W&B cohort (stage: wandb)")
    wandb_group.add_argument(
        "--wandb-run", type=str, default="upi73i4k",
        help="Run whose genes the arms are scored on. The protocol docs' "
             "reference run, whose single-arm funnel is already published.",
    )
    wandb_group.add_argument("--n-genes", type=int, default=1000,
                             help="Cohort size. 1000 is the MetaSUN power figure.")
    wandb_group.add_argument(
        "--condition", action="append", metavar="NAME=VALUE",
        default=["energy_above_hull=0"],
        help="Conditioning target, once per feature. upi73i4k conditions on "
             "energy_above_hull, so it needs one.",
    )
    wandb_group.add_argument(
        "--upload-arm", type=str, default=None,
        help="Log exactly this arm back to the run. Every arm writes the same "
             "protocol/ summary keys, so uploading more than one keeps only the "
             "last to finish.",
    )

    parser.add_argument(
        "--no-share", action="store_true",
        help="Give every arm its own screen and its own PyXtal draws. Off by "
             "default, and turning it on weakens the comparison: arms would "
             "then differ by the random draw as well as by the thing under "
             "test. See SHARED_DRAWS.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Smoke test.")
    parser.add_argument("--debug", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args.arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    unknown = [a for a in args.arms if a not in ARMS]
    if unknown:
        raise SystemExit(f"--arms: {unknown} not in {list(ARMS)}")
    args.root.mkdir(parents=True, exist_ok=True)

    stages = ("genes", "run", "score", "report") if args.stage == "all" else (args.stage,)
    for stage in stages:
        logger.info("=== %s ===", stage)
        if stage == "genes":
            stage_genes(args.root, args.oracle_root, args.per_bin, args.seed)
        elif stage == "run":
            for arm in args.arms:
                stage_run(args, arm)
        elif stage == "wandb":
            stage_wandb(args)
        elif stage == "score":
            stage_score(args)
        elif stage == "report":
            stage_report(args)


if __name__ == "__main__":
    main()
