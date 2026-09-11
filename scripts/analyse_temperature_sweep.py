#!/usr/bin/env python3
"""Tabulate a de novo protocol temperature sweep.

One arm per sampling temperature, each a directory ``T<value>/`` holding the
ordinary protocol outputs for a cohort generated at that temperature
(``wyformer-protocol-wandb --temperature``).  Every arm comes from the same
checkpoint and the same conditioning, so the only thing that differs between
them is the softmax temperature the cascade fields were drawn at.

Two things are read that the funnel does not carry.  The *shape* of the cohort
-- orbits, atoms, distinct elements per gene -- because temperature moves the
stop probability and therefore the size of what is being asked of PyXtal and
the MLIP; and a Wilson interval on every rate, because a 1000-gene cohort
resolves about 3 percentage points and a sweep read without that invites a
trend to be seen in the noise.

Usage::

    scripts/analyse_temperature_sweep.py table --sweep-dir generated/<id>/temperature
    scripts/analyse_temperature_sweep.py table --sweep-dir generated/<id>/temperature \\
        --output-dir generated/<id>/temperature/tables
"""
from __future__ import annotations

import argparse
import gzip
import json
import logging
import math
import sys
from collections import Counter
from pathlib import Path

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root / "src") not in sys.path:
    sys.path.insert(0, str(_repo_root / "src"))

import numpy as np
import pandas as pd

from wyckoff_transformer.cli.protocol import (
    FUNNEL_FILE,
    MANIFEST_FILE,
    RELAXATIONS_FILE,
    SCREEN_FILE,
    STRUCTURES_FILE,
)

logger = logging.getLogger(__name__)

GENES_FILE = "wyckoff_genes.json.gz"

#: Funnel counts worth a row, in cascade order.  Every one is reported per
#: sampled gene, which is the only denominator the protocol keeps constant.
FUNNEL_ROWS = (
    ("valid_gene", "valid gene"),
    ("unique_gene", "unique gene"),
    ("gene_novel", "novel gene"),
    ("structure", "has structure"),
    ("valid_structure", "valid structure"),
    ("unique_structure", "unique structure"),
    ("novel_structure", "novel structure"),
    ("metastable", "metastable (e_hull <= 0.1)"),
    ("stable", "stable (e_hull <= 0)"),
)

#: The two readouts the protocol ranks on.  Kept separate because they are
#: already rates per sampled gene in the funnel, not counts.
RATE_ROWS = (
    ("metasun_per_sampled_gene", "MetaSUN per sampled gene"),
    ("sun_per_sampled_gene", "SUN per sampled gene"),
)


def wilson(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial rate.

    Not the normal approximation: SUN lands at a handful of genes in a thousand,
    where the normal interval runs below zero and is worthless exactly where the
    sweep is hardest to read.
    """
    if total <= 0:
        return (float("nan"), float("nan"))
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return (max(0.0, centre - half), min(1.0, centre + half))


def gene_shape(path: Path) -> dict:
    """Per-gene size statistics of a cohort, straight off the gene file.

    Temperature acts on the STOP token as much as on the content, so an arm can
    differ from its neighbour mostly by generating longer sequences.  That
    changes the cost of every downstream stage and the difficulty of the PyXtal
    draw, so it is reported alongside the rates rather than left implicit.
    """
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        genes = json.load(handle)
    orbits, atoms, species, groups = [], [], [], Counter()
    for gene in genes:
        orbits.append(sum(len(site) for site in gene["sites"]))
        atoms.append(sum(gene["numIons"]))
        species.append(len(set(gene["species"])))
        groups[gene["group"]] += 1
    frame = pd.DataFrame({"orbits": orbits, "atoms": atoms, "species": species})
    return {
        "n_genes": len(genes),
        "mean_orbits": float(frame["orbits"].mean()),
        "median_orbits": float(frame["orbits"].median()),
        "mean_atoms": float(frame["atoms"].mean()),
        "median_atoms": float(frame["atoms"].median()),
        "mean_distinct_elements": float(frame["species"].mean()),
        "distinct_space_groups": len(groups),
        # The start token is drawn from the run's saved space-group
        # distribution, not from the model, so this is a control: it must not
        # move with the temperature.
        "top_space_group_share": groups.most_common(1)[0][1] / len(genes) if genes else None,
        "space_group_counts": dict(groups),
    }


def space_group_control(arms: list, reference_temperature: float = 1.0,
                        draws: int = 200, seed: int = 0) -> dict:
    """Check the claim that temperature leaves the space-group marginal alone.

    It should: the start token is drawn from the run's saved empirical
    distribution, not from the model.  But the cohort is truncated to the first
    ``--n-genes`` *formally valid* genes, and formal validity does move with the
    temperature, so a differential rejection could still drag the marginal
    around.  The test is a total-variation distance from the reference arm,
    read against the distance two independent draws of the *same* multinomial
    would show at this cohort size -- without that scale a TV of 0.13 looks
    like a finding when it is the noise floor.
    """
    marginals = {}
    for arm in arms:
        if "groups" not in arm:
            continue
        counts = np.zeros(231)
        for group, n in arm["groups"].items():
            counts[int(group)] += n
        marginals[arm["temperature"]] = counts / counts.sum()
    if reference_temperature not in marginals:
        return {}
    reference = marginals[reference_temperature]
    rng = np.random.default_rng(seed)
    noise = [
        0.5 * np.abs(rng.multinomial(1000, reference) / 1000
                     - rng.multinomial(1000, reference) / 1000).sum()
        for _ in range(draws)
    ]
    return {
        "reference_temperature": reference_temperature,
        "total_variation_from_reference": {
            f"{temperature:g}": round(float(0.5 * np.abs(marginal - reference).sum()), 4)
            for temperature, marginal in sorted(marginals.items())
        },
        "sampling_noise_total_variation": {
            "mean": round(float(np.mean(noise)), 4),
            "sd": round(float(np.std(noise)), 4),
            "note": "two independent 1000-gene draws of the reference marginal",
        },
    }


#: How a gene that produced no structure is classified, by what its ``error``
#: column says.  Order matters: the first pattern that matches wins.
FAILURE_KINDS = (
    ("out of memory", "cuda_oom"),
    ("timeout", "timeout"),
    ("timed out", "timeout"),
    ("pyxtal", "pyxtal"),
)


def failure_breakdown(structures: pd.DataFrame) -> dict:
    """Why the genes without a structure have none.

    Worth separating from the model's own failure rate, because one of these
    reasons is not about the model at all.  A cold sampler draws a tail of very
    large cells (see the sweep notes), and a large cell is what runs a shared
    4.6 GiB K20c out of memory -- so an arm can lose structures to the *card*
    in proportion to how cold it is, which would otherwise be read as the
    sampler being worse.
    """
    missing = structures[~structures["has_structure"].astype(bool)]
    kinds: dict[str, int] = {}
    for error in missing.get("error", pd.Series(dtype=str)).fillna(""):
        text = str(error).lower()
        for pattern, kind in FAILURE_KINDS:
            if pattern in text:
                kinds[kind] = kinds.get(kind, 0) + 1
                break
        else:
            kinds["other"] = kinds.get("other", 0) + 1
    return {"genes_without_structure": int(len(missing)), **kinds}


def oom_accounting(relaxations: pd.DataFrame, structures: pd.DataFrame | None) -> dict:
    """Exactly how much of an arm the GPU ate, and the bound that puts on MetaSUN.

    A CUDA OOM is not symmetric noise.  It can only ever *remove* a relaxation,
    never add one, so every rate it touches is biased downwards -- and it lands
    preferentially on the cold arms, whose cohorts carry the large-cell tail
    (T=0.7 has 3.0% of genes at 100+ atoms against 1.1% at T=1.0).  Left
    unmeasured it would read as the cold sampler being worse.

    Two levels, because they bound the readout differently:

    ``genes_all_trials_lost``
        The gene produced no structure at all, so it is a zero in the numerator
        of every downstream rate while still counting in the 1000.  Had those
        relaxations run, *at most* all of them would have been MetaSUN, so they
        bound the distortion from above.
    ``genes_some_trials_lost``
        The gene still has a structure, but its best energy was chosen from a
        smaller pool of trials, which can only make the kept energy higher.
        These can flip a gene out of MetaSUN but rarely do, so they enter the
        bound with the same one-sided logic and a much smaller weight.

    The bound reported is deliberately the loose, hard one: the measured
    MetaSUN cannot be too high because of OOM, and cannot be low by more than
    this.
    """
    failed = relaxations[relaxations["status"] != "ok"]
    # An arm with no failures reads its all-empty `error` column back as
    # float64, where the `.str` accessor does not exist -- so coerce rather
    # than trust the dtype.
    errors = failed["error"].astype(str) if "error" in failed else pd.Series(dtype=str)
    oom = failed[errors.str.contains("out of memory", case=False, na=False)]
    per_gene = oom.groupby("index").size() if len(oom) else pd.Series(dtype=int)

    all_lost = 0
    if structures is not None and len(per_gene):
        without = set(structures.index[~structures["has_structure"].astype(bool)])
        all_lost = sum(1 for gene in per_gene.index if gene in without)
    return {
        "trials": int(len(relaxations)),
        "trials_lost_to_oom": int(len(oom)),
        "trial_oom_rate": round(len(oom) / max(len(relaxations), 1), 5),
        "genes_touched_by_oom": int(len(per_gene)),
        "genes_all_trials_lost": int(all_lost),
        "genes_some_trials_lost": int(len(per_gene) - all_lost),
    }


def oom_metasun_bound(accounting: dict, sampled: int) -> float:
    """Upper bound on how much OOM could have suppressed MetaSUN, in rate units.

    One-sided by construction: a lost relaxation cannot have *created* a
    MetaSUN hit.  A gene that lost every trial could have been one; a gene that
    lost some could have had a better best-energy.  Both are counted at their
    maximum, so the true distortion is somewhere in ``[0, this]``.
    """
    if not accounting or sampled <= 0:
        return 0.0
    return round(accounting["genes_touched_by_oom"] / sampled, 4)


def read_arm(directory: Path) -> dict | None:
    """Everything one temperature arm can contribute, however far it got."""
    funnel_path = directory / FUNNEL_FILE
    manifest_path = directory / MANIFEST_FILE
    screen_path = directory / SCREEN_FILE
    genes_path = directory / GENES_FILE
    if not screen_path.is_file():
        logger.warning("%s: no %s, skipping", directory, SCREEN_FILE)
        return None
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    temperature = manifest.get("sampling_temperature")
    if temperature is None:
        # Fall back to the directory name (``T0.8``), so an arm produced before
        # the manifest key existed still lands in the table.
        try:
            temperature = float(directory.name.lstrip("T"))
        except ValueError:
            logger.warning("%s: no sampling_temperature and unparsable name", directory)
            return None
    arm = {
        "temperature": float(temperature),
        "dir": directory,
        "screen": json.loads(screen_path.read_text()),
        "manifest": manifest,
        "funnel": json.loads(funnel_path.read_text()) if funnel_path.is_file() else None,
    }
    if genes_path.is_file():
        arm["shape"] = gene_shape(genes_path)
        # Kept out of ``shape`` so the summary JSON stays readable; the control
        # is the only thing that wants the whole histogram.
        arm["groups"] = arm["shape"].pop("space_group_counts")
    structures_path = directory / STRUCTURES_FILE
    if structures_path.is_file():
        arm["structures"] = pd.read_csv(structures_path, index_col="index")
        arm["failures"] = failure_breakdown(arm["structures"])
    relaxations_path = directory / RELAXATIONS_FILE
    if relaxations_path.is_file():
        relaxations = pd.read_csv(relaxations_path)
        arm["oom"] = oom_accounting(relaxations, arm.get("structures"))
        arm["relax_seconds"] = float(relaxations["seconds"].sum(skipna=True))
        arm["relax_median_seconds"] = float(relaxations["seconds"].median(skipna=True))
        arm["n_relaxations"] = int(len(relaxations))
    return arm


def homogeneity_and_trend(arms: list, key: str) -> dict:
    """Is this rate temperature-dependent at all, and if so, monotonically?

    Reading a sweep by eye invites two mistakes at once: calling the largest
    gap a finding, and calling a real drift noise.  Two tests, because they
    answer different questions.

    *Homogeneity* (Pearson chi-square over the arms' 2xK table) asks whether the
    arms differ from each other by more than binomial noise.  It is sensitive to
    any difference, including a non-monotone one.

    *Trend* (Cochran-Armitage, with the temperature itself as the score) asks
    the question the sweep is actually about: does the rate move *with* the
    temperature, in one direction.  A sweep can fail homogeneity on one odd arm
    while having no trend at all, and can show a real trend too gradual for any
    single pairwise gap to reach significance.

    Both return a p-value against the null of no temperature dependence.
    """
    from scipy import stats  # noqa: PLC0415

    rows = []
    for arm in arms:
        # Same fallback as the table: the screen answers the gene-level rates
        # for every arm, so those are tested over the whole nine-point sweep
        # while the relaxed ones use only the arms that were relaxed. The arm
        # count is reported per row, since it differs.
        source = arm["funnel"] if arm["funnel"] is not None else arm["screen"]["summary"]
        if key not in source:
            continue
        total = source["sampled"]
        rate = source[key]
        successes = int(round(rate * total)) if rate <= 1 else int(rate)
        rows.append((arm["temperature"], successes, total))
    if len(rows) < 2:
        return {}

    temperatures = np.array([t for t, _, _ in rows], dtype=float)
    successes = np.array([s for _, s, _ in rows], dtype=float)
    totals = np.array([n for _, _, n in rows], dtype=float)
    failures = totals - successes

    # A rate that is 1.000 (or 0.000) in every arm -- formal gene validity is
    # both, by construction, since the cohort is truncated to valid genes --
    # has an all-zero row, which no contingency test is defined on. There is
    # nothing to test: the arms are identical.
    if successes.sum() == 0 or failures.sum() == 0:
        return {"degenerate": True, "note": "the rate is constant across every arm"}

    chi2, p_homogeneity = stats.chi2_contingency(
        np.vstack([successes, failures]))[:2]

    # Cochran-Armitage: the score is the temperature, so the statistic tests a
    # linear trend in the rate against it rather than against arm order.
    grand = successes.sum() / totals.sum()
    mean_score = (totals * temperatures).sum() / totals.sum()
    centred = temperatures - mean_score
    numerator = (centred * successes).sum()
    variance = grand * (1 - grand) * (totals * centred ** 2).sum()
    if variance <= 0:
        return {"chi2": float(chi2), "p_homogeneity": float(p_homogeneity)}
    z = numerator / math.sqrt(variance)
    p_trend = float(2 * stats.norm.sf(abs(z)))

    # The slope the trend test is about, in rate units per unit of temperature.
    weights = totals
    slope = float(
        (weights * centred * (successes / totals)).sum() / (weights * centred ** 2).sum())
    return {
        "arms": len(rows),
        "chi2": round(float(chi2), 3),
        "p_homogeneity": round(float(p_homogeneity), 4),
        "z_trend": round(float(z), 3),
        "p_trend": round(p_trend, 4),
        "slope_per_unit_temperature": round(slope, 4),
    }


def stage_table(args) -> None:
    """One column per temperature; rates, intervals, cohort shape and cost."""
    arms = []
    for directory in sorted(args.sweep_dir.glob("T*")):
        if not directory.is_dir():
            continue
        arm = read_arm(directory)
        if arm is not None:
            arms.append(arm)
    if not arms:
        raise SystemExit(f"no arms with a {SCREEN_FILE} under {args.sweep_dir}")
    arms.sort(key=lambda a: a["temperature"])
    names = [f"T={arm['temperature']:g}" for arm in arms]

    rows: list[dict] = []

    def add(label: str, values: list) -> None:
        rows.append({"metric": label, **dict(zip(names, values))})

    add("sampled genes", [arm["screen"]["n_sampled"] for arm in arms])
    # Formal validity of the raw draw, before the cohort was truncated to
    # --n-genes.  The screen only ever sees the kept genes, so without this the
    # arms look equally valid by construction.
    add("formal validity of the draw", [
        arm["manifest"].get("formal_gene_validity") for arm in arms])
    add("mean orbits per gene", [
        round(arm["shape"]["mean_orbits"], 2) if "shape" in arm else None for arm in arms])
    add("mean atoms per gene", [
        round(arm["shape"]["mean_atoms"], 2) if "shape" in arm else None for arm in arms])
    add("mean distinct elements", [
        round(arm["shape"]["mean_distinct_elements"], 2) if "shape" in arm else None
        for arm in arms])
    add("distinct space groups", [
        arm["shape"]["distinct_space_groups"] if "shape" in arm else None for arm in arms])

    for key, label in FUNNEL_ROWS:
        counts, rates = [], []
        for arm in arms:
            # The screen already answers the gene half of the funnel, so a
            # screen-only arm still belongs in those rows; only the stages that
            # need a relaxation are left blank.
            source = arm["funnel"] if arm["funnel"] is not None else arm["screen"]["summary"]
            if key not in source:
                counts.append(None)
                rates.append(None)
                continue
            sampled = source["sampled"]
            count = source[key]
            low, high = wilson(count, sampled)
            counts.append(count)
            rates.append(f"{count / sampled:.3f} [{low:.3f}, {high:.3f}]")
        add(label, counts)
        add(f"  {label} per sampled gene [95% CI]", rates)

    for key, label in RATE_ROWS:
        values, bounded = [], []
        for arm in arms:
            funnel = arm["funnel"]
            if funnel is None or key not in funnel:
                values.append(None)
                bounded.append(None)
                continue
            sampled = funnel["sampled"]
            count = int(round(funnel[key] * sampled))
            low, high = wilson(count, sampled)
            values.append(f"{funnel[key]:.3f} [{low:.3f}, {high:.3f}]")
            # OOM can only have removed structures, so it widens the interval
            # upwards alone.  Shown separately from the statistical interval
            # because it is a systematic, not noise: a bigger card shrinks it
            # to nothing without changing anything else.
            bound = oom_metasun_bound(arm.get("oom", {}), sampled)
            bounded.append(f"+{bound:.3f}" if bound else "0")
        add(f"{label} [95% CI]", values)
        add(f"  {label}: one-sided OOM bound", bounded)

    add("trials", [arm["manifest"].get("trials_total") for arm in arms])
    add("trials per gene", [arm["manifest"].get("trials_per_gene") for arm in arms])
    add("median relaxation s", [
        round(arm["relax_median_seconds"], 1) if "relax_median_seconds" in arm else None
        for arm in arms])
    add("total relaxation ks", [
        round(arm["relax_seconds"] / 1000, 1) if "relax_seconds" in arm else None
        for arm in arms])

    energy_rows = {"median e_above_hull": [], "median atoms relaxed": []}
    for arm in arms:
        frame = arm.get("structures")
        if frame is None:
            energy_rows["median e_above_hull"].append(None)
            energy_rows["median atoms relaxed"].append(None)
            continue
        energy_rows["median e_above_hull"].append(
            round(float(frame["e_above_hull"].median(skipna=True)), 4))
        energy_rows["median atoms relaxed"].append(
            round(float(frame["n_atoms"].median(skipna=True)), 1))
    for label, values in energy_rows.items():
        add(label, values)

    for key, label in (
        ("trials_lost_to_oom", "trials lost to CUDA OOM"),
        ("trial_oom_rate", "  as a fraction of trials"),
        ("genes_touched_by_oom", "genes that lost a trial to OOM"),
        ("genes_all_trials_lost", "  genes that lost every trial"),
    ):
        add(label, [arm.get("oom", {}).get(key) for arm in arms])

    for kind in ("genes_without_structure", "cuda_oom", "timeout", "pyxtal", "other"):
        values = [arm.get("failures", {}).get(kind) for arm in arms]
        if any(value for value in values):
            add(f"no structure: {kind}" if kind != "genes_without_structure"
                else "genes without a structure", values)

    table = pd.DataFrame(rows)
    out_dir = args.output_dir or (args.sweep_dir / "tables")
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / "temperature_sweep.csv", index=False)

    tests = {
        key: homogeneity_and_trend(arms, key)
        for key, _ in FUNNEL_ROWS + RATE_ROWS
    }
    tests = {key: value for key, value in tests.items() if value}

    control = space_group_control(arms)
    summary = {
        "temperature_dependence": tests,
        "space_group_control": control,
        "arms": [
            {
                "temperature": arm["temperature"],
                "dir": str(arm["dir"]),
                "screen": arm["screen"]["summary"],
                "shape": arm.get("shape"),
                "funnel": arm["funnel"],
                "n_relaxations": arm.get("n_relaxations"),
                "relax_seconds": arm.get("relax_seconds"),
                "failures": arm.get("failures"),
                "oom": arm.get("oom"),
                "oom_metasun_bound": oom_metasun_bound(
                    arm.get("oom", {}), (arm["funnel"] or {}).get("sampled", 0)),
            }
            for arm in arms
        ]
    }
    (out_dir / "temperature_sweep.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with pd.option_context("display.max_columns", None, "display.width", 250):
        print(table.to_string(index=False))
    if tests:
        print("\nIs the rate temperature-dependent? (null: it is not)")
        # The arm count differs by row on purpose: the gene-level rates come
        # from the screen, which every arm has, while anything past the
        # relaxation exists only for the arms that were relaxed.
        print(f"  {'rate':34s} {'arms':>5s} {'p homogeneity':>14s} {'p trend':>9s} {'slope/K':>9s}")
        for key, result in tests.items():
            if "p_trend" not in result:
                continue
            print(f"  {key:34s} {result['arms']:5d} {result['p_homogeneity']:14.4f} "
                  f"{result['p_trend']:9.4f} {result['slope_per_unit_temperature']:9.4f}")

    if control:
        print("\nSpace-group marginal (should not move with the temperature):")
        print("  TV from the T=%g arm: %s" % (
            control["reference_temperature"],
            ", ".join(f"T={t}: {v:.3f}"
                      for t, v in control["total_variation_from_reference"].items())))
        print("  two independent draws of the same marginal differ by "
              f"{control['sampling_noise_total_variation']['mean']:.3f} "
              f"+- {control['sampling_noise_total_variation']['sd']:.3f}")
    print(f"\n-> {out_dir}/temperature_sweep.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--debug", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)
    table = sub.add_parser("table", help="Assemble every arm into one table.")
    table.add_argument("--sweep-dir", type=Path, required=True,
                       help="Directory holding one T<value>/ arm per temperature.")
    table.add_argument("--output-dir", type=Path, default=None,
                       help="Where the table goes. Defaults to <sweep-dir>/tables.")
    table.set_defaults(func=stage_table)
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args.func(args)


if __name__ == "__main__":
    main()
