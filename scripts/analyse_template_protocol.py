#!/usr/bin/env python3
"""Compare template-matched starts against a finished de novo protocol run.

The protocol's ``template`` stage appends one template-matched start per gene
to a run's existing PyXtal draws, so a run that has already been relaxed and
scored can be extended without repeating any of it.  This script turns the
extended run into the two arms worth reading, scores each with the ordinary
``score`` stage, and tabulates them against the original.

Arms, all over the same genes and the same relaxation settings:

``random1``
    One random start per gene -- the original run's trial 0.  The comparison
    that matches ``template_first`` on budget, since that also spends about one
    trial per gene.
``random``
    The original run's trials, unchanged: PyXtal draws under the trial schedule.
``template_first``
    The proposal.  The template start where the gene has one, and the original
    run's random trials where it does not -- so the fallback is the ordinary
    protocol, not a missing structure.
``union``
    Every trial of both, best energy kept.  Not a proposal, an upper bound: it
    says how much the template start adds when nothing is taken away.

All three are scored here rather than read from the original run's
``funnel.json``.  That is not redundancy: the funnel's definitions change, and a
comparison against a funnel written by a different revision of the score stage
would attribute the difference to the arm.  ``random`` reuses the original
run's *relaxations*, which is what must not be recomputed; its scoring is cheap
and has to match its neighbours.

Usage::

    scripts/analyse_template_protocol.py arms  --run-dir generated/<id>/protocol_template
    scripts/analyse_template_protocol.py score --run-dir generated/<id>/protocol_template
    scripts/analyse_template_protocol.py table --run-dir generated/<id>/protocol_template \\
        --baseline generated/<id>/protocol
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root / "src") not in sys.path:
    sys.path.insert(0, str(_repo_root / "src"))

import pandas as pd

from wyckoff_transformer.cli.protocol import (
    FUNNEL_FILE,
    PYXTAL_COLUMNS,
    PYXTAL_TRIALS_FILE,
    RELAXATIONS_FILE,
    RELAXATION_COLUMNS,
    SCREEN_FILE,
    STRUCTURES_FILE,
    TEMPLATE_TRIAL,
    aggregate_structures,
    read_rows,
    stage_score,
)
from wyckoff_transformer.evaluation.protocol import (
    DEFAULT_REFERENCE_CACHE,
    DEFAULT_REFERENCE_SPLITS,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("template_protocol")

#: The arms built beside the extended run, as ``suffix -> description``.  The
#: extended run itself is the ``union`` arm and needs no directory.
ARM_SUFFIXES = {
    "_random1": "random1",
    "_random": "random",
    "_first": "template_first",
}


def _arm_dir(run_dir: Path, suffix: str) -> Path:
    return run_dir.parent / (run_dir.name + suffix)


def stage_arms(args) -> None:
    """Split the extended run into the ``random`` and ``template_first`` arms.

    ``union`` needs no directory of its own: it *is* the extended run, whose
    logs hold every trial.  The other two get a sibling directory with a
    filtered copy of the two per-trial logs; the CIF paths in them still point
    into the trial directories the relaxations were written to, so nothing is
    copied but a few CSVs and nothing is relaxed again.
    """
    run_dir = args.run_dir
    relaxations = read_rows(run_dir / RELAXATIONS_FILE, RELAXATION_COLUMNS)
    draws = read_rows(run_dir / PYXTAL_TRIALS_FILE, PYXTAL_COLUMNS)
    if not len(relaxations):
        raise SystemExit(f"No relaxations at {run_dir / RELAXATIONS_FILE}")

    is_template = relaxations["trial"] == TEMPLATE_TRIAL
    template = relaxations[is_template]
    has_template = set(template.loc[template["status"] == "ok", "index"])
    logger.info(
        "%d genes have a relaxed template start, %d have a template row at all, "
        "%d trials in the extended run",
        len(has_template), len(template), len(relaxations),
    )

    def masks(frame: pd.DataFrame) -> dict:
        """Which of *frame*'s rows each arm keeps, by ``(index, trial)`` alone.

        Applied to the relaxation log and the draw log alike, so a gene whose
        draws all failed keeps its row in the draw log and is still reported as
        a gene without a structure rather than vanishing from the frame.
        """
        template_row = frame["trial"] == TEMPLATE_TRIAL
        gene_has = frame["index"].isin(has_template)
        return {
            # One random start per gene: the budget-matched comparison for
            # template_first, which also spends about one trial per gene.
            "_random1": frame["trial"] == 0,
            # The original run's trials, and only those.
            "_random": ~template_row,
            # The template trial for the genes that have one, every other trial
            # for the genes that do not -- the proposal's own fallback rule.
            "_first": (gene_has & template_row) | (~gene_has & ~template_row),
        }

    relaxation_masks, draw_masks = masks(relaxations), masks(draws)
    for suffix in ARM_SUFFIXES:
        out = _arm_dir(run_dir, suffix)
        out.mkdir(parents=True, exist_ok=True)
        for name in (SCREEN_FILE, "wyckoff_genes.json.gz"):
            source = run_dir / name
            if source.is_file():
                shutil.copy2(source, out / name)
        arm = relaxations[relaxation_masks[suffix]]
        arm.to_csv(out / RELAXATIONS_FILE, index=False)
        draws[draw_masks[suffix]].to_csv(out / PYXTAL_TRIALS_FILE, index=False)
        logger.info("%s: %d trials over %d genes -> %s",
                    ARM_SUFFIXES[suffix], len(arm), arm["index"].nunique(), out)

    for directory in (run_dir, *(_arm_dir(run_dir, s) for s in ARM_SUFFIXES)):
        frame = aggregate_structures(directory)
        logger.info("%s: %d/%d genes with a structure",
                    directory.name, int(frame["has_structure"].sum()), len(frame))


def stage_score_arms(args) -> None:
    """Run the protocol's own score stage on each arm directory."""
    for directory in (args.run_dir,
                      *(_arm_dir(args.run_dir, s) for s in ARM_SUFFIXES)):
        if not (directory / STRUCTURES_FILE).is_file():
            raise SystemExit(f"{directory} has no {STRUCTURES_FILE}; run 'arms' first")
        logger.info("=== scoring %s ===", directory)
        stage_score(SimpleNamespace(
            input=args.genes or (directory / "wyckoff_genes.json.gz"),
            output_dir=directory,
            mlip=args.mlip,
            reference_cache=args.reference_cache,
            reference_splits=",".join(DEFAULT_REFERENCE_SPLITS),
            lemat_cif_csv=args.lemat_cif_csv,
        ))


#: What the comparison table reports, as ``column -> label``.  Every rate in the
#: funnel is per *sampled* gene, so the arms are directly comparable even though
#: they spend different numbers of trials.
_FUNNEL_ROWS = (
    ("structure", "genes with a structure"),
    ("valid_structure", "valid structure"),
    ("unique_structure", "unique structure"),
    ("novel_structure", "novel structure"),
    ("metastable", "unique and e_hull <= 0.1"),
    ("stable", "unique and e_hull <= 0"),
    ("metastable_among_novel", "novel and e_hull <= 0.1 (MetaSUN numerator)"),
    ("stable_among_novel", "novel and e_hull <= 0 (SUN numerator)"),
)


def stage_table(args) -> None:
    """Assemble the arms into one table, plus a per-gene join for the details."""
    arms = {
        "random1": _arm_dir(args.run_dir, "_random1"),
        "template_first": _arm_dir(args.run_dir, "_first"),
        "random": _arm_dir(args.run_dir, "_random"),
        "union": args.run_dir,
    }
    if args.baseline is not None:
        # The original run's own funnel, for the record.  It is *not* one of the
        # arms: it was written by whatever revision of the score stage was
        # current when the run finished, and the funnel's definitions change.
        published = json.loads((args.baseline / FUNNEL_FILE).read_text())
        logger.info("Originally published funnel of %s: MetaSUN %s, SUN %s",
                    args.baseline, published.get("metasun_per_sampled_gene"),
                    published.get("sun_per_sampled_gene"))
    funnels, frames, trials = {}, {}, {}
    for name, directory in arms.items():
        funnels[name] = json.loads((directory / FUNNEL_FILE).read_text())
        frames[name] = pd.read_csv(directory / STRUCTURES_FILE, index_col="index")
        relaxations = read_rows(directory / RELAXATIONS_FILE, RELAXATION_COLUMNS)
        trials[name] = len(relaxations) / max(funnels[name]["sampled"], 1)

    sampled = funnels["random"]["sampled"]
    rows = [{
        "metric": "relaxations per sampled gene",
        **{name: round(trials[name], 3) for name in arms},
    }]
    for key, label in _FUNNEL_ROWS:
        rows.append({
            "metric": label,
            **{name: funnels[name].get(key) for name in arms},
        })
        rows.append({
            "metric": f"  per sampled gene ({sampled})",
            **{name: round(funnels[name].get(key, 0) / sampled, 4) for name in arms},
        })
    table = pd.DataFrame(rows)

    out_dir = args.output_dir or (args.run_dir / "tables")
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / "arm_comparison.csv", index=False)

    joined = pd.DataFrame(index=frames["random"].index)
    for name, frame in frames.items():
        for column in ("has_structure", "energy_per_atom", "e_above_hull",
                       "valid_structure", "novel_structure", "best_trial"):
            if column in frame:
                joined[f"{name}_{column}"] = frame[column]
    joined["template_used"] = (
        joined.get("template_first_best_trial") == TEMPLATE_TRIAL)
    joined["union_used_template"] = joined.get("union_best_trial") == TEMPLATE_TRIAL
    joined.to_csv(out_dir / "per_gene_arms.csv")

    energy = {}
    for name in arms:
        column = f"{name}_e_above_hull"
        if column in joined:
            energy[name] = {
                "median_e_above_hull": float(joined[column].median(skipna=True)),
                "mean_e_above_hull": float(joined[column].mean(skipna=True)),
            }
    # Both kinds of trial were relaxed by the same five workers on the same
    # cards with the same settings, so their wall times are comparable and say
    # what a template start costs against a random one.
    union = read_rows(args.run_dir / RELAXATIONS_FILE, RELAXATION_COLUMNS)
    is_template = union["trial"] == TEMPLATE_TRIAL
    cost = {
        "median_seconds_random": float(union.loc[~is_template, "seconds"].median()),
        "median_seconds_template": float(union.loc[is_template, "seconds"].median()),
        "total_seconds_random": float(union.loc[~is_template, "seconds"].sum()),
        "total_seconds_template": float(union.loc[is_template, "seconds"].sum()),
    }

    summary = {
        "sampled": sampled,
        "relaxation_seconds": cost,
        "relaxations_per_sampled_gene": {k: round(v, 3) for k, v in trials.items()},
        "funnels": funnels,
        "energy": energy,
        "template_used_by_best_trial": int(joined["template_used"].sum()),
        "union_kept_the_template_trial": int(joined["union_used_template"].sum()),
    }
    (out_dir / "arm_comparison.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(table.to_string(index=False))
    print(json.dumps({"energy": energy, "relaxation_seconds": cost}, indent=2))


#: Trial-level columns written by :func:`stage_ceiling`.
TRIAL_COLUMNS = (
    "index", "trial", "source", "energy_per_atom", "e_above_hull",
    "valid_structure", "novel_structure", "metasun",
)


def stage_ceiling(args) -> None:
    """Score every *trial*, not every gene, and read the ceiling off it.

    The arms answer what the search *delivered*.  They cannot say how much is
    left, because a gene whose best structure is above the hull looks the same
    as one whose best structure was never sampled.  Scoring each trial
    separately separates the two:

    ``delivered``
        the verdict on the lowest-energy trial, which is what the protocol
        reports;
    ``ceiling``
        whether *any* trial of that gene is valid, novel and at or below the
        threshold.

    The difference is a selection loss, and it has exactly one mechanism here.
    At fixed composition ``e_above_hull`` is affine in the total energy, so the
    lowest-energy trial is also the lowest-``e_hull`` one; a gene can therefore
    only lose MetaSUN at the selection step when the lowest-energy trial is
    **not novel** and a higher-energy one is.  That is what a template start
    does when it lands on the LeMat-Bulk structure it was taken from, so this is
    the measurement that says whether the template trial is displacing novel
    structures rather than merely failing to add them.

    The ceiling is a *lower bound* on what the gene set could yield -- it is
    bounded by the trials that were actually run -- so it is reported next to
    the same quantity restricted to the random trials, whose gap says how much
    the template start moved the bound.

    Uniqueness is not applied.  It does not bind: every arm has exactly one
    fewer unique structure than valid one, so including it would change no digit
    and would make a per-trial verdict depend on which trial of another gene was
    selected.
    """
    from pymatgen.core import Structure

    from wyckoff_transformer.cli.protocol import GeneFingerprinter, load_genes
    from wyckoff_transformer.evaluation.hull_energy import HullEnergyCalculator
    from wyckoff_transformer.evaluation.novelty import NoveltyFilter
    from wyckoff_transformer.evaluation.protocol import METASTABLE_THRESHOLD, read_screen
    from wyckoff_transformer.evaluation.structure_novelty import build_novelty_reference
    from wyckoff_transformer.evaluation.structure_validity import is_valid

    run_dir = args.run_dir
    genes = load_genes(args.genes or (run_dir / "wyckoff_genes.json.gz"))
    screen = read_screen(run_dir / SCREEN_FILE)
    tables_dir = args.output_dir or (run_dir / "tables")
    scores_path = tables_dir / "per_trial_scores.csv"
    if scores_path.is_file():
        # Scoring 3258 trials costs a quarter of an hour and a pass over the
        # CIF export; the summary is a groupby.  Re-read rather than re-score.
        logger.info("Reusing %s", scores_path)
        _report_ceiling(pd.read_csv(scores_path), screen, tables_dir)
        return
    relaxations = read_rows(run_dir / RELAXATIONS_FILE, RELAXATION_COLUMNS)
    relaxations = relaxations[relaxations["status"] == "ok"]
    logger.info("Scoring %d trials over %d genes",
                len(relaxations), relaxations["index"].nunique())

    fingerprinter = GeneFingerprinter()
    hull = HullEnergyCalculator(args.mlip)

    gene_fingerprints = {}
    for index in sorted(set(relaxations["index"])):
        try:
            gene_fingerprints[int(index)] = fingerprinter.fingerprint(genes[int(index)])
        except Exception as exc:  # noqa: BLE001
            logger.warning("Gene %d: no fingerprint (%s)", index, exc)

    rows, structures, fingerprints, relaxed_fingerprints = [], {}, {}, {}
    for position, record in enumerate(relaxations.itertuples(index=False), start=1):
        key = (int(record.index), int(record.trial))
        path = Path(str(record.cif))
        if not path.is_file():
            continue
        try:
            structure = Structure.from_file(path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Trial %s: unreadable CIF (%s)", key, exc)
            continue
        structures[key] = structure
        fingerprints[key] = gene_fingerprints.get(key[0])
        try:
            relaxed_fingerprints[key] = fingerprinter.fingerprint_structure(structure)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Trial %s: no relaxed fingerprint (%s)", key, exc)
        try:
            energy = float(record.energy_per_atom) * len(structure)
            e_above_hull = hull.energy_above_hull(energy, structure.composition)
        except (ValueError, TypeError) as exc:
            logger.warning("Trial %s: e_above_hull failed (%s)", key, exc)
            e_above_hull = None
        rows.append({
            "index": key[0],
            "trial": key[1],
            "source": "template" if key[1] == TEMPLATE_TRIAL else "random",
            "energy_per_atom": float(record.energy_per_atom),
            "e_above_hull": e_above_hull,
            "valid_structure": bool(is_valid(structure)),
        })
        if position % 250 == 0:
            logger.info("Read %d/%d trials", position, len(relaxations))

    frame = pd.DataFrame(rows)
    scored = pd.DataFrame({
        "fingerprint": pd.Series(fingerprints),
        "structure": pd.Series(structures),
    }).dropna()
    scored["relaxed_fingerprint"] = pd.Series(relaxed_fingerprints).reindex(scored.index)
    reference = build_novelty_reference(
        pd.concat([scored["fingerprint"], scored["relaxed_fingerprint"].dropna()]),
        cache=args.reference_cache,
        lemat_cif_csv=args.lemat_cif_csv,
    )
    novelty = NoveltyFilter(reference)

    def is_novel(record: pd.Series) -> bool:
        """Novel iff no LeMat-Bulk entry sharing *either* fingerprint matches."""
        if not novelty.is_novel(record):
            return False
        relaxed = record.get("relaxed_fingerprint")
        if relaxed is None or (isinstance(relaxed, float) and pd.isna(relaxed)):
            return True
        return novelty.is_novel(
            pd.Series({"fingerprint": relaxed, "structure": record.structure}))

    novel = {key: is_novel(record) for key, record in scored.iterrows()}
    frame["novel_structure"] = [
        bool(novel.get((i, t), False)) for i, t in zip(frame["index"], frame["trial"])
    ]
    frame["metasun"] = (
        frame["valid_structure"]
        & frame["novel_structure"]
        & frame["e_above_hull"].le(METASTABLE_THRESHOLD)
    )

    tables_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(scores_path, index=False)
    _report_ceiling(frame, screen, tables_dir)


def _report_ceiling(frame: pd.DataFrame, screen, tables_dir: Path) -> None:
    """Summarise the per-trial scores, whole and over the gene-novel subset.

    The gene-novel split is the one the ceiling question is really about: a gene
    whose fingerprint LeMat-Bulk already has can still relax into a novel
    structure, but it is not where new material is expected to come from, and
    averaging the two hides how much room the novel genes have left.
    """
    novel_genes = set(screen.novel)
    # Denominators are *sampled* genes, as everywhere else in the funnel: a gene
    # sampled twice belongs once in the numerator and twice in the denominator.
    summary = {
        "all_genes": _ceiling_summary(frame, screen.n_sampled),
        "gene_novel": _ceiling_summary(
            frame[frame["index"].isin(novel_genes)], screen.n_sampled_novel),
        "gene_known": _ceiling_summary(
            frame[~frame["index"].isin(novel_genes)], screen.n_sampled_known),
    }
    summary["n_sampled"] = screen.n_sampled
    summary["n_sampled_novel"] = screen.n_sampled_novel
    summary["n_gene_novel_representatives"] = len(novel_genes)
    summary["derived_arms"] = _derived_arms(frame, novel_genes, screen.n_sampled)
    (tables_dir / "ceiling.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


def _best_by_energy(trials: pd.DataFrame) -> pd.Series:
    """MetaSUN verdict on each gene's lowest-energy trial: what the protocol picks."""
    if not len(trials):
        return pd.Series(dtype=bool)
    best = trials.loc[trials.groupby("index")["energy_per_atom"].idxmin()]
    return best.set_index("index")["metasun"]


def _derived_arms(frame: pd.DataFrame, novel_genes: set, sampled: int) -> dict:
    """Two arms that need no further relaxation, only a different choice.

    ``template_on_novel_genes``
        The template start only where the gene's own fingerprint is *absent*
        from LeMat-Bulk, and the random trials where it is present.  The
        per-trial scores say the template start's novelty cost falls entirely on
        gene-known genes -- where it lands on the very structure the gene names,
        wins on energy, and displaces a novel trial -- so withholding it there
        should keep the gain and drop the loss.  Selecting the template's own
        candidates by *augmented* fingerprint instead would do the same thing
        inside the retrieval, and is the version worth implementing.

    ``novelty_aware_selection``
        The lowest-energy trial *among the novel ones*, falling back to the
        lowest-energy trial when none is novel.  This attains the ceiling
        exactly: within one gene the composition is fixed, so the lowest-energy
        novel trial is also the lowest-``e_hull`` novel trial.  It is a change
        of readout rather than of search, and it is not free of interpretation
        -- the structure it reports is metastable with respect to a known
        polymorph the same run also found.
    """
    is_novel_gene = frame["index"].isin(novel_genes)
    random_trials = frame[frame["source"] == "random"]

    mixed = pd.concat([frame[is_novel_gene],
                       random_trials[~random_trials["index"].isin(novel_genes)]])
    novel_first = frame[frame["metasun"] | frame["novel_structure"]]

    arms = {
        "random": _best_by_energy(random_trials),
        "union": _best_by_energy(frame),
        "template_on_novel_genes": _best_by_energy(mixed),
        # Genes with no novel trial fall back to the union's own choice.
        "novelty_aware_selection": _best_by_energy(novel_first).combine_first(
            _best_by_energy(frame)),
    }
    return {
        name: {
            "delivered": int(verdict.sum()),
            "delivered_per_sampled_gene": round(float(verdict.sum()) / sampled, 4),
        }
        for name, verdict in arms.items()
    }


def _ceiling_summary(frame: pd.DataFrame, sampled: int) -> dict:
    """Delivered against any-trial MetaSUN, whole and split by trial source."""
    def arm(trials: pd.DataFrame) -> dict:
        if not len(trials):
            return {}
        best = trials.loc[trials.groupby("index")["energy_per_atom"].idxmin()]
        ceiling = trials.groupby("index")["metasun"].any()
        delivered = best.set_index("index")["metasun"]
        # A gene the selection lost: some trial is MetaSUN, the chosen one is
        # not.  At fixed composition that can only be a novelty verdict.
        lost = ceiling & ~delivered.reindex(ceiling.index).fillna(False)
        return {
            "trials": int(len(trials)),
            "genes": int(trials["index"].nunique()),
            "delivered": int(delivered.sum()),
            "delivered_per_sampled_gene": round(float(delivered.sum()) / sampled, 4),
            "ceiling": int(ceiling.sum()),
            "ceiling_per_sampled_gene": round(float(ceiling.sum()) / sampled, 4),
            "lost_to_selection": int(lost.sum()),
        }

    random_only = frame[frame["source"] == "random"]
    summary = {
        "sampled": sampled,
        "union": arm(frame),
        "random": arm(random_only),
        "template_trials_that_are_metasun": int(
            frame.loc[frame["source"] == "template", "metasun"].sum()),
        "template_trials_that_are_metastable_but_known": int(
            (frame["source"].eq("template")
             & frame["e_above_hull"].le(0.1)
             & ~frame["novel_structure"]).sum()),
    }
    # Best-of-k over the random trials, in the order the schedule drew them:
    # whether the curve is still rising says how far the search is from done.
    for k in (1, 2, 3):
        summary[f"random_best_of_{k}"] = arm(random_only[random_only["trial"] < k])
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("stage", choices=["arms", "score", "table", "ceiling"])
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="The run extended with the protocol's template stage.")
    parser.add_argument("--baseline", type=Path, default=None,
                        help=(
                            "The original run, whose published funnel is logged "
                            "for the record. Optional: the 'random' arm is "
                            "re-scored here from the same relaxations, so the "
                            "three arms come from one revision of the score "
                            "stage."
                        ))
    parser.add_argument("--genes", type=Path, default=None)
    parser.add_argument("--mlip", type=str, default="orb_conserv_inf")
    parser.add_argument("--reference-cache", type=Path, default=DEFAULT_REFERENCE_CACHE)
    parser.add_argument("--lemat-cif-csv", type=Path,
                        default=Path("data/lemat-bulk/lemat_pbe.csv.gz"))
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    if args.stage == "arms":
        stage_arms(args)
    elif args.stage == "score":
        stage_score_arms(args)
    elif args.stage == "ceiling":
        stage_ceiling(args)
    else:
        stage_table(args)


if __name__ == "__main__":
    main()
