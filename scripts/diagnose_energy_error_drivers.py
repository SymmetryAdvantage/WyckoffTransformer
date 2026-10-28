"""Explain residual variation in the pinned, one-minimum-per-gene DFT study.

Stages: prepare (verified geometries and independent descriptors), models
(composition/cohort-held-out diagnostics), report (tables, figures and W&B).
No energy calculation or energy-regressor inference is performed.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
import gzip
import hashlib
import io
import json
from pathlib import Path
import subprocess
import warnings
import zipfile

import numpy as np
import pandas as pd

from wyckoff_transformer.dataset_cache import build_info, dataset_cache_dir, load_split
from wyckoff_transformer.evaluation.energy_error_features import (
    calculation_features, composition_keys, structure_features,
)
from wyckoff_transformer.paths import runs_root, wandb_dir


MINIMA_SOURCE = "symmetry-advantage/WyckoffTransformer/energy_regressor_gene_minima_bjfoqicx:v0"
PRECISION_SOURCE = "symmetry-advantage/WyckoffTransformer/energy_regressor_precision_rycc5sdi:v1"
CODE = [
    "scripts/diagnose_energy_error_drivers.py",
    "src/wyckoff_transformer/evaluation/energy_error_features.py",
    "src/wyckoff_transformer/tests/test_energy_error_features.py",
]
SEED = 20261029


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, default=str, allow_nan=False) + "\n")


def geometry_job(task):
    from pymatgen.core import Structure
    geometry, raw, gene = task
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            result = structure_features(Structure.from_dict(json.loads(raw)), gene)
            result.update(geometry=geometry, feature_error=None)
            return result
        except Exception as exc:
            return dict(geometry=geometry, feature_error=f"{type(exc).__name__}: {exc}")


def prepare(root, run, args):
    import wandb
    from wyckoff_transformer.cli.template_novelty import fetch_entry
    from wyckoff_transformer.evaluation.protocol import GeneFingerprinter
    from wyckoff_transformer.evaluation.template_novelty import load_template_reference, template_id_from_augmented

    raw_root = runs_root() / "regressor_precision_20261028"
    minima = run.use_artifact(MINIMA_SOURCE)
    precision = run.use_artifact(PRECISION_SOURCE)
    sources = []
    for artifact, directory, local, names in [
        (minima, "minima", runs_root() / "energy_regressor_gene_minima_20261028", ["rows.csv.gz", "selection_audit.csv.gz"]),
        (precision, "precision", raw_root, ["symmetry.jsonl.gz", "figshare_metadata.json", "field_provenance.json", "provenance.json"]),
    ]:
        target = root / directory
        target.mkdir(exist_ok=True)
        for name in names:
            fetch_entry(artifact.get_entry(name), target, local)
        sources.append(dict(name=artifact.qualified_name, id=artifact.id, digest=artifact.digest,
                            files={name: artifact.manifest.entries[name].digest for name in names}))
    frame = pd.read_csv(root / "minima/rows.csv.gz", low_memory=False)
    frame = frame[frame.pool == "c_generated_dft"].copy()
    if frame.final_key.duplicated().any():
        raise ValueError("Primary DFT input is not one row per final gene")
    wanted = set(frame.geometry)
    genes = {}
    with gzip.open(root / "precision/symmetry.jsonl.gz", "rt") as handle:
        for line in handle:
            item = json.loads(line)
            if item["geometry"] in wanted:
                genes[item["geometry"]] = item
    if set(genes) != wanted:
        raise ValueError("Missing pinned final-gene symmetry records")
    for r in frame.itertuples():
        if ":".join(map(str, genes[r.geometry]["key"])) != r.final_key:
            raise ValueError("Pinned symmetry record and selected gene disagree")

    # The downloaded Figshare zip is only a working copy. Each cohort's gzip
    # bytes are verified against the metadata pinned in the original artifact.
    archive = args.figshare_zip or raw_root / "figshare_v6.zip"
    if not archive.exists():
        import requests
        response = requests.get("https://figshare.com/ndownloader/articles/29145101/versions/6", stream=True, timeout=120)
        response.raise_for_status()
        archive = root / "figshare_v6.zip"
        with archive.open("wb") as handle:
            for chunk in response.iter_content(1 << 20):
                handle.write(chunk)
    metadata = json.loads((root / "precision/figshare_metadata.json").read_text())
    file_ids = {}
    for item in metadata["files"]:
        file_ids.setdefault(item["computed_md5"], []).append(item["id"])
    tasks, calc, verified_files = [], [], []
    with zipfile.ZipFile(archive) as z, gzip.open(root / "selected_structures.jsonl.gz", "wt") as saved:
        for cohort, selected in frame.groupby("cohort"):
            blob = z.read(cohort)
            md5 = hashlib.md5(blob).hexdigest()
            if md5 not in file_ids:
                raise ValueError(f"Figshare file does not match pinned metadata: {cohort}")
            verified_files.append(dict(cohort=cohort, md5=md5, file_ids=file_ids[md5]))
            table = pd.read_csv(io.BytesIO(blob), compression="gzip")
            for r in selected.itertuples():
                row = table.iloc[int(r.index)]
                text = row.structure
                if hashlib.sha256(text.encode()).hexdigest() != r.geometry:
                    raise ValueError(f"Selected geometry digest mismatch: {cohort}/{r.index}")
                gene = genes[r.geometry]["gene"]
                tasks.append((r.geometry, text, gene))
                entry = json.loads(row.entry) if "entry" in row and pd.notna(row.entry) else None
                composition = genes[r.geometry]["composition"]
                values = calculation_features(entry, composition)
                values.update(geometry=r.geometry,
                              saved_structural_validity=row.get("structural_validity"),
                              saved_smact_validity=row.get("smact_validity"))
                if entry and "energy" in entry and abs(float(entry["energy"]) - float(r.energy)) > 1e-6:
                    raise ValueError("ComputedEntry raw energy disagrees with the pinned target")
                calc.append(values)
                saved.write(json.dumps(dict(geometry=r.geometry, structure=json.loads(text), gene=gene)) + "\n")
            print("Verified source geometries", cohort, len(selected), flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        records = []
        for i, record in enumerate(executor.map(geometry_job, tasks, chunksize=64)):
            records.append(record)
            if (i + 1) % 5000 == 0:
                print("Geometry descriptors", i + 1, "/", len(tasks), flush=True)
    features = pd.DataFrame(records)
    base = frame[["geometry", "final_key", "candidate_id", "cohort", "truth", "final_prediction", "final_gene_novel_fixed", "novel_structure_fixed", "gene_in_train", "gene_min_formation_energy_per_atom"]]
    features = base.merge(features, on="geometry", validate="one_to_one").merge(pd.DataFrame(calc), on="geometry", validate="one_to_one")
    for column in [c for c in features if c.startswith("potcar_")]:
        features[column] = features[column].fillna(0.)
    audit = pd.read_csv(root / "minima/selection_audit.csv.gz", usecols=["pool", "group_key", "selection_status"])
    observations = audit[(audit.pool == "c_generated_dft") & audit.selection_status.isin(["selected", "higher energy or tie"])].groupby("group_key").size()
    features["n_observations_available"] = features.final_key.map(observations)

    fields = json.loads((root / "precision/field_provenance.json").read_text())
    cache = dataset_cache_dir(fields["dataset"])
    if build_info(cache, "train") != fields["cache_build"]:
        raise ValueError("Training cache differs from recorded model cache")
    wanted_formulas = set(features.formula_key.dropna())
    wanted_systems = set(features.chemsys_key.dropna())
    formulas, systems, sgs = Counter(), Counter(), Counter()
    training = load_split(cache, "train", columns=["elements", "multiplicity", "spacegroup_number"])
    for i, (elements, mults, sg) in enumerate(training.itertuples(index=False, name=None)):
        formula, system = composition_keys(elements, mults)
        if formula in wanted_formulas:
            formulas[formula] += 1
        if system in wanted_systems:
            systems[system] += 1
        sgs[int(sg)] += 1
        if (i + 1) % 500000 == 0:
            print("Training coverage", i + 1, "/", len(training), flush=True)
    features["train_formula_count"] = features.formula_key.map(formulas).fillna(0)
    features["train_chemsys_count"] = features.chemsys_key.map(systems).fillna(0)
    features["train_space_group_count"] = features.space_group.map(sgs).fillna(0)
    support = dict(formula=dict(formulas), chemsys=dict(systems), space_group=dict(sgs), train_rows=len(training), cache_build=build_info(cache, "train"))
    dump(root / "training_support.json", support)
    templates, template_provenance = load_template_reference(cache=cache, splits=("train", "val", "test"))
    fp = GeneFingerprinter()
    template_ids = {g: template_id_from_augmented(fp.fingerprint(item["gene"])) for g, item in genes.items()}
    features["anonymous_template_id"] = features.geometry.map(template_ids)
    features["template_known_corpus"] = features.anonymous_template_id.isin(templates).astype(int)
    features["error"] = features.final_prediction - features.truth
    features["absolute_error"] = features.error.abs()
    features["log_absolute_error"] = np.log1p(features.absolute_error / .025)
    features.to_csv(root / "features.csv.gz", index=False)
    dump(root / "provenance.json", dict(
        date=datetime.now(UTC).isoformat(), git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        git_dirty=bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()),
        sources=sources, verified_figshare_files=verified_files, template_reference=template_provenance,
        model_field_provenance=fields, units="eV/atom", primary_pool="generated-only DFT minimum per final gene",
        code_sha256={path: hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in CODE},
        no_new_predictions_relaxations_or_energy_calculations=True,
    ))
    print("Prepared feature rows", len(features), "failures", int(features.feature_error.notna().sum()), flush=True)
    artifact = wandb.Artifact(f"energy_error_features_{run.id}", type="energy_regressor_evaluation")
    for name in ["features.csv.gz", "selected_structures.jsonl.gz", "training_support.json", "provenance.json"]:
        artifact.add_file(str(root / name))
    for path in CODE:
        artifact.add_file(path, name="code/" + path)
    run.log_artifact(artifact).wait()


def feature_matrix(frame):
    """Prespecified feature blocks; observed target energies are never predictors."""
    # A fixed POTCAR label for an element is only a proxy for that element's
    # presence. Only within-element variation could identify settings effects.
    variants = Counter(c.split("_")[1] for c in frame if c.startswith("potcar_") and frame[c].sum() > 0)
    variable_potcars = [c for c in frame if c.startswith("potcar_") and variants[c.split("_")[1]] > 1]
    groups = dict(
        novelty=["final_gene_novel_fixed", "novel_structure_fixed", "gene_in_train"],
        size=["n_atoms", "n_gene_atoms", "n_orbits", "n_elements"],
        dof=["positional_dof", "lattice_dof", "positional_dof_per_orbit", "positional_dof_per_gene_atom", "fraction_atoms_free_orbits"],
        packing=["volume", "volume_per_atom", "density", "lattice_anisotropy", "covalent_sphere_fraction", "min_pair_distance", "min_covalent_distance_ratio", "mean_nearest_covalent_ratio", "fraction_short_contacts"],
        chemistry=[c for c in frame if c.startswith("element_fraction_")] + ["mean_atomic_number", "std_atomic_number", "mean_electronegativity", "electronegativity_range", "mean_covalent_radius", "fraction_transition_metal", "fraction_lanthanide", "fraction_mp_u_element", "mp_u_chemistry_proxy"],
        support=["train_formula_count", "train_chemsys_count", "train_space_group_count", "template_known_corpus", "n_observations_available"],
        prediction=["final_prediction"],
        calculation=["dft_metadata_available", "dft_uses_hubbard", "dft_hubbard_per_atom"] + variable_potcars,
    )
    values = {}
    for group, columns in groups.items():
        for column in columns:
            values[column] = pd.to_numeric(frame[column], errors="coerce")
    matrix = pd.DataFrame(values, index=frame.index)
    for group, columns in [("cohort", ["cohort"]), ("symmetry", ["space_group", "crystal_system"]), ("calculation", ["dft_run_type"])]:
        dummy = pd.get_dummies(frame[columns].astype("string").fillna("missing"), prefix=columns, dtype=float)
        matrix = pd.concat([matrix, dummy], axis=1)
        groups.setdefault(group, []).extend(dummy.columns)
    # Constant chemical/pseudopotential columns have no diagnostic information.
    keep = matrix.nunique(dropna=False) > 1
    matrix = matrix.loc[:, keep].astype(float)
    groups = {g: [c for c in columns if c in matrix] for g, columns in groups.items()}
    return matrix, groups


def association_tables(frame, root):
    numeric = ["volume", "volume_per_atom", "n_atoms", "n_gene_atoms", "n_orbits", "n_elements", "space_group", "positional_dof", "lattice_dof", "density", "covalent_sphere_fraction", "min_covalent_distance_ratio", "fraction_short_contacts", "mean_atomic_number", "fraction_transition_metal", "fraction_lanthanide", "train_formula_count", "train_chemsys_count", "n_observations_available", "final_prediction", "mp_u_chemistry_proxy", "dft_uses_hubbard", "template_known_corpus"]
    correlations, bins = [], []
    for subset, f in [("all", frame), ("novel_genes", frame[frame.final_gene_novel_fixed == True]), ("known_genes", frame[frame.final_gene_novel_fixed == False])]:
        strata = f.cohort.astype(str) + "/" + f.final_gene_novel_fixed.astype(str) + "/" + f.novel_structure_fixed.astype(str)
        for feature in numeric:
            for outcome in [] if feature == "space_group" else ["error", "absolute_error"]:
                valid = f[[feature, outcome]].dropna()
                valid = valid[np.isfinite(valid).all(axis=1)]
                ranks = valid.rank(pct=True)
                centered = ranks - ranks.groupby(strata.loc[valid.index]).transform("mean")
                correlations.append(dict(subset=subset, feature=feature, outcome=outcome, n=len(valid),
                                         spearman=valid[feature].corr(valid[outcome], method="spearman"),
                                         cohort_novelty_adjusted_rank_correlation=centered[feature].corr(centered[outcome])))
            if feature in ["space_group", "mp_u_chemistry_proxy", "dft_uses_hubbard", "template_known_corpus"]:
                labels = f[feature].astype("string").fillna("missing")
            elif f[feature].nunique() > 1:
                labels = pd.qcut(f[feature], q=5, duplicates="drop").astype("string").fillna("missing")
            else:
                labels = f[feature].astype("string").fillna("missing")
            if feature == "positional_dof":
                labels = pd.cut(f[feature], [-.5, .5, 2.5, 5.5, 11.5, np.inf]).astype("string")
            for label, ff in f.groupby(labels, dropna=False):
                bins.append(dict(subset=subset, feature=feature, bin=str(label), n=len(ff), mae=ff.absolute_error.mean(), bias=ff.error.mean(), median_error=ff.error.median(), median_absolute_error=ff.absolute_error.median(), fraction_under_100=(ff.error < -.1).mean(), fraction_over_100=(ff.error > .1).mean()))
    pd.DataFrame(correlations).to_csv(root / "correlations.csv", index=False)
    pd.DataFrame(bins).to_csv(root / "feature_bins.csv", index=False)
    chemistry = []
    for z in range(1, 119):
        column = f"element_fraction_{z}"
        f = frame[frame.final_gene_novel_fixed == True]
        for present, ff in f.groupby(f[column] > 0):
            if present and len(ff):
                chemistry.append(dict(atomic_number=z, n=len(ff), mae=ff.absolute_error.mean(), bias=ff.error.mean(), median_absolute_error=ff.absolute_error.median()))
    pd.DataFrame(chemistry).to_csv(root / "element_errors.csv", index=False)
    frame.groupby(["cohort", "final_gene_novel_fixed", "dft_run_type"], dropna=False).agg(n=("error", "size"), mae=("absolute_error", "mean"), bias=("error", "mean")).to_csv(root / "cohort_settings.csv")


def models(root, run, args):
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.metrics import r2_score
    from sklearn.model_selection import GroupKFold

    provenance = json.loads((root / "provenance.json").read_text())
    provenance.setdefault("preparation_code_sha256", provenance["code_sha256"])
    provenance["code_sha256"] = {path: hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in CODE}
    provenance["model_parameters"] = dict(iterations=args.iterations, folds=args.folds, seed=SEED,
        max_leaf_nodes=15, min_samples_leaf=100, l2_regularization=10., learning_rate=.07)
    dump(root / "provenance.json", provenance)
    frame = pd.read_csv(root / "features.csv.gz", low_memory=False)
    frame = frame[frame.feature_error.isna() & np.isfinite(frame.error)].reset_index(drop=True)
    association_tables(frame, root)
    scores, predictions, importance = [], [], []
    for subset in ["all", "novel_genes"]:
        f = frame if subset == "all" else frame[frame.final_gene_novel_fixed == True]
        f = f.reset_index(drop=True)
        x, blocks = feature_matrix(f)
        base = blocks["novelty"] + blocks["cohort"]
        configs = {"controls": base, "full": list(x.columns)}
        if subset == "novel_genes":
            configs.update({"controls+" + g: list(dict.fromkeys(base + columns)) for g, columns in blocks.items() if g not in ["novelty", "cohort"]})
        partitions = [("heldout_formula", list(GroupKFold(n_splits=args.folds).split(x, groups=f.formula_key)))]
        large_cohorts = f.cohort.value_counts()
        large_cohorts = large_cohorts[large_cohorts >= 300].index
        partitions.append(("heldout_cohort", [(np.flatnonzero(f.cohort != cohort), np.flatnonzero(f.cohort == cohort)) for cohort in large_cohorts]))
        for protocol, folds in partitions:
            names = configs if protocol == "heldout_formula" else {g: configs[g] for g in ["controls", "full"]}
            for outcome in ["error", "log_absolute_error"]:
                y = f[outcome].to_numpy()
                for name, columns in names.items():
                    oof = np.full(len(f), np.nan)
                    for fold, (train, test) in enumerate(folds):
                        model = HistGradientBoostingRegressor(max_iter=args.iterations, max_leaf_nodes=15, min_samples_leaf=100,
                            l2_regularization=10., learning_rate=.07, early_stopping=False, random_state=SEED)
                        model.fit(x.iloc[train][columns], y[train])
                        oof[test] = model.predict(x.iloc[test][columns])
                        if name == "full" and protocol == "heldout_formula":
                            baseline = np.mean((y[test] - oof[test])**2)
                            rng = np.random.default_rng(SEED + fold)
                            for group, group_columns in blocks.items():
                                if group in ["novelty", "cohort"] or not group_columns:
                                    continue
                                permuted = x.iloc[test][columns].copy()
                                # Keep the cohort distribution unchanged while
                                # disturbing this coherent feature block.
                                for positions in f.iloc[test].groupby("cohort").indices.values():
                                    donor = rng.permutation(positions)
                                    permuted.iloc[positions, permuted.columns.get_indexer(group_columns)] = x.iloc[test].iloc[donor][group_columns].to_numpy()
                                loss = np.mean((y[test] - model.predict(permuted))**2)
                                importance.append(dict(subset=subset, outcome=outcome, fold=fold, group=group, increase_mse=loss - baseline))
                    valid = np.isfinite(oof)
                    record = dict(subset=subset, protocol=protocol, outcome=outcome, model=name, n=int(valid.sum()), r2=r2_score(y[valid], oof[valid]), rmse=float(np.sqrt(np.mean((y[valid] - oof[valid])**2))))
                    if outcome == "error":
                        record["mae_after_oof_error_prediction"] = float(np.mean(abs(y[valid] - oof[valid])))
                        record["uncorrected_mae"] = float(np.mean(abs(y[valid])))
                    else:
                        record["spearman_error_severity"] = float(pd.Series(y[valid]).corr(pd.Series(oof[valid]), method="spearman"))
                    scores.append(record)
                    if name in ["controls", "full"]:
                        predictions.append(pd.DataFrame(dict(geometry=f.geometry, subset=subset, protocol=protocol, outcome=outcome, model=name, truth=y, prediction=oof)))
                    print("Model", subset, protocol, outcome, name, "R2", round(record["r2"], 3), flush=True)
    pd.DataFrame(scores).to_csv(root / "model_scores.csv", index=False)
    pd.DataFrame(importance).to_csv(root / "block_importance.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_csv(root / "out_of_fold_predictions.csv.gz", index=False)


def report(root, run, args):
    import wandb
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scores = pd.read_csv(root / "model_scores.csv")
    bins = pd.read_csv(root / "feature_bins.csv")
    importance = pd.read_csv(root / "block_importance.csv")
    grouped = importance[(importance.subset == "novel_genes") & (importance.outcome == "log_absolute_error")].groupby("group").increase_mse.agg(["mean", "std"]).sort_values("mean", ascending=False)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), layout="constrained")
    for ax, feature in zip(axes.flat[:5], ["n_atoms", "positional_dof", "volume_per_atom", "min_covalent_distance_ratio", "train_formula_count"]):
        f = bins[(bins.subset == "novel_genes") & (bins.feature == feature) & (bins.n >= 30)]
        ax.bar(np.arange(len(f)), f.mae * 1000)
        ax.set(title=feature, ylabel="MAE (meV/atom)", xticks=np.arange(len(f)), xticklabels=f.bin)
        ax.tick_params(axis="x", labelrotation=35, labelsize=7)
    ax = axes.flat[5]
    ax.barh(grouped.index[::-1], grouped["mean"].to_numpy()[::-1])
    ax.set(title="Held-out feature-block importance", xlabel="Increase in squared log-error loss")
    for extension in ["png", "pdf"]:
        fig.savefig(root / f"error_drivers.{extension}", dpi=180)
    plt.close(fig)
    provenance = json.loads((root / "provenance.json").read_text())
    features = pd.read_csv(root / "features.csv.gz", low_memory=False)
    failed = features[features.feature_error.notna()]
    failed[["geometry", "cohort", "feature_error", "truth", "final_prediction"]].to_csv(root / "feature_exclusions.csv", index=False)
    lines = ["## F Error drivers beyond novelty", "",
        f"Measured 2026-10-29 at git `{provenance['git_commit']}` with exact working-tree "
        f"source in the [W&B analysis run]({run.url}). Inputs are the pinned "
        "one-minimum-per-final-gene generated DFT pool in §E. Geometry bytes and saved "
        "final genes are verified against the original source digests. No new energy "
        "calculation, relaxation or regressor inference was run.", "",
        f"Of {len(features):,} selected structures, {len(failed)} fail the new check that "
        "the recovered gene preserves the geometry's reduced composition; they are "
        "explicit exclusions, retained with their source IDs in feature_exclusions.csv. "
        "The prior protocol composition audit concerned initial-to-final pairs in B "
        "and did not check this C gene-to-geometry identity. A missing prediction is "
        "also excluded from error models. Previously reported C numbers remain the "
        "pinned historical results; the diagnostic denominators below reflect this audit.", "",
        "The feature table includes cell size and volume, volume per atom, density, "
        "covalent-radius packing/contact proxies, space group, atomic and lattice degrees "
        "of freedom, elemental fractions, chemistry descriptors, predicted formation energy, "
        "training-formula/chemical-system coverage, corpus template coverage, observation "
        "count, and recorded DFT Hubbard-U/pseudopotential metadata. Composition coverage "
        "uses only the original training split. Anonymous-template coverage uses all three "
        "corpus splits and is labelled accordingly. Missing metadata is not treated as GGA.", "",
        "All marginal effects are associations. Cohort/novelty-adjusted rank correlations "
        "and within-cohort block permutations supplement raw bins. Diagnostic models "
        "predict signed residual and log(1 + absolute residual / 0.025 eV/atom), never "
        "using observed target energy as a feature. The controls contain cohort and "
        "novelty. Grouped cross-validation holds out entire reduced formulas. A second "
        "check holds out each source cohort with at least 300 rows; smaller cohorts "
        "remain available for training but are not its test denominator. Hyperparameters "
        "are fixed; no model selection is performed on held-out scores. Error-model "
        "predictions diagnose reproducible patterns and are not a validated replacement "
        "energy model.", "", "### Held-out predictability", "",
        "| Population | Holdout | Outcome | Model | N | R² | Residual MAE after error prediction (meV/atom) |",
        "|---|---|---|---|---:|---:|---:|"]
    for r in scores[scores.model.isin(["controls", "full"])].itertuples():
        mae = f"{1000 * r.mae_after_oof_error_prediction:.1f}" if pd.notna(r.mae_after_oof_error_prediction) else "—"
        lines.append(f"| {r.subset} | {r.protocol} | {r.outcome} | {r.model} | {r.n} | {r.r2:.3f} | {mae} |")
    lines += ["", "### Feature blocks beyond cohort and novelty", "",
        "Single-block additions below use novel genes and formula-held-out predictions. "
        "The outcome is log absolute error; R² quantifies predictability of error severity, "
        "not energy accuracy. Correlated blocks can substitute for one another.", "",
        "| Added block | R² |", "|---|---:|"]
    f = scores[(scores.subset == "novel_genes") & (scores.protocol == "heldout_formula") & (scores.outcome == "log_absolute_error")].sort_values("r2", ascending=False)
    for r in f.itertuples():
        lines.append(f"| {r.model} | {r.r2:.3f} |")
    lines += ["", "### Raw feature bins for novel genes", "",
        "Errors below are meV/atom. These bins do not by themselves control for chemistry "
        "or other structural variables. Their counts and signed errors are included to "
        "distinguish bias from scatter.", "",
        "| Feature | Bin | N | MAE | Bias | Median absolute error |", "|---|---|---:|---:|---:|---:|"]
    for r in bins[(bins.subset == "novel_genes") & bins.feature.isin(["n_atoms", "positional_dof", "volume_per_atom", "min_covalent_distance_ratio", "train_formula_count", "mp_u_chemistry_proxy", "dft_uses_hubbard"]) & (bins.n >= 30)].itertuples():
        lines.append(f"| {r.feature} | {r.bin} | {r.n} | {1000*r.mae:.1f} | {1000*r.bias:.1f} | {1000*r.median_absolute_error:.1f} |")
    lines += ["", "The artifact preserves the feature matrix, selected geometries, source settings "
        "and coverage provenance, all bins/correlations, element and cohort summaries, "
        "out-of-fold predictions, model scores, within-cohort permutation importance, "
        "figures and exact source. Reproduce with `scripts/diagnose_energy_error_drivers.py` "
        "and the platform's documented interpreter. The stages are prepare, models and report.", ""]
    text = "\n".join(lines)
    (root / "report.md").write_text(text)
    if args.update_doc:
        doc = Path("docs/energy_regressor_precision.md")
        old = doc.read_text()
        if "## F Error drivers" in old:
            start = old.index("## F Error drivers")
            end = old.index("## Reproduction and saved outputs", start)
            old = old[:start] + old[end:]
        doc.write_text(old.replace("## Reproduction and saved outputs", text + "\n## Reproduction and saved outputs", 1))
    # Recheck saved code identity: running code must match the published snapshot.
    for path, sha in provenance["code_sha256"].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != sha:
            raise ValueError("Source changed during this analysis; rerun prepare before publishing")
    dump(root / "verification.json", dict(feature_rows=int(pd.read_csv(root / "features.csv.gz", usecols=["geometry"]).shape[0]),
         feature_code_matches_preparation=True, cv_groups="reduced formula", no_observed_energy_predictor=True,
         holdout_cohort_min_rows=300, hyperparameters=dict(iterations=args.iterations, leaves=15, min_leaf=100, l2=10, learning_rate=.07), seed=SEED))
    artifact = wandb.Artifact(f"energy_error_drivers_{run.id}", type="energy_regressor_evaluation", metadata=provenance)
    for path in root.iterdir():
        if path.is_file() and path.name not in ["wandb_result.json", "figshare_v6.zip"]:
            artifact.add_file(str(path))
    for path in CODE:
        artifact.add_file(path, name="code/" + path)
    if args.update_doc:
        artifact.add_file("docs/energy_regressor_precision.md", name="docs/energy_regressor_precision.md")
    run.log_artifact(artifact).wait()
    run.summary["novel_gene_formula_holdout_log_error"] = f.set_index("model").r2.to_dict()
    print(scores.to_string(index=False), flush=True)
    print("W&B", run.url, flush=True)


def main():
    import wandb
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "models", "report", "all"])
    parser.add_argument("--output", type=Path, default=runs_root() / "energy_regressor_error_drivers_20261029")
    parser.add_argument("--figshare-zip", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--update-doc", action="store_true")
    args = parser.parse_args()
    root = args.output
    root.mkdir(parents=True, exist_ok=True)
    state = root / "wandb_result.json"
    previous = json.loads(state.read_text()) if state.exists() else {}
    with wandb.init(entity="symmetry-advantage", project="WyckoffTransformer", job_type="energy_regressor_error_drivers",
        name="Energy error drivers beyond novelty 2026-10-29", mode="online", dir=wandb_dir(),
        id=previous.get("id"), resume="must" if previous else None) as run:
        dump(state, dict(id=run.id, url=run.url))
        for stage in ["prepare", "models", "report"] if args.stage == "all" else [args.stage]:
            globals()[stage](root, run, args)


if __name__ == "__main__":
    main()
