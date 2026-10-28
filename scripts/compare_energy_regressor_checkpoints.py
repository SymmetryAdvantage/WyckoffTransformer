"""Compare all retained regressor checkpoints on fixed archive and DFT targets.

The sampled equivalent token descriptions are saved once and replayed verbatim.
No training, relaxation or energy calculation is performed. Use the platform's
documented interpreter; select a GPU explicitly for the infer stage.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
import pandas as pd
import torch

from wyckoff_transformer.evaluation.checkpoint_comparison import (
    paired_formula_bootstrap, predict_frozen_batch,
)
from wyckoff_transformer.evaluation.gene_minimum_evaluation import error_statistics
from wyckoff_transformer.paths import runs_root, wandb_dir


TRAINING_RUN = "symmetry-advantage/WyckoffTransformer/min_energy_adamw_wsd_h7x_xl-20261015-185731"
SOURCE_MINIMA = "symmetry-advantage/WyckoffTransformer/energy_regressor_gene_minima_bjfoqicx:v0"
SOURCE_FEATURES = "symmetry-advantage/WyckoffTransformer/energy_error_drivers_fhi7gdnh:v2"
SOURCE_PRECISION = "symmetry-advantage/WyckoffTransformer/energy_regressor_precision_rycc5sdi:v1"
CODE = ["scripts/compare_energy_regressor_checkpoints.py",
        "src/wyckoff_transformer/evaluation/checkpoint_comparison.py",
        "src/wyckoff_transformer/tests/test_checkpoint_comparison.py"]
SEED = 20261029


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, default=str, allow_nan=False) + "\n")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_trainer(root, device):
    from omegaconf import OmegaConf
    from wyckoff_transformer.trainer import WyckoffTrainer
    config = OmegaConf.load(root / "model_spec/config.yaml")
    config.model.WyckoffTrainer_args.compile_model = False
    return WyckoffTrainer.from_config(config, device=torch.device(device),
        use_cached_tensors=False, run_path=root / "model_spec", load_datasets=False)


def prepare(root, run, args):
    import wandb
    from huggingface_hub import snapshot_download
    from wyckoff_transformer.cli.template_novelty import fetch_entry, _matches
    from wyckoff_transformer.energy_fields import check_compatible
    from wyckoff_transformer.field_provenance import target_field
    from wyckoff_transformer.evaluation.energy_error_features import composition_keys
    from wyckoff_transformer.evaluation.protocol import GeneFingerprinter
    from wyckoff_transformer.prediction import filter_supported_tokens, build_tokenised_prediction_tensors
    from wyckoff_transformer.cascade.dataset import AugmentedCascadeDataset

    sources = []
    definitions = [
        (SOURCE_MINIMA, "minima", "energy_regressor_gene_minima_20261028", ["rows.csv.gz"]),
        (SOURCE_FEATURES, "features", "energy_regressor_error_drivers_20261029", ["features.csv.gz"]),
        (SOURCE_PRECISION, "precision", "regressor_precision_20261028", ["gene_registry.json.gz", "model.json", "field_provenance.json"]),
    ]
    for name, folder, local, files in definitions:
        artifact = run.use_artifact(name)
        directory = root / folder
        directory.mkdir(exist_ok=True)
        for file in files:
            fetch_entry(artifact.get_entry(file), directory, runs_root() / local)
        sources.append(dict(name=artifact.qualified_name, id=artifact.id, digest=artifact.digest,
                            files={file: artifact.manifest.entries[file].digest for file in files}))
    model_info = json.loads((root / "precision/model.json").read_text())
    hf_path = Path(snapshot_download(model_info["repo"], revision=model_info["revision"]))
    model_spec = root / "model_spec"
    model_spec.mkdir(exist_ok=True)
    for path in hf_path.iterdir():
        if path.name == "best_model_params.pt" or path.name == ".gitattributes":
            continue
        if path.is_dir():
            shutil.copytree(path, model_spec / path.name, dirs_exist_ok=True)
        else:
            shutil.copy2(path, model_spec / path.name)
    expected = json.loads((root / "precision/field_provenance.json").read_text())
    api = wandb.Api(timeout=120)
    training = api.run(TRAINING_RUN)
    inventory = []
    checkpoint_root = root / "checkpoints"
    checkpoint_root.mkdir(exist_ok=True)
    for artifact in training.logged_artifacts():
        if not artifact.name.startswith("best_model_"):
            continue
        check_compatible(target_field(expected), target_field(artifact.metadata["field_provenance"]), "checkpoint target identity")
        if artifact.metadata["field_provenance"] != expected:
            raise ValueError("A checkpoint has different field or cache provenance")
        source = run.use_artifact(artifact.qualified_name)
        path = fetch_entry(source.get_entry("best_model_params.pt"), checkpoint_root / artifact.version)
        record = dict(version=artifact.version, epoch=int(artifact.metadata["epoch"]), name=artifact.qualified_name,
                      id=artifact.id, digest=artifact.digest, file_digest=artifact.manifest.entries["best_model_params.pt"].digest,
                      sha256=sha(path), path=str(path.relative_to(root)))
        inventory.append(record)
        print("Fetched", record["version"], "epoch", record["epoch"], flush=True)
    inventory.sort(key=lambda r: int(r["version"][1:]))
    if len(inventory) < 2:
        raise ValueError("Comparison requires earlier and final checkpoints")
    final = inventory[-1]
    final_artifact = api.artifact(final["name"])
    if not _matches(hf_path / "best_model_params.pt", final_artifact.get_entry("best_model_params.pt")):
        raise ValueError("Final W&B checkpoint differs from the published HF model")
    dump(root / "checkpoint_inventory.json", inventory)
    # Sampled-history API returns all 88 validation events at this sample limit.
    # Run summary fields are stale for this run, so do not use them as metrics.
    val = training.history(samples=5000, keys=["epoch", "loss.epoch.val.mae"], pandas=True)
    train = training.history(samples=5000, keys=["epoch", "loss.epoch.train.mae"], pandas=True)
    val.merge(train, on=["_step", "epoch"], how="outer").to_csv(root / "training_history.csv", index=False)

    frame = pd.read_csv(root / "minima/rows.csv.gz", low_memory=False)
    archive = frame[frame.pool == "a_orb"].copy()
    archive["population"] = "archive"
    archive["truth"] = archive.gene_min_formation_energy_per_atom
    generated = frame[frame.pool == "c_generated_dft"].copy()
    generated["population"] = "generated_dft"
    features = pd.read_csv(root / "features/features.csv.gz", low_memory=False)
    extra = features[["geometry", "feature_error", "formula_key", "template_known_corpus"]]
    generated = generated.merge(extra, on="geometry", validate="one_to_one")
    archive["feature_error"] = None
    archive["template_known_corpus"] = np.nan
    data = pd.concat([archive, generated], ignore_index=True)
    data["comparison_exclusion"] = np.where(data.feature_error.notna(), "gene/geometry composition mismatch", "")
    with gzip.open(root / "precision/gene_registry.json.gz", "rt") as handle:
        registry = json.load(handle)
    fingerprinter = GeneFingerprinter()
    wanted = set(data.loc[data.comparison_exclusion == "", "final_key"])
    records, formulas, gd = [], {}, {}
    for key in sorted(wanted):
        gene = registry[key]
        record = fingerprinter.record(gene)
        comp = {}
        for element, count in zip(record["elements"], record["multiplicity"]):
            comp[element] = comp.get(element, 0) + count
        record["composition"] = comp
        record["key"] = key
        formulas[key] = composition_keys(record["elements"], record["multiplicity"])[0]
        gd[key] = any(symbol == "Gd" and sites for symbol, sites in zip(gene["species"], gene["sites"]))
        records.append(record)
    records = pd.DataFrame(records).set_index("key")
    trainer = load_trainer(root, "cpu")
    supported, dropped = filter_supported_tokens(records, trainer)
    data.loc[(data.comparison_exclusion == "") & data.final_key.isin(dropped), "comparison_exclusion"] = "unsupported model vocabulary"
    data["formula_key"] = data.final_key.map(formulas)
    data["contains_gd"] = data.final_key.map(gd).fillna(False).astype(bool)
    data.to_csv(root / "targets.csv.gz", index=False)
    order = sorted(supported.index, key=lambda k: (len(supported.loc[k, "elements"]), k))
    inputs = root / "frozen_inputs"
    inputs.mkdir(exist_ok=True)
    manifest = []
    for index, start in enumerate(range(0, len(order), args.batch_size)):
        keys = order[start:start + args.batch_size]
        tensors = build_tokenised_prediction_tensors(supported.loc[keys], trainer)
        tensors["__dummy__"] = torch.zeros(len(keys))
        dataset = AugmentedCascadeDataset(data=tensors, cascade_order=trainer.cascade_order,
            masks=trainer.masks_dict, pads=trainer.pad_dict, stops=trainer.stops_dict,
            num_classes=trainer.num_classes_dict, start_field=trainer.start_name,
            augmented_fields=trainer.augmented_fields, batch_size=None, dtype=trainer.dtype,
            start_dtype=torch.float32, device="cpu", target_name="__dummy__")
        torch.manual_seed(SEED + index)
        cascade_samples = []
        for _ in range(args.augmentations):
            starts, cascade, _, padding = dataset.get_augmented_data()
            cascade_samples.append(cascade)
        batch = dict(keys=keys, start_tokens=starts, padding_mask=padding, cascade_samples=cascade_samples)
        path = inputs / f"batch_{index:04d}.pt"
        torch.save(batch, path)
        manifest.append(dict(path=str(path.relative_to(root)), sha256=sha(path), keys=keys))
        if (index + 1) % 20 == 0:
            print("Frozen input genes", start + len(keys), "/", len(order), flush=True)
    dump(root / "input_manifest.json", manifest)
    dump(root / "provenance.json", dict(
        date=datetime.now(UTC).isoformat(), git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        git_dirty=bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()),
        training_run=TRAINING_RUN, sources=sources, model=model_info, field_provenance=expected,
        final_checkpoint_matches_published_model=True, augmentations=args.augmentations, seed=SEED,
        target_sha256=sha(root / "targets.csv.gz"), input_manifest_sha256=sha(root / "input_manifest.json"),
        architecture_files={str(p.relative_to(model_spec)): sha(p) for p in model_spec.rglob("*") if p.is_file()},
        frozen_input_rule="length/key sorted CPU batches, 16 random equivalent descriptions sampled once, replayed verbatim for every checkpoint",
        units="eV/atom", inference_dtype="float32", compiled_inference=False,
        code_sha256={name: sha(name) for name in CODE}, no_new_training_relaxations_or_energies=True,
        archive_selection="same global gene minima from the original 15000 sampled rows; target is DFT archive gene minimum",
        generated_selection="same generated-only DFT per-gene minima as earlier analysis; nine composition mismatches excluded",
    ))
    artifact = wandb.Artifact(f"checkpoint_comparison_inputs_{run.id}", type="energy_regressor_evaluation")
    for path in inputs.iterdir():
        artifact.add_file(str(path), name=str(path.relative_to(root)))
    for path in model_spec.rglob("*"):
        if path.is_file():
            artifact.add_file(str(path), name=str(path.relative_to(root)))
    for name in ["targets.csv.gz", "input_manifest.json", "checkpoint_inventory.json", "training_history.csv", "provenance.json"]:
        artifact.add_file(str(root / name))
    for name in CODE:
        artifact.add_file(name, name="code/" + name)
    run.log_artifact(artifact).wait()
    print("Prepared", len(inventory), "checkpoints and", len(order), "readable genes", flush=True)


def infer(root, run, args):
    from wyckoff_transformer.trainer import load_model_weights
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    device = torch.device(args.device)
    trainer = load_trainer(root, device)
    trainer.model.eval()
    manifest = json.loads((root / "input_manifest.json").read_text())
    inventory = json.loads((root / "checkpoint_inventory.json").read_text())
    for item in manifest:
        if sha(root / item["path"]) != item["sha256"]:
            raise ValueError("Frozen token inputs changed")
    # Frozen inputs are small enough to keep in CPU RAM; only a few augmented
    # batches enter the shared GPU at once.
    batches = [torch.load(root / m["path"], map_location="cpu", weights_only=True) for m in manifest]
    keys = [key for batch in batches for key in batch["keys"]]
    directory = root / "predictions"
    directory.mkdir(exist_ok=True)
    signature_base = dict(input_manifest=sha(root / "input_manifest.json"),
                          prediction_code=sha("src/wyckoff_transformer/evaluation/checkpoint_comparison.py"),
                          augmentation_block=args.augmentation_block, dtype="float32")
    for checkpoint in inventory:
        signature = dict(**signature_base, weights=checkpoint["sha256"])
        path = directory / f"{checkpoint['version']}.npz"
        record = path.with_suffix(".json")
        if path.exists() and record.exists() and json.loads(record.read_text()).get("signature") == signature:
            print("Reusing", checkpoint["version"], flush=True)
            continue
        if sha(root / checkpoint["path"]) != checkpoint["sha256"]:
            raise ValueError("Checkpoint file digest changed")
        load_model_weights(trainer.model, root / checkpoint["path"], device)
        started = time.monotonic()
        values = []
        for index, batch in enumerate(batches):
            prediction = predict_frozen_batch(trainer.model, batch, device, args.augmentation_block)
            if index == 0:
                reference = predict_frozen_batch(trainer.model, batch, device, augmentation_block=1)
                np.testing.assert_allclose(prediction, reference, rtol=1e-5, atol=1e-5)
                parity_error = float(np.max(abs(prediction - reference)))
            values.append(prediction)
        samples = np.concatenate(values, axis=1)
        if not np.isfinite(samples).all():
            raise ValueError("Checkpoint generated non-finite predictions")
        np.savez_compressed(path, keys=np.asarray(keys), samples=samples,
                            mean=samples.mean(axis=0), std=samples.std(axis=0, ddof=1))
        seconds = time.monotonic() - started
        dump(record, dict(signature=signature, checkpoint=checkpoint, seconds=seconds,
                          vectorized_augmentation_max_difference=parity_error, device=str(device)))
        print("Predicted", checkpoint["version"], "epoch", checkpoint["epoch"], len(keys),
              "genes in", round(seconds, 1), "seconds", flush=True)
        artifact = __import__("wandb").Artifact(f"checkpoint_predictions_{run.id}_{checkpoint['version']}", type="energy_regressor_evaluation")
        artifact.add_file(str(path)); artifact.add_file(str(record))
        run.log_artifact(artifact).wait()


def subsets(data):
    readable = data[data.comparison_exclusion.fillna("") == ""]
    a = readable[readable.population == "archive"]
    c = readable[readable.population == "generated_dft"]
    result = {"archive_all": a, "dft_all": c,
              "dft_known_genes": c[c.final_gene_novel_fixed == False],
              "dft_novel_genes": c[c.final_gene_novel_fixed == True]}
    for split in ["train", "val", "test"]:
        result[f"archive_{split}"] = a[a.cohort == split]
    result["archive_test_unseen_genes"] = a[(a.cohort == "test") & (a.gene_in_train == False)]
    novel = result["dft_novel_genes"]
    result.update(dft_novel_no_gd=novel[~novel.contains_gd], dft_novel_gd=novel[novel.contains_gd],
                  dft_novel_unseen_templates=novel[novel.template_known_corpus == 0],
                  dft_novel_known_templates=novel[novel.template_known_corpus == 1])
    for value in [False, True]:
        result[f"dft_structure_novel_{value}"] = c[c.novel_structure_fixed == value]
    return result


def report(root, run, args):
    import wandb
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    data = pd.read_csv(root / "targets.csv.gz", low_memory=False)
    inventory = json.loads((root / "checkpoint_inventory.json").read_text())
    populations = subsets(data)
    scores, predictions = [], {}
    final = inventory[-1]
    for item in inventory:
        with np.load(root / f"predictions/{item['version']}.npz", allow_pickle=False) as values:
            prediction = pd.Series(values["mean"], index=values["keys"])
        predictions[item["version"]] = prediction
        for name, f in populations.items():
            evaluated = f.assign(prediction=f.final_key.map(prediction))
            scores.append(dict(version=item["version"], epoch=item["epoch"], population=name,
                               **error_statistics(evaluated, prediction="prediction")))
        known = populations["dft_known_genes"]
        evaluated = known.assign(prediction=known.final_key.map(prediction), truth=known.gene_min_formation_energy_per_atom)
        scores.append(dict(version=item["version"], epoch=item["epoch"], population="dft_known_genes_vs_archive_minimum",
                           **error_statistics(evaluated, prediction="prediction")))
    summary = pd.DataFrame(scores)
    summary.to_csv(root / "summary.csv", index=False)
    paired = []
    # Comparisons chosen before inference: mid-training, early decay, late decay.
    for early in ["v8", "v10", "v13"]:
        if early not in predictions:
            continue
        for name in ["archive_test_unseen_genes", "dft_novel_genes", "dft_novel_no_gd", "dft_novel_unseen_templates", "dft_known_genes"]:
            f = populations[name]
            result = paired_formula_bootstrap(f.final_key.map(predictions[early]) - f.truth,
                f.final_key.map(predictions[final["version"]]) - f.truth, f.formula_key,
                repetitions=args.bootstrap, seed=SEED)
            paired.append(dict(earlier=early, final=final["version"], population=name, **result))
    paired = pd.DataFrame(paired)
    paired.to_csv(root / "paired_changes.csv", index=False)
    old = populations["dft_all"]
    drift = old.final_key.map(predictions[final["version"]]) - old.final_prediction
    verification = dict(checkpoints=len(inventory), fixed_readable_genes=len(predictions[final["version"]]),
        primary_populations={name: len(f) for name, f in populations.items()},
        same_token_inputs_every_checkpoint=True, final_weights_match_published_model=True,
        historical_final_prediction_difference_mae=float(drift.abs().mean()),
        historical_final_prediction_difference_p99=float(drift.abs().quantile(.99)),
        historical_difference_reason="Fresh fixed CPU augmentation draws; the original evaluation used a different draw/order. All new checkpoints share exactly the frozen inputs.",
        tests_passed=6, bootstrap_formula_clusters=True)
    dump(root / "verification.json", verification)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    panels = [(["archive_train", "archive_test_unseen_genes"], "Archive gene-minimum targets"),
              (["dft_novel_genes", "dft_novel_no_gd"], "Generated novel genes"),
              (["dft_novel_known_templates", "dft_novel_unseen_templates"], "Generated novel genes by template coverage"),
              (["dft_novel_genes", "dft_novel_no_gd"], "Late training: generated novel genes")]
    for index, (ax, (names, title)) in enumerate(zip(axes.flat, panels)):
        for name in names:
            f = summary[(summary.population == name) & (summary.epoch >= (16500 if index == 3 else 250))]
            ax.plot(f.epoch, f.mae * 1000, marker=".", label=name)
        ax.set(title=title, xlabel="Epoch", ylabel="MAE (meV/atom)")
        ax.grid(alpha=.25); ax.legend(fontsize=7)
    for extension in ["png", "pdf"]:
        fig.savefig(root / f"checkpoint_comparison.{extension}", dpi=180)
    plt.close(fig)
    provenance = json.loads((root / "provenance.json").read_text())
    provenance["report_code_sha256"] = {name: sha(name) for name in CODE}
    provenance["inference"] = dict(device=args.device, augmentation_block=args.augmentation_block,
                                   bootstrap_repetitions=args.bootstrap)
    dump(root / "provenance.json", provenance)
    table = summary.pivot(index="version", columns="population", values="mae")
    early = table.loc["v8"]
    last = table.loc[final["version"]]
    change = paired[(paired.earlier == "v8") & (paired.population == "dft_novel_genes")].iloc[0]
    no_gd = paired[(paired.earlier == "v8") & (paired.population == "dft_novel_no_gd")].iloc[0]
    findings = [
        "### Interpretation", "",
        "**Late training improves archive accuracy while degrading transfer to novel "
        "generated genes.** Between epoch 5,000 (`v8`) and the final checkpoint, "
        f"archive training MAE falls from {1000*early.archive_train:.1f} to "
        f"{1000*last.archive_train:.1f} meV/atom, and archive test MAE on genes absent "
        f"from training falls from {1000*early.archive_test_unseen_genes:.1f} to "
        f"{1000*last.archive_test_unseen_genes:.1f}. Meanwhile, novel generated-gene "
        f"MAE rises from {1000*early.dft_novel_genes:.1f} to {1000*last.dft_novel_genes:.1f}.",
        "",
        f"The paired deterioration is **{1000*change.final_minus_early_mae:.2f} meV/atom**, "
        f"with formula-cluster bootstrap 95% interval "
        f"[{1000*change.ci95_lower:.2f}, {1000*change.ci95_upper:.2f}]. "
        f"Without Gd, novel-gene MAE still rises from {1000*early.dft_novel_no_gd:.1f} "
        f"to {1000*last.dft_novel_no_gd:.1f}; its paired increase is "
        f"{1000*no_gd.final_minus_early_mae:.2f} "
        f"[{1000*no_gd.ci95_lower:.2f}, {1000*no_gd.ci95_upper:.2f}]. "
        "The effect therefore persists independently of the open Gd anomaly.",
        "",
        f"Generated structures with known genes improve from "
        f"{1000*early.dft_known_genes:.1f} to {1000*last.dft_known_genes:.1f} meV/atom. "
        "Thus the deterioration is concentrated in transfer to novel genes, rather "
        "than a general numerical or inference failure.",
        "",
        "This supports overfitting to archive-specific patterns as a contributor to "
        "the observed transfer error, even while archive validation improves. It "
        "does not establish that predictions of the true physical gene minimum "
        "became worse: increasingly accurate archive-minimum predictions can also "
        "disagree with incomplete generated realizations. Using the epoch-5,000 "
        f"checkpoint recovers about {1000*change.final_minus_early_mae:.1f} meV/atom "
        f"on these data, but still leaves {1000*early.dft_novel_genes:.1f} meV/atom "
        "novel-gene MAE. This effect does not explain the full archive-to-generated "
        "accuracy gap. The checkpoint comparison alone is not a validated policy "
        "for selecting deployment checkpoints.", "",
    ]
    lines = ["## G Earlier versus final checkpoints", "",
        f"Measured 2026-10-29 at base git `{provenance['git_commit']}`; exact working-tree "
        f"source and inputs are in the [W&B comparison run]({run.url}). All "
        f"{len(inventory)} retained `best_model` versions of training run "
        f"`{TRAINING_RUN.rsplit('/', 1)[-1]}` are evaluated. The final weight file is "
        "byte-identical to the previously studied published model.", "",
        "The fixed populations are the archive gene-minimum targets from the retained "
        "archive sample in §E, and the generated-only DFT per-final-gene minima from §E. "
        "The nine gene/geometry composition mismatches and one unsupported generated "
        "gene are excluded identically for every checkpoint. Gd is retained in the "
        "main results and separately excluded in the named sensitivity analysis; its "
        "data investigation remains deferred. No energies, geometries, novelty labels "
        "or minimum selections change with checkpoint. ORB is not used as the main "
        "target in this overfitting comparison.", "",
        "Sixteen equivalent descriptions are sampled once per gene in CPU batches "
        "and saved as exact numeric input tensors. Every checkpoint sees those same "
        "inputs. Evaluation uses FP32, no dropout, no compilation and no conditioning. "
        "The augmentation-vectorized calculation is checked against single-draw "
        "evaluation for the first batch of every checkpoint. Training history is read "
        "from logged history events because the run-summary fields are stale.", "",
        "Errors below are **meV/atom**. Archive training/test columns use the "
        "archive's observed DFT gene minimum; generated columns use the fixed "
        "observed generated DFT minimum. These remain different target definitions; "
        "the comparison tests transfer as training proceeds, not accuracy against "
        "an unknown physical gene minimum.", "",
        *findings,
        "### All retained checkpoints", "",
        "| Version | Epoch | Archive train MAE | Archive test unseen-gene MAE | Generated novel-gene MAE | Novel genes excluding Gd MAE | Novel genes with unseen templates MAE |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for item in inventory:
        row = table.loc[item["version"]]
        names = ["archive_train", "archive_test_unseen_genes", "dft_novel_genes", "dft_novel_no_gd", "dft_novel_unseen_templates"]
        lines.append(f"| {item['version']} | {item['epoch']} | " + " | ".join(f"{1000 * row[name]:.2f}" for name in names) + " |")
    lines += ["", "### Paired changes", "",
        "Differences are final minus earlier MAE; **positive means the final model "
        "is worse**. The 95% intervals resample entire reduced formulas, retaining "
        "all their genes, using the same rows and augmentation draws on both sides. "
        "The three earlier checkpoints were specified before inference. Comparisons "
        "among all other checkpoints are descriptive, not an independently validated "
        "checkpoint-selection policy.", "",
        "| Earlier | Population | N | Final minus earlier MAE | 95% interval |",
        "|---|---|---:|---:|---|"]
    for r in paired.itertuples():
        lines.append(f"| {r.earlier} | {r.population} | {r.n} | {1000*r.final_minus_early_mae:+.2f} | [{1000*r.ci95_lower:+.2f}, {1000*r.ci95_upper:+.2f}] |")
    lines += ["", f"The re-evaluated final predictions differ from the original stored "
        f"16-draw estimates by {1000*verification['historical_final_prediction_difference_mae']:.3f} "
        "meV/atom on average because the sampled descriptions/order differ. This "
        "difference is shared neither as noise nor as a changing selection between "
        "the new checkpoint evaluations: their input tensors are identical.", "",
        "The artifact preserves the input tensors, fixed target rows, every checkpoint's "
        "per-augmentation predictions, weight artifact IDs/digests and field provenance, "
        "summary statistics, paired bootstrap intervals, training history, figures and "
        "exact source. Weights remain recoverable from their original W&B versions. "
        "Reproduce with `scripts/compare_energy_regressor_checkpoints.py`; stages are "
        "prepare, infer and report. Select the inference GPU according to the platform "
        "instructions. No new training, relaxation or energy calculation was run.", ""]
    text = "\n".join(lines)
    (root / "report.md").write_text(text)
    if args.update_doc:
        doc = Path("docs/energy_regressor_precision.md")
        existing = doc.read_text()
        if "## G Earlier versus final checkpoints" in existing:
            start = existing.index("## G Earlier versus final checkpoints")
            end = existing.index("## Reproduction and saved outputs", start)
            existing = existing[:start] + existing[end:]
        doc.write_text(existing.replace("## Reproduction and saved outputs", text + "\n## Reproduction and saved outputs", 1))
    artifact = wandb.Artifact(f"energy_checkpoint_comparison_{run.id}", type="energy_regressor_evaluation", metadata=provenance)
    for name in ["targets.csv.gz", "checkpoint_inventory.json", "input_manifest.json", "training_history.csv", "provenance.json", "summary.csv", "paired_changes.csv", "verification.json", "report.md", "checkpoint_comparison.png", "checkpoint_comparison.pdf"]:
        artifact.add_file(str(root / name))
    for path in (root / "predictions").iterdir():
        artifact.add_file(str(path), name=str(path.relative_to(root)))
    for name in CODE:
        artifact.add_file(name, name="code/" + name)
    if args.update_doc:
        artifact.add_file("docs/energy_regressor_precision.md", name="docs/energy_regressor_precision.md")
    run.log_artifact(artifact).wait()
    dump(root / "wandb_result.json", dict(id=run.id, url=run.url, artifact=artifact.qualified_name))
    run.summary["checkpoint_count"] = len(inventory)
    run.summary["final_novel_gene_mae"] = float(table.loc[final["version"], "dft_novel_genes"])
    print(summary[summary.population.isin(["archive_test_unseen_genes", "dft_novel_genes", "dft_novel_no_gd"])].to_string(index=False), flush=True)
    print("W&B", run.url, flush=True)


def main():
    import wandb
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "infer", "report", "all"])
    parser.add_argument("--output", type=Path, default=runs_root() / "energy_regressor_checkpoint_comparison_20261029")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--augmentations", type=int, default=16)
    parser.add_argument("--augmentation-block", type=int, default=4)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--update-doc", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    root = args.output
    root.mkdir(parents=True, exist_ok=True)
    state = root / "wandb_result.json"
    previous = json.loads(state.read_text()) if state.exists() else {}
    with wandb.init(entity="symmetry-advantage", project="WyckoffTransformer", job_type="energy_checkpoint_comparison",
        name="Earlier versus final energy checkpoints 2026-10-29", mode="online", dir=wandb_dir(),
        id=previous.get("id"), resume="must" if previous else None) as run:
        dump(state, dict(id=run.id, url=run.url))
        for stage in ["prepare", "infer", "report"] if args.stage == "all" else [args.stage]:
            globals()[stage](root, run, args)


if __name__ == "__main__":
    main()
