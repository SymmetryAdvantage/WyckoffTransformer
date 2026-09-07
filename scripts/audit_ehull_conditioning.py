"""Generate a reproducible e-hull-conditioning probe from a saved WyFormer run.

The script deliberately covers only the discrete-generation part of the audit.
Run CrySPR afterwards with the printed commands so that generation and
relaxation can be restarted independently.
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import torch
from omegaconf import OmegaConf

from wyckoff_transformer.cli import parse_condition_assignments
from wyckoff_transformer.trainer import WyckoffTrainer, load_model_weights


DEFAULT_TARGETS = (0.0, 0.025, 0.05, 0.1, 0.2)


def _target_tag(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def _feature_tag(feature: str) -> str:
    """What the swept channel is called in a path.

    `energy_above_hull` keeps its historical short name so the shell scripts that read
    `generated/upi73i4k/ehull_conditioning_audit/wyckoff_genes_ehull_*.json.gz` --
    run_ehull_audit_relaxations.sh and four others -- still find the files a rerun of the
    original audit produces.
    """
    return "ehull" if feature == "energy_above_hull" else feature


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate matched WyFormer samples at an e-hull target grid."
    )
    parser.add_argument("--run-path", type=Path, default=Path("runs/upi73i4k"))
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Defaults to generated/<run name>/<swept channel>_conditioning_audit, so a "
             "sweep of a different run or a different channel cannot land on top of an "
             "earlier audit's genes and manifest.",
    )
    parser.add_argument("--n-samples", type=int, default=250)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--device", type=torch.device, default=torch.device("cpu"))
    parser.add_argument("--targets", type=float, nargs="+", default=DEFAULT_TARGETS)
    parser.add_argument("--sweep-feature", default="energy_above_hull",
                        help="Which conditioning channel --targets moves.")
    parser.add_argument("--baseline", action="append", metavar="NAME=VALUE", default=None,
                        help="Value to hold another conditioning channel at while the "
                             "sweep runs, e.g. --baseline max_force=0. Defaults to 0 for "
                             "every channel other than --sweep-feature.")
    parser.add_argument("--force", action="store_true",
                        help="Write into an output directory that already holds an audit. "
                             "Without it an existing manifest.json is refused, because "
                             "regenerating genes on top of a completed audit leaves the "
                             "relaxations and scores downstream of it describing samples "
                             "that no longer exist.")
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = (Path("generated") / args.run_path.name
                           / f"{_feature_tag(args.sweep_feature)}_conditioning_audit")

    if args.n_samples <= 0:
        parser.error("--n-samples must be positive")
    config_path = args.run_path / "config.yaml"
    weights_path = args.run_path / "best_model_params.pt"
    if not config_path.is_file() or not weights_path.is_file():
        parser.error(f"{args.run_path} must contain config.yaml and best_model_params.pt")

    torch.manual_seed(args.seed)
    config = OmegaConf.load(config_path)
    trainer = WyckoffTrainer.from_config(
        config_dict=config,
        device=args.device,
        use_cached_tensors=False,
        run_path=args.run_path,
        load_datasets=False,
    )
    load_model_weights(trainer.model, weights_path, args.device)
    features = trainer.condition_features
    if args.sweep_feature not in features:
        raise SystemExit(
            f"This model conditions on {list(features)}, not {args.sweep_feature!r}; "
            "pass --sweep-feature.")
    baseline = {name: 0.0 for name in features}
    baseline.update(parse_condition_assignments(args.baseline or ()))
    unknown = sorted(set(baseline) - set(features))
    if unknown:
        raise SystemExit(f"--baseline names {unknown}, which are not conditioning features")
    print(f"sweeping {args.sweep_feature} with the rest held at "
          + ", ".join(f"{name}={value:g}" for name, value in baseline.items()
                      if name != args.sweep_feature))

    starts = trainer._sample_start_tokens_from_distribution(args.n_samples)

    existing = args.output_dir / "manifest.json"
    if existing.exists() and not args.force:
        raise SystemExit(
            f"{existing} already exists -- {args.output_dir} holds a completed audit. "
            "Point --output-dir somewhere else, or pass --force to overwrite it.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(starts.cpu(), args.output_dir / "fixed_start_tokens.pt")
    manifest = {
        "run_path": str(args.run_path),
        "checkpoint": str(weights_path),
        "seed": args.seed,
        "n_samples": args.n_samples,
        "sweep_feature": args.sweep_feature,
        "targets_eV_per_atom": args.targets,
        "baseline": baseline,
        "device": str(args.device),
        "cryspr_protocol": {
            "model": "MACE-MP-0a-small",
            "n_trials": 3,
            "fmax": 0.05,
        },
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    for index, target in enumerate(args.targets):
        # Keep the stochastic sequence reproducible but independent per target.
        torch.manual_seed(args.seed + 1 + index)
        # Sweep one channel, hold the others at their baseline. On a single-feature
        # model this is the (n_samples, 1) tensor it always was; on a multi-channel one
        # it is the only way the sweep means anything, since filling every column with
        # the target would move three variables at once.
        values = dict(baseline)
        values[args.sweep_feature] = target
        condition = trainer.build_condition_from_values(
            values, args.n_samples, device=args.device)
        genes = trainer.generate_structures(
            n_structures=args.n_samples,
            calibrate=False,
            start_tensor=starts.clone(),
            cond=condition,
        )
        output = (args.output_dir /
                  f"wyckoff_genes_{_feature_tag(args.sweep_feature)}_{_target_tag(target)}.json.gz")
        with gzip.open(output, "wt", encoding="utf-8") as handle:
            json.dump(genes, handle)
        print(f"target={target:g}: {len(genes)}/{args.n_samples} genes -> {output}")
        print(
            "  CrySPR: uv run python generated/upi73i4k/genbench/relax_remaining.py "
            f"{output} --output-dir "
            f"{args.output_dir / ('cryspr_' + _feature_tag(args.sweep_feature) + '_' + _target_tag(target))} "
            "--workers 20 --n-trials 3 --fmax 0.05"
        )


if __name__ == "__main__":
    main()
