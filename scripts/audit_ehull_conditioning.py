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

from wyckoff_transformer.trainer import WyckoffTrainer, load_model_weights


DEFAULT_TARGETS = (0.0, 0.025, 0.05, 0.1, 0.2)


def _target_tag(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate matched WyFormer samples at an e-hull target grid."
    )
    parser.add_argument("--run-path", type=Path, default=Path("runs/upi73i4k"))
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("generated/upi73i4k/ehull_conditioning_audit"),
    )
    parser.add_argument("--n-samples", type=int, default=250)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--device", type=torch.device, default=torch.device("cpu"))
    parser.add_argument("--targets", type=float, nargs="+", default=DEFAULT_TARGETS)
    args = parser.parse_args()

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
    starts = trainer._sample_start_tokens_from_distribution(args.n_samples)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(starts.cpu(), args.output_dir / "fixed_start_tokens.pt")
    manifest = {
        "run_path": str(args.run_path),
        "checkpoint": str(weights_path),
        "seed": args.seed,
        "n_samples": args.n_samples,
        "targets_eV_per_atom": args.targets,
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
        condition = torch.full(
            (args.n_samples, 1), target, dtype=torch.float32, device=args.device
        )
        genes = trainer.generate_structures(
            n_structures=args.n_samples,
            calibrate=False,
            start_tensor=starts.clone(),
            cond=condition,
        )
        output = args.output_dir / f"wyckoff_genes_ehull_{_target_tag(target)}.json.gz"
        with gzip.open(output, "wt", encoding="utf-8") as handle:
            json.dump(genes, handle)
        print(f"target={target:g}: {len(genes)}/{args.n_samples} genes -> {output}")
        print(
            "  CrySPR: uv run python generated/upi73i4k/genbench/relax_remaining.py "
            f"{output} --output-dir {args.output_dir / ('cryspr_ehull_' + _target_tag(target))} "
            "--workers 20 --n-trials 3 --fmax 0.05"
        )


if __name__ == "__main__":
    main()
