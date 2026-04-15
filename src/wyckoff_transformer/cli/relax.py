"""CLI entry point for CrySPR: crystal structure prediction via PyXtal + MACE."""
import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from wyckoff_transformer.cryspr.calculator import build_mace_calculator
from wyckoff_transformer.cryspr.generator import func_run

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="wyformer-cryspr",
        description=(
            "Generate and relax crystal structures from Wyckoff gene representations "
            "using PyXtal and a MACE ML force field."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "input",
        type=Path,
        help="JSON file containing a list of Wyckoff gene dicts.",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="First index (Python-style, inclusive) to process.",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="Last index (exclusive). Defaults to the end of the file.",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help=(
            "MACE model to use: either a local filesystem path or an HTTPS URL. "
            "URL-based models are downloaded once and cached in "
            "~/.cache/wyckoff_transformer/mace_models/."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("cryspr_output"),
        help="Root directory for all output files and sub-directories.",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=6,
        help="Number of independent generation + relaxation trials per Wyckoff gene.",
    )
    parser.add_argument(
        "--fmax",
        type=float,
        default=0.01,
        help="Force convergence criterion in eV/Å.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help=(
            "Label written to the results CSV 'model' column. "
            "Defaults to the stem of the model file/URL."
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="PyTorch device: 'cpu', 'cuda', or 'auto' (picks CUDA when available).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    with open(args.input) as f:
        data = json.load(f)

    end = args.end if args.end is not None else len(data)
    selected = data[args.start:end]

    model_name = args.model_name or Path(args.model.split("?")[0]).stem

    logger.info("Building MACE calculator from %s", args.model)
    calculator = build_mace_calculator(model=args.model, device=args.device)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for i, wyckoffgene in enumerate(selected, start=args.start):
        atoms, formula, energy, energy_per_atom = func_run(
            id_gene=i,
            wyckoffgene=wyckoffgene,
            calculator=calculator,
            output_dir=args.output_dir,
            model_name=model_name,
            n_trials=args.n_trials,
            fmax=args.fmax,
        )
        results.append({
            "model": model_name,
            "id": i,
            "formula": formula,
            "energy": energy,
            "energy_per_atom": energy_per_atom,
        })

    results_csv = args.output_dir / f"{model_name}_results.csv"
    pd.DataFrame(results).to_csv(results_csv, index=False)
    logger.info("Results written to %s", results_csv)


if __name__ == "__main__":
    main()
