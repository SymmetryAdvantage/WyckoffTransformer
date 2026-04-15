"""CLI entry point for CrySPR: crystal structure prediction via PyXtal + MACE."""
import argparse
import json
import gzip
import logging
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from wyckoff_transformer.cryspr.calculator import build_mace_calculator
from wyckoff_transformer.cryspr.generator import func_run

logger = logging.getLogger(__name__)

_SINGLE_THREAD_ENV_VARS = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
}


def _worker(
    id_gene: int,
    wyckoffgene: dict,
    model: str,
    device: str,
    output_dir: Path,
    model_name: str,
    n_trials: int,
    fmax: float,
) -> dict:
    """Build a per-process calculator and run func_run; returns a result dict."""
    # Belt-and-suspenders: env vars are already inherited from the parent (spawn),
    # but set them again here and also limit PyTorch's own runtime thread count.
    for key, value in _SINGLE_THREAD_ENV_VARS.items():
        os.environ[key] = value
    import torch
    torch.set_num_threads(1)

    calculator = build_mace_calculator(model=model, device=device)
    atoms, formula, energy, energy_per_atom, cif = func_run(
        id_gene=id_gene,
        wyckoffgene=wyckoffgene,
        calculator=calculator,
        output_dir=output_dir,
        model_name=model_name,
        n_trials=n_trials,
        fmax=fmax,
    )
    return {
        "model": model_name,
        "id": id_gene,
        "formula": formula,
        "energy": energy,
        "energy_per_atom": energy_per_atom,
        "cif": cif,
    }


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
        "--workers",
        type=int,
        default=1,
        help=(
            "Number of parallel worker processes. "
            "Each worker builds its own MACE calculator. "
            "Values > 1 automatically set OMP/MKL/OPENBLAS_NUM_THREADS=1."
        ),
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

    if args.input.suffix == ".gz":
        opener = gzip.open
    else:
        opener = open
    with opener(args.input, mode="rt", encoding="utf-8") as f:
        data = json.load(f)

    end = args.end if args.end is not None else len(data)
    selected = data[args.start:end]

    model_name = args.model_name or Path(args.model.split("?")[0]).stem

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.workers > 1:
        for key, value in _SINGLE_THREAD_ENV_VARS.items():
            os.environ[key] = value
        logger.info(
            "Parallel mode: %d workers; set %s",
            args.workers,
            ", ".join(f"{k}={v}" for k, v in _SINGLE_THREAD_ENV_VARS.items()),
        )

        results_map: dict[int, dict] = {}
        futures_to_id: dict = {}
        # spawn: each worker is a fresh process that inherits the OS environment
        # (including the thread-count vars above) before any Python import runs.
        # fork (the Linux default) would inherit already-initialised thread pools,
        # making the env vars ineffective.
        _spawn_ctx = multiprocessing.get_context("spawn")
        pool = ProcessPoolExecutor(max_workers=args.workers, mp_context=_spawn_ctx)
        try:
            for i, wyckoffgene in enumerate(selected, start=args.start):
                fut = pool.submit(
                    _worker,
                    i,
                    wyckoffgene,
                    args.model,
                    args.device,
                    args.output_dir,
                    model_name,
                    args.n_trials,
                    args.fmax,
                )
                futures_to_id[fut] = i

            for fut in as_completed(futures_to_id):
                i = futures_to_id[fut]
                try:
                    results_map[i] = fut.result()
                except Exception as exc:
                    logger.error("Gene %d failed: %s", i, exc)
                    results_map[i] = {
                        "model": model_name,
                        "id": i,
                        "formula": None,
                        "energy": None,
                        "energy_per_atom": None,
                        "cif": None,
                    }
        except KeyboardInterrupt:
            logger.warning("Interrupted — terminating workers.")
            for fut in futures_to_id:
                fut.cancel()
            # Terminate workers before calling shutdown so the internal
            # _ExecutorManagerThread doesn't block on dead communication
            # channels when shutdown(wait=True) tries to join it.
            for proc in pool._processes.values():
                proc.terminate()
            # shutdown(wait=False) stops the manager thread without blocking.
            pool.shutdown(wait=False, cancel_futures=True)
            # Join each OS process ourselves — this calls waitpid() and
            # prevents zombies. Fall back to SIGKILL if a worker is stuck
            # (e.g. blocked in a CUDA op that ignores SIGTERM).
            for proc in pool._processes.values():
                proc.join(timeout=10)
                if proc.is_alive():
                    proc.kill()
                    proc.join()
            raise SystemExit(130)  # 128 + SIGINT, conventional exit code
        else:
            pool.shutdown(wait=True)

        results = [results_map[i] for i in sorted(results_map)]
    else:
        logger.info("Building MACE calculator from %s", args.model)
        calculator = build_mace_calculator(model=args.model, device=args.device)

        results = []
        for i, wyckoffgene in enumerate(selected, start=args.start):
            atoms, formula, energy, energy_per_atom, cif = func_run(
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
                "cif": cif,
            })

    results_csv = args.output_dir / f"{model_name}_results.csv"
    pd.DataFrame(results).to_csv(results_csv, index=False)
    logger.info("Results written to %s", results_csv)


if __name__ == "__main__":
    main()
