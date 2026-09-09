"""Build, inspect and draw from a (chemical system, space group) prior.

    wyformer-system-prior build lemat_bulk_fmax1
    wyformer-system-prior show  --required Li --allowed Li-Na-K-Mg-...-O
    wyformer-system-prior sample -n 1000 --required Li --allowed ... -o plan.json

`build` is the expensive one and is done once per dataset; `show` answers "is this
query worth a GPU hour" in a second; `sample` writes the plan a generation run
executes. See docs/chemical_system_sampler.md.
"""
import argparse
import json
import logging
from pathlib import Path

from wyckoff_transformer.system_prior import (
    DEFAULT_SG_KAPPA,
    SystemSpaceGroupPrior,
    prior_from_tensor_cache,
)

#: Where `build` puts its artifact when the caller does not say, next to the tensors
#: it was derived from.
DEFAULT_ARTIFACT_NAME = "system_prior.npz"


def default_artifact_path(dataset: str, cache_root: Path | None = None) -> Path:
    root = Path.cwd() / "cache" if cache_root is None else Path(cache_root)
    return root / dataset / DEFAULT_ARTIFACT_NAME


def _add_query_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--prior", type=Path, required=True,
                        help="The .npz written by `build`.")
    parser.add_argument("--required", "-r", type=str, default=None, metavar="Li-S",
                        help="Elements every structure must be asked to contain. "
                             "Omit for a palette with no floor.")
    parser.add_argument("--allowed", "-a", type=str, default=None, metavar="Li-S-P-O-...",
                        help="The palette. Defaults to every element the model knows. "
                             "Required elements must be in it.")
    parser.add_argument("--min-arity", type=int, default=None,
                        help="Refuse systems with fewer elements than this.")
    parser.add_argument("--max-arity", type=int, default=None,
                        help="Refuse systems with more elements than this. With a wide "
                             "palette the observed feasible set is mostly quaternary, so "
                             "this is how a ternary campaign stays ternary.")
    parser.add_argument("--system-temperature", type=float, default=1.0,
                        help="Flattens p(system). 1 is the empirical distribution; "
                             "larger discounts how much attention the corpus paid.")
    parser.add_argument("--sg-kappa", type=float, default=DEFAULT_SG_KAPPA,
                        help="Back-off pseudo-counts in p(space group | system).")


def _build(args: argparse.Namespace) -> None:
    max_values = {}
    for item in args.max or []:
        field, _, value = item.partition("=")
        if not _:
            raise SystemExit(f"--max takes FIELD=VALUE, got {item!r}")
        max_values[field] = float(value)
    prior = prior_from_tensor_cache(
        dataset=args.dataset,
        config_name=args.tokeniser_config,
        cache_root=args.cache_root,
        splits=tuple(args.splits),
        held_out_splits=tuple(args.held_out_splits),
        max_values=max_values or None,
    )
    output = args.output or default_artifact_path(args.dataset, args.cache_root)
    prior.save(output)
    print(f"{prior.metadata['n_systems']} systems, "
          f"{prior.metadata['n_rows']:.0f} rows, "
          f"{prior.n_space_groups} space groups, "
          f"{prior.n_elements} element tokens")
    if "held_out_novelty_rate" in prior.metadata:
        print(f"held-out novelty rate {prior.metadata['held_out_novelty_rate']:.4f} "
              f"over {prior.metadata['held_out_rows']} rows -- the default novel_fraction")
    print(f"written to {output} ({output.stat().st_size / 1e6:.1f} MB)")


def _show(args: argparse.Namespace) -> None:
    prior = SystemSpaceGroupPrior.load(args.prior)
    print(f"source: {prior.metadata.get('source')}  "
          f"systems: {prior.metadata.get('n_systems')}  "
          f"rows: {prior.metadata.get('n_rows', 0):.0f}  "
          f"default novel_fraction: {prior.metadata.get('held_out_novelty_rate', 0.0):.4f}")
    print(prior.describe_query(
        required=args.required, allowed=args.allowed, top=args.top,
        min_arity=args.min_arity, max_arity=args.max_arity,
        system_temperature=args.system_temperature, sg_kappa=args.sg_kappa))


def _sample(args: argparse.Namespace) -> None:
    prior = SystemSpaceGroupPrior.load(args.prior)
    draws = prior.sample(
        args.n_structures,
        required=args.required,
        allowed=args.allowed,
        novel_fraction=args.novel_fraction,
        system_temperature=args.system_temperature,
        sg_kappa=args.sg_kappa,
        sg_temperature=args.sg_temperature,
        min_arity=args.min_arity,
        max_arity=args.max_arity,
        rng=args.seed,
    )
    print(draws.summary(top=args.top))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "wt") as handle:
            json.dump(draws.manifest(), handle, indent=1)
        print(f"plan written to {args.output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--debug", action="store_true")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser(
        "build", help="Count (system, space group) over a tokenised dataset.")
    build.add_argument("dataset", type=str, help="A directory under cache/, e.g. lemat_bulk_fmax1")
    build.add_argument("--tokeniser-config", type=str, default=None,
                       help="Tokeniser config stem, when the dataset has more than one "
                            "cached. It has to be the one the model was trained with.")
    build.add_argument("--cache-root", type=Path, default=None)
    build.add_argument("--splits", nargs="+", default=["train"],
                       help="Splits to count. Training only by default: the prior is the "
                            "model's own input distribution.")
    build.add_argument("--held-out-splits", nargs="+", default=["val"],
                       help="Splits used only to measure the novelty rate.")
    build.add_argument("--max", action="append", metavar="FIELD=VALUE", default=None,
                       help="Keep only rows with FIELD <= VALUE, e.g. "
                            "--max energy_above_hull=0.1. Repeatable. Changes the prior "
                            "from where a palette puts atoms to where it puts them stably.")
    build.add_argument("--output", "-o", type=Path, default=None)
    build.set_defaults(func=_build)

    show = subparsers.add_parser(
        "show", help="What a query admits: systems, mass, space groups.")
    _add_query_arguments(show)
    show.add_argument("--top", type=int, default=15)
    show.set_defaults(func=_show)

    sample = subparsers.add_parser(
        "sample", help="Draw (system, space group) requests and write them as a plan.")
    _add_query_arguments(sample)
    sample.add_argument("-n", "--n-structures", type=int, required=True,
                        help="How many structures the batch will hold.")
    sample.add_argument("--novel-fraction", type=float, default=None,
                        help="Share of rows drawn from systems the data has not seen. "
                             "Defaults to the novelty rate measured when the prior was built.")
    sample.add_argument("--sg-temperature", type=float, default=1.0)
    sample.add_argument("--seed", type=int, default=None,
                        help="Seed, so a campaign can be repeated exactly.")
    sample.add_argument("--top", type=int, default=15)
    sample.add_argument("--output", "-o", type=Path, default=None,
                        help="Where to write the plan as JSON.")
    sample.set_defaults(func=_sample)

    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)
    args.func(args)


if __name__ == "__main__":
    main()
