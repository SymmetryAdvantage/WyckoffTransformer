import argparse
import omegaconf
import torch
from pathlib import Path
from operator import itemgetter
import pickle
import gzip
import logging

from wyckoff_transformer.tokenization import tokenise_dataset

logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser("Retokenise a cached dataset")
    parser.add_argument("dataset", type=str, help="The name of the dataset to retokenise")
    parser.add_argument("config_file", type=Path, help="The tokeniser configuration file")
    parser.add_argument("--debug", action="store_true", help="Set the logging level to debug")
    parser.add_argument('--pilot', action='store_true', help="Run a pilot experiment")
    parser.add_argument("--n-jobs", type=int, help="Number of jobs to use")
    tokenizer_source = parser.add_mutually_exclusive_group(required=True)
    tokenizer_source.add_argument("--tokenizer-path", type=Path, help="Load a pickled tokenizer")
    tokenizer_source.add_argument("--new-tokenizer", action="store_true",
        help="Generate a new tokenizer, potentially overwriting files")
    args = parser.parse_args()
    if args.n_jobs is not None:
        raise NotImplementedError("n_jobs is not implemented yet"
            "Pandarallel will consume what it likes")
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    config = omegaconf.OmegaConf.load(args.config_file)
    cache_path = Path(__file__).parent.resolve() / "cache" / args.dataset
    cache_path.mkdir(parents=True, exist_ok=True)
    with gzip.open(cache_path / 'data.pkl.gz', "rb") as f:
        datasets_pd = pickle.load(f)
    print("Loaded the dataset. It has the following sizes:")
    for name, dataset in datasets_pd.items():
        print(f"{name}: {len(dataset)}")
    if args.pilot:
        datasets_pd = {name: dataset.sample(100) for name, dataset in datasets_pd.items()}
        print("Piloting with 100 samples")
    tensors, tokenisers, token_engineers = tokenise_dataset(
        datasets_pd, config, args.tokenizer_path, n_jobs=args.n_jobs)
    if args.debug and "multiplicity" in token_engineers:
        index = 0
        multiplicities_from_tokens = token_engineers["multiplicity"].get_feature_from_token_batch(
            tensors["val"]["spacegroup_number"].tolist(),
            [tensors["val"]["site_symmetries"][:, index].tolist(), tensors["val"]["sites_enumeration"][:, index].tolist()])
        assert (multiplicities_from_tokens == datasets_pd["val"]["multiplicity"].map(itemgetter(index))).all()
        logger.debug("Multiplicities from tokens match the original dataset")
    tokenizer_root_path = Path(__file__).parent.resolve() / "yamls" / "tokenisers"
    tokenizer_full_name = args.config_file.resolve().relative_to(tokenizer_root_path).with_suffix('')
    cache_tensors_path = cache_path / 'tensors' / tokenizer_full_name.with_suffix('.pt')
    cache_tensors_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tensors, cache_tensors_path)
    # In the future we might want to save the tokenisers in json, so that they can be distributed
    cache_tokenisers_path = cache_path / 'tokenisers' / tokenizer_full_name.with_suffix('.pkl.gz')
    cache_tokenisers_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(cache_tokenisers_path, "wb") as f:
        pickle.dump(tokenisers, f)
        pickle.dump(token_engineers, f)


if __name__ == '__main__':
    main()
