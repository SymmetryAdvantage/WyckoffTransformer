import argparse
import omegaconf
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
    tokenizer_source = parser.add_mutually_exclusive_group(required=True)
    tokenizer_source.add_argument("--tokenizer-path", type=Path, help="Load a pickled tokenizer")
    tokenizer_source.add_argument("--new-tokenizer", action="store_true",
        help="Generate a new tokenizer, potentially overwriting files")
    args = parser.parse_args()
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
    tensors, tokenisers, token_engineers = tokenise_dataset(datasets_pd, config, args.use_existing_tokenizers)
    if args.debug and "multiplicity" in token_engineers:
        index = 0
        multiplicities_from_tokens = token_engineers["multiplicity"].get_feature_from_token_batch(
            tensors["val"]["spacegroup_number"].tolist(),
            [tensors["val"]["site_symmetries"][:, index].tolist(), tensors["val"]["sites_enumeration"][:, index].tolist()])
        assert (multiplicities_from_tokens == datasets_pd["val"]["multiplicity"].map(itemgetter(index))).all()
        logger.debug("Multiplicities from tokens match the original dataset")
    tokeniser_name = args.config_file.stem
    cache_tensors_path = cache_path / 'tensors'
    cache_tensors_path.mkdir(parents=True, exist_ok=True)
    with gzip.open(cache_tensors_path / f'{tokeniser_name}.pkl.gz', "wb") as f:
        pickle.dump(tensors, f)
    # In the future we might want to save the tokenisers in json, so that they can be distributed
    cache_tokenisers_path = cache_path / 'tokenisers'
    cache_tokenisers_path.mkdir(parents=True, exist_ok=True)
    with gzip.open(cache_tokenisers_path / f'{tokeniser_name}.pkl.gz', "wb") as f:
        pickle.dump(tokenisers, f)
        pickle.dump(token_engineers, f)


if __name__ == '__main__':
    main()
