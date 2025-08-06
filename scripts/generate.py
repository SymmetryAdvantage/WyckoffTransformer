import time
from pathlib import Path
import argparse
import torch
# torch.set_float32_matmul_precision('high')
import json
import gzip
import wandb
import logging
from omegaconf import OmegaConf

import sys
sys.path.append(str(Path(__file__).parent.parent.resolve()))
from wyckoff_transformer.trainer import WyckoffTrainer


def main():
    parser = argparse.ArgumentParser(description="Generate structures using a Wyckoff transformer.")
    parser.add_argument("output", type=Path, help="The output file.")
    model_source = parser.add_mutually_exclusive_group(required=True)
    model_source.add_argument("--wandb-run", type=str, help="The W&B run to use for the model.")
    model_source.add_argument("--model-path", type=Path,
        help="The path to the model directory. Should contain a best_model_params.pt, tokenizers.pkl.gz, "
             "token_engineers.pkl.gz, config.yaml")
    parser.add_argument("--use-cached-tensors", action="store_true",
        help="Generation process requires tensors of the original dataset. Mostly for technical reasons, "
             "but also for sampling the start tokens. If you are sure that the tokenization "
             "didn't change between training and generation, you can use this flag to speed up the process.")
    parser.add_argument("--initial-n-samples", type=int, help="The number of samples to try"
        " before filtering out the invalid ones.", default=1100)
    parser.add_argument("--firm-n-samples", type=int, help="The number of samples after generation, "
        "subsampling the valid ones if nesessary.", default=1000)
    parser.add_argument("--update-wandb", action="store_true", help="Update the W&B run with the "
        "generated structures and quality metrics.")
    parser.add_argument("--device", type=torch.device, default=torch.device("cpu"), help="The device to use.")
    parser.add_argument("--calibrate", action="store_true", help="Calibrate the generator.")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode.")
    parser.add_argument("--csx", action="store_true",
                        help="Enable Chemical System eXploration (CSX).")
    parser.add_argument("--required-elements", "--r", type=str,
                        help="Required elements for CSX mode (e.g., 'Li-S'). Must be provided if --csx is used.")
    parser.add_argument("--allowed-elements", "--a", type=str, default="all",
                        help="Allowed elements for CSX mode: 'all', 'fix', or a custom set (e.g., 'Li-S-P-O').")
    args = parser.parse_args()
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    if args.output.suffixes != [".json", ".gz"]:
        raise ValueError("Output file must be a .json.gz file.")
    if args.csx and not args.required_elements:
        parser.error("--required-elements is required when --csx is enabled.")
    if args.wandb_run:
        if args.update_wandb:
            wandb_run = wandb.init(project="WyckoffTransformer", id=args.wandb_run, resume=True)
        else:
            wandb_run = wandb.Api().run(f"WyckoffTransformer/{args.wandb_run}")
        config = OmegaConf.create(dict(wandb_run.config))
        run_path = Path(__file__).parent.parent / "runs" / args.wandb_run
    elif args.model_path:
        if args.update_wandb:
            raise ValueError("Cannot update W&B run when using a local model path.")
        run_path = args.model_path
        config = OmegaConf.load(run_path / "config.yaml")

    generation_start_time = time.time()
    trainer = WyckoffTrainer.from_config(
        config, device=args.device, use_cached_tensors=args.use_cached_tensors, run_path=run_path)
    trainer.model.load_state_dict(torch.load(trainer.run_path / "best_model_params.pt", weights_only=True))

    if args.csx:
        print("--- Running in Chemical System eXploration (CSX) mode ---")
        generated_wp = trainer.generate_csx_structures(
            n_structures=args.initial_n_samples,
            calibrate=args.calibrate,
            required_element_set=args.required_elements,
            allowed_element_set=args.allowed_elements
        )
    else:
        print("--- Running in Default Generation mode ---")
        generated_wp = trainer.generate_structures(args.initial_n_samples, args.calibrate)

    generation_end_time = time.time()
    print(f"Generation in total took {generation_end_time - generation_start_time} seconds")
    # print(f"Tensor generation took {tensor_generated_time - generation_start_time} seconds")
    # print(f"Detokenizing took {generation_end_time - tensor_generated_time} seconds")
    # wp_formal_validity = len(generated_wp) / generation_size
    # print(f"Wyckchoffs formal validity: {wp_formal_validity}")
    if args.firm_n_samples is not None:
        if len(generated_wp) >= args.firm_n_samples:
            generated_wp = generated_wp[:args.firm_n_samples]
        else:
            raise ValueError("Not enough valid structures to subsample.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(args.output, "wt") as f:
        json.dump(generated_wp, f)


if __name__ == "__main__":
    main()
