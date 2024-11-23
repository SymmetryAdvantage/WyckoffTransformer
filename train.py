from pathlib import Path
import argparse
import logging
from omegaconf import OmegaConf
import torch
import wandb
from wyckoff_transformer.trainer import train_from_config


def main():
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument("config", type=Path, help="The configuration file")
    parser.add_argument("dataset", type=str, default="mp_20_biternary", help="Dataset to use")
    parser.add_argument("device", type=torch.device, help="Device to train on")
    parser.add_argument("--pilot", action="store_true", help="Run a pilot run by setting epochs to 101")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    parser.add_argument("--run-path", type=Path, default=Path("runs"), help="Set the path for saving run data")
    parser.add_argument("--torch-num-thread", type=int, help="Number of threads for torch")
    args = parser.parse_args()

    if args.debug:
        torch.autograd.set_detect_anomaly(True)
        logging.basicConfig(level=logging.DEBUG)

    if args.torch_num_thread:
        torch.set_num_threads(args.torch_num_thread)

    if args.device.type == "cuda":
        # UserWarning: TensorFloat32 tensor cores for float32 matrix multiplication available but not enabled. Consider setting `torch.set_float32_matmul_precision('high')` for better performance.
        torch.set_float32_matmul_precision('high')
        
    config = OmegaConf.load(args.config)
    if args.pilot:
        print("Pilot run; overwriting epochs to 3")
        config['optimisation']['epochs'] = 3
        config['optimisation']['validation_period'] = 1
        tags = ["pilot"]
    else:
        tags = []
    config['name'] = args.config.stem
    config['dataset'] = args.dataset

    tokeniser_config_path = Path(__file__).parent.resolve() / "yamls" / "tokenisers" / f"{config.tokeniser.name}.yaml"
    tokeniser_config = OmegaConf.load(tokeniser_config_path)
    if len(tokeniser_config.get("augmented_token_fields", [])) > 1:
        raise ValueError("Only one augmented field is supported")
    config['tokeniser'] = tokeniser_config

    wandb_config = OmegaConf.to_container(config)

    with wandb.init(
        project="WyckoffTransformer",
        job_type="train",
        tags=tags,
        config=wandb_config):

        if args.debug:
            config["model"]['WyckoffTrainer_args']['compile_model'] = False
            with torch.autograd.detect_anomaly():
                train_from_config(config, args.device, run_path=args.run_path)
        else:
            train_from_config(config, args.device, run_path=args.run_path)


if __name__ == '__main__':
    main()
