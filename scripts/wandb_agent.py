from functools import partial
import argparse
import wandb
from pathlib import Path
from omegaconf import OmegaConf
import torch
from wyckoff_transformer.trainer import train_from_config


def agent_function(device, run_path):
    wandb.init()
    base_config_name = wandb.config.base_config
    base_config = OmegaConf.load(Path(__file__).parent.parent.resolve() / "yamls" / "models" / f"{base_config_name}.yaml")
    final_config = OmegaConf.merge(base_config, dict(wandb.config))
    tokeniser_config_path = Path(__file__).parent.parent.resolve() / "yamls" / "tokenisers" / f"{final_config.tokeniser.name}.yaml"
    tokeniser_config = OmegaConf.load(tokeniser_config_path)
    final_config['tokeniser'] = tokeniser_config
    train_from_config(final_config, device, run_path=run_path)


def main():
    parser = argparse.ArgumentParser(description='Agent for WanDB sweep')
    parser.add_argument("sweep_id", type=str, help="The WanDB project name")
    parser.add_argument("device", type=torch.device, help="Device to train on")
    parser.add_argument("--project", type=str, default="WyckoffTransformer", help="The WanDB project name")
    parser.add_argument("--count", type=int, default=2, help="The number of sweep config trials to try")
    parser.add_argument("--run-path", type=Path, default=Path("runs"), help="Set the path for saving run data")
    parser.add_argument("--torch-num-thread", type=int, default=19, help="Number of threads for torch")
    args = parser.parse_args()
    if args.device.type == "cuda":
        # UserWarning: TensorFloat32 tensor cores for float32 matrix multiplication available but not enabled. Consider setting `torch.set_float32_matmul_precision('high')` for better performance.
        torch.set_float32_matmul_precision('high')

    torch.set_num_threads(args.torch_num_thread)
    wandb.agent(args.sweep_id, function=partial(agent_function, device=args.device, run_path=args.run_path),
                project=args.project, count=args.count)


if __name__ == '__main__':
    main()
