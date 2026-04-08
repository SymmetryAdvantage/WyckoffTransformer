# Original: https://github.com/txie-93/cdvae/tree/main
# Modifications copyright 2026, Nikita Kazeev. Licensed under the Apache License, Version 2.0.
"""Property model loading and evaluation functions."""
from pathlib import Path
import os
import sys

import numpy as np
import torch
from hydra import initialize_config_dir, compose
import hydra
from torch_geometric.loader import DataLoader

from .pl_data.dataset import TensorCrystDataset
from .pl_data.datamodule import worker_init_fn


def get_model_path(eval_model_name):
    return Path(__file__).resolve().parent / 'prop_models' / eval_model_name


def load_config(model_path):
    with initialize_config_dir(str(model_path), version_base="1.1"):
        cfg = compose(config_name='hparams')
    return cfg


def load_model(model_path):
    # Hydra config requires a global variable
    os.environ["PROJECT_ROOT"] = str(Path(__file__).resolve().parents[4])
    # Older checkpoints may refer to modules under `cdvae.*` or the legacy
    # `wyckoff_transformer.cdvae_evals.*` path; map both to this package.
    import cdvae_property_models as _self_pkg
    sys.modules.setdefault("cdvae", _self_pkg)
    sys.modules.setdefault("wyckoff_transformer.cdvae_evals", _self_pkg)

    with initialize_config_dir(str(model_path), version_base="1.1"):
        cfg = compose(config_name='hparams')
        model_cls = hydra.utils.get_class(cfg.model._target_)
        ckpts = list(model_path.glob('*.ckpt'))
        if not ckpts:
            raise FileNotFoundError(f"No checkpoint files found in {model_path}")
        ckpt = None
        for ck in ckpts:
            if 'last' in ck.parts[-1]:
                ckpt = str(ck)
        if ckpt is None:
            ckpt_epochs = np.array(
                [int(ck.parts[-1].split('-')[0].split('=')[1])
                 for ck in ckpts if 'last' not in ck.parts[-1]])
            ckpt = str(ckpts[ckpt_epochs.argsort()[-1]])
        model = model_cls.load_from_checkpoint(
            ckpt,
            strict=True,
            weights_only=False,
            encoder=cfg.model.encoder)

        model.lattice_scaler = torch.load(
            model_path / 'lattice_scaler.pt', weights_only=False)
        model.scaler = torch.load(
            model_path / 'prop_scaler.pt', weights_only=False)

    return model


def prop_model_eval(
        eval_model_name,
        crystal_array_list,
        device: torch.device = torch.device("cpu")):

    model_path = get_model_path(eval_model_name)
    model = load_model(model_path)
    model.to(device)
    cfg = load_config(model_path)

    dataset = TensorCrystDataset(
        crystal_array_list, cfg.data.niggli, cfg.data.primitive,
        cfg.data.graph_method, cfg.data.preprocess_workers,
        cfg.data.lattice_scale_method)

    dataset.scaler = model.scaler.copy()

    loader = DataLoader(
        dataset,
        shuffle=False,
        batch_size=256,
        num_workers=0,
        worker_init_fn=worker_init_fn)

    model.eval()
    all_preds = []

    for batch in loader:
        preds = model(batch)
        model.scaler.match_device(preds)
        scaled_preds = model.scaler.inverse_transform(preds)
        all_preds.append(scaled_preds.detach().cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0).squeeze(1)
    return all_preds.tolist()
