from typing import Tuple, Dict, Optional, List
import importlib
from random import randint
import logging
from functools import partial
from pathlib import Path
import gzip
import json
import pickle
import numpy as np
import torch
import math
from torch import nn
from torch import Tensor
from omegaconf import OmegaConf, DictConfig
from tqdm import trange
import wandb
from sklearn.metrics import roc_auc_score 

from cascade_transformer.dataset import AugmentedCascadeDataset, TargetClass, jagged_batch_randperm
from cascade_transformer.model import CascadeTransformer
from wyckoff_transformer.tokenization import (
    load_tensors_and_tokenisers, tensor_to_pyxtal,
    get_letter_from_ss_enum_idx, get_wp_index)
from wyckoff_transformer.generator import WyckoffGenerator
from wyckoff_transformer.evaluation import (
    evaluate_and_log, StatisticalEvaluator, smac_validity_from_counter)


logger = logging.getLogger(__file__)
preprocessed_wyckhoffs_cache_path = Path(__file__).parent.parent.resolve() / "cache" / "wychoffs_enumerated_by_ss.pkl.gz"

class WyckoffTrainer():
    def __init__(
        self,
        model: nn.Module,
        train_dataset: Dict[str, torch.tensor],
        val_dataset: Dict[str, torch.tensor],
        tokenisers: dict,
        token_engineers: dict,
        cascade_order: Tuple[str],
        cascade_is_target: Dict[str, bool],
        augmented_fields: List[str]|None,
        start_name: str,
        start_dtype: torch.dtype,
        target: TargetClass|str,
        evaluation_samples: int,
        multiclass_next_token_with_order_permutation: bool,
        optimisation_config: dict,
        device: torch.DeviceObjType,
        augmented_storage_device: Optional[torch.DeviceObjType] = None,
        batch_size: Optional[int] = None,
        train_batch_size: Optional[int] = None,
        val_batch_size: Optional[int] = None,
        test_batch_size: Optional[int] = None,
        run_path: Optional[Path] = None,
        target_name = None,
        kl_samples: Optional[int] = None,
        weights_path: Optional[Path] = None,
        test_dataset: Optional[Dict[str, torch.tensor]] = None,
        compile_model: bool = False,
    ):
        """
        Args:
            multiclass_next_token_with_order_permutation: Train a permutation-invariant model by permuting the sequences,
                    If the target is the next token, predict the 0-th cascade field as a multiclass target.
                    You might also want to ensure that the model is permutation-invariant; pay attention to the positional encoding.
            target: the target to predict. One of:
                next_token: predict the next token in the cascade&sequence.
                num_unique_tokens: predict the number of unique tokens in the cascade&sequence encountered so far.
                    Intended for debugging the ability of the model to count.
        """
        if isinstance(target, str):
            target = TargetClass[target]
        if target not in (TargetClass.Scalar, TargetClass.Classification):
            is_target_in_order = [cascade_is_target[field] for field in cascade_order]
            if not all(is_target_in_order[:-1]):
                raise NotImplementedError("Only one not targret field is supported "
                    "at the moment and it must be the last")
            self.cascade_target_count = sum(is_target_in_order)
        else:
            self.cascade_target_count = 0
        self.kl_samples = kl_samples
        self.token_engineers = token_engineers
        self.cascade_is_target = cascade_is_target
        # Nothing else will work in foreseeable future
        self.dtype = torch.int64
        if batch_size is not None:
            logger.warning("batch_size is deprecated, use train_batch_size, val_batch_size, test_batch_size")
            if train_batch_size is None:
                train_batch_size = batch_size
            elif train_batch_size != batch_size:
                raise ValueError("batch_size and train_batch_size differ")

        self.run_path = run_path

        if target == TargetClass.NextToken:
            # Sequences have difference lengths, so we need to make sure that
            # long sequences don't dominate the loss, so we don't average the loss
            self.criterion = nn.CrossEntropyLoss(reduction="sum")
        elif target == TargetClass.NumUniqueTokens:
            if not multiclass_next_token_with_order_permutation:
                raise NotImplementedError("NumUniqueTokens is not implemented without permutations")
            self.criterion = nn.MSELoss(reduction="none")
        elif target == TargetClass.Scalar:
            # Assumes the batch size is the same for all batches
            self.criterion = nn.MSELoss(reduction='mean')
            self.testing_criterion = nn.L1Loss(reduction='mean')
        elif target == TargetClass.Classification:
            # Classification loss
            self.criterion = nn.CrossEntropyLoss(reduction='mean')
            self.testing_criterion = nn.CrossEntropyLoss(reduction='mean')
        else:
            raise ValueError(f"Unknown target: {target}")
        
        self.model = model
        # Transformer doesn't support fullgraph=True
        self.compiled_model = torch.compile(self.model, fullgraph=False)
        if compile_model:
            self.model = self.compiled_model
        self.tokenisers = tokenisers
        self.device = device
        self.augmented_fields = augmented_fields

        self.masks_dict = {field: tokenisers[field].mask_token for field in cascade_order}
        self.pad_dict = {field: tokenisers[field].pad_token for field in cascade_order}
        self.stops_dict = {field: tokenisers[field].stop_token for field in cascade_order}
        self.num_classes_dict = {field: len(tokenisers[field]) for field in cascade_order}
        self.start_name = start_name

        self.train_dataset = AugmentedCascadeDataset(
            data=train_dataset,
            cascade_order=cascade_order,
            masks=self.masks_dict,
            pads=self.pad_dict,
            stops=self.stops_dict,
            num_classes=self.num_classes_dict,
            start_field=start_name,
            augmented_fields=augmented_fields,
            batch_size=train_batch_size,
            dtype=self.dtype,
            start_dtype=start_dtype,
            device=self.device,
            augmented_storage_device=augmented_storage_device,
            target_name=target_name)
        
        if "lr_per_sqrt_n_samples" in optimisation_config.optimiser:
            if "config" in optimisation_config.optimiser and "lr" in optimisation_config.optimiser.config:
                raise ValueError("Cannot specify both lr and lr_per_sqrt_n_samples")
            if train_batch_size is None:
                samples_per_step = len(self.train_dataset)
            else:
                samples_per_step = train_batch_size
            optimisation_config.optimiser.update(
                {"config": {"lr": optimisation_config.optimiser.lr_per_sqrt_n_samples * samples_per_step**0.5}})
        optimizer_module_obj = importlib.import_module(optimisation_config.optimiser.get("module", "torch.optim"))
        self.optimizer = getattr(optimizer_module_obj, optimisation_config.optimiser.name)(
            model.parameters(), **optimisation_config.optimiser.config)
        if "scheduler" in optimisation_config:
            self.scheduler = getattr(torch.optim.lr_scheduler, optimisation_config.scheduler.name)(
                self.optimizer, 'min', **optimisation_config.scheduler.config)
        else:
            self.scheduler = None

        self.val_dataset = AugmentedCascadeDataset(
            data=val_dataset,
            cascade_order=cascade_order,
            masks=self.masks_dict,
            pads=self.pad_dict,
            stops=self.stops_dict,
            num_classes=self.num_classes_dict,
            start_field=start_name,
            augmented_fields=augmented_fields,
            batch_size=val_batch_size,
            dtype=self.dtype,
            start_dtype=start_dtype,
            device=device,
            augmented_storage_device=augmented_storage_device,
            target_name=target_name
            )

        if test_dataset is None:
            self.test_dataset = None
        else:
            self.test_dataset = AugmentedCascadeDataset(
                data=test_dataset,
                cascade_order=cascade_order,
                masks=self.masks_dict,
                pads=self.pad_dict,
                stops=self.stops_dict,
                num_classes=self.num_classes_dict,
                start_field=start_name,
                augmented_fields=augmented_fields,
                batch_size=test_batch_size,
                dtype=self.dtype,
                start_dtype=start_dtype,
                device=device,
                augmented_storage_device=augmented_storage_device,
                target_name=target_name
            )

        assert self.train_dataset.max_sequence_length == self.val_dataset.max_sequence_length
    
        self.clip_grad_norm = optimisation_config.clip_grad_norm
        self.cascade_len = len(cascade_order)
        self.cascade_order = cascade_order
        self.epochs = optimisation_config.epochs
        if weights_path is not None:
            self.model.load_state_dict(torch.load(weights_path))

        self.validation_period = optimisation_config.validation_period
        self.early_stopping_patience_epochs = optimisation_config.early_stopping_patience_epochs
        self.target = target
        self.multiclass_next_token_with_order_permutation = multiclass_next_token_with_order_permutation
        self.evaluation_samples = evaluation_samples


    @classmethod
    def from_config(cls, config_dict: dict|DictConfig,
                    device: torch.device,
                    use_cached_tensors: bool = True,
                    run_path: Optional[Path] = Path("runs")):
        config = OmegaConf.create(config_dict)
        if config.model.WyckoffTrainer_args.get("multiclass_next_token_with_order_permutation", False) and \
            not config.model.CascadeTransformer_args.learned_positional_encoding_only_masked:

            raise ValueError("Multiclass target with order permutation requires learned positional encoding only masked, ",
                            "otherwise the Transformer is not permutation invariant.")
        tensors, tokenisers, token_engineers = load_tensors_and_tokenisers(
            config.dataset, config.tokeniser.name, use_cached_tensors=use_cached_tensors,
            tokenizer_path=run_path / "tokenizers.pkl.gz" if not use_cached_tensors else None)
        model = CascadeTransformer.from_config_and_tokenisers(config, tokenisers, device)
        # Our hihgly dynamic concat-heavy workflow doesn't benefit much from compilation
        # torch._dynamo.config.cache_size_limit = 128
        # model = torch.compile(model, dynamic=True)
        if config.model.CascadeTransformer_args.start_type == "categorial":
            start_dtype = torch.int64
        # one-hots are encoded by a linear layer
        elif config.model.CascadeTransformer_args.start_type == "one_hot":
            start_dtype = torch.float32
        else:
            raise ValueError(f"Unknown start type: {config.model.CascadeTransformer_args.start_type}")
        return cls(
            model, tensors["train"], tensors["val"], tokenisers, token_engineers, config.model.cascade.order,
            config.model.cascade.get("is_target", None),
            config.model.cascade.get("augmented", None),
            config.model.start_token,
            optimisation_config=config.optimisation, device=device,
            run_path=run_path,
            start_dtype=start_dtype,
            test_dataset=tensors["test"] if "test" in tensors else None,
            **config.model.WyckoffTrainer_args)


    # Compilation fails due to the use of jagged_batch_randperm
    # @torch.compile(fullgraph=False)
    def get_permutation_kl_compiled(
        self,
        num_samples: int,
        max_seq_len: int):

        start_perm = torch.randperm(self.train_dataset.start_tokens.size(0))
        start = self.train_dataset.start_tokens[start_perm[:num_samples]]
        generated = []
        for field in self.cascade_order:
            generated.append(torch.empty((start.size(0), 0),
                                        dtype=torch.int64, device=self.device, requires_grad=False))
        pure_sequence_lengths = torch.zeros(start.size(0), device=self.device, requires_grad=False, dtype=torch.int64)
        reference_log_proba = torch.zeros(start.size(0), device=self.device)
        is_the_end = torch.zeros(start.size(0), device=self.device, dtype=torch.bool)
        for known_seq_len in range(max_seq_len):
            for known_cascade_len, cascade_name in enumerate(self.cascade_order):
                if self.cascade_is_target[cascade_name]:
                    this_generation_input = []
                    for cascade_idx, cascade_name in enumerate(self.cascade_order):
                        if cascade_idx < known_cascade_len:
                            this_generation_input.append(generated[cascade_idx][:, :known_seq_len + 1])
                        else:
                            this_generation_input.append(torch.cat(
                                (generated[cascade_idx][:, :known_seq_len],
                                 torch.tensor(self.masks_dict[cascade_name], device=self.device).unsqueeze(0).expand(start.size(0), 1)), dim=1))
                    # We don't require padding as the sequences containing pads won't be used
                    logits = self.compiled_model(start, this_generation_input, None, known_cascade_len)
                    log_probas = torch.nn.functional.log_softmax(logits, dim=1)
                    with torch.no_grad():
                        generated_tokens = torch.multinomial(torch.exp(log_probas), num_samples=1).squeeze()
                        is_the_end = is_the_end | (
                            (generated_tokens == self.stops_dict[cascade_name]) |
                            (generated_tokens == self.masks_dict[cascade_name]) |
                            (generated_tokens == self.pad_dict[cascade_name]))
                        # pure doesn't include the stop token
                        # or any other abominations that might be generated
                        generated[known_cascade_len] = torch.cat((
                            generated[known_cascade_len], generated_tokens.unsqueeze(1)), dim=1)
                    reference_log_proba[~is_the_end] += torch.gather(
                                log_probas[~is_the_end], 1, generated_tokens[~is_the_end].unsqueeze(1)).squeeze()
                else:
                    raise NotImplementedError("Only cascades with all fields as targets are supported")
            pure_sequence_lengths[~is_the_end] = known_seq_len
        generated_data = dict(zip(self.cascade_order, generated))
        generated_data[self.start_name] = start
        generated_dataset = AugmentedCascadeDataset(
            data=generated_data,
            cascade_order=self.cascade_order,
            masks=self.masks_dict,
            pads=self.pad_dict,
            stops=self.stops_dict,
            num_classes=self.num_classes_dict,
            start_field=self.start_name,
            augmented_field=None,
            dtype=torch.long,
            start_dtype=torch.float,
            device=self.device)
        full_permutation = jagged_batch_randperm(pure_sequence_lengths, max_seq_len)
        permuted_log_likelihoods = torch.zeros_like(reference_log_proba)
        for known_seq_len in range(max_seq_len):
            for known_cascade_len, cascade_name in enumerate(self.cascade_order):
                start, this_data, target = generated_dataset.get_masked_multiclass_cascade_data(
                                known_seq_len, known_cascade_len, TargetClass.NextToken, multiclass_target=False,
                                augmented_data=None, full_permutation=full_permutation,
                                apply_permutation=False, truncate_invalid_targets=False)
                logits = self.compiled_model(start, this_data, None, known_cascade_len)
                log_probas = torch.nn.functional.log_softmax(logits, dim=1)
                is_the_end = (
                            (target == self.stops_dict[cascade_name]) |
                            (target == self.masks_dict[cascade_name]) |
                            (target == self.pad_dict[cascade_name]))
                permuted_log_likelihoods[~is_the_end] += torch.gather(
                    log_probas[~is_the_end], 1, target[~is_the_end].unsqueeze(1)).squeeze()
        # sum so it's the same order of magnitude as the ordinary loss
        return (reference_log_proba - permuted_log_likelihoods).sum()


    def get_permutation_kl(
        self,
        num_samples: int,
        max_seq_len: int):

        start_perm = torch.randperm(self.train_dataset.start_tokens.size(0))
        start = self.train_dataset.start_tokens[start_perm[:num_samples]]
        generated = []
        for field in self.cascade_order:
            generated.append(torch.empty((start.size(0), 0),
                                        dtype=torch.int64, device=self.device, requires_grad=False))
        pure_sequence_lengths = torch.zeros(start.size(0), device=self.device, requires_grad=False, dtype=torch.int64)
        reference_log_proba = torch.zeros(start.size(0), device=self.device)
        for known_seq_len in range(max_seq_len):
            for known_cascade_len, cascade_name in enumerate(self.cascade_order):
                if self.cascade_is_target[cascade_name]:
                    this_generation_input = []
                    for cascade_idx, cascade_name in enumerate(self.cascade_order):
                        if cascade_idx < known_cascade_len:
                            this_generation_input.append(generated[cascade_idx][:, :known_seq_len + 1])
                        else:
                            this_generation_input.append(torch.cat(
                                (generated[cascade_idx][:, :known_seq_len],
                                 torch.tensor(self.masks_dict[cascade_name], device=self.device).unsqueeze(0).expand(start.size(0), 1)), dim=1))
                    logits = self.model(start, this_generation_input, None, known_cascade_len)
                    log_probas = torch.nn.functional.log_softmax(logits, dim=1)
                    with torch.no_grad():
                        generated_tokens = torch.multinomial(torch.exp(log_probas), num_samples=1).squeeze()
                        is_the_end = (
                            (generated_tokens == self.stops_dict[cascade_name]) |
                            (generated_tokens == self.masks_dict[cascade_name]) |
                            (generated_tokens == self.pad_dict[cascade_name]))
                        # pure doesn't include the stop token
                        # or any other abominations that might be generated
                        pure_sequence_lengths[is_the_end] = known_seq_len
                        generated[known_cascade_len] = torch.cat((
                            generated[known_cascade_len], generated_tokens.unsqueeze(1)), dim=1)
                    reference_log_proba += torch.gather(
                                log_probas, 1, generated_tokens.unsqueeze(1)).squeeze()
                else:
                    raise NotImplementedError("Only cascades with all fields as targets are supported")
        generated_data = dict(zip(self.cascade_order, generated))
        generated_data[self.start_name] = start
        generated_dataset = AugmentedCascadeDataset(
            data=generated_data,
            cascade_order=self.cascade_order,
            masks=self.masks_dict,
            pads=self.pad_dict,
            stops=self.stops_dict,
            num_classes=self.num_classes_dict,
            start_field=self.start_name,
            augmented_fields=None,
            dtype=torch.long,
            start_dtype=torch.float,
            device=self.device)
        full_permutation = jagged_batch_randperm(pure_sequence_lengths, max_seq_len)
        permuted_log_likelihoods = torch.zeros_like(reference_log_proba)
        for known_seq_len in range(max_seq_len):
            for known_cascade_len, cascade_name in enumerate(self.cascade_order):
                start, this_data, target, batch_target_is_viable = generated_dataset.get_masked_multiclass_cascade_data(
                                known_seq_len, known_cascade_len, TargetClass.NextToken, multiclass_target=False,
                                full_permutation=full_permutation,
                                apply_permutation=False, return_chosen_indices=True)
                logits = self.model(start, this_data, None, known_cascade_len)
                log_probas = torch.nn.functional.log_softmax(logits, dim=1)
                permuted_log_likelihoods[batch_target_is_viable] += torch.gather(
                    log_probas, 1, target.unsqueeze(1)).squeeze()
        return (reference_log_proba - permuted_log_likelihoods).mean()


    def get_loss(
        self,
        dataset: AugmentedCascadeDataset,
        known_seq_len: int,
        known_cascade_len: int|None,
        no_batch: bool = False,
        testing: bool = False) -> Tensor:
        """
        Computes loss on the dataset. Advances the dataset to the next batch.
        """
        logging.debug("Known sequence length: %i", known_seq_len)
        logging.debug("Known cascade length: %s", str(known_cascade_len))
        # Step 1: Get the data
        if self.multiclass_next_token_with_order_permutation:
            if self.target == TargetClass.NextToken:
                # Once we have sampled the first cascade field, the prediction target is no longer mutliclass
                # However, we still need to permute the sequence so that the autoregression is
                # permutation-invariant.
                start_tokens, masked_data, target = dataset.get_masked_multiclass_cascade_data(
                    known_seq_len, known_cascade_len, multiclass_target=(known_cascade_len == 0),
                    target_type=self.target, no_batch=no_batch)
            elif self.target == TargetClass.NumUniqueTokens:
                start_tokens, masked_data, target = dataset.get_masked_multiclass_cascade_data(
                    known_seq_len, known_cascade_len, multiclass_target=False, target_type=self.target,
                    no_batch=no_batch)
                logging.debug("Target: %s", target)
                # Counts are integers, as they should be, but MSE needs a float
                target = target.float()
            else:
                raise ValueError(f"Target {self.target} is not supported by "
                                  "multiclass_next_token_with_order_permutation")
        else:
            if self.target in (TargetClass.Scalar, TargetClass.Classification):
                start_tokens, masked_data, target, padding_mask = dataset.get_augmented_data(no_batch=no_batch)
            else:
                start_tokens, masked_data, target = dataset.get_masked_cascade_data(known_seq_len, known_cascade_len)
        # Step 2: Get the prediction
        if self.target == TargetClass.NextToken:
            # No padding, as we have already discarded the padding
            prediction = self.model(start_tokens, masked_data, None, known_cascade_len)
        elif self.target == TargetClass.NumUniqueTokens:
            # No padding, as we have already discarded the padding
            prediction = self.model(start_tokens, masked_data, None, None)
        elif self.target == TargetClass.Scalar:
            logger.debug("Start tokens size: %s", start_tokens.size())
            #logger.debug("Start tokens isnan: %s", start_tokens.isnan().any())
            #logger.debug("Masked data isnan: %s", any((a.isnan().any() for a in masked_data)))
            #logger.debug("Padding mask isnan: %s", padding_mask.isnan().any())
            prediction = self.model(start_tokens, masked_data, padding_mask, None).squeeze()
            #logger.debug("Prediction isnan: %s", prediction.isnan().any())
        elif self.target == TargetClass.Classification:
            logger.debug("Start tokens size: %s", start_tokens.size())
            # Sequence-level classification: expect logits of shape [batch_size, num_classes]
            prediction = self.model(start_tokens, masked_data, padding_mask, None)
            if prediction.dim() == 1:
                prediction = prediction.unsqueeze(1)
            assert prediction.dim() == 2, f"Classification logits must be [N, C], got {tuple(prediction.shape)}"
            assert prediction.size(1) >= 2, (
                "Binary classification with CrossEntropyLoss requires 2 logits ([N,2]). "
                f"Got C={prediction.size(1)}."
            )
            if target.dim() > 1:
                target = target.squeeze(-1)
            target = target.long()
        else:
            raise ValueError(f"Unknown target: {self.target}")
        # Step 3: Calculate the loss
        # logger.debug("Target isnan: %s", target.isnan().any())
        logger.debug("Target min: %s, max: %s", target.min(), target.max())
        logger.debug("Prediction shape: %s", prediction.shape)
        if self.target == TargetClass.Classification:
            # CrossEntropyLoss expects class indices as int64
            target = target.long()
        if testing:
            return self.testing_criterion(prediction, target)
        return self.criterion(prediction, target)


    def train_epoch(self):
        self.model.train()
        if hasattr(self.optimizer, "train"):
            self.optimizer.train()
        for _ in trange(self.train_dataset.batches_per_epoch, leave=False):
            self.optimizer.zero_grad(set_to_none=True)
            if self.target == TargetClass.NextToken:
                known_cascade_len = randint(0, self.cascade_target_count - 1)
                known_seq_len = randint(0, self.train_dataset.max_sequence_length - 1)
            elif self.target == TargetClass.NumUniqueTokens:
                known_cascade_len = 0
                known_seq_len = randint(0, self.train_dataset.max_sequence_length - 1)
            elif self.target in (TargetClass.Scalar, TargetClass.Classification):
                # Use full sequences
                known_cascade_len = None
                known_seq_len = self.train_dataset.max_sequence_length - 1
                
            else:
                raise ValueError(f"Unknown target: {self.target}")
            loss = self.get_loss(self.train_dataset, known_seq_len, known_cascade_len)
            if self.kl_samples is not None:
                kl_loss = self.get_permutation_kl_compiled(self.kl_samples, self.train_dataset.max_sequence_length)
                loss += kl_loss
                wandb.log({"kl_loss": kl_loss}, commit=False)
            if self.target == TargetClass.NumUniqueTokens:
                # Predictions are [batch_size, cascade_size]
                # Unreduced MSE is [batch_size, cascade_size]
                # We avoid averaging them at the level of self.criterion, so we can log
                # the loss for each cascade field separately.
                loss = loss.mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
            self.optimizer.step()
            wandb.log({"loss.batch.train": loss,
                       "known_seq_len": known_seq_len,
                       "known_cascade_len": known_cascade_len})


    @torch.no_grad()
    def evaluate(self, dataset: AugmentedCascadeDataset) -> Tensor:
        """
        Evaluates the model by calculating the average loss on the dataset.
        Args:
            dataset: The dataset to evaluate on.
        Returns:
            The average loss on the dataset.
        """
        self.model.eval()
        if hasattr(self.optimizer, "eval"):
            self.optimizer.eval()
        if dataset.batch_size is not None and not dataset.fix_batch_size:
            raise NotImplementedError("Only fixed batch size is supported")
        if self.target in (TargetClass.Scalar, TargetClass.Classification):
            loss = torch.zeros(1, device=self.device)
            # Augmentation
            for sample_ids in range(self.evaluation_samples):
                for batch_idx in range(dataset.batches_per_epoch):
                    loss += self.get_loss(dataset, None, None, False, True)
            # Above we check that the batch size is the same for all batches
            return loss / self.evaluation_samples / dataset.batches_per_epoch

        loss = torch.zeros(self.cascade_len, device=self.device)
        for _ in range(self.evaluation_samples):
            for known_seq_len in range(dataset.max_sequence_length):
                if self.target == TargetClass.NextToken:
                    for known_cascade_len in range(self.cascade_target_count):
                        loss[known_cascade_len] += self.get_loss(
                            dataset, known_seq_len, known_cascade_len, True)
                else: # NumUniqueTokens
                    loss += self.get_loss(dataset, known_seq_len, 0, True).sum(dim=0)
            # ln(P) = ln p(t_n|t_n-1, ..., t_1) + ... + ln p(t_2|t_1)
            # We are minimising the negative log likelihood of the whole sequences
        return loss / self.evaluation_samples / len(dataset)

    def evaluate_classification_metrics(self, dataset: AugmentedCascadeDataset) -> dict:
        """
        Compute Accuracy, Precision, Recall, F1, and AUC for a classification dataset.
        Returns a dict with keys: accuracy, precision, recall, f1, auc.
        - For binary classification (C=2): positive class is index 1.
        - For multiclass (C>2): macro-averaged precision/recall/F1; AUC is OVO macro if sklearn is available.
        """
        self.model.eval()
        if hasattr(self.optimizer, "eval"):
            self.optimizer.eval()

        with torch.no_grad():
            logits_list = []
            targets_list = []
            for _ in range(dataset.batches_per_epoch):
                start_tokens, masked_data, target, padding_mask = dataset.get_augmented_data(no_batch=False)
                logits = self.model(start_tokens, masked_data, padding_mask, None)
                if logits.dim() == 1:
                    logits = logits.unsqueeze(1)
                logits_list.append(logits.detach().cpu())
                targets_list.append(target.detach().long().view(-1).cpu())
            logits = torch.cat(logits_list, dim=0)
            targets = torch.cat(targets_list, dim=0)
            num_classes = logits.size(1)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
                    # Accuracy
            accuracy = (preds == targets).float().mean().item()

            eps = 1e-12
            if num_classes == 2:
                # Binary metrics (positive=1)
                pos = 1
                tp = ((preds == pos) & (targets == pos)).sum().item()
                fp = ((preds == pos) & (targets != pos)).sum().item()
                fn = ((preds != pos) & (targets == pos)).sum().item()
                precision = tp / (tp + fp + eps)
                recall = tp / (tp + fn + eps)
                f1 = 2.0 * precision * recall / (precision + recall + eps)
                # AUC (if sklearn available)
                try:
                    auc = float(roc_auc_score(targets.numpy(), probs[:, 1].numpy()))
                except Exception:
                    auc = float('nan')
            else:
                raise ValueError(f"Classification metrics currently only support binary classification, but got num_classes={num_classes}")

            return {
                "accuracy": float(accuracy),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "auc": float(auc),
            }
    
    def train(self):
        best_val_loss = float('inf')
        best_val_epoch = 0

        run_path = Path(__file__).parent.parent / "runs"

        self.run_path = run_path / wandb.run.id
        self.run_path.mkdir(parents=True, exist_ok=True)

        best_model_params_path = self.run_path / "best_model_params.pt"
        wandb.define_metric("loss.epoch.val.total", step_metric="epoch", summary="min")
        wandb.define_metric("loss.epoch.train.total", step_metric="epoch", summary="min")
        wandb.define_metric("loss.epoch.val_best", step_metric="epoch", summary="min")
        wandb.define_metric("lr", step_metric="epoch")
        wandb.define_metric("known_seq_len", hidden=True)
        wandb.define_metric("known_cascade_len", hidden=True)

        for epoch in (train_tqdm := trange(self.epochs)):
            self.train_epoch()
            if epoch % self.validation_period == 0 or epoch == self.epochs - 1:
                raw_losses = {
                    "train": self.evaluate(self.train_dataset),
                    "val": self.evaluate(self.val_dataset)
                }
                if self.test_dataset is not None:
                    raw_losses['test'] = self.evaluate(self.test_dataset)
                loss_dict = {}
                metrics_dict = {}
                if self.target == TargetClass.Scalar:
                    total_val_loss = raw_losses['val']
                    for name, loss in raw_losses.items():
                        loss_dict[name] = {"mae": loss.item()}
                elif self.target == TargetClass.Classification:
                    total_val_loss = raw_losses['val']
                    for name, loss in raw_losses.items():
                        loss_dict[name] = {"cross_entropy": loss.item()}
                    # Compute classification metrics (Acc, Precision, Recall, F1, AUC)
                    metrics_dict = {}
                    metrics_dict["train"] = self.evaluate_classification_metrics(self.train_dataset)
                    metrics_dict["val"] = self.evaluate_classification_metrics(self.val_dataset)
                    if self.test_dataset is not None:
                        metrics_dict["test"] = self.evaluate_classification_metrics(self.test_dataset)
                else:
                    total_val_loss = raw_losses['val'].sum()
                    for name, loss in raw_losses.items():
                        loss_dict[name] = {name: loss[i] for i, name in enumerate(self.cascade_order)}
                        loss_dict[name]["total"] = loss.sum().item()
                # wandb.log({"loss.epoch": loss_dict,
                #            "lr": self.optimizer.param_groups[0]['lr'],
                #            "epoch": epoch}, commit=False)
                wandb.log({
                    "loss.epoch": loss_dict,
                    "metrics.epoch": metrics_dict if self.target == TargetClass.Classification else {},
                    "lr": self.optimizer.param_groups[0]['lr'],
                    "epoch": epoch,
                }, commit=False)
                # Persist latest classification metrics into summary
                if self.target == TargetClass.Classification:
                    try:
                        # Ensure a stable nested structure in summary
                        if "metrics" not in wandb.run.summary:
                            wandb.run.summary["metrics"] = {}
                        wandb.run.summary["metrics"]["train"] = metrics_dict.get("train", {})
                        wandb.run.summary["metrics"]["val"] = metrics_dict.get("val", {})
                        if "test" in metrics_dict:
                            wandb.run.summary["metrics"]["test"] = metrics_dict["test"]
                        # Track best validation AUC
                        val_auc = metrics_dict.get("val", {}).get("auc", float("nan"))
                        if not hasattr(self, "_best_val_auc"):
                            self._best_val_auc = float("-inf")
                        if not math.isnan(val_auc) and val_auc > self._best_val_auc:
                            self._best_val_auc = val_auc
                            wandb.run.summary["best.val.auc"] = val_auc
                    except Exception as e:
                        logger.warning("Failed to update W&B summary metrics: %s", e)
                if total_val_loss < best_val_loss:
                    best_val_loss = total_val_loss
                    best_val_epoch = epoch
                    torch.save(self.model.state_dict(), best_model_params_path)
                    wandb.save(best_model_params_path, base_path=self.run_path, policy="live")
                    if self.target == TargetClass.Classification:
                        train_tqdm.set_description(
                            f"Epoch {epoch}; loss_epoch.val {total_val_loss.item():.4f} "
                            f"saved to {best_model_params_path} | "
                            f"Accuracy {metrics_dict['val'].get('accuracy', float('nan')):.4f}, "
                            f"Precision {metrics_dict['val'].get('precision', float('nan')):.4f}, "
                            f"Recall {metrics_dict['val'].get('recall', float('nan')):.4f}, "
                            f"F1 {metrics_dict['val'].get('f1', float('nan')):.4f}, "
                            f"AUC {metrics_dict['val'].get('auc', float('nan')):.4f}")
                    else:        
                        train_tqdm.set_description(
                            f"Epoch {epoch}; loss_epoch.val {total_val_loss.item():.4f} "
                            f"saved to {best_model_params_path}")
                    wandb.log({"loss.epoch.val_best": best_val_loss}, commit=False)
                if epoch - best_val_epoch > self.early_stopping_patience_epochs:
                    print(f"Early stopping at epoch {epoch} after more than "
                          f"{self.early_stopping_patience_epochs} epochs without improvement")
                    break
                # Don't step the scheduler on the tail epoch to presereve
                # patience behaviour
                if self.scheduler and epoch % self.validation_period == 0:
                    self.scheduler.step(total_val_loss)

        # Make sure we log the last evaluation results
        wandb.log({}, commit=True)


    def generate_structures(self, n_structures: int, calibrate: bool) -> List[dict]:
        generator = WyckoffGenerator(
            self.model, self.cascade_order, self.cascade_is_target, self.token_engineers,
            self.train_dataset.masks, self.train_dataset.max_sequence_length)
        if calibrate:
            generator.calibrate(self.val_dataset)
        if self.model.start_type == "categorial":
            max_start = self.model.start_embedding.num_embeddings
            start_counts = torch.bincount(
                self.train_dataset.start_tokens, minlength=max_start) + torch.bincount(
                    self.val_dataset.start_tokens, minlength=max_start)
            start_tensor = torch.distributions.Categorical(probs=start_counts.float()).sample((n_structures,))
        elif self.model.start_type == "one_hot":
            # We are going to sample with replacement from train+val datasets
            all_starts = torch.cat([self.train_dataset.start_tokens, self.val_dataset.start_tokens], dim=0)
            start_tensor = all_starts[torch.randint(len(all_starts), (n_structures,))]
        else:
            raise ValueError(f"Unknown start type: {self.model.start_type}")
        generated_tensors = generator.generate_tensors(start_tensor)
        generated_cascade_order = self.cascade_order
        if self.cascade_order[-1] == "harmonic_site_symmetries":
            del generated_tensors[-1]
            generated_cascade_order = generated_cascade_order[:-1]
        generated_tensors = torch.stack(generated_tensors, dim=-1)

        if 'sites_enumeration' in self.tokenisers:
            letter_from_ss_enum_idx = get_letter_from_ss_enum_idx(self.tokenisers['sites_enumeration'])
        else:
            letter_from_ss_enum_idx = None
        with gzip.open(preprocessed_wyckhoffs_cache_path, "rb") as f:
            ss_from_letter = pickle.load(f)[2]
        to_pyxtal = partial(tensor_to_pyxtal,
                            tokenisers=self.tokenisers,
                            token_engineers=self.token_engineers,
                            cascade_order=generated_cascade_order,
                            letter_from_ss_enum_idx=letter_from_ss_enum_idx,
                            ss_from_letter=ss_from_letter,
                            wp_index=get_wp_index())
        #with Pool() as pool:
            #structures = pool.starmap(to_pyxtal, zip(start_tensor.detach().cpu(), generated_tensors.detach().cpu()))
        structures = list(map(to_pyxtal, start_tensor.detach().cpu(), generated_tensors.detach().cpu()))
        print(f"Generated {len(structures)} Wyckoffs")
        valid_structures = [s for s in structures if s is not None]
        print(f"From which {len(valid_structures)} are valid")
        return valid_structures
    
    def generate_evaluate_and_log_wp(
        self,
        generation_name: str,
        calibrate: bool,
        n_structures: int,
        evaluator: StatisticalEvaluator):
    
        generated_wp = self.generate_structures(n_structures, calibrate)
        file_name = self.run_path / f"generated_wp_{generation_name}.json.gz"
        with gzip.open(file_name, "wt") as f:
            json.dump(generated_wp, f)
        wandb.save(file_name, base_path=self.run_path, policy="now")
        evaluate_and_log(generated_wp, generation_name, n_structures, evaluator)




def train_from_config(
    config_dict: dict,
    device: torch.device,
    run_path: Path = Path(__file__).parent.parent / "runs"):

    if wandb.run is None:
        raise ValueError("W&B run must be initialized")
    this_run_path = run_path / wandb.run.id
    trainer = WyckoffTrainer.from_config(config_dict, device, this_run_path)    
    trainer.train()
    this_run_path.mkdir(parents=True, exist_ok=True)
    with gzip.open(this_run_path / "tokenizers.pkl.gz", "wb") as f:
        pickle.dump(trainer.tokenisers, f)
    wandb.save(this_run_path / "tokenizers.pkl.gz", base_path=this_run_path, policy="now")
    with gzip.open(this_run_path / "token_engineers.pkl.gz", "wb") as f:
        pickle.dump(trainer.token_engineers, f)
    wandb.save(this_run_path / "token_engineers.pkl.gz", base_path=this_run_path, policy="now")
    config = OmegaConf.create(config_dict)
    if config.model.WyckoffTrainer_args.target == "NextToken" and \
        config.evaluation.get("n_structures_to_generate", 0) > 0:

        print("Training complete, loading the best model")
        trainer.model.load_state_dict(torch.load(trainer.run_path / "best_model_params.pt", weights_only=True))
        data_cache_path = Path(__file__).parent.parent.resolve() / "cache" / config.dataset / "data.pkl.gz"
        with gzip.open(data_cache_path, "rb") as f:
            datasets_pd = pickle.load(f)
        del datasets_pd["train"]
        del datasets_pd["val"]

        test_no_sites = datasets_pd['test']['site_symmetries'].map(len).values
        num_sites_bins = np.arange(0, 21)
        test_no_sites_hist = np.histogram(test_no_sites, bins=num_sites_bins)
        wandb.run.summary["num_sites"] = {"test": {"hist": wandb.Histogram(np_histogram=test_no_sites_hist)}}

        print(f"Test dataset size: {len(datasets_pd['test']['site_symmetries'])}")
        wandb.run.summary["test_dataset_size"] = len(datasets_pd["test"]["site_symmetries"])
        evaluator = StatisticalEvaluator(datasets_pd["test"])
        test_smact_validity = datasets_pd["test"]["composition"].map(smac_validity_from_counter).mean()
        print(f"SMAC-T validity on the test dataset: {test_smact_validity}")
        wandb.run.summary["smact_validity"] = {"test": test_smact_validity}
        wandb.run.summary["formal_validity"] = {}
        wandb.run.summary["wp"] = {}
        print("No calibration:")
        trainer.generate_evaluate_and_log_wp(
            "no_calibration", calibrate=False, n_structures=config.evaluation.n_structures_to_generate,
            evaluator=evaluator)
        print("Temperature calibration:")
        trainer.generate_evaluate_and_log_wp(
            "temperature_calibration", calibrate=True, n_structures=config.evaluation.n_structures_to_generate,
            evaluator=evaluator)