from typing import Tuple, List, Dict
from pathlib import Path
import logging
import torch
from torch import nn, Tensor

from cascade_transformer.dataset import AugmentedCascadeDataset, TargetClass, jagged_batch_randperm
from wyckoff_transformer.tokenization import load_tensors_and_tokenisers

logger = logging.getLogger(__name__)

class TemperatureScaling(nn.Module):
    def __init__(self):
        super(TemperatureScaling, self).__init__()
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, logits):
        # Apply temperature scaling
        return self.temperature_scale(logits)

    def temperature_scale(self, logits):
        # Temperature scaling of logits
        temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
        return logits / temperature

    def fit(self, logits, labels):
        # Fit the temperature parameter using the validation data
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=50)

        def eval_():
            optimizer.zero_grad()
            loss = criterion(self.temperature_scale(logits), labels)
            loss.backward()
            return loss

        logger.info("Cross entropy before calibration: %f", criterion(logits, labels).item())
        optimizer.step(eval_)
        logger.info("Cross entropy after calibration: %f", criterion(self.temperature_scale(logits), labels).item())
        logger.info("Temperature: %f", self.temperature.item())
        return self


class WyckoffGenerator():
    @classmethod
    def from_wandb_run(
        cls,
        wandb_run_id,
        update_wandb: bool = False,
        device: torch.device = torch.device("cpu")):
        import wandb
        from omegaconf import OmegaConf
        from cascade_transformer.model import CascadeTransformer

        if update_wandb:
            wandb_run = wandb.init(project="WyckoffTransformer", id=wandb_run_id, resume=True)
        else:
            wandb_run = wandb.Api().run(f"WyckoffTransformer/{wandb_run_id}")

        config = OmegaConf.create(dict(wandb_run.config))

        # The start tokens will be sampled from the train+validation datasets,
        # to preserve the sanctity of the test dataset and ex nihilo generation.
        tensors, tokenisers, engineers = load_tensors_and_tokenisers(config.dataset, config.tokeniser.name)

        model = CascadeTransformer.from_config_and_tokenisers(config, tokenisers, device)
        model.load_state_dict(torch.load(Path("runs", wandb_run_id, "best_model_params.pt"), map_location=device))
        # We need to grab any tensor from the train dataset
        max_sequence_len = tensors["train"][config.model.cascade.order[0]].size(1)

        masks_dict = {field: tokenisers[field].mask_token for field in config.model.cascade.order}

        generator = WyckoffGenerator(model,
            config.model.cascade.order,
            config.model.cascade.is_target,
            engineers,
            masks_dict, max_sequence_len)
        return generator, wandb_run, tensors, tokenisers, engineers


    def __init__(self,
                 model: nn.Module,
                 cascade_order: Tuple,
                 cascade_is_target: Dict,
                 token_engineers: Dict,
                 masks: dict,
                 max_sequence_len: int):
        self.model = model
        self.max_sequence_len = max_sequence_len
        self.cascade_order = cascade_order
        self.cascade_is_target = cascade_is_target
        self.masks = masks
        self.calibrators = None
        self.tail_calibrators = None
        self.token_engineers = token_engineers


    def calibrate(self, dataset: AugmentedCascadeDataset, calibration_element_count_threshold: int = 100):
        """
        The calibraiton is going to be per cascade field.
        We will generate p_predicted and p_true for each cascade field for each
        known sequence length.
        """
        assert dataset.cascade_order == self.cascade_order
        with torch.no_grad():
            self.model.eval()
            self.calibrators = []
            self.tail_calibrators = []
            for known_cascade_len, cascade_name in enumerate(self.cascade_order):
                tail_predictions = []
                tail_targets = []
                self.calibrators.append([])
                if not self.cascade_is_target[cascade_name]:
                    logging.info("Cascade field %s is not a target doesn't need calibration", cascade_name)
                    continue
                logging.info("Calibrating cascade field %s", cascade_name)
                for known_seq_len in range(dataset.max_sequence_length):
                    if known_cascade_len == 0:
                        start_tokens, masked_data, target = dataset.get_masked_multiclass_cascade_data(
                            known_seq_len, known_cascade_len,
                            target_type=TargetClass.NextToken, multiclass_target=True)
                    else:
                        start_tokens, masked_data, target = dataset.get_masked_multiclass_cascade_data(
                            known_seq_len, known_cascade_len,
                            target_type=TargetClass.NextToken, multiclass_target=False)
                    model_output = self.model(start_tokens, masked_data, None, known_cascade_len)
                    # Enought data for separate calibration
                    if target.size(0) >= calibration_element_count_threshold:
                        # If model_output and target are on different devices, not our problem
                        self.calibrators[known_cascade_len].append(TemperatureScaling().to(target.device).fit(
                            model_output, target))
                    # Not enough data for separate calibration, so we gather all the data in the tail
                    else:
                        tail_predictions.append(model_output)
                        tail_targets.append(target)
                tail_predictions = torch.concatenate(tail_predictions, axis=0)
                tail_targets = torch.concatenate(tail_targets, axis=0)
                if tail_predictions.size(0) < calibration_element_count_threshold:
                    logger.warning("Tail too small, %i when requested %i", tail_predictions.size(0), calibration_element_count_threshold)
                self.tail_calibrators.append(TemperatureScaling().to(tail_targets.device).fit(tail_predictions, tail_targets))


    def compute_likelihoods(self,
        dataset: AugmentedCascadeDataset,
        n_permutations: int = 1,
        n_augmentations: int = 1) -> Tensor:
        """
        Computes the likelihood of the dataset.
        """
        with torch.no_grad():
            self.model.eval()
            log_likelihoods = torch.zeros(
                n_permutations, n_augmentations, dataset.start_tokens.size()[0],
                device=dataset.device, dtype=torch.float32)
            if n_augmentations > 1:
                augmentations = [dataset.get_augmentation() for _ in range(n_augmentations)]
            else:
                augmentations = [None]
            for permutation_idx in range(n_permutations):
                if n_permutations > 1:
                    full_permutation = jagged_batch_randperm(
                        dataset.pure_sequences_lengths, dataset.max_sequence_length)
                    applied_permutation = True
                else:
                    full_permutation = None
                    applied_permutation = False
                for augmentation_idx, augmented_data in enumerate(augmentations):
                    for known_seq_len in range(self.max_sequence_len):
                        for known_cascade_len in range(len(self.cascade_order)):
                            start, this_data, target, batch_target_is_viable = dataset.get_masked_multiclass_cascade_data(
                                known_seq_len, known_cascade_len, TargetClass.NextToken, multiclass_target=False,
                                augmented_data=augmented_data, full_permutation=full_permutation,
                                apply_permutation=applied_permutation, return_chosen_indices=True)
                            logits = self.model(start, this_data, None, known_cascade_len)
                            log_probas = torch.nn.functional.log_softmax(logits, dim=1)
                            log_likelihoods[permutation_idx, augmentation_idx, batch_target_is_viable] += torch.gather(
                                log_probas, 1, target.unsqueeze(1)).squeeze()
            return log_likelihoods

    @torch.no_grad()
    def generate_tensors(
        self,
        start: Tensor,
        temperature: float = 1) -> List[Tensor]:
        """
        Generates a sequence of tokens.

        Arguments:
            start: The start token. It should be a tensor of shape [batch_size].
            temperature: The temperature to use for the generation

        Returns:
            The generated sequence of tokens. It has shape [batch_size, max_len, len(cascade_order)].
            It doesn't include the start token.
        """
        self.model.eval()
        batch_size = start.size(0)
        # Since we are doing in-place operations, we can just pre-assign
        # everything to be a mask
        generated = []
        for field in self.cascade_order:
            if self.masks[field].dim() == 0:
                generated.append(torch.full((batch_size, self.max_sequence_len), self.masks[field],
                                            dtype=self.masks[field].dtype, device=start.device))
            else:
                generated.append(torch.tile(self.masks[field].unsqueeze(0), (batch_size, self.max_sequence_len)))   

        cascade_index_by_name = {name: idx for idx, name in enumerate(self.cascade_order)}
        if len(start.size()) > 1:
            start_converted = list(map(tuple, start.tolist()))
        else:
            start_converted = start.tolist()
        for known_seq_len in range(self.max_sequence_len):
            for known_cascade_len, cascade_name in enumerate(self.cascade_order):
                if self.cascade_is_target[cascade_name]:
                    this_generation_input = [generated_cascade[:, :known_seq_len + 1] for generated_cascade in generated]
                    logits = self.model(start, this_generation_input, None, known_cascade_len)
                    if self.calibrators is not None:
                        if known_seq_len < len(self.calibrators[known_cascade_len]):
                            logits = self.calibrators[known_cascade_len][known_seq_len](logits)
                        else:
                            logits = self.tail_calibrators[known_cascade_len](logits)
                    logits = logits / temperature
                    # binary/ternary hack
                    # if known_cascade_len == 0 and known_seq_len > 2:
                    #    logits *= 3.
                    # CONSIDER: remove probas for all special tokens aside from STOP
                    calibrated_probas = torch.nn.functional.softmax(logits, dim=1)
                    # calibrated_probas = probas.numpy()
                    # calibrated_probas = calibrator.predict_proba(probas.numpy())
                    generated[known_cascade_len][:, known_seq_len] = \
                        torch.multinomial(calibrated_probas, num_samples=1).squeeze()
                else:
                    if known_cascade_len != len(self.cascade_order) - 1:
                        raise NotImplementedError("Only the last cascade field can be non-target")
                    # If a field is not in the cascade, by exclusion it is the start token
                    this_engineer_input = [generated[cascade_index_by_name[name]][:, known_seq_len].tolist() 
                                            for name in self.token_engineers[cascade_name].inputs
                                                if name in cascade_index_by_name]
                    #print(this_engineer_input)
                    #print(len(this_engineer_input))
                    #print(this_generation_input[0].shape)
                    generated[known_cascade_len][:, known_seq_len] = \
                        torch.from_numpy(
                            self.token_engineers[cascade_name].get_feature_from_token_batch(start_converted, this_engineer_input))
        return generated
