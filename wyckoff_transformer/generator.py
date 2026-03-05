from typing import Tuple, List, Dict, Optional
from pathlib import Path
import logging
import torch
from torch import nn, Tensor
import numpy as np

from cascade_transformer.dataset import AugmentedCascadeDataset, TargetClass, jagged_batch_randperm
from wyckoff_transformer.tokenization import load_tensors_and_tokenisers, FeatureEngineer
from preprocess_wychoffs import inverse_series

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
        logger.warning("Loading current tensors and tokenisers for %s, not the saved ones", config.dataset)
        tensors, tokenisers, engineers = load_tensors_and_tokenisers(config.dataset, config.tokeniser.name)

        model = CascadeTransformer.from_config_and_tokenisers(config, tokenisers, device)
        model.load_state_dict(torch.load(Path("runs", wandb_run_id, "best_model_params.pt"), map_location=device))
        # We need to grab any tensor from the train dataset
        max_sequence_len = tensors["train"][config.model.cascade.order[0]].size(1)

        masks_dict = {field: tokenisers[field].mask_token for field in config.model.cascade.order}
        stops_dict = {field: tokenisers[field].stop_token for field in config.model.cascade.order}

        generator = WyckoffGenerator(model,
            config.model.cascade.order,
            config.model.cascade.is_target,
            engineers,
            masks_dict, max_sequence_len, stops=stops_dict)
        return generator, wandb_run, tensors, tokenisers, engineers


    def __init__(self,
                 model: nn.Module,
                 cascade_order: Tuple,
                 cascade_is_target: Dict,
                 token_engineers: Dict,
                 masks: dict,
                 max_sequence_len: int,
                 stops: Optional[Dict] = None):
        self.model = model
        self.max_sequence_len = int(max_sequence_len)
        self.cascade_order = cascade_order
        self.cascade_is_target = cascade_is_target
        self.masks = masks
        self.stops = stops
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
            # print(f"Generating {field} with mask shape {self.masks[field].shape}")
            if np.issubdtype(type(self.masks[field]), np.integer) or self.masks[field].ndim == 0:
                if np.issubdtype(type(self.masks[field]), np.integer):
                    dtype = torch.int64
                else:
                    dtype = self.masks[field].dtype
                generated.append(torch.full((batch_size, self.max_sequence_len), self.masks[field],
                                             dtype=dtype, device=start.device))
            else:
                unsqueezed_mask = self.masks[field].unsqueeze(0)
                if self.masks[field].dim() == 0:
                    generated.append(torch.tile(unsqueezed_mask, (batch_size, self.max_sequence_len)))
                elif self.masks[field].dim() == 1:
                    unsqueezed_mask = unsqueezed_mask.unsqueeze(0)
                    generated.append(torch.tile(unsqueezed_mask, (batch_size, self.max_sequence_len, 1)))
                else:
                    raise NotImplementedError("Mask should be a scalar or a vector")
            # print(f"Generated {field} with shape {generated[-1].shape}")
        # We have a problem. Engeineers by default can only work with data, not output of other engineers.
        if 'harmonic_site_symmetries' in self.cascade_order and 'sites_enumeration' not in self.cascade_order:
            cluster_to_enum = inverse_series(self.token_engineers["harmonic_cluster"].db)
            self.token_engineers['sites_enumeration'] = FeatureEngineer(
                cluster_to_enum, mask_token=None, stop_token=None, pad_token=None)
            
        cascade_index_by_name = {name: idx for idx, name in enumerate(self.cascade_order)}
        if len(start.size()) > 1:
            start_converted = list(map(tuple, start.tolist()))
        else:
            start_converted = start.tolist()
        stop_generated = np.zeros(batch_size, dtype=bool)
        for known_seq_len in range(self.max_sequence_len):
            for known_cascade_len, cascade_name in enumerate(self.cascade_order):
                if self.cascade_is_target[cascade_name]:
                    # +1 for MASK
                    this_generation_input = [generated_cascade[:, :known_seq_len + 1] for generated_cascade in generated]
                    #import pdb
                    #pdb.set_trace()
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
                    if self.stops is not None:
                        stop_generated |= generated[known_cascade_len][:, known_seq_len] == self.stops[cascade_name]
                else:
                    if known_cascade_len != len(self.cascade_order) - 1:
                        raise NotImplementedError("Only the last cascade field can be non-target")
                    if self.token_engineers[cascade_name].inputs[0] != 'spacegroup_number':
                        raise NotImplementedError("Only engineers with spacegroup_number first input are supported")
                    this_engineer_input = []
                    for input_field in self.token_engineers[cascade_name].inputs[1:]:
                        if input_field in cascade_index_by_name:
                            this_cascade_input = generated[cascade_index_by_name[input_field]][:, known_seq_len]
                        elif cascade_name == 'harmonic_site_symmetries' and input_field == 'sites_enumeration':
                            # Since we don't natively support either two engineers for one field or
                            # chainging engineers, we do this hack
                            enumerations = self.token_engineers['sites_enumeration'].get_feature_from_token_batch(
                                start_converted, [
                                    generated[cascade_index_by_name['site_symmetries']][:, known_seq_len].tolist(),
                                    generated[cascade_index_by_name['harmonic_cluster']][:, known_seq_len].tolist()])
                            this_cascade_input = enumerations
                            #import pdb
                            #pdb.set_trace()
                        else:
                            raise NotImplementedError(
                                f"Unknown input field {input_field} for engineer {cascade_name}")
                        this_engineer_input.append(this_cascade_input.tolist())
                    #print(this_engineer_input)
                    #print(len(this_engineer_input))
                    #print(this_generation_input[0].shape)
                    # print(f"Generating {cascade_name}")
                    feature_np = self.token_engineers[cascade_name].get_feature_from_token_batch(
                        start_converted, this_engineer_input)
                    if feature_np.dtype == "O": # Object, in this case array of array
                        # import pdb
                        # pdb.set_trace()
                        feature_np = np.stack(feature_np)
                    generated[known_cascade_len][:, known_seq_len] = \
                        torch.from_numpy(feature_np)
            ss_validitity = []
            enum_validity = []
            for structure_index, this_start in enumerate(start_converted):
                if stop_generated[structure_index]:
                    continue
                ss_validitity.append(
                    (this_start,
                     generated[cascade_index_by_name['site_symmetries']][structure_index, known_seq_len].item()
                     ) in self.token_engineers["multiplicity"].db)
                if 'sites_enumeration' in cascade_index_by_name:
                    enum_validity.append(
                        (this_start,
                            generated[cascade_index_by_name['site_symmetries']][structure_index, known_seq_len].item(),
                            generated[cascade_index_by_name['sites_enumeration']][structure_index, known_seq_len].item()
                        ) in self.token_engineers["multiplicity"].db)
                elif "harmonic_cluster" in cascade_index_by_name:
                    enum_validity.append(
                        (this_start,
                            generated[cascade_index_by_name['site_symmetries']][structure_index, known_seq_len].item(),
                            generated[cascade_index_by_name['harmonic_cluster']][structure_index, known_seq_len].item()
                        ) in self.token_engineers["sites_enumeration"].db)
            print(f"Known sequence length: {known_seq_len}")
            print(f"SS validity: {sum(ss_validitity) / len(ss_validitity) if len(ss_validitity) > 0 else '0/0'}")
            print(f"ENUM validity: {sum(enum_validity) / len(enum_validity) if len(enum_validity) > 0 else '0/0'}")
        return generated
