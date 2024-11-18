from typing import Tuple, List, Optional, Iterable
import logging
import numpy as np
import torch
from torch import nn
from torch import Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from omegaconf import OmegaConf

from cascade_transformer.dataset import batched_bincount


class CascadeEmbedding(nn.Module):
    def __init__(self,
                 cascade,
                 **kwargs):
        """
        Arguments:
            cascade: a tuple of tuples (N_i, d_i, pad_i), where N_i is the number of possible values for the i-th token,
                d_i is the dimensionality of the i-th token embedding. If d_i is None, the token is not embedded.
                pad_i is padding_idx to be passed to torch.nn.Embedding
        """
        super().__init__()
        self.embeddings = torch.nn.ModuleList()
        self.total_embedding_dim = 0
        for n, d, pad, _ in cascade:
            if d is None:
                self.embeddings.append(None)
                self.total_embedding_dim += 1
            else:
                self.embeddings.append(nn.Embedding(n, d, padding_idx=pad, **kwargs))
                self.total_embedding_dim += d


    def forward(self, x: List[Tensor]) -> Tensor:
        """
        Arguments:
            x: Tensor of shape ``[batch_size, seq_len, len(cascade)]``

        Returns:
            Tensor of shape ``[batch_size, seq_len, self.total_embedding_dim]``
        """
        list_of_embeddings = []
        for tensor, emb in zip(x, self.embeddings):
            if emb is None:
                list_of_embeddings.append(tensor.unsqueeze(-1))
            else:
                list_of_embeddings.append(emb(tensor))
        return torch.cat(list_of_embeddings, dim=2)


def get_perceptron(input_dim: int, output_dim:int, num_layers:int) -> torch.nn.Module:
    """
    Returns a perceptron with num_layers layers, with ReLU activation.
    If num_layers is 1, returns a single linear layer.
    """

    if num_layers == 1:
        return nn.Linear(input_dim, output_dim)
    else:
        this_sequence = []
        for _ in range(num_layers - 1):
            this_sequence.append(nn.Linear(input_dim, input_dim))
            this_sequence.append(nn.ReLU())
        this_sequence.append(nn.Linear(input_dim, output_dim))
    return nn.Sequential(*this_sequence)


def get_pyramid_perceptron(
    input_dim: int,
    output_dim:int,
    num_layers:int,
    dropout: Optional[float] = None) -> torch.nn.Module:
    """
    Returns a perceptron with num_layers layers, with ReLU activation.
    If num_layers is 1, returns a single linear layer.
    """

    if num_layers == 1:
        return nn.Linear(input_dim, output_dim)
    else:
        this_sequence = []
        layer_sizes = np.linspace(input_dim, output_dim, num_layers + 1, dtype=int)
        for input_layer_size, output_layer_size in zip(layer_sizes[:-1], layer_sizes[1:]):
            this_sequence.append(nn.Linear(input_layer_size, output_layer_size))
            if dropout is not None:
                this_sequence.append(nn.Dropout(dropout))
            if len(this_sequence) < num_layers * 2 - 1:
                this_sequence.append(nn.ReLU())
    return nn.Sequential(*this_sequence)


class CascadeTransformer(nn.Module):
    @classmethod
    def from_config_and_tokenisers(cls, config: OmegaConf,
        tokenisers: dict, device: torch.device):

        if len(config.tokeniser.augmented_token_fields) > 1:
            raise ValueError("Only one augmented field is supported")

        cascade_order = config.model.cascade.order

        full_cascade = dict()
        for field in cascade_order:
            full_cascade[field] = (len(tokenisers[field]),
                                   config.model.cascade.embedding_size[field],
                                   tokenisers[field].pad_token,
                                   config.model.cascade.is_target[field])
        if config.model.CascadeTransformer_args.start_type == "categorial":
            n_start = len(tokenisers[config.model.start_token])
        elif config.model.CascadeTransformer_args.start_type == "one_hot":
            n_start = len(next(iter(tokenisers[config.model.start_token].values())))
        else:
            raise ValueError(f"Unknown start_type {config.start_type}")

        # perhaps it can be needed
        # full_cascade = {'elements': (92, 16, 89, True), 'site_symmetries': (78, 16, 75, True), 'sites_enumeration': (11, 8, 8, True)}
        # n_start = 109

        return cls(
            n_start=n_start,
            cascade=full_cascade.values(),
            **config.model.CascadeTransformer_args
            ).to(device)


    def __init__(self,
                 start_type: str,    
                 n_start: int|None,
                 cascade: Tuple[Tuple[int, int|None, int], ...],
                 learned_positional_encoding_max_size: Optional[int],
                 learned_positional_encoding_only_masked: bool,
                 token_aggregation: str|None,
                 aggregate_after_encoder: bool,
                 include_start_in_aggregation: bool,
                 aggregation_inclsion: str,
                 concat_token_counts: bool,
                 concat_token_presence: bool,
                 num_fully_connected_layers: int,
                 mixer_layers: int,
                 outputs: str|int,
                 perceptron_shape: str,
                 TransformerEncoderLayer_args: dict,
                 TransformerEncoder_args: dict,
                 compile_perceptrons: bool = True):
        """
        Expects tokens in the following format:
        START_k -> [] -> STOP -> PAD
        START_k is the start token, it can take one of the K values, will be embedded.
            In Wychoff transformer, we store the space group here.
        [] is the cascading token. Each element of the token can take values from 0 to N_i - 1.
         During input, some of them are embedded, some not, as per the cascade. We predict the probability for each value, from 
         0 to N_i. Non-embedded values are expected to be floats.
        STOP is stop.
        PAD is padding. Not predicted.

        Arguments:
            start_type: "one_hot" or "categorial". If categorial, n_start must be provided and the start token will be embedded.
                If one_hot, n_start is ignored and the start token will be treated as a one-hot vector, embedded via a linear layer.
            n_start: Number of possible start tokens or dimensionality of the one-hot start token.
            cascade: a tuple of tuples (N_i, d_i, pad_i, is_target), where N_i is the number of possible values for the i-th token,
                d_i is the dimensionality of the i-th token embedding. If d_i is None, the token is not embedded.
                pad_i is padding_idx to be passed to torch.nn.Embedding
                is_target is a boolean, if True, the model will have a separate head for predicting this token.
            token_aggregation: When predicting, concatenate to the MASK token aggregated values of all the tokens.
        """
        super().__init__()
        self.embedding = CascadeEmbedding(cascade)
        self.d_model = self.embedding.total_embedding_dim
        self.encoder_layers = TransformerEncoderLayer(self.d_model, batch_first=True, **TransformerEncoderLayer_args)
        self.transformer_encoder = TransformerEncoder(self.encoder_layers, **TransformerEncoder_args)
        self.start_type = start_type
        if start_type == "categorial":
            self.start_embedding = nn.Embedding(n_start, self.d_model)
        elif start_type == "one_hot":
            self.start_embedding = nn.Linear(n_start, self.d_model)
        self.learned_positional_encoding_max_size = learned_positional_encoding_max_size
        self.learned_positional_encoding_only_masked = learned_positional_encoding_only_masked
        if perceptron_shape == "pyramid":
            percepron_generator_raw = get_pyramid_perceptron
        elif perceptron_shape == "input":
            percepron_generator_raw = get_perceptron
        else:
            raise ValueError(f"Unknown perceptron_shape {perceptron_shape}")
        if compile_perceptrons:
            def percepron_generator(*args, **kwargs):
                return torch.compile(percepron_generator_raw(*args, **kwargs), fullgraph=True)
        else:
            percepron_generator = percepron_generator_raw
        if learned_positional_encoding_max_size != 0:
            self.positions_embedding = nn.Embedding(
                learned_positional_encoding_max_size,
                self.d_model)
        if mixer_layers == 1:
            # Since our tokens are concatenated, we need to mix the embeddings
            # before we can use multuple attention heads.
            # Actually, a fully-connected layer is an overparametrisation
            # but it's easier to implement. Completely redundant if nhead == 1.
            self.mixer = nn.Linear(self.d_model, self.d_model, bias=False)
        else:
            # Hypthesis: since we concatenate embeddings, the mixer is a possible place to add non-linearity.
            self.mixer = percepron_generator(self.d_model, self.d_model, mixer_layers)
        # Note that in the normal usage, we want to condition the cascade element prediction
        # on the previous element, so care should be taken as to which head to call.
        self.token_aggregation = token_aggregation
        if aggregation_inclsion == "None":
            self.aggregation_inclsion = None
        else:
            self.aggregation_inclsion = aggregation_inclsion
        self.aggregate_after_encoder = aggregate_after_encoder
        self.concat_token_counts = concat_token_counts
        self.concat_token_presence = concat_token_presence
        self.include_start_in_aggregation = include_start_in_aggregation
        prediction_head_size = 2 * self.d_model if aggregation_inclsion == "concat" else self.d_model
        self.prediction_heads = torch.nn.ModuleList()
        if num_fully_connected_layers == 0:
            raise ValueError("num_fully_connected_layers must be at least 1 for dimensionality reasons.")
        self.cascade = tuple(cascade)
        if outputs == "token_scores":
            for output_size, _, _, is_target in cascade:
                if is_target:
                    this_head_size = prediction_head_size
                    if concat_token_counts:
                        this_head_size += output_size
                    if concat_token_presence:
                        this_head_size += output_size
                    self.prediction_heads.append(percepron_generator(this_head_size, output_size, num_fully_connected_layers))
                else:
                    self.prediction_heads.append(None)
        else:
            self.the_prediction_head = percepron_generator(prediction_head_size, outputs, num_fully_connected_layers)


    def forward(self,
                start: Tensor,
                cascade: List[Tensor],
                padding_mask: Tensor|None,
                prediction_head: int|None) -> Tensor:
        logging.debug("Cascade len: %i", len(cascade))
        cascade_embedding = self.embedding(cascade)
        logging.debug("Cascade reported embedding dim: %i", self.embedding.total_embedding_dim)
        logging.debug("Cascade embedding size: (%i, %i, %i)", *cascade_embedding.size())
        cascade_embedding = self.mixer(cascade_embedding)
        if self.learned_positional_encoding_max_size != 0:
            if self.learned_positional_encoding_only_masked:
                positional_encoding = self.positions_embedding(
                    torch.tensor([cascade_embedding.size(1) - 1], device=start.device, dtype=start.dtype))
                cascade_embedding[:, -1] += positional_encoding
            else:
                sequence_range = torch.arange(0, cascade_embedding.size(1), device=start.device, dtype=start.dtype)
                positional_encoding = self.positions_embedding(sequence_range)
                cascade_embedding += positional_encoding.unsqueeze(0)

        data = torch.cat([self.start_embedding(start).unsqueeze(1), cascade_embedding], dim=1)

        transformer_output = self.transformer_encoder(data, src_key_padding_mask=padding_mask)

        logging.debug("Transforer output size: %s", transformer_output.size())
        if self.aggregate_after_encoder:
            aggregation_input = transformer_output
        else:
            aggregation_input = data

        aggregation_start_idx = int(not self.include_start_in_aggregation)
        if self.token_aggregation == "sum":
            aggregation = aggregation_input[:, aggregation_start_idx:-1].sum(dim=1)
        elif self.token_aggregation == "max":
            # 2 for start and MASK. 0 or 1 would be a bug, so we let the code crash.
            if aggregation_input.size(1) == 2 and not self.include_start_in_aggregation:
                # Possible opmisation: just 0. for aggregation_inclsion == "add"
                aggregation = torch.zeros_like(aggregation_input[:, 0])
            else:
                aggregation = aggregation_input[:, aggregation_start_idx:-1].max(dim=1).values
        elif self.token_aggregation == "mean":
            aggregation = (
                aggregation_input[:, aggregation_start_idx:] * (1 -padding_mask.float())[..., None]
            ).sum(dim=1) / (1 - padding_mask.float() + 1e-6).sum(dim=1)[..., None]

    
        if self.aggregation_inclsion == "concat":
            prediction_inputs = [transformer_output[:, -1], aggregation]
        elif self.aggregation_inclsion == "add":
            prediction_inputs = [transformer_output[:, -1] + aggregation]
        elif self.aggregation_inclsion is None:
            prediction_inputs = [transformer_output[:, -1]]
        elif self.aggregation_inclsion == "aggr":
            prediction_inputs = [aggregation]
        else:
            raise ValueError(f"Unknown aggregation_inclsion {self.aggregation_inclsion}")

        if self.concat_token_counts:
            token_counts = batched_bincount(cascade[prediction_head], dim=1, max_value=self.cascade[prediction_head][0])
            prediction_inputs.append(token_counts)

        if self.concat_token_presence:
            token_counts = batched_bincount(cascade[prediction_head], dim=1, max_value=self.cascade[prediction_head][0], dtype=torch.bool)
            prediction_inputs.append(token_counts)
        
        prediction_input = torch.cat(prediction_inputs, dim=1)

        if prediction_head is None:
            return self.the_prediction_head(prediction_input)
        return self.prediction_heads[prediction_head](prediction_input)