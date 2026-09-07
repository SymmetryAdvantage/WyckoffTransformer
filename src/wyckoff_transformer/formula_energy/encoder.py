"""A composition encoder with two heads: the floor, and how loose the bound is.

The encoder is the CrabNet shape -- learned element embeddings, a sinusoidal
encoding of stoichiometric fraction, a small transformer over the element
multiset, attention pooling. It is written here rather than imported because
the literature says the encoder is not where the remaining accuracy lives, and
because the loss is. Feeding CrabNet and Roost embeddings to a 2026 in-context
foundation model gave "only marginal or inconsistent improvements ... suggesting
that their attention-based encoders already saturate the accessible
compositional information" (npj Comput Mater 2026); the cross-modal transfer
work of the same year reaches state of the art on 25 of 32 tasks but only
*approaches* CrabNet on composition, using pretraining aimed at small data. We
have 2.2M formulas. Meanwhile every published composition model regresses the
energy of an entry, and we need the floor beneath a set of entries, which is a
different estimand and needs :mod:`wyckoff_transformer.censored`.

The two heads are the design. Both read a trunk built from chemistry alone; only
the scale head additionally reads provenance. So the estimate of ``f*(X)`` is a
function of the elements and their proportions and of nothing else, while the
width of the excess above it is free to depend on who looked and how hard. See
:mod:`.features` for why that separation is not optional.

Output is ``[N, 2]`` -- ``(location, log_scale)`` -- which is exactly the shape
:class:`wyckoff_transformer.censored.CensoredMinLoss` splits, and also the shape
of Wren's two-output robust-L1 head, so the two are directly comparable.
"""
from __future__ import annotations

import math
from typing import Optional, Sequence

import torch
from torch import Tensor, nn

from wyckoff_transformer.formula_energy.features import N_ELEMENT_SLOTS


class FractionalEncoder(nn.Module):
    """Sinusoidal encoding of a stoichiometric fraction, CrabNet's device.

    A fraction is a continuous quantity that has to be compared across formulas,
    and an embedding table cannot interpolate between 0.33 and 0.34. Projecting
    it through sines of many periods gives a representation where nearby
    fractions are nearby vectors, which is the same reason positional encodings
    are built this way.

    Half the width is spent on a linear scale and half on a logarithmic one: the
    linear half resolves the difference between a third and a half, the log half
    resolves the difference between a dopant at 1% and one at 0.1%.
    """

    def __init__(self, d_model: int, resolution: int = 100, log_floor: float = 1e-3) -> None:
        super().__init__()
        if d_model % 4:
            raise ValueError(f"d_model must be divisible by 4, got {d_model}")
        self.resolution = resolution
        self.log_floor = log_floor
        half = d_model // 2
        frequencies = torch.exp(
            -math.log(10_000.0) * torch.arange(0, half, 2, dtype=torch.float32) / half
        )
        self.register_buffer("frequencies", frequencies, persistent=False)

    def _sinusoids(self, values: Tensor) -> Tensor:
        angles = values.unsqueeze(-1) * self.frequencies
        return torch.cat([angles.sin(), angles.cos()], dim=-1)

    def forward(self, fractions: Tensor) -> Tensor:
        """``[B, L]`` fractions in [0, 1] into ``[B, L, d_model]``."""
        linear = fractions * self.resolution
        logarithmic = (
            torch.log(fractions.clamp(min=self.log_floor)) / math.log(self.log_floor)
        ) * self.resolution
        return torch.cat([self._sinusoids(linear), self._sinusoids(logarithmic)], dim=-1)


def _mlp(widths: Sequence[int], dropout: float) -> nn.Sequential:
    """Wren's output-network shape: a few narrowing ReLU layers onto one scalar."""
    layers: list[nn.Module] = []
    for inputs, outputs in zip(widths[:-1], widths[1:]):
        layers += [nn.Linear(inputs, outputs), nn.ReLU(), nn.Dropout(dropout)]
    layers.append(nn.Linear(widths[-1], 1))
    return nn.Sequential(*layers)


class CompositionEncoder(nn.Module):
    """Element multiset to one fixed-width vector, invariant to element order."""

    def __init__(
        self,
        d_model: int = 256,
        n_layers: int = 3,
        n_heads: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.element = nn.Embedding(N_ELEMENT_SLOTS, d_model, padding_idx=0)
        self.fraction = FractionalEncoder(d_model)
        # A learned scalar rather than a fixed sum, so the model can decide how
        # much of the signal is identity and how much is proportion.
        self.fraction_scale = nn.Parameter(torch.tensor(1.0))
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model, n_heads, dim_feedforward, dropout,
                activation="gelu", batch_first=True, norm_first=True,
            ),
            num_layers=n_layers,
            # The fast nested-tensor path is silently unavailable under
            # norm_first anyway; asking for it only produces a warning per call.
            enable_nested_tensor=False,
        )
        self.pool = nn.Linear(d_model, 1)
        self.d_model = d_model

    def forward(self, element_ids: Tensor, fractions: Tensor, padding_mask: Tensor) -> Tensor:
        """``[B, L]`` inputs into a ``[B, d_model]`` trunk.

        Args:
            element_ids: Atomic numbers, 0 for padding.
            fractions: Stoichiometric fractions, summing to 1 over real slots.
            padding_mask: ``True`` where the slot is padding.
        """
        hidden = self.element(element_ids) + self.fraction_scale * self.fraction(fractions)
        hidden = self.transformer(hidden, src_key_padding_mask=padding_mask)
        scores = self.pool(hidden).squeeze(-1).masked_fill(padding_mask, float("-inf"))
        return (scores.softmax(dim=-1).unsqueeze(-1) * hidden).sum(dim=1)


class FormulaEnergyModel(nn.Module):
    """Predict ``(location, log_scale)`` for the censored-minimum likelihood.

    Args:
        n_provenance: Width of the provenance vector, ``len(PROVENANCE_FEATURES)``.
        location_feature_indices: Columns of the provenance vector the *location*
            head may also read. Empty by default, which is the exclusion
            restriction: the floor is a function of chemistry alone. Naming a
            column here is a deliberate relaxation, for a feature argued to carry
            chemistry rather than selection -- neighbourhood density being the
            case in point. Everything not named remains invisible to the floor.
        location_bias: Initial value of the location head's output bias. Targets
            are formation energies in eV/atom and are deliberately not
            standardised -- ``CensoredMinLoss`` carries a label-noise scale and a
            scale floor in those units -- so the head starts near the data mean
            instead of at zero.
        detach_scale_trunk: Stop gradients from the scale head reshaping the
            trunk. The location is a function of chemistry either way; this
            additionally keeps the *representation* from being fitted to explain
            search effort.
    """

    def __init__(
        self,
        n_provenance: int,
        d_model: int = 256,
        n_layers: int = 3,
        n_heads: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        head_widths: Sequence[int] = (256, 256, 128, 64),
        location_bias: float = 0.0,
        detach_scale_trunk: bool = False,
        location_feature_indices: Sequence[int] = (),
    ) -> None:
        super().__init__()
        if any(index >= n_provenance or index < 0 for index in location_feature_indices):
            raise ValueError(
                f"location_feature_indices {list(location_feature_indices)} out of range "
                f"for {n_provenance} provenance features"
            )
        self.encoder = CompositionEncoder(d_model, n_layers, n_heads, dim_feedforward, dropout)
        self.location_feature_indices = tuple(location_feature_indices)
        self.location_head = _mlp(
            [d_model + len(self.location_feature_indices), *head_widths], dropout)
        self.scale_head = _mlp([d_model + n_provenance, *head_widths], dropout)
        self.n_provenance = n_provenance
        self.detach_scale_trunk = detach_scale_trunk
        with torch.no_grad():
            self.location_head[-1].bias.fill_(location_bias)

    def forward(
        self,
        element_ids: Tensor,
        fractions: Tensor,
        padding_mask: Tensor,
        provenance: Tensor,
    ) -> Tensor:
        """Returns ``[B, 2]``: column 0 the floor, column 1 its log excess scale."""
        if provenance.size(-1) != self.n_provenance:
            raise ValueError(
                f"Expected {self.n_provenance} provenance features, got {provenance.size(-1)}"
            )
        trunk = self.encoder(element_ids, fractions, padding_mask)
        location = self.location_head(self._location_input(trunk, provenance))
        scale_trunk = trunk.detach() if self.detach_scale_trunk else trunk
        log_scale = self.scale_head(torch.cat([scale_trunk, provenance], dim=-1))
        return torch.cat([location, log_scale], dim=-1)

    def _location_input(self, trunk: Tensor, provenance: Tensor) -> Tensor:
        """The trunk, plus whichever provenance columns the floor is allowed to see."""
        if not self.location_feature_indices:
            return trunk
        columns = provenance[..., list(self.location_feature_indices)]
        return torch.cat([trunk, columns], dim=-1)

    @torch.no_grad()
    def predict_floor(
        self,
        element_ids: Tensor,
        fractions: Tensor,
        padding_mask: Tensor,
        provenance: Optional[Tensor] = None,
    ) -> Tensor:
        """The floor alone -- what screening a novel formula needs.

        With no named location features this takes no provenance at all, which is
        the point: a formula nobody has computed has none to supply. When some are
        named they have to be passed, and they are exactly the ones computable for
        an uncomputed composition.
        """
        trunk = self.encoder(element_ids, fractions, padding_mask)
        if self.location_feature_indices and provenance is None:
            raise ValueError(
                "This model's floor reads "
                f"{len(self.location_feature_indices)} provenance features; pass them"
            )
        if provenance is None:
            provenance = torch.zeros(trunk.size(0), self.n_provenance, device=trunk.device)
        return self.location_head(self._location_input(trunk, provenance)).squeeze(-1)
