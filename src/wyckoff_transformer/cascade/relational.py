"""Crystallographic and chemical pairwise relational attention biases.

Wyckoff sites are an unordered set, and the backbone treats them as one: there is no
positional encoding over the site axis, and the trainer permutes the sequence at every
step so that nothing can smuggle order back in. What that buys in permutation invariance
it pays for in isotropy -- every pair of sites enters attention alike, though a
(Cs, F) pair and a (Cs, K) pair are not alike, and neither are two sites on `4/mmm` and
two on `1`.

This module injects the difference as an additive bias on the attention logits:

    A_ij = Q_i . K_j / sqrt(d_k) + B_chem(e_i, e_j) + B_wyckoff(G, ss_i, ss_j)

which is the only place it can go without breaking permutation invariance: B depends on
the *contents* of sites i and j, never on where they sit in the tensor, so permuting the
sites permutes B's rows and columns with them.

Both terms are symmetric in (i, j) by construction, and both are **zero at
initialisation** -- `elem_interaction`, the last layer of `chem_mlp` and `ss_pair_bias`
are all zero-initialised, in the same spirit as the AdaLN-Zero modulation next door. A
run therefore starts numerically identical to the same config without the bias, which is
what makes the comparison an ablation rather than a coincidence.

Cost is kept off the batch axis wherever it can be. Both biases depend only on the pair
of *token ids*, so the per-head tables are built once per forward over the vocabulary
(92 x 92 elements, 81 x 81 site symmetries) and then gathered into [B, N, N, H]. The
gather is unavoidable -- it is the bias tensor -- but the MLP and the embedding product
never see the batch.
"""

from typing import Tuple
import logging
import warnings

import numpy as np
import torch
from torch import nn, Tensor

logger = logging.getLogger(__name__)

#: Symmetric pairwise chemical descriptors, in the order the MLP receives them.
#: The write-up asks for [delta_chi, |r_i - r_j|, r_i / r_j]; the bare ratio is not
#: symmetric under i <-> j, so it is read as min/max, which is the same quantity for
#: the ordered pair and keeps B_chem(i, j) = B_chem(j, i) exactly.
CHEM_PAIR_FEATURES: Tuple[str, ...] = (
    "abs_electronegativity_difference",
    "abs_radius_difference",
    "radius_ratio_min_over_max",
)

#: Tokeniser entries that are not elements. Their rows and columns of the feature table
#: are zeroed (the post-standardisation mean), so a MASK/STOP/PAD site contributes no
#: physical signal -- only the learned element embedding, which has a row of its own.
_SERVICE_TOKENS = frozenset(("MASK", "STOP", "PAD"))


def _element_scalars(symbol: str) -> Tuple[float, float]:
    """Pauling electronegativity and a radius for one element symbol.

    Both are missing for parts of the table -- the noble gases have no Pauling
    electronegativity, and `atomic_radius` is `None` wherever Slater never published one
    -- so the radius falls back through the calculated and mean-ionic values. Whatever is
    still missing comes back as NaN and is filled with the vocabulary mean by the caller.
    """
    from pymatgen.core import Element  # noqa: PLC0415  # heavy import, model-construction only

    element = Element(symbol)
    with warnings.catch_warnings():
        # Both absences are expected and handled; pymatgen warns about them anyway.
        warnings.filterwarnings("ignore", r"No Pauling electronegativity for \w+\.")
        warnings.filterwarnings("ignore", r"No data available for \w+ for \w+")
        electronegativity = float(element.X)  # NaN for the noble gases, by pymatgen's design
        radius = np.nan
        # Lazily, so a later fallback is only consulted -- and only warns -- when needed.
        for attribute in ("atomic_radius", "atomic_radius_calculated", "average_ionic_radius"):
            candidate = getattr(element, attribute)
            if candidate is None:
                continue
            value = float(candidate)
            if value > 0:
                radius = value
                break
    return electronegativity, radius


def build_element_pair_features(element_tokeniser) -> np.ndarray:
    """Standardised symmetric pair descriptors for every pair of element tokens.

    Args:
        element_tokeniser: mapping element symbol -> token id, as produced by
            `EnumeratingTokeniser`; service tokens are included and handled.

    Returns:
        `[n_tokens, n_tokens, len(CHEM_PAIR_FEATURES)]` float32. Standardised over the
        element-element pairs, zero wherever either token is a service token.
    """
    n_tokens = len(element_tokeniser)
    electronegativity = np.full(n_tokens, np.nan)
    radius = np.full(n_tokens, np.nan)
    is_element = np.zeros(n_tokens, dtype=bool)
    for token, index in element_tokeniser.items():
        if token in _SERVICE_TOKENS:
            continue
        is_element[index] = True
        electronegativity[index], radius[index] = _element_scalars(token)

    if not is_element.any():
        raise ValueError("The element tokeniser holds no elements")
    # A handful of entries survive the fallbacks as NaN (electronegativity of He/Ne/Ar).
    # They become the mean of what is known, i.e. the least informative value available,
    # rather than poisoning every pair they appear in.
    for values in (electronegativity, radius):
        known = is_element & np.isfinite(values)
        if not known.all():
            logger.info("Filling %i missing element scalars with the vocabulary mean",
                        int(is_element.sum() - known.sum()))
        values[~known] = values[known].mean()

    delta_x = np.abs(electronegativity[:, None] - electronegativity[None, :])
    delta_r = np.abs(radius[:, None] - radius[None, :])
    pairwise_min = np.minimum(radius[:, None], radius[None, :])
    pairwise_max = np.maximum(radius[:, None], radius[None, :])
    ratio = pairwise_min / pairwise_max

    features = np.stack([delta_x, delta_r, ratio], axis=-1)
    element_pairs = is_element[:, None] & is_element[None, :]
    population = features[element_pairs]
    # A vocabulary of one element leaves every descriptor constant; keep the scale at 1
    # rather than dividing by zero and handing the MLP a table of NaN.
    spread = population.std(axis=0)
    features = (features - population.mean(axis=0)) / np.where(spread > 0, spread, 1.)
    features[~element_pairs] = 0.0
    return np.ascontiguousarray(features, dtype=np.float32)


class RelationalAttentionBias(nn.Module):
    """Additive `[B * nhead, L, L]` attention bias over a set of Wyckoff sites.

    The returned tensor is laid out for `nn.MultiheadAttention`'s 3-D `attn_mask`, which
    is where it is consumed: passing it as the encoder's `mask` leaves the rest of the
    attention -- the padding mask included -- exactly as it was.

    The sequence the encoder sees is `[start] + sites`, one longer than the cascade
    tensors. The start token carries the space group and is not a site, so its row and
    column of the bias are zero: it is not a member of the set the relations are defined
    over, and biasing its attention would make the bias depend on a position.
    """

    def __init__(self,
                 nhead: int,
                 num_elements: int,
                 num_site_symmetries: int,
                 element_pair_features: Tensor,
                 n_start: int,
                 start_type: str,
                 element_embedding_dim: int = 16,
                 chem_mlp_hidden: int = 32,
                 space_group_gate: bool = True):
        """
        Args:
            nhead: number of attention heads; the bias is per-head.
            num_elements: size of the element token vocabulary, service tokens included.
            num_site_symmetries: size of the site-symmetry token vocabulary.
            element_pair_features: `[num_elements, num_elements, n_features]` table from
                `build_element_pair_features`. Held as a buffer: not learned, but it
                travels with the checkpoint, so generation cannot silently disagree with
                training about what the physical features were.
            n_start: dimensionality of the start token (one-hot width, or vocabulary
                size when categorial).
            start_type: "one_hot" or "categorial", matching `CascadeTransformer`.
            element_embedding_dim: width of the element embedding whose elementwise
                product feeds the learned chemical interaction term.
            chem_mlp_hidden: hidden width of the MLP over the physical pair features.
            space_group_gate: whether the Wyckoff term is modulated by the space group.
        """
        super().__init__()
        if element_pair_features.dim() != 3 or element_pair_features.size(0) != num_elements \
                or element_pair_features.size(1) != num_elements:
            raise ValueError(
                f"element_pair_features must be [{num_elements}, {num_elements}, n_features], "
                f"got {tuple(element_pair_features.size())}")
        self.nhead = nhead
        self.num_elements = num_elements
        self.num_site_symmetries = num_site_symmetries

        # B_chem, learned part: W_e^T (v_i (*) v_j). Zero-initialised output, so the term
        # is exactly 0 at init while the embeddings below still start random -- W_e's own
        # gradient is proportional to v_i (*) v_j and would vanish if they did not.
        self.elem_embeddings = nn.Embedding(num_elements, element_embedding_dim)
        self.elem_interaction = nn.Linear(element_embedding_dim, nhead, bias=False)
        nn.init.zeros_(self.elem_interaction.weight)

        # B_chem, physical part: MLP over the standardised pair descriptors.
        self.register_buffer("element_pair_features", element_pair_features)
        self.chem_mlp = nn.Sequential(
            nn.Linear(element_pair_features.size(-1), chem_mlp_hidden),
            nn.SiLU(),
            nn.Linear(chem_mlp_hidden, nhead),
        )
        nn.init.zeros_(self.chem_mlp[-1].weight)
        nn.init.zeros_(self.chem_mlp[-1].bias)

        # B_wyckoff: a free symmetric table over site-symmetry pairs. Stored unsymmetrised
        # and averaged with its transpose on use, so both orderings share one parameter
        # and weight decay sees a single tensor.
        self.ss_pair_bias = nn.Parameter(torch.zeros(num_site_symmetries, num_site_symmetries, nhead))

        # The write-up adds E_sg(G) to B_wyckoff. As an additive term it is a no-op: it is
        # constant along j, and softmax is invariant to a constant per row. The space group
        # does condition the Wyckoff term here, but multiplicatively -- a per-head gate
        # (1 + g(G)), zero-initialised so it starts as the identity.
        self.start_type = start_type
        if space_group_gate:
            if start_type == "categorial":
                self.sg_gate = nn.Embedding(n_start, nhead)
                nn.init.zeros_(self.sg_gate.weight)
            elif start_type == "one_hot":
                self.sg_gate = nn.Linear(n_start, nhead)
                nn.init.zeros_(self.sg_gate.weight)
                nn.init.zeros_(self.sg_gate.bias)
            else:
                raise ValueError(f"Unknown start_type {start_type}")
        else:
            self.sg_gate = None

    def pair_tables(self) -> Tuple[Tensor, Tensor]:
        """Per-head bias tables over the vocabularies: `[V_e, V_e, H]`, `[V_ss, V_ss, H]`.

        Built once per forward, off the batch axis. Both are symmetric.
        """
        embeddings = self.elem_embeddings.weight
        chem = self.elem_interaction(embeddings.unsqueeze(1) * embeddings.unsqueeze(0))
        chem = chem + self.chem_mlp(self.element_pair_features)
        site_symmetry = 0.5 * (self.ss_pair_bias + self.ss_pair_bias.transpose(0, 1))
        return chem, site_symmetry

    def forward(self,
                elements: Tensor,
                site_symmetries: Tensor,
                start: Tensor) -> Tensor:
        """
        Args:
            elements: `[B, N]` element token ids, one per site.
            site_symmetries: `[B, N]` site-symmetry token ids, one per site.
            start: the start token, `[B, n_start]` one-hot or `[B]` categorial.

        Returns:
            `[B * nhead, N + 1, N + 1]` additive bias, ready to be passed as
            `nn.MultiheadAttention`'s `attn_mask`.
        """
        if elements.dim() != 2 or site_symmetries.dim() != 2:
            raise ValueError(
                "The relational bias needs categorial element and site-symmetry tokens, got "
                f"shapes {tuple(elements.size())} and {tuple(site_symmetries.size())}")
        batch_size, n_sites = elements.shape
        chem_table, ss_table = self.pair_tables()

        bias = chem_table[elements.unsqueeze(2), elements.unsqueeze(1)]  # [B, N, N, H]
        wyckoff = ss_table[site_symmetries.unsqueeze(2), site_symmetries.unsqueeze(1)]
        if self.sg_gate is not None:
            wyckoff = wyckoff * (1. + self.sg_gate(start).unsqueeze(1).unsqueeze(1))
        bias = bias + wyckoff

        # [B, N, N, H] -> [B, H, N + 1, N + 1], the leading row and column being the
        # start token, which takes no relational bias.
        bias = torch.nn.functional.pad(bias.permute(0, 3, 1, 2), (1, 0, 1, 0))
        return bias.reshape(batch_size * self.nhead, n_sites + 1, n_sites + 1)


def relational_bias_from_tokenisers(
    tokenisers: dict,
    cascade_order,
    nhead: int,
    n_start: int,
    start_type: str,
    element_field: str = "elements",
    site_symmetry_field: str = "site_symmetries",
    **kwargs) -> Tuple[RelationalAttentionBias, int, int]:
    """Builds the bias module and resolves the cascade indices it reads.

    Returns:
        (module, element cascade index, site-symmetry cascade index).
    """
    order = list(cascade_order)
    for field in (element_field, site_symmetry_field):
        if field not in order:
            raise ValueError(
                f"relational_attention_bias needs {field!r} in the cascade order, got {order}")
    element_index = order.index(element_field)
    site_symmetry_index = order.index(site_symmetry_field)
    features = torch.from_numpy(build_element_pair_features(tokenisers[element_field]))
    module = RelationalAttentionBias(
        nhead=nhead,
        num_elements=len(tokenisers[element_field]),
        num_site_symmetries=len(tokenisers[site_symmetry_field]),
        element_pair_features=features,
        n_start=n_start,
        start_type=start_type,
        **kwargs)
    return module, element_index, site_symmetry_index
