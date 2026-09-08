r"""Novelty of a Wyckoff gene, read off the generative WyFormer's own likelihood.

The energy screen (:mod:`wyckoff_transformer.cli.dft_screen`) trades novelty for
stability: it finds low-lying genes partly by finding compositions the reference
archive already holds.  This module supplies the opposite lever, from the
generator rather than from a critic.

A gene the model emits often is a gene its training archive is full of.  Its
self-information

.. math:: I(G) = -\\log p_\\theta(G)

is therefore a continuous novelty score, and unlike the fingerprint lookup the
protocol already does -- which only answers *is this exact gene in the reference
set* -- it orders the genes that lookup calls novel, separating a gene that is
one Wyckoff letter away from a known one from a gene the model can barely
express.

Making that a real density takes some care, because WyFormer does not generate a
sequence: it generates a *set* of Wyckoff sites, in a uniformly random order,
under any of the equivalent enumerations of the same positions.  Write
:math:`R(G)` for the set of distinct token sequences that decode to ``G``.  Then

.. math:: p_\\theta(G) = \\sum_{r\\in R(G)} p_\\theta(r)
                       = |R(G)|\; \\mathbb E_{r\\sim U(R(G))}\\,p_\\theta(r),

which is estimated here by drawing ``permutation_samples`` representations
uniformly from :math:`R(G)` and taking the importance-weighted mean.  Both the
log-mean-exp estimate and the Jensen (ELBO) lower bound are reported; they
bracket the truth from below and agree when the model's likelihood is
order-invariant.  ``log |R(G)|`` is computed exactly rather than estimated: it is
a combinatorial count, it is of the order of ``log n!``, and dropping it would
make the score a measure of gene size as much as of novelty.

The space group is not predicted by the model -- generation draws it from the
empirical training distribution -- so its log-probability comes from the saved
start-token distribution and is reported as its own column.
"""
from __future__ import annotations

import logging
from collections import Counter
from math import lgamma
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from wyckoff_transformer.cascade.dataset import AugmentedCascadeDataset, TargetClass
from wyckoff_transformer.prediction import (
    build_tokenised_prediction_tensors,
    filter_supported_tokens,
)

logger = logging.getLogger(__name__)

#: Fields a scored record must carry, beyond the space group.
SITE_FIELDS = ("elements", "site_symmetries", "sites_enumeration")

#: Columns :func:`score_gene_likelihood` writes, in output order.
LIKELIHOOD_COLUMNS = (
    "n_sites",
    "log_representations",
    "log_p_spacegroup",
    "mean_representation_log_likelihood",
    "std_representation_log_likelihood",
    "log_likelihood",
    "log_likelihood_elbo",
    "surprisal",
    "surprisal_per_site",
)


def log_orderings(sites: Sequence[tuple]) -> float:
    """``log`` of the number of distinct orderings of a multiset of sites.

    Two sites that carry the same (element, site symmetry, enumeration) token
    triple are the same token to the model, so swapping them does not give a new
    sequence and must not be counted as one.
    """
    counts = Counter(sites)
    total = lgamma(len(sites) + 1)
    for multiplicity in counts.values():
        total -= lgamma(multiplicity + 1)
    return total


def gene_representations(record: Dict) -> Tuple[List[Tuple[tuple, ...]], np.ndarray]:
    """Every distinct site multiset that decodes to one gene, and its ordering count.

    The equivalent Wyckoff enumerations are the model's other axis of redundancy
    alongside site order; ``pyxtal_notation_to_sites`` supplies them as
    ``sites_enumeration_augmented``.  Two enumerations can collapse to the same
    multiset of site triples, so they are deduplicated here -- double-counting a
    representation would inflate ``|R(G)|`` and make the gene look commoner than
    the model actually makes it.

    Returns:
        The distinct multisets, each as one arbitrary ordering of its site
        triples, and the log count of distinct orderings of each.
    """
    elements = list(record["elements"])
    symmetries = list(record["site_symmetries"])
    variants = record.get("sites_enumeration_augmented")
    # A record missing the column reads back as NaN once it has been through a
    # DataFrame, which is neither None nor empty and is not iterable either.
    if variants is None or not isinstance(variants, (list, tuple, set, frozenset)):
        variants = [record["sites_enumeration"]]
    elif not variants:
        variants = [record["sites_enumeration"]]
    distinct: Dict[frozenset, Tuple[tuple, ...]] = {}
    for enumeration in variants:
        sites = tuple(zip(elements, symmetries, enumeration))
        if len(sites) != len(elements):
            raise ValueError(
                "An augmented enumeration has a different number of sites than the gene")
        key = frozenset(Counter(sites).items())
        distinct.setdefault(key, sites)
    multisets = list(distinct.values())
    return multisets, np.fromiter(
        (log_orderings(sites) for sites in multisets), dtype=float, count=len(multisets))


def start_log_prior(trainer, records: pd.DataFrame) -> pd.Series:
    """log p(space group) under the distribution generation actually samples from.

    A space group the archive holds ten structures in is a rarer thing to be
    handed than one it holds ten thousand of, and the sampler knows that; the
    model does not, since the start token is given to it rather than predicted.
    """
    distribution = trainer.start_token_distribution
    if distribution is None:
        path = trainer.get_start_token_distribution_path(trainer.run_path)
        if not path.exists():
            raise FileNotFoundError(
                f"No start-token distribution at {path}; the space-group prior is part "
                "of the generative density and cannot be skipped silently.")
        distribution = trainer.load_start_token_distribution_file(path)
        trainer.start_token_distribution = distribution

    counts = np.asarray(distribution["counts"], dtype=float)
    if counts.sum() <= 0:
        raise ValueError("Start-token distribution counts are empty")
    log_probabilities = np.log(counts) - np.log(counts.sum())

    tokeniser = trainer.tokenisers[trainer.start_name]
    start_type = distribution["start_type"]
    if start_type == "categorial":
        by_key = {index: value for index, value in enumerate(log_probabilities)}

        def lookup(space_group):
            return by_key.get(tokeniser[space_group], -np.inf)
    elif start_type == "one_hot":
        by_key = {
            tuple(vector): value
            for vector, value in zip(distribution["vectors"], log_probabilities)
        }

        def lookup(space_group):
            vector = tuple(float(x) for x in tokeniser.np_dict[space_group])
            return by_key.get(vector, -np.inf)
    else:
        raise ValueError(f"Unknown start type in distribution: {start_type!r}")

    values = records[trainer.start_name].map(lookup)
    unseen = int(np.isneginf(values.to_numpy(dtype=float)).sum())
    if unseen:
        logger.warning(
            "%d genes are in a space group the start-token distribution never draws; "
            "their likelihood is zero and their surprisal infinite", unseen)
    return values


def _first_target_index(trainer) -> int:
    """The cascade field that decides where a sequence stops.

    Every target field carries STOP at the stopping position, so counting all of
    them would charge one stopping decision three times over.  Generation reads
    the first one, so that is the term that belongs in the density; the rest are
    nuisance draws the decoder discards, and they marginalise away.
    """
    return trainer.cascade_target_indices[0]


@torch.no_grad()
def representation_log_likelihood(
    trainer,
    frame: pd.DataFrame,
    cond: Optional[Tensor] = None,
) -> Tensor:
    """log p(sequence | space group) for rows already in a fixed order and enumeration.

    ``frame`` must carry the plain (non-augmented) site fields: this is the
    likelihood of one *representation*, not of the gene.
    """
    data = build_tokenised_prediction_tensors(frame, trainer)
    dataset = AugmentedCascadeDataset(
        data=data,
        cascade_order=trainer.cascade_order,
        masks=trainer.masks_dict,
        pads=trainer.pad_dict,
        stops=trainer.stops_dict,
        num_classes=trainer.num_classes_dict,
        start_field=trainer.start_name,
        augmented_fields=[],
        batch_size=None,
        dtype=trainer.dtype,
        start_dtype=(
            trainer.train_dataset.start_tokens.dtype
            if getattr(trainer, "train_dataset", None) is not None
            else (torch.int64 if trainer.model.start_type == "categorial" else torch.float32)
        ),
        device=trainer.device,
        augmented_storage_device=None,
        target_name=None,
    )
    lengths = dataset.pure_sequences_lengths.to(torch.int64)
    total = torch.zeros(len(frame), dtype=torch.float64, device=trainer.device)
    stop_index = _first_target_index(trainer)

    was_training = trainer.model.training
    trainer.model.eval()
    try:
        for known_seq_len in range(dataset.max_sequence_length):
            viable = (lengths >= known_seq_len).nonzero(as_tuple=True)[0]
            if viable.numel() == 0:
                break
            stopping = lengths[viable] == known_seq_len
            batch_cond = None if cond is None else cond[viable]
            for known_cascade_len in trainer.cascade_target_indices:
                if known_cascade_len != stop_index and bool(stopping.all()):
                    # Every row here has already stopped, so the only term that
                    # would be added is the redundant STOP copy.
                    continue
                start, cascade, target = dataset.get_masked_multiclass_cascade_data(
                    known_seq_len,
                    known_cascade_len,
                    target_type=TargetClass.NextToken,
                    multiclass_target=False,
                    batch_target_is_viable=viable,
                    apply_permutation=False,
                    truncate_invalid_targets=False,
                )
                logits = trainer.model(start, cascade, None, known_cascade_len, cond=batch_cond)
                token_log_probability = torch.log_softmax(logits.float(), dim=-1).gather(
                    1, target.to(torch.int64).unsqueeze(1)).squeeze(1)
                if known_cascade_len != stop_index:
                    token_log_probability = token_log_probability.masked_fill(stopping, 0.0)
                total.index_add_(0, viable, token_log_probability.to(torch.float64))
    finally:
        if was_training:
            trainer.model.train()
    return total


def _draw_representations(
    records: pd.DataFrame,
    multisets: Sequence[Sequence[Tuple[tuple, ...]]],
    weights: Sequence[np.ndarray],
    rng: np.random.Generator,
) -> pd.DataFrame:
    """One representation per gene, drawn uniformly from ``R(G)``.

    A multiset is chosen in proportion to how many distinct orderings it has, and
    then permuted uniformly at random: together that is a uniform draw over the
    representations, which is what the importance weight ``|R(G)|`` assumes.
    """
    drawn = {field: [] for field in SITE_FIELDS}
    for gene_multisets, gene_weights in zip(multisets, weights):
        probabilities = np.exp(gene_weights - gene_weights.max())
        probabilities /= probabilities.sum()
        sites = list(gene_multisets[rng.choice(len(gene_multisets), p=probabilities)])
        rng.shuffle(sites)
        for position, field in enumerate(SITE_FIELDS):
            drawn[field].append([site[position] for site in sites])
    frame = records.copy()
    for field in SITE_FIELDS:
        frame[field] = pd.Series(drawn[field], index=records.index)
    return frame.drop(columns=[
        column for column in frame.columns if column.endswith("_augmented")])


def score_gene_likelihood(
    records: pd.DataFrame,
    trainer,
    cond: Optional[Tensor] = None,
    permutation_samples: int = 8,
    seed: int = 0,
    batch_size: Optional[int] = None,
) -> pd.DataFrame:
    """The generative model's log-density, and its surprisal, for each gene.

    Args:
        records: Wyckoff records, as ``GeneFingerprinter.record`` builds them,
            indexed by whatever the caller wants carried through.
        trainer: A loaded generative ``WyckoffTrainer`` (``target: NextToken``).
        cond: Conditioning values in physical units, one row per record. Pass
            what generation was run at: the density a sample came from is the
            conditional one, and scoring it under a different condition measures
            a different generator.
        permutation_samples: Representations drawn per gene. The estimator is a
            lower bound that tightens with this; the spread across draws is
            reported so a caller can see whether it has enough.
        seed: Seeds the representation draws.
        batch_size: Genes per forward sweep. ``None`` scores the whole pool at
            once, which is what a GPU wants and what a large pool cannot afford.

    Returns:
        One row per record, with :data:`LIKELIHOOD_COLUMNS`.
    """
    if trainer.target != TargetClass.NextToken:
        raise ValueError(
            "Gene novelty is read from the generative model's likelihood; this "
            f"checkpoint has target {trainer.target}.")
    if permutation_samples < 1:
        raise ValueError("permutation_samples must be at least 1")
    if getattr(trainer, "composition_conditioning", False):
        raise ValueError(
            "A composition-conditioned generator needs a composition vector per gene, "
            "which this scorer does not build.")
    if cond is not None:
        if not trainer.condition_features:
            raise ValueError("cond was supplied, but this model has no condition features.")
        if cond.shape[0] != len(records):
            raise ValueError(f"cond has {cond.shape[0]} rows for {len(records)} genes")
        trainer._validate_condition_values(cond)
        cond = trainer.transform_condition(cond.to(trainer.device, dtype=torch.float32))
    elif trainer.condition_features:
        raise ValueError(
            f"This generator is conditioned on {list(trainer.condition_features)}; pass "
            "cond in physical units, at the value the pool was generated at.")

    multisets, weights = [], []
    for _, record in records.iterrows():
        gene_multisets, gene_weights = gene_representations(record)
        multisets.append(gene_multisets)
        weights.append(gene_weights)

    rng = np.random.default_rng(seed)
    chunk = len(records) if batch_size is None else max(1, int(batch_size))
    samples = np.empty((permutation_samples, len(records)), dtype=float)
    for draw in range(permutation_samples):
        for start in range(0, len(records), chunk):
            stop = min(start + chunk, len(records))
            frame = _draw_representations(
                records.iloc[start:stop], multisets[start:stop], weights[start:stop], rng)
            samples[draw, start:stop] = representation_log_likelihood(
                trainer, frame, cond=None if cond is None else cond[start:stop]
            ).cpu().numpy()

    log_representations = np.fromiter(
        (float(np.logaddexp.reduce(gene_weights)) for gene_weights in weights),
        dtype=float, count=len(records))
    log_p_spacegroup = start_log_prior(trainer, records).to_numpy(dtype=float)
    # log-mean-exp over the draws: the importance-sampling estimate of the sum
    # over representations, which is the quantity the ELBO below bounds.
    log_mean_exp = (
        np.logaddexp.reduce(samples, axis=0) - np.log(permutation_samples))

    result = pd.DataFrame(index=records.index)
    result["n_sites"] = records["elements"].map(len)
    result["log_representations"] = log_representations
    result["log_p_spacegroup"] = log_p_spacegroup
    result["mean_representation_log_likelihood"] = samples.mean(axis=0)
    result["std_representation_log_likelihood"] = (
        samples.std(axis=0, ddof=1) if permutation_samples > 1 else 0.0)
    result["log_likelihood"] = log_representations + log_mean_exp + log_p_spacegroup
    result["log_likelihood_elbo"] = (
        log_representations + samples.mean(axis=0) + log_p_spacegroup)
    result["surprisal"] = -result["log_likelihood"]
    result["surprisal_per_site"] = result["surprisal"] / result["n_sites"]
    return result[list(LIKELIHOOD_COLUMNS)]


def records_from_genes(genes: Sequence[dict], trainer) -> Tuple[pd.DataFrame, pd.Series]:
    """Wyckoff records for genes the model can read, plus a reason for each drop.

    Mirrors the front half of ``wyformer-gene-screen``: a gene that is not a
    legal Wyckoff assignment, or that names a token outside the vocabulary, has
    no likelihood under this model rather than a low one.
    """
    from wyckoff_transformer.evaluation.protocol import GeneFingerprinter  # noqa: PLC0415

    fingerprinter = GeneFingerprinter()
    reasons = pd.Series(pd.NA, index=pd.RangeIndex(len(genes), name="index"), dtype=object)
    rows = []
    for index, gene in enumerate(genes):
        try:
            record = fingerprinter.record(gene)
        except (KeyError, TypeError, ValueError) as error:
            reasons.iloc[index] = f"{type(error).__name__}: {error}"
            continue
        # `build_tokenised_prediction_tensors` reads the composition counter for the
        # tokeniser's `counters` field. A repeated element occupies several sites, so
        # its occupancies are summed rather than overwritten.
        composition: Dict = {}
        for element, count in zip(record["elements"], record["multiplicity"]):
            composition[element] = composition.get(element, 0) + count
        record["composition"] = composition
        record["source_index"] = index
        rows.append(record)
    if not rows:
        return pd.DataFrame(), reasons
    frame = pd.DataFrame.from_records(rows).set_index("source_index")
    frame.index.name = "index"
    supported, dropped = filter_supported_tokens(frame, trainer)
    for index in dropped:
        reasons.loc[index] = "Gene is outside the model's vocabulary"
    return supported, reasons
