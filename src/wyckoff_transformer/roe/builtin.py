"""Adapters that fill a mode's slots with code that already exists.

Nothing here implements a screen, a generator, an energy model or a relaxation.
Each class is a thin, declarative wrapper whose job is to say what it needs, run
the existing function, and write its verdict into the cohort in the one format
the accounting understands.  When a component looks like it is doing real work,
that work belongs in the module it wraps.
"""
from __future__ import annotations

import json
import logging
from argparse import Namespace
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd
import torch

from wyckoff_transformer.evaluation.protocol import (
    DEFAULT_REFERENCE_CACHE,
    DEFAULT_REFERENCE_SPLITS,
    GeneFingerprinter,
    default_fingerprint_cache,
    load_genes,
    load_reference_fingerprints,
    reference_identity,
    screen_genes,
)
from wyckoff_transformer.paths import resolve_store_path
from wyckoff_transformer.roe.cohort import Cohort, timed

logger = logging.getLogger(__name__)

#: Drawn per gene wanted, before formal validity removes some.  1.1 is
#: ``wyformer-generate``'s own ratio (1100 attempts for 1000 genes).
DEFAULT_OVERSAMPLE = 1.1


# --------------------------------------------------------------------------- #
# Always applied: formal validity
# --------------------------------------------------------------------------- #
def apply_validity(cohort: Cohort) -> None:
    """Drop genes that are not legal Wyckoff assignments.  Every mode does this.

    It is not a filter slot and no mode gets credit for it: a gene naming a
    Wyckoff letter its space group does not have is not a cheap reconstruction
    that a careless mode would attempt and a careful one would skip -- it is one
    PyXtal cannot attempt at all.  Charging it to ``broadside`` would credit the
    other modes with avoiding a cost that does not exist.
    """
    fingerprinter = GeneFingerprinter()
    valid, reasons = [], {}
    with timed() as elapsed:
        for index in cohort.kept_indices():
            try:
                fingerprinter.fingerprint(cohort.genes[index])
            except Exception as exc:  # noqa: BLE001 - the mappings raise several types
                reasons[index] = f"{type(exc).__name__}: {exc}"
            else:
                valid.append(index)
    cohort.record(
        "validity", "validity",
        passed=valid,
        columns={"invalid_reason": pd.Series(reasons, dtype="object")},
        seconds=elapsed(),
        detail={"invalid": len(reasons)},
    )


# --------------------------------------------------------------------------- #
# Samplers
# --------------------------------------------------------------------------- #
class SystemPriorSampler:
    """Draws (chemical system, space group) per structure from a built prior.

    Wraps :class:`~wyckoff_transformer.system_prior.SystemSpaceGroupPrior`; every
    knob is that sampler's own, documented in ``docs/chemical_system_sampler.md``.
    """

    name = "system_prior"

    def __init__(
        self,
        prior_path: Path,
        *,
        required: Optional[str] = None,
        allowed: Optional[str] = None,
        novel_fraction: Optional[float] = None,
        system_temperature: float = 1.0,
        sg_temperature: float = 1.0,
        min_arity: Optional[int] = None,
        max_arity: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> None:
        from wyckoff_transformer.system_prior import SystemSpaceGroupPrior

        self.prior_path = Path(prior_path)
        self.prior = SystemSpaceGroupPrior.load(self.prior_path)
        self.query = {
            "required": required, "allowed": allowed, "novel_fraction": novel_fraction,
            "system_temperature": system_temperature, "sg_temperature": sg_temperature,
            "min_arity": min_arity, "max_arity": max_arity, "seed": seed,
        }
        # One generator for the sampler's lifetime: re-seeding per call would hand
        # every top-up round the identical plan.
        self._rng = np.random.default_rng(seed)

    def plan(self, n_structures: int):
        return self.prior.sample(
            n_structures,
            required=self.query["required"],
            allowed=self.query["allowed"],
            novel_fraction=self.query["novel_fraction"],
            system_temperature=self.query["system_temperature"],
            sg_temperature=self.query["sg_temperature"],
            min_arity=self.query["min_arity"],
            max_arity=self.query["max_arity"],
            rng=self._rng,
        )

    def describe(self) -> dict:
        return {"component": self.name, "prior": str(self.prior_path), **self.query}


class SubsystemClosureSampler:
    """Aims at a set of target systems and requests every subsystem of each too.

    Exploring the A-B-C hull means generating A-B, A-C and B-C as well: a ternary
    is only below the hull if it is below the binaries, and the binaries a
    campaign could find belong to that hull as much as the ones the archive holds.
    The targets are drawn once, at construction, from the prior (or named);
    every plan then spreads its rows over the targets' closures with
    :meth:`~wyckoff_transformer.system_prior.SystemSpaceGroupPrior.closure_plan`.
    """

    name = "subsystem_closure"

    def __init__(
        self,
        prior_path: Path,
        *,
        targets: Optional[Sequence[str]] = None,
        n_targets: Optional[int] = None,
        target_arity: int = 3,
        min_arity: int = 2,
        target_share: float = 0.5,
        required: Optional[str] = None,
        allowed: Optional[str] = None,
        novel_fraction: Optional[float] = None,
        system_temperature: float = 1.0,
        sg_temperature: float = 1.0,
        seed: Optional[int] = None,
    ) -> None:
        from wyckoff_transformer.system_prior import SYSTEM_DELIMITER, SystemSpaceGroupPrior

        if (targets is None) == (n_targets is None):
            raise ValueError("Name the targets or ask for n_targets of them, not both")
        self.prior_path = Path(prior_path)
        self.prior = SystemSpaceGroupPrior.load(self.prior_path)
        self._rng = np.random.default_rng(seed)
        if targets is None:
            tokens = self.prior.sample_targets(
                n_targets, target_arity, required, allowed,
                novel_fraction=novel_fraction, system_temperature=system_temperature,
                rng=self._rng)
        else:
            tokens = [self.prior.parse_elements(target) for target in targets]
        self.targets = [
            SYSTEM_DELIMITER.join(self.prior.element_symbols[t] for t in system)
            for system in tokens]
        self.query = {
            "targets": self.targets, "n_targets": len(self.targets),
            "target_arity": target_arity, "min_arity": min_arity,
            "target_share": target_share, "required": required, "allowed": allowed,
            "novel_fraction": novel_fraction, "system_temperature": system_temperature,
            "sg_temperature": sg_temperature, "seed": seed,
        }
        logger.info("Aiming at %d targets: %s", len(self.targets), ", ".join(self.targets))

    def plan(self, n_structures: int):
        return self.prior.closure_plan(
            self.targets, n_structures,
            min_arity=self.query["min_arity"],
            target_share=self.query["target_share"],
            sg_temperature=self.query["sg_temperature"],
            rng=self._rng,
        )

    def describe(self) -> dict:
        return {"component": self.name, "prior": str(self.prior_path), **self.query}


class PlanFileSampler:
    """Replays a plan ``wyformer-system-prior sample`` already wrote.

    A campaign that has to be reproduced exactly, or split across machines, is
    reproduced from its plan rather than from the prior and a seed: the plan is
    the record of what was actually asked for.

    ``vocabulary`` is checked against the one the plan carries; omitted, the
    plan's own is used, which is what re-filtering a pool drawn from it needs.
    """

    name = "plan_file"

    def __init__(self, plan_path: Path, vocabulary: Optional[Sequence[str]] = None) -> None:
        from wyckoff_transformer.system_prior import SystemDraws

        self.plan_path = Path(plan_path)
        with open(self.plan_path, "rt", encoding="utf-8") as handle:
            manifest = json.load(handle)
        self.query = manifest.get("query", {})
        self.draws = SystemDraws.from_manifest(
            manifest, list(vocabulary) if vocabulary is not None else None)

    def plan(self, n_structures: int):
        if n_structures != len(self.draws):
            logger.warning(
                "The plan at %s holds %d structures; generating that many rather than %d.",
                self.plan_path, len(self.draws), n_structures)
        return self.draws

    def describe(self) -> dict:
        return {"component": self.name, "plan": str(self.plan_path), "rows": len(self.draws),
                "query": self.query}


# --------------------------------------------------------------------------- #
# Gene sources
# --------------------------------------------------------------------------- #
class WyFormerGeneSource:
    """Genes drawn from a loaded WyFormer checkpoint.

    The generation itself is ``WyckoffTrainer.generate_structures``, reached
    through the same module-level helpers ``wyformer-generate`` uses, so that a
    mode and the CLI cannot drift into drawing differently.
    """

    name = "wyformer"

    def __init__(
        self,
        trainer,
        *,
        condition_values: Optional[dict] = None,
        chemical_system: Optional[str] = None,
        space_groups: Optional[Sequence[int]] = None,
        required_elements: Optional[str] = None,
        allowed_elements: Optional[str] = None,
        temperature: float = 1.0,
        oversample: float = DEFAULT_OVERSAMPLE,
        allow_fewer: bool = False,
        device=None,
        model_id: Optional[str] = None,
    ) -> None:
        self.trainer = trainer
        self.condition_values = dict(condition_values) if condition_values else None
        self.chemical_system = chemical_system
        self.space_groups = list(space_groups) if space_groups else None
        self.required_elements = required_elements
        self.allowed_elements = allowed_elements
        self.temperature = temperature
        self.oversample = oversample
        self.allow_fewer = allow_fewer
        self.device = device if device is not None else trainer.device
        self.model_id = model_id
        self.formal_gene_validity: Optional[float] = None
        if chemical_system is not None and not trainer.chemical_system_conditioning:
            raise ValueError(
                "A chemical system was named, but this checkpoint was not trained with "
                "chemical_system_conditioning. Pass allowed_elements to mask an "
                "unconditioned model instead.")

    @property
    def requires_plan(self) -> bool:
        """A chemical-system-conditioned checkpoint asked for no fixed system needs one.

        Such a model has no unconditional mode: something has to say which
        elements each row is for, and leaving it to the training distribution
        would make the cohort a sample of the corpus rather than of a campaign.
        """
        return bool(
            getattr(self.trainer, "chemical_system_conditioning", False)
            and self.chemical_system is None)

    def attempts(self, n_structures: int) -> int:
        """Draws needed to end up with *n* genes, since formal validity removes some."""
        return max(n_structures + 1, int(round(n_structures * self.oversample)))

    def draw(self, n_structures: int, plan: Optional[Any] = None) -> list[dict]:
        from wyckoff_transformer.chemical_system import parse_chemical_system
        from wyckoff_transformer.cli.generate import (
            chemical_system_vector_for_generation,
            keep_required_elements,
            prepare_start_tensor_for_space_groups,
        )

        trainer = self.trainer
        attempted = len(plan) if plan is not None else self.attempts(n_structures)

        cond = None
        if self.condition_values:
            cond = trainer.build_condition_from_values(
                self.condition_values, attempted, device=self.device)

        system_cond = None
        element_mask = None
        start_tensor = None
        allowed_elements = self.allowed_elements

        if plan is not None:
            elements_tokeniser = trainer.tokenisers["elements"]
            vocabulary = [str(symbol) for symbol in elements_tokeniser.to_token]
            plan_vocabulary = list(getattr(plan, "element_symbols", vocabulary))
            if plan_vocabulary != vocabulary:
                raise ValueError(
                    "The plan was drawn over a different element vocabulary than the "
                    "checkpoint knows; they have to come from the same dataset, or a "
                    "system would decode into different elements.")
            system_cond = plan.conditioning_block(len(elements_tokeniser), device=self.device)
            start_tensor = plan.start_tensor(
                trainer.tokenisers[trainer.start_name], trainer.model.start_type,
                device=self.device)
            element_mask = plan.element_mask(
                len(elements_tokeniser), stop_token=elements_tokeniser.stop_token,
                device=self.device)
            allowed_elements = None
        else:
            if self.chemical_system is not None:
                elements_tokeniser = trainer.tokenisers["elements"]
                symbols, _ = parse_chemical_system(self.chemical_system, elements_tokeniser)
                system_cond = chemical_system_vector_for_generation(
                    self.chemical_system, elements_tokeniser, attempted, self.device)
                if allowed_elements is None:
                    allowed_elements = "-".join(symbols)
            if self.space_groups:
                start_tensor = prepare_start_tensor_for_space_groups(
                    trainer, self.space_groups, attempted)

        use_element_constraints = plan is None and (
            self.required_elements is not None or allowed_elements is not None)

        genes = trainer.generate_structures(
            n_structures=attempted,
            calibrate=False,
            temperature=self.temperature,
            start_tensor=start_tensor,
            cond=cond,
            composition_cond=system_cond,
            required_element_set=self.required_elements if use_element_constraints else None,
            allowed_element_set=allowed_elements if allowed_elements is not None else "all",
            allowed_element_mask=element_mask,
        )
        self.formal_gene_validity = len(genes) / attempted if attempted else None
        if plan is not None:
            genes = keep_required_elements(genes, plan.query.get("required"))

        if len(genes) < n_structures and not self.allow_fewer:
            raise ValueError(
                f"{len(genes)} of {attempted} draws were formally valid, short of the "
                f"{n_structures} asked for. Raise the oversample ratio, or allow fewer.")
        return genes[:n_structures]

    def describe(self) -> dict:
        return {
            "component": self.name,
            "model": self.model_id,
            "condition": self.condition_values,
            "chemical_system": self.chemical_system,
            "space_groups": self.space_groups,
            "required_elements": self.required_elements,
            "allowed_elements": self.allowed_elements,
            "temperature": self.temperature,
            "oversample": self.oversample,
            "formal_gene_validity": self.formal_gene_validity,
        }


class GeneFileSource:
    """A pool of genes already on disk, handed out in order.

    Two things it makes possible.  A cohort drawn once can be re-filtered
    without a GPU, which is how a mode is re-scored after a change to a filter.
    And several modes can be run against the *same* pool, which turns the
    comparison between them into a paired one: they then differ in what they
    select and in nothing else, with no sampling noise between the arms.

    Each call hands out the next unread slice, so a mode topping up to a budget
    consumes exactly as much of the pool as it needs -- and ``n_sampled`` ends up
    being what that mode actually spent on generation, which is the quantity the
    per-sampled-gene rates are against.
    """

    name = "gene_file"
    requires_plan = False

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._genes: Optional[list] = None
        self.consumed = 0

    def genes(self) -> list:
        if self._genes is None:
            self._genes = load_genes(self.path)
            logger.info("Pool at %s holds %d genes", self.path, len(self._genes))
        return self._genes

    def attempts(self, n_structures: int) -> int:
        return n_structures

    def draw(self, n_structures: int, plan: Optional[Any] = None) -> list[dict]:
        genes = self.genes()
        take = genes[self.consumed: self.consumed + n_structures]
        self.consumed += len(take)
        if not take:
            logger.warning("The pool at %s is exhausted after %d genes",
                           self.path, self.consumed)
        return take

    def describe(self) -> dict:
        return {"component": self.name, "path": str(self.path),
                "pool": None if self._genes is None else len(self._genes),
                "consumed": self.consumed}


# --------------------------------------------------------------------------- #
# Filters
# --------------------------------------------------------------------------- #
class NoveltyUniquenessScreen:
    """Uniqueness and novelty by augmented Wyckoff fingerprint.

    Two interchangeable backends answering the same question:

    ``python``
        :func:`~wyckoff_transformer.evaluation.protocol.screen_genes` against the
        reference fingerprint set -- 4.8M nested frozensets, ~17 GB resident.
    ``tensor``
        the 128-bit canonical keys of
        :mod:`wyckoff_transformer.evaluation.gene_hash`: uniqueness by
        ``torch.unique``, novelty by ``torch.searchsorted`` against a sorted
        table two orders of magnitude smaller.  Exactly equivalent, and pinned
        to be so by ``tests/test_gene_hash.py``.

    Either way the cohort columns are identical -- which is why this is one
    class with a backend rather than two components -- and which backend ran is
    recorded in the manifest.

    Run over the genes still kept rather than over the whole cohort.  That
    matters when another filter ran first: the representative of a fingerprint
    class has to be a gene that survived, or a class would be dropped because
    the member that happened to come first did not.

    ``novelty`` can be turned off to get uniqueness alone, which is the honest
    arm for a campaign whose reference archive is not the one it will be judged
    against.
    """

    name = "gene_screen"
    slot = "screen"
    provides = ("unique_representative", "duplicates", "gene_novel")
    requires = ()

    BACKENDS = ("python", "tensor")

    def __init__(
        self,
        *,
        reference_cache: Path = DEFAULT_REFERENCE_CACHE,
        reference_splits: Sequence[str] = DEFAULT_REFERENCE_SPLITS,
        fingerprint_cache: Optional[Path] = None,
        key_table_path: Optional[Path] = None,
        backend: str = "python",
        uniqueness: bool = True,
        novelty: bool = True,
    ) -> None:
        if not uniqueness and not novelty:
            raise ValueError(
                "A screen that checks neither uniqueness nor novelty does nothing; "
                "leave the slot out of the mode instead.")
        if backend not in self.BACKENDS:
            raise ValueError(f"backend is one of {self.BACKENDS}, got {backend!r}")
        self.reference_cache = Path(reference_cache)
        self.reference_splits = tuple(reference_splits)
        self.fingerprint_cache = fingerprint_cache
        self.key_table_path = key_table_path
        self.backend = backend
        self.uniqueness = uniqueness
        self.novelty = novelty
        self._reference: Optional[set] = None
        self._table = None

    # ------------------------------------------------------------------ #
    def reference(self) -> set:
        """The fingerprint set, for the ``python`` backend."""
        if self._reference is None:
            if not self.novelty:
                self._reference = set()
            else:
                self._reference = load_reference_fingerprints(
                    self.reference_cache,
                    self.reference_splits,
                    fingerprint_cache=self.fingerprint_cache
                    or default_fingerprint_cache(self.reference_cache, self.reference_splits),
                )
        return self._reference

    def key_table(self):
        """The sorted key table, for the ``tensor`` backend.  Built once if absent."""
        from wyckoff_transformer.evaluation.gene_hash import (
            GeneKeyTable,
            build_reference_table,
            default_key_table_path,
        )

        if self._table is None:
            if not self.novelty:
                self._table = GeneKeyTable.from_keys(torch.empty((0, 2), dtype=torch.int64))
            else:
                path = Path(self.key_table_path or default_key_table_path(
                    resolve_store_path(self.reference_cache), self.reference_splits))
                if path.is_file():
                    self._table = GeneKeyTable.load(path)
                    logger.info("Loaded %d reference gene keys from %s", len(self._table), path)
                else:
                    logger.info("No key table at %s; building it from the reference", path)
                    self._table = build_reference_table(
                        self.reference_cache, self.reference_splits, output=path)
        return self._table

    # ------------------------------------------------------------------ #
    def _verdicts_python(self, genes: list) -> tuple:
        """``(representative -> count, local -> novel, valid locals, reference size)``."""
        reference = self.reference()
        screen = screen_genes(genes, reference, GeneFingerprinter())
        novel_fingerprints = {screen.fingerprint[local] for local in screen.novel}
        gene_novel = {
            local: screen.fingerprint[local] in novel_fingerprints for local in screen.valid}
        return dict(screen.counts), gene_novel, list(screen.valid), len(reference)

    def _verdicts_tensor(self, genes: list) -> tuple:
        from wyckoff_transformer.evaluation.gene_hash import (
            gene_keys,
            unique_representatives,
        )

        fingerprinter = GeneFingerprinter()
        valid, records = [], []
        for local, gene in enumerate(genes):
            try:
                records.append(fingerprinter.record(gene))
            except Exception as exc:  # noqa: BLE001 - mirrors screen_genes
                logger.debug("Gene %d is not a legal assignment (%s)", local, exc)
            else:
                valid.append(local)
        keys = gene_keys(records)
        representatives, counts = unique_representatives(keys)
        counts_by_local = {
            valid[int(row)]: int(count) for row, count in zip(representatives, counts)}

        known = self.key_table().contains(keys)
        gene_novel = {valid[row]: not bool(known[row]) for row in range(len(valid))}
        return counts_by_local, gene_novel, valid, len(self.key_table())

    # ------------------------------------------------------------------ #
    def apply(self, cohort: Cohort) -> None:
        indices = cohort.kept_indices()
        genes = [cohort.genes[i] for i in indices]
        with timed() as elapsed:
            if self.backend == "tensor":
                counts, novel_by_local, valid_locals, reference_size = self._verdicts_tensor(genes)
            else:
                counts, novel_by_local, valid_locals, reference_size = self._verdicts_python(genes)

        representatives = {indices[local]: count for local, count in counts.items()}
        # Novelty is a property of the fingerprint, so it belongs to every gene in a
        # class and not only to the representative: without a uniqueness screen the
        # duplicates are kept, and each has to carry its own verdict.
        gene_novel = {indices[local]: is_novel for local, is_novel in novel_by_local.items()}
        novel = {index for index, is_novel in gene_novel.items() if is_novel}

        survivors = set(representatives) if self.uniqueness else {
            indices[local] for local in valid_locals}
        if self.novelty:
            survivors &= novel

        columns = {
            "unique_representative": pd.Series(
                {index: True for index in representatives}, dtype="boolean"),
            "duplicates": pd.Series(representatives, dtype="Int64"),
            "gene_novel": pd.Series(gene_novel, dtype="boolean"),
        }
        # Only a uniqueness screen may move weight: it is the one that knows a
        # representative now stands for the draws that collapsed onto it.
        weights = None
        if self.uniqueness:
            weights = {index: 0 for index in indices}
            weights.update(representatives)

        cohort.record(
            "filter", self.name,
            passed=sorted(survivors),
            columns=columns,
            weights=weights,
            seconds=elapsed(),
            detail={
                "backend": self.backend,
                "uniqueness": self.uniqueness,
                "novelty": self.novelty,
                "reference": reference_identity(self.reference_cache, self.reference_splits)
                if self.novelty else None,
                "reference_size": reference_size if self.novelty else 0,
                "unique_genes": len(representatives),
                "gene_novel": len(novel),
                "invalid": len(indices) - len(valid_locals),
            },
        )

    def describe(self) -> dict:
        return {
            "component": self.name,
            "backend": self.backend,
            "uniqueness": self.uniqueness,
            "novelty": self.novelty,
            "reference_cache": str(self.reference_cache),
            "reference_splits": list(self.reference_splits),
        }


class PredictedHullFilter:
    """Selects genes by their predicted energy above the hull at their own composition.

    Wraps :func:`~wyckoff_transformer.cli.gene_screen.score_genes`, which predicts
    the gene's attainable formation energy with ``max_force`` pinned at zero --
    the gene has not been relaxed -- and subtracts the reference hull at its
    formula.  The result is a *predicted* ``e_hull``, and it is not the MLIP
    ``e_above_hull`` the protocol later scores with (``docs/e_hull_definitions.md``).

    Two ways to select on it, and which one is right depends on what is fixed:

    ``threshold``
        keep everything at or below ``margin``.  The natural reading of "predicted
        below the hull", and the one to use when the question is how many such
        genes a generator produces.
    ``top``
        keep the ``top_k`` lowest.  The one to use when the *reconstruction
        budget* is fixed, which is how a mode is actually run: a threshold at 0
        keeps about 5% of novel genes, so filling a budget of 1000 from it would
        take tens of thousands of draws.  Taking the best ``k`` spends the budget
        on the best the pool has, and the manifest records what predicted
        ``e_hull`` the cut actually landed at.

    **A gene whose composition the reference hull does not cover is kept, not
    dropped** (``on_missing_hull='keep'``).  There is no hull to compare it with,
    and a novel composition is exactly what a discovery campaign is looking for;
    dropping it would make this filter select against novel chemistry by
    construction, which is the failure ``docs/generative_novelty_screen.md``
    measured for an energy ranker left to itself.  Under ``top`` an undecided
    gene is ranked last, so it is kept only if the budget is not filled without
    it.

    ``rank``
        keep everything (under ``margin`` when one is given) and leave the cut to
        the budget, which :meth:`~wyckoff_transformer.roe.plan.Engagement.run`
        applies after *every* filter by :attr:`rank_column`.  The one to use when
        another filter runs after this one: a ``top`` cut taken before a screen
        spends budget on duplicates and known genes the screen then removes.

    **Which energy and which hull** (``energy`` and ``hull``):

    * ``energy='raw'`` is the regressor's prediction.  ``'corrected'`` replaces
      it, for a gene the archive holds, with that gene's DFT energy, and for
      every other gene subtracts the regressor's local residual
      (:mod:`wyckoff_transformer.gene_energy_residuals`).
    * ``hull='reference'`` compares each gene with the DFT hull alone.
      ``'joint'`` puts the candidates on the hull too and scores each against
      the hull of everything else
      (:func:`~wyckoff_transformer.formula_energy.joint_hull.joint_hull_scores`),
      so candidates compete with each other and a ternary is measured against
      the binaries the same cohort found.

    All four combinations are written as ``predicted_e_hull_<hull>_<energy>`` for
    analysis whenever their inputs were given; ``predicted_e_hull`` is the one
    selected on.
    """

    name = "predicted_hull"
    slot = "energy"
    provides = ("predicted_formation_energy", "hull_energy", "predicted_e_hull")
    requires = ()

    SELECTORS = ("threshold", "top", "rank")
    HULLS = ("reference", "joint")
    ENERGIES = ("raw", "corrected")

    def __init__(
        self,
        regressor,
        reference: pd.DataFrame,
        *,
        margin: Optional[float] = 0.0,
        select: str = "threshold",
        top_k: Optional[int] = None,
        on_missing_hull: str = "keep",
        augmentation_samples: int = 1,
        regressor_id: Optional[str] = None,
        reference_id: Optional[str] = None,
        hull: str = "reference",
        energy: str = "raw",
        correction=None,
        known_genes=None,
        residuals_id: Optional[str] = None,
    ) -> None:
        if on_missing_hull not in ("keep", "drop"):
            raise ValueError("on_missing_hull is 'keep' or 'drop'")
        if select not in self.SELECTORS:
            raise ValueError(f"select is one of {self.SELECTORS}, got {select!r}")
        if select == "top" and not top_k:
            raise ValueError("select='top' needs top_k, the reconstruction budget")
        if select != "rank" and margin is None:
            raise ValueError(f"select={select!r} needs a margin")
        if hull not in self.HULLS:
            raise ValueError(f"hull is one of {self.HULLS}, got {hull!r}")
        if energy not in self.ENERGIES:
            raise ValueError(f"energy is one of {self.ENERGIES}, got {energy!r}")
        if energy == "corrected" and correction is None and known_genes is None:
            raise ValueError(
                "energy='corrected' needs a residual correction or the known-gene "
                "energies; without either it is the raw prediction under another name")
        self.hull = hull
        self.energy = energy
        self.correction = correction
        self.known_genes = known_genes
        self.residuals_id = residuals_id
        #: The column the budget ranks on, when the cut is left to it.
        self.rank_column = "predicted_e_hull" if select == "rank" else None
        self.regressor = regressor
        self.reference = reference
        self.margin = margin
        self.select = select
        self.top_k = top_k
        self.on_missing_hull = on_missing_hull
        self.augmentation_samples = augmentation_samples
        self.regressor_id = regressor_id
        self.reference_id = reference_id
        self._lookup = None
        #: Gene key -> its scored row.  Scoring is per-gene and deterministic, so
        #: a cohort topped up over several rounds must not pay for the genes it
        #: already scored -- that would make the loop quadratic in the pool.
        self._scored: dict = {}

    def hull_lookup(self):
        from wyckoff_transformer.formula_energy.screen import HullLookup

        if self._lookup is None:
            logger.info("Indexing the PBE reference by element for the hull lookup")
            self._lookup = HullLookup(self.reference)
        return self._lookup

    def _score(self, cohort: Cohort, indices: list) -> pd.DataFrame:
        """Score the genes not already scored, and return rows for all of *indices*."""
        from wyckoff_transformer.cli.gene_screen import score_genes

        keys = self._gene_keys(cohort, indices)
        fresh = [index for index in indices
                 if keys[index] is None or keys[index] not in self._scored]
        scored = None
        if fresh:
            scored = score_genes(
                [cohort.genes[i] for i in fresh],
                self.regressor,
                self.reference,
                augmentation_samples=self.augmentation_samples,
                hull_lookup=self.hull_lookup(),
            ).sort_index()
            scored.index = pd.Index(fresh, name="index")
            for index in fresh:
                if keys[index] is not None:
                    self._scored[keys[index]] = scored.loc[index]
        logger.info("Scored %d new genes; %d came from the cache", len(fresh),
                    len(indices) - len(fresh))

        rows = []
        for index in indices:
            key = keys[index]
            if key is not None and key in self._scored:
                rows.append(self._scored[key])
            elif scored is not None:
                # An unkeyable gene: scored every time, never cached, because
                # there is nothing to cache it under.
                rows.append(scored.loc[index])
            else:
                raise RuntimeError(
                    f"Gene {index} has neither a cached score nor a fresh one; this "
                    "cannot happen unless the key cache and the fresh list disagree.")
        frame = pd.DataFrame(rows)
        frame.index = pd.Index(indices, name="index")
        return frame

    @staticmethod
    def _gene_keys(cohort: Cohort, indices: list) -> dict:
        """Index -> 128-bit gene key, or ``None`` for a gene that has none."""
        from wyckoff_transformer.evaluation.gene_hash import gene_key
        from wyckoff_transformer.evaluation.protocol import GeneFingerprinter

        fingerprinter = GeneFingerprinter()
        keys = {}
        for index in indices:
            try:
                keys[index] = gene_key(fingerprinter.record(cohort.genes[index]))
            except Exception:  # noqa: BLE001 - an invalid gene is scored as unusable
                keys[index] = None
        return keys

    def _variants(self, cohort: Cohort, indices: list, scored: pd.DataFrame) -> dict:
        """Every (hull, energy) predicted ``e_hull`` this filter has the inputs for."""
        from wyckoff_transformer.gene_energy_residuals import chemical_system

        columns = {"predicted_e_hull_reference_raw": scored["score"].astype(float)}
        energies = {"raw": scored["predicted_formation_energy"].astype(float)}
        needs_keys = self.known_genes is not None or self.hull == "joint"
        keys = self._gene_keys(cohort, indices) if needs_keys else {}

        if self.correction is not None or self.known_genes is not None:
            from pymatgen.core.composition import Composition

            corrected = energies["raw"].copy()
            if self.correction is not None:
                systems = {
                    formula: chemical_system(str(e) for e in Composition(formula).elements)
                    for formula in scored["formula"].dropna().unique()}
                has_formula = scored["formula"].notna()
                bias = pd.Series(np.nan, index=scored.index)
                bias[has_formula] = self.correction.biases(
                    scored.loc[has_formula, "formula"].map(systems))
                columns["residual_correction"] = bias
                corrected = corrected - bias.fillna(0.0)
            if self.known_genes is not None:
                rows = [index for index in indices if keys[index] is not None]
                dft = pd.Series(np.nan, index=scored.index)
                if rows:
                    dft.loc[rows] = self.known_genes.lookup([keys[i] for i in rows])
                columns["known_dft_formation_energy"] = dft
                corrected = dft.where(dft.notna(), corrected)
            corrected = corrected.where(scored["predicted_formation_energy"].notna())
            energies["corrected"] = corrected
            columns["corrected_formation_energy"] = corrected
            columns["predicted_e_hull_reference_corrected"] = (
                corrected - scored["hull_energy"].astype(float))

        if self.hull == "joint":
            from wyckoff_transformer.formula_energy.joint_hull import joint_hull_scores

            representative = {}
            for index in indices:
                if keys[index] is not None:
                    representative.setdefault(keys[index], index)
            reps = list(representative.values())
            for name, energy in energies.items():
                candidates = pd.DataFrame(
                    {"formula": scored.loc[reps, "formula"], "energy": energy.loc[reps]})
                joint = joint_hull_scores(candidates, self.hull_lookup())
                by_key = {keys[i]: joint.at[i, "joint_e_hull"] for i in reps}
                columns[f"predicted_e_hull_joint_{name}"] = pd.Series(
                    [by_key.get(keys[i], np.nan) if keys[i] is not None else np.nan
                     for i in indices], index=scored.index, dtype=float)
        return columns

    def apply(self, cohort: Cohort) -> None:
        indices = cohort.kept_indices()
        with timed() as elapsed:
            scored = self._score(cohort, indices)
            variants = self._variants(cohort, indices, scored)

        values = variants[f"predicted_e_hull_{self.hull}_{self.energy}"]
        decided = values.notna()
        below = decided & (values <= (self.margin if self.margin is not None else 0.0))
        undecided = ~decided

        if self.select == "rank":
            keep = decided if self.margin is None else below
            if self.on_missing_hull == "keep":
                keep = keep | undecided
            cut = self.margin
        elif self.select == "top":
            # Undecided genes rank last: they are kept only if the budget is not
            # filled without them, and dropped outright if it is.
            order = values.fillna(float("inf")).sort_values(kind="stable")
            chosen = order.index[: self.top_k]
            keep = pd.Series(scored.index.isin(chosen), index=scored.index)
            if self.on_missing_hull == "drop":
                keep &= decided
            cut = float(values.loc[keep & decided].max()) if bool((keep & decided).any()) else None
        else:
            keep = below | (undecided if self.on_missing_hull == "keep" else False)
            cut = self.margin

        cohort.record(
            "filter", self.name,
            passed=scored.index[keep].tolist(),
            columns={
                "formula": scored["formula"],
                "predicted_formation_energy": scored["predicted_formation_energy"],
                "hull_energy": scored["hull_energy"],
                # The quantity's established name. The column the wrapped screen
                # calls `score` is a predicted e_hull and is named as one here.
                "predicted_e_hull": values,
                "hull_undecided": undecided,
                **variants,
            },
            seconds=elapsed(),
            detail={
                "select": self.select,
                "margin": self.margin,
                "top_k": self.top_k,
                "selection_cut_e_hull": cut,
                "on_missing_hull": self.on_missing_hull,
                "hull": self.hull,
                "energy": self.energy,
                "decided": int(decided.sum()),
                "below_hull": int(below.sum()),
                "undecided": int(undecided.sum()),
                "regressor": self.regressor_id,
                "reference": self.reference_id,
                "residuals": self.residuals_id,
                "correction": (self.correction.describe()
                               if self.correction is not None else None),
            },
        )

    def describe(self) -> dict:
        return {
            "component": self.name,
            "regressor": self.regressor_id,
            "reference": self.reference_id,
            "select": self.select,
            "margin": self.margin,
            "top_k": self.top_k,
            "on_missing_hull": self.on_missing_hull,
            "hull": self.hull,
            "energy": self.energy,
            "residuals": self.residuals_id,
        }


class SurprisalBandFilter:
    """Keeps a band of the generator's own log-density.

    The reference-free novelty estimator of ``docs/generative_novelty_screen.md``.
    Its measured use is a *band*, not a direction: ranking on surprisal either way
    was worse than random there, while keeping a band and ranking the survivors on
    energy was the best lookup-free arm.  So the filter takes two quantiles and
    keeps what falls between them.

    Refused for a chemical-system-conditioned generator, by
    :func:`~wyckoff_transformer.gene_likelihood.score_gene_likelihood` itself:
    each gene would be scored under its own conditioning and the numbers would
    not be comparable across genes.
    """

    name = "surprisal_band"
    slot = "surprisal"
    provides = ("surprisal", "surprisal_quantile")
    requires = ()

    def __init__(
        self,
        trainer,
        *,
        lower_quantile: float = 0.0,
        upper_quantile: float = 0.3,
        condition_values: Optional[dict] = None,
        permutation_samples: int = 32,
        seed: int = 0,
        batch_size: Optional[int] = None,
    ) -> None:
        if not 0.0 <= lower_quantile < upper_quantile <= 1.0:
            raise ValueError("The band needs 0 <= lower < upper <= 1")
        self.trainer = trainer
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile
        self.condition_values = dict(condition_values) if condition_values else None
        self.permutation_samples = permutation_samples
        self.seed = seed
        self.batch_size = batch_size

    def apply(self, cohort: Cohort) -> None:
        from wyckoff_transformer.gene_likelihood import (
            records_from_genes,
            score_gene_likelihood,
        )

        indices = cohort.kept_indices()
        with timed() as elapsed:
            records, _ = records_from_genes([cohort.genes[i] for i in indices], self.trainer)
            cond = None
            if self.condition_values:
                cond = self.trainer.build_condition_from_values(
                    self.condition_values, len(records), device=self.trainer.device)
            scored = score_gene_likelihood(
                records, self.trainer, cond=cond,
                permutation_samples=self.permutation_samples,
                seed=self.seed, batch_size=self.batch_size)

        column = "surprisal" if "surprisal" in scored else scored.columns[0]
        surprisal = pd.Series(
            scored[column].to_numpy(),
            index=pd.Index([indices[local] for local in range(len(scored))], name="index"))
        quantile = surprisal.rank(pct=True)
        keep = (quantile > self.lower_quantile) & (quantile <= self.upper_quantile)

        cohort.record(
            "filter", self.name,
            passed=surprisal.index[keep].tolist(),
            columns={"surprisal": surprisal, "surprisal_quantile": quantile},
            seconds=elapsed(),
            detail={
                "lower_quantile": self.lower_quantile,
                "upper_quantile": self.upper_quantile,
                "permutation_samples": self.permutation_samples,
            },
        )

    def describe(self) -> dict:
        return {
            "component": self.name,
            "band": [self.lower_quantile, self.upper_quantile],
            "permutation_samples": self.permutation_samples,
            "condition": self.condition_values,
        }


# --------------------------------------------------------------------------- #
# Reconstructors
# --------------------------------------------------------------------------- #
class CrySPRReconstructor:
    """Runs the de novo ranking protocol on the genes a mode kept.

    Delegates to :mod:`wyckoff_transformer.cli.protocol` rather than driving
    PyXtal and the MLIP here.  Everything that makes that module long -- the
    worker pool, the device faults, the trial schedule, the lineage checks, the
    two scoring tracks -- is what a reconstruction needs to survive a night on
    shared hardware, and a second implementation of it would be a second thing
    to get wrong.

    The protocol re-runs its own screen on the filtered gene file. That is not
    wasted: it is the audit, computed from the gene file as it was handed over,
    and ``score`` refuses to mix its verdict with a novelty reference other than
    its own.
    """

    name = "cryspr"

    def __init__(self, protocol_argv: Sequence[str] = (), stages: Sequence[str] = ("all",)) -> None:
        self.protocol_argv = list(protocol_argv)
        self.stages = tuple(stages)

    def _args(self, gene_file: Path, output_dir: Path) -> Namespace:
        from wyckoff_transformer.cli.protocol import build_parser

        argv = [str(gene_file), "--output-dir", str(output_dir), *self.protocol_argv]
        return build_parser().parse_args(argv)

    def reconstruct(self, cohort: Cohort, output_dir: Path) -> dict:
        from wyckoff_transformer.cli import protocol as protocol_cli
        from wyckoff_transformer.roe.cohort import ENGAGED_GENES_FILE

        protocol_dir = Path(output_dir) / "protocol"
        protocol_dir.mkdir(parents=True, exist_ok=True)
        args = self._args(Path(output_dir) / ENGAGED_GENES_FILE, protocol_dir)

        stages = protocol_cli.STAGES if self.stages == ("all",) else self.stages
        for stage in stages:
            protocol_cli.run_stage(stage, args)

        funnel_path = protocol_dir / "funnel.json"
        if funnel_path.is_file():
            with open(funnel_path, "rt", encoding="utf-8") as handle:
                return json.load(handle)
        return {}

    def describe(self) -> dict:
        return {"component": self.name, "protocol_args": self.protocol_argv,
                "stages": list(self.stages)}


class DiffCSPReconstructor:
    """DiffCSP++ as a reconstructor.  Not wired up.

    The repository reads DiffCSP++ *output* -- ``evaluation/DiffCSP_to_sites.py``
    turns its structures into Wyckoff records -- but has no path that hands it
    genes and gets structures back.  The slot exists so that adding one is a
    component rather than a change to every mode, and this class refuses loudly
    rather than letting a mode look runnable when it is not.
    """

    name = "diffcsp++"

    def reconstruct(self, cohort: Cohort, output_dir: Path) -> dict:
        raise NotImplementedError(
            "No DiffCSP++ reconstruction path exists in this repository; only the "
            "importer for structures it has already produced "
            "(wyckoff_transformer.evaluation.DiffCSP_to_sites). Write the kept genes "
            "with --no-reconstruct, run DiffCSP++ on them, and score the result.")

    def describe(self) -> dict:
        return {"component": self.name, "implemented": False}
