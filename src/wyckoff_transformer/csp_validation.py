"""Gene-level CSP match rate on a held-out split, logged while a CSP backbone trains.

Validation loss says how well the model predicts the next token; crystal structure prediction
asks whether sampling recovers the right structure, and the two can come apart. A backbone's
space-group head can overfit within a thousand epochs while the site heads are still
improving, and a checkpoint chosen on total loss then serves neither (DiffCSPNew
docs/archive/csp-benchmark-results.md). This module measures the CSP outcome directly.

For a fixed, seeded subset of the split's structures it samples ``samples`` Wyckoff genes per
formula exactly as the CSP benchmark does: the space group from the backbone's start head,
renormalised over the groups that can express the composition, then the gene from
`csp.ConstrainedDecoder` (``strategy="sample"``) in that group. With ``given_cell`` (the
default) the decoder may only finish at the structure's own atom count: by default its
*primitive* count, as the DiffCSP protocol gives it, so that a candidate group's conventional
cell holds that count times the group's lattice points. The conventional count would also tell
the model the ground truth's lattice centring.
It reports, over the subset:

- ``gene_mr@N``: the share of formulas whose first N samples include the structure's own gene,
  compared by the augmentation-invariant fingerprint (`evaluation.protocol.GeneFingerprinter`);
- ``sg_mr@N``: the same for its space group;
- ``decoded``: the share of requested samples that decoding produced.

This is a match rate of *genes*. The structural match rate of the benchmark also needs a
gene-to-structure model, a relaxation and StructureMatcher, which do not belong in a training
loop. A structure is usually recovered from its own gene; some are also recovered from an
inequivalent one.

A backbone conditioned on a scalar (formation energy, say) is given each structure's own
value from the split: this monitors conditional generation, and is more optimistic than a
benchmark that has to choose the value without seeing the structure.

Under DDP every rank evaluates its share of the subset and the counts are summed, so all ranks
have to call `evaluate` together.
"""
from __future__ import annotations

import logging
import time
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

logger = logging.getLogger(__name__)

NS = (1, 5, 20)


@dataclass(frozen=True)
class CSPValidationConfig:
    """``WyckoffTrainer_args.csp_evaluation`` in a config.

    Args:
        period: Epochs between evaluations; a multiple of ``validation_period``, since the
            evaluation runs inside the validation step.
        n_structures: Structures of the split to evaluate on, drawn once with ``seed``.
        samples: Genes sampled per formula; the largest N reported.
        given_cell: Give the decoder the structure's atom count (the DiffCSP protocol).
            Required for a backbone conditioned on the cell size.
        cell_count: Which atom count ``given_cell`` gives: ``primitive``, the protocol's, or
            ``conventional``, which also reveals the lattice centring. A backbone whose
            cell-size channel was trained on conventional counts needs ``conventional``: its
            space-group head reads that count before the group, and with it the centring, is
            chosen.
        max_atoms: Largest conventional cell the decoder may build when the cell is not given.
        split: The split to read, ``val`` by default.
        seed: Draws the subset and seeds the sampling, which is the same at every evaluation,
            so that successive evaluations differ by the model and not by the draw.
    """
    period: int
    n_structures: int = 128
    samples: int = 20
    given_cell: bool = True
    cell_count: str = "primitive"
    max_atoms: int = 80
    split: str = "val"
    seed: int = 0


#: Lattice points of a conventional cell, by the first letter of the space group's symbol.
#: PyXtal uses the hexagonal setting for the rhombohedral groups, hence R = 3.
LATTICE_POINTS = {"P": 1, "A": 2, "B": 2, "C": 2, "I": 2, "F": 4, "R": 3}


def lattice_points(spacegroup: int) -> int:
    """Lattice points of a space group's conventional cell, in PyXtal's settings."""
    from pyxtal.symmetry import Group  # noqa: PLC0415

    return LATTICE_POINTS[Group(int(spacegroup)).symbol[0]]


def hits_at_n(hits: Sequence[Sequence[bool]], ns: Sequence[int] = NS) -> Dict[int, int]:
    """How many formulas have a hit among their first N samples, for each N."""
    return {n: int(sum(any(h[:n]) for h in hits)) for n in ns}


def _pyxtal_gene(spacegroup: int, elements, letters, multiplicities) -> dict:
    """A dataset row as the PyXtal dict the decoder emits."""
    species = sorted({str(e) for e in elements})
    sites = [[f"{m}{l}" for e, l, m in zip(elements, letters, multiplicities) if str(e) == s] for s in species]
    return {"group": int(spacegroup), "species": species, "sites": sites,
            "numIons": [sum(int(lab[:-1]) for lab in labs) for labs in sites]}


class CSPValidation:
    """Samples genes for a fixed subset of a split and scores them against the structures' own."""

    def __init__(self, trainer, config: CSPValidationConfig):
        from wyckoff_transformer.csp import ConstrainedDecoder  # noqa: PLC0415
        from wyckoff_transformer.evaluation.protocol import GeneFingerprinter  # noqa: PLC0415

        if not trainer.composition_conditioning:
            raise ValueError("csp_evaluation needs a composition-conditioned backbone")
        if not getattr(trainer, "predict_start", False):
            raise ValueError("csp_evaluation samples the space group from the start head (predict_start)")
        if trainer.condition_on_cell_size and not config.given_cell:
            raise ValueError("a backbone conditioned on the cell size needs given_cell: true")
        if config.cell_count not in ("primitive", "conventional"):
            raise ValueError(f"cell_count must be primitive or conventional, not {config.cell_count!r}")
        if trainer.condition_on_cell_size and config.cell_count != "conventional":
            raise ValueError("this backbone's cell-size channel reads the conventional atom count before "
                             "the space group is chosen; it needs cell_count: conventional")
        if config.period % trainer.validation_period:
            raise ValueError(f"csp_evaluation.period ({config.period}) must be a multiple of "
                             f"validation_period ({trainer.validation_period})")
        if trainer.dataset is None:
            raise ValueError("csp_evaluation reads the split from the dataset cache; the trainer has no dataset")
        self.trainer, self.config = trainer, config
        self.fingerprinter = GeneFingerprinter()
        model = getattr(trainer.model, "_orig_mod", trainer.model)  # decode eagerly, not through torch.compile
        self.decoder = ConstrainedDecoder(
            model=model, cascade_order=trainer.cascade_order, cascade_is_target=trainer.cascade_is_target,
            tokenisers=trainer.tokenisers, token_engineers=trainer.token_engineers, masks=trainer.masks_dict,
            stops=trainer.stops_dict, max_sequence_len=trainer.max_sequence_length, device=trainer.device,
            model_dir=trainer.run_path)
        self.model = model
        self.space_groups = sorted(trainer.tokenisers[trainer.start_name])
        self.lattice_points = {sg: lattice_points(sg) for sg in self.space_groups}
        self.tasks = self._load_tasks()
        self._feasible: Dict[int, Dict[int, List[int]]] = {}
        logger.info("CSP validation: %d %s structures x %d samples every %d epochs",
                    len(self.tasks), config.split, config.samples, config.period)

    def _load_tasks(self) -> List[dict]:
        from pymatgen.core import Composition  # noqa: PLC0415
        from wyckoff_transformer.dataset_cache import dataset_cache_dir, load_split  # noqa: PLC0415

        t = self.trainer
        columns = ["spacegroup_number", "elements", "wyckoff_letters", "multiplicity", "composition"]
        columns += list(t.condition_features)
        df = load_split(dataset_cache_dir(t.dataset), self.config.split, columns=columns)
        rng = np.random.default_rng(self.config.seed)
        rows = rng.choice(len(df), size=min(self.config.n_structures, len(df)), replace=False)
        tasks = []
        for i in sorted(rows):
            row = df.iloc[i]
            gene = _pyxtal_gene(row.spacegroup_number, row.elements, row.wyckoff_letters, row.multiplicity)
            counts = {str(k): int(v) for k, v in row.composition.items()}
            reduced = Composition(counts).reduced_composition.as_dict()
            n_conventional = sum(counts.values())
            points = lattice_points(row.spacegroup_number)
            if n_conventional % points:
                raise ValueError(f"row {i}: {n_conventional} atoms in a cell with {points} lattice points")
            tasks.append({
                "row": int(i),
                # Plain element counts: parse_formula rejects pymatgen's parenthesised groups.
                "formula": "".join(f"{el}{int(n)}" for el, n in reduced.items()),
                "n_atoms": n_conventional,
                "n_primitive": n_conventional // points,
                "spacegroup": int(row.spacegroup_number),
                "fingerprint": self.fingerprinter.fingerprint(gene),
                "condition": {f: float(row[f]) for f in t.condition_features} or None,
            })
        return tasks

    def _cond(self, ratio, z: int, condition: Optional[dict]) -> torch.Tensor:
        from wyckoff_transformer.composition import composition_vector  # noqa: PLC0415

        t = self.trainer
        parts = []
        if t.condition_features:
            parts.append(t.transform_condition(t.build_condition_from_values(condition, 1, device=t.device)))
        parts.append(composition_vector(ratio.element_tokens, ratio.counts_at(z), t.n_elements, device=t.device,
                                        condition_on_cell_size=t.condition_on_cell_size).unsqueeze(0))
        cond = torch.cat(parts, dim=-1)
        return t.with_null_indicator(cond) if t.classifier_free_guidance else cond

    def _cell_atoms(self, task: dict, sg: int) -> Optional[int]:
        """Atoms the decoder must place in space group *sg*'s conventional cell; None if free."""
        if not self.config.given_cell:
            return None
        if self.config.cell_count == "conventional":
            return task["n_atoms"]
        return task["n_primitive"] * self.lattice_points[sg]

    def _feasible_groups(self, index: int, ratio, task: dict) -> Dict[int, List[int]]:
        """Space group -> cell sizes it can build the formula at; fixed per task, so cached."""
        if index not in self._feasible:
            from wyckoff_transformer.cli.csp import start_tensor_for  # noqa: PLC0415

            t, unit = self.trainer, sum(ratio.unit_counts)
            out = {}
            for sg in self.space_groups:
                comb = self.decoder.combinatorics(sg, start_tensor_for(t, sg, t.device))
                zs = ratio.feasible_z(comb, t.max_sequence_length, self.config.max_atoms // unit)
                n_atoms = self._cell_atoms(task, sg)
                if n_atoms is not None:
                    zs = [z for z in zs if z * unit == n_atoms]
                if zs:
                    out[sg] = zs
            self._feasible[index] = out
        return self._feasible[index]

    @torch.no_grad()
    def _sample(self, index: int, task: dict) -> List[Optional[dict]]:
        """``samples`` genes for one task in sampling order, each with its space group; None if not decoded."""
        from wyckoff_transformer.cli.csp import start_tensor_for  # noqa: PLC0415
        from wyckoff_transformer.csp import CompositionRatio, candidates_to_pyxtal  # noqa: PLC0415

        t, n = self.trainer, self.config.samples
        ratio = CompositionRatio.for_formula(task["formula"], t.tokenisers["elements"])
        feasible = self._feasible_groups(index, ratio, task)
        if not feasible:
            return [None] * n
        # Only a cell-size channel reads z, and it is then the conventional count, the same
        # for every group; the composition fractions do not depend on z.
        z = task["n_atoms"] // sum(ratio.unit_counts) if self.trainer.condition_on_cell_size else 1
        cond = self._cond(ratio, z, task["condition"])
        log_p = torch.log_softmax(self.model.forward_start(1, cond=cond).float()[0], -1).cpu()
        log_p = torch.where(torch.tensor([sg in feasible for sg in self.space_groups]), log_p,
                            torch.tensor(-np.inf))
        seed = self.config.seed * 1_000_003 + task["row"]
        cpu = torch.Generator().manual_seed(seed)
        dev = torch.Generator(device=t.device).manual_seed(seed)
        trial_sg = [self.space_groups[i] for i in
                    torch.multinomial((log_p - log_p.logsumexp(0)).exp(), n, replacement=True, generator=cpu)]
        out: List[Optional[dict]] = [None] * n
        for sg in sorted(set(trial_sg)):
            trials = [i for i, s in enumerate(trial_sg) if s == sg]
            start = start_tensor_for(t, sg, t.device)
            comb = self.decoder.combinatorics(sg, start)
            got = []
            for _ in range(5):  # a path can die at the site cap; draw again
                need = len(trials) - len(got)
                if need <= 0:
                    break
                cands = self.decoder.decode(start=start, sg_number=sg, ratio=ratio, allowed_z=feasible[sg],
                                            max_z=max(feasible[sg]), n_candidates=need, strategy="sample",
                                            cond=cond, generator=dev)
                got += [g for g in candidates_to_pyxtal(cands, t, start, comb) if g is not None]
            got = [got[j] for j in torch.randperm(len(got), generator=cpu).tolist()]
            for trial, gene in zip(trials, got):
                out[trial] = {"gene": gene, "sg": sg}
        return out

    def evaluate(self) -> Dict[str, float]:
        """Match rates over the subset; every rank must call it."""
        t = self.trainer
        dist = t.distributed
        was_training = self.model.training
        self.model.eval()
        started = time.time()
        gene_hits, sg_hits, decoded = [], [], 0
        try:
            for index in range(dist.rank, len(self.tasks), dist.world_size):
                task = self.tasks[index]
                try:
                    samples = self._sample(index, task)
                except Exception as error:  # noqa: BLE001 - one formula must not end a training run
                    logger.warning("CSP validation: %s failed (%s: %s)", task["formula"], type(error).__name__, error)
                    samples = [None] * self.config.samples
                decoded += sum(s is not None for s in samples)
                gene_hits.append([s is not None and self.fingerprinter.fingerprint(s["gene"]) == task["fingerprint"]
                                  for s in samples])
                sg_hits.append([s is not None and s["sg"] == task["spacegroup"] for s in samples])
        finally:
            self.model.train(was_training)
        ns = [n for n in NS if n <= self.config.samples] or [self.config.samples]
        g, s = hits_at_n(gene_hits, ns), hits_at_n(sg_hits, ns)
        local = torch.tensor([*g.values(), *s.values(), len(gene_hits), decoded], dtype=torch.float64,
                             device=t.device)
        # all_reduce_mean is the collective the context offers; times world_size it is the sum.
        total = dist.all_reduce_mean(local) * dist.world_size if dist.enabled else local
        total = total.tolist()
        k = len(ns)
        n_tasks = total[2 * k]
        metrics = {f"gene_mr@{n}": total[i] / n_tasks for i, n in enumerate(ns)}
        metrics.update({f"sg_mr@{n}": total[k + i] / n_tasks for i, n in enumerate(ns)})
        metrics["decoded"] = total[2 * k + 1] / (n_tasks * self.config.samples)
        metrics["seconds"] = time.time() - started
        return metrics
