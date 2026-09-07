# How much does Wyckoff → structure cost us?

WyFormer emits a Wyckoff gene; every reported metric is computed on a structure
CrySPR reconstructs from it. This study measures the loss in that step alone, by
feeding CrySPR genes extracted from *real* structures and asking how often it
gets the original back. The result is a ceiling: no improvement to the
generative model can recover what the reconstruction stage throws away.

## Protocol

1. **Sample.** 1000 structures from LeMat-Bulk with `e_above_hull ≤ 0.1 eV/atom`,
   uniformly at random under a fixed seed. Record the `nsites` and space-group
   distributions of the draw.
2. **Relax with ORB.** `orb_conserv_inf` — `orb-v3-conservative-inf-omat-20250404`,
   the same checkpoint and hull as the ranking protocol. This puts the target
   and the reconstruction on one PES, so a failure to recover is attributable to
   PyXtal sampling rather than to a DFT/MLIP disagreement. These relaxed
   structures are the targets.
3. **Extract the gene** from each relaxed structure with
   `wyckoff_transformer.data.structure_to_sites`, i.e.
   `kick_pyxtal_until_it_works(tol=0.1, a_tol=5.0)` — the same call
   `scripts/process_lemat_symmetry.py` used to build the training set. The kick
   means the tolerance is not constant across structures: log the attempt index
   that succeeded, and treat any gene that needed a tolerance far from 0.1 as a
   separate stratum when reading the results.

   Then **deduplicate genes**. The gene → structure map is one-to-many; if two
   sampled targets reduce to the same gene, they become one gene whose target
   set has two members, and a match against either counts. Report the number of
   genes as well as the number of structures — every rate below is per gene.
4. **Reconstruct.** Each gene through the full CrySPR process with ORB:
   - 10 trials per gene (see [note on the trial count](#why-10-trials-here));
   - stages 1–3 as in `stepwise_relax`, plus stage 4 below;
   - `fmax 0.02`, `steps_limit 500`, on GPU.
5. **Score** each gene against its target(s) with the results specification
   below.

### Stage 4: rattle and re-relax

Stage 3 releases the symmetry constraint but almost never moves: for a
symmetry-invariant potential the forces along symmetry-breaking modes vanish
identically, so a structure converged in stage 2 sits at a stationary point it
cannot leave — measured at >1 meV/atom in 0.4% of trials
([trial and stage spread](cryspr_trial_and_stage_spread.md)). Stage 4 exists to
seed that descent explicitly, which is exactly the failure mode that would make
a reconstruction land on the wrong side of a symmetry-lowering distortion.

| parameter | value |
|---|---|
| displacement | `atoms.rattle(stdev=0.05, seed=...)`, Gaussian per Cartesian component |
| cell | random strain, each component ~ N(0, 0.01), symmetrised |
| seed | deterministic in `(gene_id, trial_index)`, so the run is reproducible |
| relaxation | as stage 3 — free cell, no symmetry constraint, same `fmax`/`steps_limit` |
| acceptance | keep stage 4 **only if** it lowers E/atom by > 1 meV/atom; otherwise the trial's final structure is stage 3's |

The acceptance rule matters for the metric, not just for the energy. An
unconditional rattle would displace an already-converged structure by ~0.05 Å of
noise and could turn a match into a non-match for nothing. With the rule, stage 4
can only ever help the energy, and its effect on recovery is measurable — score
each trial at stage 3 and at stage 4 separately and report both columns.

0.05 Å sits below the 0.1 Å symmetry tolerance of step 3, so the rattle
perturbs without by itself changing the detected group; it breaks the exact
stationarity, which is all it needs to do.

## Results specification

### Matching

`pymatgen.analysis.structure_matcher.StructureMatcher(ltol=0.2, stol=0.3,
angle_tol=5, primitive_cell=True, scale=True)` — the defaults. Report a second
match rate at the looser CDVAE-style `ltol=0.3, stol=0.5, angle_tol=10` for
comparability with published match rates, and record the volume ratio per match
so a stricter, `scale=False` reading is possible after the fact without re-running.

### One row per gene

| column | meaning |
|---|---|
| `gene_id`, `n_targets` | gene identity; how many sampled structures reduced to it |
| `spacegroup`, `crystal_system`, `nsites`, `n_wyckoff_sites` | breakdown keys |
| `dof_positional` | Σ `dof` over the gene's sites — free coordinates PyXtal must guess |
| `dof_total` | `dof_positional` + lattice DOF of the crystal system (1 cubic … 6 triclinic) |
| `symmetry_tol_attempt` | which `kick_pyxtal_until_it_works` attempt produced the gene |
| `n_trials_generated`, `n_trials_relaxed` | PyXtal and relaxation survival, out of 10 |
| `n_matching_trials_s3`, `n_matching_trials_s4` | trials matching a target, by stage |
| `matched_kept` | the lowest-energy trial matches a target |
| `matched_any` | any trial matches a target |
| `e_kept`, `e_target` | final E/atom of the kept trial and of the best target |
| `de_kept` | `e_kept − e_target` |
| `verdict` | below |

### Verdicts

Evaluated in order, so every gene lands in exactly one:

| verdict | condition | reading |
|---|---|---|
| `recovered` | `matched_kept` | what CrySPR actually delivers |
| `sampled_not_selected` | `matched_any` | PyXtal found it; the energy criterion discarded it |
| `lower_energy_alternative` | `de_kept < −1 meV/atom` | **not a failure** — the gene is genuinely ambiguous and CrySPR found a better ORB minimum for it |
| `missed` | otherwise | the real failure |
| `generation_failed` | `n_trials_relaxed == 0` | PyXtal or the relaxation never produced a structure |

Three headline numbers:

- **recovery rate** = `recovered` — the quantity the study is for;
- **sampling ceiling** = `recovered + sampled_not_selected` — what a perfect
  selection rule on the same 10 trials would give, and therefore how much of the
  loss is sampling versus ranking;
- **ambiguity rate** = `lower_energy_alternative` — genes that do not determine
  a structure, where recovery is not the right thing to ask for.

Separating the second and third is the point of the spec. Collapsing all three
into "not recovered" would attribute to CrySPR both the cases where it found the
target and rejected it, and the cases where it beat the target.

### Breakdowns

Recovery rate against, each as a table plus the count in each bin:

| key | bins |
|---|---|
| `dof_positional` | 0, 1–2, 3–5, 6–10, >10 |
| `dof_total` | same bins |
| `nsites` | ≤10, 11–20, 21–40, >40 |
| `n_wyckoff_sites` | 1, 2–3, 4–6, >6 |
| `crystal_system` | all 7 |
| `spacegroup` | appendix table only |

Aggregate to crystal system for the readable result; 230 space groups over 1000
genes is too sparse for anything else. A bin holding 100 genes carries about
±5 percentage points, so read the DOF trend, not individual cells.

The DOF curve is the actionable output. The single scalar answers "how much do
we lose"; the curve answers "on which genes", which is what would tell us
whether to bias the model toward reconstructable genes or to fix the
reconstruction.

### Sanity guard

Reject any trial with `|E| > 50 eV/atom` before scoring. Collapsed cells reach
−10¹⁵ eV and MACE reports them without complaint (1% of trials in the
`upi73i4k` run); one such trial would otherwise win its gene's energy
comparison and register as a spurious `lower_energy_alternative`.

## Why 10 trials here

The ranking protocol uses 1 trial and the spread analysis recommends 3, both on
the grounds that the marginal *energy* gain falls ~3.6× per added trial. That
argument does not apply here. This study measures coverage — whether the target
basin is reachable at all — and best-of-N coverage keeps growing after the mean
energy has stopped improving. 10 trials also makes `sampled_not_selected` a
meaningful number rather than a rounding artefact.

## Cost

1000 genes × 10 trials × 4 stages = 40,000 relaxations, about 1.8× the 22,500 of
the `upi73i4k` spread run, with a heavier potential. Run on GPU. A 100-gene pilot
first is worth it: it fixes the wall-clock estimate and shakes out the stage-4
plumbing, which does not exist yet — `stepwise_relax` is three stages, and
`func_run` returns only the lowest-energy trial, so the driver has to collect the
per-trial finals from the `trial-N/` directories to compute `matched_any`.

## See also

- [CrySPR trial and stage spread](cryspr_trial_and_stage_spread.md) — where the trial and stage numbers come from
- [The de novo ranking protocol](de_novo_ranking_protocol.md) — the ORB checkpoint and hull pairing
