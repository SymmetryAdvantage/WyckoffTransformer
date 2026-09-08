# DFT fixed-hull adversarial screening

> **Scope:** offline proof of concept. The target is SUN against the immutable
> LeMat-Bulk PBE hull. No MLIP energy is used as a label or reward, no generated
> structure is added to the hull, and the model is not updated from campaign
> outcomes.

## Claim

Given a pool of generated Wyckoff genes and a fixed DFT relaxation budget, enrich
the submitted pool for structures below the published DFT hull. This first screen
models energetic margins only: validity, uniqueness, and novelty remain downstream
outcomes measured after relaxation. It is a benchmark-targeted triage claim, not
yet a calibrated SUN-probability model or a claim of stability against a complete
thermodynamic hull.

For gene `G`, stochastic reconstruction and DFT relaxation `R(G, ω)`, and fixed
reference `D`, the strict-SUN objective is

\[
J(q)=\mathbb E_{G\sim q,\omega}
\left[V(R)U(R)N_D(R)\mathbf 1(E_D(R)-h_D(X)\le 0)\right].
\]

The first implementation changes the selection distribution over a fixed
candidate pool. With equal relaxation costs, ranking by the predicted energetic margins is the
first approximation to this finite-budget decision problem.

## Why DFT only

LeMat-Bulk supplies DFT energies, so Stage 1 trains and screens on that energy
scale. Training a reward directly on the benchmark MLIPs while DFT labels are
available would make it difficult to distinguish chemical signal from
potential-specific reward hacking.

MLIP energies may become appropriate later when:

1. generated, MLIP-relaxed structures are added in an active-learning loop; or
2. a DFT calculation has not converged (`f_max` remains large) and a consistent
   MLIP relaxation is demonstrably a better estimate.

Neither exception is part of this command. `wyformer-dft-screen` is intentionally
separate from `wyformer-protocol`, whose energies and hulls are MLIP-specific.

## Two estimands, one fixed hull

The available models answer similar but non-identical questions.

### Composition floor

\[
f^*(X)=\min_{S:\,c(S)=X}E_{DFT}(S).
\]

The censored formula ensemble estimates this latent floor. LeMat-Bulk provenance
labels affect the model of how loose an observed upper bound is, while the floor
head is restricted to chemistry and, optionally, neighbourhood features that can
be computed for a never-observed formula.

Unstable DFT entries are useful here. They contribute evidence about which search
processes attempted a formula and how loose archive bounds are. The current table
collapses each formula to its observed minimum; non-minimum entries survive
through provenance counts, cell-size diversity, force summaries, and system
neighbourhood density. They are not treated as discovered stable structures.

The component scores are

\[
s_F(X)=\hat f^*(X)-h_D(X)
\]

and the winner's-curse-adjusted version

\[
s_F^+(X)=\hat f^*(X)+\sigma_{epi}(X)-h_D(X).
\]

The excess scale predicted by the censored likelihood is not epistemic uncertainty
and does not enter `s_F+`.

### Gene attainable energy

\[
g^*(G)=\min_{S\in\mathcal R(G)}E_{DFT}(S).
\]

The initial gene critic approximates this with the lowest PBE formation energy
observed for the augmentation-invariant gene. It is queried at `max_force = 0`,
which means the clean-relaxation limit, not an already available DFT result.

\[
s_G(G)=\hat g^*(G)-h_D(c(G)).
\]

### Conservative joint score

The two estimates are not added: doing so would count the hull twice and would
have no physical estimand. Until a calibrated joint outcome model exists, the
screen uses

\[
s_J(G)=\max(s_F^+(c(G)),s_G(G)).
\]

Therefore `s_J <= 0` only when both estimators reach or clear the same fixed DFT hull. With
perfect estimators the formula condition would be redundant because
`f*(X) <= g*(G)`; in practice it is a composition-opportunity consistency check
between two imperfect models. All component scores are written so this fusion
rule can be ablated rather than assumed.

## Existing components and boundaries

| Component | Estimand | Role here |
|---|---|---|
| `wyformer-screen` | latent composition floor | unchanged, reused through its model APIs |
| `wyformer-gene-screen` | observed gene-level attainable-energy proxy | unchanged, reused as the gene scorer |
| `wyformer-dft-screen` | conservative conjunction against one PBE hull | new offline selection layer |
| `wyformer-protocol` | relaxed MLIP energy against a matching MLIP hull | not used for the DFT claim |

A fingerprint collision is not a proof of structural non-novelty, so the DFT
screen reports formula membership but does not discard known formulas or genes.
Validity, uniqueness, and novelty must still be evaluated on the relaxed
structures.

## Usage

Train the composition-floor ensemble from its shipped config:

```bash
uv run python -m wyckoff_transformer.formula_energy.train \
  --config yamls/models/formula_energy/censored_floor.yaml \
  --device cuda
```

CLI options such as `--out`, `--epochs`, and `--models` override the YAML. Train
the gene critic as documented in [composition screening](composition_screening.md),
then rank a generated pool:

```bash
uv run wyformer-dft-screen generated/<run>/wyckoff_genes.json.gz \
  --formula-ensemble runs/formula_energy/ensemble.pt \
  --regressor-path runs/<gene-energy-run> \
  --formula-table data/formula_energy/formula_table.parquet \
  --reference data/lemat-bulk/lemat_pbe_ehull.csv.gz \
  --rank-by joint_score_adjusted \
  --top 1000 \
  --out generated/<run>/dft_screen.csv
```

Useful fixed-pool ablations are available without recomputing predictions:

- `composition_score_naive`
- `composition_score_adjusted`
- `gene_score`
- `joint_score_naive`
- `joint_score_adjusted`

`--joint-below-hull-only` keeps only adjusted joint passes at or below zero. Ranking is a
pre-relaxation screen; the selected genes still require the external DFT
relaxation and evaluation workflow.

## Proof-of-concept experiment

Generate one large candidate pool and assign candidates from that same pool to
selection arms:

1. random baseline;
2. generator likelihood baseline;
3. composition floor;
4. gene critic;
5. conservative joint score.

Keep the number of submitted DFT relaxations equal. During development,
`E_hull <= 0.1 eV/atom` provides more statistical power, but freeze the selector
before the final strict-SUN (`E_hull <= 0`) comparison.

Report all three denominators:

1. SUN per submitted structure;
2. SUN per completed DFT relaxation;
3. SUN per raw sampled gene.

This distinguishes a useful relaxation-budget screen from a generator that has
itself learned a higher unfiltered SUN rate.

### Measuring it against the MLIP protocol first

The DFT relaxations are the point, but the same selection question can be asked
much sooner against the [de novo ranking
protocol](de_novo_ranking_protocol.md), whose relaxations are MLIP. Score one
pool with `wyformer-dft-screen`, relax the *whole* pool with `wyformer-protocol`
so every arm draws from the same measured candidates, then:

```bash
python scripts/analyse_dft_screen_uplift.py generated/<run>
```

which reads `dft_screen.csv` and `protocol/structures.csv` and reports each
arm's MetaSUN and SUN rate at a fixed budget against a random draw of the same
size. `scripts/protocol_relax.pbs` runs the relaxation half as a self-chaining
PBS job (`qsub -v POOL=generated/<run> scripts/protocol_relax.pbs`).

This is a proxy and not the claim: the screen's estimators and its hull are PBE,
while the protocol's energies and hull are ORB, so a candidate can clear one and
miss the other. What it does test cheaply is whether the ranking carries any
signal about relaxed stability at all, which is a precondition for the DFT
version being worth its budget.

The first such measurement, over 5,000 `e9ywwsie` genes, is in
[`docs/archive/e9ywwsie_dft_screen_uplift_report.md`](archive/e9ywwsie_dft_screen_uplift_report.md).
Two results from it shape how the screen should be used:

- The ranking is strong (Spearman +0.59 against ORB-relaxed `e_above_hull`, an
  8x spread in metastable rate between the best and worst decile) but the
  conservative joint score is not usable as a *filter*: 8 genes of 5,000 clear
  `joint_score_adjusted <= 0`.
- Stability and novelty are anti-correlated under the screen -- its best decile
  is 83% already-known formulas -- so ranking the raw pool nets only ~1.4x
  MetaSUN. Filtering on gene novelty *before* ranking gives 2.1-2.4x at the
  small budgets where a screen earns its keep. Rank within the novel subset,
  not across the pool.

## Assumptions and exclusions

1. The LeMat-Bulk PBE reference and its elemental references are immutable across
   training, screening, and evaluation. Hull lookup explicitly subtracts the
   elemental-reference contribution from pymatgen's absolute hull energy; a
   regression test with nonzero elemental energies protects this conversion.
   Current checkpoints record the energy
   convention; custom paths and legacy checkpoints fail closed unless
   `--allow-unverified-energy-scale` is supplied explicitly. Known candidate
   formulas are also cross-checked between the formula table and live hull.
2. The formula floor checkpoint uses the censored objective. An MSE formula model
   predicts the observed archive bound and is rejected.
3. Candidate-specific provenance is unavailable for a novel formula and cannot
   feed the floor head. Only system-neighbourhood location features are accepted.
4. The gene critic uses the observed gene-minimum PBE target, not a per-structure
   mean and not an MLIP energy.
5. The screen does not claim a calibrated joint probability; `max` is a
   conservative ranking rule in eV/atom.
6. This stage does not update the hull or either model after evaluating generated
   structures. Repeated hull-building campaigns are a later stage.
