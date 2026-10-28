# Energy regressor: architectural explanations for the out-of-distribution gap

Written 2026-10-29 at git `87b686523741aa2a0c1fc1a91f09465090f521e8`. This is a
**code and config review, not an experiment**: no training, inference or energy
calculation was run. Every number quoted comes from
[`energy_regressor_precision.md`](energy_regressor_precision.md), which records its own
commits and W&B runs. The model reviewed is `SymmetryAdvantage/WyFormer-LeMat-Energy`, trained by
[`min_energy_adamw_wsd_h7x_xl-20261015-185731`](https://wandb.ai/symmetry-advantage/WyckoffTransformer/runs/min_energy_adamw_wsd_h7x_xl-20261015-185731)
from [`yamls/models/lemat_fixed/min_energy_adamw_wsd_h7x_xl.yaml`](../yamls/models/lemat_fixed/min_energy_adamw_wsd_h7x_xl.yaml)
with tokeniser `der_tokenizer_v1`.

The question: which architectural choices could explain why the archive test MAE is
24.4 meV/atom, but novel generated genes have 121.7 meV/atom, and novel genes with unseen
anonymous templates 231.6 meV/atom?

**Status: hypotheses.** None of the items below has been tested by an ablation. The
experiments that would test them are listed at the end.

## What the precision report already rules in or out

- **The unseen-template gap is present from the start of training, so it is not caused
  by overfitting.** In §G of the precision report, the epoch-250 checkpoint (`v1`)
  already has 241.0 meV/atom on novel genes with unseen templates, against 125.8 on all
  novel genes. Across all later checkpoints the unseen-template MAE stays between 216
  and 232. Late-training overfitting adds 8–10 meV/atom (paired v8→final: +8.0 on novel
  genes, +10.2 on unseen templates). That is real, but small next to a gap of more than
  100 meV/atom. This points at inductive bias and input information, not at capacity or
  training length.
- **The archive test does not measure transfer to new templates.** The archive split is
  random, so test genes almost always use templates that occur in training. The
  24.4 meV/atom measures element substitution within known templates.
- **Part of the gap is the target, not the model.** §D: for novel structures whose genes
  are known, an oracle predicting the archive minimum exactly would score
  131.0 meV/atom, the same as the regressor. The generated DFT energies are single
  realisations, while the training target is the archive's observed gene minimum. So the
  121.7 meV/atom on novel genes is an upper bound on the error against those genes'
  minima.

## Architectural candidates, most to least likely

### 1. The model never sees geometry

The input is the space group, plus site symmetry, Wyckoff enumeration and element for
each orbit. It contains no coordinates, lattice parameters, interatomic distances or
degrees of freedom. For a known template, the template's typical geometry is effectively
memorised across its many element substitutions; prototype substitution is exactly
what the archive test rewards. For a new template the model has nothing that tells it
whether the atoms fit. The strongest driver in §F is packing and short contacts: MAE is
168.7 vs 82.2 meV/atom between the lowest and highest fifths of the closest-contact
ratio, and the sign is the same in all 11 large cohorts.

This limit belongs to the gene representation itself. It can be reduced by giving the model
geometric facts that the gene determines (fixed Wyckoff coordinates, free-parameter
counts, relations between orbits), but a change to the network alone cannot remove it.

### 2. Wyckoff positions are learned labels with no geometric meaning

From the [model config](../yamls/models/lemat_fixed/min_energy_adamw_wsd_h7x_xl.yaml) (L66–L71):

- `site_symmetries` is a learned `nn.Embedding` of the oriented point-group symbol. It is
  shared across space groups, so some transfer is possible.
- `sites_enumeration` is a learned embedding of an arbitrary index within the site
  symmetry. Augmentation shuffles it, so it only says "a different orbit", and the index
  means different things in different space groups.
- The space group is better encoded: with `start_type: one_hot`,
  [`SpaceGroupEncoder`](../src/wyckoff_transformer/tokenization.py) supplies a multi-hot
  vector of its symmetry operations plus the centring, through a linear layer.

Nothing tells the model how orbits relate to each other geometrically, for example which
positions are neighbours, their fixed coordinates, or their coordination. Attention has to
learn those relations from which tokens occur together, which only works for
combinations seen in training. `der_tokenizer_v1` already builds `site_symmetry_ops_id`, a
frozen table of each `(space group, site symmetry)` pair's symmetry operations, expanded by
[`FrozenTableEmbedding`](../src/wyckoff_transformer/cascade/model.py). This model does not
use it.

### 3. The encoder counts orbits, not atoms

`multiplicity: 0` in the cascade embedding (config L71) means multiplicity is not an
encoder input. It is used only as the pooling weight after the encoder
(`token_aggregation: weighted_mean`, `aggregation_weight: 3`; see
`CascadeTransformer.forward` in [`model.py`](../src/wyckoff_transformer/cascade/model.py)).
Inside attention, a 1-fold orbit and a 48-fold orbit count equally in every other
site's context, so each site's view of the surrounding composition is distorted in a
template-dependent way. The network can learn a correction for templates it has seen,
but nothing makes that correction carry over to new combinations. Formation energy is
dominated by composition, so this distortion matters.

Multiplicity is implicitly recoverable from the space group and the site symmetry, so
the per-site value itself transfers for any seen `(sg, ss)` pair. What does not
transfer is the attention weighting. A cheap fix is to add `log m_j` as a key-side
attention bias. Then the softmax weight is proportional to `m_j exp(q·k)`, exactly as if
each orbit were repeated `m_j` times. Feeding multiplicity as a scalar input is the
alternative.

### 4. Symmetry invariance is learned, not built in

Equivalent Wyckoff descriptions are handled by training-time augmentation and by averaging 16
random descriptions at inference. That invariance is only learned on the training
templates, so the 16 descriptions should disagree more on new templates. The §G artifact
keeps the per-augmentation predictions, so this can be checked without new inference.

### 5. Element embeddings are learned IDs with no physical features

`elements` is a learned embedding of the element token, so transfer between chemistries
relies only on co-occurrence. This is likely a minor contributor here. The Gd tail in
§F is suspected to be a data or energy-convention problem (open), and much of the carbon
error goes together with short-contact geometry.

### 6. No regularisation, selected on in-template validation

Encoder dropout is 0.0 (config L100) and weight decay 0.01 (L109). The run lasts 20,000
epochs at about 4–5× the parameters of the earlier reference. Validation draws on the same
templates as training, so validation-based selection rewards memorisation. §G measures this
at about 8 meV/atom on novel genes.

Not suspected: positional encoding is disabled (`learned_positional_encoding_max_size:
0`), so the encoder treats each structure as a set of orbits, and the start token is
excluded from pooling.

## Experiments that would discriminate

1. **Template-held-out archive split.** Hold out whole anonymous templates from training and
   compute gene-minimum labels without the held-out rows. This gives an archive
   out-of-distribution error on DFT minima, free of the target mismatch in §D, and is a
   prerequisite for judging any change below.
2. **Ablations on that split:** multiplicity as an attention bias or scalar input (§3);
   adding `site_symmetry_ops_id` (§2); replacing `sites_enumeration` with the ops vector
   (§2); dropout or earlier stopping (§6).
3. **Augmentation spread vs. template novelty** from the existing per-augmentation
   predictions in the [§G comparison run](https://wandb.ai/symmetry-advantage/WyckoffTransformer/runs/77z0yc9r)
   (§4). Needs no new inference.
4. **Template geometry descriptors** (§1): fixed coordinates of special positions, free
   parameter counts, and pairwise orbit relations derived from the space group. Add them
   as frozen-table or pass-through features and evaluate on the template-held-out split.
