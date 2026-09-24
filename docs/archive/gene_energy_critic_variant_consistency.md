# Gene-energy critics and equivalent Wyckoff descriptions

> **STATUS: MEASUREMENT, 2026-09-24.** Code at `7778507` (branch `der-min-energy`),
> script `scripts/analyse_variant_consistency.py`. Both critics were scored on the
> full validation split of `lemat_bulk_fmax1_stress`. The augmented critic was still
> training when this was written, so its absolute accuracy is provisional. Its
> consistency across descriptions is the result that matters here.

## The question

A Wyckoff gene has several equivalent descriptions: the relabellings the affine
normaliser allows. In 26 orthorhombic space groups these change the site-symmetry
symbol itself (`2..` becomes `.2.`), not just the enumeration index; see
[the augmentation audit](../wyckoff_augmentation_audit.md). The critic's target,
`gene_min_formation_energy_per_atom`, is defined per augmentation-invariant gene, so
every description of a structure has the same target.

`wyformer-gene-screen` scores whatever description the generator emits, without
canonicalising it (`GeneFingerprinter.record`). WyFormer is trained with
augmentation, so it emits descriptions spread across the equivalent labellings, not
in the dataset's stored convention. A critic that has only seen the stored
descriptions is being used outside its training distribution.

## Method

For each of the 100,000 validation structures, every distinct
(`site_symmetries`, `sites_enumeration`) pair in its `*_augmented` columns was
scored, along with the stored description. That gives 420,224 distinct
descriptions, and 83.0% of structures have more than one. Each description is
tokenised as a single variant, so the model scores exactly what it is given.
`multiplicity` is recomputed from (space group, symbol, enumeration).

The **uniform MAE** averages each structure's error over all its descriptions, then
averages over structures. It approximates what a generator trained with augmentation
feeds the screener. **Spread** is the max − min of a structure's predictions across
its descriptions; a critic that respects the symmetry gives 0.

## Results

| | 5x baseline, no augmentation | der critic, augmented |
|---|---|---|
| W&B run | `min_energy_5x_adamw_wsd-20260921-125850` | `min_energy_adamw_wsd-20260924-102431` |
| config | `yamls/models/lemat/min_energy_5x_adamw_wsd.yaml` | `yamls/models/lemat/der/min_energy_adamw_wsd.yaml` |
| tokeniser | `lemat_bulk_fmax1_sg_multiplicity` | `der_tokenizer_v1` |
| augmented fields in training | none | `site_symmetries` + `sites_enumeration` |
| checkpoint | finished, 2859 epochs | epoch 250 of 2859, mid-run |
| MAE, stored descriptions | **0.0356** | 0.0595 |
| MAE, other descriptions (pooled) | 0.1252 | 0.0566 |
| **MAE, uniform over descriptions** | **0.0728** | **0.0595** |
| MAE, descriptions with a relabelled symbol | **0.3317** | 0.0665 |
| spread, median / mean / p90 / max | 0.035 / 0.137 / 0.335 / 5.29 | 0.014 / 0.022 / 0.050 / 1.19 |
| structures with spread > their model's stored MAE | 49.7% | 6.9% |
| structures with spread > 0.1 eV/atom | 23.4% | 2.1% |
| spread, enumeration only (n = 76,670), median / p90 | 0.032 / 0.224 | 0.013 / 0.042 |
| spread, symbol relabelled (n = 6,348), median / p90 | 0.253 / 1.686 | 0.045 / 0.119 |

All values are in eV/atom. The two models differ in tokeniser, augmentation and
training length. They share the architecture (d_model 96, 3 layers,
dim_feedforward 320), the loss, the batch of 12500, the learning rate of 0.003 and
the data.

## What it means

**A critic trained without augmentation has learned the stored description, not the
gene.** Its validation MAE of 0.036 eV/atom holds only for the dataset's own
labelling. Averaged over equivalent descriptions it is 0.073, twice as large. Where
the symbol is relabelled it is 0.33, nine times as large and large enough to put a
structure on the wrong side of the hull. The spread for one structure exceeds the
model's own MAE for half the validation set.

**The augmented critic is close to invariant.** Its MAE is the same on stored and
other descriptions. Even for relabelled symbols it is 0.066 against 0.060. At epoch
250 of 2859, and after the loss spike at step ~7040, it already has a lower uniform
MAE (0.060) than the fully trained baseline (0.073). Its stored-description MAE is
worse, which is expected at this point in training.

**Consequences:**
- The number to report for a screening critic is the uniform MAE over equivalent
  descriptions, not the validation loss on stored descriptions. The validation loss
  logged during training does not measure this.
- Screening results produced with a critic trained without augmentation are noisier
  than that critic's reported MAE suggests, especially for orthorhombic genes.
- Augmentation is one fix. Canonicalising every description to one fixed
  representative, at training and at scoring, is the other, and it gives exact
  invariance at no training cost. It has not been built.

## Caveats

- The augmented critic is a mid-run checkpoint of a run that had a loss spike at
  step ~7040. Rerun this on the final checkpoint before quoting its absolute MAE.
- "Stored description" means the labelling in the dataset cache. It is
  deterministic per structure, not a canonical choice per gene.
- One seed per model.
