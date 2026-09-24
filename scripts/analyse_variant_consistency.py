"""How consistently does a trained gene-energy regressor score equivalent Wyckoff descriptions?

Every validation structure is expanded into all of its augmentation variants -- the
(site_symmetries, sites_enumeration) pairs the corrected augmentation emits, which name
the same gene -- and each variant is scored separately. A symmetry-respecting model gives
one number per structure; the spread across variants is the inconsistency the screener
inherits, since wyformer-gene-screen scores whatever description a generator emits.

    python scripts/analyse_variant_consistency.py <run-id> <n-structures> <device> [out.csv]

See docs/archive/gene_energy_critic_variant_consistency.md for the results.
"""
import sys

import numpy as np
import pandas as pd
import torch

from wyckoff_transformer.cli.csp import load_trainer
from wyckoff_transformer.dataset_cache import dataset_cache_dir, load_split
from wyckoff_transformer.paths import runs_root
from wyckoff_transformer.prediction import (
    build_tokenised_prediction_tensors,
    filter_supported_tokens,
)

TARGET = "gene_min_formation_energy_per_atom"
CHUNK = 20000

run_id, n_structures, device = sys.argv[1], int(sys.argv[2]), torch.device(sys.argv[3])
out_path = sys.argv[4] if len(sys.argv) > 4 else None

trainer = load_trainer(device, model_path=runs_root() / run_id)
print(f"run {run_id}: tokeniser {trainer.tokeniser_config.name}, "
      f"trained with augmented fields {trainer.augmented_fields}")

columns = ["spacegroup_number", "elements", "site_symmetries", "sites_enumeration",
           "multiplicity", "composition", "site_symmetries_augmented",
           "sites_enumeration_augmented", TARGET]
val = load_split(dataset_cache_dir("lemat_bulk_fmax1_stress"), "val", columns=columns)
val = val.dropna(subset=[TARGET])
if n_structures < len(val):
    val = val.sample(n=n_structures, random_state=0)
print(f"{len(val)} validation structures")

rows = []
for structure, record in val.iterrows():
    variants = list(zip(record["site_symmetries_augmented"],
                        record["sites_enumeration_augmented"]))
    original = (list(record["site_symmetries"]), list(record["sites_enumeration"]))
    seen = set()
    for kind, (ss, en) in [("original", original)] + [("variant", v) for v in variants]:
        key = (tuple(ss), tuple(int(e) for e in en))
        if key in seen:
            continue
        seen.add(key)
        rows.append({
            "structure": structure, "kind": kind,
            "symbol_changed": tuple(ss) != tuple(original[0]),
            "spacegroup_number": record["spacegroup_number"],
            "elements": record["elements"], "site_symmetries": list(ss),
            "sites_enumeration": [int(e) for e in en],
            "multiplicity": record["multiplicity"], "composition": record["composition"],
            TARGET: record[TARGET],
        })
frame = pd.DataFrame(rows)
print(f"{len(frame)} distinct descriptions, "
      f"{frame.groupby('structure').size().gt(1).mean():.1%} of structures have more than one")

predictions = pd.Series(np.nan, index=frame.index)
dropped = 0
for start in range(0, len(frame), CHUNK):
    chunk = frame.iloc[start:start + CHUNK]
    supported, unsupported = filter_supported_tokens(chunk, trainer)
    dropped += len(unsupported)
    # No *_augmented columns: each description is tokenised as its own single variant,
    # so a model trained with augmentation scores exactly the description it is given.
    tensors = build_tokenised_prediction_tensors(supported, trainer)
    with torch.no_grad():
        mean, _ = trainer.predict_scalars(tensors, augmentation_samples=1)
    predictions.loc[supported.index] = mean.float().cpu().numpy()
frame["prediction"] = predictions
print(f"{dropped} descriptions outside the model's vocabulary, left unscored")
if out_path:
    frame.drop(columns=["elements", "site_symmetries", "sites_enumeration", "multiplicity",
                        "composition"]).to_csv(out_path)

scored = frame.dropna(subset=["prediction"])
scored = scored.assign(abs_error=(scored["prediction"] - scored[TARGET]).abs())
per = scored.groupby("structure").agg(
    n=("prediction", "size"), spread=("prediction", lambda p: p.max() - p.min()),
    std=("prediction", "std"), symbol_changed=("symbol_changed", "any"))
multi = per[per["n"] > 1]
original_mae = scored.loc[scored["kind"] == "original", "abs_error"].mean()
variant_mae = scored.loc[scored["kind"] == "variant", "abs_error"].mean()
# What a generator trained with augmentation feeds the screener: any equivalent
# description, so each structure's error averaged uniformly over its descriptions.
uniform_mae = scored.groupby("structure")["abs_error"].mean().mean()
changed_mae = scored.loc[scored["symbol_changed"], "abs_error"].mean()

print(f"\nMAE on original descriptions      {original_mae:.4f} eV/atom")
print(f"MAE on other variants (pooled)    {variant_mae:.4f} eV/atom")
print(f"MAE, uniform over each structure's descriptions  {uniform_mae:.4f} eV/atom")
print(f"MAE on descriptions with a relabelled symbol     {changed_mae:.4f} eV/atom")
print(f"structures with >1 description    {len(multi)}")
print(f"  prediction spread (max-min)     median {multi['spread'].median():.4f}, "
      f"mean {multi['spread'].mean():.4f}, p90 {multi['spread'].quantile(0.9):.4f}, "
      f"max {multi['spread'].max():.4f}")
print(f"  per-structure std               mean {multi['std'].mean():.4f}")
print(f"  spread > original MAE           {(multi['spread'] > original_mae).mean():.1%}")
print(f"  spread > 0.1 eV/atom            {(multi['spread'] > 0.1).mean():.1%}")
for changed, group in multi.groupby("symbol_changed"):
    label = "symbol relabelled" if changed else "enumeration only "
    print(f"  {label}  n={len(group):6d}  median spread {group['spread'].median():.4f}  "
          f"p90 {group['spread'].quantile(0.9):.4f}")
