# Re-scoring the protocol artifacts from `lemat_bulk_ehull` to `lemat_bulk_fmax1_stress`

*Archived record of a one-time migration, carried out 2026-09-15/16 on top of
commit `600a2ab`. Not maintained. The rule it leaves behind lives in
[the de novo ranking protocol](../de_novo_ranking_protocol.md#the-novelty-reference);
this page is the evidence.*

## What changed

Protocol artifacts scored before 2026-09-15 judged novelty against
`cache/lemat_bulk_ehull`. That variant lacks 1.12M of the current variant's
rows — everything above `max_force` 0.02 eV/Å, the Materials Project rows with
empty forces, Yb and actinide chemistry — so a generated structure matching one
of them counted as novel.

Gene novelty, `novel_structure`, MetaSUN, SUN and the novelty crossings of such
an artifact are **not comparable** with a run scored against
`lemat_bulk_fmax1_stress`. Validity, uniqueness, `metastable` and `stable` do
not depend on the reference.

A second, quieter part of the same change: the reference fingerprint set used
to be written to one fixed path, so before 2026-09-15 passing another
`--reference-cache` alone silently reused `lemat_bulk_ehull`'s fingerprints.
`--reference-fingerprint-cache` now defaults to `gene_fingerprints.pkl.gz`
beside the reference it was computed from, with the splits in the name when
they are not all three.

`lemat_bulk_ehull` held 4,207,723 rows over 3,959,797 distinct gene
fingerprints, against 5,327,342 rows over 4,826,004 fingerprints for
`lemat_bulk_fmax1_stress`. Every `immutable_id` of both is present in the
5,335,299-row `lemat_pbe.csv.gz` export (checked 2026-09-15), so no artifact
hit `UnresolvedReferenceError` during the migration.

## The re-scores

Re-scored on 2026-09-15 with `--from-artifact --stages screen,score`,
`lemat_bulk_ehull` → `lemat_bulk_fmax1_stress` (free readout unless marked;
every other funnel entry unchanged):

| run | artifact | gene novelty | novel structure | MetaSUN | SUN | MetaSUN, fixed symmetry |
|---|---|---|---|---|---|---|
| `e9ywwsie` | v4 → **v5** | 0.672 → 0.663 | 0.665 → 0.653 | 0.278 → 0.268 | 0.006 → 0.005 | 0.161 → 0.153 |
| `ehull-ssops-20260904-235534` | v2 → **v3** | 0.643 → 0.631 | 0.622 → 0.612 | 0.289 → 0.281 | 0.006 → 0.006 | 0.197 → 0.190 |
| `ehull5x-20260904-213346` | v3 → **v4** | 0.580 → 0.557 | 0.594 → 0.579 | 0.268 → 0.255 | 0.014 → 0.012 | 0.185 → 0.172 |
| `19qbxo6l` | v3 → **v4** | 0.673 → 0.665 | 0.588 → 0.580 | 0.193 → 0.188 | 0.007 → 0.007 | 0.103 → 0.099 |
| `e_all_adamw_wsd-20260909-001225` | v1 → **v2** | 0.698 → 0.690 | 0.608 → 0.603 | 0.210 → 0.206 | 0.003 → 0.002 | 0.107 → 0.103 |
| `relational_e_all_adamw_wsd-20260909-234259` | v0 → **v1** | 0.725 → 0.686 | 0.672 → 0.636 | 0.197 → 0.169 | 0.011 → 0.006 | 0.110 → 0.085 |
| `upi73i4k`\* | v3 → **v4** | 0.670 → 0.660 | 0.638 → 0.633 | 0.256 → 0.250 | 0.005 → 0.005 | 0.161 → 0.154 |

\* Not a pure re-score; see below.

The relational run loses the most: 2.8 points of MetaSUN and half its SUN. Its
generated structures are disproportionately ones the current variant has and
`lemat_bulk_ehull` did not, so its MetaSUN lead over `e_all_adamw_wsd` went
from −0.013 to −0.037. In every run the re-screen kept validity, uniqueness and
the counts exactly, and no gene went from known to novel.

## `upi73i4k` also had its gene 860 re-relaxed

`score` refused v3 (`IncompleteStageError`): all three trials of gene 860
(K24Cl36H90O132, 282 atoms) had failed with a CUDA out-of-memory error on a
2 GiB card. v3 was scored on 2026-09-12, before that check existed, and counted
the gene as having no structure. On 2026-09-16 those three trials were re-run
on zeus with `relax --resume` (one worker on each RTX 6000 Ada,
`--relax-timeout 1800`, the same draws from `pyxtal.extxyz`), then `screen` and
`score` against `lemat_bulk_fmax1_stress`. The other 2365 trials were kept.

So v4 differs from v3 by that one gene as well as by the reference:
`structure` 997 → 998, `valid_structure` 904 → 905. The gene relaxed to a
valid, novel structure at 0.101 eV/atom above the hull, just outside
`metastable`, so `metastable` and `stable` are unchanged.

## The collision measurement

The per-run novelty reference is affordable because only LeMat-Bulk entries
whose fingerprint collides with a generated one can reach `StructureMatcher`.
Measured against `lemat_bulk_ehull` on `upi73i4k`'s 2500 genes: 627
fingerprints collided, over 795 reference structures — a median of 1 candidate
each and never more than 6. Not re-measured against
`lemat_bulk_fmax1_stress`, whose 4,826,004 fingerprints are 22% more than
`lemat_bulk_ehull`'s, so expect somewhat more of both.
