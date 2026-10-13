# Stale Materials Project energies in the LeMat-Bulk PBE hull

> **STATUS: MEASUREMENT, 2026-10-14. Not fixed.** Nothing in the data store or in
> any training cache has been changed yet. See [what to do](#what-to-do).
> Audit outputs and scripts: `$WYFORMER_RUNS/kappagen_genes/hull_audit/`.

## Summary

Some LeMat-Bulk PBE entries that come from the Materials Project carry an energy that
MP itself has since replaced. Where such an entry is too low, it becomes a vertex of
the PBE convex hull. It then pushes up the `e_hull` of every other structure in its
chemical system, by up to 1.5 eV/atom.

The entries fall into two groups:

1. **Bad GGA relaxation energies, no +U involved.** LeMat has the final energy of an
   old MP *structure optimization*. A later MP *static* run on the same final
   structure is 0.9–2.1 eV/atom higher. Examples: Sb₂Te₃ `mp-1079094`, Bi₂Te₃
   `mp-1182783`, MoTe `mp-1237627`.
2. **Mixed plain-GGA and GGA+U energies.** For oxides and fluorides of the +U elements
   (V, Cr, Mn, Fe, Co, Ni, Mo, W), LeMat has a plain-GGA energy where MP now uses
   GGA+U. LeMat-Bulk was meant to make this mixing impossible. Examples: W–O
   `mp-636148`, K–Mo–O `mp-28305`.

Materials Project's current data is fine; the errors are in LeMat-Bulk. Every
LeMat-Bulk-derived PBE hull in this repository inherits them. That covers `e_hull` and
`e_form` in `lemat_pbe_ehull.csv.gz`, the `energy_above_hull` and gene-minimum
formation-energy training labels, and `HullLookup` (items 1–3 of
[`e_hull_definitions.md`](e_hull_definitions.md)). The published `dft` split of
`LeMat-Bulk-MLIP-Hull` contains the same entries.

## How it was found

The kappagen gene campaign kept genes predicted within 0.15 eV/atom of this hull. The
campaign generated 100k genes for each of 215 ternaries and their 115 binaries, and
was merged in `5c2cdf6` (W&B run `a8734z77`).

The Sb–Te systems kept almost nothing. Sb–Te itself kept 24 genes and Fe–Sb–Te 51,
against a median of about 8,500 per system. The hull at Sb₂Te₃ sat at a formation
energy of −1.144 eV/atom, while experiment gives about −0.11.

## The two telluride entries

| | `mp-1079094` Sb₂Te₃ | `mp-1201` Sb₂Te₃ | `mp-1182783` Bi₂Te₃ | `mp-34202` Bi₂Te₃ |
|---|---|---|---|---|
| space group | C2/m (high-pressure phase) | R-3m | C2/m | R-3m |
| volume per atom (Å³) | 27.66 | 34.05 | 71.74 | 36.66 |
| LeMat PBE `e_form` (eV/atom) | **−1.144** | −0.126 | **−0.469** | −0.239 |
| ORB-v3 `e_form`, single point on the same structure | +0.337 | −0.137 | +0.400 | −0.237 |
| LeMat `max_force` (eV/Å) | 0.075 | 0.005 | 0.084 | 0.0004 |

- **The Sb₂Te₃ value is impossible.** `mp-1079094` is 19% denser than the ground
  state. Compressing a crystal raises its energy, so it cannot sit 1 eV/atom below it.
- **ORB isolates the two entries.** ORB was trained on the same kind of PBE data. On
  every other Sb–Te, Bi–Te and Sb–Se entry it reproduces the PBE `e_form` to within a
  median of ±0.01 eV/atom. It disagrees by 1.48 and 0.87 eV/atom on these two.
- **The damage to the rest of the system is large.** With `mp-1079094` on the hull,
  the experimental ground state `mp-1201` gets `e_hull` = 1.018 eV/atom in
  `lemat_pbe_ehull.csv.gz`.

### Where the energies come from

These were queried from the MP API on 2026-10-14. The API now returns new-style
IDs, so `mp-1079094` is `mp-aaacjkhq`. The match was confirmed by cell volume.

| MP task | type | date | energy (eV/atom) |
|---|---|---|---|
| `aaacjkhq` | GGA Structure Optimization | 2018-04-14 | **−4.6812**, which is LeMat's value |
| `aaaecswc` | GGA Static, same structure | 2021-01-23 | −3.2009, MP's current value |
| `aaaeizrz` | GGA Static, same structure | 2021-03-09 | −3.1869 |

- **The runs are otherwise identical.** All three end at the same volume
  (276.55 Å³) with the same POTCARs (`PAW_PBE Sb 06Sep2000`, `Te 08Apr2002`), ENCUT 520
  and ISPIN 2. None sets PSTRESS, +U or a meta-GGA functional.
- **MP's current verdict for `mp-1079094`:** `e_form` = +0.084 eV/atom and
  `e_above_hull` = 0.467 eV/atom.
- **`mp-1182783` follows the same pattern.** LeMat's −3.9079 eV/atom is exactly the
  2018 relaxation `aaacphrr`. MP's 2021 static run `aaaeatue` gives −3.0438, and MP's
  current `e_above_hull` for it is 0.634 eV/atom.

## How widespread

### The detector

`LeMat-Bulk-MLIP-Hull`'s `dft` split holds every LeMat-Bulk entry within 1 meV/atom
of the PBE hull: 144,127 rows. It stores each row's PBE energy (`true_energy`)
alongside five MLIP energies (`orb_conserv_inf`, `orb_direct_20`, `uma`,
`mace_omat`, `mace_mp`).

An entry is flagged when its PBE energy is below **all five** MLIPs by more than a
threshold, measured per atom:

| PBE below all five MLIPs by more than | entries |
|---|---|
| 0.1 eV/atom | 2,466 |
| 0.2 eV/atom | 424 |
| 0.3 eV/atom | 208 |
| 0.5 eV/atom | 139 |
| 1.0 eV/atom | 48 |

- **Sources of the 424 entries at 0.2 eV/atom:** 223 from MP, 186 from Alexandria
  (`agm`) and 15 from OQMD.
- **Element count:** 25 binaries, 289 ternaries, 99 quaternaries and 11 with five or
  six elements.
- **The worst one is `mp-1400697`** (Bi–O–Pd–Pt). It has a 40.6 eV/atom gap,
  `max_force` = 80.9 eV/Å and `e_form` = −37.9 eV/atom.

### Checking the MP-sourced entries against MP

The current MP `GGA_GGA+U` uncorrected energy was fetched for each of the 223 flagged
MP entries. Our hull uses LeMat's raw `energy` with no MP2020 correction, so the
uncorrected energy is the comparable number. A control set of 200 random unflagged MP
hull entries was fetched the same way.

| | entries | found in MP | \|LeMat − MP\| < 0.01 | LeMat below MP by > 0.1 | LeMat − MP, 5% / 50% / 95% |
|---|---|---|---|---|---|
| flagged | 223 | 223 | 4 | 219 | −1.486 / −0.554 / −0.204 |
| control | 200 | 198 | 191 | 1 | −0.008 / 0.000 / +0.005 |

- **The flag tracks the real error.** Across the flagged entries, the LeMat − MP gap
  correlates with the PBE − MLIP gap at **0.997**. Wherever the MLIPs disagree with
  LeMat, they agree with MP's current value. The MLIP consensus is therefore a
  reliable detector of this defect.
- **Sorting the 219 by composition** puts 203 in the +U group: oxides or fluorides
  with a +U element, median gap −0.535 eV/atom. The other 16 have no +U involved,
  median gap −0.833. This split is by composition only. Task by task, I checked just
  these:
  - `mp-636148` (W–O): LeMat −9.0861 is the 2019 plain-GGA relaxation `aaabkfbg`. MP
    now uses GGA+U static `aaaeitbs` at −7.4859.
  - `mp-28305` (K–Mo–O): LeMat −8.3331 is the 2014 plain-GGA relaxation `aaaabpwr`. MP
    now uses GGA+U static `aaaedfbx` at −6.8470.
  - `mp-1237627` (Mo–Te): LeMat −10.1358 is the 2019 GGA relaxation `aaacskvb`. MP's
    GGA statics give −7.998.
- **Four flagged entries match MP.** `mp-863` (Co–Pu), `mp-681150` (Pu–Rh) and
  `mp-1091393` (Ga–Ni–Pu) are probably the MLIPs mishandling plutonium.
  `mp-1518724` (Ba–Cr–Fe–Gd–O, GGA+U only, 3.07 eV/atom gap) is unexplained.

### Not checked yet

The 186 Alexandria entries and 15 OQMD entries can't be checked against MP. Many
involve Pu, Np or Pa, where the MLIPs are weak. Others are Cr fluorides and oxides
with gaps of 2–4 eV/atom, which look like the same defect.

Below 0.2 eV/atom the flag starts to catch MLIP errors as well. The campaign's own
weak flags are bcc Fe `mp-13`, Fe–Ni, Co–Fe, Bi–Co, Ag–Bi and Fe–Se, at 0.10–0.27
eV/atom. They involve magnetic 3d metals and are not evidently bad labels.

## Effect on the kappagen campaign

Inside the campaign's 330 systems, only `mp-1079094` and `mp-1182783` are flagged at
0.2 eV/atom. Rebuilding the hull without them and keeping the regressor's predictions
unchanged gives:

- **Affected systems:** the 25 that contain both Sb and Te, or both Bi and Te.
- **Affected genes:** 534,204 unique genes.
- **Hull shift:** a median of 0.03–0.72 eV/atom upward, depending on the system.
- **Genes kept at e_hull ≤ 0.15:** 61,487 rise to **297,618**.

| system | kept before | kept with a clean hull |
|---|---|---|
| Sb–Te | 24 | 8,923 |
| Fe–Sb–Te | 51 | 9,054 |
| Cu–Sb–Te | 74 | 12,963 |
| Ag–Sb–Te | 174 | 18,195 |
| Bi–Te | 2,573 | 11,151 |

The other 305 systems don't change.

**The regressor learned the bad label too.** `min_energy_adamw_wsd_h7x_xl` predicts
−1.143 eV/atom for the Sb₈Te₁₂ C2/m gene (sg 12), which is `mp-1079094`'s own gene.
Against a clean hull it would rank as a candidate 1.02 eV/atom below the hull. Any
regressor trained on gene-minimum formation energies from this archive will do the
same for every flagged entry's gene.

## What is affected

- **`lemat_pbe_ehull.csv.gz`:** `e_form` and `e_hull`, and every hull built from them.
  That includes `HullLookup`, `wyformer-gene-screen` and `wyformer-dft-screen`.
- **Training labels:** `energy_above_hull`, clipped for conditioning, and the
  gene-minimum formation energy. The flagged entries themselves carry wrong labels,
  and so does every structure in their chemical systems, through the hull.
- **The published `dft` split of `LeMat-Bulk-MLIP-Hull`.**
- **Not directly affected:** hulls built from one MLIP's own energies (item 4 of
  [`e_hull_definitions.md`](e_hull_definitions.md)). Neither telluride entry is in the
  `orb_conserv_inf` split. Models trained on the bad PBE labels still feed into those
  workflows indirectly.

## What to do

1. **MP-sourced entries:** replace LeMat's energy with MP's current `GGA_GGA+U`
   uncorrected energy. Then rebuild `lemat_pbe_ehull.csv.gz` under a new name, as the
   store is read-only.
2. **Alexandria and OQMD entries:** exclude the flagged ones, or check them against
   their source databases.
3. **Rebuild the training caches and retrain** the energy regressors and the e_hull
   CFG models.
4. **Report upstream** to LeMat-Bulk, with `mp-1079094` (bad relaxation energy) and
   `mp-636148` (plain GGA where GGA+U is used) as minimal examples.
5. **Interim fix for the kappagen campaign:** re-select against a hull without the
   flagged entries, and drop genes that match flagged entries.

## Reproducing

Scripts are in `$WYFORMER_RUNS/kappagen_genes/hull_audit/scripts/`; each takes the
scratch directory as its argument:

- `dftscan.py`: the MLIP-consensus scan of the `dft` split. Writes
  `suspect_dft_hull_entries.csv`.
- `sbte_orb.py`: ORB single points on the Sb/Bi chalcogenide entries. Writes
  `sbte_orb.parquet`.
- `mp_audit.py`: flagged and control entries against the current MP thermo
  documents. Writes `mp_audit.parquet`. It needs `MP_API_KEY` from `.env`; it talks to
  the plain REST API, because `mp_api` is not installed in the venv.
- `mpq.py`, `mpt.py`, `mpw.py`: the per-task provenance queries above.
- `rescore.py`: the clean-hull re-score of the campaign. Writes
  `rescored_affected.parquet`.

Run them inside the image with `scripts/platforms/aspire2a/run_in_singularity.sh`.
