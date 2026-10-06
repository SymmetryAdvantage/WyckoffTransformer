# Can the Cu–Ge–Te hull states make a switchable material?

> **STATUS: MEASUREMENT, 2026-10-07.** Branch `cu-ge-te-switching` (from `main`
> at `ebd0ad6`). Scripts: `scripts/cu_ge_te_switching/s0`–`s9`. Outputs:
> `artifacts/cu_ge_te_switching/` (gitignored, on zeus). Input: the hull
> campaign of [`cu_ge_te_hull_mapping_20260929.md`](archive/cu_ge_te_hull_mapping_20260929.md).
> Compute: one RTX 6000 Ada (GPU 0, borrowed for ~7 h) and the zeus CPUs.

The question: the hull campaign found ~17,700 Cu–Ge–Te structures within
100 meV/atom of the ORB hull. Can some of them be used for a material that
switches reliably between metastable states under an external stimulus? A switch
needs states of the **same composition** that (i) are distinct minima, (ii) are
separated by a barrier high enough to hold the state and low enough for a
stimulus to cross it, and (iii) differ in a property that the stimulus couples
to (volume for pressure, polarisation for a field, entropy for heat).

## Summary

1. **The map overstates how many states there are.** Correcting the reference
   hull, removing duplicates and dropping vacuum-gap artefacts leaves **55
   distinct ternary states within 50 meV/atom, in 26 compositions** (the campaign
   counted 65 ternary entries). ORB also compresses polymorph energy gaps:
   PBE puts them **2–4× further apart**.
2. **Within a composition, the low-energy ternary states are almost all
   different Cu/Ge arrangements on one fcc Te framework** (zinc-blende
   derived), plus one wurtzite-type stacking variant. Moving between them means
   exchanging cations, which is diffusion. They are not separate switchable
   states; at temperature they belong to one cation-disordered phase or stay
   frozen. In the binaries, GeTe's extra "polymorphs" are ferroelastic domain
   variants of R3m, and Cu₂Te/Cu₃Te₂ have dozens of nearly identical Cu
   arrangements (mobile Cu sublattice).
3. **The switches that do exist are the known ones**, and the computation puts
   numbers on them:
   * **Heat, amorphous ↔ crystal (phase-change memory).** ORB+D3 melt-quench
     glasses lie 79 (Cu₂GeTe₃), 98 (Cu₃GeTe₄) and 156 (GeTe) meV/atom above
     their crystals. The Cu ternaries are predicted to *densify* by 11% on
     amorphisation, the opposite of GeTe (+2.5%). That sign depends on the
     dispersion correction, so it needs an experimental check.
     Crystallisation at 600 K: PENDING.
   * **Heat, ferroelectric ↔ paraelectric GeTe.** ORB reproduces the
     transition: the average rhombohedral strain disappears at 675 ± 25 K
     (ORB+D3: 575 ± 25 K; experiment ~700 K) while the local Peierls
     distortion survives (order–disorder character). This switch is reversible
     and has no memory.
   * **Electric field, +P ↔ −P in GeTe.** The coherent reversal barrier is
     56 meV/atom at fixed cell; relaxing the cell gives a 57 meV/f.u. double
     well in ORB versus 38 meV/f.u. in PBE. The polar Cu ternaries are not
     field-switchable: reversing a tetrahedral (zinc-blende or wurtzite
     derived) polar structure moves atoms by ~1–1.5 Å and costs
     200–360 meV/atom, and Cu₃GeTe₄ is metallic by electron count anyway.
   * **Pressure, tetrahedral → dense Cu₂GeTe₃.** Enthalpies cross at 1.7 GPa in
     ORB and ~4 GPa in PBE, with an 11% volume collapse and Cu/Ge going from
     4- to 8–9-fold coordination. This is a real, large property change, but at
     gigapascals and reconstructive (large hysteresis expected).
4. **Verdict.** The hull map is a good list of candidate *compositions* and of
   which crystal each composition crystallises into. It is not a list of
   switchable states: the multiplicity it shows is configurational, not
   polymorphic. For a device in this system, the realistic route is the
   phase-change one (amorphous ↔ crystalline Cu₂GeTe₃ / Cu-doped GeTe), and the
   computation that matters next is crystallisation kinetics and electronic
   contrast, not more polymorph search.

## 1. Corrections to the hull map

**Reference relaxation (`s0_relax_references.py`).** The `LeMat-Bulk-MLIP-Hull`
ORB energies are single points on DFT geometries
(`verify_hull_energies`), while every generated structure was relaxed with ORB.
Relaxing the 42 references with ORB lowers them by 0–11 meV/atom (CuTe 11.1,
Te 8.4, GeTe 2.8, Cu₃Te₂ 1.9, Cu₃GeTe₄ 0.3). Consequences:

| Campaign claim | After relaxing references |
|---|---|
| New GeTe polymorph lowers the hull by 2.7 meV/atom | **Gone**: it is the reference relaxing (2.8 meV) |
| New CuTe polymorph lowers the hull by 11.0 meV/atom | **Gone**: reference relaxes by 11.1 meV |
| Cu₃Te₂ lowered by 12.3 meV/atom | Survives, 10.4 meV/atom |
| Cu₃GeTe₄ lowered by 2.9 meV/atom | Survives, 2.6 meV/atom (I‑42m below the Pmn2₁ reference) |
| ≤ 50 / ≤ 100 meV/atom generated entries: 3,932 / 17,734 | 3,796 / 17,246 |

The corrected hull vertices are Cu, Ge, Te, GeTe, CuTe, Cu₃Te₂ (generated),
Cu₃GeTe₄ (generated, I‑42m) and Cu₅Ge (generated). The campaign report's
"9 vertices / 4 elemental vertices" has 8 vertices and 3 elements.

**Duplicates and artefacts (`s1_screen_pairs.py`).** Grouping with pymatgen's
default `StructureMatcher`: the 3,838 entries within 50 meV/atom are 1,974
distinct states. Of these, 101 have an atom-free slab wider than 4 Å (GeTe 30,
Ge–Te layered compounds most of the rest): stacks of layers that ORB, which
has no dispersion, barely binds. Examples: GeTe P6₃/mmc at +7.6 meV/atom with
49% more volume than R3m. Real Cu–Te layer gaps reach ~3.6 Å, hence the 4 Å
cut.

**ORB versus PBE (`s5_dft.py`, pslibrary 1.0.0 PAW, 50/400 Ry, PBE single points
at ORB geometries).** The ordering is preserved, the gaps are not:

| Pair | ORB (meV/atom) | PBE (meV/atom) |
|---|---:|---:|
| Cu₃GeTe₄ Pmn2₁ − I‑42m | 2.6 | 8.0 |
| Cu₃GeTe₄ P‑43m − I‑42m | 4.7 | 12.0 |
| Cu₂GeTe₃ Cmc2₁ − Imm2 | 4.8 | 21.8 |
| Cu₂GeTe₃ dense Imm2 − Imm2 | 28.6 | 55.4 |
| Cu₂GeTe₃ dense Cmcm − Imm2 | 29.8 | 65.0 |
| GeTe Fm‑3m − R3m (relaxed cells, per atom) | 28.3 | 19.0 |

ORB's systematic softening matters for counting: a 50 meV/atom window in ORB is
roughly a 100+ meV/atom window in PBE.

## 2. What the states are

`s1` pairs every two states of a composition and tests whether a loose matcher
(`ltol 0.3, stol 0.6, angle 15°`, supercells allowed) maps one onto the other.
For the ternaries, the Te sublattice alone was compared with fcc and hcp:

| Composition | States ≤ 60 meV/atom (ORB) | Te sublattice | Difference between states |
|---|---|---|---|
| Cu₃GeTe₄ | I‑42m (0), Pmn2₁ (2.6), P‑43m (4.7) | fcc, hcp, fcc | Cu/Ge order (I‑42m ↔ P‑43m); stacking (Pmn2₁, enargite-like) |
| Cu₂GeTe₃ | Imm2 (15), Cmc2₁ (20) | fcc, fcc | Cu/Ge order |
| Cu₂GeTe₃ | Imm2 (44), Cmcm (45) | neither | dense, Cu–Ge and Te–Te "wrong" bonds, Cu CN 5–9 |
| Cu₇GeTe₈ | F‑43m ×2, I‑42m ×2, Pm | all fcc | Cu/Ge order and vacancies |
| CuGeTe₂ | six C2/m, Pnma, P2₁/m, Pbcm within 6 meV | fcc (one hcp) | Cu/Ge order |

All tetrahedral ternary states share a Te framework, and the I‑42m ↔ P‑43m and
Imm2 ↔ Cmc2₁ pairs have no species-preserving displacive map. Every switch
among them needs Cu–Ge exchange. Electron counting separates the two main
ternaries: Cu₂GeTe₃ is charge-balanced (2×1 + 4 = 3×2) and should be a
semiconductor; Cu₃GeTe₄ has one hole per formula unit (3 + 4 < 8) and is a
metal. PBE makes both metallic (a known PBE failure for Cu tellurides), so no
gap is reported here.

In GeTe the extra low-energy states (P2₁, Pmn2₁ at +7–8 meV/atom) map onto R3m
with 5% length strain and a 4° angle change: they are rotated ferroelastic
domain variants of R3m, not new phases.

## 3. Dynamical stability and vibrational entropy (`s2`, `s6`)

Harmonic phonons (phonopy, ORB, 0.01 Å displacements, supercells ≥ 11 Å,
q-mesh density 40) for 81 structures: the ternary states ≤ 50 meV/atom, the main
binaries and the hull vertices. 78 have no imaginary modes; CuTe#2 and two Te
polymorphs are saddles. The hull at temperature (`hull_vs_T.csv`) keeps the
same vertices to 900 K, except that Cu₃Ge comes back on the hull above ~600 K
through vibrational entropy.

Vibrational entropy reverses the order of a ternary pair only once: the two
F‑43m Cu₇GeTe₈ states cross at ~650 K (ΔS = 0.12 k_B/atom). The dense Cu₂GeTe₃
Cmcm state gains 0.20 k_B/atom over Imm2 but is still 13 meV/atom above it at
900 K. Harmonic, fixed volume, no configurational entropy, which for the
cation-ordering variants is the larger term.

## 4. Stimuli

### 4.1 Heat: amorphous ↔ crystal

Melt-quench MD (`s4_md.py --mode meltquench --d3`): 216–224 atoms, Langevin
BAOAB at zero pressure, 10 ps at 1500 K, 15 ps at 1000–1100 K, quench at
10 K/ps, 10 ps at 300 K, then relaxation. Without dispersion the liquid is far
too dilute (GeTe at 1100 K: 39 Å³/atom, ~30% above experiment); ORB+D3(BJ)
(orb-models' `D3SumModel`) gives 30.6 Å³/atom against ~29.7 experimentally, so
all amorphous numbers are ORB+D3.

| | GeTe | Cu₂GeTe₃ | Cu₃GeTe₄ |
|---|---:|---:|---:|
| Crystal V (Å³/atom, ORB+D3, 0 K) | 28.49 | 26.29 | 25.59 |
| Liquid V at ~1000 K | 30.46 | 25.11 | 24.59 |
| Glass V (relaxed) | 29.20 | 23.36 | 22.74 |
| Volume change crystal → glass | +2.5% | −11.1% | −11.2% |
| E(glass) − E(crystal), meV/atom | 156 | 79 | 98 |
| Same gap, PBE-D3(BJ) single points | 151 | 80 | 101 |
| PBE-D3 pressure on the ORB+D3 glass / crystal (GPa) | −0.33 / −0.41 | 0.28 / 0.32 | 0.48 / 0.52 |

In the Cu₂GeTe₃ glass, Cu coordination rises from 4 to 7.2, with 0.43 Cu–Cu
contacts per atom at 2.5–2.6 Å, plus Cu–Ge and Te–Te bonds, and only 36% of Ge
stays tetrahedral. That is where the extra density comes from. Without D3 the
same liquid is 12% *less* dense than the crystal, so the sign of the volume
change is decided by the dispersion term. PBE-D3 agrees with ORB+D3: the
glass−crystal energies match to 1–5 meV/atom, and PBE-D3 leaves both phases at
the same residual pressure (within 0.1 GPa for all three), so the 11%
densification is not an ORB fitting artefact. Whether D3 is the right physics
for Cu tellurides is a separate question that measured film densities should
settle. PBE-D3 at Γ gives the glasses gaps of 5 meV (GeTe) and 31 meV
(Cu₂GeTe₃), against 0.54 eV for crystalline GeTe and a metallic Cu₂GeTe₃
crystal; with the PBE gap problem and a 10 K/ps quench, these are not a
contrast prediction.

Crystallisation at 600 K (`--mode anneal`, 150 ps, crystalline fraction from
Te-sublattice q6 bond order): PENDING.

### 4.2 Heat: ferroelectric GeTe

Heating R3m GeTe at 5 K/ps in a 216-atom pseudo-cubic supercell with a fully
flexible cell (`--mode heat`; `figures/heating.png`, binned in 25 K steps).
The cell-angle deviation from 90° (macroscopic rhombohedral strain) is ~3° at
300 K and falls below 1° between 650 and 700 K with ORB and between 550 and
600 K with ORB+D3; experiment gives ~700 K (625–705 K depending on Ge
vacancies). Above that, the residual 0.4–0.8° is thermal noise of a 216-atom
cell. The mean long-minus-short Ge–Te bond difference barely moves:
0.45–0.50 Å (ORB) and 0.40–0.47 Å (ORB+D3) on both sides of the transition. The
transition is order–disorder: the crystal becomes cubic on average while each
Ge stays off-centre, which is also what EXAFS on GeTe shows. The heating rate
(5 K/ps) and cell size make these ±25 K estimates.

The same protocol on Cu₂GeTe₃ (ORB+D3) keeps the crystal intact to ~850 K.
Cu starts to diffuse at ~850 K (mean-square displacement 3.2 Å² at 900 K, Ge
0.9 Å², Te 0.5 Å²), and the lattice melts at ~890 K on this fast ramp, falling
to the denser liquid's volume. There is no wide temperature window in which
the cations are mobile on a fixed Te framework, so the cation-ordering
variants of §2 interconvert only near melting on MD timescales.

The transition is reversible on cooling and holds no state, so it is a sensor
or actuator mechanism, not a memory.

### 4.3 Electric field: polarisation reversal

`s3b_polar_switching.py` builds the −P variant by inversion, translation and
permutation, removes the net translation, and runs a fixed-cell climbing-image
NEB (ASE, 7 images, fmax 0.03 eV/Å).

| State | Barrier (meV/atom) | Max polar displacement (Å) | P, nominal charges (C/m²) |
|---|---:|---:|---:|
| GeTe R3m | 56.4 | 0.41 | 0.23 |
| Cu₂GeTe₃ Imm2 (ground state) | 207* | 1.44 | 0.88 |
| Cu₂GeTe₃ Cmc2₁ | 264 | 0.96 | 0.50 |
| Cu₃GeTe₄ Pmn2₁ (wurtzite type) | 353* | 0.95 | 0.54 |
| Cu₂GeTe₂ Fmm2 | 356* | 1.17 | 0.95 |
| Cu₂GeTe₂ Pmn2₁ | 212* | 1.52 | 0.29 |
| Cu₂Te Pca2₁ | 43 | 1.05 | 0.24 |
| Cu₃Te₂ Pm | 190* | 1.02 | 0.31 |

\* NEB not converged in 600 steps; the number is the highest image, an upper
bound of the right order. Only GeTe has a sub-0.5 Å reversal path. In the
tetrahedral ternaries every cation has to pass through the plane of its Te
neighbours, which breaks bonds; Cu₂Te and Cu₃Te₂ are metals, where a
polarisation is not defined.

GeTe's Born charges are 3–4× its nominal ±2, so its real polarisation is closer
to the ~0.6 C/m² in the literature. With relaxed cells the GeTe double well
(rocksalt − R3m) is 57 meV/f.u. in ORB and 38 meV/f.u. in PBE at the same
geometries: ORB over-distorts GeTe (Peierls bonds 2.83/3.34 Å versus
2.84/3.16 Å in experiment). Practical caveat: GeTe is a degenerate p-type
conductor (Ge vacancies), so field switching needs thin films with suppressed
carriers.

### 4.4 Pressure

`s8_pressure.py` relaxes each state at 0–10 GPa (ORB, CPU) and compares
enthalpies (figure: `figures/pressure.png`).

* **Cu₂GeTe₃**: tetrahedral Imm2 gives way to the dense Cmcm state at
  ~1.7 GPa in ORB; volume falls 11% (26.5 → 23.5 Å³/atom at 2 GPa). PBE at the
  ORB geometries (0 and 2 GPa) moves the crossing to **~4 GPa**.
* **GeTe**: R3m relaxes continuously to Fm‑3m between 6 and 8 GPa in ORB
  (experiment ~3–4 GPa at room temperature; ORB's deep double well raises it).
* **CuGeTe₂**: a wurtzite-type state becomes lowest above ~2 GPa.
* **Cu₃GeTe₄**: no crossing below 10 GPa; Pmn2₁ approaches I‑42m (0.7 meV/atom
  at 10 GPa).

The Cu₂GeTe₃ transition changes coordination and probably conductivity, but
it needs gigapascals; epitaxial strain gives at most a few tenths of a GPa.

## 5. What did not work and was dropped

A concerted NEB between polymorphs was attempted first. With both endpoints on
the average cell, the 2–5% strain between ferroelastic variants pushed state B
into A's basin. A variable-cell-per-image NEB written for this study was not
trustworthy (images climbed during optimisation) and was removed. Since the
pairs differ by cation order, stacking or domain orientation, a small-cell
concerted barrier would not describe their kinetics in any case: those
transitions go through diffusion, partial dislocations or domain walls.

## 6. Recommendations

1. Treat the hull map as composition-level information. For each composition,
   collapse cation-ordering variants on a common anion framework into one
   phase (the disorder-parent analysis, `cu_ge_te_disorder_parent_inference_20260929.md`,
   does this for the binaries already).
2. If the goal is a device, pursue phase-change behaviour along the
   GeTe–Cu₂GeTe₃ line: crystallisation temperature and speed (longer anneals,
   several temperatures, larger cells), and electronic contrast with a
   functional that opens the Cu-telluride gap (HSE or DFT+U/mBJ).
3. Settle the amorphous density of Cu₂GeTe₃ against experiment before trusting
   ORB+D3 for the Cu ternaries; it determines whether a Cu-rich composition
   reduces the volume change on switching.
4. For pressure switching of Cu₂GeTe₃, compute the reverse path on
   decompression (is the dense phase retained at 0 GPa?) and its electronic
   structure.

## 7. Reproduction

From the worktree root, with `artifacts/cu_ge_te_hull` linked to the campaign
outputs:

```bash
cd scripts/cu_ge_te_switching
../../.venv/bin/python s0_relax_references.py --device cuda
../../.venv/bin/python s1_screen_pairs.py --workers 20 --max-states 12
../../.venv/bin/python s2_phonons.py --per-formula 6 --window 0.05
../../.venv/bin/python s6_free_energy.py
../../.venv/bin/python s3b_polar_switching.py --only "GeTe#0" "Cu2GeTe3#0" ...
../../.venv/bin/python s4_md.py --formula Cu2GeTe3 --mode meltquench --t-liquid 1000 --d3
../../.venv/bin/python s4_md.py --formula GeTe --mode heat --atoms 216 --t-max 1000
../../.venv/bin/python s4_md.py --formula GeTe --mode anneal --from-run meltquench_GeTe_216_s0_d3 --t-anneal 600 --anneal-ps 150 --d3
../../.venv/bin/python s8_pressure.py --workers 5 --threads 3
../../.venv/bin/python s5_dft.py --nproc 16 --states "Cu2GeTe3#0" "Cu2GeTe3#1"
../../.venv/bin/python s7_analyze_md.py && ../../.venv/bin/python s9_figures.py
```

`s5_dft.py` needs the conda Quantum ESPRESSO 7.5 in `/home/kna/mamba/envs/qe`
(the system `pw.x` 6.7 aborts on input) and the pslibrary 1.0.0 PAW files in
`artifacts/cu_ge_te_switching/pseudo/`. `phonopy` is not a project dependency;
install it into the venv with `uv pip install phonopy`.
