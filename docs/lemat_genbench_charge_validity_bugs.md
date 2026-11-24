# Bug Report: Numerical Division-by-Zero and Combinatorial Explosion in LeMat-GenBench Oxidation State and Validity Heuristics

**Target Repository:** [`LeMaterial/lemat-genbench`](https://github.com/LeMaterial/lemat-genbench)  
**Affected Modules:** `utils/oxidation_state.py`, `metrics/validity_metrics.py`  
**Severity:** 
- **Critical / Process Hang (fix submitted):** Combinatorial explosion in `compositional_oxi_state_guesses` stalls evaluation for hours on unit cells with $\ge 30$ atoms of a single species.
- **Moderate / False Negatives (merged upstream):** Division by zero in `electronegativity_correlation` emitted `RuntimeWarning: invalid value encountered in divide` and turned valid charge-balanced solutions into `NaN`, causing spurious validity rejections.
- **Design / Domain Limitation:** SMACT metallicity cutoff ($0.70$) and integer-per-site ionic solver reject near-hull DFT phases ($3.8\%$ of 1,000 near-hull LeMat-Bulk structures) as well as $1.91\%$ ($31 / 1,621$) of experimentally observed ICSD-backed structures in `lemat_bulk_fixed_fmax1_stress`.

---

## 1. Bug 1: Division-by-Zero & Spurious Rejection when $\Delta\chi = 0$

**Status:** Fixed in [`kazeevn/lemat-genbench` commit `1ccfa11b`](https://github.com/kazeevn/lemat-genbench/commit/1ccfa11b) (2026-09-30) and merged upstream in [`LeMaterial/lemat-genbench` PR #91](https://github.com/LeMaterial/lemat-genbench/pull/91) (commit `2a708c8a`, 2026-11-20, closing [issue #90](https://github.com/LeMaterial/lemat-genbench/issues/90)).

### Location (pre-fix code)
`lemat_genbench/utils/oxidation_state.py`, function `electronegativity_correlation`:

```python
def electronegativity_correlation(
    elements: list[str],
    oxidation_states: list[int | float]
) -> float:
    ...
    en_vals = []
    for el in elements:
        try:
            en_vals.append(Element(el).X)
        ...
    if len(en_vals) != len(oxidation_states):
        return np.nan
    else:
        corr = np.corrcoef(oxidation_states, en_vals)[0, 1]
        return corr
```

### Mechanism & Failure Mode
When candidate oxidation state solutions are ranked for materials containing elements with identical Pauling electronegativities—such as **Copper ($\chi = 1.90$)** and **Silicon ($\chi = 1.90$)**—the vector `en_vals` has zero variance. The earlier Ni/P example was incorrect: pymatgen reports $\chi(\mathrm{Ni}) = 1.91$ and $\chi(\mathrm{P}) = 2.19$.

$$\text{std}(\mathbf{en\_vals}) = 0.0$$

Inside NumPy's `np.corrcoef` implementation (`numpy/lib/_function_base_impl.py:3036-3037`):
```python
c = cov(x, y, rowvar, dtype=dtype)
stddev = np.sqrt(np.diag(c))
c /= stddev[:, None]  # line 3036: divides by 0.0
c /= stddev[None, :]  # line 3037: divides by 0.0
```
Dividing by `stddev = 0.0` emits:
```
RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
```
and returns `corr = np.nan`.

Downstream in the pre-fix fallback of `ChargeNeutralityMetric.compute_structure`:
```python
try:
    score = -compositional_oxi_state_guesses(
        composition,
        all_oxi_states=True,
        max_sites=-1,
        target_charge=0,
        oxi_states_override=None,
    )[2][0]
except IndexError:
    return 10.0
return 0.0 if score > 0.0 else 10.0
```
Because `score` is `NaN`, `score > 0.0` evaluates to `False`. If earlier validity checks did not accept the structure, this fallback assigned a charge deviation of `10.0` even though charge-balanced oxidation-state solutions were found (e.g. $4\text{Cu}^{4+} + 3\text{Cu}^{3+} + 25\text{Si}^- = 0$ for $\text{Cu}_7\text{Si}_{25}$).

### Minimal Reproduction
```python
from pymatgen.core import Composition
from lemat_genbench.utils.oxidation_state import compositional_oxi_state_guesses

# Both Cu and Si have Pauling electronegativity 1.9
comp = Composition("CuSi")
guesses = compositional_oxi_state_guesses(
    comp, all_oxi_states=True, max_sites=-1, target_charge=0, oxi_states_override=None
)
print("Scores:", guesses[2])
# Before the fix: RuntimeWarning; scores are NaN
# After the fix: no warning; all returned scores are 0.0
```

`CuSi` reproduces Bug 1 without the large-site search cost described in Bug 2.

### Implemented Fix
When either correlation vector has zero variance ($\text{std} < 10^{-6}$), Pearson correlation is undefined. The function returns a neutral score (`0.0`) without calling `np.corrcoef`. Missing electronegativity data and mismatched inputs still return `NaN`:

```python
def electronegativity_correlation(
    elements: list[str],
    oxidation_states: list[int | float]
) -> float:
    ...
    if len(en_vals) != len(oxidation_states):
        logger.error("Mismatch in array lengths for correlation calculation")
        return np.nan

    if len(en_vals) <= 1:
        return 0.0
    if np.std(en_vals) < 1e-6 or np.std(oxidation_states) < 1e-6:
        return 0.0

    return float(np.corrcoef(oxidation_states, en_vals)[0, 1])
```

In `validity_metrics.py`:
```python
score = -output[2][0]
return 0.0 if score >= 0.0 else 10.0
```

Regression tests cover constant electronegativity, constant oxidation states, and acceptance of a neutral Cu/Si fallback score.

---

## 2. Bug 2: Combinatorial Explosion on Structures with $\ge 30$ Sites per Element

**Status:** Fixed on [`kazeevn/lemat-genbench` branch `fix/oxi-state-combinatorial-explosion`](https://github.com/kazeevn/lemat-genbench/commit/0ed87311) (commit `0ed87311`, 2026-11-24).

### Location
`lemat_genbench/utils/oxidation_state.py`, function `compositional_oxi_state_guesses`:

```python
for idx, el in enumerate(elements):
    ...
    for oxid_combo in combinations_with_replacement(oxids, int(el_amt[el])):
        # check to make sure none of the oxidation states deviate by more than 1 
        if max(oxid_combo) - min(oxid_combo) <= 1: 
            oxid_sum = sum(oxid_combo)
            ...
```

### Mechanism & Failure Mode
The number of elements generated by `itertools.combinations_with_replacement(oxids, N)` is:

$$\binom{N + |\text{oxids}| - 1}{N}$$

For standard compounds in materials databases:
* In OQMD structure `oqmd-9905762` ($\text{CuSi}_{64}$): $N = 64$, $|\text{oxids}| = 8$ for Si.
  $$\binom{64 + 8 - 1}{64} = \binom{71}{7} = \mathbf{1,304,969,540 \text{ combinations}}$$
  Evaluating this single structure in pure Python takes **several hours** and consumes 100% CPU.
* Even for moderate unit cells with $N = 30$ and $|\text{oxids}| = 6$:
  $$\binom{30 + 6 - 1}{30} = \binom{35}{5} = \mathbf{324,632 \text{ combinations}}$$

However, the subsequent line:
```python
if max(oxid_combo) - min(oxid_combo) <= 1:
```
discards **$>99.999\%$** of these tuples! 

Any multiset where $\max - \min \le 1$ can only contain at most two adjacent values $(k, k+1)$ from `oxids`. For an element with $N$ atoms, there are only two possibilities:
1. All $N$ atoms share the exact same state $k$ ($1$ combination per $k$).
2. $m$ atoms have state $k$ and $(N - m)$ atoms have state $k+1$, with $1 \le m < N$ ($N - 1$ combinations per adjacent pair).

Total combinations satisfying $\max - \min \le 1$ is at most:
$$|\text{oxids}| + (|\text{oxids}| - 1)(N - 1) \approx \mathbf{K \cdot N}$$
Generating $1.3 \times 10^9$ combinations to find $\sim 500$ valid ones is an unnecessary algorithmic bottleneck that causes benchmark evaluations on large cells to freeze.

### Minimal Reproduction
```python
import time
from pymatgen.core import Composition
from lemat_genbench.utils.oxidation_state import compositional_oxi_state_guesses

# CuSi32 (modest 33-atom cell)
start = time.time()
compositional_oxi_state_guesses(
    Composition("CuSi32"), all_oxi_states=True, max_sites=-1, target_charge=0, oxi_states_override=None
)
print(f"CuSi32 took: {time.time() - start:.2f} s")  # > 15 seconds

# CuSi64 (standard 65-atom cell from OQMD)
# Takes > 2 hours!
```

### Implemented Patch (`0ed87311`)
Replaced brute-force enumeration in `lemat_genbench/utils/oxidation_state.py` with two exact, output-preserving helpers:

1. **`_bounded_oxi_combos(oxids, n_atoms)`:** Direct $O(K \cdot N)$ construction of `(k,) * n_atoms` and adjacent `(k1,) * m + (k2,) * (n_atoms - m)` (`k2 - k1 == 1`) combinations for integer oxidation states, with a windowed `combinations_with_replacement` fallback when non-integer overrides are supplied.
2. **`_iter_target_sum_tuples(el_sums, target_charge)`:** Suffix min/max-bounded DFS with $O(1)$ `Counter` lookup at the final element, replacing `product(*el_sums)` + `sum(x) == target_charge` for high-site multi-element compositions.
3. Per-element `oxi_probs` precomputation when `not all_oxi_states`, avoiding repeated `Species` instantiation.

**Measured post-fix runtime** (`all_oxi_states=True`, `max_sites=-1`, `target_charge=0`):
- `CuSi32`: **1.15 ms** (down from >15 s; returns `()`, no charge-balanced solution exists because $32 \times 1 > \max(\text{Cu}) = 4$).
- `CuSi64`: **0.63 ms** (down from >2 hours; returns `()`).
- `Cu17Si64` (coprime, 64 reduced Si sites): **0.72 ms** (returns 5 charge-balanced solutions, top score `0.0`).

---

## 3. Domain Limitation: False Negatives on Experimental (ICSD) and Near-Hull Intermetallics, Subvalent Phases, Polyanions & Electrides

### Problem
In `validity_metrics.py`, non-insulating structures are supposed to be exempted from charge neutrality checks via SMACT's `metallicity_score`:
```python
if metallicity_score(Composition(structure.formula)) > 0.70:
    return 0.0
```

However, the SMACT metallicity heuristic and integer-per-site ionic solver suffer from several blindspots:
1. **Metalloids (Si, Ge, B, As, Sb, Te) are counted as non-metals.**
   Any intermetallic silicide, boride, or germanide with $\ge 35\text{ at}\%$ metalloid falls below the $0.70$ threshold.
2. **Hard threshold cliff edge.**
   Ordered metallic antiperovskites like $\text{Tb}_2\text{TmMgC}$ (Alexandria `agm005013222`, $E_\text{hull} = 0.0299$ eV/atom) have a metallicity score of **0.6945**, failing the bypass by $0.0055$.
3. **Subvalent, metallic chalcogenide/pnictide/halide, and electride phases.**
   Phases with metal–metal bonding or delocalized electrons—such as $\text{LuSe}$ ($\text{Lu}^{3+}\text{Se}^{2-}\cdot e^-$), $\text{Ti}_2\text{P}$, or layered $\text{Ba}_2\text{Ca}_4\text{As}_3\text{H}_2$ ($[(\text{Ba}_2\text{Ca}_4)^{12+} (\text{As}^{3-})_3 (\text{H}^-)_2] \cdot e^-$)—cannot be balanced by integer site charges.
4. **Polyanion / homonuclear-bonded species.**
   Experimental salts with homonuclear bonds (e.g. azide $\text{BaN}_6$ with $\text{N}_3^-$, polyphosphide $\text{SrP}_3$, Zintl $\text{Rb}_3\text{As}_7$) require fractional average oxidation states (such as $-1/3$) that do not exist in element-wise integer lists.

### Experimental (ICSD-Backed) Audit (2026-11-24)
Reproducible via [`scripts/audit_lemat_genbench_charge_validity.py`](../scripts/audit_lemat_genbench_charge_validity.py) on all **1,621 ICSD-backed** (`icsd_backed == 1.0` in `lemat_bulk_fixed_fmax1_stress_icsd`) structures in the `val` + `test` splits of `lemat_bulk_fixed_fmax1_stress`, evaluated against `lemat-genbench` commit `0ed87311`:

```bash
.venv/bin/python scripts/audit_lemat_genbench_charge_validity.py \
  --lemat-python /path/to/lemat-genbench/.venv/bin/python
```

- **Overall ICSD rejection rate (`charge_deviation > 0.1`):** **31 / 1,621 (1.91%)** (0 evaluation errors).
- **Near-hull ICSD ($E_\text{hull} < 0.05\text{ eV/atom}$):** **24 / 1,145 (2.10%)**.
- **Mutually exclusive breakdown of the 31 rejected ICSD structures:**
  1. **All metal / metalloid (`15 / 31`, `0.93%` of ICSD):** e.g. `mp-17185` (`Ca4Si3Pd4`, $E_\text{hull}=0.0000$, `metal_score=0.6365`), `mp-1197413` (`Sc9Te2`, $0.0000$, `0.6552`), `mp-972463` (`Tm5Sb3`, $0.0000$, `0.6324`), `mp-569281` (`HoB4`, $0.0001$, `0.3299`), `mp-28612` (`Na(LiSi2)3`, $0.0003$, `0.4235`), `mp-570539` (`Rb3As7`, $0.0049$, `0.3099`), `mp-8234` (`BaTe3`, $0.0124$, `0.3249`), `mp-980420` (`Ti3Si`, $0.0166$, `0.6487`).
  2. **Non-all-metal/metalloid with dominant species $\ge 85\text{ at}\%$ (`3 / 31`):** `mp-554098` (`Ti6O`, $0.0020$, `0.6306`, $85.7\%\text{ Ti}$ interstitial suboxide), `mp-1707` (`BaN6`, $0.0413$, `0.2377`, $85.7\%\text{ N}$ azide), `mp-28905` (`SrC20`, $0.2852$, `0.2242`, $95.2\%\text{ C}$ intercalated graphite).
  3. **Other non-all-metal/metalloid (`13 / 31`, `0.80%` of ICSD):** e.g. `mp-414` (`LuSe`, $0.0000$, `0.3735`), `mp-561153` (`XeF3`, $0.0000$, `0.2169`), `mp-631194` (`La3Ga4Cl`, $0.0017$, `0.6178`), `mp-11108` (`SrP3`, $0.0017$, `0.3094`), `mp-1179773` (`Sr2(PPd)3`, $0.0033$, `0.6190`), `mp-866714` (`La3NiBr3`, $0.0034$, `0.6142`), `mp-29807` (`Hf27P16`, $0.0101$, `0.5681`), `mp-30218` (`Ti2P`, $0.0161$, `0.5987`), `mp-582242` (`La8Ni4Br7`, $0.0269$, `0.6486`).

In an earlier audit of 1,000 mixed (DFT + experimental) structures from LeMat-Bulk with $E_\text{hull} < 0.1\text{ eV/atom}$, **3.8% were rejected solely by `charge_deviation > 0.1`** (including `Tb2TmMgC`, `CuSi64`, and `Ba2Ca4As3H2`).

### Candidate Improvements & Design Trade-offs

1. **All-Metal/Metalloid Intermetallic Bypass:**
   If all constituent elements in a crystal are metals or metalloids (no halogens, chalcogens except Te, or N/P/C/H), bypass charge balancing:
   ```python
   if all(Element(el).is_metal or Element(el).is_metalloid for el in composition.elements):
       return 0.0
   ```
   *Trade-off:* Exempts roughly half of rejected ICSD structures (`15 / 31`), but also exempts III-V / Zintl semiconductors (e.g. `GaAs`, `AlAs`, `InSb`) and alters benchmark validity numbers (including bypassing the equal-electronegativity `CuSi` fallback path tested in PR #91).
2. **Dilute Solid Solution / Doped Matrix Bypass:**
   Bypass charge balancing when a single elemental metal or covalent semiconductor (e.g. Si, Ge, C, Fe) comprises $\ge 85\text{ at}\%$ of a crystal (covering `Ti6O`, `SrC20`, and `CuSi64`, though care is needed to avoid matching polyanion salts like `BaN6`).
3. **Remaining Gap:**
   Even with (1) and (2), `13 / 31` (`0.80%`) of rejected ICSD structures—predominantly subvalent/metallic phosphides, halides, and chalcogenides (`LuSe`, `Ti2P`, `Hf27P16`, `La3NiBr3`) and polyphosphides (`SrP3`)—still require a non-ionic/band-structure or homonuclear-bond criterion.
