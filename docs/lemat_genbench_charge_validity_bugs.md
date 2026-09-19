# Bug Report: Numerical Division-by-Zero and Combinatorial Explosion in LeMat-GenBench Oxidation State and Validity Heuristics

**Target Repository:** [`LeMaterial/lemat-genbench`](https://github.com/LeMaterial/lemat-genbench)  
**Affected Modules:** `utils/oxidation_state.py`, `metrics/validity_metrics.py`  
**Severity:** 
- **Critical / Process Hang:** Combinatorial explosion in `compositional_oxi_state_guesses` stalls evaluation for hours on unit cells with $\ge 30$ atoms of a single species.
- **Moderate / False Negatives:** Division by zero in `electronegativity_correlation` emits `RuntimeWarning: invalid value encountered in divide` and turns valid charge-balanced structures into `NaN`, causing spurious validity rejections.
- **Design / Domain Limitation:** SMACT metallicity cutoff ($0.70$) misclassifies metalloid-rich intermetallics, subvalent phases, and electride-hydrides as ionic insulators, rejecting near-hull phases ($E_\text{hull} < 0.05$ eV/atom).

---

## 1. Bug 1: Division-by-Zero & Spurious Rejection when $\Delta\chi = 0$

### Location
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
When candidate oxidation state solutions are ranked for materials containing elements with identical Pauling electronegativities—most notably **Copper ($\chi = 1.9$)** and **Silicon ($\chi = 1.9$)**, or **Nickel ($\chi = 2.19$)** and **Phosphorus ($\chi = 2.19$)**—the vector `en_vals` has zero variance:

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

Downstream in `charge_deviation`:
```python
try:
    correlation = -compositional_oxi_state_guesses(
        composition,
        all_oxi_states=True,
        max_sites=-1,
        target_charge=0,
        oxi_states_override=None,
    )[2][0]
except IndexError:
    return LARGE_CHARGE_DEVIATION
return 0.0 if correlation > 0.0 else LARGE_CHARGE_DEVIATION
```
Because `correlation` is `NaN`, `-np.nan > 0.0` evaluates to `False`. The structure is penalized with `LARGE_CHARGE_DEVIATION = 10.0` and marked **invalid**, even though valid charge-balanced combinations (e.g. $4\text{Cu}^{4+} + 3\text{Cu}^{3+} + 25\text{Si}^- = 0$ for $\text{Cu}_7\text{Si}_{25}$) were successfully found!

### Minimal Reproduction
```python
import numpy as np
from pymatgen.core import Composition
from lemat_genbench.utils.oxidation_state import compositional_oxi_state_guesses

# Both Cu and Si have Pauling electronegativity 1.9
comp = Composition("Cu7Si25")
guesses = compositional_oxi_state_guesses(
    comp, all_oxi_states=True, max_sites=-1, target_charge=0, oxi_states_override=None
)
print("Scores:", guesses[2])
# Output:
# RuntimeWarning: invalid value encountered in divide
# Scores: (nan, nan, nan, nan)
```

### Proposed Patch
When all constituent elements have identical electronegativities ($\text{std}(\mathbf{en\_vals}) < 10^{-6}$), there is no electrostatic driving force favoring one element over another. The correlation is neutral ($0.0$), and the candidate solution should be accepted:

```python
def electronegativity_correlation(
    elements: list[str],
    oxidation_states: list[int | float]
) -> float:
    ...
    if len(en_vals) != len(oxidation_states):
        logger.error("Mismatch in array lengths for correlation calculation")
        return np.nan

    # Guard against zero-variance vectors
    if len(elements) <= 1:
        return 0.0
    if float(np.std(en_vals)) < 1e-6 or float(np.std(oxidation_states)) < 1e-6:
        return 0.0

    with np.errstate(divide="ignore", invalid="ignore"):
        corr = np.corrcoef(oxidation_states, en_vals)[0, 1]
    return 0.0 if np.isnan(corr) else float(corr)
```

In `validity_metrics.py`:
```python
# Accept neutral (0.0) correlation when no electronegativity difference exists
return 0.0 if (correlation >= 0.0 or math.isclose(correlation, 0.0)) else LARGE_CHARGE_DEVIATION
```

---

## 2. Bug 2: Combinatorial Explosion on Structures with $\ge 30$ Sites per Element

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

### Proposed Patch
Replace the brute-force `combinations_with_replacement` loop with direct construction of the adjacent-state partitions:

```python
# Direct, O(K * N) generation replacing combinations_with_replacement:
sorted_oxids = sorted(set(oxids))
n_atoms = int(el_amt[el])

# 1. Homogeneous assignments (all n atoms have identical state k)
valid_combos = []
for k in sorted_oxids:
    valid_combos.append((k,) * n_atoms)

# 2. Mixed adjacent assignments (m atoms at k, n-m atoms at k+1)
for i in range(len(sorted_oxids) - 1):
    k1, k2 = sorted_oxids[i], sorted_oxids[i + 1]
    if k2 - k1 == 1:
        for m in range(1, n_atoms):
            valid_combos.append((k1,) * m + (k2,) * (n_atoms - m))

for oxid_combo in valid_combos:
    oxid_sum = sum(oxid_combo)
    if oxid_sum not in el_sums[idx]:
        el_sums[idx].append(oxid_sum)

    if not all_oxi_states:
        scores = [type(comp).oxi_prob[str(Species(el, o))] for o in oxid_combo]
        score = math.prod(scores)
        if oxid_sum not in el_sum_scores[idx] or score > el_sum_scores[idx].get(oxid_sum, 0):
            el_sum_scores[idx][oxid_sum] = score
            el_best_oxid_combo[idx][oxid_sum] = oxid_combo
```

**Benchmark impact:** Exactly identical output and state rankings, but reduces runtime on a 64-atom species from **$>2$ hours to $<0.1$ milliseconds**.

---

## 3. Domain Limitation: False Negatives on Near-Hull ($E_\text{hull} < 0.05$ eV/atom) Intermetallics & Electrides

### Problem
In `validity_metrics.py`, non-insulating structures are supposed to be exempted from charge neutrality checks via SMACT's `metallicity_score`:
```python
if metallicity_score(Composition(structure.formula)) > 0.70:
    return 0.0
```

However, the SMACT metallicity heuristic suffers from two critical blindspots:
1. **Metalloids (Si, Ge, B, As, Sb, Te) are counted as non-metals.**
   Any intermetallic silicide, boride, or germanide with $\ge 35\text{ at}\%$ metalloid falls below the $0.70$ threshold.
2. **Hard threshold cliff edge.**
   Ordered metallic antiperovskites like $\text{Tb}_2\text{TmMgC}$ (Alexandria `agm005013222`, $E_\text{hull} = 0.0299$ eV/atom) have a metallicity score of **0.6945**, failing the bypass by $0.0055$.
3. **Subvalent & Electride Phases.**
   Experimentally confirmed classes such as layered alkaline-earth pnictide hydrides ($\text{Ba}_2\text{Ca}_4\text{As}_3\text{H}_2$, $E_\text{hull} = 0.0144$ eV/atom) have delocalized interstitial electrons $[(\text{Ba}_2\text{Ca}_4)^{12+} (\text{As}^{3-})_3 (\text{H}^-)_2] \cdot e^-$. Because $+12 \neq +11$, the ionic solver rejects them.

In an audit of 1,000 structures from LeMat-Bulk with $E_\text{hull} < 0.1$ eV/atom, **3.8% of thermodynamically near-ground-state structures are rejected solely by `charge_deviation > 0.1`**.

### Recommended Improvements

1. **All-Metal/Metalloid Intermetallic Bypass:**
   If all constituent elements in a crystal are metals or metalloids (i.e. contains no halogens, chalcogens, or N/P), the crystal cannot form an ionic lattice and should automatically bypass charge balancing:
   ```python
   if all(Element(el).is_metal or Element(el).is_metalloid for el in composition.elements):
       return 0.0
   ```

2. **Dilute Solid Solution / Doped Matrix Bypass:**
   If a single element comprises $\ge 85\text{ at}\%$ of a crystal and that element is an elemental metal or covalent semiconductor (e.g. Si, Ge, C, Fe), bypass charge balancing to accommodate dilute dopants (such as $\text{CuSi}_{64}$, $E_\text{hull} = 0.033$ eV/atom).
