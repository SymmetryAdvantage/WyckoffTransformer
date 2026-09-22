# The Wyckoff augmentation: a group, and a representation that is not equivariant

> **STATUS (2026-09-22, commit `9145354`): defect established, reproduced, and
> fixed in code.** The augmentation now carries the oriented site-symmetry
> symbol alongside the enumeration index. **No cache has been migrated yet**,
> and until one is, anything reading an old cache raises rather than producing
> a wrong fingerprint. See *What has to be rebuilt* below.

## The question, and why it was asked

`evaluation/novelty.py::record_to_augmented_fingerprint` decides when two
Wyckoff genes are the same gene. Every uniqueness and novelty verdict in this
repository rests on it, and through them MetaSUN and SUN.

It is invariant to the equivalent enumerations of the same positions, by
comparing the **set** of relabelled variants. Building the tensor-space key of
[the rules of engagement](rules_of_engagement.md#the-tensor-space-screen) raised
the obvious optimisation: if the relabellings form a group, orbits partition,
and the cheaper *minimum over the orbit* is a valid canonical form. That is what
was checked.

## The relabellings do form a group

For all 230 space groups, with no exceptions: the identity is present, every map
is a bijection on the letters, the set is closed under composition in both
orders, and every inverse is present. `alternatives_letters[0]` is
`('a','b','c',...)` in all 230, so the assertion at
`preprocess_wychoffs.py:454` is safe.

It is also a theorem rather than a coincidence. `Group(n).get_alternatives()` is
a lookup of PyXtal's static `wyckoff_sets.json`, whose rows are a transversal of
the space group's normaliser quotient. Conjugation by a normaliser element
permutes Wyckoff positions, and the map from normaliser cosets to letter
permutations is a homomorphism; the set in the table is its **image**, and the
image of a homomorphism is a subgroup. The map is frequently non-injective, so
the `frozenset` deduplication at `preprocess_wychoffs.py:461` is not merely
harmless -- it collapses exactly the kernel and computes the image.

**PyXtal documents none of this.** `get_alternatives` is three lines returning
`wyckoff_sets.json`, with no mention of Bilbao, WYCKSETS, International Tables or
normalisers, and the PyXtal paper cites Bilbao only in passing about
group-subgroup relations. The file's fields -- `No.`, `Coset Representative`,
`Geometrical Interpretation`, `Transformed WP` -- match the documented output of
Bilbao's WYCKSETS field for field, and the content matches, but the
identification is an inference and not an attested fact. There is no upstream
guarantee of the semantics, of the direction of the permutation, or of which
normaliser was used, which is a reason to keep the closure check below as a
regression guard rather than to rely on provenance.

No published source states "the induced letter permutations form a group". What
the literature states is the *action*: the IUCr dictionary defines a Wyckoff set
as the points whose site-symmetry groups are conjugate under the normaliser --
orbits of a group action by definition -- and Koch & Fischer count equivalent
descriptions by the index of a quotient group. The group property follows
immediately; cite the computation, not a theorem nobody wrote down.

Group orders over the 230: 66 trivial, 94 of order 2, 34 of order 4, and a tail
to order 48 (space groups 16, 22, 47, 48).

## But the group property was the wrong question

The fingerprint is not built from letters. `data.py::pyxtal_notation_to_sites`
(lines 63-73) builds `sites_enumeration_augmented` by taking the enumeration of
the **relabelled** letter, while `site_symmetries` keeps the symbol of the
**original** letter. The fingerprint then zips the two.

The enumeration index is defined *within* a site-symmetry class. Pairing an old
symbol with a new index is therefore a category error whenever a relabelling
changes the symbol -- and it does, in **26 space groups**, by permuting the
crystal axes (`2..` -> `.2.` -> `..2`, `m..` -> `.m.`):

    16, 17, 20, 21, 22, 23, 24, 25, 35, 42, 44, 47, 48, 49, 50, 59,
    65, 66, 67, 68, 69, 70, 71, 72, 73, 74

1,920 `(space group, letter, map)` triples change the symbol. The resulting
`(old symbol, new index)` pair is always a *legal* pair in that space group,
which is why the collision is silent rather than raising.

**This is known crystallography, and ITA gives the textbook case.** Volume A1,
Müller's *Remarks on Wyckoff positions*, uses `I222` -- space group **23**, on the
list above -- to make exactly this point: `4e, 4f, 4g, 4h, 4i, 4j` form one
Wyckoff set, but the three pairs lie on differently oriented axes and are not
equivalent once `a != b != c`. Those three pairs are precisely the ones whose
oriented symbols are `2..`, `.2.` and `..2`. PyXtal's table for space group 23
contains all six axis permutations, and its `z,x,y` row relabels
`a b c d e f g h i j k` to `a c d b i j e f g h k` -- mixing `{e,f}`, `{g,h}` and
`{i,j}` just as ITA describes.

The theory says why. A Wyckoff position is a class of points whose site-symmetry
groups are *conjugate*, and ITA §1.4.4 notes that conjugate site-symmetry groups
"contain operations of the same kind but at least some of the symmetry elements
are located or oriented differently in space". The oriented symbol encodes that
orientation -- its constituents are ordered by the primary, secondary and
tertiary symmetry directions of the lattice. So the normaliser preserves the
site-symmetry group **up to conjugacy**, i.e. its point-group *type*, and not the
oriented symbol. What it does preserve is multiplicity, the number of free
parameters, and that type.

The 26 groups are all orthorhombic, which is where this and only this can happen:
only there are the three symmetry directions interchangeable by a
lattice-preserving affine map not already in the group. In tetragonal, hexagonal
and cubic groups the map that would exchange secondary and tertiary directions
does not preserve the lattice type; in monoclinic and triclinic there are no such
slots to permute.

## The counterexample, checked by hand

Space group 68, `Ccce`. Its letters, as `letter: (symbol, index within symbol)`:

    a:('222',0) b:('222',1) c:('-1',0) d:('-1',1) e:('2..',0)
    f:('.2.',0) g:('..2',0) h:('..2',1) i:('1',0)

Its four relabellings, as images of `a..i`:

    abcdefghi (identity)   abdcfeghi (c<->d with e<->f)
    bacdefghi (a<->b)      badcfeghi (both)

```python
A = {"group": 68, "species": ["Li", "Be"], "numIons": [1, 1], "sites": [["1e"], ["1c"]]}
B = {"group": 68, "species": ["Li", "Be"], "numIons": [1, 1], "sites": [["1e"], ["1d"]]}
```

`record_to_augmented_fingerprint(A) == record_to_augmented_fingerprint(B)` is
**True**, and **no relabelling takes A to B** -- every map that swaps `c` and `d`
also moves `e` to `f`. The collision is caused by `f` having index 0 within
`.2.` while `e` has index 0 within `2..`: the token `(Li, '2..', 0)` silently
means both "Li on e" and "Li on f".

The mirror failure, a missed merge, is space group 16: `1t` (symbol `..2`) and
`1o` (symbol `.2.`) have identical variant sets but different recorded symbols,
so they get different fingerprints although the axis-permuting setting relates
them.

## How much it matters -- partly measured

| | |
|---|---|
| broadside's 1000 novelty decisions in the 26 exposed groups | **108 (10.8%)** -- 80 called novel, 28 called known |
| the 10,000-gene pool in the 26 exposed groups | 1,000 (10.0%) |
| observable false or missed merges *within* that pool | **0** |

**The zero is weak evidence, and should not be read as reassurance.** The pool
yields 9,987 distinct genes from 10,000 draws, so there is almost nothing there
to merge and the test has very little power. The failure mode that matters is
novelty against the *dense* 4.8M-gene reference, where a missed merge means a
gene is called novel although the archive holds a relabelled equivalent. **That
has not been measured.** On an adversarial sample restricted to the 26 groups,
missed merges were common and false merges ran at about 11 in 20,000.

Directions, so the bias is not mistaken for noise:

- a **missed merge** calls a gene novel when the archive holds an equivalent, so
  it **overstates** novelty, gene novelty rate, and MetaSUN;
- a **false merge** calls a gene known when it is not, so it **understates** them.

The adversarial sample says missed merges dominate, so the expected net effect
is an overstatement -- concentrated in the 26 groups, which include `Pmmm` (47),
`Cmmm` (65), `Fmmm` (69), `Immm` (71) and `Pmm2` (25) and are not rare.

**Comparisons between arms are unaffected.** Every arm of the rules-of-engagement
comparison uses the same screen against the same reference, so whatever bias
exists is common to all of them. What inherits the defect is the *absolute*
novelty and MetaSUN of any run -- this one and every protocol artifact already
stored.

## The fix, and what it would cost

Carry the augmented **letter** rather than the `(old symbol, new index)` pair, or
relabel the site symmetry alongside the letter, in
`data.py::pyxtal_notation_to_sites` lines 63-73. Either makes the variant set a
true orbit, removes both failure modes, and makes min-over-orbit exactly valid on
all 230 space groups -- which also buys the cheaper vectorisable canonical form.

**Carrying the letter is what everyone else does.** `pymatgen`'s protostructure
labelling (`pymatgen/analysis/prototypes`, upstreamed from aviary; Goodall et al.
2022, Parackal et al. 2024) canonicalises by exactly the minimum over the
normaliser relabelling orbit, and stores only letters, multiplicities and
free-parameter counts -- no site-symmetry symbol travels through the relabelling,
so the failure mode cannot arise. AFLOW's prototype label does the same by
re-deriving the label from a reoriented cell. spglib takes the other route: one
standardised setting, with `wyckoffs` and `site_symmetry_symbols` derived from
the same cell, consistent because nothing is relabelled afterwards. The hazard is
specific to relabelling letters while keeping symbols computed before the
relabelling, and no source or codebase documents it as a pitfall -- which is
presumably why it survived here.

The same `(symbol, index)` pairing flows into `record_to_anonymous_fingerprint`,
`record_to_relaxed_AFLOW_fingerprint` and `record_to_strict_AFLOW_fingerprint`,
so all four inherit it and all four would change.

The cost is a migration, which is why nothing has been changed:

- every fingerprint in the 26 groups changes, so
  `cache/lemat_bulk_fmax1_stress/gene_fingerprints.pkl.gz` (4,826,004 entries)
  and `gene_keys.npz` must be rebuilt;
- gene novelty, `novel_structure`, MetaSUN, SUN and the novelty crossings of
  every stored protocol artifact would move, and would not be comparable with
  numbers produced before the change -- the same situation as
  [the `lemat_bulk_ehull` re-score](archive/lemat_bulk_ehull_rescore.md), and it
  would want the same treatment: re-score, and record what moved.

## A separate question the audit raised: affine or Euclidean?

Bilbao's WYCKSETS uses the **affine** normaliser (and the Euclidean one for the
triclinic and monoclinic groups 1-15, whose affine normalisers have continuous
linear parts). Its EQUIVSTRU, which enumerates the physically equivalent
descriptions of a *given structure*, uses the **Euclidean** normaliser instead.
The two differ exactly on the axis permutations, which is the same 26 groups.

So the affine table is deliberately coarser than structural equivalence: it calls
`I222`'s `4e` and `4g` one Wyckoff set, and ITA's own remark is that they are not
equivalent once `a != b != c`.

That is not a second defect, and it is probably the right choice here. A WyFormer
gene is an abstract assignment of elements to Wyckoff letters with no metric --
the cell is drawn later by PyXtal -- so the affine group is the natural one, and
it is what `pymatgen` and AFLOW use for protostructure labels. It is worth
recording because the coarser grouping errs toward *merging*, which at the gene
level sends more candidates to `StructureMatcher` rather than fewer, and the
matcher is the actual verdict. The direction is conservative. It would be the
wrong choice if a gene fingerprint were ever treated as a structural identity on
its own.

## What has to be rebuilt -- and what does not

**The whole dataset does not have to be re-derived from structures.** The
augmentation is a pure function of `(spacegroup_number, wyckoff_letters)`, and
both are already columns in every cache. So no CIF is parsed, no PyXtal symmetry
detection is re-run, and no structure is touched: two columns are recomputed and
rewritten. `scripts/migrate_augmented_site_symmetries.py` does exactly that, in
place, keeping the old file beside it.

| artifact | rebuild? | what it costs |
|---|---|---|
| structures, CIFs, labels, splits | **no** | untouched |
| `cache/<dataset>/` augmentation columns | **yes**, recomputed in place | one pass over the rows; ~25 GB resident for LeMat-Bulk |
| `gene_fingerprints.pkl.gz` | **yes** | ~11 min; the migration deletes it so it cannot be read stale |
| `gene_keys.npz` | **yes** | ~7 min; likewise deleted |
| `cache/<dataset>/tensors/*.safetensors` | **only if you will train** | re-tokenise from the migrated cache; nothing upstream of it re-runs |
| existing checkpoints | **no** -- and they cannot be | they were trained on the old augmentation; that is a property of the model, not of a cache |
| stored protocol artifacts | **re-score** to be comparable | `--from-artifact --stages screen,score`, as in [the `lemat_bulk_ehull` re-score](archive/lemat_bulk_ehull_rescore.md) |

Two consequences worth stating plainly.

**Every trained checkpoint saw the corrupted augmentation.** In the 26 space
groups the model was shown pairs naming a third Wyckoff position as an
"equivalent description" of the one it was given. Retraining is the only way to
remove that, and whether it is worth it is a question about how much of the
training set falls in those groups -- 10.0% of a 10,000-gene sample from
`unconditional_5x_ehull01-20260915-151250`, which is a sample of the model's
output rather than of the corpus, so the corpus share has still to be measured.

**Numbers produced before and after the fix are not comparable.** Gene novelty,
`novel_structure`, MetaSUN, SUN and the novelty crossings all move. This is the
same situation as the `lemat_bulk_ehull` migration and wants the same handling:
re-score, and record what moved.

## What the tests now cover

The defect survived because every test compared the fingerprint to itself.
`test_the_key_induces_the_same_partition_as_the_fingerprint` pins the tensor key
to `record_to_augmented_fingerprint` -- the right contract for the key, since it
is a faster spelling of the existing verdict rather than a correction of it, but
it means a defect the two share is invisible. The tests added with the fix, in
`tests/test_gene_hash.py`, compare against something external instead:

- **the equivariance tests** -- over 3000 random genes, the partition induced by
  the fingerprint, and separately by the tensor key, must equal the partition
  induced by the **letter orbit**: the multiset of `(element, relabelled letter)`
  over all relabellings. Letters are what the relabellings act on, so this is
  independent of how a record encodes a position. These are the tests that would
  have caught the defect, and they fail against the old code;
- the two hand-checkable cases, both directions: space group 68 must *separate*
  two genes no relabelling relates, and space group 16 must *merge* two that an
  axis permutation does;
- the relabellings are a group, in all 230 space groups -- the premise the
  letter-orbit truth rests on, guarded because PyXtal's table is undocumented
  and could change under us;
- the relabellings do **not** preserve the oriented symbol, asserting exactly
  the 26-group set, so the reason for the pairing cannot quietly stop applying;
- the augmented columns stay aligned, one symmetry tuple per enumeration tuple;
- the variant order does not move with `PYTHONHASHSEED`. The relabellings are
  held in a `frozenset`, so before they were sorted the *set* was stable but the
  order was not -- and the order is what a paired representation and a tokenised
  cache are indexed by;
- an unmigrated record is refused by both the fingerprint and the key.

`wyckoff_processor.validate_augmented_token_fields` refuses a tokeniser config
that augments `sites_enumeration` without `site_symmetries`, or either without
`wyckoff_letters` where that field is present. The combination that caused this
can no longer be configured.

## How it was checked

Three independent routes, 2026-09-22, commit `9145354`:

1. **Mathematics** -- identified the table as the normaliser-quotient image and
   showed the group property follows from it being the image of a homomorphism.
   Concluded, incorrectly, that the existing fingerprint was therefore sound:
   the argument is about the action on *letters*, and the code does not carry
   letters.
2. **Computation** -- exhaustive closure, identity and inverse checks over all
   230 groups; the site-symbol-changing count; and partition tests over uniform
   and adversarial gene samples, which found the defect.
4. **An independent re-check** of the load-bearing claims before they were
   written down here: the 26-group list, the 1,920 triples, and the space group
   68 counterexample, all reproduced directly.

3. **Literature** -- confirmed the ITA `I222` example, the oriented-symbol
   theory, the affine/Euclidean split, and that the two widely used
   canonicalisers carry letters rather than symbols. IUCr and Bilbao pages are
   behind Cloudflare, so several ITA passages are as reported by search
   extraction rather than read directly; the `I222` relabelling was verified
   against PyXtal's own table instead.

## See also

- [Rules of engagement](rules_of_engagement.md#the-tensor-space-screen) -- the
  tensor key this audit came out of
- [The de novo ranking protocol](de_novo_ranking_protocol.md#novelty-and-uniqueness-are-two-stage) -- what the fingerprint is used for
