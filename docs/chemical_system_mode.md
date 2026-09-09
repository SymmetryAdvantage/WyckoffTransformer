# Chemical-system mode: generating within a set of allowed elements

CSP mode fixes the formula and asks which structure it adopts. This is the
relaxed variant of it: the caller names a **set of allowed elements** — Ba-Ti-O
— and leaves the stoichiometry, the cell size and the space group to the model.

It is the question a chemist asks more often than the CSP one, because a
chemical system is what a synthesis campaign is scoped by, while a formula is
already an answer.

## What is supplied, and what is not

| | CSP ([csp_mode.md](csp_mode.md)) | chemical system (here) |
| --- | --- | --- |
| conditioning input | element fractions + `log1p` cell size | indicator over the element set |
| stoichiometry | fixed by the caller | chosen by the model |
| cell size | an input, or decoded per `z` | chosen by the model |
| decoding | `ConstrainedDecoder`, exact-composition masking | a plain mask on the `elements` logits |
| acceptance rate | one, by construction | one, by construction |

The second row is the whole difference, and it propagates to the fourth. A
composition is reachable only through a particular sum of Wyckoff
multiplicities, so most prefixes strand it and `csp.SpaceGroupCombinatorics` has
to answer a reachability question before every placement. A *set* strands
nothing: any allowed element can be placed at any multiplicity, any number of
times. So there is no combinatorics here at all — the constraint is one mask on
the `elements` logits, which `WyckoffGenerator.generate_tensors` already applied
through `allowed_element_set`, and every draw is accepted.

The space group is also a conditioning input, in both modes and for a reason
that predates both: it is this architecture's **start token**, so it enters as
the sequence's first element rather than through AdaLN. `--space-group` pins it;
omitting it samples from the training distribution, which is the right default
when the question is "what does this system crystallise as".

## The representation

`wyckoff_transformer.chemical_system` turns the set into a fixed-width indicator
over the element vocabulary — one where the element is present, zero elsewhere —
which joins whatever scalars the model is already conditioned on and reaches
every encoder layer through the same AdaLN path `energy_above_hull` takes. The
same vector is built from `--chemical-system` at sampling time, so training and
sampling see one representation.

Widths follow the element tokeniser, service tokens included, exactly as
`composition` does it: the vector stays indexable by token id with no offset
table to get wrong, at the cost of three columns that are always zero.

Two properties are worth stating because they are choices:

- **The counts do not reach the vector.** `index_fill_`, not `index_add_`: an
  element named twice is still one element. BaTiO3 and Ba2TiO4 are one chemical
  system and encode identically, which is the point of the mode.
- **The L1 norm is the arity.** 3 for Ba-Ti-O. The model is told how many
  elements it is working with, without being told the proportions. A binary and
  a quaternary system have very different gene-length distributions, and an
  encoding that hid the difference would make the model infer it from the space
  group alone.

Nothing is transformed or rescaled on the way in. `condition_transform` and
`condition_scale` name the scalar features only, and an indicator is already on
the scale AdaLN wants: its per-column standard deviation over the training set
is at most 0.5, the same order as `energy_above_hull` at 0.317 post-`log1p` and
below the 0.525 that made `max_force` at the wrong scale diverge (see
[dirty_data_conditioning.md](dirty_data_conditioning.md)).

## What the label means

Every training structure carries the set of elements it actually contains, so
the model learns

    p(gene | exactly these elements are present)

It does **not** learn `p(gene | a subset of these elements)`: nothing in the
training data pairs a gene with a system larger than its own.

That is the honest reading of the label and it has one consequence that can
surprise. Asking for Ba-Ti-O asks for a compound containing all three. If what
you want is "any oxide of barium and titanium, including BaO and TiO2", the
conditioning vector is the wrong instrument. Two ways round it: run the model
once per subsystem and pool the results, which keeps the conditioning honest; or
mask without conditioning — `--allowed-elements Ba-Ti-O` and no
`--chemical-system`, which leaves the conditioning to be sampled from the
training distribution and so needs `--use-cached-tensors`.

The two halves of `--chemical-system` behave differently on this point, and
deliberately:

- the **mask** permits any subset of the requested set, so a model that decides
  a Ba-Ti-O request is best answered by TiO2 is not prevented from saying so;
- the **conditioning** asks for all of them, so it is never encouraged to.

An obvious extension, not implemented: augment training by occasionally
presenting a superset of the structure's own elements, which would teach the
model that "allowed" and "present" are different claims. It would need the
conditioning vector resampled per epoch rather than precomputed once, which is a
change to how the block is stored, not to what it means.

## Training a backbone

`yamls/models/lemat_bulk_fmax1/chemsys_e_all_adamw_wsd.yaml` is
`yamls/models/lemat_bulk_ehull/e_all_adamw_wsd.yaml` — the same architecture,
optimiser and WSD schedule, the same three energy conditioning channels — plus
`chemical_system_conditioning: true`, and minus its `condition_dim`. The width
stops being a free choice once it follows the element vocabulary, so
`from_config` derives it (95 for `lemat_bulk_fmax1`: three energy columns, then
92 element columns) and refuses a config that hardcodes a different number.

    bash scripts/train_in_pb.sh \
        yamls/models/lemat_bulk_fmax1/chemsys_e_all_adamw_wsd.yaml lemat_bulk_fmax1

No re-cache is needed. `chemical_system_conditioning` reads the
`counters: {composition: elements}` field the tokeniser already emits and keeps
only its keys, so one tensor cache serves this mode, `composition_conditioning`
and the unconditioned runs alike.

`composition_conditioning` and `chemical_system_conditioning` are alternatives,
not layers, and are refused together: a composition already names the elements,
so conditioning on both would give the element vocabulary two blocks of columns
saying overlapping things.

## Generating

    wyformer-generate out.json.gz --model-path runs/<run> \
        --chemical-system Ba-Ti-O \
        --space-group 221 \
        --condition energy_above_hull=0 \
        --condition delta_e_polymorph=0 \
        --condition max_force=0

`--chemical-system` does two things at once — it builds the conditioning vector
*and* narrows `--allowed-elements` to the same set — so the set is a guarantee
and not merely a preference. Passing `--allowed-elements` explicitly overrides
the second half, which is how you ask for the asymmetric case above.

`--space-group` takes one number or a comma-separated list, and spreads the
batch evenly over them. It is refused together with `--sg-dist`, which sets the
same tokens from a cached dataset's distribution.

A model conditioned on several scalars needs all of them named: `--condition`
once per feature. Omitting them samples whole conditioning rows from the
training data, which needs `--use-cached-tensors` and pairs the energies with
the systems as they actually occur.

## CSP with a chemical-system backbone

`wyformer-csp` accepts one. The formula still enters as an exact decoding
constraint through `ConstrainedDecoder`; what the backbone is *told* is only
which elements are involved. It is the weaker conditioning of the two, and its
one advantage is that it has no cell-size channel — so `z` is decoded in a
single pass over all feasible values rather than one pass per `z`, which is the
3.6x budget saving described in
[csp_mode.md](csp_mode.md#the-model-chooses-z-not-the-caller). Whether the
weaker conditioning costs more than the wider decoding buys is unmeasured.

## What has not been measured

- **Whether the conditioning helps.** A 3-epoch pilot on `lemat_bulk_fmax1_pilot`
  trains, checkpoints, reloads at the derived width (88 there, for the pilot's
  smaller vocabulary) and generates under both the element mask and a fixed
  space group, with `grad_norm` at 1.73 against 0.52 for the unconditioned
  parent on the same data — a wider input, well inside the clip at 10. Three
  epochs says the plumbing works and nothing else.
- **How often a trained model uses the whole system it was given.** The obvious
  first read on a trained checkpoint, and the number that says whether the
  conditioning is doing anything: generate for a set of known systems and count
  the structures whose element set is exactly the request.
- **Gene novelty scoring is refused for these models.** The likelihood would be
  `p(gene | its own chemical system)`, scored under a different conditioning for
  every gene, which is not comparable across genes.
  `gene_likelihood.gene_log_likelihoods` raises rather than producing it
  silently.
