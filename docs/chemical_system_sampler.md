# Sampling (chemical system, space group) inside an element palette

[chemical_system_mode.md](chemical_system_mode.md) conditions the model on *one*
system the caller names. This is the piece that stands in front of it when the
caller does not name one, which is the shape the operating mode actually has: a
user supplies a small set of **required** elements and a large set of **allowed**
ones — 1-3 against 20-40 — and every generated structure needs a system and a
space group of its own.

`wyckoff_transformer.system_prior` reads `p(system, space group)` off the
training split, restricts it to the region `required ⊆ S ⊆ allowed`, and draws a
pair per structure. `wyformer-system-prior` builds, inspects and samples from it.

## Why the palette cannot stay a filter

The mode this replaces samples unconditionally and discards what does not match.
That is biased twice, and both biases are measurable on `lemat_bulk_fmax1`'s
training split (5127874 rows, 661816 distinct systems, 228 space groups).

**The space group is drawn before any element is.** It is this architecture's
start token, so it is fixed at the start of the sequence and cannot react to a
mask applied later on the `elements` logits. Over a 36-element palette the space
groups of the rows that survive the filter carry KL 0.21 nats against the
unconditional distribution once Li is required, and 0.48 for Li-S. Per *system* —
which is what a system-conditioned model is actually given — the row-weighted
mean KL is 0.92 nats over the 55334 systems with at least 20 rows:

| | top space groups |
| --- | --- |
| unconditional | 123: 0.133, 216: 0.094, 12: 0.076, 166: 0.061, 221: 0.052 |
| Li required, 36-element palette | 123: 0.101, 216: 0.070, 12: 0.065, 1: 0.055, 221: 0.050 |
| Li-Mn-Co-O | 1: 0.32, 2: 0.21, 8: 0.13, 12: 0.12 |

Li-Mn-Co-O is a triclinic and monoclinic system; the unconditional sampler starts
a third of its batch in 123 and 216, and no element mask moves a start token.

**The mask is flat where chemistry is not.** Masking the `elements` logits makes
every allowed element reachable, but nothing says which *combinations* occur, so
Li-Kr-O is as reachable as Li-Mn-O except through whatever the unconditioned
model already believes. Forcing the required elements in by argmax — what
`generate_tensors` does when `required_element_set` is non-empty — is a third
distortion on top of that.

**And rejection is expensive where it is not wrong.** The share of training rows
a query admits is 11.4% for a bare 36-element palette, 1.4% with Li required and
0.07% for Li-S: 9x to 1400x oversampling to fill a batch.

## The model

    p(S, G | required, allowed) = p(S | required ⊆ S ⊆ allowed) · p(G | S)

**`p(S)`** is the training row count of each feasible system, renormalised, mixed
with a proposal for systems the data has never seen:

    p(S) = (1 - α)·p_empirical(S) + α·p_novel(S)

`p_novel` draws an arity, then the elements the required set does not already fix,
without replacement, both weighted by how many **distinct systems** each already
appears in rather than by how many rows — a corpus that enumerated ten thousand
quaternary oxides says more about its own campaign than about how often oxygen
belongs in a new system. Candidates that land on an observed system are rejected,
so the two components really are a partition. α defaults to the novelty rate
measured on the held-out split when the prior was built (0.0554 for
`lemat_bulk_fmax1`, against 0.0555 on test), so the sampler is as adventurous as
the data says the world is; raise it deliberately to explore.

That default is a property of the corpus and the split, not of chemistry: the
same measurement on the 200000-row `lemat_bulk_fmax1_pilot` gives 0.31, because a
smaller corpus leaves more of the world unseen. `--max` raises it too — a prior
built from stable rows only has fewer systems in it — and the filter is applied
to the held-out split as well, so the number stays a like-for-like measurement of
"a system with no counted member in training" rather than sliding into "a system
with no *stable* member".

**`p(G | S)`** is the system's own space group counts shrunk toward a back-off:

    p(G|S) = (n(S, G) + κ·p_backoff(G|S)) / (n(S) + κ)

The shrinkage is not optional. The median system has 54 training rows spread over
228 space groups, so the raw table is mostly zeros and scores -150.9 nats/row on
held-out data. The back-off is a product of experts against the global
distribution as the reference measure — the geometric mean of the per-element
space group distributions, times the arity's, divided by the global one.

Both smoothing constants were fitted on the held-out split rather than chosen:

| p(G\|S) on held-out rows whose **system was seen** | nats/row |
| --- | --- |
| κ = 0 (raw counts) | -150.95 |
| κ = 30 | -3.275 |
| κ = 50 | -3.265 |
| **κ = 100 (default)** | **-3.265** |
| κ = 150 | -3.270 |
| κ = ∞ (back-off only) | -3.356 |

| p(G\|S) on held-out rows whose **system was not seen** | nats/row |
| --- | --- |
| global p(G) | -2.686 |
| elements alone (geometric mean) | -2.411 |
| arity alone | -2.233 |
| **elements × arity ÷ global (default)** | **-1.997** |

| the novel-system proposal, on held-out systems training never saw | nats/system |
| --- | --- |
| row-weighted arity, uniform elements | -15.98 |
| row-weighted arity, row-weighted elements | -15.64 |
| distinct-system arity, uniform elements | -14.93 |
| **distinct-system arity, distinct-system elements (default)** | **-14.55** |

`β`, the pseudo-counts holding the per-element and per-arity tables away from
zero, is flat from 1 to 1000 (-1.9965 to -1.9993) and is there to keep the
geometric mean finite, not to fit anything.

## Using it

Build once per dataset, from the same tensor cache the model trains on — that is
what ties the element tokens, the space group vocabulary and the train/val split
to the checkpoint:

    wyformer-system-prior build lemat_bulk_fmax1

57 s, 10.8 MB, written to `cache/lemat_bulk_fmax1/system_prior.npz`. `--max
energy_above_hull=0.1` (repeatable) counts only rows under a bound, which changes
the question from where a palette puts atoms to where it puts them *stably* — one
worth asking when the run is going to condition on a low hull energy anyway.

Look before spending a GPU hour. `show` answers in two seconds:

    wyformer-system-prior show --prior cache/lemat_bulk_fmax1/system_prior.npz \
        --required Li --allowed Li-Na-K-Mg-Ca-Al-Si-P-S-Cl-O-N-F-B-C-Ti-V-Cr-Mn-Fe-...

    observed feasible systems: 5037 (72681 training rows, 1.42% of the corpus --
    the acceptance rate a rejection sampler would have had)
    by arity: {1: 1, 2: 35, 3: 595, 4: 3845, 5: 477, 6: 80, 7: 3, 8: 1}
    top 8 systems by sampling weight:
      Li-Mn-Co-O   p=0.0302 rows=2197  1:0.32 2:0.21 8:0.13 12:0.12
      ...
    KL(palette || unconditional) = 0.209 nats

Then draw a plan:

    wyformer-system-prior sample --prior cache/lemat_bulk_fmax1/system_prior.npz \
        --required Li-S --allowed <palette> -n 1000 --seed 0 -o plan.json

    1000 structures over 329 systems and 851 (system, space group) cells;
    56 rows in systems absent from the training data

The knobs worth knowing: `--novel-fraction` (α above), `--system-temperature`
(1 is the empirical distribution, larger discounts how much attention the corpus
paid to a system, which is a defensible thing to want from a database assembled
by campaigns rather than by sampling nature), `--sg-temperature`, `--sg-kappa`,
and `--min-arity` / `--max-arity` — with a wide palette the observed feasible set
is overwhelmingly quaternary, so a ternary campaign has to say so.

## What the model is handed

`SystemDraws` carries one request per structure and turns them into the three
tensors generation needs, all in row order:

```python
draws = SystemSpaceGroupPrior.load(path).sample(n, required="Li", allowed=palette)
composition_cond = draws.conditioning_block(len(trainer.tokenisers["elements"]), device)
start_tensor     = draws.start_tensor(trainer.tokenisers[trainer.start_name],
                                      trainer.model.start_type, device)
element_mask     = draws.element_mask(len(trainer.tokenisers["elements"]),
                                      stop_token=trainer.tokenisers["elements"].stop_token)
trainer.generate_structures(len(draws), calibrate=False, cond=scalars,
                            composition_cond=composition_cond, start_tensor=start_tensor,
                            allowed_element_mask=element_mask)
```

`conditioning_block` goes through `chemical_system.chemical_system_vector` rather
than reimplementing it, so the sampler cannot drift from what training saw;
`start_tensor` produces exactly what `SpaceGroupEncoder.encode_spacegroups` would
for those space groups.

`allowed_element_mask` is the per-row form of `allowed_element_set`, and it is
what makes the sampling stick. A batch-wide set can only be the *union* of the
drawn systems, which lets every structure back into the whole palette. Measured
on a trained `lemat_bulk_fmax1` checkpoint, 100 rows split evenly over Li-Mn-O,
Li-Fe-P-O, Na-S and Ba-Ti-O:

| | valid | built from the elements their row asked for |
| --- | --- | --- |
| per-row mask | 67 | **67 of 67** |
| one palette-wide set (the union) | 79 | 29 of 79 |

Two thirds of the batch crosses systems without it, and a structure that mixes
Ba with S is not what its conditioning vector asked for.

Nothing forces a required element in. The forcing `required_element_set` applies
is an argmax that plants the element at whatever site comes next rather than
where the model wanted it; with a per-row mask it is not needed to keep the
elements right, only to keep a *required* one from being dropped, and rejection
after the fact does that without editing the sample. A row asked for Li-Fe-P-O
may answer with Li-P, which is inside its mask and outside the request.

## Generating from a plan

`wyformer-generate` takes either the prior or a plan file:

    wyformer-generate out.json.gz --model-path runs/<run> \
        --system-prior cache/lemat_bulk_fmax1/system_prior.npz \
        --required-elements Li --allowed-elements <palette> \
        --initial-n-samples 1100 --sampler-seed 0 \
        --condition energy_above_hull=0 --condition delta_e_polymorph=0 \
        --condition max_force=0

    --- Sampled plan: 1100 structures over 695 systems and 1049 cells;
        59 rows in systems absent from the training data ---
    --- Plan written to out.plan.json ---
    --- Running in per-row chemical system generation mode ---

`--required-elements` and `--allowed-elements` become the sampler's floor and
ceiling instead of a decoding constraint, and the structures that dropped a
required element are removed after generation rather than having one forced in —
the run says how many, on the line that follows the generation timing. How large
that loss is on a system-conditioned checkpoint is not yet measured: the only one
that exists is a few hundred epochs into a 20000-epoch schedule and decodes no
valid structures at all yet.

The plan is written next to the output as `out.plan.json`, which is the run's
provenance: it names every (system, space group) asked for and the query that
produced them.

`--system-plan out.plan.json` regenerates from a plan already drawn — the batch
size then comes from the plan — which is how a campaign is repeated exactly, or
how the full set of sampler knobs (`--system-temperature`, `--sg-kappa`,
`--min-arity`, ...) reaches a generation run: draw the plan with
`wyformer-system-prior sample`, then feed it. A plan carries the element
vocabulary it was drawn against and is refused against a model with a different
one, since a system is stored as element *tokens*.

`--system-prior` and `--system-plan` are refused together with
`--chemical-system`, `--space-group` and `--sg-dist`: a plan already says what
each row is asked for and where it starts.

## What this does not claim

- **It is a prior over inputs, not over structures.** Nothing here estimates
  `p(gene | S, G)`; whether the conditioned model honours the request is a
  question about the checkpoint, and the first read on it is to generate from a
  plan and count the structures whose element set is exactly what their row asked
  for.
- **A drawn system is a request.** The mask permits any subset of it, so a
  Li-Mn-Co-O row may come back as Li-Mn-O. `SystemDraws` records what was asked;
  what was produced has to be read off the output.
- **Which row produced which structure is not recorded.** Invalid genes are
  dropped inside `generate_structures`, so the output list is shorter than the
  plan and no longer aligned to it. The plan is therefore campaign-level
  provenance, not per-structure: "how often does the model use the whole system
  it was given" needs the mask of dropped rows, which is a change to
  `generate_structures` rather than to this module.
- **The counts are the corpus's, with all its campaigns in them.** LeMat-Bulk is
  not a sample of nature: 83% of its distinct systems are quaternary, and
  Li-Mn-Co-O has 2197 rows because someone enumerated it.
  `--system-temperature` and the distinct-system weighting in the novel proposal
  are two partial answers to that; neither is measured against an external
  ground truth, because there isn't one.
- **The novelty rate is a held-out estimate, not a target.** 5.5% is how often
  the val split lands in a system train never saw, under the same random split.
  It is not a claim about how often a *useful* new system appears.
