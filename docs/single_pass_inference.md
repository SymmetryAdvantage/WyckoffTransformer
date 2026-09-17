 ▐▛███▛█   Claude Code v2.1.272
▝▜██████▀  Opus 5 · Claude Pro
  ▝▝ ▝▝    ~/WyckoffTransformer/.claude/worktrees/breezy-wobbling-moth


❯ CascadeTransformer is notoriously inefficient during training - it requires a 
  separate pass per known_seq_length/known_cascade_lenght - unlike the          
  non-cascade Transformer next token likelihoods can be computed during         
  training via one pass. Is it possible to retain the cascade mathematics, but  
  also gain the computational efficiency?                                       

● I'll start by reading how the cascade model and its training loop are
  implemented.

● Bash(git log --oneline -1 && ls && ls wyckoff_transformer && grep -rn "class
      .*Cascade" --include=*.py . | grep -v .venv | head -20)
  ⎿  Error: Exit code 2
     600a2ab Merge remote-tracking branch 'origin/training-loss-fixes'
     AGENTS.md
     CITATION.cff
     CLAUDE.md
     conftest.py
     data
     docs
     generated
     ICML_eval
     … +10 lines (ctrl+o to expand)

  Searched for 1 pattern, read 3 files, listed 2 directories (ctrl+o to expand)

● Bash(ls yamls/models yamls/models/lemat_bulk_fmax1_stress 2>/dev/null; grep
      -rl "multiclass_next_token_with_order_permutation: true" yamls/models |
      xargs ls -t | hea…)
  ⎿  yamls/models:
     alex_mp_20
     band_gap_adamw_wsd.yaml
     … +40 lines (ctrl+o to expand)

  Read 2 files, listed 1 directory (ctrl+o to expand)

● Bash(.venv/bin/python -m wyckoff_transformer.paths 2>&1 | head; ls
      /home/kna/.local/share/wyformer/cache 2>/dev/null | head -30)
  ⎿  /bin/bash: line 1: .venv/bin/python: No such file or directory
     alexandria_trajectories
     alex_mp_20
     … +19 lines (ctrl+o to expand)

  Listed 1 directory (ctrl+o to expand)

● I can lazily open the safetensors file to read just the length tensor. Since
  the worktree has no venv and the script doesn't depend on the project, I'll
  run it using the main checkout's interpreter.

● Bash(cat > /tmp/claude-1000/-home-kna-WyckoffTransformer/f4a009df-3fa2-4e82-88
      5c-7e9603244cd9/scratchpad/keys.py <<'EOF'
      from safetensors import safe_open…)
  ⎿  tensor_0 [5127342, 62] I16
     tensor_1 [5127342, 62] I16
     tensor_10 [5127342] I64
     … +45 lines (ctrl+o to expand)

● Bash(cat > /tmp/claude-1000/-home-kna-WyckoffTransformer/f4a009df-3fa2-4e82-88
      5c-7e9603244cd9/scratchpad/meta.py <<'EOF'
      from safetensors import safe_open…)
  ⎿  wyckoff_transformer_tensor_cache_structure {"type":"dict","items":{"tr
     ain":{"type":"dict","items":{"site_symmetries":{"type":"tensor","key":
     "tensor_0"},"elements":{"type":"tensor","key":"tensor_1"},"sites_enume
     … +20 lines (ctrl+o to expand)

● Bash(cat > /tmp/claude-1000/-home-kna-WyckoffTransformer/f4a009df-3fa2-4e82-88
      5c-7e9603244cd9/scratchpad/lengths.py <<'EOF'
      import numpy as np, torch…)
  ⎿  torch 2.14.0+cu133 flex_attention: yes
     N = 5127342 max = 61
     quantiles 50/75/90/99/99.9: [ 4.  5.  6. 14. 38.]
     … +11 lines (ctrl+o to expand)

❯ You can create .venv in the worktree, it's fast                               

● Yes, mostly. The cascade itself (predicting a site field by field, the soft
  target for the first field, and the training objective) can all be kept
  exactly in a one-pass model. What can't be kept is the combination the current
  encoder has: each prediction being an exact function of the unordered set of
  known sites, with those sites also attending to each other. One of those two
  has to give.

  Why the cost exists

  For each (n, c), trainer.py:1426 runs the full bidirectional encoder over
  [START, t_1..t_n, q]. Here q is site n+1 with fields c and later set to MASK.
  Every known site attends to every other site and to q, so the state of t_1
  changes with n and with c, and nothing can be reused between passes.

  A causal language model avoids this because token i's state depends only on
  tokens up to i, so it is valid for every later prediction. The same logic
  applies here: in one pass, a known site gets one state per layer, reused for
  every n at which it is known. That state can't depend on sites that are still
  unknown at some of those n. You get at most two of three properties:

  ┌─────────────────────┬────────────────────┬──────────────┬──────────────┐
  │                     │       exact        │ sites attend │ passes per   │
  │                     │   set-invariant    │   to each    │  structure   │
  │                     │     prediction     │    other     │              │
  ├─────────────────────┼────────────────────┼──────────────┼──────────────┤
  │ current             │ yes                │ yes          │ (L+1)·C      │
  ├─────────────────────┼────────────────────┼──────────────┼──────────────┤
  │ A: shared context,  │ yes                │ yes          │ L+1          │
  │ many queries        │                    │              │              │
  ├─────────────────────┼────────────────────┼──────────────┼──────────────┤
  │ B: two-stream,      │ only on average    │ yes          │ 1            │
  │ causal over sites   │ over orders        │              │              │
  ├─────────────────────┼────────────────────┼──────────────┼──────────────┤
  │ C: sites encoded    │ yes                │ no           │ 1            │
  │ independently       │                    │              │              │
  └─────────────────────┴────────────────────┴──────────────┴──────────────┘

  C throws away the site–site pair terms that relational.py exists to add, so
  I'd skip it.

  A: exact set conditioning, C× fewer passes

  - Layout: fix n and keep [START, t_1..t_n, M] bidirectional as now, where M is
    the all-MASK token. Append a query token for every remaining site j and
    every c ≥ 1 (site j with fields c and later masked). Queries attend to the
    context and to themselves; nothing attends to a query.
  - What stays the same: c = 0 is the current computation unchanged, because M
    stays in the context. For c ≥ 1 the only change is that known sites no
    longer see the partly filled site.
  - Targets: c ≥ 1 is trained on every remaining site, each weighted 1/(L−n),
    instead of only the one the permutation put next. This is what the
    multiclass target already does for c = 0. The expected loss is exactly the
    current one, with lower variance.

  B: one pass (XLNet-style two-stream attention)

  - Layout: permute once and build [START, t_1..t_L, q_{0,0} .. q_{L,C−1}]. Each
    q_{n,c} is exactly the last input the current model gets for (n, c).
  - Mask: t_i sees START and t_1..t_i. q_{n,c} sees START, t_1..t_n and itself;
    head c reads it. No query sees a later site or another query, so nothing
    leaks.
  - Targets: the c = 0 soft target for every n at once is a reverse cumulative
    sum of one-hots. For c ≥ 1 the target is t_{n+1}[c].
  - Kept exactly:
    - The objective. Today a step draws n in proportion to how many structures
      are that long, and c uniformly (trainer.py:1481-1491). Summing B's loss
      over all targets and dividing by a fixed target count estimates the same
      quantity without bias, so lr and clip_grad_norm keep their scale.
    - The parameters. The embeddings, heads, AdaLN conditioning and relational
      pair tables all carry over. The presence features become a running max of
      one-hots over sites.
  - What changes: the prediction now depends on the order of the known sites. It
    is invariant only on average, because training draws random orders; a
    causal mask leaks order even with no positional encoding. Validation NLL
    becomes an estimate over sampled orders, which evaluation_samples already
    averages.
  - Side benefits: evaluation, calibrate and gene_likelihood drop from (L+1)·C
    passes to one per structure, and generation can cache the known sites (a KV
    cache).

  Numbers on lemat_bulk_fmax1_stress

  Computed today at commit 600a2ab from the train split's pure_sequence_length
  (5,127,342 structures); no runs involved. Median L is 4, the 99th percentile
  14, the maximum 61. With C = 3 that is 16.2 targets per structure.

  ┌────────────────────────────┬─────────┬───────┬───────┐
  │ per structure, all targets │ current │   A   │   B   │
  ├────────────────────────────┼─────────┼───────┼───────┤
  │ forward passes             │ 16.2    │ 5.4   │ 1     │
  ├────────────────────────────┼─────────┼───────┼───────┤
  │ tokens × layers            │ 1×      │ 1/1.4 │ 1/3.6 │
  └────────────────────────────┴─────────┴───────┴───────┘

  - Per training step at the same batch size: the current scheme spends about
    4.8 tokens per target, B about 1.3, and B covers 16× more targets. Those
    targets share a structure and an order, so they are correlated and the real
    gain is less than 16×.
  - Where the gain comes from: because LeMat sequences are short, the compute
    saving is only 3.6×; it rises with length (3× at L=4, 9× at L=20, 25× at
    L=62). The bigger win is fewer passes and more targets per step, since your
    config notes this model is limited by kernel launches rather than
    arithmetic.
  - Memory: at batch 25k, a dense mask padded to the 62-site cap would be about
    25 GB. B needs batches grouped by length (length_sorted_indices already
    sorts by length) or FlexAttention with a block mask. FlexAttention is in the
    torch 2.14 installed here and can also apply the relational bias.

  Recommendation

  Build B behind a config flag and A as its ablation. A gets the extra targets
  per step while keeping exact invariance. If A matches B at equal wall-clock
  time, the order dependence isn't worth it. Compare them on generation metrics
  such as validity and S.U.N., not only validation NLL, since B's NLL depends on
  the sampled orders.

  On the venv: this analysis only read one tensor from the cache, without
  importing any project code, so I ran it with the main checkout's Python. I'll
  run env_init.sh in this worktree when we start prototyping. Want me to start
  with B, plus a test that each query's output matches a separate masked pass?

✻ Worked for 7m 51s · done Tuesday 7:18 PM

❯ /usage 
  ⎿  Settings dialog dismissed

❯ /usage 
  ⎿  Settings dialog dismissed

❯ /usage 
  ⎿  Settings dialog dismissed

❯ /usage 
  ⎿  Settings dialog dismissed

❯ /usage 
  ⎿  Settings dialog dismissed