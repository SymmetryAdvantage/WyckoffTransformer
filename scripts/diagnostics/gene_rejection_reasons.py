"""Classify why WyFormer-generated Wyckoff genes get rejected, by site index."""
import sys, json, logging
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
import torch
from omegaconf import OmegaConf

from wyckoff_transformer.trainer import WyckoffTrainer, load_model_weights
from wyckoff_transformer.generator import WyckoffGenerator
from wyckoff_transformer.tokenization import get_wp_index

logging.basicConfig(level=logging.WARNING)

RUN = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/upi73i4k")
N = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
COND = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0

config = OmegaConf.load(RUN / "config.yaml")
trainer = WyckoffTrainer.from_config(config, device=torch.device("cpu"),
                                     use_cached_tensors=False, run_path=RUN, load_datasets=False)
load_model_weights(trainer.model, RUN / "best_model_params.pt", torch.device("cpu"))

start = trainer._sample_start_tokens_from_distribution(N)
# decode start tensor rows back to space group numbers
_sgtok = trainer.tokenisers["spacegroup_number"]
if start.dim() == 1:
    sg_of = [_sgtok.to_token[i.item()] for i in start]
else:
    _sgs = list(_sgtok.keys())
    _ref = _sgtok.encode_spacegroups(_sgs, dtype=start.dtype, device="cpu").cpu()
    _rmap = {tuple(r.tolist()): sg for r, sg in zip(_ref, _sgs)}
    sg_of = [_rmap[tuple(r.tolist())] for r in start.cpu()]
cond = None
if trainer.condition_feature is not None:
    cd = getattr(trainer.model, "condition_dim", None) or 1
    cond = trainer.transform_condition(torch.full((N, cd), COND, dtype=torch.float32))

gen = WyckoffGenerator(trainer.model, trainer.cascade_order, trainer.cascade_is_target,
                       trainer.token_engineers, trainer.masks_dict, trainer.max_sequence_length)
tensors = gen.generate_tensors(start, cond=cond)
order = list(trainer.cascade_order)
keep = ["elements", "site_symmetries", "sites_enumeration"]
T = torch.stack([tensors[order.index(f)] for f in keep], dim=-1)
order = keep

toks = trainer.tokenisers
letter_from_ss_enum = toks['sites_enumeration'].get_letter_from_ss_enum_idx(RUN)
wp_index = get_wp_index()
sg_tok = toks["spacegroup_number"]
i_el, i_ss, i_en = order.index("elements"), order.index("site_symmetries"), order.index("sites_enumeration")
stop = {f: toks[f].stop_token for f in ("elements", "site_symmetries", "sites_enumeration")}

# per-site-index outcome counters
first_fail = Counter()             # reason -> count of sequences
fail_at_index = Counter()          # site index of first failure
attempts_at_index = Counter()      # sequences that reached site index i alive
fails_by_reason_index = defaultdict(Counter)
len_dist = Counter()
attempted_len = []
accepted_flag = []
reached_len = Counter()

for b in range(N):
    sg = sg_of[b]
    avail = deepcopy(wp_index[sg])
    reason = None
    nsites = 0
    for i in range(T.size(1)):
        el, ss_i, en_i = T[b, i, i_el].item(), T[b, i, i_ss].item(), T[b, i, i_en].item()
        if el == stop["elements"] or ss_i == stop["site_symmetries"] or en_i == stop["sites_enumeration"]:
            break
        attempts_at_index[i] += 1
        ss = toks["site_symmetries"].to_token[ss_i]
        # 1. is the site symmetry legal in this space group at all?
        if ss not in letter_from_ss_enum[sg]:
            reason = "ss_not_in_sg"
        elif en_i not in letter_from_ss_enum[sg][ss]:
            reason = "enum_out_of_range"
        else:
            letter = letter_from_ss_enum[sg][ss][en_i]
            if letter not in avail.get(ss, {}):
                reason = "repeated_0dof_wp"
            else:
                mult, dof = avail[ss][letter]
                if dof == 0:
                    del avail[ss][letter]
        if reason is not None:
            fail_at_index[i] += 1
            fails_by_reason_index[reason][i] += 1
            first_fail[reason] += 1
            break
        nsites += 1
    attempted_len.append(nsites + (1 if reason is not None else 0))
    accepted_flag.append(reason is None)
    reached_len[nsites] += 1
    if reason is None:
        len_dist[nsites] += 1

print(f"run={RUN} N={N}")
print(f"accepted: {sum(len_dist.values())}/{N} = {sum(len_dist.values())/N:.3f}")
print("\nfirst-failure reason (share of all sequences):")
for r, c in first_fail.most_common():
    print(f"  {r:20s} {c:6d}  {c/N:.4f}")
print("\nper-site conditional validity (of sequences alive at index i, fraction that survive site i):")
print(f"  {'i':>3} {'alive':>7} {'fail':>6} {'p_ok':>7}  " + "  ".join(f"{r:>18}" for r in ("ss_not_in_sg","enum_out_of_range","repeated_0dof_wp")))
cum = 1.0
for i in sorted(attempts_at_index):
    a, f = attempts_at_index[i], fail_at_index[i]
    p = 1 - f / a
    cum *= p
    parts = "  ".join(f"{fails_by_reason_index[r][i]/a:18.4f}" for r in ("ss_not_in_sg","enum_out_of_range","repeated_0dof_wp"))
    print(f"  {i:>3} {a:>7} {f:>6} {p:>7.4f}  {parts}   cum={cum:.4f}")
# P(gene accepted | it attempted >= n sites)
print("\nP(gene accepted | attempted >= n sites):")
print(f"  {'n':>3} {'attempted>=n':>13} {'accepted':>9} {'rate':>7}")
import numpy as np
att = np.array(attempted_len)
acc = np.array(accepted_flag)
for n in range(1, 13):
    m = att >= n
    if m.sum() == 0:
        break
    print(f"  {n:>3} {int(m.sum()):>13} {int(acc[m].sum()):>9} {acc[m].mean():>7.3f}")
