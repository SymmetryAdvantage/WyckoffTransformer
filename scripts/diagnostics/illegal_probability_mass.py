"""Probability mass the model puts on Wyckoff positions that cannot exist.

Conditioned on sequences that are still ALIVE (have not emitted STOP), and split into
"illegal in the space group at all" vs "legal but already consumed".
"""
import sys, logging
from collections import defaultdict
from pathlib import Path
import numpy as np, torch
from omegaconf import OmegaConf
from wyckoff_transformer.trainer import WyckoffTrainer, load_model_weights
from wyckoff_transformer.tokenization import get_wp_index
logging.basicConfig(level=logging.ERROR); torch.manual_seed(0)
RUN = Path(sys.argv[1]); N = int(sys.argv[2]); COND = float(sys.argv[3]) if len(sys.argv) > 3 else None
cfg = OmegaConf.load(RUN / "config.yaml")
tr = WyckoffTrainer.from_config(cfg, torch.device("cpu"), use_cached_tensors=False,
                                run_path=RUN, load_datasets=False)
load_model_weights(tr.model, RUN / "best_model_params.pt", torch.device("cpu")); model = tr.model.eval()
toks = tr.tokenisers; order = list(tr.cascade_order)
n_ss, n_en = len(toks["site_symmetries"]), len(toks["sites_enumeration"])
stop_ss, stop_en, stop_el = (toks[f].stop_token for f in ("site_symmetries", "sites_enumeration", "elements"))
lfse = toks["sites_enumeration"].get_letter_from_ss_enum_idx(RUN); wpi = get_wp_index()
ss_to_id = {toks["site_symmetries"].to_token[i]: i for i in range(n_ss)}
start = tr._sample_start_tokens_from_distribution(N)
_t = toks["spacegroup_number"]
if start.dim() == 1:
    sg_of = [_t.to_token[i.item()] for i in start]
else:
    _s = list(_t.keys()); _r = _t.encode_spacegroups(_s, dtype=start.dtype, device="cpu").cpu()
    _m = {tuple(r.tolist()): sg for r, sg in zip(_r, _s)}; sg_of = [_m[tuple(r.tolist())] for r in start.cpu()]
cond = None
if tr.condition_feature is not None:
    cd = getattr(model, "condition_dim", None) or 1
    cond = tr.transform_condition(torch.full((N, cd), 0.0 if COND is None else COND))

in_sg = torch.zeros(N, n_ss, dtype=torch.bool)      # site symmetry exists in this group
enum_in_sg = [defaultdict(set) for _ in range(N)]   # (ss_id) -> legal enum ids
avail = [defaultdict(set) for _ in range(N)]        # (ss_id) -> enum ids not yet consumed
dof0 = [set() for _ in range(N)]
for b in range(N):
    for ss, pe in lfse[sg_of[b]].items():
        if ss not in ss_to_id: continue
        s = ss_to_id[ss]
        in_sg[b, s] = True
        for en, letter in pe.items():
            enum_in_sg[b][s].add(en); avail[b][s].add(en)
            if wpi[sg_of[b]][ss][letter][1] == 0:
                dof0[b].add((s, en))
    in_sg[b, stop_ss] = True

MAX = int(sys.argv[4]) if len(sys.argv) > 4 else 10
gen = [torch.full((N, tr.max_sequence_length), toks[f].mask_token, dtype=torch.int64) for f in order]
alive = np.ones(N, dtype=bool)
rows = []
with torch.no_grad():
    for k in range(MAX):
        idx = np.where(alive)[0]
        if len(idx) < 20: break
        cur = None; ss_imp = ss_used = en_imp = en_used = None
        for ci, name in enumerate(order):
            if not tr.cascade_is_target.get(name, False): continue
            p = torch.softmax(model(start, [g[:, :k+1] for g in gen], None, ci, cond=cond), 1)
            if name == "site_symmetries":
                # available = exists in group AND has at least one unconsumed enumeration
                av = torch.zeros(N, n_ss, dtype=torch.bool)
                for b in idx:
                    ok = [s for s, es in avail[b].items() if es]
                    if ok: av[b, ok] = True
                    av[b, stop_ss] = True
                ss_imp = (p * (~in_sg)).sum(1)[idx].mean().item()          # not in the group at all
                ss_used = (p * (in_sg & ~av)).sum(1)[idx].mean().item()    # in group, fully consumed
            if name == "sites_enumeration":
                legal = torch.zeros(N, n_en, dtype=torch.bool)
                free = torch.zeros(N, n_en, dtype=torch.bool)
                for b in idx:
                    s = cur[b]
                    if s == stop_ss:
                        legal[b, stop_en] = True; free[b, stop_en] = True; continue
                    if enum_in_sg[b].get(s): legal[b, list(enum_in_sg[b][s])] = True
                    if avail[b].get(s): free[b, list(avail[b][s])] = True
                    legal[b, stop_en] = True; free[b, stop_en] = True
                en_imp = (p * (~legal)).sum(1)[idx].mean().item()
                en_used = (p * (legal & ~free)).sum(1)[idx].mean().item()
            tok = torch.multinomial(p, 1).squeeze(1)
            gen[ci][:, k] = tok
            if name == "site_symmetries": cur = tok.tolist()
            if name == "elements": newly = (tok == stop_el).numpy()
        rows.append((k, len(idx), ss_imp, ss_used, en_imp, en_used))
        en_tok = gen[order.index("sites_enumeration")][:, k].tolist()
        for b in idx:
            s, e = cur[b], en_tok[b]
            if (s, e) in dof0[b]: avail[b][s].discard(e)
        alive &= ~newly
        alive &= (gen[order.index("site_symmetries")][:, k] != stop_ss).numpy()
print(f"run={RUN}  N={N}   (means over sequences still alive at that index)")
print(f"  {'i':>3} {'alive':>6} {'p(ss not in SG)':>16} {'p(ss exhausted)':>16} {'p(enum illegal)':>16} {'p(enum consumed)':>17}")
for k, n, a, b_, c, d in rows:
    print(f"  {k:>3} {n:>6} {a:>16.4f} {b_:>16.4f} {c:>16.4f} {d:>17.4f}")
