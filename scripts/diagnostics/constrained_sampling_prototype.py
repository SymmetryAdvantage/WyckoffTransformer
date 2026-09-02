"""Prototype: symmetry-constrained sampling for WyFormer.

At every step the logits for `site_symmetries` and `sites_enumeration` are masked to the
Wyckoff positions that are (a) legal in the sampled space group and (b) not already consumed
(0-DoF positions can be used once). Compares against the unconstrained sampler.
"""
import sys, logging
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
import numpy as np
import torch
from omegaconf import OmegaConf

from wyckoff_transformer.trainer import WyckoffTrainer, load_model_weights
from wyckoff_transformer.tokenization import get_wp_index

logging.basicConfig(level=logging.ERROR)
torch.manual_seed(0)

RUN = Path(sys.argv[1]); N = int(sys.argv[2])
COND = float(sys.argv[3]) if len(sys.argv) > 3 else None
# Optional: dump the accepted genes' space groups and site counts for downstream comparison.
DUMP = Path(sys.argv[4]) if len(sys.argv) > 4 else None

config = OmegaConf.load(RUN / "config.yaml")
trainer = WyckoffTrainer.from_config(config, device=torch.device("cpu"),
                                     use_cached_tensors=False, run_path=RUN, load_datasets=False)
load_model_weights(trainer.model, RUN / "best_model_params.pt", torch.device("cpu"))
model = trainer.model.eval()
toks = trainer.tokenisers
order = list(trainer.cascade_order)
i_el, i_ss, i_en = order.index("elements"), order.index("site_symmetries"), order.index("sites_enumeration")
n_ss, n_en = len(toks["site_symmetries"]), len(toks["sites_enumeration"])
stop_el = toks["elements"].stop_token
stop_ss, stop_en = toks["site_symmetries"].stop_token, toks["sites_enumeration"].stop_token
letter_from_ss_enum = toks["sites_enumeration"].get_letter_from_ss_enum_idx(RUN)
wp_index = get_wp_index()
ss_id = {toks["site_symmetries"].to_token[i]: i for i in range(n_ss)
         if not isinstance(toks["site_symmetries"].to_token[i], str) or True}
ss_to_id = {}
for i in range(n_ss):
    t = toks["site_symmetries"].to_token[i]
    ss_to_id[t] = i

# Precompute per space group: list of (ss_token_id, enum_token_id, letter, dof)
legal = {}
for sg, per_ss in letter_from_ss_enum.items():
    entries = []
    for ss, per_enum in per_ss.items():
        if ss not in ss_to_id:
            continue
        for en, letter in per_enum.items():
            mult, dof = wp_index[sg][ss][letter]
            entries.append((ss_to_id[ss], en, ss, letter, dof))
    legal[sg] = entries

start = trainer._sample_start_tokens_from_distribution(N)
_sgtok = toks["spacegroup_number"]
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
    v = 0.0 if COND is None else COND
    cond = trainer.transform_condition(torch.full((N, cd), v, dtype=torch.float32))

MAXLEN = trainer.max_sequence_length
NEG = float("-inf")


@torch.no_grad()
def generate(masked: bool):
    gen = [torch.full((N, MAXLEN), toks[f].mask_token, dtype=torch.int64) for f in order]
    # availability: per sample, dict ss_id -> set of enum ids still usable
    avail = []
    for b in range(N):
        d = defaultdict(set)
        for ssid, en, ss, letter, dof in legal[sg_of[b]]:
            d[ssid].add(en)
        avail.append(d)
    dof0 = [{(ssid, en) for ssid, en, ss, letter, dof in legal[sg_of[b]] if dof == 0} for b in range(N)]
    done = np.zeros(N, dtype=bool)
    premask_ss = []
    for k in range(MAXLEN):
        cur_ss = None
        for ci, name in enumerate(order):
            if not trainer.cascade_is_target.get(name, False):
                continue
            inp = [g[:, :k + 1] for g in gen]
            logits = model(start, inp, None, ci, cond=cond)
            if masked and name == "site_symmetries":
                # The diagnostic is taken BEFORE the mask: illegal probability mass is
                # observable whether or not the sample is constrained, so keeping the
                # proxy costs nothing.
                _p = torch.softmax(logits, dim=1)
                _live = [b for b in range(N) if not done[b]]
                if _live:
                    _leg = torch.zeros(len(_live), n_ss, dtype=torch.bool)
                    for j, b in enumerate(_live):
                        _leg[j, list(avail[b].keys())] = True
                        _leg[j, stop_ss] = True
                    premask_ss.append(((_p[_live] * (~_leg)).sum(1).mean().item(), len(_live)))
                m = torch.full((N, n_ss), NEG)
                for b in range(N):
                    if done[b]:
                        m[b, stop_ss] = 0.
                        continue
                    ok = [s for s, es in avail[b].items() if es]
                    m[b, ok] = 0.
                    m[b, stop_ss] = 0.
                logits = logits + m
            if masked and name == "sites_enumeration":
                m = torch.full((N, n_en), NEG)
                for b in range(N):
                    s = cur_ss[b]
                    if done[b] or s == stop_ss:
                        m[b, stop_en] = 0.
                        continue
                    m[b, list(avail[b][s])] = 0.
                logits = logits + m
            p = torch.softmax(logits, dim=1)
            tok = torch.multinomial(p, 1).squeeze(1)
            gen[ci][:, k] = tok
            if name == "elements":
                done |= (tok == stop_el).numpy()
            if name == "site_symmetries":
                cur_ss = tok.tolist()
                if masked:
                    done |= (tok == stop_ss).numpy()
        if masked:
            en_tok = gen[i_en][:, k].tolist()
            for b in range(N):
                if done[b]:
                    continue
                s, e = cur_ss[b], en_tok[b]
                if (s, e) in dof0[b]:
                    avail[b][s].discard(e)
        if done.all():
            break
    if masked and premask_ss:
        print("\npre-mask diagnostic, still available with constrained sampling:")
        print(f"  {'i':>3} {'live':>6} {'p(illegal site symmetry)':>25}")
        for i, (v, n) in enumerate(premask_ss):
            if n >= 20:
                print(f"  {i:>3} {n:>6} {v:>25.4f}")
    return torch.stack([gen[i_el], gen[i_ss], gen[i_en]], dim=-1)


def audit(T, tag):
    reasons = Counter(); att = []; acc = []
    nsites_all = []
    for b in range(N):
        sg = sg_of[b]; a = deepcopy(wp_index[sg]); reason = None; n = 0
        for i in range(T.size(1)):
            el, si, ei = T[b, i, 0].item(), T[b, i, 1].item(), T[b, i, 2].item()
            if el == stop_el or si == stop_ss or ei == stop_en:
                break
            ss = toks["site_symmetries"].to_token[si]
            if ss not in letter_from_ss_enum[sg]:
                reason = "ss_not_in_sg"
            elif ei not in letter_from_ss_enum[sg][ss]:
                reason = "enum_out_of_range"
            else:
                letter = letter_from_ss_enum[sg][ss][ei]
                if letter not in a.get(ss, {}):
                    reason = "repeated_0dof_wp"
                else:
                    if a[ss][letter][1] == 0:
                        del a[ss][letter]
            if reason:
                reasons[reason] += 1
                break
            n += 1
        att.append(n + (1 if reason else 0)); acc.append(reason is None and n > 0)
        nsites_all.append(n)
    att = np.array(att); acc = np.array(acc)
    print(f"\n=== {tag} ===")
    print(f"accepted {acc.sum()}/{N} = {acc.mean():.3f}")
    for r, c in reasons.most_common():
        print(f"  {r:20s} {c/N:.4f}")
    print("  n_sites of ACCEPTED genes: mean %.2f  hist %s" % (
        np.mean([x for x, ok in zip(nsites_all, acc) if ok]),
        Counter(x for x, ok in zip(nsites_all, acc) if ok).most_common(8)))
    print("  n_sites ATTEMPTED (all):   mean %.2f" % att.mean())
    return acc, att, nsites_all


Tu = generate(masked=False)
au = audit(Tu, "unconstrained (current)")
Tm = generate(masked=True)
am = audit(Tm, "symmetry-constrained sampling")

# --- Does whole-sequence rejection distort the space-group distribution? ---
import collections
accu, _, _ = au
accm, _, _ = am
req = collections.Counter(sg_of)
got_u = collections.Counter(sg for sg, ok in zip(sg_of, accu) if ok)
got_m = collections.Counter(sg for sg, ok in zip(sg_of, accm) if ok)
rows = []
for sg, n in req.items():
    if n >= 15:
        rows.append((sg, n, got_u[sg] / n, got_m[sg] / n, len(legal[sg])))
rows.sort(key=lambda r: r[2])
print("\n=== per-space-group acceptance (groups with >=15 requests) ===")
print(f"  {'sg':>4} {'asked':>6} {'accept_unconstrained':>21} {'accept_masked':>14} {'n_WP_in_group':>14}")
for sg, n, pu, pm, nw in rows:
    print(f"  {sg:>4} {n:>6} {pu:>21.3f} {pm:>14.3f} {nw:>14}")
ru = np.array([r[2] for r in rows]); nw = np.array([float(r[4]) for r in rows])
print(f"\n  spread of per-SG acceptance, unconstrained: min {ru.min():.3f} max {ru.max():.3f} sd {ru.std():.3f}")
print(f"  corr(accept_rate, n_Wyckoff_positions_in_group) = {np.corrcoef(ru, nw)[0,1]:.3f}")
# total variation distance between requested and delivered SG distribution
def tvd(a, b):
    keys = set(a) | set(b)
    ta, tb = sum(a.values()), sum(b.values())
    return 0.5 * sum(abs(a[k]/ta - b[k]/tb) for k in keys)
print(f"  TVD(requested SG dist, delivered SG dist) unconstrained = {tvd(req, got_u):.4f}")
print(f"  TVD(requested SG dist, delivered SG dist) masked        = {tvd(req, got_m):.4f}")

if DUMP is not None:
    import json
    payload = {}
    for tag, (accepted, _attempted, n_sites) in (("unconstrained", au), ("masked", am)):
        payload[tag] = {
            "sg": [int(sg_of[i]) for i in range(N) if accepted[i]],
            "nsites": [int(n_sites[i]) for i in range(N) if accepted[i]],
        }
    DUMP.write_text(json.dumps(payload))
    print(f"\nwrote {DUMP}")
