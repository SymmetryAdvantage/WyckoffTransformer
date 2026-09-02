"""Illegal-site-symmetry mass at the FIRST site, per space group, vs training frequency.

At site 0 there is no occupancy confound and no dead-sequence confound: the only thing the
model has to know is which site symmetries exist in the group it was handed.
"""
import sys, logging, json, pickle, gzip
from pathlib import Path
import numpy as np, torch
from omegaconf import OmegaConf
from wyckoff_transformer.trainer import WyckoffTrainer, load_model_weights
logging.basicConfig(level=logging.ERROR)
RUN = Path(sys.argv[1]); DATA = sys.argv[2]; COND = float(sys.argv[3]) if len(sys.argv) > 3 else None
cfg = OmegaConf.load(RUN / "config.yaml")
tr = WyckoffTrainer.from_config(cfg, torch.device("cpu"), use_cached_tensors=False,
                                run_path=RUN, load_datasets=False)
load_model_weights(tr.model, RUN / "best_model_params.pt", torch.device("cpu")); model = tr.model.eval()
toks = tr.tokenisers; order = list(tr.cascade_order)
n_ss = len(toks["site_symmetries"]); stop_ss = toks["site_symmetries"].stop_token
lfse = toks["sites_enumeration"].get_letter_from_ss_enum_idx(RUN)
ss_to_id = {toks["site_symmetries"].to_token[i]: i for i in range(n_ss)}
sgtok = toks["spacegroup_number"]
sgs = sorted(sgtok.keys())

d = pickle.load(gzip.open(f"cache/{DATA}/data.pkl.gz", "rb"))
from collections import Counter
freq = Counter()
for split in d.values():
    freq.update(split["spacegroup_number"])
total = sum(freq.values())

R = int(sys.argv[4]) if len(sys.argv) > 4 else 64   # samples per space group
sgs_rep = [s for s in sgs for _ in range(R)]
start = sgtok.encode_spacegroups(sgs_rep, dtype=torch.float32, device="cpu") \
    if tr.model.start_type == "one_hot" else torch.tensor([sgtok[s] for s in sgs_rep])
B = len(sgs_rep)
cond = None
if tr.condition_feature is not None:
    cd = getattr(model, "condition_dim", None) or 1
    cond = tr.transform_condition(torch.full((B, cd), 0.0 if COND is None else COND))
legal = torch.zeros(B, n_ss, dtype=torch.bool)
for r, s in enumerate(sgs_rep):
    for ss in lfse[s]:
        if ss in ss_to_id: legal[r, ss_to_id[ss]] = True
    legal[r, stop_ss] = True
inp = [torch.full((B, 1), toks[f].mask_token, dtype=torch.int64) for f in order]
ci_el, ci = order.index("elements"), order.index("site_symmetries")
with torch.no_grad():
    # the site-symmetry head never sees MASK in the element slot: sample the element first
    pe = torch.softmax(model(start, inp, None, ci_el, cond=cond), 1)
    inp[ci_el][:, 0] = torch.multinomial(pe, 1).squeeze(1)
    p = torch.softmax(model(start, inp, None, ci, cond=cond), 1)
bad_raw = (p * (~legal)).sum(1).numpy()
bad = bad_raw.reshape(len(sgs), R).mean(1)
n_tr = np.array([freq.get(s, 0) for s in sgs], dtype=float)
print(f"run={RUN}  data={DATA}  {B} space groups in the tokeniser, {total:,} training structures")
print(f"mean p(illegal site symmetry) at site 0, unweighted over groups : {bad.mean():.4f}")
w = n_tr / n_tr.sum()
print(f"                                weighted by training frequency  : {(bad*w).sum():.4f}")
bins = [(0, 0), (1, 30), (31, 300), (301, 3000), (3001, 10**9)]
print(f"\n  {'training structures in group':>30} {'groups':>7} {'mean p(illegal)':>16} {'median':>9}")
for lo, hi in bins:
    m = (n_tr >= lo) & (n_tr <= hi)
    if m.sum() == 0: continue
    lab = "0 (unseen)" if hi == 0 else f"{lo}-{hi}" if hi < 10**8 else f"{lo}+"
    print(f"  {lab:>30} {int(m.sum()):>7} {bad[m].mean():>16.4f} {np.median(bad[m]):>9.4f}")
o = np.argsort(-bad)
print("\n  worst 12 groups:")
print(f"  {'sg':>4} {'p(illegal)':>11} {'train count':>12}")
for i in o[:12]:
    print(f"  {sgs[i]:>4} {bad[i]:>11.4f} {int(n_tr[i]):>12}")
lg = np.log10(n_tr + 1)
print(f"\n  corr(p_illegal, log10 training count) = {np.corrcoef(bad, lg)[0,1]:.3f}")
