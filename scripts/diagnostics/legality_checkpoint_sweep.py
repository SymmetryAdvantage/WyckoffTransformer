"""Frequency-weighted p(illegal site symmetry) at site 0, across every local checkpoint."""
import sys, logging, pickle, gzip, warnings
from pathlib import Path
from collections import Counter
import numpy as np, torch
from omegaconf import OmegaConf
logging.basicConfig(level=logging.CRITICAL); warnings.filterwarnings("ignore")
from wyckoff_transformer.trainer import WyckoffTrainer, load_model_weights

R = 24
_freq_cache = {}
def freq_for(ds):
    if ds not in _freq_cache:
        d = pickle.load(gzip.open(f"cache/{ds}/data.pkl.gz", "rb"))
        c = Counter()
        for split in d.values():
            c.update(split["spacegroup_number"])
        _freq_cache[ds] = c
    return _freq_cache[ds]

def measure(run: Path):
    cfg = OmegaConf.load(run / "config.yaml")
    ds = cfg.get("dataset") or cfg.tokeniser.name.replace("_sg_multiplicity", "").replace("_energy", "")
    tr = WyckoffTrainer.from_config(cfg, torch.device("cpu"), use_cached_tensors=False,
                                    run_path=run, load_datasets=False)
    load_model_weights(tr.model, run / "best_model_params.pt", torch.device("cpu"))
    model = tr.model.eval(); toks = tr.tokenisers; order = list(tr.cascade_order)
    if "sites_enumeration" not in order:
        return None
    n_ss = len(toks["site_symmetries"]); stop_ss = toks["site_symmetries"].stop_token
    lfse = toks["sites_enumeration"].get_letter_from_ss_enum_idx(run)
    ss_to_id = {toks["site_symmetries"].to_token[i]: i for i in range(n_ss)}
    sgtok = toks["spacegroup_number"]; sgs = sorted(sgtok.keys())
    rep = [s for s in sgs for _ in range(R)]
    start = (sgtok.encode_spacegroups(rep, dtype=torch.float32, device="cpu")
             if model.start_type == "one_hot" else torch.tensor([sgtok[s] for s in rep]))
    B = len(rep)
    cond = None
    if tr.condition_feature is not None:
        cd = getattr(model, "condition_dim", None) or 1
        cond = tr.transform_condition(torch.zeros(B, cd))
    legal = torch.zeros(B, n_ss, dtype=torch.bool)
    for r, s in enumerate(rep):
        for ss in lfse[s]:
            if ss in ss_to_id: legal[r, ss_to_id[ss]] = True
        legal[r, stop_ss] = True
    inp = [torch.full((B, 1), toks[f].mask_token, dtype=torch.int64) for f in order]
    ie, ic = order.index("elements"), order.index("site_symmetries")
    with torch.no_grad():
        inp[ie][:, 0] = torch.multinomial(torch.softmax(model(start, inp, None, ie, cond=cond), 1), 1).squeeze(1)
        p = torch.softmax(model(start, inp, None, ic, cond=cond), 1)
    bad = (p * (~legal)).sum(1).numpy().reshape(len(sgs), R).mean(1)
    f = freq_for(ds); n = np.array([f.get(s, 0) for s in sgs], dtype=float)
    w = n / n.sum() if n.sum() else np.ones(len(sgs)) / len(sgs)
    nparam = sum(q.numel() for q in model.parameters())
    common = n >= 3000
    return ds, (bad * w).sum(), bad.mean(), (bad[common].mean() if common.any() else float("nan")), nparam

runs = sorted(p for p in Path("runs").iterdir() if (p / "best_model_params.pt").exists())
print(f"{'run':>10} {'dataset':>22} {'params':>9} {'weighted':>9} {'unweighted':>11} {'common SGs':>11}")
for r in runs:
    try:
        out = measure(r)
    except Exception as e:
        print(f"{r.name:>10} {'-- ' + type(e).__name__:>22}"); continue
    if out is None:
        print(f"{r.name:>10} {'(no enumeration head)':>22}"); continue
    ds, w, u, c, npar = out
    print(f"{r.name:>10} {ds:>22} {npar:>9,} {w:>9.4f} {u:>11.4f} {c:>11.4f}", flush=True)
