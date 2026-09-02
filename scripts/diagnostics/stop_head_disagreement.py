"""How often do the three cascade heads disagree about where the sequence ends?"""
import sys, logging
from collections import Counter
from pathlib import Path
import numpy as np, torch
from omegaconf import OmegaConf
from wyckoff_transformer.trainer import WyckoffTrainer, load_model_weights
from wyckoff_transformer.generator import WyckoffGenerator
logging.basicConfig(level=logging.ERROR)
torch.manual_seed(0)
RUN = Path(sys.argv[1]); N = int(sys.argv[2]); COND = float(sys.argv[3]) if len(sys.argv) > 3 else None
cfg = OmegaConf.load(RUN / "config.yaml")
tr = WyckoffTrainer.from_config(cfg, device=torch.device("cpu"), use_cached_tensors=False,
                                run_path=RUN, load_datasets=False)
load_model_weights(tr.model, RUN / "best_model_params.pt", torch.device("cpu"))
start = tr._sample_start_tokens_from_distribution(N)
cond = None
if tr.condition_feature is not None:
    cd = getattr(tr.model, "condition_dim", None) or 1
    cond = tr.transform_condition(torch.full((N, cd), 0.0 if COND is None else COND))
g = WyckoffGenerator(tr.model, tr.cascade_order, tr.cascade_is_target, tr.token_engineers,
                     tr.masks_dict, tr.max_sequence_length)
T = g.generate_tensors(start, cond=cond)
order = list(tr.cascade_order)
fields = ["elements", "site_symmetries", "sites_enumeration"]
stops = {f: tr.tokenisers[f].stop_token for f in fields}
first = {}
for f in fields:
    t = T[order.index(f)]
    hit = (t == stops[f])
    idx = torch.where(hit.any(1), hit.float().argmax(1), torch.tensor(t.size(1)))
    first[f] = idx.numpy()
a, b, c = first["elements"], first["site_symmetries"], first["sites_enumeration"]
agree = (a == b) & (b == c)
print(f"run={RUN} N={N}")
print(f"all three heads agree on sequence end: {agree.mean():.4f}")
print(f"  elements == site_symmetries : {(a==b).mean():.4f}")
print(f"  elements == sites_enumeration: {(a==c).mean():.4f}")
eff = np.minimum(np.minimum(a, b), c)
print(f"  mean end index: elements {a.mean():.2f}  ss {b.mean():.2f}  enum {c.mean():.2f}  effective(min) {eff.mean():.2f}")
trunc = eff < a
print(f"  genes truncated EARLY because a non-element head said STOP first: {trunc.mean():.4f}")
print(f"    sites lost when that happens: mean {(a-eff)[trunc].mean():.2f}" if trunc.any() else "")
