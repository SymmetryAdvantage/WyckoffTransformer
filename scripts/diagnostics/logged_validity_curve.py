"""Reproduce the W&B ss_validity / enumeration_validity curve, and correct it.

`WyckoffTrainer.generate_structures` builds `WyckoffGenerator` without `stops`, so
`self.stops is None`, `stop_generated` stays all-False, and the skip at generator.py:348
never fires. The logged curve at known_seq_len k therefore averages over every generated
structure, including the majority that emitted STOP long before k and are being asked to
continue a sequence that already ended.

This prints both: the metric as logged, and the same metric over live sequences only.
"""
import sys, logging
from pathlib import Path
import numpy as np, torch
from omegaconf import OmegaConf
from wyckoff_transformer.trainer import WyckoffTrainer, load_model_weights
from wyckoff_transformer.generator import WyckoffGenerator
logging.basicConfig(level=logging.ERROR); torch.manual_seed(0)

RUN = Path(sys.argv[1]); N = int(sys.argv[2]) if len(sys.argv) > 2 else 1100
COND = float(sys.argv[3]) if len(sys.argv) > 3 else None
cfg = OmegaConf.load(RUN / "config.yaml")
tr = WyckoffTrainer.from_config(cfg, torch.device("cpu"), use_cached_tensors=False,
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
i_ss, i_en = order.index("site_symmetries"), order.index("sites_enumeration")
i_el = order.index("elements")
stop_el = tr.tokenisers["elements"].stop_token
stop_ss = tr.tokenisers["site_symmetries"].stop_token
stop_en = tr.tokenisers["sites_enumeration"].stop_token
db = tr.token_engineers["multiplicity"].db
sc = list(map(tuple, start.tolist())) if start.dim() > 1 else start.tolist()

print(f"run={RUN}  N={N}")
print(f"  {'k':>3} {'live':>6} {'stop@k':>7} | {'ss AS LOGGED':>13} | {'1-P(stop|live)':>15} | {'ss REAL SITES':>14} {'enum REAL SITES':>16}")
alive = np.ones(N, dtype=bool)
for k in range(min(21, T[0].size(1))):
    stopped = ((T[i_el][:, k] == stop_el) | (T[i_ss][:, k] == stop_ss) | (T[i_en][:, k] == stop_en)).numpy()
    ss_all, ss_real, en_real = [], [], []
    for b in range(N):
        s = T[i_ss][b, k].item(); e = T[i_en][b, k].item()
        ss_all.append((sc[b], s) in db)
        if alive[b] and not stopped[b]:
            ss_real.append((sc[b], s) in db)
            en_real.append((sc[b], s, e) in db)
    nl = int(alive.sum()); ns = int((alive & stopped).sum())
    pstop = 1 - ns / nl if nl else float("nan")
    r_ss = np.mean(ss_real) if ss_real else float("nan")
    r_en = np.mean(en_real) if en_real else float("nan")
    print(f"  {k:>3} {nl:>6} {ns:>7} | {np.mean(ss_all):>13.4f} | {pstop:>15.4f} | {r_ss:>14.4f} {r_en:>16.4f}")
    alive &= ~stopped
    if alive.sum() < 10:
        break
