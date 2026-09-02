"""Is the SG -> legal-site-symmetry map learnable from the encoding the model is given?

Upper bound on difficulty: fit a LINEAR map from SpaceGroupEncoder's vector to the
multi-hot indicator of which site symmetries exist in the group. If linear suffices, a
140k-parameter transformer cannot be blamed on capacity for getting this wrong.
"""
import sys
import numpy as np, torch, json
from pathlib import Path
from wyckoff_transformer.tokenization import SpaceGroupEncoder, load_wyckoff_mappings
from pyxtal.symmetry import Group
from collections import Counter

# Any run directory works: only its `wyckoffs_enumerated_by_ss.json` is read.
RUN = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("runs/upi73i4k")
maps = load_wyckoff_mappings(RUN)
sgs = sorted(int(k) for k in maps.ss_from_letter.keys())
enc = SpaceGroupEncoder.from_sg_set(set(sgs))
X = np.stack([np.asarray(enc[s], dtype=np.float64) for s in sgs])
all_ss = sorted({ss for s in sgs for ss in maps.ss_from_letter[s].values()})
ssi = {s: i for i, s in enumerate(all_ss)}
Y = np.zeros((len(sgs), len(all_ss)))
for r, s in enumerate(sgs):
    for ss in set(maps.ss_from_letter[s].values()):
        Y[r, ssi[ss]] = 1.
print(f"space groups {len(sgs)}   encoding dim {X.shape[1]}   site symmetry vocab {len(all_ss)}")
print(f"legal set size: mean {Y.sum(1).mean():.1f}  max {int(Y.sum(1).max())}  "
      f"so a uniform-over-vocab guess would be {Y.sum(1).mean()/len(all_ss):.3f} legal mass")

Xa = np.hstack([X, np.ones((len(sgs), 1))])
W, *_ = np.linalg.lstsq(Xa, Y, rcond=None)
P = Xa @ W
pred = (P > 0.5)
exact = (pred == (Y > 0.5)).all(1).mean()
err = (pred != (Y > 0.5)).sum()
print(f"\nLINEAR least-squares readout ({Xa.shape[1]}x{Y.shape[1]} = {Xa.shape[1]*Y.shape[1]:,} params):")
print(f"  space groups with the legal set recovered EXACTLY: {exact:.3f}")
print(f"  total bit errors over {Y.size:,} (group, site symmetry) cells: {err}")
print(f"  rank of the encoding matrix: {np.linalg.matrix_rank(Xa)} of {Xa.shape[1]}")

# how many hidden units does an MLP need?
import torch.nn as nn
Xt = torch.tensor(X, dtype=torch.float32); Yt = torch.tensor(Y, dtype=torch.float32)
for h in (8, 16, 32, 64):
    torch.manual_seed(0)
    m = nn.Sequential(nn.Linear(X.shape[1], h), nn.ReLU(), nn.Linear(h, len(all_ss)))
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    for _ in range(4000):
        opt.zero_grad(); l = nn.functional.binary_cross_entropy_with_logits(m(Xt), Yt); l.backward(); opt.step()
    with torch.no_grad():
        acc = ((m(Xt) > 0).float() == Yt).all(1).float().mean().item()
        bits = ((m(Xt) > 0).float() != Yt).sum().item()
    n = sum(p.numel() for p in m.parameters())
    print(f"  MLP hidden={h:>3} ({n:>6,} params): exact-set accuracy {acc:.3f}, bit errors {int(bits)}")
