"""Per reviewed MP row: compare geometry with each candidate task and choose the energy source.

Writes decisions.parquet (one row per reviewed LeMat row) and replacements.jsonl (the new
per-row payload for every row whose DFT record is replaced).
"""
import sys, json, glob, numpy as np, pandas as pd
from pathlib import Path
from scipy.optimize import linear_sum_assignment

W = Path(sys.argv[1])
T_OK = 0.2       # eV/atom: |DFT - MLIP median| bound for "MLIP-consistent" (99.5th pct on clean rows)
REF_GAIN = 0.05  # eV/atom: a live row switches only if the candidate is this much closer to the MLIPs
SAME, NEAR = 1e-3, 0.05  # A, lattice and site deviation

f = pd.read_parquet(W / "identify.parquet")
f = f[f.review != ""].set_index("immutable_id")
cand = pd.read_parquet(W / "candidates.parquet")
cand_by = {k: g for k, g in cand.groupby("immutable_id")}
ext = {}
for fn in glob.glob(str(W / "s3/*.jsonl")):
    for line in open(fn):
        d = json.loads(line)
        ext[d["task_id"]] = d
L = pd.read_parquet(W / "lemat_review_rows.parquet",
                    columns=["immutable_id", "lattice_vectors", "cartesian_site_positions",
                             "species_at_sites"]).drop_duplicates("immutable_id").set_index("immutable_id")


def compare(lat, sp, xyz, d):
    """Max lattice deviation, permutation (task sites -> LeMat order), max site deviation."""
    tl = np.asarray(d["lattice"], float); tsp = np.asarray(d["species"]); txyz = np.asarray(d["xyz"], float)
    sp = np.asarray(sp)
    if len(tsp) != len(sp) or sorted(tsp) != sorted(sp):
        return np.inf, None, np.inf
    dlat = float(np.abs(tl - lat).max())
    if dlat > NEAR:
        return dlat, None, np.inf
    inv = np.linalg.inv(lat); perm = np.empty(len(sp), int)
    for el in np.unique(sp):
        i = np.flatnonzero(sp == el); j = np.flatnonzero(tsp == el)
        fr = (txyz[j][None, :, :] - xyz[i][:, None, :]) @ inv; fr -= np.round(fr)
        rr, cc = linear_sum_assignment(np.linalg.norm(fr @ lat, axis=2)); perm[i[rr]] = j[cc]
    fr = (txyz[perm] - xyz) @ inv; fr -= np.round(fr)
    return dlat, perm, float(np.linalg.norm(fr @ lat, axis=1).max())


records, payloads = [], []
for mid, r in f.iterrows():
    lr = L.loc[mid]
    lat = np.stack(lr.lattice_vectors); xyz = np.stack(lr.cartesian_site_positions); sp = list(lr.species_at_sites)
    mlip = r.epa - r.d_med if pd.notna(r.d_med) else np.nan
    a = abs(r.d_med) if pd.notna(r.d_med) else np.nan
    cs = []
    for c in (cand_by.get(mid).itertuples() if mid in cand_by else []):
        d = ext.get(c.task_id)
        if d is None or "error" in d or d.get("energy") is None:
            continue
        dlat, perm, dpos = compare(lat, sp, xyz, d)
        geo = "same" if max(dlat, dpos) <= SAME else "near" if max(dlat, dpos) <= NEAR else "different"
        e = d["energy"] / len(d["species"])
        cs.append(dict(task_id=c.task_id, dep=c.task_deprecated, is_lemat=c.is_lemat_task, is_mp=c.is_mp_task,
                       calc_type=c.calc_type, geo=geo, dlat=dlat, dpos=dpos, epa=e,
                       b=abs(e - mlip) if geo != "different" else np.nan, perm=perm, doc=d))
    live = [c for c in cs if not c["dep"] and not (c["is_lemat"] and r.identified)
            and (not isinstance(r.mp_run, str) or c["calc_type"].split(" ")[0] == r.mp_run)]
    live_geo = [c for c in live if c["geo"] != "different"]
    mp_c = next((c for c in live if c["is_mp"]), None)
    choice, action = None, None
    if r.review in ("control_same", "control_live"):
        action = "control"
        choice = next((c for c in cs if c["is_lemat"]), None)
    elif r.review == "stale":
        if mp_c is not None and mp_c["geo"] != "different" and (np.isnan(a) or mp_c["b"] <= max(T_OK, a + REF_GAIN)):
            choice, action = mp_c, "replace_energy"
        elif live_geo:
            best = min(live_geo, key=lambda c: (c["b"], c["calc_type"].endswith("Optimization")))
            if np.isnan(a) or best["b"] <= max(T_OK, a + REF_GAIN):
                choice, action = best, "replace_energy"
        if choice is None and mp_c is not None and mp_c["geo"] == "different":
            # MLIPs were evaluated on LeMat's geometry; the re-relaxed one is close (median dV 2e-4),
            # so the same referee applies with the MLIP energy at LeMat's geometry as the proxy.
            b_s = abs(mp_c["epa"] - mlip) if pd.notna(mlip) else np.nan
            if np.isnan(a) or np.isnan(b_s) or b_s <= max(T_OK, a + REF_GAIN):
                choice, action = mp_c, "replace_structure"
        if choice is None:
            action = "keep_stale_mlip_consistent" if (pd.notna(a) and a <= T_OK and r.material_deprecated is False) else "remove"
    else:  # live_big
        if live_geo and pd.notna(a) and a > T_OK:
            best = min(live_geo, key=lambda c: c["b"])
            if best["b"] < a - REF_GAIN and best["b"] <= T_OK:
                choice, action = best, "replace_energy"
        if action is None and mp_c is not None and mp_c["geo"] == "different" and pd.notna(a) and a > T_OK:
            b_s = abs(mp_c["epa"] - mlip)
            if b_s < a - REF_GAIN and b_s <= T_OK:
                choice, action = mp_c, "replace_structure"
        if action is None:
            action = "keep"
    rec = dict(immutable_id=mid, review=r.review, grp=r.grp, action=action, n_cand=len(cs), n_live_geo=len(live_geo),
               old_epa=r.epa, mlip_epa=mlip, old_vs_mlip=r.d_med, lemat_task=r.lemat_task, lemat_calc_type=r.lemat_calc_type,
               lemat_batch=r.lemat_batch, lemat_task_deprecated=r.lemat_task_deprecated, mp_task=r.mp_task,
               mp_calc_type=r.mp_calc_type, material_deprecated=r.material_deprecated)
    if choice is not None:
        d = choice["doc"]
        rec.update(new_task=choice["task_id"], new_calc_type=choice["calc_type"], new_epa=choice["epa"],
                   new_vs_mlip=choice["epa"] - mlip, geo=choice["geo"], dlat=choice["dlat"], dpos=choice["dpos"],
                   new_is_mp_task=choice["is_mp"])
        if action in ("replace_energy", "replace_structure", "control"):
            perm = choice["perm"] if choice["perm"] is not None else np.arange(len(d["species"]))
            take = lambda v: None if v is None else [v[i] for i in perm]
            payloads.append(dict(immutable_id=mid, action=action, task_id=choice["task_id"], energy=d["energy"],
                                 perm=[int(i) for i in perm],
                                 lattice=d["lattice"], species=take(d["species"]), xyz=take(d["xyz"]),
                                 forces=take(d["forces"]) if d.get("forces") else None, stress=d.get("stress"),
                                 magmom=take(d["magmom"]), total_magnetization=d.get("total_magnetization"),
                                 efermi=d.get("efermi"), last_updated=d.get("last_updated"),
                                 potcars=d.get("potcars"), hubbards=d.get("hubbards")))
    records.append(rec)

out = pd.DataFrame(records)
out.to_parquet(W / "decisions.parquet")
with open(W / "replacements.jsonl", "w") as fh:
    for p in payloads:
        fh.write(json.dumps(p) + "\n")
pd.set_option("display.width", 220)
print(pd.crosstab([out.review, out.grp], out.action, margins=True))
print(pd.crosstab(out.action, out.geo.fillna("-")))
