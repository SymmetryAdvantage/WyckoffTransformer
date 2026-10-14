"""Apply the decisions to LeMat-Bulk compatible_pbe and write the fixed shards plus the audit table.

usage: build_fixed.py WORK OUT
"""
import sys, json, glob, os, collections, numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq
from pathlib import Path

W, OUT = Path(sys.argv[1]), Path(sys.argv[2])
SRC = Path(os.path.expanduser("~/.cache/huggingface/hub/datasets--LeMaterial--LeMat-Bulk/snapshots/"
                              "0dc17eea904b860ad7288141e9870f67f8e6bb2c/compatible_pbe"))

sys.path.insert(0, str(Path(__file__).parent))
from tsutil import lemat_time
dec = pd.read_parquet(W / "decisions.parquet").set_index("immutable_id")
task_time = {t["task_id"]: t.get("last_updated") for t in json.load(open(W / "tasks_meta.json"))}
rep = {}
for line in open(W / "replacements.jsonl"):
    p = json.loads(line)
    if p["action"] != "control":
        rep[p["immutable_id"]] = p
joined = pd.read_parquet(W / "joined.parquet").drop_duplicates("immutable_id")
ml = ["orb_direct_20", "orb_conserv_inf", "uma", "mace_mp", "mace_omat"]

# --- forces and stress that upstream (or a replacement task's output) left empty
# Untouched rows: scripts/recover_mp_forces.py matched LeMat's own task bit-exactly and took the
# values from its last ionic step, already in LeMat's site order.
RECOVERY = Path(os.path.expanduser("~/.local/share/wyformer/cache/mp_forces_recovery/runs/full/results.parquet"))
recovered = pd.read_parquet(RECOVERY, columns=["immutable_id", "group", "status", "forces", "stress"])
recovered = recovered[(recovered.group == "missing") & recovered.status.str.startswith("exact")].set_index("immutable_id")
# Replacement tasks: the last ionic step, kept only when it is the task's output (s3_steps.py).
steps = {}
if (W / "s3_steps" / "steps.jsonl").exists():
    for line in open(W / "s3_steps" / "steps.jsonl"):
        d = json.loads(line)
        if d["ok"]:
            steps[d["task_id"]] = d
for i, p in rep.items():
    st = steps.get(p["task_id"])
    if st is not None:
        if not p.get("forces"):
            p["forces"] = [st["forces"][j] for j in p["perm"]]; p["forces_source"] = "last_ionic_step"
        if not p.get("stress"):
            p["stress"] = st["stress"]; p["stress_source"] = "last_ionic_step"
print("replacements filled from the last ionic step:", sum("forces_source" in p for p in rep.values()),
      "| still without forces:", sum(not p.get("forces") for p in rep.values()))

# --- removals, each with a calculation-level cause (removal_causes.py); nothing is removed for
# disagreeing with the MLIPs alone
from removal_causes import CAUSES, UNEXPLAINED, POTCAR_OFFSETS
ident = pd.read_parquet(W / "identify.parquet").set_index("immutable_id")
x = joined.set_index("immutable_id")
remove, cause, flagged = {}, {}, {}
for i, r in dec[dec.action == "remove"].iterrows():
    pots = [p for p in str(ident.at[i, "lemat_potcars"]).split(",") if p in POTCAR_OFFSETS]
    if pots:
        remove[i] = "nonstandard_potcar"
        cause[i] = "; ".join(f"{p}: measured offset +{POTCAR_OFFSETS[p][0]:.3f} eV per atom of that element "
                             f"vs the standard POTCAR on {POTCAR_OFFSETS[p][1]} same-structure pairs" for p in pots) \
                   + "; MP deprecated the material and has no valid replacement calculation"
    else:
        flagged[i] = ("mp_material_deprecated_no_stated_reason",
                      "MP deprecated the material without a stated reason; standard POTCARs and U; kept")
for i, why in CAUSES.items():
    remove[i] = "calculation_defect"; cause[i] = why
for i, why in UNEXPLAINED.items():
    flagged[i] = ("unexplained_mlip_gap", "> 0.5 eV/atom below all five MLIPs; " + why + "; kept")
print("removals:", pd.Series(remove).value_counts().to_dict(), "flagged:", len(flagged))

def replacement_cause(i, r):
    """Why the upstream DFT record was replaced, from the MP provenance of row i."""
    t = ident.loc[i]
    if r.grp == "unidentified":
        return "upstream energy matches no GGA/GGA+U task MP still serves for this material"
    if r.grp == "live_other_task":
        return (f"upstream task {t.lemat_task} is live but {abs(r.old_vs_mlip):.2f} eV/atom from the MLIP median; "
                f"MP's task {r.new_task} on the same structure is within {abs(r.new_vs_mlip):.2f}")
    bits = [f"upstream task {t.lemat_task} ({t.lemat_calc_type}, batch {t.lemat_batch}) is deprecated by MP"
            if t.lemat_task_deprecated else f"upstream task {t.lemat_task} ({t.lemat_calc_type})"]
    if isinstance(t.mp_run, str) and t.lemat_run != t.mp_run:
        bits.append(f"it is {t.lemat_run} where MP's mixing scheme uses {t.mp_run}")
    old_p, new_p = set(str(t.lemat_potcars).split(",")), set(str(t.mp_potcars).split(","))
    if isinstance(t.mp_potcars, str) and old_p != new_p:
        bits.append("non-standard POTCAR " + ", ".join(sorted(old_p - new_p)))
    hu = lambda h: {k: v for k, v in (json.loads(h) or {}).items() if v} if isinstance(h, str) else {}
    if t.lemat_run == t.mp_run and hu(t.lemat_task_hubbards) != hu(t.mp_hubbards):
        bits.append(f"U values {hu(t.lemat_task_hubbards)} vs MP's {hu(t.mp_hubbards)}")
    return "; ".join(bits)


# --- audit table
aud = []
for i, p in rep.items():
    if i in remove:
        continue
    r = dec.loc[i]
    aud.append(dict(immutable_id=i, change=p["action"], reason=r.grp, cause=replacement_cause(i, r),
                    old_energy=float(x.at[i, "energy"]),
                    new_energy=p["energy"], nsites=int(x.at[i, "nsites"]), new_nsites=len(p["species"]), old_mp_task=r.lemat_task if r.grp != "unidentified" else None,
                    old_calc_type=r.lemat_calc_type if r.grp != "unidentified" else None, old_batch=r.lemat_batch if r.grp != "unidentified" else None,
                    new_mp_task=p["task_id"], new_calc_type=r.new_calc_type, geometry_match=r.geo,
                    forces_source=p.get("forces_source", "output") if p.get("forces") else None,
                    mlip_median_energy_per_atom=r.mlip_epa))
for i, why in remove.items():
    r = dec.loc[i] if i in dec.index else None
    aud.append(dict(immutable_id=i, change="removed", reason=why, cause=cause[i],
                    old_energy=float(x.at[i, "energy"]) if i in x.index else np.nan, new_energy=np.nan,
                    nsites=int(x.at[i, "nsites"]) if i in x.index else -1,
                    old_mp_task=None if r is None else r.lemat_task, old_calc_type=None if r is None else r.lemat_calc_type,
                    old_batch=None if r is None else r.lemat_batch, new_mp_task=None, new_calc_type=None, geometry_match=None,
                    mlip_median_energy_per_atom=float(x.at[i, "epa"] - x.at[i, "d_med"]) if i in x.index else np.nan))
for i, (why, note) in flagged.items():
    r = dec.loc[i] if i in dec.index else None
    aud.append(dict(immutable_id=i, change="kept_flagged", reason=why, cause=note,
                    old_energy=float(x.at[i, "energy"]), new_energy=float(x.at[i, "energy"]),
                    nsites=int(x.at[i, "nsites"]), old_mp_task=None if r is None else r.lemat_task,
                    old_calc_type=None if r is None else r.lemat_calc_type, old_batch=None if r is None else r.lemat_batch,
                    new_mp_task=None, new_calc_type=None, geometry_match=None,
                    mlip_median_energy_per_atom=float(x.at[i, "epa"] - x.at[i, "d_med"])))
for i, r in dec[dec.action == "keep_stale_mlip_consistent"].iterrows():
    if i not in remove:
        aud.append(dict(immutable_id=i, change="kept_flagged", reason=r.grp, cause="stale MP task with no valid replacement; agrees with the MLIPs; kept", old_energy=float(x.at[i, "energy"]),
                        new_energy=float(x.at[i, "energy"]), nsites=int(x.at[i, "nsites"]), old_mp_task=r.lemat_task,
                        old_calc_type=r.lemat_calc_type, old_batch=r.lemat_batch, new_mp_task=None, new_calc_type=None,
                        geometry_match=None, mlip_median_energy_per_atom=r.mlip_epa))
for i in recovered.index:
    if i in rep or i in remove or i not in x.index:
        continue
    aud.append(dict(immutable_id=i, change="forces_stress_added", reason="upstream_empty",
                    cause=("forces and stress_tensor were empty upstream; taken from the last ionic step of "
                           "the same MP task, matched to the upstream energy and geometry "
                           "(scripts/recover_mp_forces.py)"),
                    old_energy=float(x.at[i, "energy"]), new_energy=float(x.at[i, "energy"]),
                    nsites=int(x.at[i, "nsites"]), forces_source="last_ionic_step"))
aud = pd.DataFrame(aud)
# a flagged row that also gained forces/stress keeps one audit entry: the flag, with the forces noted
dup = aud.immutable_id.duplicated(keep=False)
both = set(aud.loc[dup & aud.change.eq("forces_stress_added"), "immutable_id"])
m = aud.immutable_id.isin(both) & aud.change.eq("kept_flagged")
aud.loc[m, "forces_source"] = "last_ionic_step"
aud.loc[m, "cause"] = aud.loc[m, "cause"] + "; forces and stress_tensor (empty upstream) added from the last ionic step of the same MP task"
aud = aud[~(aud.immutable_id.isin(both) & aud.change.eq("forces_stress_added"))].reset_index(drop=True)
assert not aud.immutable_id.duplicated().any()
geo = pd.read_parquet(W / "rs_geo.parquet").set_index("immutable_id")
m = aud.change.eq("replace_structure")
aud.loc[m, "geometry_match"] = aud.loc[m, "immutable_id"].map(geo.geometry_match)
aud["volume_change"] = aud.immutable_id.map(geo.dvol)
aud["old_energy_per_atom"] = aud.old_energy / aud.nsites
aud["new_nsites"] = aud.new_nsites.fillna(aud.nsites).astype(int)
aud["new_energy_per_atom"] = aud.new_energy / aud.new_nsites
(OUT / "audit").mkdir(parents=True, exist_ok=True)
aud.to_parquet(OUT / "audit" / "changes.parquet", index=False)
print(aud.groupby(["change", "reason"]).size())

# --- rewrite the shards
(OUT / "compatible_pbe").mkdir(parents=True, exist_ok=True)
files = sorted(SRC.glob("*.parquet"))
n_in = n_out = n_rep = n_recovered = 0
for fn in files:
    t = pq.read_table(fn)
    schema = t.schema
    ids = t.column("immutable_id").to_pylist()
    keep = np.array([i not in remove for i in ids])
    hit = [k for k, i in enumerate(ids) if i in rep and i not in remove]
    if hit:
        cols = {name: t.column(name).to_pylist() for name in
                ("energy", "forces", "stress_tensor", "magnetic_moments", "total_magnetization", "dos_ef",
                 "last_modified", "lattice_vectors", "cartesian_site_positions", "species_at_sites",
                 "nsites", "chemical_formula_descriptive")}
        for k in hit:
            p = rep[ids[k]]
            assert set(p["species"]) == set(cols["species_at_sites"][k]), ids[k]
            if len(p["species"]) != cols["nsites"][k]:
                counts = collections.Counter(p["species"])
                cols["nsites"][k] = len(p["species"])
                cols["chemical_formula_descriptive"][k] = " ".join(f"{e}{counts[e]}" for e in sorted(counts))
            cols["energy"][k] = p["energy"]
            cols["lattice_vectors"][k] = p["lattice"]
            cols["cartesian_site_positions"][k] = p["xyz"]
            cols["species_at_sites"][k] = p["species"]
            cols["forces"][k] = p["forces"] if p["forces"] is not None else []
            cols["stress_tensor"][k] = p["stress"] if p["stress"] is not None else []
            magmom = [0.0 if m is None else m for m in p["magmom"]] if p["magmom"] and any(m is not None for m in p["magmom"]) else []
            cols["magnetic_moments"][k] = magmom
            # LeMat's total_magnetization is the sum of the site moments (95% of control rows)
            cols["total_magnetization"][k] = float(sum(magmom)) if magmom else p["total_magnetization"]
            cols["dos_ef"][k] = p["efermi"]
            # LeMat's last_modified is the API task's last_updated (98.5% of control rows)
            cols["last_modified"][k] = lemat_time(task_time.get(p["task_id"]) or p["last_updated"])
        for name, values in cols.items():
            idx = schema.get_field_index(name)
            t = t.set_column(idx, schema.field(name), pa.array(values, type=schema.field(name).type))
        n_rep += len(hit)
    # untouched rows whose forces/stress upstream left empty: the recovered values
    rec_hit = [k for k, i in enumerate(ids) if i in recovered.index and i not in rep and i not in remove]
    if rec_hit:
        forces = t.column("forces").to_pylist(); stress = t.column("stress_tensor").to_pylist()
        for k in rec_hit:
            r = recovered.loc[ids[k]]
            if not forces[k]:
                forces[k] = [list(map(float, f)) for f in r.forces]
            if not stress[k]:
                stress[k] = [list(map(float, row)) for row in r.stress]
        for name, values in (("forces", forces), ("stress_tensor", stress)):
            idx = schema.get_field_index(name)
            t = t.set_column(idx, schema.field(name), pa.array(values, type=schema.field(name).type))
        n_recovered += len(rec_hit)
    t = t.filter(pa.array(keep))
    n_in += len(ids); n_out += t.num_rows
    pq.write_table(t.replace_schema_metadata(schema.metadata), OUT / "compatible_pbe" / fn.name,
                   compression="snappy", row_group_size=20000)
    print(fn.name, len(ids), "->", t.num_rows, flush=True)
print(f"rows {n_in} -> {n_out}; replaced {n_rep}; removed {n_in - n_out}; forces/stress recovered for {n_recovered} untouched rows")
