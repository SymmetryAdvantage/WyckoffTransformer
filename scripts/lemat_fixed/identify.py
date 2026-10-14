import json, os, re, sys, numpy as np, pandas as pd
W=sys.argv[1]
def legacy(i):
    p,_,b=i.rpartition("-")
    if not re.fullmatch(r"[a-z]+",b): return i
    n=0
    for c in b: n=n*26+ord(c)-97
    return f"{p or 'mp'}-{n}"
core={d["material_id"]:d for d in json.load(open(W+"/core_lemat.json"))}
tasks={t["task_id"]:t for t in json.load(open(W+"/tasks_meta.json"))}
m=pd.read_parquet(W+"/mp_vs_thermo.parquet")
# LeMat ids that MP has merged into another material: follow them to the parent material
alias=json.load(open(W+"/core_alias.json")) if os.path.exists(W+"/core_alias.json") else {}
thermo={d["material_id"]:d for d in json.load(open(W+"/thermo_gga.json"))}
for i,parent in alias.items():
    if i in core: continue
    core[i]=core[parent]
    th=thermo.get(parent)
    if th and th.get("origins"):
        m.loc[m.immutable_id==i,"thermo_task"]=th["origins"][0]["task_id"]
rows=[]
for r in m.itertuples(index=False):
    d=core.get(r.immutable_id); rec={"immutable_id":r.immutable_id}
    if d is None: rec["status"]="material_not_found"; rows.append(rec); continue
    dep={legacy(t) for t in d.get("deprecated_tasks") or []}
    rec["material_deprecated"]=d["deprecated"]
    cands=[]
    for k,ct in (d.get("calc_types") or {}).items():
        t=tasks.get(legacy(k))
        if t is None or (t.get("output") or {}).get("energy") is None: continue
        cands.append((legacy(k),t))
    rec["n_cand"]=len(cands)
    best=None
    for tid,t in cands:
        dE=t["output"]["energy"]-r.energy
        if best is None or abs(dE)<abs(best[1]): best=(tid,dE,t)
    if best is not None:
        tid,dE,t=best
        rec.update(lemat_task=tid, lemat_task_dE=dE, lemat_calc_type=t["calc_type"], lemat_batch=t.get("batch_id"),
                   lemat_task_deprecated=tid in dep, lemat_task_vol=t.get("volume"), lemat_task_nsites=t.get("nsites"),
                   lemat_task_hubbards=json.dumps(t["input"].get("hubbards")) if t.get("input") else None,
                   lemat_potcars=",".join(p["titel"] for p in (t.get("input") or {}).get("potcar_spec") or []))
    th=r.thermo_task
    if isinstance(th,str):
        th=legacy(th); t=tasks.get(th)
        rec.update(mp_task=th, mp_task_found=t is not None)
        if t is not None:
            rec.update(mp_calc_type=t["calc_type"], mp_energy=t["output"]["energy"], mp_nsites=t["nsites"], mp_task_vol=t.get("volume"),
                       mp_batch=t.get("batch_id"), mp_task_deprecated=th in dep,
                       mp_potcars=",".join(p["titel"] for p in (t.get("input") or {}).get("potcar_spec") or []),
                       mp_hubbards=json.dumps(t["input"].get("hubbards")) if t.get("input") else None)
    rows.append(rec)
f=pd.DataFrame(rows)
if "status" not in f: f["status"]=None
f=m[["immutable_id","energy","nsites","vol","epa","d_med","d_min","d_max","dft_hull","elements","chemical_formula_reduced","fmax","p_kbar","last_modified","uncorrected_energy_per_atom"]].merge(f,on="immutable_id")
f["identified"]=f.lemat_task_dE.abs()<1e-3
f["mp_epa"]=f.mp_energy/f.mp_nsites
f["dE_mp"]=f.epa-f.mp_epa
f["same_task"]=f.lemat_task==f.mp_task
f["lemat_run"]=f.lemat_calc_type.str.split(" ").str[0]; f["mp_run"]=f.mp_calc_type.str.split(" ").str[0]
f["dvol_mp"]=(f.vol/f.nsites)/(f.mp_task_vol/f.mp_nsites)-1
f["lemat_task_deprecated"]=f.lemat_task_deprecated.astype("boolean").fillna(False).astype(bool)
f["mlip_epa"]=f.epa-f.d_med
f["mp_vs_mlip"]=f.mp_epa-f.mlip_epa
rm=(f.lemat_run!=f.mp_run)&f.mp_run.notna()
f["grp"]=np.select([f.status.eq("material_not_found").values,(~f.identified).values,f.same_task.values,
                    (f.lemat_task_deprecated&rm).values,f.lemat_task_deprecated.values,rm.values],
                   ["not_found","unidentified","same_task","dep_runmismatch","deprecated","runmismatch"],"live_other_task")
f.to_parquet(W+"/identify.parquet")
pd.set_option("display.width",250)
print("identified:",f.identified.sum(),"of",len(f)); print(f.status.value_counts(dropna=False).to_dict() if "status" in f else "")
print(pd.crosstab(f.lemat_calc_type,f.lemat_task_deprecated,margins=True))
print(pd.crosstab([f.lemat_run,f.lemat_task_deprecated],f.mp_run,margins=True))
print("same task:",f.same_task.sum())
