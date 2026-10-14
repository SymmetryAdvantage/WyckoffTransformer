"""LeMat MP ids that MP no longer serves as materials: find the material that now holds them as a task,
and fetch that parent's core doc and task metadata. Writes core_alias.json {lemat_id: parent_material_id}
and appends the parents to core_lemat.json and their tasks to tasks_meta.json."""
import sys, json, re, pandas as pd
from mpclient import Rest
W=sys.argv[1]; r=Rest(W+"/api_cache")
def legacy(i):
    p,_,b=i.rpartition("-")
    if not re.fullmatch(r"[a-z]+",b): return i
    n=0
    for c in b: n=n*26+ord(c)-97
    return f"{p or 'mp'}-{n}"
core=json.load(open(W+"/core_lemat.json")); have={d["material_id"] for d in core}
ids=sorted(set(pd.read_parquet(W+"/mp_vs_thermo.parquet").immutable_id)-have)
F="material_id,task_ids,task_types,calc_types,run_types,deprecated_tasks,deprecated,origins,nsites,volume,elements,formula_pretty"
alias={}; parents={}
for dep in ("false","true"):
    for s in range(0,len(ids),100):
        for d in r.get("/materials/core/",{"task_ids":",".join(ids[s:s+100]),"deprecated":dep,"id_format":"legacy","_fields":F,"_limit":1000})["data"]:
            for t in d["task_ids"]:
                if t in ids and t not in alias: alias[t]=d["material_id"]; parents[d["material_id"]]=d
print("aliases",len(alias),"of",len(ids),"parents",len(parents))
new=[d for m,d in parents.items() if m not in have]
json.dump(core+new,open(W+"/core_lemat.json","w")); json.dump(alias,open(W+"/core_alias.json","w"))
tm=json.load(open(W+"/tasks_meta.json")); known={t["task_id"] for t in tm}
want=sorted({legacy(k) for d in parents.values() for k,ct in (d.get("calc_types") or {}).items()
             if ct.split(" ")[0] in ("GGA","GGA+U") and ct.split(" ",1)[1] in ("Static","Structure Optimization")}-known)
FT="task_id,calc_type,task_type,batch_id,last_updated,completed_at,nsites,volume,output.energy,input.hubbards,input.potcar_spec,input.is_hubbard,tags"
for s in range(0,len(want),300):
    for t in r.get("/materials/tasks/",{"task_ids":",".join(want[s:s+300]),"id_format":"legacy","_fields":FT,"_limit":1000})["data"]:
        for p in (t.get("input") or {}).get("potcar_spec") or []: p.pop("summary_stats",None); p.pop("hash",None)
        tm.append(t)
json.dump(tm,open(W+"/tasks_meta.json","w")); print("added tasks",len(want))
