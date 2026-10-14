import sys, json, pandas as pd
from mpclient import Rest
W=sys.argv[1]; r=Rest(W+"/api_cache")
ids=sorted(pd.read_parquet(W+"/mp_vs_thermo.parquet").immutable_id.unique())
out={}
F="material_id,task_ids,task_types,calc_types,run_types,deprecated_tasks,deprecated,origins,nsites,volume,elements,formula_pretty"
for dep in ("false","true"):
    todo=[i for i in ids if i not in out]
    for s in range(0,len(todo),500):
        b=todo[s:s+500]
        d=r.get("/materials/core/",{"material_ids":",".join(b),"deprecated":dep,"id_format":"legacy","_fields":F,"_limit":1000})
        for x in d["data"]: out[x["material_id"]]=x
        if s%20000==0: print(dep,s,len(todo),len(out),flush=True)
print("resolved",len(out),"of",len(ids))
json.dump(list(out.values()), open(W+"/core_lemat.json","w"))
