import sys, json, re
from concurrent.futures import ThreadPoolExecutor
from mpclient import Rest
W=sys.argv[1]; r=Rest(W+"/api_cache")
def legacy(i):
    p,_,b=i.rpartition("-")
    if not re.fullmatch(r"[a-z]+",b): return i
    n=0
    for c in b: n=n*26+ord(c)-97
    return f"{p or 'mp'}-{n}"
core=json.load(open(W+"/core_lemat.json"))
want=set()
for d in core:
    for k,ct in (d.get("calc_types") or {}).items():
        if ct.split(" ",1)[0] in ("GGA","GGA+U") and ct.split(" ",1)[1] in ("Static","Structure Optimization"):
            want.add(legacy(k))
want=sorted(want); print("tasks",len(want),flush=True)
F="task_id,calc_type,task_type,batch_id,last_updated,completed_at,nsites,volume,output.energy,input.hubbards,input.potcar_spec,input.is_hubbard,tags"
chunks=[want[s:s+400] for s in range(0,len(want),400)]
def job(b): return r.get("/materials/tasks/",{"task_ids":",".join(b),"id_format":"legacy","_fields":F,"_limit":1000})["data"]
out=[]
with ThreadPoolExecutor(4) as ex:
    for n,res in enumerate(ex.map(job,chunks)):
        out+=res
        if n%100==0: print(n,len(chunks),len(out),flush=True)
for t in out:
    for p in (t.get("input") or {}).get("potcar_spec") or []: p.pop("summary_stats",None); p.pop("hash",None)
json.dump(out,open(W+"/tasks_meta.json","w")); print("done",len(out),"unique",len({t['task_id'] for t in out}))
