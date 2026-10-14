"""Select candidate MP tasks for the rows under review and extract their full outputs from S3."""
import sys, json, re, gzip, time, numpy as np, pandas as pd, requests
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
W=Path(sys.argv[1]); OUT=W/"s3"; OUT.mkdir(exist_ok=True)
MAN=Path("/home/kna/.local/share/wyformer/cache/mp_forces_recovery")
S3="https://materialsproject-parsed.s3.amazonaws.com/"
TASK_ID_RE=re.compile(rb'"task_id":\s*"([a-z]+-\d+)"')
def legacy(i):
    p,_,b=i.rpartition("-")
    if not re.fullmatch(r"[a-z]+",b): return i
    n=0
    for c in b: n=n*26+ord(c)-97
    return f"{p or 'mp'}-{n}"

def select():
    f=pd.read_parquet(W/"identify.parquet")
    core={d["material_id"]:d for d in json.load(open(W/"core_lemat.json"))}
    alias=json.load(open(W/"core_alias.json")) if (W/"core_alias.json").exists() else {}
    for i,parent in alias.items(): core.setdefault(i,core[parent])
    tasks={t["task_id"]:t for t in json.load(open(W/"tasks_meta.json"))}
    stale=f.grp.isin(["deprecated","dep_runmismatch","unidentified","runmismatch"])
    livebig=(f.grp=="live_other_task")&(f.dE_mp.abs()>0.05)
    ctrl1=f[f.grp=="same_task"].sample(300,random_state=0).index
    ctrl2=f[(f.grp=="live_other_task")&(f.dE_mp.abs()<=0.05)].sample(300,random_state=0).index
    f["review"]=np.where(stale,"stale",np.where(livebig,"live_big",""))
    f.loc[ctrl1,"review"]="control_same"; f.loc[ctrl2,"review"]="control_live"
    r=f[f.review!=""]
    pairs=[]
    for row in r.itertuples():
        d=core.get(row.immutable_id)
        if d is None: continue
        dep={legacy(t) for t in d.get("deprecated_tasks") or []}
        for k,ct in (d.get("calc_types") or {}).items():
            tid=legacy(k); t=tasks.get(tid)
            if t is None: continue
            run,_,typ=ct.partition(" ")
            if run not in ("GGA","GGA+U") or typ not in ("Static","Structure Optimization"): continue
            is_lemat = tid==row.lemat_task
            is_mp = tid==row.mp_task
            if is_lemat or is_mp or (tid not in dep and t.get("nsites")==row.nsites and (not isinstance(row.mp_run,str) or run==row.mp_run)):
                pairs.append((row.immutable_id,tid,tid in dep,is_lemat,is_mp,ct))
    p=pd.DataFrame(pairs,columns=["immutable_id","task_id","task_deprecated","is_lemat_task","is_mp_task","calc_type"])
    f.to_parquet(W/"identify.parquet"); p.to_parquet(W/"candidates.parquet")
    print(f.review.value_counts().to_dict(), "pairs",len(p),"tasks",p.task_id.nunique(), flush=True)
    return p

def extract(doc):
    o=doc.get("output") or {}; s=o.get("structure") or {}
    c=(doc.get("calcs_reversed") or [{}])[0]; co=c.get("output") or {}; oc=co.get("outcar") or {}
    inp=doc.get("input") or {}
    lat=s["lattice"]["matrix"]
    return {"task_id":doc["task_id"],"energy":o.get("energy"),"lattice":lat,
            "species":[x["species"][0]["element"] for x in s["sites"]],
            "xyz":[x["xyz"] if "xyz" in x else list(np.dot(x["abc"],lat)) for x in s["sites"]],
            "magmom":[(x.get("properties") or {}).get("magmom") for x in s["sites"]],
            "forces":o.get("forces"),"stress":o.get("stress"),
            "total_magnetization":oc.get("total_magnetization"),"efermi":co.get("efermi"),
            "potcars":[p.get("titel") for p in (inp.get("potcar_spec") or [])],
            "hubbards":inp.get("hubbards"),"last_updated":str(doc.get("last_updated")),
            "calc_type":doc.get("calc_type"),"nions_steps":len(co.get("ionic_steps") or [])}

def scan(key, needed):
    wanted=set(needed); got={}
    for a in range(4):
        try:
            with requests.get(S3+key,stream=True,timeout=900) as resp:
                resp.raise_for_status()
                with gzip.GzipFile(fileobj=resp.raw) as fh:
                    for line in fh:
                        ids={m.decode() for m in TASK_ID_RE.findall(line[:4000])} or {m.decode() for m in TASK_ID_RE.findall(line)}
                        if not ids & wanted: continue
                        d=json.loads(line); t=d.get("task_id")
                        if t in wanted:
                            try: got[t]=extract(d)
                            except Exception as e: got[t]={"task_id":t,"error":repr(e)}
                            wanted.discard(t)
                            if not wanted: break
            return key, got, sorted(wanted), True
        except Exception as e:
            time.sleep(5*(a+1))
    return key, got, sorted(wanted), False

def fetch(p, workers):
    need=set(p.task_id)
    done=set()
    for fn in OUT.glob("*.jsonl"):
        for line in open(fn): done.add(json.loads(line)["task_id"])
    need-=done
    a2=pd.read_parquet(MAN/"manifest_atomate2.parquet",columns=["task_id","sourced_from_path"])
    a2=a2[a2.task_id.isin(need)]
    leg=pd.read_parquet(MAN/"manifest_legacy.parquet",columns=["task_id","nelements","spacegroup","dt"])
    leg=leg[leg.task_id.isin(need-set(a2.task_id))]
    keys={}
    for t,k in zip(a2.task_id,a2.sourced_from_path.str.replace("s3://materialsproject-parsed/","",regex=False)): keys.setdefault(k,[]).append(t)
    for r in leg.itertuples(): keys.setdefault(f"tasks-legacy/tasks/nelements={r.nelements}/output_spacegroup_number={r.spacegroup}/dt={r.dt}.jsonl.gz",[]).append(r.task_id)
    nowhere=need-set(a2.task_id)-set(leg.task_id)
    print("to fetch",len(need),"files",len(keys),"in no manifest",len(nowhere),flush=True)
    out=open(OUT/f"part_{int(time.time())}.jsonl","w"); miss=open(OUT/"missing.txt","a")
    with ProcessPoolExecutor(workers) as ex:
        futs=[ex.submit(scan,k,v) for k,v in keys.items()]
        for i,fu in enumerate(as_completed(futs),1):
            k,got,left,ok=fu.result()
            for d in got.values(): out.write(json.dumps(d)+"\n")
            for t in left: miss.write(f"{t}\t{k}\t{ok}\n")
            if i%100==0 or i==len(futs): out.flush(); print(f"files {i}/{len(futs)}",flush=True)
    for t in nowhere: miss.write(f"{t}\tNO_MANIFEST\tTrue\n")
    out.close(); miss.close()

if __name__=="__main__":
    p=select() if not (W/"candidates.parquet").exists() else pd.read_parquet(W/"candidates.parquet")
    fetch(p, int(sys.argv[2]) if len(sys.argv)>2 else 12)
    print("DONE",flush=True)
