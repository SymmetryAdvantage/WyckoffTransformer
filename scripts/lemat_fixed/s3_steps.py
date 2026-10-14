"""Last-ionic-step forces/stress for replacement tasks whose output.forces/stress are empty.

MP's task documents often leave output.forces/stress empty for relaxations; the values are
in calcs_reversed[0].output.ionic_steps[-1]. Kept only if that step is the output: same
species order, lattice and positions to 1e-4 A, and an energy term equal to the output
energy to 1e-5 eV (as in scripts/recover_mp_forces.py)."""
import sys, json, re, gzip, time, numpy as np, pandas as pd, requests
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
W=Path(sys.argv[1]); OUT=W/"s3_steps"; OUT.mkdir(exist_ok=True)
MAN=Path("/home/kna/.local/share/wyformer/cache/mp_forces_recovery")
S3="https://materialsproject-parsed.s3.amazonaws.com/"
TASK_ID_RE=re.compile(rb'"task_id":\s*"([a-z]+-\d+)"')
TOL_X, TOL_E = 1e-4, 1e-5

def xyz(s):
    lat=np.asarray(s["lattice"]["matrix"],float)
    return lat,[x["species"][0]["element"] for x in s["sites"]],np.array([x["xyz"] if "xyz" in x else np.dot(x["abc"],lat) for x in s["sites"]],float)

def extract(doc):
    o=doc.get("output") or {}; c=(doc.get("calcs_reversed") or [{}])[0]
    steps=(c.get("output") or {}).get("ionic_steps") or []
    rec={"task_id":doc["task_id"],"ok":False}
    if not steps or not o.get("structure"): rec["why"]="no ionic steps"; return rec
    st=steps[-1]
    if not st.get("forces") or not st.get("stress") or not st.get("structure"): rec["why"]="step lacks forces/stress"; return rec
    la,sa,xa=xyz(o["structure"]); lb,sb,xb=xyz(st["structure"])
    if sa!=sb: rec["why"]="species order differs"; return rec
    fr=(xb-xa)@np.linalg.inv(la); fr-=np.round(fr)
    dpos=float(np.linalg.norm(fr@la,axis=1).max()); dlat=float(np.abs(lb-la).max())
    dE=min(abs(st[k]-o["energy"]) for k in ("e_fr_energy","e_wo_entrp","e_0_energy") if isinstance(st.get(k),(int,float)))
    rec.update(dpos=dpos,dlat=dlat,dE=dE)
    if dpos>TOL_X or dlat>TOL_X or dE>TOL_E: rec["why"]="last step is not the output"; return rec
    rec.update(ok=True,forces=st["forces"],stress=st["stress"])
    return rec

def scan(key, needed):
    wanted=set(needed); got=[]
    for a in range(4):
        try:
            with requests.get(S3+key,stream=True,timeout=900) as resp:
                resp.raise_for_status()
                with gzip.GzipFile(fileobj=resp.raw) as fh:
                    for line in fh:
                        if not {m.decode() for m in TASK_ID_RE.findall(line)} & wanted: continue
                        d=json.loads(line)
                        if d.get("task_id") in wanted:
                            got.append(extract(d)); wanted.discard(d["task_id"])
                            if not wanted: break
            return got, sorted(wanted)
        except Exception: time.sleep(5*(a+1))
    return got, sorted(wanted)

if __name__=="__main__":
    need=set(open(W/"need_steps.txt").read().split())
    a2=pd.read_parquet(MAN/"manifest_atomate2.parquet",columns=["task_id","sourced_from_path"]); a2=a2[a2.task_id.isin(need)]
    leg=pd.read_parquet(MAN/"manifest_legacy.parquet",columns=["task_id","nelements","spacegroup","dt"]); leg=leg[leg.task_id.isin(need)]
    keys={}
    # same bucket preference as s3_extract.py: atomate2 first, legacy otherwise
    for t,k in zip(a2.task_id,a2.sourced_from_path.str.replace("s3://materialsproject-parsed/","",regex=False)): keys.setdefault(k,[]).append(t)
    for r in leg[~leg.task_id.isin(set(a2.task_id))].itertuples():
        keys.setdefault(f"tasks-legacy/tasks/nelements={r.nelements}/output_spacegroup_number={r.spacegroup}/dt={r.dt}.jsonl.gz",[]).append(r.task_id)
    print("tasks",len(need),"files",len(keys),flush=True)
    out=open(OUT/"steps.jsonl","w"); missing=[]
    with ProcessPoolExecutor(12) as ex:
        futs=[ex.submit(scan,k,v) for k,v in keys.items()]
        for i,fu in enumerate(as_completed(futs),1):
            got,left=fu.result(); missing+=left
            for g in got: out.write(json.dumps(g)+"\n")
            if i%100==0: print("files",i,len(futs),flush=True)
    out.close(); print("DONE missing",len(missing),flush=True)
