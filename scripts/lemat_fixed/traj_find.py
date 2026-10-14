import sys, pandas as pd, pyarrow.parquet as pq, pyarrow.compute as pc, pyarrow as pa
from concurrent.futures import ThreadPoolExecutor
from huggingface_hub import HfFileSystem
W=sys.argv[1]
a=pd.read_parquet("/home/kna/.local/share/wyformer/staging/lemat_bulk_fixed/audit/changes.parquet")
ids=pa.array(a[a.reason=="below_all_mlips"].immutable_id.tolist())
fs=HfFileSystem()
files=sorted(f for f in fs.ls("datasets/LeMaterial/LeMat-Traj/compatible_pbe",detail=False) if f.endswith(".parquet"))
cols=["immutable_id","id","relaxation_step","relaxation_number","energy","nsites","lattice_vectors","cartesian_site_positions","species_at_sites","forces","stress_tensor","last_modified"]
def job(f):
    for att in range(4):
        try:
            p=pq.ParquetFile(fs.open(f,block_size=8<<20))
            out=[]
            for rg in range(p.num_row_groups):
                t=p.read_row_group(rg,columns=["immutable_id"])
                m=pc.is_in(t.column("immutable_id"),value_set=ids)
                if pc.any(m).as_py():
                    full=p.read_row_group(rg,columns=cols); out.append(full.filter(m).to_pandas())
            return f,(pd.concat(out) if out else None)
        except Exception as e:
            err=e
    return f,err
res=[]
with ThreadPoolExecutor(16) as ex:
    for n,(f,r) in enumerate(ex.map(job,files)):
        if isinstance(r,Exception): print("FAIL",f,r,flush=True)
        elif r is not None: res.append(r); print(f.split("/")[-1], r.immutable_id.unique().tolist(), flush=True)
        if n%50==0: print("done",n,flush=True)
df=pd.concat(res) if res else pd.DataFrame()
df.to_parquet(W+"/traj_removed.parquet"); print("frames",len(df),"ids",df.immutable_id.nunique() if len(df) else 0)
