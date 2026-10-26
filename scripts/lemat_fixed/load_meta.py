import pyarrow.parquet as pq, glob, os, numpy as np, pandas as pd, sys
W=sys.argv[1]
S=os.path.expanduser("~/.cache/huggingface/hub/datasets--LeMaterial--LeMat-Bulk/snapshots/0dc17eea904b860ad7288141e9870f67f8e6bb2c/compatible_pbe")
dfs=[]
for f in sorted(glob.glob(S+"/*.parquet")):
    t=pq.read_table(f,columns=["immutable_id","nsites","elements","chemical_formula_reduced","energy","last_modified","functional","cross_compatibility","forces","stress_tensor","lattice_vectors"])
    d=t.select(["immutable_id","nsites","chemical_formula_reduced","energy","last_modified","functional","cross_compatibility"]).to_pandas()
    d["elements"]=[",".join(sorted(e)) for e in t.column("elements").to_pylist()]
    fr=t.column("forces").to_pylist()
    d["fmax"]=[float(np.linalg.norm(np.asarray(x),axis=1).max()) if x else np.nan for x in fr]
    st=t.column("stress_tensor").to_pylist()
    d["p_kbar"]=[float(-np.trace(np.asarray(x))/3) if x is not None and len(x)==3 else np.nan for x in st]
    lv=t.column("lattice_vectors").to_pylist()
    d["vol"]=[abs(np.linalg.det(np.asarray(x))) for x in lv]
    d["shard"]=os.path.basename(f)
    dfs.append(d); print(f, len(d), flush=True)
df=pd.concat(dfs,ignore_index=True)
df["src"]=df.immutable_id.str.extract(r"^([a-z]+)")[0]
df.to_parquet(W+"/lemat_meta.parquet")
print(df.src.value_counts()); print(df.functional.value_counts(), df.cross_compatibility.value_counts())
