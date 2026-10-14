"""Full upstream rows for every MP row under review."""
import sys, os, glob, pandas as pd, pyarrow as pa, pyarrow.parquet as pq, pyarrow.compute as pc
W=sys.argv[1]
S=os.path.expanduser("~/.cache/huggingface/hub/datasets--LeMaterial--LeMat-Bulk/snapshots/0dc17eea904b860ad7288141e9870f67f8e6bb2c/compatible_pbe")
f=pd.read_parquet(W+"/identify.parquet"); ids=pa.array(f[f.review!=""].immutable_id.tolist())
t=pa.concat_tables([pq.read_table(fn).filter(pc.is_in(pq.read_table(fn,columns=["immutable_id"]).column("immutable_id"),value_set=ids)) for fn in sorted(glob.glob(S+"/*.parquet"))])
pq.write_table(t,W+"/lemat_review_rows.parquet"); print("rows",t.num_rows)
