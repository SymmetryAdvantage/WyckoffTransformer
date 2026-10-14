"""Shortest interatomic distance of each replacement structure vs LeMat's original."""
import sys, json, numpy as np, pandas as pd
from multiprocessing import Pool
from pymatgen.core import Structure, Lattice
W=sys.argv[1]
L=pd.read_parquet(W+"/lemat_review_rows.parquet",columns=["immutable_id","lattice_vectors","cartesian_site_positions","species_at_sites"]).drop_duplicates("immutable_id").set_index("immutable_id")
reps=[json.loads(l) for l in open(W+"/replacements.jsonl")]
reps=[p for p in reps if p["action"] in ("replace_structure","replace_energy")]
def mind(s):
    dm=s.distance_matrix+np.eye(len(s))*99
    return float(dm.min()) if len(s)>1 else float(min(s.lattice.abc))
def job(p):
    l=L.loc[p["immutable_id"]]
    a=Structure(Lattice(np.stack(l.lattice_vectors)),list(l.species_at_sites),np.stack(l.cartesian_site_positions),coords_are_cartesian=True)
    b=Structure(Lattice(p["lattice"]),p["species"],p["xyz"],coords_are_cartesian=True)
    return p["immutable_id"],p["action"],mind(a),mind(b)
with Pool(16) as pool: r=pd.DataFrame(pool.map(job,reps,chunksize=50),columns=["immutable_id","action","dmin_old","dmin_new"])
r["ratio"]=r.dmin_new/r.dmin_old
r.to_parquet(W+"/contacts.parquet")
for a,g in r.groupby("action"):
    print(a,len(g),"ratio<0.9:",(g.ratio<0.9).sum(),"<0.8:",(g.ratio<0.8).sum(),"new dmin<1.0A (non-H):",(g.dmin_new<1.0).sum(), "ratio q01:",round(g.ratio.quantile(0.01),3))
print(r[r.ratio<0.9].sort_values("ratio").head(15).to_string())
