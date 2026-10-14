import sys, json, numpy as np, pandas as pd
from multiprocessing import Pool
from pymatgen.core import Structure, Lattice
from pymatgen.analysis.structure_matcher import StructureMatcher
W=sys.argv[1]
L=pd.read_parquet(W+"/lemat_review_rows.parquet",columns=["immutable_id","lattice_vectors","cartesian_site_positions","species_at_sites"]).drop_duplicates("immutable_id").set_index("immutable_id")
reps=[json.loads(l) for l in open(W+"/replacements.jsonl")]
reps=[p for p in reps if p["action"]=="replace_structure"]
sm=StructureMatcher(ltol=0.02,stol=0.05,angle_tol=1,primitive_cell=False,scale=False,attempt_supercell=True)
def job(p):
    l=L.loc[p["immutable_id"]]
    a=Structure(Lattice(np.stack(l.lattice_vectors)),list(l.species_at_sites),np.stack(l.cartesian_site_positions),coords_are_cartesian=True)
    b=Structure(Lattice(p["lattice"]),p["species"],p["xyz"],coords_are_cartesian=True)
    try: m=sm.fit(a,b)
    except Exception: m=False
    return p["immutable_id"], "same_crystal_new_cell" if m else "re_relaxed", b.volume/len(b)/(a.volume/len(a))-1
with Pool(16) as pool: res=pool.map(job,reps,chunksize=20)
r=pd.DataFrame(res,columns=["immutable_id","geometry_match","dvol"]); r.to_parquet(W+"/rs_geo.parquet")
print(r.geometry_match.value_counts().to_dict(), r.dvol.abs().quantile([.5,.9,.99,1]).round(4).to_dict())
