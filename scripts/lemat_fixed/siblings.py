import sys, numpy as np, pandas as pd, pyarrow.parquet as pq, pyarrow.compute as pc, pyarrow as pa, glob, os
from pymatgen.core import Structure, Lattice, Composition
from pymatgen.analysis.structure_matcher import StructureMatcher
W=sys.argv[1]
a=pd.read_parquet("/home/kna/.local/share/wyformer/staging/lemat_bulk_fixed/audit/changes.parquet")
rem=a[a.reason=="below_all_mlips"].immutable_id.tolist()
j=pd.read_parquet(W+"/joined.parquet").drop_duplicates("immutable_id")
red=lambda f: Composition(f).reduced_formula
j["red"]=j.chemical_formula_reduced.map(red)
targets=j[j.immutable_id.isin(rem)]
sib=j[j.red.isin(set(targets.red))]
S=os.path.expanduser("~/.cache/huggingface/hub/datasets--LeMaterial--LeMat-Bulk/snapshots/0dc17eea904b860ad7288141e9870f67f8e6bb2c/compatible_pbe")
ids=pa.array(sib.immutable_id.tolist()); rows=[]
for fn in sorted(glob.glob(S+"/*.parquet")):
    t=pq.read_table(fn,columns=["immutable_id","lattice_vectors","cartesian_site_positions","species_at_sites"])
    rows.append(t.filter(pc.is_in(t.column("immutable_id"),value_set=ids)).to_pandas())
rows=pd.concat(rows).drop_duplicates("immutable_id").set_index("immutable_id")
st={i:Structure(Lattice(np.stack(r.lattice_vectors)),list(r.species_at_sites),np.stack(r.cartesian_site_positions),coords_are_cartesian=True) for i,r in rows.iterrows()}
sm=StructureMatcher(ltol=0.2,stol=0.3,angle_tol=5,scale=True)
J=j.set_index("immutable_id"); out=[]
for t in targets.itertuples():
    for s in sib[(sib.red==t.red)&(sib.immutable_id!=t.immutable_id)].itertuples():
        if sm.fit(st[t.immutable_id],st[s.immutable_id]):
            out.append(dict(removed=t.immutable_id,formula=t.red,sibling=s.immutable_id,dE_dft=t.epa-s.epa,dE_mlip=(t.epa-t.d_med)-(s.epa-s.d_med),
                            vol_ratio=(st[t.immutable_id].volume/len(st[t.immutable_id]))/(st[s.immutable_id].volume/len(st[s.immutable_id])), sib_vs_mlip=s.d_med))
o=pd.DataFrame(out); o.to_parquet(W+"/siblings.parquet")
pd.set_option("display.width",250)
print(o.round(3).to_string())
print("removed rows with a matching sibling:",o.removed.nunique(),"of",len(targets))
