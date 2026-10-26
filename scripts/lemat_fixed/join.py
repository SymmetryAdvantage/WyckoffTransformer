import pandas as pd, glob, os, sys
W=sys.argv[1]
H=os.path.expanduser("~/.cache/huggingface/hub/datasets--LeMaterial--LeMat-Bulk-MLIP-Hull-All/snapshots/b7bfc3c6ddfdef49bd03875fb5d3d6e1b5f80d53/data")
h=pd.concat([pd.read_parquet(f,columns=[c for c in ["immutable_id","nsites","dft_hull","true_energy","orb_direct_20_energy","orb_conserv_inf_energy","uma_energy","mace_mp_energy","mace_omat_energy"]]) for f in sorted(glob.glob(H+"/*.parquet"))])
m=pd.read_parquet(W+"/lemat_meta.parquet")
print(len(h), h.immutable_id.nunique(), len(m), m.immutable_id.nunique())
j=m.merge(h.drop(columns="nsites"),on="immutable_id",how="left")
print("missing in hull-all", j.true_energy.isna().sum(), "energy mismatch", (abs(j.energy-j.true_energy)>1e-6).sum())
ml=["orb_direct_20_energy","orb_conserv_inf_energy","uma_energy","mace_mp_energy","mace_omat_energy"]
for c in ml: j[c.replace("_energy","_d")]=(j.energy-j[c])/j.nsites
D=j[[c.replace("_energy","_d") for c in ml]]
j["d_max"]=D.max(axis=1); j["d_min"]=D.min(axis=1); j["d_med"]=D.median(axis=1)
j["epa"]=j.energy/j.nsites
j.to_parquet(W+"/joined.parquet")
for s in ["mp","agm","oqmd"]:
    x=j[j.src==s]
    print(s, len(x), "below all by>0.2:",(x.d_max<-0.2).sum(),"above all by>0.2:",(x.d_min>0.2).sum(), "Yb rows:", x.elements.str.contains(r"(^|,)Yb(,|$)").sum())
for i in ["mp-1400697","mp-725221","mp-554674","mp-1079094","mp-636148"]:
    print(j[j.immutable_id==i][["immutable_id","chemical_formula_reduced","epa","d_med","d_min","d_max","fmax","last_modified"]].to_string(header=False))
