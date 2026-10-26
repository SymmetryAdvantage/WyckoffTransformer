import sys, json
from mpclient import Rest
W=sys.argv[1]; r=Rest(W+"/api_cache")
def pull(endpoint, params, name):
    out=[]; skip=0
    while True:
        d=r.get(endpoint, {**params, "_limit":1000, "_skip":skip})
        out+=d["data"]; total=d["meta"]["total_doc"]
        skip+=1000
        if skip%20000==0: print(name, skip, total, flush=True)
        if skip>=total or not d["data"]: break
    json.dump(out, open(f"{W}/{name}.json","w")); print(name, len(out), flush=True)
pull("/materials/thermo/", {"thermo_types":"GGA_GGA+U","id_format":"legacy","_fields":"material_id,uncorrected_energy_per_atom,energy_per_atom,nsites,volume,origins,deprecated,energy_above_hull,elements,formula_pretty"}, "thermo_gga")
for dep in ("false","true"):
    pull("/materials/core/", {"deprecated":dep,"id_format":"legacy","_fields":"material_id,task_ids,task_types,calc_types,run_types,deprecated_tasks,deprecated,origins,nsites,volume,elements,formula_pretty"}, f"core_dep{dep}")
