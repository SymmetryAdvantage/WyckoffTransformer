import hashlib, json, time, os, requests
from pathlib import Path
API="https://api.materialsproject.org"
class Rest:
    def __init__(self, cache, env="/home/kna/WyckoffTransformer/.env"):
        key=None
        for line in Path(env).read_text().splitlines():
            if line.startswith("MP_API_KEY="): key=line.split("=",1)[1].strip().strip("'\"")
        self.s=requests.Session(); self.s.headers["X-API-KEY"]=key
        self.cache=Path(cache); self.cache.mkdir(parents=True, exist_ok=True)
    def get(self, endpoint, params, use_cache=True):
        tag=hashlib.sha1(json.dumps([endpoint,params],sort_keys=True).encode()).hexdigest()
        p=self.cache/f"{tag}.json"
        if use_cache and p.exists(): return json.loads(p.read_text())
        for a in range(8):
            try:
                r=self.s.get(API+endpoint, params=params, timeout=600)
            except requests.RequestException as e:
                time.sleep(2**a); continue
            if r.status_code==429 or r.status_code>=500: time.sleep(2**a); continue
            if r.status_code>=400: raise RuntimeError(f"{r.status_code} {r.text[:500]}")
            d=r.json(); tmp=p.with_suffix(".tmp"); tmp.write_text(json.dumps(d)); tmp.rename(p); return d
        raise RuntimeError("failed "+endpoint)
