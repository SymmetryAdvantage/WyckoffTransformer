#!/usr/bin/env bash
# Fetch a large remote checkpoint into the `cached_path` cache using parallel
# byte-range requests, and verify it against the server's ETag.
#
#     scripts/platforms/aspire2a/prefetch_cached_path.sh <url> [streams]
#
# WHY THIS EXISTS
#
# The ORB checkpoints live in an S3 bucket in us-west-1 that serves ASPIRE 2A at
# ~30 kB/s *per connection* -- measured 2026-09-08 from asp2a-gpu002. The
# throttle is per-connection, not per-host, so 16 range requests in parallel get
# ~400 kB/s and the 102 MB orb-v3 checkpoint lands in ~8 minutes instead of ~53.
# Tunnelling through a login node does not help: the bucket is just as slow from
# there (24 kB/s through socks5h://asp2a-login-nus02:1081, against 4.5 MB/s to
# HuggingFace over the same proxy), so the login node is not a faster egress --
# it is the same egress.
#
# Without this, `--stage relax` looks like a deadlock rather than a download:
# every --workers-per-device worker blocks on the one `cached_path` file lock
# while the first of them crawls through the checkpoint.
#
# WHAT IT WRITES
#
# `cached_path` keys its cache as sha256(url).sha256(etag), with a JSON meta file
# alongside. This writes both, so the next `cached_path(url)` is a cache hit and
# no job downloads anything.
#
# Run it from a login node or a dev node -- anywhere with network. It is
# idempotent: an already-cached, ETag-verified file is left alone.
set -euo pipefail

URL=${1:?usage: prefetch_cached_path.sh <url> [streams]}
STREAMS=${2:-16}
CACHE_DIR=${CACHED_PATH_CACHE_ROOT:-$HOME/.cache/cached_path}
WORK=$(mktemp -d "${TMPDIR:-/tmp}/prefetch.XXXXXX")
trap 'rm -rf "$WORK"' EXIT

command -v curl >/dev/null || { echo "curl not found" >&2; exit 1; }
mkdir -p "$CACHE_DIR"

# --- ask the server for size and ETag ---------------------------------------
head=$(curl -sIL --retry 3 "$URL")
SIZE=$(awk 'BEGIN{IGNORECASE=1} /^content-length:/ {gsub(/\r/,"",$2); n=$2} END{print n}' <<<"$head")
ETAG=$(awk 'BEGIN{IGNORECASE=1} /^etag:/ {gsub(/\r/,"",$2); e=$2} END{print e}' <<<"$head")
[ -n "$SIZE" ] && [ -n "$ETAG" ] || { echo "could not read Content-Length/ETag from $URL" >&2; exit 1; }
echo "url    : $URL"
echo "size   : $SIZE bytes"
echo "etag   : $ETAG"

name=$(python3 -c 'import hashlib,sys;print(hashlib.sha256(sys.argv[1].encode()).hexdigest())' "$URL")
etag_h=$(python3 -c 'import hashlib,sys;print(hashlib.sha256(sys.argv[1].encode()).hexdigest())' "$ETAG")
DEST="$CACHE_DIR/$name.$etag_h"

if [ -f "$DEST" ] && [ "$(stat -c %s "$DEST")" = "$SIZE" ]; then
    echo "already cached: $DEST"
    exit 0
fi

# --- parallel byte ranges ----------------------------------------------------
CHUNK=$(( (SIZE + STREAMS - 1) / STREAMS ))
echo "fetching with $STREAMS streams (chunk $CHUNK)"
start=$(date +%s)
for i in $(seq 0 $((STREAMS-1))); do
    s=$((i*CHUNK)); e=$((s+CHUNK-1)); [ $e -ge $SIZE ] && e=$((SIZE-1))
    [ $s -ge $SIZE ] && break
    curl -sL --retry 5 -o "$WORK/part.$(printf %04d "$i")" -r "$s-$e" "$URL" &
done
wait
echo "elapsed: $(( $(date +%s) - start ))s"

# Re-fetch any stream that came back short rather than trusting the byte total:
# a truncated part and a complete one look identical in an aggregate size check.
for i in $(seq 0 $((STREAMS-1))); do
    s=$((i*CHUNK)); e=$((s+CHUNK-1)); [ $e -ge $SIZE ] && e=$((SIZE-1))
    [ $s -ge $SIZE ] && break
    f="$WORK/part.$(printf %04d "$i")"; want=$((e-s+1))
    have=$(stat -c %s "$f" 2>/dev/null || echo 0)
    if [ "$have" != "$want" ]; then
        echo "part $i short ($have/$want) -> refetching"
        curl -sL --retry 5 -o "$f" -r "$s-$e" "$URL"
        have=$(stat -c %s "$f")
        [ "$have" = "$want" ] || { echo "part $i still short" >&2; exit 1; }
    fi
done

cat "$WORK"/part.* > "$WORK/full"
got=$(stat -c %s "$WORK/full")
[ "$got" = "$SIZE" ] || { echo "assembled size $got != $SIZE" >&2; exit 1; }

# --- verify against the ETag -------------------------------------------------
# A plain ETag is the MD5. An "<md5>-<n>" ETag is S3 multipart: the MD5 of the
# concatenated per-part MD5s. The part size is not advertised, so infer it from
# the part count -- S3's uploader uses a power-of-two MiB size.
python3 - "$WORK/full" "$ETAG" <<'PY'
import hashlib, sys
path, etag = sys.argv[1], sys.argv[2].strip('"')
data_md5 = lambda b: hashlib.md5(b).digest()
if '-' not in etag:
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    if h.hexdigest() != etag: sys.exit(f"ETag MISMATCH: {h.hexdigest()} != {etag}")
    print("ETag OK (single-part MD5)"); raise SystemExit
want_md5, want_n = etag.split('-'); want_n = int(want_n)
size = __import__('os').path.getsize(path)
for mib in (5, 8, 16, 32, 64, 128, 256, 512):
    part = mib * 1024 * 1024
    if -(-size // part) != want_n: continue
    digests = []
    with open(path, 'rb') as f:
        while True:
            b = f.read(part)
            if not b: break
            digests.append(data_md5(b))
    got = hashlib.md5(b''.join(digests)).hexdigest()
    if got == want_md5:
        print(f"ETag OK (multipart, {want_n} x {mib} MiB)"); raise SystemExit
sys.exit(f"ETag MISMATCH: no part size reproduces {etag}")
PY

# --- install ------------------------------------------------------------------
cp "$WORK/full" "$DEST.incoming"
mv -f "$DEST.incoming" "$DEST"
python3 - "$URL" "$DEST" "$ETAG" <<'PY'
import json, os, sys, time
url, dest, etag = sys.argv[1:4]
json.dump(dict(resource=url, cached_path=dest, creation_time=time.time(),
               size=os.path.getsize(dest), etag=etag, extraction_dir=False),
          open(dest + ".json", "w"))
PY
echo "installed: $DEST"
