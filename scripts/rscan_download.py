#!/usr/bin/env python3
"""
Download the 3RScan files the adapter needs (src/evaluation/rscan.py) for
the environments of the given benchmark splits, from the URL scheme of the
official download script (gist WaldJohannaU/55f5e35992ea91157b789b15eac4d432).
Use requires agreeing to the 3RScan Terms of Use
(http://campar.in.tum.de/public_datasets/3RScan/3RScanTOU.pdf).

Per scan: sequence.zip (unpacked to <scan>/sequence/, then removed),
labels.instances.annotated.v2.ply, semseg.v2.json. Rerunnable: files whose
size matches the server are skipped; downloads go to a temporary name and
are renamed when complete.

  python scripts/rscan_download.py --sizes-only
  python scripts/rscan_download.py --splits rscan_train rscan_val rscan_test
"""

import argparse
import json
import os
import shutil
import sys
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))
import conftest  # noqa: E402,F401

from hiper.evaluation import paths as hpaths  # noqa: E402
from hiper.evaluation import rscan  # noqa: E402

DATA_URL = "https://campar.in.tum.de/public_datasets/3RScan/Dataset"
FILES = ("labels.instances.annotated.v2.ply", "semseg.v2.json",
         "sequence.zip")


def scan_ids(meta, splits):
    index = rscan.load_index(meta)
    out = []
    for env_id, env in sorted(index.items()):
        if rscan.assign_split(env_id, env["type"]) in splits:
            out += [s for s, _, _ in rscan.env_scans(env)]
    return out


def remote_size(url, tries=3):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, method="HEAD")
            with urllib.request.urlopen(req, timeout=60) as r:
                return int(r.headers["Content-Length"])
        except Exception:                               # noqa: BLE001
            if i == tries - 1:
                raise
            time.sleep(5 * (i + 1))


def done_marker(scan_dir):
    return os.path.join(scan_dir, "sequence", ".complete")


def fetch(scan_id, root, keep_zip=False, tries=3):
    """Download (and unpack) one scan; returns (scan id, bytes, note)."""
    d = os.path.join(root, scan_id)
    os.makedirs(d, exist_ok=True)
    got = 0
    for name in FILES:
        if name == "sequence.zip" and os.path.exists(done_marker(d)):
            continue
        url = f"{DATA_URL}/{scan_id}/{name}"
        out = os.path.join(d, name)
        size = remote_size(url)
        if os.path.exists(out) and os.path.getsize(out) == size:
            pass
        else:
            for i in range(tries):
                tmp = out + ".part"
                try:
                    with urllib.request.urlopen(url, timeout=120) as r, \
                            open(tmp, "wb") as f:
                        shutil.copyfileobj(r, f, length=1 << 22)
                    if os.path.getsize(tmp) != size:
                        raise IOError(f"size {os.path.getsize(tmp)} != {size}")
                    os.replace(tmp, out)
                    got += size
                    break
                except Exception:                       # noqa: BLE001
                    if os.path.exists(tmp):
                        os.remove(tmp)
                    if i == tries - 1:
                        raise
                    time.sleep(10 * (i + 1))
        if name == "sequence.zip":
            seq = os.path.join(d, "sequence")
            with zipfile.ZipFile(out) as z:
                members = z.namelist()
                z.extractall(d if any(m.startswith("sequence/")
                                      for m in members) else seq)
            info = rscan.read_info(os.path.join(seq, "_info.txt"))
            n = len([f for f in os.listdir(seq) if f.endswith(".color.jpg")])
            with open(os.path.join(seq, "_info.txt")) as f:
                listed = next((int(l.split("=")[1]) for l in f
                               if l.startswith("m_frames.size")), n)
            if n != listed:
                return scan_id, got, f"frames {n} != _info.txt {listed}"
            open(done_marker(d), "w").write(f"{n} frames, color "
                                            f"{info['color_size']}\n")
            if not keep_zip:
                os.remove(out)
    return scan_id, got, "ok"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", default=str(hpaths.RSCAN_SCANS))
    ap.add_argument("--meta", default=str(hpaths.RSCAN_META))
    ap.add_argument("--splits", nargs="+",
                    default=["rscan_train", "rscan_val", "rscan_test"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--sizes-only", action="store_true")
    ap.add_argument("--keep-zip", action="store_true")
    ap.add_argument("--log", default=None, help="JSON lines progress log")
    args = ap.parse_args()
    ids = scan_ids(args.meta, set(args.splits))
    print(f"{len(ids)} scans in {args.splits}", flush=True)
    if args.sizes_only:
        tot = {n: 0 for n in FILES}
        with ThreadPoolExecutor(16) as ex:
            futs = {ex.submit(remote_size, f"{DATA_URL}/{s}/{n}"): n
                    for s in ids for n in FILES}
            for fu in as_completed(futs):
                tot[futs[fu]] += fu.result()
        for n, b in tot.items():
            print(f"{n:36s} {b / 1e9:8.1f} GB")
        print(f"{'total':36s} {sum(tot.values()) / 1e9:8.1f} GB")
        return
    log = open(args.log, "a") if args.log else None
    t0, total, bad = time.time(), 0, []
    with ThreadPoolExecutor(args.workers) as ex:
        futs = [ex.submit(fetch, s, args.root, args.keep_zip) for s in ids]
        for i, fu in enumerate(as_completed(futs), 1):
            try:
                sid, got, note = fu.result()
            except Exception as e:                      # noqa: BLE001
                sid, got, note = "?", 0, f"ERROR {type(e).__name__}: {e}"
            total += got
            if note != "ok":
                bad.append((sid, note))
            rec = {"i": i, "n": len(ids), "scan": sid, "note": note,
                   "gb": round(total / 1e9, 2),
                   "min": round((time.time() - t0) / 60, 1)}
            print(json.dumps(rec), flush=True)
            if log:
                log.write(json.dumps(rec) + "\n")
                log.flush()
    print(f"done: {len(ids)} scans, {total / 1e9:.1f} GB new, "
          f"{len(bad)} problems: {bad[:10]}", flush=True)


if __name__ == "__main__":
    main()
