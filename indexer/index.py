import argparse, time, sys
from pathlib import Path
from ingest import ensure, ingest_file, READERS

def scan_once(root="/data"):
    ensure()
    total = 0
    for p in Path(root).rglob("*"):
        if p.suffix.lower() in READERS:
            total += ingest_file(p)
    return total

def watch(root, interval=30):
    print(f"👀 watching {root} every {interval}s")
    seen = set()
    while True:
        for p in Path(root).rglob("*"):
            if p.suffix.lower() in READERS and str(p) not in seen:
                n = ingest_file(p)
                print(f"➕ {p} → {n}")
                seen.add(str(p))
        time.sleep(interval)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch", action="store_true")
    ap.add_argument("--root", default="/data")
    a = ap.parse_args()
    if a.watch:
        watch(a.root)
    else:
        print(f"Indexed {scan_once(a.root)} chunks")
