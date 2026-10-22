"""Lock данных: sha256 золотого набора, фикстур и эталона UI → data/data.lock.json.

    python scripts/data_lock.py            # пересчитать lock
    python scripts/data_lock.py --check    # CI: данные не менялись без обновления lock
    python scripts/data_lock.py --hash     # печатает data_hash (идёт в registry)

Работает рядом с DVC (dvc.yaml/dvc.lock): lock-файл маленький и живёт в git,
поэтому CI проверяет целостность даже без доступа к DVC-remote.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOCK = ROOT / "data" / "data.lock.json"
TRACKED = ["evals/golden", "evals/fixtures", "data/reference"]


def sha256(p: Path) -> str:
    data = p.read_bytes()
    if p.suffix in (".jsonl", ".json", ".md", ".txt"):
        data = data.replace(b"\r\n", b"\n")  # одинаковый хэш на Windows и Linux
    return hashlib.sha256(data).hexdigest()


def compute() -> dict:
    files = {}
    for d in TRACKED:
        for p in sorted((ROOT / d).rglob("*")):
            if p.is_file():
                files[p.relative_to(ROOT).as_posix()] = sha256(p)
    digest = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    return {"data_hash": f"sha256:{digest[:16]}", "files": files}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--hash", action="store_true")
    args = ap.parse_args()
    cur = compute()
    if args.hash:
        print(cur["data_hash"])
        return 0
    if args.check:
        old = json.loads(LOCK.read_text(encoding="utf-8"))
        if old["data_hash"] != cur["data_hash"]:
            changed = sorted(set(old["files"].items()) ^ set(cur["files"].items()))
            print(f"DATA LOCK MISMATCH: {old['data_hash']} != {cur['data_hash']}")
            for path, _ in changed[:20]:
                print("  changed:", path)
            return 1
        print(f"data lock ok: {cur['data_hash']} ({len(cur['files'])} files)")
        return 0
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    LOCK.write_text(json.dumps(cur, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {LOCK.relative_to(ROOT)}: {cur['data_hash']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
