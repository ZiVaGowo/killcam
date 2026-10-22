"""Офлайн-отчёт дрифта по папке скринов (nightly).

    python scripts/drift_report.py data/vod/final_g5
    python scripts/drift_report.py data/vod/after_patch_7_36 --fail-on-alert
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from PIL import Image  # noqa: E402

from app.config import load_params  # noqa: E402
from app.monitoring.drift import DriftMonitor  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--fail-on-alert", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cfg = load_params()["drift"]
    m = DriftMonitor(ROOT / "data/reference/ui_reference.json", cfg["window"], cfg["threshold"])
    files = sorted(Path(args.folder).glob("*.png"))
    for p in files:
        m.observe(Image.open(p).convert("RGB"))
    status = {"folder": args.folder, "frames": len(files), **m.status()}
    print(json.dumps(status, ensure_ascii=False))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    return 1 if args.fail_on_alert and m.alert else 0


if __name__ == "__main__":
    sys.exit(main())
