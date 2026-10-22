"""CLI eval-гейта.

    python -m evals.run_eval                          # Production-версия vs baseline
    python -m evals.run_eval --version 2              # кандидат из registry
    python -m evals.run_eval --prompt caster:v2       # ad-hoc: промпт из рабочей копии (PR)
    python -m evals.run_eval --update-baseline        # зафиксировать прод-базлайн

Код возврата 1, если гейт красный — CI/promote не пропускают версию.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

from app.config import ROOT, get_settings
from app.registry import Registry
from evals.gate import BASELINE, compare, evaluate, load_baseline, write_report


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows-консоль
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", type=int, default=None, help="версия из registry (по умолчанию Production)")
    ap.add_argument("--prompt", default=None, help="ad-hoc prompt_version поверх Production, например caster:v2")
    ap.add_argument("--update-baseline", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "evals" / "reports" / "latest.json"))
    args = ap.parse_args()

    target: int | dict | None = args.version
    if args.prompt:
        target = {**Registry(get_settings().registry_path).production(), "prompt_version": args.prompt, "version": 0}

    report = asyncio.run(evaluate(target, get_settings()))
    m = report["metrics"]
    baseline = load_baseline()
    passed, reasons = compare(m, baseline) if baseline else (True, ["no baseline yet"])
    report["gate"] = {"passed": passed, "reasons": reasons, "baseline": baseline}
    write_report(report, __import__("pathlib").Path(args.out))

    print(f"\n== eval {report['version']} ==")
    width = max(len(k) for k in m)
    for k, v in m.items():
        b = baseline.get(k)
        print(f"  {k:<{width}}  {v!s:<10} {'(baseline ' + str(b) + ')' if b is not None else ''}")
    failed = [c for c in report["cases"] if not c["passed"]]
    if failed:
        print(f"\n  failed cases ({len(failed)}):")
        for c in failed[:25]:
            print(f"    - {c['id']}: {'; '.join(c['fails'])}")
    if args.update_baseline:
        BASELINE.write_text(json.dumps({"version": report["version"], "metrics": m}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"\n  baseline updated -> {BASELINE.relative_to(ROOT)}")
        return 0
    print(f"\n  GATE: {'PASSED' if passed else 'FAILED'}" + ("" if passed else "\n    " + "\n    ".join(reasons)))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
