"""CLI Model Registry.

    python scripts/registry_cli.py list
    python scripts/registry_cli.py register --prompt caster:v2 [--model mock://llm/card-v1] [--vlm vlm_score_ocr:1.0@7.35]
    python scripts/registry_cli.py promote 2            # прогон eval-гейта; красный → не едет в Production
    python scripts/registry_cli.py rollback             # вернуть предыдущую Production-версию
    python scripts/registry_cli.py gate --stage Staging # nightly train-gate: оценить кандидатов без промоута
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import get_settings  # noqa: E402
from app.registry import Registry  # noqa: E402
from evals.gate import gate_version  # noqa: E402
from scripts.data_lock import compute  # noqa: E402


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    r = sub.add_parser("register")
    r.add_argument("--prompt", required=True)
    r.add_argument("--model", default=None)
    r.add_argument("--vlm", default=None)
    r.add_argument("--notes", default="")
    p = sub.add_parser("promote")
    p.add_argument("version", type=int)
    p.add_argument("--stage", default="Production")
    sub.add_parser("rollback")
    g = sub.add_parser("gate")
    g.add_argument("--stage", default="Staging")
    args = ap.parse_args()

    s = get_settings()
    reg = Registry(s.registry_path)
    if args.cmd == "list":
        for v in reg.versions():
            mark = "*" if v["stage"] == "Production" else " "
            print(f"{mark} v{v['version']:<3} {v['stage']:<11} {v['prompt_version']:<11} {v['vlm']:<24} {v['model_uri']:<22} {v['data_hash']}")
        return 0
    if args.cmd == "register":
        prod = reg.production()
        v = reg.register(args.model or prod["model_uri"], args.prompt, args.vlm or prod["vlm"], compute()["data_hash"], args.notes)
        print(f"registered v{v['version']} -> Staging ({v['prompt_version']}, {v['vlm']})")
        return 0
    if args.cmd == "promote":
        if args.stage != "Production":
            reg.transition(args.version, args.stage, reason="manual")
            print(f"v{args.version} -> {args.stage}")
            return 0
        passed, report = asyncio.run(gate_version(args.version, s))
        m = report["metrics"]
        print(f"eval v{args.version}: halluc={m['raw_score_halluc_rate']} golden={m['golden_pass_rate']} "
              f"redteam={m['redteam_block_rate']} faith={m['faithfulness']}")
        if not passed:
            print("GATE FAILED — v{} остаётся в Staging:\n  - {}".format(args.version, "\n  - ".join(report["gate"]["reasons"])))
            return 1
        reg.transition(args.version, "Production", m, reason="gate passed")
        print(f"GATE PASSED — v{args.version} -> Production")
        return 0
    if args.cmd == "gate":
        cands = [v for v in reg.versions() if v["stage"] == args.stage]
        if not cands:
            print(f"нет версий в {args.stage}")
            return 0
        rc = 0
        for v in cands:
            passed, report = asyncio.run(gate_version(v["version"], s))
            print(f"v{v['version']} {v['prompt_version']}: {'PASSED' if passed else 'FAILED'} {report['gate']['reasons']}")
            rc |= 0 if passed else 1
        return rc
    if args.cmd == "rollback":
        v = reg.rollback()
        print(f"rolled back: v{v['version']} -> Production")
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
