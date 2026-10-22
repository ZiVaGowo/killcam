"""Эталон UI текущей патч-версии для дрифт-монитора (DVC stage `reference`).

    python scripts/build_reference.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import yaml  # noqa: E402

from app.backends.hud import HEROES, FrameSpec, render_frame  # noqa: E402
from app.monitoring.drift import build_reference, embed  # noqa: E402


def main() -> None:
    params = yaml.safe_load((ROOT / "params.yaml").read_text(encoding="utf-8"))
    patch = params["game"]["ui_patch"]
    heroes = list(HEROES)
    embs = []
    for i in range(80):
        spec = FrameSpec(f"{i % 9}:{i % 7}", f"{5 + i % 30}:{(i * 7) % 60:02d}", heroes[i % len(heroes)],
                         ["kill", "objective", "none"][i % 3], patch=patch, seed=90_000 + i)
        embs.append(embed(render_frame(spec)))
    ref = build_reference(embs, patch)
    out = ROOT / "data" / "reference" / "ui_reference.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(ref) + "\n", encoding="utf-8")
    print(f"reference for UI patch {patch}: {ref['n']} frames -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
