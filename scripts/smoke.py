"""Смоук живого сервиса: /health + стрим карточки по килл-скрину.

    python scripts/smoke.py --url http://127.0.0.1:8000
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--wait", type=float, default=30, help="сколько ждать старта сервиса, с")
    args = ap.parse_args()

    deadline = time.time() + args.wait
    while True:
        try:
            h = httpx.get(f"{args.url}/health", timeout=3)
            if h.status_code == 200:
                break
        except httpx.HTTPError:
            pass
        if time.time() > deadline:
            print("health: service did not start")
            return 1
        time.sleep(1)
    health = h.json()
    print(f"health ok: v{health['production']['version']} {health['production']['prompt_version']} device={health['device']}")

    img = base64.b64encode((ROOT / "evals/fixtures/screens/ok_01.png").read_bytes()).decode()
    body = {"messages": [{"role": "user", "content": [{"type": "text", "text": "карточка"},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img}"}}]}], "stream": True, "match_id": "smoke"}
    content, final = "", None
    with httpx.stream("POST", f"{args.url}/v1/chat/completions", json=body, timeout=30) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if line.startswith("data: ") and line != "data: [DONE]":
                chunk = json.loads(line[6:])
                content += chunk["choices"][0]["delta"].get("content", "")
                final = chunk
    card = json.loads(content)
    assert card["score"] == "3:2", card
    print(f"stream ok: score={card['score']} ttft={final['killcam']['ttft_s']}s cost=${final['killcam']['cost_usd']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
