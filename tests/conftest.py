from __future__ import annotations

import base64
import json
import shutil
from dataclasses import replace
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import ROOT, get_settings
from app.main import create_app

GOLD = ROOT / "evals" / "golden"


def b64(path: str | Path) -> str:
    return base64.b64encode((ROOT / path).read_bytes()).decode()


def load_jsonl(name: str) -> list[dict]:
    return [json.loads(line) for line in (GOLD / f"{name}.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]


def chat_body(image: str | None, audio: str | None = None, text: str = "Сделай карточку момента", chat=None,
              stream: bool = False, overlay_text: str | None = None, **extra) -> dict:
    parts: list[dict] = [{"type": "text", "text": text}]
    if image:
        parts.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64(image)}"}})
    if audio:
        parts.append({"type": "input_audio", "input_audio": {"data": b64(audio), "format": "wav"}})
    body = {"model": "killcam-card", "messages": [{"role": "user", "content": parts}], "stream": stream,
            "match_id": "final_g5", "chat": chat or []}
    if overlay_text:
        body["overlay_text"] = overlay_text
    body.update(extra)
    return body


def parse_sse(text: str) -> tuple[str, list[dict]]:
    chunks = []
    for line in text.splitlines():
        if line.startswith("data: ") and line != "data: [DONE]":
            chunks.append(json.loads(line[6:]))
    content = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks)
    return content, chunks


@pytest.fixture()
def settings(tmp_path):
    reg = tmp_path / "registry.json"
    shutil.copy(ROOT / "registry" / "registry.json", reg)
    s = replace(get_settings(), registry_path=reg, runtime_dir=tmp_path / "runtime", stream_delay_s=0.0, redis_url="", inline_worker=True)
    s.runtime_dir.mkdir(parents=True, exist_ok=True)
    return s


@pytest.fixture()
def client(settings):
    with TestClient(create_app(settings)) as c:
        yield c


OPERATOR = {"X-Operator-Key": "dev-operator"}
