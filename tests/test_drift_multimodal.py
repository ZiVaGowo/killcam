"""Мониторинг дрифта после патча UI + 5 мультимодальных кейсов через API."""

import json
from pathlib import Path

import pytest
from PIL import Image

from app.config import ROOT, load_params
from app.monitoring.drift import DriftMonitor
from tests.conftest import chat_body, load_jsonl


def _feed(monitor: DriftMonitor, folder: Path, n: int) -> float:
    for p in sorted(folder.glob("*.png"))[:n]:
        monitor.observe(Image.open(p).convert("RGB"))
    return monitor.last_score


def test_drift_alert_after_patch():
    cfg = load_params()["drift"]
    m = DriftMonitor(ROOT / "data/reference/ui_reference.json", cfg["window"], cfg["threshold"])
    _feed(m, ROOT / "data/vod/final_g5", 50)
    assert not m.alert, m.status()
    m.reset()
    _feed(m, ROOT / "data/vod/after_patch_7_36", 40)
    assert m.alert, m.status()
    assert m.status()["reference_patch"] == "7.35"


@pytest.mark.parametrize("case", load_jsonl("multimodal"), ids=lambda c: c["id"])
def test_multimodal_case(client, case):
    r = client.post("/v1/chat/completions", json=chat_body(case["image"], audio=case["audio"], chat=case["chat"]))
    assert r.status_code == 200
    body = r.json()
    card = json.loads(body["choices"][0]["message"]["content"])
    for k, v in case["expect"].items():
        if k == "transcript_contains":
            assert v in body["killcam"]["transcript"]
        else:
            assert card[k] == v, (k, card)
    for bad in case.get("forbid", []):
        assert bad not in body["choices"][0]["message"]["content"]
