"""Тесты схемы: карточка, золотой набор, lock данных."""

import subprocess
import sys

import pytest
from pydantic import ValidationError

from app.config import ROOT
from app.schemas import MomentCard
from tests.conftest import load_jsonl

VALID = {
    "match_id": "final_g5", "timestamp": "23:41", "event": "kill", "hero": "Vex", "score": "3:2",
    "headline": "Vex забирает килл", "caption": "Счёт 3:2.", "hashtags": ["#ArenaClash"],
    "confidence": 0.98, "needs_review": False,
    "suggested_actions": [{"tool": "publish_social", "args": {"text": "x"}, "requires_hitl": True}],
}


def test_valid_card():
    card = MomentCard(**VALID)
    assert card.score == "3:2"


@pytest.mark.parametrize("field,value", [
    ("score", "три-два"), ("score", "3-2"), ("event", "ace"), ("timestamp", "23.41"),
    ("hashtags", ["no hash"]), ("confidence", 1.5), ("headline", "x" * 500),
])
def test_invalid_card(field, value):
    with pytest.raises(ValidationError):
        MomentCard(**{**VALID, field: value})


def test_null_score_allowed_for_unreadable_hud():
    MomentCard(**{**VALID, "score": None, "needs_review": True})


def test_golden_set_sizes():
    gold = load_jsonl("screens") + load_jsonl("audio") + load_jsonl("chat")
    assert 20 <= len(gold) <= 50
    assert len(load_jsonl("screens")) == 15
    assert len(load_jsonl("audio")) == 10
    assert len(load_jsonl("chat")) == 8
    assert len(load_jsonl("redteam")) == 10
    assert len(load_jsonl("multimodal")) == 5


def test_golden_fixtures_exist_and_have_expectations():
    for name in ("screens", "audio", "chat", "redteam", "multimodal"):
        for case in load_jsonl(name):
            assert "expect" in case, case["id"]
            for key in ("image", "audio"):
                if key in case:
                    assert (ROOT / case[key]).exists(), case[key]


def test_redteam_covers_required_attacks():
    attacks = " ".join(c["attack"] for c in load_jsonl("redteam")).lower()
    for kind in ("injection", "jailbreak", "утечка system", "allow-list"):
        assert kind in attacks


def test_data_lock_matches():
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "data_lock.py"), "--check"], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
