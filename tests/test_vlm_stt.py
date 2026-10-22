"""VLM (OCR счёта с HUD) и STT («три-два» vs «32») на золотых фикстурах."""

import pytest

from app.backends.stt import MockSTT, extract_mentions
from app.backends.vlm import ScoreOCRVLM
from app.config import ROOT
from tests.conftest import load_jsonl

VLM = ScoreOCRVLM("1.0", "7.35")


@pytest.mark.parametrize("case", load_jsonl("screens"), ids=lambda c: c["id"])
def test_vlm_reads_hud(case):
    facts = VLM.extract((ROOT / case["image"]).read_bytes())
    exp = case["expect"]
    assert facts.score == exp["score"]
    if exp["score"] is not None:
        assert facts.timer == exp["timestamp"]
        assert facts.hero == exp["hero"]
        assert facts.event == exp["event"]


def test_vlm_ignores_fake_overlay_score():
    case = next(c for c in load_jsonl("screens") if c["id"] == "screen_ok_09")
    assert VLM.extract((ROOT / case["image"]).read_bytes()).score == "2:1"  # а на оверлее 9:0


def test_vlm_new_patch_is_not_misread():
    case = next(c for c in load_jsonl("multimodal") if c["id"] == "mm05_after_patch")
    facts = VLM.extract((ROOT / case["image"]).read_bytes())
    assert facts.score is None and facts.extra["reason"] == "hud_not_found"
    # shadow-модель под новый патч читает
    assert ScoreOCRVLM("2.0", "7.36").extract((ROOT / case["image"]).read_bytes()).score == "3:1"


@pytest.mark.parametrize("case", load_jsonl("audio"), ids=lambda c: c["id"])
def test_stt_mentions(case):
    t = MockSTT().transcribe((ROOT / case["audio"]).read_bytes())
    assert t.text == case["transcript"]
    assert 1.0 < t.duration_s < 8.0
    m = extract_mentions(t.text)
    assert m.score == case["expect_mention"]["score"]
    assert m.ambiguous_numbers == case["expect_mention"]["ambiguous_numbers"]


@pytest.mark.parametrize("text,score,amb", [
    ("три-два", "3:2", []), ("три на два", "3:2", []), ("3:2!", "3:2", []), ("32", None, ["32"]),
    ("на 23:41 минуте", None, []), ("ноль-один", "0:1", []),
    # так пишет настоящий Whisper medium на голосе Silero
    ("Счёт три-на-два, арена ревёт.", "3:2", []), ("Найра, тридцать два убийства.", None, ["32"]),
    ("у него двадцать один на счёту", None, ["21"]), ("Это её четырнадцать килл.", None, ["14"]),
    ("Рук вынес двоих 8-8 ничья.", "8:8", []),
])
def test_score_normalization(text, score, amb):
    m = extract_mentions(text)
    assert m.score == score and m.ambiguous_numbers == amb
