"""Речь: нормализация чисел для TTS, эндпоинт STT, опционально — реальные Whisper + Silero.

Реальные модели тяжёлые, поэтому их тест запускается только явно:
    KILLCAM_REAL_MODELS=1 pytest tests/test_speech.py -q
"""

import base64
import os

import pytest

from app.backends.tts import number_ru, spell_numbers
from app.config import ROOT


@pytest.mark.parametrize("n,words", [(0, "ноль"), (7, "семь"), (12, "двенадцать"), (23, "двадцать три"), (40, "сорок"), (41, "сорок один"), (115, "сто пятнадцать")])
def test_number_ru(n, words):
    assert number_ru(n) == words


@pytest.mark.parametrize("text,spoken", [
    ("Vex забирает килл на 23:41!", "Vex забирает килл на двадцать три сорок один!"),
    ("Счёт 3:2.", "Счёт три два."),
    ("на 5:07", "на пять ноль семь"),
    ("#Vex 32 фрага", "Vex тридцать два фрага"),
])
def test_spell_numbers(text, spoken):
    assert spell_numbers(text) == spoken


def test_stt_endpoint_mock(client):
    audio = base64.b64encode((ROOT / "evals/fixtures/audio/a01.wav").read_bytes()).decode()
    d = client.post("/v1/media/stt", json={"audio_b64": audio}).json()
    assert d["text"] == "Векс забирает килл! Три-два!"
    assert d["mentions"] == {"score": "3:2", "ambiguous_numbers": [], "hero": "Vex", "event": "kill"}
    assert d["stt"].startswith("mock_stt") and 1.0 < d["duration_s"] < 8.0


def test_stt_endpoint_rejects_garbage(client):
    assert client.post("/v1/media/stt", json={}).status_code == 422
    assert client.post("/v1/media/stt", json={"audio_b64": base64.b64encode(b"not a wav").decode()}).status_code == 422


@pytest.mark.skipif(os.environ.get("KILLCAM_REAL_MODELS") != "1", reason="реальные модели: KILLCAM_REAL_MODELS=1")
def test_real_silero_to_whisper_roundtrip():
    """Silero произносит отбивку, Whisper её слышит — сквозная проверка настоящих моделей."""
    from app.backends.stt import WhisperSTT, extract_mentions
    from app.backends.tts import SileroTTS
    from app.config import detect_device
    from app.media import synth_bumper

    device = detect_device()
    wav = synth_bumper("Векс забирает килл! Счёт три-два!", 8, SileroTTS(device=device))
    text = WhisperSTT(device=device).transcribe(wav).text
    m = extract_mentions(text)
    assert m.hero == "Vex" and m.event == "kill", text


def test_character_voice_effects():
    import numpy as np

    from app.backends.voice_fx import CHARACTER_VOICES

    sr = 48000
    t = np.arange(sr * 2) / sr
    voice = (0.5 * np.sin(2 * np.pi * 180 * t)).astype(np.float32)
    for name, (base, fx) in CHARACTER_VOICES.items():
        out = fx(voice, sr)
        assert base == "eugene"
        assert np.isfinite(out).all() and np.max(np.abs(out)) <= 0.91, name
        assert len(out) > len(voice) * 0.9, name  # сдвиг тона/темпа + хвост реверберации
