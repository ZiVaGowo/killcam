"""TTS-бэкенды для голосовой отбивки.

* ``MockTTS`` — CPU, без моделей: синтетический «голос» (формантные слоги). Для CI и офлайна.
* ``SileroTTS`` — Silero TTS v5.6 (ru) + обработка: голоса Олег (oleg) и Ромчик (romchik).
  Работает и на CPU, на GPU — быстрее. KILLCAM_TTS_BACKEND=silero, модель — KILLCAM_TTS_MODEL, голос — KILLCAM_TTS_SPEAKER.

Silero не читает цифры, поэтому ``spell_numbers`` превращает «23:41» в «двадцать три сорок один»,
а счёт «3:2» — в «три два».
"""

from __future__ import annotations

import os
import re
import threading

import numpy as np

UNITS = ["ноль", "один", "два", "три", "четыре", "пять", "шесть", "семь", "восемь", "девять", "десять",
         "одиннадцать", "двенадцать", "тринадцать", "четырнадцать", "пятнадцать", "шестнадцать",
         "семнадцать", "восемнадцать", "девятнадцать"]
TENS = ["", "", "двадцать", "тридцать", "сорок", "пятьдесят", "шестьдесят", "семьдесят", "восемьдесят", "девяносто"]
HUNDREDS = ["", "сто", "двести", "триста", "четыреста", "пятьсот", "шестьсот", "семьсот", "восемьсот", "девятьсот"]


def number_ru(n: int) -> str:
    if n < 20:
        return UNITS[n]
    if n < 100:
        t, u = divmod(n, 10)
        return TENS[t] + (f" {UNITS[u]}" if u else "")
    if n < 1000:
        h, r = divmod(n, 100)
        return HUNDREDS[h] + (f" {number_ru(r)}" if r else "")
    return " ".join(UNITS[int(d)] for d in str(n))


def spell_numbers(text: str) -> str:
    def time_or_score(m: re.Match) -> str:
        a, b = m.group(1), m.group(2)
        if len(b) == 2 and b.startswith("0"):  # 5:07 -> «пять ноль семь»
            return f"{number_ru(int(a))} ноль {number_ru(int(b))}"
        return f"{number_ru(int(a))} {number_ru(int(b))}"

    text = re.sub(r"(\d{1,2})\s*[:\-]\s*(\d{1,2})", time_or_score, text)
    text = re.sub(r"\d+", lambda m: number_ru(int(m.group(0))), text)
    text = re.sub(r"[#@]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


class MockTTS:
    version = "mock_tts:1.0"
    sample_rate = 16000

    def voices(self) -> dict:
        return {"models": {"mock": ["synthetic"]}, "default": "mock/synthetic"}

    def synth(self, text: str, seconds: float, voice: str | None = None) -> np.ndarray:
        import hashlib

        from app.backends.stt import speechlike

        seed = int(hashlib.sha1(text.encode()).hexdigest()[:8], 16)
        return speechlike(text, min(seconds - 1, max(2, len(text) // 10)), self.sample_rate, seed=seed)


# Оставлены голоса, которые на слух звучат естественнее всего (по итогам сравнения в UI)
# Голоса отбивки: базовый Silero + обработка (app/backends/voice_fx.py)
SILERO_MODELS = {
    "v5_6_ru": ["oleg", "romchik"],
}


class SileroTTS:
    """Silero TTS: несколько версий модели, голос выбирается на каждый запрос ("v5_6_ru/romchik")."""

    sample_rate = 48000
    _lock = threading.Lock()

    def __init__(self, device: str = "cpu", speaker: str | None = None, model_id: str | None = None):
        self.device = device
        self.model_id = model_id or os.environ.get("KILLCAM_TTS_MODEL", "v5_6_ru")
        self.speaker = speaker or os.environ.get("KILLCAM_TTS_SPEAKER", "oleg")
        self._models: dict[str, object] = {}
        self.version = f"silero_tts:{self.model_id}/{self.speaker}"

    def voices(self) -> dict:
        return {"models": SILERO_MODELS, "default": f"{self.model_id}/{self.speaker}"}

    def _resolve(self, voice: str | None) -> tuple[str, str]:
        model_id, _, speaker = (voice or "").partition("/")
        model_id = model_id if model_id in SILERO_MODELS else self.model_id
        speaker = speaker if speaker in SILERO_MODELS[model_id] else self.speaker
        return model_id, speaker

    def _load(self, model_id: str | None = None):
        model_id = model_id or self.model_id
        with self._lock:
            if model_id not in self._models:
                import torch

                model, _ = torch.hub.load("snakers4/silero-models", "silero_tts", language="ru", speaker=model_id, trust_repo=True)
                model.to(torch.device(self.device))
                # TorchScript компилируется на первых вызовах — прогреваем, чтобы отбивка в эфир шла за доли секунды
                for _ in range(3):
                    model.apply_tts(text="разминка голоса", speaker="eugene", sample_rate=self.sample_rate)
                self._models[model_id] = model
                self.version = f"silero_tts:{self.model_id}/{self.speaker}@{self.device}"
        return self._models[model_id]

    def synth(self, text: str, seconds: float, voice: str | None = None) -> np.ndarray:
        import torch

        from app.backends.voice_fx import CHARACTER_VOICES

        model_id, speaker = self._resolve(voice)
        model = self._load(model_id)
        base, fx = CHARACTER_VOICES.get(speaker, (speaker, None))
        with torch.inference_mode():
            audio = model.apply_tts(text=spell_numbers(text), speaker=base, sample_rate=self.sample_rate)
        audio = audio.detach().cpu().numpy().astype(np.float32)
        return fx(audio, self.sample_rate) if fx else audio


_TTS_CACHE: dict[tuple, object] = {}


def get_tts(backend: str, device: str):
    if backend == "silero":
        key = (backend, device)
        if key not in _TTS_CACHE:
            _TTS_CACHE[key] = SileroTTS(device=device)
        return _TTS_CACHE[key]
    return MockTTS()
