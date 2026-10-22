"""TTS-бэкенды для голосовой отбивки.

* ``MockTTS`` — CPU, без моделей: синтетический «голос» (формантные слоги). Для CI и офлайна.
* ``SileroTTS`` — Silero TTS v4 (ru), естественные русские голоса (aidar, baya, kseniya, xenia, eugene).
  Работает и на CPU, на GPU — быстрее. KILLCAM_TTS_BACKEND=silero, голос — KILLCAM_TTS_SPEAKER.

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

    def synth(self, text: str, seconds: float) -> np.ndarray:
        import hashlib

        from app.backends.stt import speechlike

        seed = int(hashlib.sha1(text.encode()).hexdigest()[:8], 16)
        return speechlike(text, min(seconds - 1, max(2, len(text) // 10)), self.sample_rate, seed=seed)


class SileroTTS:
    sample_rate = 48000
    _lock = threading.Lock()

    def __init__(self, device: str = "cpu", speaker: str | None = None):
        self.device = device
        self.speaker = speaker or os.environ.get("KILLCAM_TTS_SPEAKER", "aidar")
        self._model = None
        self.version = f"silero_tts:v4_ru/{self.speaker}"

    def _load(self):
        with self._lock:
            if self._model is None:
                import torch

                model, _ = torch.hub.load("snakers4/silero-models", "silero_tts", language="ru", speaker="v4_ru", trust_repo=True)
                model.to(torch.device(self.device))
                # TorchScript компилируется на первых вызовах — прогреваем, чтобы отбивка в эфир шла за доли секунды
                for _ in range(3):
                    model.apply_tts(text="разминка голоса", speaker=self.speaker, sample_rate=self.sample_rate, put_accent=True, put_yo=True)
                self._model = model
                self.version = f"silero_tts:v4_ru/{self.speaker}@{self.device}"
        return self._model

    def synth(self, text: str, seconds: float) -> np.ndarray:
        import torch

        model = self._load()
        with torch.inference_mode():
            audio = model.apply_tts(text=spell_numbers(text), speaker=self.speaker, sample_rate=self.sample_rate,
                                    put_accent=True, put_yo=True)
        return audio.detach().cpu().numpy().astype(np.float32)


_TTS_CACHE: dict[tuple, object] = {}


def get_tts(backend: str, device: str):
    if backend == "silero":
        key = (backend, device)
        if key not in _TTS_CACHE:
            _TTS_CACHE[key] = SileroTTS(device=device)
        return _TTS_CACHE[key]
    return MockTTS()
