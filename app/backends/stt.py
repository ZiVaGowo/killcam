"""STT + нормализация реплик кастера.

* ``MockSTT`` — CPU, без моделей: читает «эталонную расшифровку» из RIFF LIST/INFO/ICMT
  чанка WAV (так устроены фикстуры). Это честная заглушка, интерфейс как у реального STT.
* ``WhisperSTT`` — опционально (``pip install faster-whisper``, KILLCAM_STT_BACKEND=whisper),
  на GPU, если есть, иначе CPU int8.

Главное тут — ``extract_mentions``: «три-два» → счёт 3:2, а «32» — это НЕ счёт,
а двусмысленное число (32 килла? 3:2?) — помечаем ``ambiguous`` и счёт из аудио не берём.
"""

from __future__ import annotations

import math
import os
import re
import struct
import threading
from dataclasses import asdict, dataclass, field

import numpy as np

NUM_WORDS = {
    "ноль": 0, "один": 1, "одна": 1, "два": 2, "две": 2, "три": 3, "четыре": 4, "пять": 5,
    "шесть": 6, "семь": 7, "восемь": 8, "девять": 9, "десять": 10,
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10,
}
HERO_ALIASES = {
    "vex": "Vex", "векс": "Vex", "rook": "Rook", "рук": "Rook", "nyra": "Nyra", "найра": "Nyra",
    "kade": "Kade", "кейд": "Kade", "sol": "Sol", "сол": "Sol", "mira": "Mira", "мира": "Mira",
}
KILL_WORDS = ("килл", "убил", "убийств", "фраг", "kill", "вынес", "снёс", "снес", "забирает")
OBJECTIVE_WORDS = ("дракон", "барон", "башн", "objective", "объект")

_W = "|".join(sorted(NUM_WORDS, key=len, reverse=True))
WORD_SCORE_RE = re.compile(rf"\b({_W})\s*(?:-|–|:|на|\s)\s*({_W})\b", re.IGNORECASE)
DIGIT_SCORE_RE = re.compile(r"\b(\d{1,2})\s*[:\-–]\s*(\d{1,2})\b")
BARE_NUMBER_RE = re.compile(r"\b\d{2,3}\b")


@dataclass
class Transcript:
    text: str
    duration_s: float
    language: str = "ru"
    stt_version: str = "mock_stt:1.0"


@dataclass
class Mentions:
    score: str | None = None
    ambiguous_numbers: list[str] = field(default_factory=list)
    hero: str | None = None
    event: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


TEENS_TENS = {
    "одиннадцать": 11, "двенадцать": 12, "тринадцать": 13, "четырнадцать": 14, "пятнадцать": 15, "шестнадцать": 16,
    "семнадцать": 17, "восемнадцать": 18, "девятнадцать": 19, "двадцать": 20, "тридцать": 30, "сорок": 40,
    "пятьдесят": 50, "шестьдесят": 60, "семьдесят": 70, "восемьдесят": 80, "девяносто": 90,
}
_UNIT_W = "один|одна|два|две|три|четыре|пять|шесть|семь|восемь|девять"
SPELLED_RE = re.compile(rf"\b({'|'.join(TEENS_TENS)})(?:\s+({_UNIT_W}))?\b")


def _normalize(low: str) -> str:
    """Реальный Whisper пишет «три-на-два» и «тридцать два» — приводим к виду, который понимает разбор."""
    low = re.sub(r"-\s*на\s*-", " на ", low)

    def num(m: re.Match) -> str:
        base = TEENS_TENS[m.group(1)]
        return str(base + (NUM_WORDS[m.group(2)] if m.group(2) and base >= 20 else 0)) + ("" if not m.group(2) or base >= 20 else f" {m.group(2)}")

    return SPELLED_RE.sub(num, low)


def extract_mentions(text: str) -> Mentions:
    low = _normalize(text.lower())
    m = Mentions()
    ws = WORD_SCORE_RE.search(low)
    ds = DIGIT_SCORE_RE.search(low)
    if ws:
        m.score = f"{NUM_WORDS[ws.group(1)]}:{NUM_WORDS[ws.group(2)]}"
    elif ds and not (len(ds.group(2)) == 2 and int(ds.group(1)) >= 10):  # "23:41" — это таймер, не счёт
        m.score = f"{int(ds.group(1))}:{int(ds.group(2))}"
    spans = [(x.start(), x.end()) for x in DIGIT_SCORE_RE.finditer(low)]
    for num in BARE_NUMBER_RE.finditer(low):
        if not any(s <= num.start() < e for s, e in spans):
            m.ambiguous_numbers.append(num.group(0))
    for token in re.findall(r"[a-zа-яё]+", low):
        if token in HERO_ALIASES:
            m.hero = HERO_ALIASES[token]
            break
    if any(w in low for w in KILL_WORDS):
        m.event = "kill"
    elif any(w in low for w in OBJECTIVE_WORDS):
        m.event = "objective"
    return m


# ---------------------------------------------------------------- WAV I/O

def make_wav(samples: np.ndarray, sr: int, comment: str | None = None) -> bytes:
    """PCM16 mono WAV с опциональным LIST/INFO/ICMT (расшифровка для MockSTT)."""
    pcm = (np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes()
    fmt = struct.pack("<4sIHHIIHH", b"fmt ", 16, 1, 1, sr, sr * 2, 2, 16)
    chunks = fmt
    if comment is not None:
        txt = comment.encode("utf-8") + b"\x00"
        if len(txt) % 2:
            txt += b"\x00"
        icmt = struct.pack("<4sI", b"ICMT", len(txt)) + txt
        lst = b"INFO" + icmt
        chunks += struct.pack("<4sI", b"LIST", len(lst)) + lst
    chunks += struct.pack("<4sI", b"data", len(pcm)) + pcm
    return b"RIFF" + struct.pack("<I", 4 + len(chunks)) + b"WAVE" + chunks


def parse_wav(data: bytes) -> tuple[np.ndarray, int, str | None]:
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError("not a WAV file")
    pos, sr, comment, samples = 12, 16000, None, np.zeros(0, dtype=np.float32)
    while pos + 8 <= len(data):
        cid, size = struct.unpack("<4sI", data[pos : pos + 8])
        body = data[pos + 8 : pos + 8 + size]
        if cid == b"fmt ":
            sr = struct.unpack("<I", body[4:8])[0]
        elif cid == b"LIST" and body[:4] == b"INFO":
            p = 4
            while p + 8 <= len(body):
                sid, ssz = struct.unpack("<4sI", body[p : p + 8])
                if sid == b"ICMT":
                    comment = body[p + 8 : p + 8 + ssz].rstrip(b"\x00").decode("utf-8", "replace")
                p += 8 + ssz + (ssz % 2)
        elif cid == b"data":
            samples = np.frombuffer(body[: len(body) // 2 * 2], dtype="<i2").astype(np.float32) / 32767
        pos += 8 + size + (size % 2)
    return samples, sr, comment


def speechlike(text: str, seconds: float, sr: int = 16000, seed: int = 0) -> np.ndarray:
    """Синтетический «голос»: формантные слоги по символам текста (детерминированно)."""
    n = int(seconds * sr)
    t = np.arange(n) / sr
    out = np.zeros(n, dtype=np.float32)
    chars = [c for c in text if not c.isspace()] or ["a"]
    syl = max(1, n // len(chars))
    rng = np.random.default_rng(seed)
    for i, ch in enumerate(chars):
        s, e = i * syl, min(n, (i + 1) * syl)
        f0 = 110 + (ord(ch) % 40) * 3
        seg = t[s:e]
        env = np.sin(np.linspace(0, math.pi, e - s)) ** 2
        out[s:e] = env * (0.5 * np.sin(2 * math.pi * f0 * seg) + 0.25 * np.sin(2 * math.pi * f0 * 2.7 * seg))
    out += rng.normal(0, 0.01, n).astype(np.float32)
    return out * 0.6


# ---------------------------------------------------------------- backends

class MockSTT:
    version = "mock_stt:1.0"

    def transcribe(self, audio: bytes) -> Transcript:
        samples, sr, comment = parse_wav(audio)
        return Transcript(text=comment or "", duration_s=len(samples) / sr, stt_version=self.version)


# Подсказка Whisper'у: словарь трансляции, чтобы «Векс» не превращался в «Вэкс», а «три-два» в «32».
WHISPER_PROMPT = "Киберспорт, Arena Clash. Герои: Векс, Рук, Найра, Кейд, Сол, Мира. Килл, фраг, дракон, барон. Счёт три-два."


class WhisperSTT:
    """faster-whisper: на GPU (float16), при ошибке CUDA — честный фолбэк на CPU (int8).

    Модель грузится лениво, при первом запросе, и одна на процесс.
    """

    _lock = threading.Lock()

    def __init__(self, device: str = "cpu", model: str | None = None):
        self.model_name = model or os.environ.get("KILLCAM_WHISPER_MODEL", "medium")
        self.device = device
        self._model = None
        self.version = f"faster_whisper:{self.model_name}"

    def _load(self):
        with self._lock:
            if self._model is None:
                try:
                    import torch  # noqa: F401 — подтягивает CUDA-библиотеки (cuBLAS/cuDNN) для CTranslate2 на Windows
                except ImportError:
                    pass
                from faster_whisper import WhisperModel  # type: ignore

                try:
                    self._model = WhisperModel(self.model_name, device=self.device,
                                               compute_type="float16" if self.device == "cuda" else "int8")
                except Exception:  # noqa: BLE001 — нет CUDA/драйвера: явный CPU-фолбэк
                    self.device = "cpu"
                    self._model = WhisperModel(self.model_name, device="cpu", compute_type="int8")
                self.version = f"faster_whisper:{self.model_name}@{self.device}"
        return self._model

    def transcribe(self, audio: bytes) -> Transcript:
        # WAV читаем сами (PyAV из faster-whisper капризничает с версиями) и приводим к 16 кГц моно
        samples, sr, _ = parse_wav(audio)
        if sr != 16000 and len(samples):
            n = int(len(samples) * 16000 / sr)
            samples = np.interp(np.linspace(0, len(samples) - 1, n), np.arange(len(samples)), samples).astype(np.float32)
        segments, info = self._load().transcribe(
            samples.astype(np.float32), language="ru", initial_prompt=WHISPER_PROMPT, vad_filter=True, beam_size=5)
        speech = [s for s in segments if not (s.no_speech_prob > 0.6 and s.avg_logprob < -0.8)]
        text = " ".join(s.text.strip() for s in speech).strip()
        return Transcript(text=text, duration_s=info.duration, language=info.language, stt_version=self.version)


_STT_CACHE: dict[tuple, object] = {}


def get_stt(backend: str, device: str):
    if backend == "whisper":
        key = (backend, device)
        if key not in _STT_CACHE:
            _STT_CACHE[key] = WhisperSTT(device=device)
        return _STT_CACHE[key]
    return MockSTT()
