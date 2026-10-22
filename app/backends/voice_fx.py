"""Обработка голоса отбивки (чистый numpy, без новых зависимостей).

Базовый голос Silero звучит синтетично; лёгкая обработка — сдвиг тона, тёплая сатурация,
хорус и комнатная акустика — делает его живее. Так получаются голоса «Олег» и «Ромчик».
"""

from __future__ import annotations

import numpy as np


def _fft_convolve(x: np.ndarray, ir: np.ndarray) -> np.ndarray:
    n = len(x) + len(ir) - 1
    size = 1 << (n - 1).bit_length()
    return np.fft.irfft(np.fft.rfft(x, size) * np.fft.rfft(ir, size), size)[:n].astype(np.float32)


def pitch_down(x: np.ndarray, factor: float) -> np.ndarray:
    """factor < 1: ниже и медленнее (как замедленная плёнка) — грубо, зато «тяжело» звучит."""
    n = int(len(x) / factor)
    return np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype(np.float32)


def reverb(x: np.ndarray, sr: int, seconds: float, decay: float, wet: float, seed: int = 7, dark: float = 0.0) -> np.ndarray:
    """Импульсный отклик = затухающий шум (большой зал/склеп). dark — сглаживание «верхов» хвоста."""
    rng = np.random.default_rng(seed)
    n = int(seconds * sr)
    ir = rng.standard_normal(n).astype(np.float32) * np.exp(-np.linspace(0, decay, n)).astype(np.float32)
    if dark > 0:
        k = max(1, int(dark * sr / 1000))
        ir = np.convolve(ir, np.ones(k, dtype=np.float32) / k, mode="same")
    ir /= np.sqrt(np.sum(ir**2)) + 1e-9
    tail = _fft_convolve(x, ir)
    dry = np.concatenate([x, np.zeros(len(tail) - len(x), dtype=np.float32)])
    return (1 - wet) * dry + wet * tail


def echo(x: np.ndarray, sr: int, delay_s: float, feedback: float, repeats: int) -> np.ndarray:
    d = int(delay_s * sr)
    out = np.concatenate([x, np.zeros(d * repeats, dtype=np.float32)])
    for i in range(1, repeats + 1):
        out[d * i : d * i + len(x)] += x * (feedback**i)
    return out


def chorus(x: np.ndarray, sr: int, depth_ms: float = 6, rate_hz: float = 0.8, mix: float = 0.5) -> np.ndarray:
    t = np.arange(len(x)) / sr
    delay = (depth_ms / 1000 * sr) * (1 + np.sin(2 * np.pi * rate_hz * t)) / 2 + 0.01 * sr
    idx = np.clip(np.arange(len(x)) - delay, 0, len(x) - 1)
    return (1 - mix) * x + mix * np.interp(idx, np.arange(len(x)), x).astype(np.float32)


def normalize(x: np.ndarray, peak: float = 0.9) -> np.ndarray:
    m = float(np.max(np.abs(x))) or 1.0
    return (x / m * peak).astype(np.float32)


def oleg(x: np.ndarray, sr: int) -> np.ndarray:
    """Олег: низкий бархатный голос. Лёгкое понижение тона, мягкая «ламповая» теплота
    и небольшая комнатная акустика — она же маскирует синтетичность Silero."""
    y = pitch_down(x, 0.9)
    y = np.tanh(1.3 * normalize(y)) * 0.9                       # мягкая сатурация — «тёплый» тембр
    y = 0.85 * y + 0.15 * pitch_down(y, 0.5)[: len(y)]          # чуть суб-баса для глубины
    y = reverb(y, sr, seconds=0.9, decay=6.0, wet=0.18, dark=1.5)
    return normalize(y)


def romchik(x: np.ndarray, sr: int) -> np.ndarray:
    """Ромчик: бодрый энергичный кастер. Тон и темп чуть выше, лёгкий хорус для «живого» объёма,
    короткая комната."""
    y = pitch_down(x, 1.05)                                     # factor > 1: чуть выше и быстрее
    y = chorus(y, sr, depth_ms=4, rate_hz=0.5, mix=0.18)
    y = np.tanh(1.15 * normalize(y)) * 0.9
    y = reverb(y, sr, seconds=0.7, decay=6.5, wet=0.14, dark=1.0)
    return normalize(y)


# виртуальный голос -> (базовый голос Silero, обработка). Голоса подобраны на слух по итогам сравнения в UI.
CHARACTER_VOICES = {
    "oleg": ("eugene", oleg),
    "romchik": ("eugene", romchik),
}
