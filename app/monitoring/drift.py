"""Дрифт эмбеддингов скринов относительно эталона UI текущего патча (data/reference, DVC).

Эмбеддинг — дешёвый и детерминированный (CPU): 8x8 grayscale-миниатюра + 4x4x4 цветовая гистограмма.
Для прод-версии можно подставить CLIP — интерфейс ``embed(img) -> np.ndarray`` тот же.
drift_score = средняя стандартизованная разница средних (окно vs эталон) по измерениям.
"""

from __future__ import annotations

import json
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image

from app.monitoring.metrics import DRIFT, STORE


def embed(img: Image.Image) -> np.ndarray:
    small = np.asarray(img.convert("L").resize((8, 8), Image.BILINEAR), dtype=np.float32).ravel() / 255.0
    q = (np.asarray(img.convert("RGB").resize((64, 36)), dtype=np.uint8) // 64).reshape(-1, 3).astype(np.int64)
    idx = q[:, 0] * 16 + q[:, 1] * 4 + q[:, 2]
    hist = np.bincount(idx, minlength=64).astype(np.float32)
    hist /= hist.sum()
    return np.concatenate([small, hist * 8])


def build_reference(embeddings: list[np.ndarray], patch: str) -> dict:
    arr = np.stack(embeddings)
    return {"patch": patch, "n": len(embeddings), "mean": arr.mean(0).round(6).tolist(), "std": arr.std(0).round(6).tolist()}


def drift_score(window: np.ndarray, ref: dict) -> float:
    mu, sd = np.array(ref["mean"]), np.array(ref["std"])
    return float(np.mean(np.abs(window.mean(0) - mu) / (sd + 0.05)))


class DriftMonitor:
    def __init__(self, reference_path: Path, window: int = 50, threshold: float = 0.35, min_window: int = 20):
        p = Path(reference_path)
        self.ref = json.loads(p.read_text()) if p.exists() else None
        self.window: deque[np.ndarray] = deque(maxlen=window)
        self.threshold = threshold
        self.min_window = min_window
        self.lock = threading.Lock()
        self.last_score = 0.0

    def observe(self, img: Image.Image) -> float | None:
        if self.ref is None:
            return None
        with self.lock:
            self.window.append(embed(img))
            if len(self.window) < self.min_window:  # на паре кадров оценка шумная — ложные алерты
                return None
            self.last_score = drift_score(np.stack(self.window), self.ref)
        DRIFT.set(self.last_score)
        STORE.drift_series.append((time.time(), round(self.last_score, 4)))
        return self.last_score

    def reset(self) -> None:
        with self.lock:
            self.window.clear()
            self.last_score = 0.0

    @property
    def alert(self) -> bool:
        return self.last_score > self.threshold

    def status(self) -> dict:
        return {
            "reference_patch": self.ref["patch"] if self.ref else None,
            "window": len(self.window),
            "drift_score": round(self.last_score, 4),
            "threshold": self.threshold,
            "alert": self.alert,
        }
