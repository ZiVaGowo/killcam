"""Конфигурация сервиса: env-переменные + params.yaml (общие с DVC/eval)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def detect_device() -> str:
    """GPU, если есть torch+CUDA, иначе явный CPU-фолбэк."""
    forced = os.environ.get("KILLCAM_DEVICE")
    if forced:
        return forced
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


@lru_cache
def load_params() -> dict:
    with open(ROOT / "params.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


@dataclass
class Settings:
    registry_path: Path = field(default_factory=lambda: Path(_env("KILLCAM_REGISTRY", str(ROOT / "registry" / "registry.json"))))
    prompts_dir: Path = ROOT / "prompts"
    runtime_dir: Path = field(default_factory=lambda: Path(_env("KILLCAM_RUNTIME_DIR", str(ROOT / "data" / "runtime"))))
    reference_path: Path = ROOT / "data" / "reference" / "ui_reference.json"
    redis_url: str = field(default_factory=lambda: _env("REDIS_URL", ""))
    inline_worker: bool = field(default_factory=lambda: _env("KILLCAM_INLINE_WORKER", "1") == "1")
    operator_key: str = field(default_factory=lambda: _env("KILLCAM_OPERATOR_KEY", "dev-operator"))
    # Бэкенды: mock (детерминированный, CPU, без сети) или openai (любой OpenAI-совместимый endpoint)
    llm_backend: str = field(default_factory=lambda: _env("KILLCAM_LLM_BACKEND", "mock"))
    llm_base_url: str = field(default_factory=lambda: _env("KILLCAM_LLM_BASE_URL", "http://localhost:8001/v1"))
    llm_api_key: str = field(default_factory=lambda: _env("KILLCAM_LLM_API_KEY", ""))
    llm_model: str = field(default_factory=lambda: _env("KILLCAM_LLM_MODEL", "gpt-4o-mini"))
    stt_backend: str = field(default_factory=lambda: _env("KILLCAM_STT_BACKEND", "mock"))      # mock | whisper
    tts_backend: str = field(default_factory=lambda: _env("KILLCAM_TTS_BACKEND", "mock"))      # mock | silero
    stream_delay_s: float = field(default_factory=lambda: float(_env("KILLCAM_STREAM_DELAY", "0.01")))
    device: str = field(default_factory=detect_device)

    @property
    def params(self) -> dict:
        return load_params()


@lru_cache
def get_settings() -> Settings:
    s = Settings()
    s.runtime_dir.mkdir(parents=True, exist_ok=True)
    return s
