"""Оценка стоимости вызовов по тарифам из params.yaml."""

from __future__ import annotations

from dataclasses import dataclass, field

from app.config import load_params
from app.monitoring.metrics import COST


@dataclass
class CostMeter:
    parts: dict[str, float] = field(default_factory=dict)

    @property
    def rates(self) -> dict:
        return load_params()["cost"]

    def add(self, component: str, usd: float) -> float:
        self.parts[component] = self.parts.get(component, 0.0) + usd
        COST.labels(component).inc(usd)
        return usd

    def vlm(self, images: int = 1) -> float:
        return self.add("vlm", images * self.rates["vlm_per_image"])

    def stt(self, seconds: float) -> float:
        return self.add("stt", seconds * self.rates["stt_per_second"])

    def llm(self, tokens_in: int, tokens_out: int) -> float:
        r = self.rates
        return self.add("llm", tokens_in / 1000 * r["llm_in_per_1k_tokens"] + tokens_out / 1000 * r["llm_out_per_1k_tokens"])

    def tts(self, chars: int) -> float:
        return self.add("tts", chars * self.rates["tts_per_char"])

    def image_gen(self, n: int = 1) -> float:
        return self.add("image_gen", n * self.rates["image_gen_per_image"])

    @property
    def total(self) -> float:
        return round(sum(self.parts.values()), 6)
