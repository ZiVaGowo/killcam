"""Оркестрация момента: скрин + голос кастера + чат → проверенная факт-карточка (стрим).

    image ──► VLM (vlm_score_ocr) ──► FACTS (счёт/герой/событие/таймер) ──┐
    audio ──► STT ──► упоминания кастера (только как подсказка) ─────────┤
    chat/overlay ──► sanitize ──► <data>...</data> ───────────────────────┼─► LLM ─► StreamGuard ─► карточка
    инструкция продюсера ──► check_instruction (jailbreak? → отказ) ─────┘

Один и тот же код используется API (/v1/chat/completions), батч-воркером и eval-гейтом.
"""

from __future__ import annotations

import os
import secrets
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from app.backends.llm import LLMRequest, MockLLM, NaiveLLM, OpenAICompatLLM, estimate_tokens
from app.backends.stt import Mentions, Transcript, extract_mentions, get_stt
from app.backends.vlm import OpenAIVLM, ScoreOCRVLM, VLMFacts, load_image
from app.config import Settings
from app.guards import DataReport, FieldGuard, GuardVerdict, StreamGuard, check_instruction, sanitize_untrusted
from app.monitoring.cost import CostMeter
from app.monitoring.drift import DriftMonitor
from app.monitoring.metrics import CARDS, GUARD_EVENTS, SCORE_HALLUC
from app.registry import Registry, load_prompt
from app.schemas import MomentCard

CANARY = os.environ.get("KILLCAM_CANARY") or f"KC-CANARY-{secrets.token_hex(6)}"
FACT_FIELDS = (("score", "score"), ("hero", "hero"), ("event", "event"), ("timestamp", "timer"))


@dataclass
class MomentInput:
    match_id: str = "unknown"
    image: bytes | None = None
    audio: bytes | None = None
    user_text: str = ""
    chat: list[str] = field(default_factory=list)
    overlay_text: str | None = None
    highlights_only: bool = False  # батч: LLM зовём только на килл/объект (детектор «это вообще килл?»)


@dataclass
class ModelBundle:
    version: dict
    prompt: str
    llm: Any
    vlm: Any

    @property
    def tag(self) -> str:
        return f"v{self.version['version']}|{self.version['prompt_version']}|{self.version['vlm']}"


@dataclass
class Analysis:
    inp: MomentInput
    verdict: GuardVerdict
    vlm: VLMFacts
    transcript: Transcript | None
    mentions: Mentions
    data: DataReport
    facts: dict
    cost: CostMeter
    shadow: dict | None = None


@dataclass
class CardResult:
    card: dict | None
    refused: bool
    refusal_reasons: list[str]
    guard: dict
    raw: dict
    facts: dict
    data_report: dict
    schema_valid: bool
    cost_usd: float
    ttft: float | None
    latency: float
    model_tag: str
    transcript: str | None = None
    skipped: bool = False

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def make_vlm(spec: str, settings: Settings):
    """'vlm_score_ocr:1.0@7.35' -> ScoreOCRVLM; 'vlm_openai:<model>' -> OpenAIVLM."""
    name, _, rest = spec.partition(":")
    if name == "vlm_openai":
        return OpenAIVLM(settings.llm_base_url, settings.llm_api_key, rest)
    ver, _, patch = rest.partition("@")
    return ScoreOCRVLM(version=ver or "1.0", patch=patch or "7.35")


class Pipeline:
    def __init__(self, settings: Settings, registry: Registry, drift: DriftMonitor | None = None, llm_override: str | None = None):
        self.settings = settings
        self.registry = registry
        self.drift = drift
        self.llm_override = llm_override
        self.stt = get_stt(settings.stt_backend, settings.device)
        self._bundles: dict[tuple, ModelBundle] = {}
        shadow = os.environ.get("KILLCAM_SHADOW_VLM")
        self.shadow_vlm = make_vlm(shadow, settings) if shadow else None

    # ------------------------------------------------------------- models
    def bundle(self, version: int | None = None) -> ModelBundle:
        v = self.registry.get(version) if version is not None else self.registry.production()
        key = (v["version"], v["model_uri"], v["prompt_version"], v["vlm"], self.llm_override)
        if key not in self._bundles:
            prompt = load_prompt(self.settings.prompts_dir, v["prompt_version"]).replace("{CANARY}", CANARY)
            uri = self.llm_override or v["model_uri"]
            if self.settings.llm_backend == "openai" and uri.startswith("openai://"):
                llm = OpenAICompatLLM(prompt, self.settings.llm_base_url, self.settings.llm_api_key, uri.removeprefix("openai://"))
            elif uri.startswith("mock://naive"):
                llm = NaiveLLM(prompt, uri, delay=self.settings.stream_delay_s)
            else:
                llm = MockLLM(prompt, uri, delay=self.settings.stream_delay_s)
            self._bundles[key] = ModelBundle(version=v, prompt=prompt, llm=llm, vlm=make_vlm(v["vlm"], self.settings))
        return self._bundles[key]

    # ------------------------------------------------------------- analysis
    def analyze(self, inp: MomentInput, b: ModelBundle) -> Analysis:
        cost = CostMeter()
        verdict = check_instruction(inp.user_text)
        vlm = VLMFacts(vlm_version=b.vlm.id)
        shadow = None
        if inp.image:
            vlm = b.vlm.extract(inp.image)
            cost.vlm()
            if self.drift is not None:
                self.drift.observe(load_image(inp.image)[0])
            if self.shadow_vlm is not None:
                sf = self.shadow_vlm.extract(inp.image)
                shadow = {"vlm": self.shadow_vlm.id, "score": sf.score, "agree": sf.score == vlm.score}
        transcript, mentions = None, Mentions()
        if inp.audio:
            transcript = self.stt.transcribe(inp.audio)
            cost.stt(transcript.duration_s)
            mentions = extract_mentions(transcript.text)
        overlay = inp.overlay_text or vlm.overlay_text
        data = sanitize_untrusted(inp.chat, overlay)
        for flag in data.flags:
            GUARD_EVENTS.labels(flag["flag"]).inc()
        facts = {
            "game": vlm.game,
            "score": vlm.score,
            "timer": vlm.timer,
            "event": vlm.event if vlm.game == "arena_clash" else "none",
            "hero": vlm.hero,
            "hud_confidence": vlm.hud_confidence,
            "kill_feed": vlm.kill_feed,
            "towers": vlm.towers,
            "caster": mentions.to_dict(),  # подсказка, НЕ источник счёта
        }
        return Analysis(inp, verdict, vlm, transcript, mentions, data, facts, cost, shadow)

    # ------------------------------------------------------------- generation
    async def stream(self, inp: MomentInput, version: int | None = None) -> AsyncIterator[dict]:
        """События: {"type": "delta", "text"} ... {"type": "final", "result": CardResult}."""
        t0 = time.perf_counter()
        b = self.bundle(version)
        a = self.analyze(inp, b)
        if not a.verdict.allowed:
            for r in a.verdict.reasons:
                GUARD_EVENTS.labels(f"input_{r.split(':')[0]}").inc()
            res = CardResult(
                card=None, refused=True, refusal_reasons=a.verdict.reasons, guard={"events": []}, raw={},
                facts=a.facts, data_report=a.data.to_dict(), schema_valid=True, cost_usd=a.cost.total,
                ttft=None, latency=time.perf_counter() - t0, model_tag=b.tag,
                transcript=a.transcript.text if a.transcript else None,
            )
            yield {"type": "refusal", "text": "Запрос отклонён guard'ом: " + ", ".join(a.verdict.reasons)}
            yield {"type": "final", "result": res}
            return
        if inp.highlights_only and a.facts["event"] not in ("kill", "objective"):
            yield {"type": "final", "result": CardResult(
                card=None, refused=False, refusal_reasons=[], guard={"events": []}, raw={}, facts=a.facts,
                data_report=a.data.to_dict(), schema_valid=True, cost_usd=a.cost.total, ttft=None,
                latency=time.perf_counter() - t0, model_tag=b.tag, skipped=True,
            )}
            return

        req =LLMRequest(system_prompt=b.prompt, facts=a.facts, data_block=a.data.block, user_text=inp.user_text, match_id=inp.match_id)
        fg = FieldGuard(facts=a.facts, canary=CANARY, system_prompt=b.prompt)
        sg = StreamGuard(fg)
        ttft, out_text = None, []
        async for chunk in b.llm.stream(req):
            out_text.append(chunk)
            safe = sg.feed(chunk)
            if safe:
                if ttft is None:
                    ttft = time.perf_counter() - t0
                yield {"type": "delta", "text": safe}
        a.cost.llm(estimate_tokens(b.prompt + str(req.messages())), estimate_tokens("".join(out_text)))

        card, schema_valid = sg.card, True
        try:
            card = MomentCard(**sg.card).model_dump()
        except ValidationError as e:
            schema_valid = False
            fg._event("schema_invalid", "*", str(e.errors()[:2]))
        for ev in fg.events:
            GUARD_EVENTS.labels(ev["kind"]).inc()
        if fg.hallucinated_score:
            SCORE_HALLUC.inc()
        if card:
            CARDS.labels(card.get("event", "none")).inc()
        guard = fg.report()
        if a.shadow:
            guard["shadow"] = a.shadow
        res = CardResult(
            card=card, refused=False, refusal_reasons=[], guard=guard, raw=fg.raw, facts=a.facts,
            data_report=a.data.to_dict(), schema_valid=schema_valid, cost_usd=a.cost.total, ttft=ttft,
            latency=time.perf_counter() - t0, model_tag=b.tag, transcript=a.transcript.text if a.transcript else None,
        )
        yield {"type": "final", "result": res}

    async def run(self, inp: MomentInput, version: int | None = None) -> CardResult:
        result = None
        async for ev in self.stream(inp, version):
            if ev["type"] == "final":
                result = ev["result"]
        assert result is not None
        return result


def faithfulness(raw: dict, facts: dict) -> float:
    """Доля фактических полей, которые модель (до guard'а) указала верно."""
    checks = [raw.get(k) == facts.get(fk) for k, fk in FACT_FIELDS if k in raw]
    return sum(checks) / len(checks) if checks else 1.0
