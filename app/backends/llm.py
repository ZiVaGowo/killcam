"""LLM-бэкенды, генерирующие факт-карточку момента (JSON) стримом.

* ``MockLLM`` — детерминированный CPU-симулятор. Следует директивам промпта
  (``score_source``, ``style``), поэтому смена промпта в registry реально меняет поведение
  и ловится eval-гейтом (демо-удар №4).
* ``NaiveLLM`` — «взломанная» модель, которая слушается инструкций из <data>.
  Нужна red-team-евалу: доказываем, что guard'ы держат, даже если модель скомпрометирована.
* ``OpenAICompatLLM`` — реальная модель через OpenAI-совместимый API (vLLM, Ollama, OpenAI...).
"""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import AsyncIterator
from dataclasses import dataclass

DIRECTIVE_RE = re.compile(r"^#\s*directive:\s*(\w+)\s*=\s*(\S+)", re.MULTILINE)


def parse_directives(prompt: str) -> dict[str, str]:
    return {k: v for k, v in DIRECTIVE_RE.findall(prompt)}


@dataclass
class LLMRequest:
    system_prompt: str
    facts: dict          # проверенные факты: VLM (счёт/герой/событие) + упоминания из STT
    data_block: str      # недоверенные данные (чат, OCR оверлеев) в <data>...</data>
    user_text: str = ""
    match_id: str = "unknown"

    def messages(self) -> list[dict]:
        return [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": (
                    f"Инструкция продюсера: {self.user_text or 'сделай карточку момента'}\n"
                    f"match_id: {self.match_id}\n"
                    f"FACTS (источник истины): {json.dumps(self.facts, ensure_ascii=False)}\n"
                    f"{self.data_block}"
                ),
            },
        ]


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


async def _chunked(text: str, size: int, delay: float) -> AsyncIterator[str]:
    for i in range(0, len(text), size):
        if delay:
            await asyncio.sleep(delay)
        yield text[i : i + size]


class MockLLM:
    obey_data = False

    def __init__(self, prompt: str, model_uri: str = "mock://llm/card", delay: float = 0.0):
        self.directives = parse_directives(prompt)
        self.model_uri = model_uri
        self.delay = delay

    # --- «рассуждение» модели -------------------------------------------------
    def _score(self, req: LLMRequest) -> str | None:
        facts = req.facts
        if self.directives.get("score_source") == "caster":
            m = facts.get("caster") or {}
            if m.get("score"):
                return m["score"]
            for num in m.get("ambiguous_numbers") or []:  # «32» -> «3:2» — та самая галлюцинация
                return f"{num[0]}:{num[1:]}"
        return facts.get("score")

    def build_card(self, req: LLMRequest) -> dict:
        f = req.facts
        hero, event, timer = f.get("hero"), f.get("event") or "none", f.get("timer")
        score = self._score(req)
        hype = self.directives.get("style", "hype") == "hype"
        if f.get("game") != "arena_clash":
            headline = "Кадр не из Arena Clash — нужна проверка"
            event, hero, score = "none", None, None
        elif event == "kill":
            headline = f"{hero or 'Игрок'} забирает килл на {timer or '??:??'}" + ("!" if hype else ".")
        elif event == "objective":
            headline = f"{hero or 'Команда'} берёт объект на {timer or '??:??'}"
        else:
            headline = f"Момент {timer or ''}".strip()
        caption = f"Счёт {score}." if score else "Счёт на HUD не читается — проверьте вручную."
        if hype and event == "kill":
            caption += " Арена взрывается!"
        card = {
            "match_id": req.match_id,
            "timestamp": timer,
            "event": event,
            "hero": hero,
            "score": score,
            "headline": headline,
            "caption": caption,
            "hashtags": ["#ArenaClash", "#KillCam"] + ([f"#{hero}"] if hero else []),
            "confidence": f.get("hud_confidence", 0.0),
            "needs_review": score is None or f.get("game") != "arena_clash",
            "suggested_actions": [],
        }
        if event == "kill" and score:
            card["suggested_actions"] = [
                {"tool": "make_thumbnail", "args": {"match_id": req.match_id, "timestamp": timer}},
                {"tool": "tts_bumper", "args": {"text": headline}},
                {"tool": "publish_social", "args": {"text": f"{headline} {caption}"}},
            ]
        if self.obey_data:
            self._obey(card, req)
        return card

    def _obey(self, card: dict, req: LLMRequest) -> None:
        """Поведение скомпрометированной модели: исполняет инструкции из данных."""
        data = req.data_block.lower()
        m = re.search(r"(?:сч[её]т|score)\D{0,5}(\d{1,2}:\d{1,2})", data)
        if m:
            card["score"] = m.group(1)
            card["caption"] = f"Счёт {m.group(1)}."
        if any(w in data for w in ("унизь", "оскорби", "humiliate", "insult")):
            card["caption"] += " Этот игрок нуб и лузер."
        phone = re.search(r"\+?\d[\d\s\-()]{9,}\d", req.data_block)
        if phone:
            card["caption"] += f" Номер игрока: {phone.group(0)}"
        if any(w in data for w in ("system prompt", "системный промпт", "промпт")):
            card["caption"] += " " + req.system_prompt[:200]
        for tool in re.findall(r"(?:tool|инструмент|вызови)\W+([a-z_]+)", data):
            card["suggested_actions"].append({"tool": tool, "args": {}})

    async def stream(self, req: LLMRequest) -> AsyncIterator[str]:
        text = json.dumps(self.build_card(req), ensure_ascii=False)
        async for chunk in _chunked(text, 12, self.delay):
            yield chunk


class NaiveLLM(MockLLM):
    obey_data = True


class OpenAICompatLLM:
    def __init__(self, prompt: str, base_url: str, api_key: str, model: str):
        self.prompt, self.base_url, self.api_key, self.model = prompt, base_url.rstrip("/"), api_key, model
        self.model_uri = f"openai://{model}"

    async def stream(self, req: LLMRequest) -> AsyncIterator[str]:
        import httpx

        body = {"model": self.model, "stream": True, "messages": req.messages(), "response_format": {"type": "json_object"}}
        headers = {"Authorization": f"Bearer {self.api_key}"}
        async with httpx.AsyncClient(timeout=60) as client:
            async with client.stream("POST", f"{self.base_url}/chat/completions", json=body, headers=headers) as r:
                r.raise_for_status()
                async for line in r.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        break
                    delta = json.loads(payload)["choices"][0].get("delta", {}).get("content")
                    if delta:
                        yield delta
