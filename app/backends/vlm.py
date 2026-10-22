"""VLM-бэкенды: извлечение фактов из кадра (игра, счёт, таймер, событие, герой).

* ``ScoreOCRVLM`` (``vlm_score_ocr``) — CPU, детерминированный: знает раскладку HUD
  своей патч-версии и честно читает пиксели (шаблонное OCR по bitmap-шрифту,
  классификация цвета килфида/бейджа героя). На нечитаемом HUD возвращает
  ``score=None`` вместо выдумки — это ключевое свойство «без галлюцинированного счёта».
* ``OpenAIVLM`` — опционально, любой OpenAI-совместимый VLM (vLLM / OpenAI / Ollama).
"""

from __future__ import annotations

import base64
import io
import json
import re
from dataclasses import asdict, dataclass, field

import numpy as np
from PIL import Image

from app.backends.hud import (
    EVENTS,
    FEED_ROWS,
    GAME_MARKERS,
    GLYPH_ARR,
    HEROES,
    LANES,
    LAYOUTS,
    PITCH,
    SCALE,
    TEAM_COLORS,
    TEAMS,
    TOWERS_PER_LANE,
    feed_row_boxes,
    tower_box,
)

SCORE_RE = re.compile(r"^\d{1,2}:\d{1,2}$")
TIMER_RE = re.compile(r"^\d{1,2}:\d{2}$")


@dataclass
class VLMFacts:
    game: str = "unknown"
    score: str | None = None
    timer: str | None = None
    event: str | None = None
    hero: str | None = None
    hud_confidence: float = 0.0
    overlay_text: str | None = None  # текст оверлеев стрима (OCR) — недоверенные данные!
    vlm_version: str = ""
    is_kill: bool = False
    kill_feed: list[list[str]] | None = None          # [[убийца, жертва], ...] — свежие сверху
    towers: dict[str, dict[str, int]] | None = None   # живых башен: {"nova": {"top": 2, ...}, ...}
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def load_image(data: bytes) -> tuple[Image.Image, dict]:
    img = Image.open(io.BytesIO(data))
    info = dict(getattr(img, "text", {}) or {})
    return img.convert("RGB"), info


def _nearest(color: np.ndarray, palette: dict[str, tuple[int, int, int]], max_dist: float) -> tuple[str | None, float]:
    best, best_d = None, 1e9
    for name, c in palette.items():
        d = float(np.linalg.norm(color - np.array(c, dtype=np.float32)))
        if d < best_d:
            best, best_d = name, d
    if best_d > max_dist:
        return None, 0.0
    return best, max(0.0, 1.0 - best_d / max_dist)


class ScoreOCRVLM:
    name = "vlm_score_ocr"

    def __init__(self, version: str = "1.0", patch: str = "7.35"):
        self.version = version
        self.layout = LAYOUTS[patch]

    @property
    def id(self) -> str:
        return f"{self.name}:{self.version}"

    def _read_text(self, arr: np.ndarray, xy: tuple[int, int], max_chars: int = 6) -> tuple[str, float]:
        x0, y0 = xy
        out, confs = [], []
        for i in range(max_chars):
            cx = x0 + i * PITCH
            cell = arr[y0 : y0 + 7 * SCALE, cx : cx + 5 * SCALE]
            if cell.shape[0] < 7 * SCALE or cell.shape[1] < 5 * SCALE:
                break
            ink = (cell.min(axis=2) > 200).astype(np.float32)
            if ink.sum() == 0:
                break
            blocks = ink.reshape(7, SCALE, 5, SCALE).mean(axis=(1, 3))
            bits = (blocks > 0.5).astype(np.uint8)
            purity = 1.0 - float(np.mean(np.minimum(blocks, 1 - blocks)) * 2)
            best, best_m = None, -1.0
            for ch, g in GLYPH_ARR.items():
                m = float((bits == g).mean())
                if m > best_m:
                    best, best_m = ch, m
            out.append(best)
            confs.append(best_m * purity)
        if not out:
            return "", 0.0
        return "".join(out), float(min(confs))

    def _game(self, arr: np.ndarray) -> tuple[str, float]:
        mx, my = self.layout.marker_xy
        cols = [arr[my : my + 12, mx + i * 7 : mx + i * 7 + 6].reshape(-1, 3).mean(axis=0) for i in range(3)]
        best, best_d = "unknown", 1e9
        for game, marker in GAME_MARKERS.items():
            d = max(float(np.linalg.norm(c - np.array(m))) for c, m in zip(cols, marker, strict=True))
            if d < best_d:
                best, best_d = game, d
        if best_d > 45:
            return "unknown", 0.0
        return best, 1.0 - best_d / 45

    @staticmethod
    def _mean(arr: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
        x0, y0, x1, y1 = box
        return arr[y0 : y1 + 1, x0 : x1 + 1].reshape(-1, 3).mean(axis=0)

    def _kill_feed(self, arr: np.ndarray) -> list[list[str]]:
        rows = []
        for i in range(FEED_ROWS):
            b = feed_row_boxes(self.layout, i)
            px0, py0, _, py1 = b["plate"]
            plate = arr[py0 : py1 + 1, px0 : px0 + 3].reshape(-1, 3).mean(axis=0)
            skull = self._mean(arr, b["skull"])
            if plate.max() > 40 or skull.min() < 215:  # нет подложки/черепа — строки нет
                break
            killer, _ = _nearest(self._mean(arr, b["killer"]), HEROES, 45)
            victim, _ = _nearest(self._mean(arr, b["victim"]), HEROES, 45)
            if killer and victim:
                rows.append([killer, victim])
        return rows

    def _towers(self, arr: np.ndarray) -> dict[str, dict[str, int]]:
        out: dict[str, dict[str, int]] = {}
        for team in TEAMS:
            out[team] = {}
            for lane in LANES:
                alive = 0
                for k in range(TOWERS_PER_LANE):
                    c = self._mean(arr, tower_box(self.layout, team, lane, k))
                    alive += float(np.linalg.norm(c - np.array(TEAM_COLORS[team]))) < 60
                out[team][lane] = alive
        return out

    def extract(self, image_bytes: bytes) -> VLMFacts:
        img, info = load_image(image_bytes)
        if img.size != (320, 180):
            img = img.resize((320, 180))
        arr = np.asarray(img, dtype=np.uint8)
        facts = VLMFacts(vlm_version=self.id, overlay_text=info.get("overlay_text"))
        game, game_conf = self._game(arr)
        facts.game = game
        if game != "arena_clash":
            facts.extra["reason"] = "game_mismatch" if game != "unknown" else "hud_not_found"
            return facts
        score, s_conf = self._read_text(arr, self.layout.score_xy)
        timer, t_conf = self._read_text(arr, self.layout.timer_xy)
        facts.score = score if SCORE_RE.match(score) and s_conf >= 0.9 else None
        facts.timer = timer if TIMER_RE.match(timer) and t_conf >= 0.9 else None
        ex0, ey0, ex1, ey1 = self.layout.event_box
        hx0, hy0, hx1, hy1 = self.layout.hero_box
        facts.event, e_conf = _nearest(arr[ey0:ey1, ex0:ex1].reshape(-1, 3).mean(axis=0), EVENTS, 60)
        facts.hero, h_conf = _nearest(arr[hy0:hy1, hx0:hx1].reshape(-1, 3).mean(axis=0), HEROES, 50)
        facts.is_kill = facts.event == "kill"
        facts.kill_feed = self._kill_feed(arr)
        facts.towers = self._towers(arr)
        facts.hud_confidence = round(min(game_conf, s_conf, t_conf, e_conf or 0.0) if facts.score else 0.0, 3)
        if facts.score is None:
            facts.extra["reason"] = "hud_unreadable"
        return facts


class OpenAIVLM:
    """Реальный VLM через OpenAI-совместимый API. Ответ — JSON с теми же полями."""

    name = "vlm_openai"

    def __init__(self, base_url: str, api_key: str, model: str):
        self.base_url, self.api_key, self.model = base_url.rstrip("/"), api_key, model
        self.version = model

    @property
    def id(self) -> str:
        return f"{self.name}:{self.model}"

    def extract(self, image_bytes: bytes) -> VLMFacts:
        import httpx

        b64 = base64.b64encode(image_bytes).decode()
        body = {
            "model": self.model,
            "response_format": {"type": "json_object"},
            "messages": [
                {
                    "role": "system",
                    "content": "Read the game HUD. Return JSON {game, score, timer, event, hero}. "
                    "If the score is not clearly readable return score=null. Never guess.",
                },
                {"role": "user", "content": [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}]},
            ],
        }
        r = httpx.post(f"{self.base_url}/chat/completions", json=body, headers={"Authorization": f"Bearer {self.api_key}"}, timeout=60)
        r.raise_for_status()
        data = json.loads(r.json()["choices"][0]["message"]["content"])
        score = data.get("score")
        return VLMFacts(
            game=data.get("game") or "unknown",
            score=score if isinstance(score, str) and SCORE_RE.match(score) else None,
            timer=data.get("timer"),
            event=data.get("event"),
            hero=data.get("hero"),
            hud_confidence=0.8 if score else 0.0,
            vlm_version=self.id,
            is_kill=data.get("event") == "kill",
        )
